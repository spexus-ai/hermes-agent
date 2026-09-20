"""Document Q&A uses neither the autonomous agent nor normal media delivery paths."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from agent.message_gates import GateDecision
from gateway.config import Platform
from gateway.document_qa import DocumentQA, start_document_qa
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource


def setup_service(tmp_path):
    (tmp_path / "documents.json").write_text(json.dumps([{"id": "manual", "title": "Guide", "text": "Install X."}]))
    cfg = {"document_qa": {"corpus_file": "documents.json", "model_endpoint": "https://model.test/chat/completions",
                           "model": "test", "policy_version": "v1"},
           "message_gates": {"required": {"before_model_request": ["test"], "before_response_delivery": ["test"]}}}
    return cfg


def event(text="How to install?", user="alice"):
    return MessageEvent(text=text, message_id="123", source=SessionSource(
        platform=Platform.TELEGRAM, chat_id="-100", chat_type="group", user_id=user, thread_id="7"))


@pytest.mark.asyncio
async def test_real_gateway_handler_and_queue_drain_are_stateless(tmp_path, monkeypatch):
    from gateway.run import GatewayRunner
    cfg = setup_service(tmp_path)
    seen, provider_requests = [], []

    async def check(ctx):
        seen.append((ctx.stage, ctx.principal_id))
        return GateDecision("allow", "ok")

    service = DocumentQA(cfg, registry={(s, "test"): check for s in cfg["message_gates"]["required"]},
                         home=tmp_path, api_key="fake")

    def provider(req):
        provider_requests.append(json.loads(req.content))
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {
            "content": "Install X. [source:manual]"}}]})

    original_client = httpx.AsyncClient
    monkeypatch.setattr("agent.gated_chat.httpx.AsyncClient", lambda **kw: original_client(
        transport=httpx.MockTransport(provider), **kw))
    adapter = SimpleNamespace(send_plain_text=AsyncMock(return_value=SimpleNamespace(success=True)),
                              _pending_messages={})
    runner = object.__new__(GatewayRunner)
    runner._document_qa = service
    runner._is_user_authorized_for_source = lambda src: True
    runner._delivery_adapter_for = lambda src: adapter
    runner._hm_admit_event = AsyncMock(side_effect=lambda e: (e, e.source, False))
    runner._handle_message_with_agent = AsyncMock(side_effect=AssertionError("Must not run agent"))
    assert await runner._handle_message(event()) is None
    queued = event("Second independent question", "bob")
    assert await runner._handle_active_session_busy_message(queued, "bob-lane") is True
    assert adapter._pending_messages["bob-lane"] is queued
    await runner._handle_message(adapter._pending_messages.pop("bob-lane"))
    assert [p for s, p in seen if s == "before_model_request"] == ["alice", "bob"]
    assert len(provider_requests) == 2
    assert len(provider_requests[1]["messages"]) == 2
    assert "How to install?" not in str(provider_requests[1])
    assert all("tools" not in req for req in provider_requests)
    assert adapter.send_plain_text.await_count == 2
    assert adapter.send_plain_text.call_args.kwargs == {"reply_to": "123", "thread_id": "7"}
    runner._handle_message_with_agent.assert_not_called()


@pytest.mark.asyncio
async def test_unsupported_and_internal_events_cannot_call_model(tmp_path):
    cfg = setup_service(tmp_path)

    async def never(ctx):
        pytest.fail("Unsupported event reached classifier")

    service = DocumentQA(cfg, registry={(s, "test"): never for s in cfg["message_gates"]["required"]},
                         home=tmp_path, api_key="fake")
    adapter = SimpleNamespace(send_plain_text=AsyncMock(return_value=SimpleNamespace(success=True)))
    runner = SimpleNamespace(_is_user_authorized_for_source=lambda src: True,
                             _delivery_adapter_for=lambda src: adapter)
    for e in (event("/model dangerous"), event("x" * 8193), event("")):
        await service.handle(runner, e)
    e = event()
    e.internal = True
    await service.handle(runner, e)
    e = event()
    e.message_type = MessageType.VOICE
    await service.handle(runner, e)
    assert adapter.send_plain_text.await_count == 4
    assert service.active == 0


@pytest.mark.asyncio
async def test_gates_cannot_be_enabled_silently_on_unprotected_mode():
    from agent.message_gates import GateConfigurationError
    with pytest.raises(GateConfigurationError):
        await start_document_qa(None, {"message_gates": {"required": {"before_model_request": ["x"]}}})


@pytest.mark.asyncio
async def test_startup_reads_effective_config_and_skips_autonomous_services(tmp_path, monkeypatch):
    import yaml
    from unittest.mock import Mock
    from gateway.run import GatewayRunner, _load_gateway_config
    from hermes_cli.plugins import _reset_plugin_managers_for_tests
    from hermes_constants import get_hermes_home
    home = get_hermes_home()
    monkeypatch.setattr("gateway.run._hermes_home", home)
    cfg = setup_service(home)
    cfg["document_qa"]["enabled"] = True
    cfg["gateway"] = {"multiplex_profiles": False}
    cfg["group_sessions_per_user"] = cfg["thread_sessions_per_user"] = True
    cfg["plugins"] = {"enabled": ["test"]}
    plugin = home / "plugins" / "test"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text("name: test\nversion: 0.1.0\ndescription: Test\n")
    (plugin / "__init__.py").write_text(
        "from agent.message_gates import GateDecision\n"
        "async def evaluate(context): return GateDecision('allow', 'ok')\n"
        "def register(ctx):\n"
        "    ctx.register_message_gate('before_model_request', evaluate)\n"
        "    ctx.register_message_gate('before_response_delivery', evaluate)\n")
    (home / "config.yaml").write_text(yaml.safe_dump(cfg))
    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:test")
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "1")
    _reset_plugin_managers_for_tests()
    runner = GatewayRunner()
    runner._start_install_faulthandler = Mock()
    runner._start_log_startup_environment = Mock()
    runner._start_check_access_policy = Mock(return_value=False)
    runner._start_recover_previous_run = AsyncMock(side_effect=AssertionError("No agent recovery"))
    runner._start_startup_warmup = Mock(side_effect=AssertionError("No agent warmup"))
    runner._start_finish_wiring = AsyncMock(side_effect=AssertionError("No background services"))
    adapter = SimpleNamespace(send_plain_text=AsyncMock())
    runner._start_prefilter_platforms = AsyncMock(return_value=(False, 1, [], [(Platform.TELEGRAM, None, adapter)]))
    runner._start_connect_pending = AsyncMock(return_value=[])
    runner._start_aggregate_connect_results = AsyncMock(return_value=1)
    runner._start_handle_no_connections = Mock(return_value=False)
    runner._update_runtime_status = Mock()
    runner._spawn_reconnect_watcher = Mock()
    try:
        assert _load_gateway_config()["document_qa"]["enabled"] is True
        assert await runner.start() is True
        assert isinstance(runner._document_qa, DocumentQA)
        assert runner._running
        runner._start_recover_previous_run.assert_not_called()
        runner._start_finish_wiring.assert_not_called()
    finally:
        _reset_plugin_managers_for_tests()
