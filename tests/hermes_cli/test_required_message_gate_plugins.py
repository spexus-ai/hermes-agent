"""Real directory discovery, profile alternation and registration teardown for required gates."""

import pytest
import yaml

from agent.message_gates import GateContext, RequiredGates
from hermes_cli.plugins import PluginManager


@pytest.mark.asyncio
async def test_profile_scoped_discovery_and_unload(tmp_path, monkeypatch):
    managers = []
    contexts = []
    for profile, action in (("a", "allow"), ("b", "deny")):
        home = tmp_path / profile
        plugin = home / "plugins" / "local-gate"
        plugin.mkdir(parents=True)
        (plugin / "plugin.yaml").write_text("name: local-gate\nversion: 0.1.0\ndescription: Test gate\n")
        (plugin / "__init__.py").write_text(
            "from agent.message_gates import GateDecision\n"
            f"async def evaluate(context): return GateDecision('{action}', 'profile-policy')\n"
            "def register(ctx):\n"
            "    ctx.register_message_gate('before_model_request', evaluate)\n"
            "    ctx.register_message_gate('before_response_delivery', evaluate)\n")
        (home / "config.yaml").write_text(yaml.safe_dump({"plugins": {"enabled": ["local-gate"]}}))
        monkeypatch.setenv("HERMES_HOME", str(home))
        manager = PluginManager()
        manager.discover_and_load()
        assert manager._plugins["local-gate"].enabled, manager._plugins["local-gate"].error
        managers.append(manager)
        contexts.append(GateContext.create(stage="before_model_request", profile_id=profile,
                         request_id=profile, principal_id="u", conversation_id="c", destination_id="d",
                         policy_version="v1", payload={"text": "same text"}))
    cfg = {"required": {"before_model_request": ["local-gate"], "before_response_delivery": ["local-gate"]}}
    chains = [RequiredGates(cfg, m._message_gates) for m in managers]
    try:
        for i, expected in ((0, "allow"), (1, "deny"), (0, "allow")):
            monkeypatch.setenv("HERMES_HOME", str(tmp_path / ("a" if i == 0 else "b")))
            assert (await chains[i].evaluate(contexts[i])).action == expected
        managers[0].unload()
        assert (await chains[0].evaluate(contexts[0])).action == "deny"
        assert (await chains[1].evaluate(contexts[1])).reason_code == "profile-policy"
    finally:
        for manager in managers:
            manager.unload()
