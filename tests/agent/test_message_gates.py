"""Required gates mediate actual transport/delivery calls, including false/invalid decisions."""

import asyncio
import json
from dataclasses import replace

import httpx
import pytest

from agent.gated_chat import GatedChat, RequestIdentity
from agent.message_gates import GateConfigurationError, GateContext, GateDecision, RequiredGates


def config(**extra):
    return {"required": {"before_model_request": ["test"], "before_response_delivery": ["test"]},
            "templates": {"off_topic": "OFF_TOPIC", "unavailable": "UNAVAILABLE",
                          "output_rejected": "OUTPUT_REJECTED", "no_answer": "NO_ANSWER"}, **extra}


def context(stage="before_model_request"):
    return GateContext.create(stage=stage, profile_id="a", request_id="r", principal_id="u",
                              conversation_id="c", destination_id="d", policy_version="v1",
                              payload={"nested": {"text": "original"}})


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["input", "output", "invalid", "error", "timeout", "unload", "none"])
async def test_gate_failure_never_reaches_forbidden_sink(failure):
    calls, delivered, seen = [], [], []

    async def check(ctx):
        seen.append(ctx.stage)
        if ctx.stage == "before_response_delivery" and failure == "output":
            return GateDecision("deny", "topic", "output_rejected")
        if ctx.stage == "before_model_request":
            if failure == "input":
                return GateDecision("deny", "topic", "off_topic")
            if failure == "invalid":
                return {"action": "allow"}
            if failure == "none":
                return None
            if failure == "error":
                raise RuntimeError("DO NOT DISCLOSE THIS")
            if failure == "timeout":
                await asyncio.Event().wait()
        return GateDecision("allow", "ok")

    registry = {(stage, "test"): check for stage in config()["required"]}
    gates = RequiredGates(config(stage_deadline_seconds=.05), registry)
    if failure == "unload":
        registry.clear()

    def provider(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {
            "content": "BLOCKED ANSWER [source:manual]"}}]})

    async def send(text):
        delivered.append(text)

    async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as client:
        chat = GatedChat(gates=gates, endpoint="https://model.test/v1/chat/completions",
                         model="test", api_key="test-key", policy_version="v1")
        await chat.answer(identity=RequestIdentity("a", "u", "c", "d"), question="question",
                          documents=[{"id": "manual", "text": "reference"}], corpus_version="v1",
                          send=send, client=client)
    assert len(calls) == (1 if failure == "output" else 0)
    assert delivered == [{"input": "OFF_TOPIC", "output": "OUTPUT_REJECTED"}.get(failure, "UNAVAILABLE")]


@pytest.mark.asyncio
async def test_snapshot_binding_composition_and_tool_free_transport():
    observed, calls, delivered = [], [], []

    async def first(ctx):
        payload = ctx.payload
        if ctx.stage == "before_model_request":
            payload["request"]["messages"] = [{"role": "system", "content": "MUTATED"}]
        else:
            payload["answer"] = "MUTATED"
        observed.append((ctx.stage, ctx.payload_digest, ctx.destination_id, ctx.principal_id))
        return GateDecision("allow", "ok")

    async def second(ctx):
        assert "MUTATED" not in ctx.payload_json
        return GateDecision("allow", "ok")

    cfg = config()
    cfg["required"] = {s: ["first", "second"] for s in cfg["required"]}
    registry = {(s, name): cb for s in cfg["required"] for name, cb in (("first", first), ("second", second))}
    gates = RequiredGates(cfg, registry)

    def provider(request):
        calls.append(request)
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {
            "content": "Use the documented command. [source:manual]"}}]})

    async def send(text):
        delivered.append(text)

    async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as client:
        await GatedChat(gates=gates, endpoint="https://model.test/chat/completions", model="test",
                        api_key="credential", policy_version="v1").answer(
            identity=RequestIdentity("a", "alice", "chat", "group:topic"), question="Install?",
            documents=[{"id": "manual", "text": "Use the documented command."}], corpus_version="hash",
            send=send, client=client)
    wire = json.loads(calls[0].content)
    assert "tools" not in wire and wire["stream"] is False
    assert "MUTATED" not in str(wire)
    assert delivered == ["Use the documented command. [source:manual]"]
    assert all(item[2:] == ("group:topic", "alice") for item in observed)
    assert observed[0][1] != observed[1][1]

    async def deny(ctx):
        return GateDecision("deny", "no", "off_topic")

    async def forbidden(ctx):
        pytest.fail("A later allow must not override deny")

    registry[("before_model_request", "first")] = deny
    registry[("before_model_request", "second")] = forbidden
    assert (await gates.evaluate(context())).action == "deny"


@pytest.mark.asyncio
async def test_cancellation_is_not_allow_and_bad_registration_is_rejected():
    started = asyncio.Event()

    async def check(ctx):
        started.set()
        await asyncio.Event().wait()

    gates = RequiredGates(config(), {(s, "test"): check for s in config()["required"]})
    task = asyncio.create_task(gates.evaluate(context()))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    with pytest.raises(GateConfigurationError):
        RequiredGates(config(), {})
    with pytest.raises(GateConfigurationError):
        RequiredGates(config(on_error="allow"), {})
    changed = replace(context(), destination_id="someone-else")
    assert changed != context()
