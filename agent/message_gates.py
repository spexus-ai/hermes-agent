"""Required, fail-closed plugin gates. Observer hook failure semantics do not apply here."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import math
from dataclasses import dataclass
from typing import Callable, Literal

STAGES = frozenset({"before_model_request", "before_response_delivery"})
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GateDecision:
    action: Literal["allow", "deny"]
    reason_code: str
    template_id: str | None = None


@dataclass(frozen=True)
class GateContext:
    stage: str
    profile_id: str
    request_id: str
    principal_id: str | None
    conversation_id: str
    destination_id: str
    policy_version: str
    payload_json: str

    @property
    def payload(self) -> dict:
        # Each reader owns a copy; nested mutation cannot change the send snapshot.
        return json.loads(self.payload_json)

    @property
    def payload_digest(self) -> str:
        return hashlib.sha256(self.payload_json.encode()).hexdigest()

    @classmethod
    def create(cls, *, payload: dict, **binding) -> GateContext:
        return cls(payload_json=json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                           separators=(",", ":"), allow_nan=False), **binding)


class GateConfigurationError(ValueError):
    pass


def positive_number(value, name: str, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GateConfigurationError(f"{name} must be a number")
    if not math.isfinite(value) or not 0 < value <= maximum:
        raise GateConfigurationError(f"{name} must be in (0, {maximum}]")
    return float(value)


class RequiredGates:
    """A profile-local chain. Configuration is validated before serving traffic."""

    def __init__(self, config: dict, registry: dict[tuple[str, str], Callable]):
        if not isinstance(config, dict) or config.get("on_error", "deny") != "deny":
            raise GateConfigurationError("message_gates requires on_error: deny")
        required = config.get("required")
        if not isinstance(required, dict) or set(required) != STAGES:
            raise GateConfigurationError("Both message gate stages must be configured")
        self.deadline = positive_number(config.get("stage_deadline_seconds", 10), "gate deadline", 120)
        self._registry = registry
        self._required = {}
        for stage, names in required.items():
            if (not isinstance(names, list) or not names
                    or any(not isinstance(n, str) or not n for n in names)
                    or len(set(names)) != len(names)):
                raise GateConfigurationError("Required gate lists must be nonempty and unique")
            for name in names:
                callback = registry.get((stage, name))
                if not inspect.iscoroutinefunction(callback):
                    raise GateConfigurationError(f"Missing async required gate: {stage}/{name}")
            self._required[stage] = tuple(names)
        templates = config.get("templates", {})
        if not isinstance(templates, dict) or any(
            not isinstance(k, str) or not isinstance(v, str) or not v.strip()
            or len(v.encode()) > 3000 for k, v in templates.items()
        ):
            raise GateConfigurationError("Templates must be bounded static strings")
        self._templates = {
            "unavailable": "Request checking is temporarily unavailable. Please try again later.",
            "off_topic": "Please ask a question within this bot's documented subject area.",
            "no_answer": "The available documentation does not contain an answer.",
            "output_rejected": "I could not prepare an answer within the documentation's scope.",
            "busy": "Too many requests. Please try again later.",
            "unsupported": "This bot accepts text questions only. Commands and attachments are disabled.",
            **templates,
        }

    def template(self, name: str | None) -> str:
        return self._templates.get(name, self._templates["unavailable"])

    async def evaluate(self, context: GateContext) -> GateDecision:
        try:
            async with asyncio.timeout(self.deadline):
                for name in self._required[context.stage]:
                    # Re-read the live registry: unload/replacement cannot leave a captured allow callback.
                    callback = self._registry.get((context.stage, name))
                    if not inspect.iscoroutinefunction(callback):
                        raise GateConfigurationError("Required gate is no longer registered")
                    decision = await callback(context)
                    if (type(decision) is not GateDecision or decision.action not in {"allow", "deny"}
                            or not isinstance(decision.reason_code, str) or not decision.reason_code
                            or len(decision.reason_code) > 128
                            or (decision.template_id is not None and not isinstance(decision.template_id, str))):
                        raise ValueError("Invalid gate decision")
                    if decision.action == "deny":
                        return decision
        except Exception as exc:
            # Exception messages can contain provider bodies, credentials or user text.
            logger.warning("Required gate failed stage=%s request=%s type=%s",
                           context.stage, context.request_id, type(exc).__name__)
            return GateDecision("deny", "gate_error", "unavailable")
        return GateDecision("allow", "all_required_gates_allowed")
