"""Opt-in, stateless document Q&A gateway mode using the generic message gate API."""

from __future__ import annotations

import asyncio
from collections import OrderedDict, deque
import hashlib
import json
import logging
from pathlib import Path
import time

from agent.gated_chat import GatedChat, RequestIdentity
from agent.message_gates import GateConfigurationError, RequiredGates, positive_number
from gateway.config import Platform
from gateway.platforms.event import MessageType

logger = logging.getLogger(__name__)


def bounded_int(value, name, maximum):
    if isinstance(value, bool) or not isinstance(value, int):
        raise GateConfigurationError(f"{name} must be an integer")
    return int(positive_number(value, name, maximum))


class DocumentQA:
    def __init__(self, config: dict, *, registry: dict, home: Path, api_key: str):
        settings = config["document_qa"]
        self.gates = RequiredGates(config.get("message_gates"), registry)
        corpus_path = Path(settings["corpus_file"])
        if not corpus_path.is_absolute():
            corpus_path = home / corpus_path
        with corpus_path.open("rb") as file:
            raw = file.read(48001)
        if len(raw) > 48000:
            raise GateConfigurationError("Corpus exceeds the 48000-byte MVP limit")
        documents = json.loads(raw)
        if not isinstance(documents, list) or not documents or len(documents) > 100:
            raise GateConfigurationError("Corpus must contain 1-100 document objects")
        ids = set()
        for doc in documents:
            if (not isinstance(doc, dict) or set(doc) != {"id", "title", "text"}
                    or any(not isinstance(v, str) or not v.strip() for v in doc.values())):
                raise GateConfigurationError("Each document requires string id, title and text")
            import re
            if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", doc["id"]) or doc["id"] in ids:
                raise GateConfigurationError("Document IDs must be unique citation-safe identifiers")
            ids.add(doc["id"])
        self.documents_json = json.dumps(documents, ensure_ascii=False)
        self.corpus_version = hashlib.sha256(raw).hexdigest()
        self.home = str(home.resolve())
        self.chat = GatedChat(
            gates=self.gates, endpoint=settings["model_endpoint"], model=settings["model"],
            api_key=api_key, policy_version=settings["policy_version"],
            timeout_seconds=settings.get("model_timeout_seconds", 45),
        )
        self.max_concurrent = bounded_int(settings.get("max_concurrent", 4), "max_concurrent", 64)
        self.per_user = bounded_int(settings.get("requests_per_user_minute", 6), "user rate", 600)
        self.global_rate = bounded_int(settings.get("requests_per_minute", 60), "global rate", 6000)
        self.active = 0
        self._users = OrderedDict()
        self._all = deque()

    def _admit(self, principal: str) -> bool:
        now = time.monotonic()
        bucket = self._users.setdefault(principal, deque())
        self._users.move_to_end(principal)
        if len(self._users) > 4096:
            self._users.popitem(last=False)
        for queue in (bucket, self._all):
            while queue and queue[0] <= now - 60:
                queue.popleft()
        if self.active >= self.max_concurrent or len(bucket) >= self.per_user or len(self._all) >= self.global_rate:
            return False
        bucket.append(now)
        self._all.append(now)
        return True

    async def handle(self, runner, event):
        source = event.source
        # Snapshot identity before any callback/await. Anonymous/channel senders are not principals.
        if (event.internal or source.platform != Platform.TELEGRAM or not source.user_id
                or getattr(source, "profile_route_rejected", False)
                or not runner._is_user_authorized_for_source(source)):
            return
        adapter = runner._delivery_adapter_for(source)
        if adapter is None:
            return
        chat_id, thread_id, reply_to = str(source.chat_id), source.thread_id, event.message_id

        async def send(text):
            # Bypass media extraction, rich cards, auto-TTS and normal agent response transforms.
            result = await adapter.send_plain_text(chat_id, text, reply_to=reply_to, thread_id=thread_id)
            if not result.success:
                logger.warning("Document Q&A delivery failed (content omitted)")

        if not self._admit(str(source.user_id)):
            await send(self.gates.template("busy"))
            return
        if (event.message_type != MessageType.TEXT or event.media_urls or event.media_types
                or event.text.lstrip().startswith("/") or not event.text.strip()
                or len(event.text.encode()) > 8192):
            await send(self.gates.template("unsupported"))
            return
        identity = RequestIdentity(self.home, str(source.user_id),
                                   f"{chat_id}:{thread_id or ''}:{source.user_id}",
                                   json.dumps([chat_id, thread_id]))
        self.active += 1
        try:
            await self.chat.answer(identity=identity, question=event.text,
                                   documents=json.loads(self.documents_json),
                                   corpus_version=self.corpus_version, send=send)
        finally:
            self.active -= 1

    async def busy(self, runner, event, session_key):
        """One bounded follow-up per lane, drained through the same handler. Never merge authors."""
        if (event.internal or not event.source.user_id
                or not runner._is_user_authorized_for_source(event.source)):
            return True
        adapter = runner._delivery_adapter_for(event.source)
        if adapter is None:
            return True
        if (len(event.text.encode()) > 8192 or event.media_urls
                or session_key in adapter._pending_messages or len(adapter._pending_messages) >= 16):
            await adapter.send_plain_text(str(event.source.chat_id), self.gates.template("busy"),
                                          reply_to=event.message_id, thread_id=event.source.thread_id)
            return True
        adapter._pending_messages[session_key] = event
        event._gateway_accepted = True
        return True


async def start_document_qa(runner, config: dict) -> bool:
    """Return False for normal mode; otherwise start only the messaging transport.

    No agent warmup, recovery, cron/kanban services, auxiliary calls or autonomous
    background watchers are started in this mode. Configuration is restart-bound.
    """
    settings = config.get("document_qa", {})
    if not isinstance(settings, dict):
        raise GateConfigurationError("document_qa must be a mapping")
    enabled = settings.get("enabled", False)
    if type(enabled) is not bool:
        raise GateConfigurationError("document_qa.enabled must be boolean")
    if not enabled:
        if (config.get("message_gates") or {}).get("required"):
            raise GateConfigurationError("Required message gates currently require document_qa.enabled")
        return False
    if runner._multiplex_on() or getattr(runner.config, "profile_routes", None):
        raise GateConfigurationError("Document Q&A MVP requires a standalone, non-routing profile")
    platforms = {p for p, cfg in runner.config.platforms.items() if cfg.enabled}
    if platforms != {Platform.TELEGRAM}:
        raise GateConfigurationError("Document Q&A MVP supports Telegram only")
    if not runner.config.group_sessions_per_user or not runner.config.thread_sessions_per_user:
        raise GateConfigurationError("Document Q&A requires per-user group AND thread sessions")
    from hermes_constants import get_hermes_home
    from agent.secret_scope import get_secret
    from hermes_cli.plugins import discover_plugins, get_plugin_manager
    await asyncio.to_thread(discover_plugins)
    key_name = settings.get("api_key_env", "OPENAI_API_KEY")
    if not isinstance(key_name, str) or not key_name:
        raise GateConfigurationError("api_key_env must name a credential")
    runner._document_qa = DocumentQA(config, registry=get_plugin_manager()._message_gates,
                                     home=Path(get_hermes_home()), api_key=get_secret(key_name) or "")
    runner._startup_restore_in_progress = False
    runner._startup_restore_queue = []
    runner._startup_restore_tasks = []
    runner._startup_parked_platforms = False
    aborted, enabled_count, _, pending = await runner._start_prefilter_platforms()
    if aborted:
        return True
    if any(not callable(getattr(adapter, "send_plain_text", None)) for _, _, adapter in pending):
        raise GateConfigurationError("Adapter has no gated plain-text delivery capability")
    results = await runner._start_connect_pending(pending)
    if results is None:
        return True
    nonretryable, retryable = [], []
    connected = await runner._start_aggregate_connect_results(results, retryable, nonretryable)
    if runner._start_handle_no_connections(connected, enabled_count, retryable, nonretryable):
        return True
    runner.delivery_router.adapters = runner.adapters
    runner._running = True
    runner._update_runtime_status(runner._serving_state())
    runner._spawn_reconnect_watcher()
    return True
