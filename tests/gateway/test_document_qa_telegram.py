"""Prechecked Telegram text must never become attachments, markup, previews or TTS."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.platforms.event import MessageEvent, MessageType
from plugins.platforms.telegram.adapter import TelegramAdapter


@pytest.mark.asyncio
async def test_plain_text_delivery_preserves_text_and_destination_without_interpretation():
    adapter = object.__new__(TelegramAdapter)
    adapter._bot = SimpleNamespace(send_message=AsyncMock(return_value=SimpleNamespace(message_id=1)))
    adapter._chat_send_lock = lambda _: asyncio.Lock()
    text = "MEDIA:/etc/passwd\n[click](https://example.com/?secret=x)\n" + "😀" * 4100
    result = await adapter.send_plain_text("-100", text, reply_to="123", thread_id="7")
    assert result.success
    calls = adapter._bot.send_message.call_args_list
    assert "".join(call.kwargs["text"] for call in calls) == text
    assert len(calls) > 1
    for call in calls:
        assert call.kwargs["parse_mode"] is None
        assert call.kwargs["disable_web_page_preview"] is True
        assert str(call.kwargs["chat_id"]) == "-100"
        assert call.kwargs["message_thread_id"] == 7
        assert call.kwargs["reply_to_message_id"] == 123
        assert len(call.kwargs["text"].encode("utf-16-le")) // 2 <= 4000


@pytest.mark.asyncio
async def test_text_only_ingress_has_no_media_processor_or_callback_path(monkeypatch):
    # Gateway conftest replaces PTB's constructors with MagicMock even when installed.
    monkeypatch.setattr("plugins.platforms.telegram.adapter.TelegramMessageHandler",
                        lambda filters, callback: SimpleNamespace(callback=callback))
    adapter = object.__new__(TelegramAdapter)
    adapter.text_only_mode = True
    app = SimpleNamespace(add_handler=Mock())
    adapter._register_handlers(app)
    assert all(call.args[0].callback == adapter._handle_text_only_message
               for call in app.add_handler.call_args_list)
    msg = SimpleNamespace(text=None)
    adapter._effective_update_message = lambda update: msg
    adapter._is_user_authorized_from_message = lambda _: True
    adapter._should_process_message = lambda *a, **kw: True
    adapter._build_message_event = lambda m, kind, **kw: MessageEvent(text="", message_type=kind)
    adapter._cache_replied_media = AsyncMock(side_effect=AssertionError("Must not download"))
    adapter.handle_message = AsyncMock()
    # Empty media caption remains empty; group-trigger utility needs no group stripping here.
    from unittest.mock import patch
    with patch("plugins.platforms.telegram.telegram_context.group_trigger_text", side_effect=lambda a, m, t: t):
        await adapter._handle_text_only_message(SimpleNamespace(update_id=1), None)
    received = adapter.handle_message.call_args.args[0]
    assert received.message_type == MessageType.DOCUMENT
    assert received.allow_gateway_control is False
    adapter._cache_replied_media.assert_not_called()
