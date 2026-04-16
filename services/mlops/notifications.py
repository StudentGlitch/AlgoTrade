from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Optional

import requests


class WebhookNotifier:
    """
    Supports:
    1) Generic webhook (`WEBHOOK_URL`)
    2) Discord webhook (`DISCORD_WEBHOOK_URL`)
    3) Telegram bot sendMessage (`TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`)
    """

    def __init__(self, webhook_url: Optional[str] = None, timeout: int = 8):
        self.webhook_url = (webhook_url or os.getenv("WEBHOOK_URL", "")).strip()
        self.discord_webhook = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
        self.telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
        self.timeout = timeout

    def enabled(self) -> bool:
        return bool(self.webhook_url or self.discord_webhook or (self.telegram_bot_token and self.telegram_chat_id))

    def _fmt_text(self, event: str, message: str, payload: Optional[dict]) -> str:
        ts = datetime.now(timezone.utc).isoformat()
        return f"[{event}] {message}\nUTC: {ts}\nPayload: {json.dumps(payload or {}, ensure_ascii=False)}"

    def _send_generic(self, body: dict) -> bool:
        if not self.webhook_url:
            return True
        r = requests.post(
            self.webhook_url,
            data=json.dumps(body),
            headers={"Content-Type": "application/json"},
            timeout=self.timeout,
        )
        return r.status_code < 300

    def _send_discord(self, event: str, message: str, payload: Optional[dict]) -> bool:
        if not self.discord_webhook:
            return True
        content = self._fmt_text(event, message, payload)
        r = requests.post(
            self.discord_webhook,
            json={"content": content[:1900]},
            timeout=self.timeout,
        )
        return r.status_code < 300

    def _send_telegram(self, event: str, message: str, payload: Optional[dict]) -> bool:
        if not (self.telegram_bot_token and self.telegram_chat_id):
            return True
        text = self._fmt_text(event, message, payload)
        url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
        r = requests.post(
            url,
            json={"chat_id": self.telegram_chat_id, "text": text[:3900]},
            timeout=self.timeout,
        )
        return r.status_code < 300

    def send(self, event: str, message: str, payload: Optional[dict] = None) -> bool:
        if not self.enabled():
            return False
        body = {
            "event": event,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": payload or {},
        }
        try:
            ok_generic = self._send_generic(body)
            ok_discord = self._send_discord(event, message, payload)
            ok_telegram = self._send_telegram(event, message, payload)
            return bool(ok_generic and ok_discord and ok_telegram)
        except Exception:
            return False
