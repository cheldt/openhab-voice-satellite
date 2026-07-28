"""openHAB REST client: send text to the voice interpreter, get the answer."""

from __future__ import annotations

import asyncio
import logging

import aiohttp

from .config import OpenHABConfig

log = logging.getLogger(__name__)


def make_session(config: OpenHABConfig) -> aiohttp.ClientSession:
    """ClientSession honoring `verify_ssl` (self-signed certificates)."""
    connector = None
    if not config.verify_ssl:
        log.warning("TLS certificate verification disabled (openhab.verify_ssl)")
        connector = aiohttp.TCPConnector(ssl=False)
    return aiohttp.ClientSession(connector=connector)


class OpenHABClient:
    """Interpreter client. Each POST may carry a conversation id so the
    server keeps the chat context; the answer comes back as the plain-text
    HTTP response. Conversations are deleted server-side when they end."""

    def __init__(self, config: OpenHABConfig, session: aiohttp.ClientSession) -> None:
        self._config = config
        self._session = session

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        token = self._config.token
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    async def ping(self) -> None:
        url = f"{self._config.url}/rest/"
        async with self._session.get(url, headers=self._headers()) as resp:
            resp.raise_for_status()

    async def send_command(self, text: str, conversation_id: str | None = None) -> str:
        url = f"{self._config.url}/rest/voice/interpreters"
        params: dict[str, str] = {}
        if self._config.llm_tools:
            params["llmTools"] = self._config.llm_tools
        if conversation_id:
            params["conversation"] = conversation_id
        headers = self._headers() | {
            "Content-Type": "text/plain",
            "Accept": "text/plain",
        }
        timeout = aiohttp.ClientTimeout(total=self._config.response_timeout_s)
        async with self._session.post(
            url, data=text.encode(), params=params or None, headers=headers, timeout=timeout
        ) as resp:
            body = (await resp.text()).strip()
            if resp.status >= 400:
                # the interpreter puts the actual error message in the body
                log.error("interpreter returned HTTP %d: %s", resp.status, body[:500])
            resp.raise_for_status()
            return body

    async def end_conversation(self, conversation_id: str) -> None:
        """Best-effort DELETE of a server-side conversation; never raises."""
        url = f"{self._config.url}/rest/voice/conversations/{conversation_id}"
        timeout = aiohttp.ClientTimeout(total=5.0)
        try:
            async with self._session.delete(
                url, headers=self._headers(), timeout=timeout
            ) as resp:
                if resp.status >= 400:
                    log.warning(
                        "conversation DELETE returned HTTP %d for %s",
                        resp.status,
                        conversation_id,
                    )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("failed to end conversation %s: %s", conversation_id, exc)
