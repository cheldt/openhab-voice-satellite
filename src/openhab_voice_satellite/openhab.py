"""openHAB REST client: send text to the voice interpreter, get the answer."""

from __future__ import annotations

import asyncio
import logging

import aiohttp

from .config import OpenHABConfig

log = logging.getLogger(__name__)

# Budget for one conversation DELETE. Pipeline.close() derives its shutdown
# drain bound from this so one slow DELETE fails itself (and is logged)
# instead of tripping the drain warning.
CONVERSATION_END_TIMEOUT_S = 5.0


class OpenHABTimeoutError(Exception):
    """openHAB did not answer within response_timeout_s."""


def make_session(config: OpenHABConfig) -> aiohttp.ClientSession:
    """ClientSession honoring `verify_ssl` (self-signed certificates).

    The session-level timeout bounds every request that does not pass its
    own (i.e. `ping`); without it aiohttp's default allows a 5-minute stall.
    """
    connector = None
    if not config.verify_ssl:
        log.warning("TLS certificate verification disabled (openhab.verify_ssl)")
        connector = aiohttp.TCPConnector(ssl=False)
    timeout = aiohttp.ClientTimeout(total=config.response_timeout_s)
    return aiohttp.ClientSession(connector=connector, timeout=timeout)


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
        # Not `GET /rest/`: that answers anonymously even with security
        # enabled, so it proves reachability only. The interpreters list
        # needs auth when security is on and proves the voice subsystem
        # is present — a bad token fails here, not at the first utterance.
        url = f"{self._config.url}/rest/voice/interpreters"
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
            # explicit charset: the body is UTF-8 either way, but a bare
            # text/plain lets the servlet default (ISO-8859-1) mangle umlauts
            "Content-Type": "text/plain; charset=utf-8",
            "Accept": "text/plain",
        }
        timeout = aiohttp.ClientTimeout(total=self._config.response_timeout_s)
        try:
            async with self._session.post(
                url, data=text.encode(), params=params or None, headers=headers, timeout=timeout
            ) as resp:
                body = (await resp.text()).strip()
                if resp.status in (401, 403):
                    # the single most likely misconfiguration deserves a name
                    log.error(
                        "openHAB rejected the request (HTTP %d) — check "
                        "api_token / OPENHAB_TOKEN",
                        resp.status,
                    )
                elif resp.status >= 400:
                    # the interpreter puts the actual error message in the body
                    log.error("interpreter returned HTTP %d: %s", resp.status, body[:500])
                resp.raise_for_status()
                return body
        except TimeoutError as exc:
            raise OpenHABTimeoutError(
                f"no answer within {self._config.response_timeout_s:.0f}s"
            ) from exc

    async def end_conversation(self, conversation_id: str) -> None:
        """Best-effort DELETE of a server-side conversation; never raises."""
        url = f"{self._config.url}/rest/voice/conversations/{conversation_id}"
        timeout = aiohttp.ClientTimeout(total=CONVERSATION_END_TIMEOUT_S)
        try:
            async with self._session.delete(
                url, headers=self._headers(), timeout=timeout
            ) as resp:
                if resp.status == 404:
                    # a barge-in before openHAB created the conversation
                    # deletes an id the server never saw — expected, not a signal
                    log.debug(
                        "conversation %s unknown to server (HTTP 404)",
                        conversation_id,
                    )
                elif resp.status >= 400:
                    log.warning(
                        "conversation DELETE returned HTTP %d for %s",
                        resp.status,
                        conversation_id,
                    )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("failed to end conversation %s: %s", conversation_id, exc)
