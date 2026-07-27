"""openHAB REST client: send text to the voice interpreter, get the answer."""

from __future__ import annotations

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
    """Stateless client: one interpreter POST per interaction, answer comes
    back as the plain-text HTTP response."""

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

    async def send_command(self, text: str) -> str:
        url = f"{self._config.url}/rest/voice/interpreters"
        params = {"llmTools": self._config.llm_tools} if self._config.llm_tools else None
        headers = self._headers() | {
            "Content-Type": "text/plain",
            "Accept": "text/plain",
        }
        timeout = aiohttp.ClientTimeout(total=self._config.response_timeout_s)
        async with self._session.post(
            url, data=text.encode(), params=params, headers=headers, timeout=timeout
        ) as resp:
            resp.raise_for_status()
            return (await resp.text()).strip()
