"""Helpers shared by the cloud STT/TTS providers (gemini, deepgram).

Deliberately free functions, not a client base class: the providers differ
in auth scheme, endpoint shapes and error types, and keeping those visible
per provider is worth more than the ~12 lines a base class would save.
"""

from __future__ import annotations

import aiohttp


async def raise_for_status(
    resp: aiohttp.ClientResponse, exc_type: type[Exception], what: str
) -> None:
    """Raise the provider's error type with a truncated body on HTTP >= 400."""
    if resp.status >= 400:
        body = await resp.text()
        raise exc_type(f"HTTP {resp.status} from {what}: {body[:500]}")


def pick_voice(voices: dict[str, str], language: str, default_language: str) -> str | None:
    """Voice for `language`, falling back to the default language's voice.

    None means the map is empty — the caller raises its own provider error.
    """
    return voices.get(language) or voices.get(default_language)
