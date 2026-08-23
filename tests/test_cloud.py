"""cloud.py helpers: voice selection and the shared error-status path."""

from __future__ import annotations

import pytest

from openhab_voice_satellite.cloud import pick_voice, raise_for_status
from openhab_voice_satellite.fallback import FALLBACK_ERRORS, CloudEngineError

VOICES = {"de": "viktoria", "en": "thalia"}


def test_pick_voice_prefers_language():
    assert pick_voice(VOICES, "en", "de") == "thalia"


def test_pick_voice_unknown_language_uses_default():
    assert pick_voice(VOICES, "fr", "de") == "viktoria"


def test_pick_voice_empty_map_returns_none():
    assert pick_voice({}, "de", "de") is None


class _Resp:
    def __init__(self, status: int, body: bytes) -> None:
        self.status = status
        self._body = body

    async def read(self) -> bytes:
        return self._body


async def test_ok_status_raises_nothing():
    await raise_for_status(_Resp(200, b"{}"), CloudEngineError, "x")


async def test_error_status_carries_the_body():
    with pytest.raises(CloudEngineError, match="HTTP 503 from /v1/listen: nope"):
        await raise_for_status(_Resp(503, b"nope"), CloudEngineError, "/v1/listen")


async def test_an_undecodable_error_body_still_names_the_status():
    """An error body is a diagnostic, so it is decoded leniently.

    resp.text() decodes strictly, so a gateway answering 502 with an
    ISO-8859-1 page raised UnicodeDecodeError *instead of* the provider error
    naming the status — the useful half of the information lost to the
    useless half.
    """
    with pytest.raises(CloudEngineError, match="HTTP 502"):
        await raise_for_status(
            _Resp(502, b"<html>Bad gateway: caf\xe9 f\xfcr alle</html>"),
            CloudEngineError, "/v1/speak",
        )


def test_body_decode_failures_are_in_the_fallback_taxonomy():
    # a 200 body that is not decodable still raises UnicodeDecodeError out of
    # json.loads; it is a ValueError, so without being named here it was the
    # one malformed-payload shape that escaped the fallback wrappers
    assert UnicodeDecodeError in FALLBACK_ERRORS
