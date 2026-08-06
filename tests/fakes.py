"""Test doubles: audio sources/sinks, engine stubs, fake cloud/openHAB servers."""

from __future__ import annotations

import asyncio
import base64
import json
from typing import AsyncIterator

import numpy as np
from aiohttp import web

from openhab_voice_satellite.stt import Transcript

FRAME = 1280  # 80 ms at 16 kHz


class FakeEndpointer:
    """Scripted endpointer: speech starts at frame `speech_at`, ends at `endpoint_at`.

    `later` holds (speech_at, endpoint_at) scripts for follow-up recordings;
    each `record_utterance` reset advances to the next one (last repeats).
    """

    def __init__(
        self,
        speech_at: int | None,
        endpoint_at: int | None,
        later: list[tuple[int | None, int | None]] | None = None,
    ) -> None:
        self._scripts = [(speech_at, endpoint_at)] + list(later or [])
        self._resets = 0
        self.reset()

    def reset(self) -> None:
        # __init__ and the first recording both use scripts[0].
        idx = min(max(self._resets - 1, 0), len(self._scripts) - 1)
        self._speech_at, self._endpoint_at = self._scripts[idx]
        self._resets += 1
        self._frames = 0
        self.speech_started = False

    def update(self, frame: np.ndarray) -> bool:
        self._frames += 1
        if self._speech_at is not None and self._frames >= self._speech_at:
            self.speech_started = True
        return self.speech_started

    @property
    def endpoint_reached(self) -> bool:
        return self._endpoint_at is not None and self._frames >= self._endpoint_at

    @property
    def elapsed_s(self) -> float:
        return self._frames * FRAME / 16000


class FakeTranscriber:
    def __init__(self, texts: str | list[str] = "schalte das licht an", language: str = "de") -> None:
        self._texts = [texts] if isinstance(texts, str) else list(texts)
        self._language = language

    async def transcribe(self, pcm: np.ndarray) -> Transcript:
        await asyncio.sleep(0.01)
        text = self._texts.pop(0) if len(self._texts) > 1 else self._texts[0]
        return Transcript(text=text, language=self._language)


class FakeSpeaker:
    def __init__(self, sink: BufferAudioSink, duration_s: float = 0.0) -> None:
        self._sink = sink
        self._duration_s = duration_s
        self.spoken: list[tuple[str, str]] = []

    async def speak(self, text: str, language: str) -> None:
        self.spoken.append((text, language))
        await self._sink.play(np.zeros(160, dtype=np.int16), 16000)
        if self._duration_s:
            await asyncio.sleep(self._duration_s)


class NullEarcons:
    async def play(self, name: str) -> None:
        pass


class LocalTranscriberStub:
    """Fallback target: records calls, answers a fixed transcript."""

    def __init__(self) -> None:
        self.calls: list[np.ndarray] = []

    async def transcribe(self, pcm: np.ndarray) -> Transcript:
        self.calls.append(pcm)
        return Transcript(text="local fallback", language="de")


class LocalSpeakerStub:
    """Fallback target: records what it was asked to speak."""

    def __init__(self) -> None:
        self.spoken: list[tuple[str, str]] = []

    async def speak(self, text: str, language: str) -> None:
        self.spoken.append((text, language))


class SilenceAudioSource:
    """Endless silence frames — keeps queues alive without triggering anything."""

    def __init__(self, frame_samples: int = 1280) -> None:
        self._frame_samples = frame_samples
        self.closed = False

    async def frames(self) -> AsyncIterator[np.ndarray]:
        while not self.closed:
            await asyncio.sleep(0)
            yield np.zeros(self._frame_samples, dtype=np.int16)

    def close(self) -> None:
        self.closed = True


class BufferAudioSink:
    """Records played PCM and duck/stop calls."""

    def __init__(self) -> None:
        self.played: list[tuple[np.ndarray, int]] = []
        self.stopped = False
        self.duck_calls: list[float] = []

    async def play(self, pcm: np.ndarray, sample_rate: int) -> None:
        self.played.append((pcm, sample_rate))

    def stop(self) -> None:
        self.stopped = True

    def duck(self, factor: float) -> None:
        self.duck_calls.append(factor)

    def unduck(self) -> None:
        pass


class FakeGemini:
    """aiohttp app implementing the two Gemini endpoints the client uses.

    - POST /v1beta/models/{model}:generateContent — dispatches on the payload:
      responseModalities AUDIO -> returns `tts_pcm` as base64 inlineData;
      anything else -> a text part with the scripted STT JSON.
    - GET /v1beta/models/{model} — model metadata (self-test probe)
    """

    def __init__(self, stt_text: str = "licht an", stt_language: str = "de") -> None:
        self.requests: list[tuple[str, dict]] = []  # (model, payload) per generateContent
        self.api_keys: list[str | None] = []  # x-goog-api-key header per call
        self.checked_models: list[str] = []  # models probed via GET
        self.stt_response: str | None = None  # raw text part; overrides stt_text/language
        self.stt_text = stt_text
        self.stt_language = stt_language
        self.tts_pcm = np.arange(0, 2400, dtype=np.int16)  # short ramp
        self.tts_mime = "audio/L16;codec=pcm;rate=24000"
        self.status = 200
        self.response_delay_s = 0.0

    def build_app(self) -> web.Application:
        app = web.Application()
        # aiohttp routing can't put ':' in a variable segment, match manually
        app.router.add_post("/v1beta/models/{tail:.+}", self._generate)
        app.router.add_get("/v1beta/models/{model}", self._get_model)
        return app

    async def _get_model(self, request: web.Request) -> web.Response:
        self.api_keys.append(request.headers.get("x-goog-api-key"))
        model = request.match_info["model"]
        self.checked_models.append(model)
        if self.status != 200:
            return web.json_response({"error": {"message": "boom"}}, status=self.status)
        return web.json_response({"name": f"models/{model}"})

    async def _generate(self, request: web.Request) -> web.Response:
        tail = request.match_info["tail"]
        model = tail.removesuffix(":generateContent")
        payload = await request.json()
        self.requests.append((model, payload))
        self.api_keys.append(request.headers.get("x-goog-api-key"))
        await asyncio.sleep(self.response_delay_s)
        if self.status != 200:
            return web.json_response({"error": {"message": "boom"}}, status=self.status)

        modalities = payload.get("generationConfig", {}).get("responseModalities")
        if modalities == ["AUDIO"]:
            data = base64.b64encode(self.tts_pcm.tobytes()).decode()
            part = {"inlineData": {"mimeType": self.tts_mime, "data": data}}
        else:
            text = self.stt_response
            if text is None:
                text = json.dumps({"text": self.stt_text, "language": self.stt_language})
            part = {"text": text}
        return web.json_response(
            {"candidates": [{"content": {"parts": [part]}}]}
        )


class FakeDeepgram:
    """aiohttp app implementing the three Deepgram endpoints the client uses.

    - POST /v1/listen — records query/auth/body, answers nova-shaped JSON;
      detected_language only included when the request asked for detection
    - POST /v1/speak — records query and JSON body, answers `tts_pcm` raw bytes
    - GET /v1/auth/token — key probe (self-test)
    """

    def __init__(self, stt_text: str = "licht an", stt_language: str = "de") -> None:
        self.listen_requests: list[tuple[list[tuple[str, str]], bytes]] = []
        self.speak_requests: list[tuple[list[tuple[str, str]], dict]] = []
        self.auth_headers: list[str | None] = []  # Authorization header per call
        self.stt_text = stt_text
        self.stt_language = stt_language
        self.tts_pcm = np.arange(0, 2400, dtype=np.int16)  # short ramp
        self.status = 200
        self.speak_statuses: list[int] = []  # FIFO per /v1/speak call; falls back to `status`
        self.response_delay_s = 0.0

    def build_app(self) -> web.Application:
        app = web.Application()
        app.router.add_post("/v1/listen", self._listen)
        app.router.add_post("/v1/speak", self._speak)
        app.router.add_get("/v1/auth/token", self._auth)
        return app

    async def _listen(self, request: web.Request) -> web.Response:
        query = [(k, v) for k, v in request.query.items()]
        self.listen_requests.append((query, await request.read()))
        self.auth_headers.append(request.headers.get("Authorization"))
        await asyncio.sleep(self.response_delay_s)
        if self.status != 200:
            return web.json_response({"err_msg": "boom"}, status=self.status)
        channel: dict = {"alternatives": [{"transcript": self.stt_text}]}
        if "detect_language" in request.query:
            channel["detected_language"] = self.stt_language
        return web.json_response({"results": {"channels": [channel]}})

    async def _speak(self, request: web.Request) -> web.Response:
        query = [(k, v) for k, v in request.query.items()]
        self.speak_requests.append((query, await request.json()))
        self.auth_headers.append(request.headers.get("Authorization"))
        await asyncio.sleep(self.response_delay_s)
        status = self.speak_statuses.pop(0) if self.speak_statuses else self.status
        if status != 200:
            return web.json_response({"err_msg": "boom"}, status=status)
        return web.Response(
            body=self.tts_pcm.tobytes(), content_type="application/octet-stream"
        )

    async def _auth(self, request: web.Request) -> web.Response:
        self.auth_headers.append(request.headers.get("Authorization"))
        if self.status != 200:
            return web.json_response({"err_msg": "boom"}, status=self.status)
        return web.json_response({"api_key_id": "fake"})


class FakeOpenHAB:
    """aiohttp app implementing the endpoints the client uses.

    - GET /rest/: health ping
    - POST /rest/voice/interpreters: records the text and answers with the
      scripted plain-text response after `response_delay_s`
    - DELETE /rest/voice/conversations/{cid}: records the deleted id
    """

    def __init__(self, response: str = "Okay.", response_delay_s: float = 0.05) -> None:
        self.commands: list[str] = []
        self.llm_tools: list[str | None] = []  # ?llmTools= value per call
        self.conversations: list[str | None] = []  # ?conversation= value per call
        self.deleted: list[str] = []  # conversation ids DELETEd
        self.headers: list[dict[str, str]] = []
        self.response = response
        self.responses: list[str] = []  # FIFO script; falls back to `response`
        self.response_delay_s = response_delay_s
        self.status = 200  # set e.g. 500 to test error handling
        self.delete_status = 200
        self.error_body = ""  # body sent along with a non-200 status

    def build_app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/rest/", self._root)
        app.router.add_post("/rest/voice/interpreters", self._interpret)
        app.router.add_delete("/rest/voice/conversations/{cid}", self._delete_conversation)
        return app

    async def _root(self, request: web.Request) -> web.Response:
        return web.json_response({"version": "8"})

    async def _interpret(self, request: web.Request) -> web.Response:
        self.commands.append(await request.text())
        self.llm_tools.append(request.query.get("llmTools"))
        self.conversations.append(request.query.get("conversation"))
        self.headers.append(dict(request.headers))
        await asyncio.sleep(self.response_delay_s)
        if self.status != 200:
            return web.Response(status=self.status, text=self.error_body)
        text = self.responses.pop(0) if self.responses else self.response
        return web.Response(text=text, content_type="text/plain")

    async def _delete_conversation(self, request: web.Request) -> web.Response:
        self.deleted.append(request.match_info["cid"])
        return web.Response(status=self.delete_status)
