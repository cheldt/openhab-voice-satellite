"""Test doubles: wav-backed audio source, buffering sink, fake openHAB server."""

from __future__ import annotations

import asyncio
import wave
from pathlib import Path
from typing import AsyncIterator

import numpy as np
from aiohttp import web


class WavAudioSource:
    """Streams one or more wav files (16 kHz mono s16) as fixed-size frames."""

    def __init__(self, paths: list[Path], frame_samples: int = 1280, realtime: bool = False) -> None:
        self._frame_samples = frame_samples
        self._realtime = realtime
        parts = []
        for path in paths:
            with wave.open(str(path), "rb") as wav:
                assert wav.getframerate() == 16000 and wav.getnchannels() == 1
                parts.append(np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16))
        self._pcm = np.concatenate(parts)
        self.closed = False

    async def frames(self) -> AsyncIterator[np.ndarray]:
        n = self._frame_samples
        for i in range(0, len(self._pcm) - n + 1, n):
            if self.closed:
                return
            if self._realtime:
                await asyncio.sleep(n / 16000)
            else:
                await asyncio.sleep(0)
            yield self._pcm[i:i + n]

    def close(self) -> None:
        self.closed = True


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
    """Records played PCM; playback duration is simulated so cancel works."""

    def __init__(self, play_duration_s: float = 0.0) -> None:
        self.played: list[tuple[np.ndarray, int]] = []
        self.stopped = False
        self.duck_calls: list[float] = []
        self._play_duration_s = play_duration_s

    async def play(self, pcm: np.ndarray, sample_rate: int) -> None:
        self.played.append((pcm, sample_rate))
        if self._play_duration_s:
            await asyncio.sleep(self._play_duration_s)

    def stop(self) -> None:
        self.stopped = True

    def duck(self, factor: float) -> None:
        self.duck_calls.append(factor)

    def unduck(self) -> None:
        pass


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
