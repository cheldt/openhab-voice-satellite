"""Deepgram cloud STT (Nova) / TTS (Aura-2) via plain REST."""

from __future__ import annotations

import asyncio
import json
import logging

import aiohttp
import numpy as np

from .audio.sink import AudioSink
from .audio.wav import pcm_to_wav_bytes
from .cloud import pick_voice, raise_for_status
from .config import SAMPLE_RATE, DeepgramConfig, SttConfig, TtsConfig
from .fallback import FALLBACK_ERRORS, CloudEngineError, PartialSpeechError
from .stt import Transcript
from .tts import split_sentences

log = logging.getLogger(__name__)

# /v1/speak synthesizes the whole request before responding, so long texts
# (e.g. "list all items" answers) blow the timeout. Chunks this size return
# in a few seconds and are pipelined with playback.
TTS_CHUNK_CHARS = 400


def tts_chunks(text: str) -> list[str]:
    """Sentences, with oversized ones split further at commas/spaces."""
    chunks = []
    for sentence in split_sentences(text):
        while len(sentence) > TTS_CHUNK_CHARS:
            cut = sentence.rfind(", ", 0, TTS_CHUNK_CHARS)
            if cut == -1:
                cut = sentence.rfind(" ", 0, TTS_CHUNK_CHARS)
            if cut == -1:
                cut = TTS_CHUNK_CHARS
            head, sentence = sentence[:cut].rstrip(","), sentence[cut:].lstrip(", ")
            if head:
                chunks.append(head)
        if sentence:
            chunks.append(sentence)
    return chunks


class DeepgramError(CloudEngineError):
    """HTTP error or malformed response from the Deepgram API."""


class DeepgramClient:
    def __init__(self, config: DeepgramConfig, session: aiohttp.ClientSession) -> None:
        self.config = config
        self._session = session

    def _headers(self, content_type: str) -> dict[str, str]:
        # header instead of ?key= keeps the key out of URLs and logs
        return {
            "Authorization": f"Token {self.config.key or ''}",
            "Content-Type": content_type,
        }

    async def listen(
        self, wav: bytes, params: list[tuple[str, str]], timeout_s: float
    ) -> dict:
        url = f"{self.config.base_url}/v1/listen"
        timeout = aiohttp.ClientTimeout(total=timeout_s)
        async with self._session.post(
            url,
            data=wav,
            params=params,
            headers=self._headers("audio/wav"),
            timeout=timeout,
        ) as resp:
            await raise_for_status(resp, DeepgramError, "/v1/listen")
            return json.loads(await resp.text())

    async def speak(
        self, text: str, params: list[tuple[str, str]], timeout_s: float
    ) -> bytes:
        url = f"{self.config.base_url}/v1/speak"
        timeout = aiohttp.ClientTimeout(total=timeout_s)
        async with self._session.post(
            url,
            json={"text": text},
            params=params,
            headers=self._headers("application/json"),
            timeout=timeout,
        ) as resp:
            await raise_for_status(resp, DeepgramError, "/v1/speak")
            return await resp.read()

    async def check_auth(self) -> None:
        """GET the token metadata: validates key and reachability."""
        url = f"{self.config.base_url}/v1/auth/token"
        timeout = aiohttp.ClientTimeout(total=self.config.stt_timeout_s)
        async with self._session.get(
            url, headers=self._headers("application/json"), timeout=timeout
        ) as resp:
            await raise_for_status(resp, DeepgramError, "/v1/auth/token")


class DeepgramTranscriber:
    def __init__(
        self, client: DeepgramClient, config: SttConfig, default_language: str
    ) -> None:
        self._client = client
        self._config = config
        self._default_language = default_language

    async def transcribe(self, pcm: np.ndarray) -> Transcript:
        languages = self._config.languages
        params = [("model", self._client.config.stt_model), ("smart_format", "true")]
        if len(languages) == 1:
            params.append(("language", languages[0]))
        else:
            # repeated detect_language params restrict the candidate set
            params.extend(("detect_language", lang) for lang in languages)
        response = await self._client.listen(
            pcm_to_wav_bytes(pcm, SAMPLE_RATE), params, self._client.config.stt_timeout_s
        )
        try:
            channel = response["results"]["channels"][0]
            text = str(channel["alternatives"][0]["transcript"]).strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise DeepgramError(f"malformed response: {exc}") from exc
        detected = channel.get("detected_language") or languages[0]
        language = detected.split("-")[0]  # BCP-47 tag -> bare code
        if language not in languages:
            language = self._default_language
        return Transcript(text=text, language=language)


class DeepgramSpeaker:
    def __init__(self, client: DeepgramClient, tts_config: TtsConfig, sink: AudioSink) -> None:
        self._client = client
        self._tts_config = tts_config
        self._sink = sink

    def _voice(self, language: str) -> str:
        voice = pick_voice(
            self._client.config.tts_voices, language, self._tts_config.default_language
        )
        if voice is None:
            raise DeepgramError("deepgram.tts_voices is empty")
        return voice

    async def speak(self, text: str, language: str) -> None:
        """Speak `text`, overlapping the fetch of chunk N+1 with playback of N."""
        chunks = tts_chunks(text)
        if not chunks:
            return
        config = self._client.config
        rate = config.tts_sample_rate
        params = [
            ("model", self._voice(language)),
            ("encoding", "linear16"),
            ("sample_rate", str(rate)),
            ("container", "none"),  # raw PCM; default would be a WAV container
        ]

        async def fetch(chunk: str) -> np.ndarray:
            raw = await self._client.speak(chunk, params, config.tts_timeout_s)
            return np.frombuffer(raw[: len(raw) & ~1], dtype=np.int16)

        pending: asyncio.Task | None = asyncio.create_task(fetch(chunks[0]))
        played = False
        try:
            for i in range(len(chunks)):
                current, pending = pending, None
                try:
                    pcm = await current
                except FALLBACK_ERRORS as exc:
                    if played:
                        # some audio already out — hand only the rest to the fallback
                        raise PartialSpeechError(
                            str(exc), remaining=" ".join(chunks[i:])
                        ) from exc
                    raise
                if i + 1 < len(chunks):
                    pending = asyncio.create_task(fetch(chunks[i + 1]))
                if len(pcm):
                    await self._sink.play(pcm, rate)
                    played = True
        finally:
            if pending is not None:
                pending.cancel()
