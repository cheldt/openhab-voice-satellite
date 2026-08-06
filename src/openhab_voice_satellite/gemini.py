"""Gemini cloud STT/TTS via plain REST."""

from __future__ import annotations

import base64
import io
import json
import logging
import re
import wave

import aiohttp
import numpy as np

from .audio.sink import AudioSink
from .config import SAMPLE_RATE, GeminiConfig, SttConfig, TtsConfig
from .fallback import CloudEngineError
from .stt import Transcript

log = logging.getLogger(__name__)

_RATE_RE = re.compile(r"rate=(\d+)")
DEFAULT_TTS_RATE = 24000


class GeminiError(CloudEngineError):
    """HTTP error or malformed response from the Gemini API."""


def pcm_to_wav_bytes(pcm: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.astype(np.int16).tobytes())
    return buf.getvalue()


class GeminiClient:
    def __init__(self, config: GeminiConfig, session: aiohttp.ClientSession) -> None:
        self._config = config
        self._session = session

    def _headers(self) -> dict[str, str]:
        # header instead of ?key= keeps the key out of URLs and logs
        return {"x-goog-api-key": self._config.key or "", "Content-Type": "application/json"}

    async def generate(self, model: str, payload: dict, timeout_s: float) -> dict:
        url = f"{self._config.base_url}/v1beta/models/{model}:generateContent"
        timeout = aiohttp.ClientTimeout(total=timeout_s)
        async with self._session.post(
            url, json=payload, headers=self._headers(), timeout=timeout
        ) as resp:
            body = await resp.text()
            if resp.status >= 400:
                raise GeminiError(f"HTTP {resp.status} from {model}: {body[:500]}")
            return json.loads(body)

    async def check_model(self, model: str) -> None:
        """GET the model metadata: validates key, model name and reachability."""
        url = f"{self._config.base_url}/v1beta/models/{model}"
        timeout = aiohttp.ClientTimeout(total=self._config.stt_timeout_s)
        async with self._session.get(url, headers=self._headers(), timeout=timeout) as resp:
            if resp.status >= 400:
                body = await resp.text()
                raise GeminiError(f"HTTP {resp.status} for model {model}: {body[:500]}")


def _response_text(response: dict) -> str:
    try:
        parts = response["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts)
    except (KeyError, IndexError, TypeError) as exc:
        raise GeminiError(f"malformed response: {exc}") from exc


class GeminiTranscriber:
    def __init__(self, client: GeminiClient, config: SttConfig, default_language: str) -> None:
        self._client = client
        self._config = config
        self._default_language = default_language
        self._model = client._config.stt_model
        self._timeout_s = client._config.stt_timeout_s

    async def transcribe(self, pcm: np.ndarray) -> Transcript:
        wav_b64 = base64.b64encode(pcm_to_wav_bytes(pcm, SAMPLE_RATE)).decode()
        languages = self._config.languages
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": (
                                "Transcribe this audio verbatim. The speech is in one of "
                                f"these languages: {', '.join(languages)}. Respond with "
                                "JSON only."
                            )
                        },
                        {"inlineData": {"mimeType": "audio/wav", "data": wav_b64}},
                    ]
                }
            ],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": {
                    "type": "OBJECT",
                    "properties": {
                        "text": {"type": "STRING"},
                        "language": {"type": "STRING", "enum": languages},
                    },
                    "required": ["text", "language"],
                },
            },
        }
        response = await self._client.generate(self._model, payload, self._timeout_s)
        raw = _response_text(response)
        try:
            data = json.loads(raw)
            text = str(data["text"]).strip()
            language = data.get("language")
        except (json.JSONDecodeError, KeyError, TypeError):
            log.warning("gemini STT returned non-JSON, using raw text")
            text, language = raw.strip(), None
        if language not in languages:
            language = self._default_language
        return Transcript(text=text, language=language)


class GeminiSpeaker:
    def __init__(
        self,
        client: GeminiClient,
        config: GeminiConfig,
        tts_config: TtsConfig,
        sink: AudioSink,
    ) -> None:
        self._client = client
        self._config = config
        self._tts_config = tts_config
        self._sink = sink

    def _voice(self, language: str) -> str:
        voices = self._config.tts_voices
        voice = voices.get(language) or voices.get(self._tts_config.default_language)
        if voice is None:
            raise GeminiError("gemini.tts_voices is empty")
        return voice

    async def speak(self, text: str, language: str) -> None:
        if not text.strip():
            return
        payload = {
            "contents": [{"parts": [{"text": text}]}],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {
                        "prebuiltVoiceConfig": {"voiceName": self._voice(language)}
                    }
                },
            },
        }
        response = await self._client.generate(
            self._config.tts_model, payload, self._config.tts_timeout_s
        )
        try:
            part = response["candidates"][0]["content"]["parts"][0]["inlineData"]
            raw = base64.b64decode(part["data"])
            mime = part.get("mimeType", "")
        except (KeyError, IndexError, TypeError) as exc:
            raise GeminiError(f"malformed TTS response: {exc}") from exc
        match = _RATE_RE.search(mime)
        rate = int(match.group(1)) if match else DEFAULT_TTS_RATE
        pcm = np.frombuffer(raw, dtype=np.int16)
        if len(pcm):
            await self._sink.play(pcm, rate)
