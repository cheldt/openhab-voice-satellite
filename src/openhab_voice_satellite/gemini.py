"""Gemini cloud STT/TTS via plain REST."""

from __future__ import annotations

import base64
import json
import logging
import re

import aiohttp
import numpy as np

from .audio.sink import AudioSink
from .audio.wav import pcm_to_wav_bytes
from .cloud import pick_voice, raise_for_status
from .config import SAMPLE_RATE, GeminiConfig, SttConfig, TtsConfig
from .fallback import CloudEngineError
from .stt import Transcript

log = logging.getLogger(__name__)

_RATE_RE = re.compile(r"rate=(\d+)")
DEFAULT_TTS_RATE = 24000


class GeminiError(CloudEngineError):
    """HTTP error or malformed response from the Gemini API."""


class GeminiClient:
    def __init__(self, config: GeminiConfig, session: aiohttp.ClientSession) -> None:
        self.config = config
        self._session = session

    def _headers(self) -> dict[str, str]:
        # header instead of ?key= keeps the key out of URLs and logs
        return {"x-goog-api-key": self.config.key or "", "Content-Type": "application/json"}

    async def generate(self, model: str, payload: dict, timeout_s: float) -> dict:
        url = f"{self.config.base_url}/v1beta/models/{model}:generateContent"
        timeout = aiohttp.ClientTimeout(total=timeout_s)
        async with self._session.post(
            url, json=payload, headers=self._headers(), timeout=timeout
        ) as resp:
            await raise_for_status(resp, GeminiError, model)
            return json.loads(await resp.text())

    async def check_model(self, model: str) -> None:
        """GET the model metadata: validates key, model name and reachability."""
        url = f"{self.config.base_url}/v1beta/models/{model}"
        timeout = aiohttp.ClientTimeout(total=self.config.stt_timeout_s)
        async with self._session.get(url, headers=self._headers(), timeout=timeout) as resp:
            await raise_for_status(resp, GeminiError, f"model {model}")


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
        response = await self._client.generate(
            self._client.config.stt_model, payload, self._client.config.stt_timeout_s
        )
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
    def __init__(self, client: GeminiClient, tts_config: TtsConfig, sink: AudioSink) -> None:
        self._client = client
        self._tts_config = tts_config
        self._sink = sink

    def _voice(self, language: str) -> str:
        voice = pick_voice(
            self._client.config.tts_voices, language, self._tts_config.default_language
        )
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
        config = self._client.config
        response = await self._client.generate(
            config.tts_model, payload, config.tts_timeout_s
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
