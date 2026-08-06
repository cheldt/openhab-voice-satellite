"""Configuration loading and validation (YAML + pydantic)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, ClassVar, Literal

import yaml
from pydantic import AfterValidator, BaseModel, Field, field_validator, model_validator

# Capture rate is not configurable: Silero VAD, openWakeWord and whisper are
# all hardwired to 16 kHz.
SAMPLE_RATE = 16000

# URL fields: no trailing slash so f"{base_url}/path" composes cleanly
BaseUrl = Annotated[str, AfterValidator(lambda v: v.rstrip("/"))]


def resolve_path(p: str, base: Path) -> Path:
    """Resolve a config path: relative paths are relative to the config file."""
    return Path(p) if Path(p).is_absolute() else base / p


class AudioConfig(BaseModel):
    # substring of a PipeWire node name/description (see --list-devices); null = default node
    input_device: str | None = None
    output_device: str | None = None
    # ClassVar keeps `config.audio.sample_rate` reads working while pydantic
    # ignores the key in old config files.
    sample_rate: ClassVar[int] = SAMPLE_RATE
    frame_ms: int = 80
    # ramped noise before a sound that follows an idle period; wakes powered
    # speakers whose signal-sensing mute ignores the keep-alive dither.
    # 0 = off.
    wakeup_preamble_ms: int = Field(0, ge=0)
    # quiet gap after which the next sound gets the preamble; match this to
    # how fast the speaker's mute kicks in (0 = before every sound)
    wakeup_preamble_idle_s: float = Field(60.0, ge=0.0)

    @property
    def frame_samples(self) -> int:
        return self.sample_rate * self.frame_ms // 1000


class WakewordConfig(BaseModel):
    model: str = "hey_jarvis"
    threshold: float = Field(0.5, ge=0.0, le=1.0)
    threshold_speaking: float = Field(0.7, ge=0.0, le=1.0)
    stop_model: str | None = None
    stop_threshold: float = Field(0.5, ge=0.0, le=1.0)


class VadConfig(BaseModel):
    threshold: float = Field(0.5, ge=0.0, le=1.0)
    silence_ms: int = 1200
    no_speech_timeout_s: float = 8.0
    max_utterance_s: float = 15.0


class SttConfig(BaseModel):
    engine: Literal["local", "gemini", "deepgram"] = "local"
    model: str = "small"
    compute_type: str = "int8"
    cpu_threads: int = 4
    beam_size: int = Field(1, ge=1)
    languages: list[str] = Field(default_factory=lambda: ["de", "en"])

    @field_validator("languages")
    @classmethod
    def _non_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("stt.languages must not be empty")
        return v


class OpenHABConfig(BaseModel):
    url: BaseUrl = "http://openhab.local:8080"
    api_token: str | None = None
    llm_tools: str | None = "item-send-command"  # ?llmTools= param; null = omit
    response_timeout_s: float = 30.0
    verify_ssl: bool = True  # False accepts self-signed certificates

    @property
    def token(self) -> str | None:
        """Env var OPENHAB_TOKEN wins over the config file value."""
        return os.environ.get("OPENHAB_TOKEN") or self.api_token


class TtsConfig(BaseModel):
    engine: Literal["kokoro", "gemini", "deepgram", "piper"] = "kokoro"
    default_language: str = "de"


class GeminiConfig(BaseModel):
    api_key: str | None = None
    base_url: BaseUrl = "https://generativelanguage.googleapis.com"  # test override
    stt_model: str = "gemini-3.6-flash"
    tts_model: str = "gemini-3.1-flash-tts-preview"
    stt_timeout_s: float = 10.0
    tts_timeout_s: float = 20.0
    tts_voices: dict[str, str] = Field(
        default_factory=lambda: {"de": "Kore", "en": "Puck"}
    )

    @property
    def key(self) -> str | None:
        """Env var GEMINI_API_KEY wins over the config file value."""
        return os.environ.get("GEMINI_API_KEY") or self.api_key


class DeepgramConfig(BaseModel):
    api_key: str | None = None
    base_url: BaseUrl = "https://api.deepgram.com"  # test override
    stt_model: str = "nova-3"
    stt_timeout_s: float = 10.0
    tts_timeout_s: float = 20.0
    tts_sample_rate: int = 24000  # linear16 rate requested from /v1/speak
    tts_voices: dict[str, str] = Field(
        default_factory=lambda: {"de": "aura-2-viktoria-de", "en": "aura-2-thalia-en"}
    )

    @property
    def key(self) -> str | None:
        """Env var DEEPGRAM_API_KEY wins over the config file value."""
        return os.environ.get("DEEPGRAM_API_KEY") or self.api_key


class KokoroVoiceConfig(BaseModel):
    model: str  # Kokoro .onnx model path
    voices: str  # voice-styles file (.bin / .npz)
    voice: str  # voice name inside the styles file, e.g. "bf_emma", "martin"
    lang: str  # espeak phonemizer code: "en-gb", "de"
    speed: float = Field(1.0, gt=0.5, le=2.0)


class KokoroConfig(BaseModel):
    threads: int = Field(2, ge=1)  # ORT intra-op threads per kokoro session
    voices: dict[str, KokoroVoiceConfig] = Field(
        default_factory=lambda: {
            "de": KokoroVoiceConfig(
                model="models/kokoro/kokoro-martin.onnx",
                voices="models/kokoro/voices-martin.npz",
                voice="martin",
                lang="de",
            ),
            "en": KokoroVoiceConfig(
                model="models/kokoro/kokoro-v1.0.onnx",
                voices="models/kokoro/voices-v1.0.bin",
                voice="bf_emma",
                lang="en-gb",
            ),
        }
    )


class PiperConfig(BaseModel):
    voices: dict[str, str] = Field(  # language -> .onnx model path (sidecar .onnx.json expected)
        default_factory=lambda: {
            "de": "models/piper/de_DE-thorsten-medium.onnx",
            "en": "models/piper/en_GB-alba-medium.onnx",
        }
    )


class BargeInConfig(BaseModel):
    resume_listening: bool = True


class DialogConfig(BaseModel):
    enabled: bool = True
    followup_timeout_s: float = Field(6.0, gt=0.0, le=60.0)  # silence that ends the conversation
    earcon: Literal["wake", "ack"] = "wake"  # cue when re-opening the mic


class EarconsConfig(BaseModel):
    enabled: bool = True
    wake: str = "sounds/wake.wav"
    ack: str = "sounds/ack.wav"
    error: str = "sounds/error.wav"
    idle: str = "sounds/idle.wav"


class LoggingConfig(BaseModel):
    level: str = "INFO"


class Config(BaseModel):
    audio: AudioConfig = Field(default_factory=AudioConfig)
    wakeword: WakewordConfig = Field(default_factory=WakewordConfig)
    vad: VadConfig = Field(default_factory=VadConfig)
    stt: SttConfig = Field(default_factory=SttConfig)
    openhab: OpenHABConfig = Field(default_factory=OpenHABConfig)
    tts: TtsConfig = Field(default_factory=TtsConfig)
    gemini: GeminiConfig = Field(default_factory=GeminiConfig)
    deepgram: DeepgramConfig = Field(default_factory=DeepgramConfig)
    kokoro: KokoroConfig = Field(default_factory=KokoroConfig)
    piper: PiperConfig = Field(default_factory=PiperConfig)
    barge_in: BargeInConfig = Field(default_factory=BargeInConfig)
    dialog: DialogConfig = Field(default_factory=DialogConfig)
    earcons: EarconsConfig = Field(default_factory=EarconsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @model_validator(mode="after")
    def _cloud_key_required(self) -> Config:
        engines = (self.stt.engine, self.tts.engine)
        if "gemini" in engines and not self.gemini.key:
            raise ValueError(
                "engine 'gemini' requires gemini.api_key or env GEMINI_API_KEY"
            )
        if "deepgram" in engines and not self.deepgram.key:
            raise ValueError(
                "engine 'deepgram' requires deepgram.api_key or env DEEPGRAM_API_KEY"
            )
        return self

    @model_validator(mode="after")
    def _tts_default_voice(self) -> Config:
        lang = self.tts.default_language
        if self.tts.engine == "piper":
            if lang not in self.piper.voices:
                raise ValueError(
                    f"tts.default_language {lang!r} has no piper voice configured"
                )
        # kokoro is the engine or the cloud fallback
        elif lang not in self.kokoro.voices:
            raise ValueError(
                f"tts.default_language {lang!r} has no kokoro voice configured"
            )
        return self


def load_config(path: str | Path) -> Config:
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return Config.model_validate(data)
