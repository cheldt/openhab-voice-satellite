"""Configuration loading and validation (YAML + pydantic)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class AudioConfig(BaseModel):
    input_device: str | None = None
    output_device: str | None = None
    sample_rate: int = 16000
    frame_ms: int = 80

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
    silence_ms: int = 700
    no_speech_timeout_s: float = 8.0
    max_utterance_s: float = 15.0


class SttConfig(BaseModel):
    model: str = "small"
    compute_type: str = "int8"
    cpu_threads: int = 4
    languages: list[str] = Field(default_factory=lambda: ["de", "en"])

    @field_validator("languages")
    @classmethod
    def _non_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("stt.languages must not be empty")
        return v


class OpenHABConfig(BaseModel):
    url: str = "http://openhab.local:8080"
    api_token: str | None = None
    llm_tools: str | None = "item-send-command"  # ?llmTools= param; null = omit
    response_timeout_s: float = 30.0
    verify_ssl: bool = True  # False accepts self-signed certificates

    @field_validator("url")
    @classmethod
    def _strip_slash(cls, v: str) -> str:
        return v.rstrip("/")

    @property
    def token(self) -> str | None:
        """Env var OPENHAB_TOKEN wins over the config file value."""
        return os.environ.get("OPENHAB_TOKEN") or self.api_token


class TtsConfig(BaseModel):
    voices: dict[str, str] = Field(
        default_factory=lambda: {
            "de": "models/piper/de_DE-thorsten-medium.onnx",
            "en": "models/piper/en_US-lessac-medium.onnx",
        }
    )
    default_language: str = "de"

    @field_validator("default_language")
    @classmethod
    def _known_default(cls, v: str, info) -> str:
        voices = info.data.get("voices")
        if voices and v not in voices:
            raise ValueError(f"tts.default_language {v!r} has no voice configured")
        return v


class BargeInConfig(BaseModel):
    resume_listening: bool = True


class DialogConfig(BaseModel):
    enabled: bool = True
    max_turns: int = Field(3, ge=1, le=10)  # max follow-up rounds after the initial command
    context_mode: Literal["verbatim", "stitch"] = "verbatim"
    earcon: Literal["wake", "ack"] = "wake"  # cue when re-opening the mic


class EarconsConfig(BaseModel):
    enabled: bool = True
    wake: str = "sounds/wake.wav"
    ack: str = "sounds/ack.wav"
    error: str = "sounds/error.wav"


class LoggingConfig(BaseModel):
    level: str = "INFO"


class Config(BaseModel):
    audio: AudioConfig = Field(default_factory=AudioConfig)
    wakeword: WakewordConfig = Field(default_factory=WakewordConfig)
    vad: VadConfig = Field(default_factory=VadConfig)
    stt: SttConfig = Field(default_factory=SttConfig)
    openhab: OpenHABConfig = Field(default_factory=OpenHABConfig)
    tts: TtsConfig = Field(default_factory=TtsConfig)
    barge_in: BargeInConfig = Field(default_factory=BargeInConfig)
    dialog: DialogConfig = Field(default_factory=DialogConfig)
    earcons: EarconsConfig = Field(default_factory=EarconsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def load_config(path: str | Path) -> Config:
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return Config.model_validate(data)
