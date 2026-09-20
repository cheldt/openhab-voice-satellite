"""Configuration loading and validation (YAML + pydantic)."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Annotated, ClassVar, Literal

import yaml
from pydantic import AfterValidator, BaseModel, Field, field_validator, model_validator

log = logging.getLogger(__name__)

# Capture rate is not configurable: Silero VAD, openWakeWord and whisper are
# all hardwired to 16 kHz.
SAMPLE_RATE = 16000

# URL fields: no trailing slash so f"{base_url}/path" composes cleanly
BaseUrl = Annotated[str, AfterValidator(lambda v: v.rstrip("/"))]


def _resolve_path(p: str, base: Path) -> str:
    """Resolve a config path: relative paths are relative to the config file."""
    return p if Path(p).is_absolute() else str(base / p)


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


class LivekitConfig(BaseModel):
    # livekit's predict() is stateless: every call recomputes the
    # melspectrogram and all 16 speech embeddings over the whole 2 s window,
    # where openWakeWord's frontend updates incrementally and costs ~0.9 ms a
    # frame. Measured single-threaded on x86, one call is ~13 ms — affordable
    # at 12.5 frames/s there, not on the Pi 5. So the engine scores every Nth
    # frame and reports the rest as skipped, which makes `patience` count
    # engine evaluations rather than mic frames, and delays a detection by up
    # to (hop_frames - 1) * audio.frame_ms.
    hop_frames: int = Field(4, ge=1, le=12)


class WakewordConfig(BaseModel):
    # "openwakeword": pretrained phrase name or a custom .onnx.
    # "livekit": path to a livekit-wakeword .onnx classifier; it ships no
    # pretrained phrase library, so there is no name form.
    engine: Literal["openwakeword", "livekit"] = "openwakeword"
    model: str = "hey_jarvis"
    threshold: float = Field(0.5, ge=0.0, le=1.0)
    threshold_speaking: float = Field(0.7, ge=0.0, le=1.0)
    stop_model: str | None = None
    stop_threshold: float = Field(0.5, ge=0.0, le=1.0)
    # None = reuse stop_threshold. The stop model runs during playback by
    # definition and usually carries the lowest threshold in the system, so
    # it is the first place echo false-accepts show up.
    stop_threshold_speaking: float | None = Field(None, ge=0.0, le=1.0)
    # consecutive observations above threshold before a detection fires. 1
    # keeps the historical single-frame trigger; 2 costs one observation of
    # latency and rejects the transient spikes that make up most false
    # accepts. Counts mic frames on openwakeword and engine evaluations on
    # livekit — see LivekitConfig.hop_frames.
    patience: int = Field(1, ge=1, le=10)
    stop_patience: int = Field(1, ge=1, le=10)
    # optional per-speaker verifier models (openwakeword custom verifiers).
    # These are unpickled at startup: treat them like executable code.
    verifier_model: str | None = None
    stop_verifier_model: str | None = None
    verifier_threshold: float = Field(0.1, ge=0.0, le=1.0)
    livekit: LivekitConfig = Field(default_factory=LivekitConfig)

    @property
    def effective_stop_threshold_speaking(self) -> float:
        if self.stop_threshold_speaking is None:
            return self.stop_threshold
        return self.stop_threshold_speaking

    @model_validator(mode="after")
    def _speaking_thresholds_are_raised(self) -> WakewordConfig:
        """Warn when the speaking threshold sits below the idle one.

        The speaking variants exist to raise the bar while our own output is
        audible; setting one lower makes the detector easiest to trigger
        exactly when the room contains our TTS. Legal — a lower bar is how you
        would deliberately favour barge-in — so this warns rather than raises,
        but it is almost always a leftover from tuning the idle threshold up.
        """
        for name, idle, speaking in (
            ("threshold", self.threshold, self.threshold_speaking),
            (
                "stop_threshold",
                self.stop_threshold,
                self.effective_stop_threshold_speaking,
            ),
        ):
            if speaking < idle:
                log.warning(
                    "wakeword.%s_speaking (%.2f) is below wakeword.%s (%.2f): "
                    "the bar drops while our own output is audible, so echo is "
                    "more likely to trigger a detection than room speech is",
                    name,
                    speaking,
                    name,
                    idle,
                )
        return self

    @model_validator(mode="after")
    def _engine_supports_settings(self) -> WakewordConfig:
        """Reject settings the selected engine would silently ignore."""
        if self.engine == "livekit":
            unsupported = [
                name
                for name in ("verifier_model", "stop_verifier_model")
                if getattr(self, name)
            ]
            if unsupported:
                raise ValueError(
                    f"wakeword.{', wakeword.'.join(unsupported)} "
                    "only applies to engine 'openwakeword' — livekit has no "
                    "per-speaker verifier, so the setting would do nothing"
                )
            if self.model == type(self).model_fields["model"].default:
                # the default is an openwakeword pretrained phrase name;
                # livekit ships no phrase library, so leaving it unset would
                # fail later as a confusing missing-file error
                raise ValueError(
                    f"wakeword.model is still the openwakeword default "
                    f"{self.model!r}; engine 'livekit' needs a path to a "
                    "classifier .onnx"
                )
        return self


class VadConfig(BaseModel):
    threshold: float = Field(0.5, ge=0.0, le=1.0)
    silence_ms: int = 1200
    no_speech_timeout_s: float = 8.0
    max_utterance_s: float = 15.0


class SttConfig(BaseModel):
    engine: Literal["local", "gemini", "deepgram"] = "local"
    model: str = "small"
    compute_type: str = "int8"
    # 3, not 4: ctranslate2 runs with the GIL released and saturates its
    # threads; on the 4-core Pi 5 one core must stay free or the 80 ms
    # wakeword cadence starves during THINKING (barge-in goes deaf)
    cpu_threads: int = 3
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
    engine: Literal["piper", "gemini", "deepgram"] = "piper"
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
        # piper is the engine or the cloud engines' fallback either way
        lang = self.tts.default_language
        if lang not in self.piper.voices:
            raise ValueError(
                f"tts.default_language {lang!r} has no piper voice configured"
            )
        return self


def _resolve_config_paths(config: Config, base: Path) -> Config:
    """Rewrite config-relative paths to absolute ones, relative to `base`.

    Done once at load time so no consumer needs to know where the config
    file lives. A Config constructed directly (tests) keeps its paths
    verbatim — they then resolve relative to the working directory.
    """
    config.piper.voices = {
        lang: _resolve_path(p, base) for lang, p in config.piper.voices.items()
    }
    earcons = config.earcons
    earcons.wake, earcons.ack, earcons.error, earcons.idle = (
        _resolve_path(p, base)
        for p in (earcons.wake, earcons.ack, earcons.error, earcons.idle)
    )
    return config


def load_config(path: str | Path) -> Config:
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return _resolve_config_paths(Config.model_validate(data), path.resolve().parent)
