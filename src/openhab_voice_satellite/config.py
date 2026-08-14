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

# violawake's streaming unit (violawake_sdk._constants.FRAME_SAMPLES) and the
# largest chunk its _process_core accepts before raising
VIOLA_FRAME_MS = 20
VIOLA_FRAME_SAMPLES = SAMPLE_RATE * VIOLA_FRAME_MS // 1000
VIOLA_MAX_FRAME_SAMPLES = VIOLA_FRAME_SAMPLES * 10

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


class ViolaAdaptiveConfig(BaseModel):
    """Noise-adaptive threshold (violawake's NoiseProfiler).

    Tracks a rolling noise floor and moves the wake threshold with the SNR.
    Off by default: it overrides `wakeword.threshold`, so enabling it hands
    tuning to the profiler's [min_threshold, max_threshold] band.
    """

    enabled: bool = False
    noise_window_s: float = Field(5.0, gt=0.0)
    min_threshold: float = Field(0.6, ge=0.0, le=1.0)
    max_threshold: float = Field(0.95, ge=0.0, le=1.0)
    snr_boost_db: float = 6.0
    snr_penalty_db: float = 3.0

    @model_validator(mode="after")
    def _band_ordered(self) -> ViolaAdaptiveConfig:
        if self.min_threshold > self.max_threshold:
            raise ValueError(
                f"min_threshold {self.min_threshold} exceeds max_threshold {self.max_threshold}"
            )
        return self


class ViolaPowerConfig(BaseModel):
    """Frame skipping (violawake's PowerManager).

    Silence skipping is the useful part here; duty_cycle_n above 1 drops
    frames the model has already committed to streaming over, so leave it at
    1 unless you have measured what the discontinuity costs in recall.
    """

    enabled: bool = False
    duty_cycle_n: int = Field(1, ge=1)
    silence_rms: float = Field(10.0, ge=0.0)  # int16 scale, matching upstream
    activity_threshold: float = Field(0.3, ge=0.0, le=1.0)
    active_window_s: float = Field(3.0, gt=0.0)


class ViolaVerifierConfig(BaseModel):
    """Stage-2 verifier: a mel-PCEN CNN re-scoring each wake trigger.

    The stage-1 temporal_cnn trades false accepts against recall on a hard
    frontier (its OWW embeddings cannot separate confusables). Running it at
    a low threshold and letting a larger CNN re-score the captured 1.5 s
    window breaks that trade: measured 0.73 FA/hour at 2 % FRR versus
    0.73 FA/hour at 28 % FRR for stage 1 alone. Training and calibration:
    violawakeword/TRAINING.md.

    `model` and `mel_basis` come as a pair — the .onnx scores (40, 151)
    mel-PCEN features, and the .npy is the mel filterbank the librosa-free
    frontend (verifier_mel.py) needs to produce them. The delay exists
    because stage 1 crosses its threshold before the phrase is finished;
    verifying immediately would score a truncated phrase.
    """

    model: str | None = None
    mel_basis: str | None = None
    threshold: float = Field(0.1, ge=0.0, le=1.0)
    delay_ms: int = Field(300, ge=0, le=1000)


class ViolaConfig(BaseModel):
    adaptive: ViolaAdaptiveConfig = Field(default_factory=ViolaAdaptiveConfig)
    power: ViolaPowerConfig = Field(default_factory=ViolaPowerConfig)
    verifier: ViolaVerifierConfig = Field(default_factory=ViolaVerifierConfig)


class WakewordConfig(BaseModel):
    # "openwakeword": pretrained phrase name or custom .onnx.
    # "violawake": registry name ("temporal_cnn") or a path to a .onnx you
    # trained — it ships no pretrained phrase library.
    engine: Literal["openwakeword", "violawake"] = "openwakeword"
    model: str = "hey_jarvis"
    threshold: float = Field(0.5, ge=0.0, le=1.0)
    threshold_speaking: float = Field(0.7, ge=0.0, le=1.0)
    stop_model: str | None = None
    stop_threshold: float = Field(0.5, ge=0.0, le=1.0)
    # None = reuse stop_threshold. The stop model runs during playback by
    # definition and usually carries the lowest threshold in the system, so
    # it is the first place echo false-accepts show up.
    stop_threshold_speaking: float | None = Field(None, ge=0.0, le=1.0)
    # consecutive frames above threshold before a detection fires. 1 keeps
    # the historical single-frame trigger; 2 costs one frame of latency and
    # rejects the transient spikes that make up most false accepts.
    patience: int = Field(1, ge=1, le=10)
    stop_patience: int = Field(1, ge=1, le=10)
    # optional per-speaker verifier models (openwakeword custom verifiers).
    # These are unpickled at startup: treat them like executable code.
    verifier_model: str | None = None
    stop_verifier_model: str | None = None
    verifier_threshold: float = Field(0.1, ge=0.0, le=1.0)
    # engine == "violawake" only; ignored otherwise
    viola: ViolaConfig = Field(default_factory=ViolaConfig)

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

    @model_validator(mode="after")
    def _violawake_supported(self) -> Config:
        """Reject the violawake settings that fail silently at runtime."""
        if self.wakeword.engine != "violawake":
            return self
        frame = self.audio.frame_samples
        # violawake scores whole 20 ms frames; a remainder makes it log once
        # per frame and return 0.0 forever, i.e. a detector that never fires
        if frame % VIOLA_FRAME_SAMPLES or frame > VIOLA_MAX_FRAME_SAMPLES:
            raise ValueError(
                f"engine 'violawake' needs audio.frame_ms a multiple of "
                f"{VIOLA_FRAME_MS} and at most "
                f"{VIOLA_MAX_FRAME_SAMPLES * 1000 // SAMPLE_RATE}; "
                f"{self.audio.frame_ms} gives {frame} samples"
            )
        if self.wakeword.verifier_model or self.wakeword.stop_verifier_model:
            raise ValueError(
                "wakeword.verifier_model is an openwakeword feature and has no "
                "effect with engine 'violawake' — remove it or switch engines"
            )
        verifier = self.wakeword.viola.verifier
        if bool(verifier.model) != bool(verifier.mel_basis):
            raise ValueError(
                "wakeword.viola.verifier needs `model` and `mel_basis` as a "
                "pair — the .onnx scores features only the .npy filterbank "
                "can produce; set both or neither"
            )
        if self.wakeword.model == WakewordConfig.model_fields["model"].default:
            # the default is an openwakeword phrase name; violawake ships no
            # pretrained phrases, so leaving it alone is always a mistake
            raise ValueError(
                f"engine 'violawake' needs wakeword.model set to a violawake "
                f"registry name or a path to a .onnx trained with violawake-train; "
                f"{self.wakeword.model!r} is an openwakeword phrase"
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
    wakeword = config.wakeword
    # only path-shaped values: pretrained openwakeword phrases ("hey_jarvis")
    # and violawake registry names ("temporal_cnn") must pass through verbatim
    for field in ("model", "stop_model", "verifier_model", "stop_verifier_model"):
        value = getattr(wakeword, field)
        if value and value.endswith((".onnx", ".tflite", ".pkl")):
            setattr(wakeword, field, _resolve_path(value, base))
    verifier = wakeword.viola.verifier
    for field in ("model", "mel_basis"):
        value = getattr(verifier, field)
        if value:
            setattr(verifier, field, _resolve_path(value, base))
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
