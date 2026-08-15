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


class Stage2Config(BaseModel):
    """Stage-2 verifier: a mel-PCEN CNN re-scoring each wake trigger.

    Engine-neutral on purpose. A single stage trades false accepts against
    recall on a hard frontier, and that is a property of running one model over
    a stream rather than of any one architecture: violawake's temporal_cnn
    needs threshold 0.90 for <1 FA/hour and loses 28 % of positives there, and
    a wakeforge model trained on the same corpus fires 128 times an hour at the
    threshold where it keeps every positive. Letting a larger CNN re-score the
    captured 1.5 s window breaks the trade for both — measured on 5.48 h of
    LibriSpeech test-clean, it rejects 99 % of either engine's triggers.

    `model` and `mel_basis` come as a pair — the .onnx scores (40, 151)
    mel-PCEN features, and the .npy is the mel filterbank the librosa-free
    frontend (verifier_mel.py) needs to produce them. The delay exists
    because stage 1 crosses its threshold before the phrase is finished;
    verifying immediately would score a truncated phrase.

    Training and calibration: violawakeword/TRAINING.md.
    """

    model: str | None = None
    mel_basis: str | None = None
    threshold: float = Field(0.1, ge=0.0, le=1.0)
    delay_ms: int = Field(300, ge=0, le=1000)

    @model_validator(mode="after")
    def _pair(self) -> Stage2Config:
        if bool(self.model) != bool(self.mel_basis):
            raise ValueError(
                "wakeword.stage2 needs `model` and `mel_basis` as a pair — the "
                ".onnx scores features only the .npy filterbank can produce; "
                "set both or neither"
            )
        return self


class ViolaConfig(BaseModel):
    # Two blocks used to live here. `power` was violawake's PowerManager, which
    # duty-cycled frames on an RMS floor and spliced the backbone exactly as
    # described at ENGINE_CONTEXT_MS below; wakeword.vad_gate replaces it and
    # replays the context. `verifier` moved to wakeword.stage2, which is
    # engine-neutral — it works behind wakeforge just as well. A stale `power:`
    # is ignored like any unknown key, but a stale `verifier:` is rejected
    # below: silently dropping it would take this deployment from 0.7 false
    # accepts an hour to 67, with nothing in the logs to say why.
    adaptive: ViolaAdaptiveConfig = Field(default_factory=ViolaAdaptiveConfig)

    @model_validator(mode="before")
    @classmethod
    def _verifier_moved(cls, data):
        if isinstance(data, dict) and data.get("verifier"):
            raise ValueError(
                "wakeword.viola.verifier has moved to wakeword.stage2 — it is "
                "engine-neutral now. Move the block up one level; the fields "
                "are unchanged."
            )
        return data


# How much continuous audio each engine needs before its scores mean anything.
# Skipping a frame does not pause these backbones, it splices them: both
# openWakeWord and violawake compute their streaming melspectrogram over
# `tail(n_samples + 480)`, so a frame that never entered the ring puts audio
# from 80 ms earlier directly against the next one, and the mel frames across
# that seam read like a plosive. Contamination clears only once a full context
# has passed: 76 mel frames (760 ms) behind the head's own window, which is
# 9 embeddings for violawake and 16 for openWakeWord. wakeforge featurises each
# chunk independently, so only its 50-frame feature cache is affected.
ENGINE_CONTEXT_MS = {"violawake": 1480, "openwakeword": 2040, "wakeforge": 500}


class WakewordVadGateConfig(BaseModel):
    """Silero in front of the wakeword model, to skip scoring silence.

    Off by default, and it stays off until the saving is measured on the
    hardware in question: Silero runs on every frame, so the gate only pays for
    itself above roughly a 15-20 % skip fraction, and a pre-roll burst on every
    speech onset eats into that.

    `preroll_ms` is the frames replayed when the gate opens, and `None` means
    the engine's own context requirement — see ENGINE_CONTEXT_MS. Setting it
    lower is rejected rather than warned about: Silero decides on 32 ms chunks
    and will clip a phrase's onset, so a short pre-roll passes every scripted
    test and quietly loses recall in the room.

    `bypass_while_speaking` skips the gate entirely during playback. Silero
    calls our own TTS speech, so the gate would be open anyway; what it buys is
    that barge-in never pays onset-clipping latency, which is the one detection
    that cannot afford it.
    """

    enabled: bool = False
    threshold: float = Field(0.5, ge=0.0, le=1.0)
    hangover_ms: int = Field(1000, ge=0)
    preroll_ms: int | None = Field(None, ge=0)
    bypass_while_speaking: bool = True


class WakeforgeConfig(BaseModel):
    """Filenames inside a wakeforge training output directory.

    `wakeword.model` names the directory, not a file, because wakeforge exports
    a featurizer and a head that only work as the pair they were trained as.
    Pointing at one file and picking up the other from elsewhere in the config
    is a mismatch nothing downstream can detect, so the pair is addressed by
    the one thing that keeps them together.

    The defaults are what `ww_trainer-quickstart` writes; `ww_trainer-train`
    names them after the featurizer and head instead, hence the overrides.
    An absolute override wins outright, since Path(dir) / "/abs" is "/abs".
    """

    featurizer: str = "best_f1_featurizer.onnx"
    head: str = "best_f1.onnx"


class WakewordConfig(BaseModel):
    # "openwakeword": pretrained phrase name or custom .onnx.
    # "violawake": registry name ("temporal_cnn") or a path to a .onnx you
    # trained — it ships no pretrained phrase library.
    # "wakeforge": a training output directory holding a featurizer + head pair.
    engine: Literal["openwakeword", "violawake", "wakeforge"] = "openwakeword"
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
    # engine-neutral second stage; unset = single stage
    stage2: Stage2Config = Field(default_factory=Stage2Config)
    # engine == "violawake" only; ignored otherwise
    viola: ViolaConfig = Field(default_factory=ViolaConfig)
    # engine == "wakeforge" only; ignored otherwise
    wakeforge: WakeforgeConfig = Field(default_factory=WakeforgeConfig)
    vad_gate: WakewordVadGateConfig = Field(default_factory=WakewordVadGateConfig)

    @property
    def gate_preroll_ms(self) -> int:
        """The pre-roll the gate will actually use, engine default resolved."""
        if self.vad_gate.preroll_ms is not None:
            return self.vad_gate.preroll_ms
        return ENGINE_CONTEXT_MS[self.engine]

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
        if self.wakeword.model == WakewordConfig.model_fields["model"].default:
            # the default is an openwakeword phrase name; violawake ships no
            # pretrained phrases, so leaving it alone is always a mistake
            raise ValueError(
                f"engine 'violawake' needs wakeword.model set to a violawake "
                f"registry name or a path to a .onnx trained with violawake-train; "
                f"{self.wakeword.model!r} is an openwakeword phrase"
            )
        return self

    @model_validator(mode="after")
    def _wakeforge_supported(self) -> Config:
        """Reject the wakeforge settings that fail silently at runtime."""
        if self.wakeword.engine != "wakeforge":
            return self
        if self.wakeword.verifier_model or self.wakeword.stop_verifier_model:
            raise ValueError(
                "wakeword.verifier_model is an openwakeword feature and has no "
                "effect with engine 'wakeforge' — remove it or switch engines"
            )
        if self.wakeword.viola != ViolaConfig():
            # --compare swaps engine and model but carries the rest of the
            # config across, so a viola block outlives the engine that read it
            raise ValueError(
                "wakeword.viola settings have no effect with engine 'wakeforge' "
                "— remove them or switch engines"
            )
        if self.wakeword.model == WakewordConfig.model_fields["model"].default:
            raise ValueError(
                f"engine 'wakeforge' needs wakeword.model set to a directory "
                f"holding a featurizer and a head trained with ww_trainer; "
                f"{self.wakeword.model!r} is an openwakeword phrase"
            )
        # no frame_ms rule here on purpose: violawake's 20 ms unit is a fixed
        # SDK constant, but wakeforge's feature hop is a property of the model
        # you trained. The equivalent check runs against the real .onnx at
        # detector construction, which is what --check exercises.
        return self

    @model_validator(mode="after")
    def _vad_gate_supported(self) -> Config:
        """Reject a gate configuration that would cost recall silently."""
        gate = self.wakeword.vad_gate
        if not gate.enabled:
            return self
        engine = self.wakeword.engine
        context = ENGINE_CONTEXT_MS[engine]
        if engine == "openwakeword":
            # 26 frames of pre-roll is 81-108 ms of burst on a Pi against an
            # 80 ms frame budget, and this is not the engine the gate is for
            raise ValueError(
                f"wakeword.vad_gate is not supported on engine 'openwakeword': "
                f"its {context} ms of context makes every gate opening a burst "
                f"longer than one frame period"
            )
        if gate.preroll_ms is not None and gate.preroll_ms < context:
            raise ValueError(
                f"wakeword.vad_gate.preroll_ms {gate.preroll_ms} is below the "
                f"{context} ms engine '{engine}' needs to recover from a gap "
                f"(760 ms of mel context behind its head window) — a shorter "
                f"pre-roll scores the wake phrase on spliced audio. Leave it "
                f"null to take the engine's own figure."
            )
        if gate.hangover_ms < 700:
            log.warning(
                "wakeword.vad_gate.hangover_ms (%d) is shorter than a wake "
                "phrase; a pause between words can close the gate mid-phrase",
                gate.hangover_ms,
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
    # and violawake registry names ("temporal_cnn") must pass through verbatim.
    # wakeforge's model is a directory, so suffix-matching cannot spot it.
    for field in ("model", "stop_model", "verifier_model", "stop_verifier_model"):
        value = getattr(wakeword, field)
        directory = wakeword.engine == "wakeforge" and field in ("model", "stop_model")
        if value and (directory or value.endswith((".onnx", ".tflite", ".pkl"))):
            setattr(wakeword, field, _resolve_path(value, base))
    for field in ("model", "mel_basis"):
        value = getattr(wakeword.stage2, field)
        if value:
            setattr(wakeword.stage2, field, _resolve_path(value, base))
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
