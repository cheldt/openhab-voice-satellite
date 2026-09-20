"""Self-test (--check): load config + all models, open audio, ping openHAB.

Each check imports its dependencies lazily so one broken stack (e.g. no
GStreamer) reports as a failed step instead of killing the whole run.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Awaitable, Callable

import numpy as np

from .config import Config


def check_audio(config: Config) -> None:
    from .audio.gst_devices import probe_capture, resolve_node

    input_node = resolve_node(config.audio.input_device, "input")
    resolve_node(config.audio.output_device, "output")
    # opens the real capture pipeline and requires one sample — also
    # catches a node WirePlumber cannot link (which stalls silently)
    probe_capture(input_node, config.audio.sample_rate)


# enough frames for the engine's context to fill and then be scored a few
# times over: openwakeword needs 16 embeddings behind 760 ms of mel, livekit
# is muted until a full 2 s window exists and then only scores every
# hop_frames-th frame
WAKEWORD_CHECK_SECONDS = 4.0


def _require_probability(score: float, what: str) -> None:
    if not np.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(
            f"{what} scored {score}, which is not a probability — "
            f"thresholds cannot be read against it"
        )


def check_wakeword(config: Config) -> None:
    from .wakeword import STOP, WAKE, build_detector

    detector = build_detector(config)
    frame = np.zeros(config.audio.frame_samples, dtype=np.int16)
    # one frame proves nothing: the engine is muted until its context window
    # fills, so a model whose scores are not probabilities at all would pass
    # while still reporting its startup zero
    frames = int(WAKEWORD_CHECK_SECONDS * 1000 / config.audio.frame_ms)
    heads = [(WAKE, "wakeword model")]
    if config.wakeword.stop_model:
        # the stop head is read against stop_threshold at runtime exactly like
        # the wake head, and openwakeword passes raw model output through
        # unclamped — a stop model exported without its sigmoid otherwise
        # passes here and then fires on nearly any speech, or never at all
        heads.append((STOP, "stop model"))
    scored = False
    for _ in range(frames):
        detector.process(frame)
        if not getattr(detector, "scored_last_frame", True):
            continue
        scored = True
        for key, what in heads:
            _require_probability(detector.score(key), what)
    if not scored:
        # EdgeTrigger.last returns 0.0 on an empty history and 0.0 is a valid
        # probability, so a detector that never scored would otherwise pass
        # this check without having run a single inference
        raise ValueError(
            f"wakeword engine {config.wakeword.engine!r} scored no frame in "
            f"{WAKEWORD_CHECK_SECONDS:.1f}s — its context window never filled"
        )


def check_vad(config: Config) -> None:
    from .vad import SpeechEndpointer

    endpointer = SpeechEndpointer(config.vad)
    endpointer.update(np.zeros(config.audio.frame_samples, dtype=np.int16))


def check_stt(config: Config) -> None:
    from .stt import Transcriber

    transcriber = Transcriber(config.stt, config.tts.default_language)
    transcriber._transcribe_sync(np.zeros(16000, dtype=np.int16))


def check_piper(config: Config) -> None:
    from piper import PiperVoice

    for lang, model_path in config.piper.voices.items():
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"{lang}: piper model missing: {path}")
        PiperVoice.load(str(path))


async def check_gemini(config: Config) -> None:
    import aiohttp

    from .gemini import GeminiClient

    async with aiohttp.ClientSession() as session:
        client = GeminiClient(config.gemini, session)
        if config.stt.engine == "gemini":
            await client.check_model(config.gemini.stt_model)
        if config.tts.engine == "gemini":
            await client.check_model(config.gemini.tts_model)


async def check_deepgram(config: Config) -> None:
    import aiohttp

    from .deepgram import DeepgramClient

    async with aiohttp.ClientSession() as session:
        await DeepgramClient(config.deepgram, session).check_auth()


async def check_openhab(config: Config) -> None:
    from .openhab import OpenHABClient, make_session

    async with make_session(config.openhab) as session:
        await OpenHABClient(config.openhab, session).ping()


Check = Callable[[Config], None | Awaitable[None]]


def select_checks(config: Config) -> list[tuple[str, Check]]:
    """The check list for this config's engine selection."""
    checks: list[tuple[str, Check]] = [
        ("audio devices", check_audio),
        ("wakeword model", check_wakeword),
        ("vad model", check_vad),
        ("whisper model (incl. warmup)", check_stt),
    ]
    # piper is the engine or the cloud engines' fallback either way
    checks.append(("piper voices", check_piper))
    if "gemini" in (config.stt.engine, config.tts.engine):
        checks.append(("gemini API", check_gemini))
    if "deepgram" in (config.stt.engine, config.tts.engine):
        checks.append(("deepgram API", check_deepgram))
    checks.append(("openHAB REST", check_openhab))
    return checks


async def run_checks(config: Config, checks: list[tuple[str, Check]] | None = None) -> int:
    """Run all selected checks; returns the process exit code."""
    if checks is None:
        checks = select_checks(config)
    print("openhab-voice-satellite self-test")
    failures = 0
    for name, check in checks:
        start = time.monotonic()
        try:
            result = check(config)
            if asyncio.iscoroutine(result):
                await result
            print(f"  ok   {name} ({time.monotonic() - start:.1f}s)")
        except Exception as exc:
            failures += 1
            print(f"  FAIL {name}: {exc}")
    print("all checks passed" if not failures else f"{failures} check(s) failed")
    return 1 if failures else 0
