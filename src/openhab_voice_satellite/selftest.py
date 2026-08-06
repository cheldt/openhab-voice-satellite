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


def check_wakeword(config: Config) -> None:
    from .wakeword import WakewordDetector

    detector = WakewordDetector(config.wakeword)
    detector.process(np.zeros(config.audio.frame_samples, dtype=np.int16))


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
