"""CLI entry point: run the assistant, list audio devices, or self-test."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

from .config import load_config


def _list_devices() -> None:
    import sounddevice as sd

    print(sd.query_devices())


def _check(config_path: Path) -> int:
    """Self-test: load config + all models, open audio devices, ping openHAB."""
    import numpy as np

    config = load_config(config_path)
    base_dir = config_path.resolve().parent
    failures = 0

    def step(name: str, fn) -> None:
        nonlocal failures
        start = time.monotonic()
        try:
            fn()
            print(f"  ok   {name} ({time.monotonic() - start:.1f}s)")
        except Exception as exc:
            failures += 1
            print(f"  FAIL {name}: {exc}")

    print("stt-proxy self-test")

    def check_audio() -> None:
        import sounddevice as sd

        from .audio.source import find_device

        find_device(config.audio.input_device, "input")
        find_device(config.audio.output_device, "output")
        sd.check_input_settings(
            device=find_device(config.audio.input_device, "input"),
            samplerate=config.audio.sample_rate,
            channels=1,
            dtype="int16",
        )

    def check_wakeword() -> None:
        from .wakeword import WakewordDetector

        detector = WakewordDetector(config.wakeword)
        detector.process(np.zeros(config.audio.frame_samples, dtype=np.int16))

    def check_vad() -> None:
        from .vad import SpeechEndpointer

        endpointer = SpeechEndpointer(config.vad, config.audio.sample_rate)
        endpointer.update(np.zeros(config.audio.frame_samples, dtype=np.int16))

    def check_stt() -> None:
        from .stt import Transcriber

        transcriber = Transcriber(config.stt, config.tts.default_language)
        transcriber._transcribe_sync(np.zeros(16000, dtype=np.int16))

    def check_tts() -> None:
        from kokoro_onnx import Kokoro

        def resolve(p: str) -> Path:
            return Path(p) if Path(p).is_absolute() else base_dir / p

        for lang, vc in config.tts.voices.items():
            kokoro = Kokoro(str(resolve(vc.model)), str(resolve(vc.voices)))
            if vc.voice not in kokoro.get_voices():
                raise ValueError(f"{lang}: voice {vc.voice!r} not in {Path(vc.voices).name}")

    def check_piper() -> None:
        from piper import PiperVoice

        def resolve(p: str) -> Path:
            return Path(p) if Path(p).is_absolute() else base_dir / p

        for lang, model_path in config.piper.voices.items():
            path = resolve(model_path)
            if not path.exists():
                raise FileNotFoundError(f"{lang}: piper model missing: {path}")
            PiperVoice.load(str(path))

    def check_gemini() -> None:
        import aiohttp

        from .gemini import GeminiClient

        async def probe() -> None:
            async with aiohttp.ClientSession() as session:
                client = GeminiClient(config.gemini, session)
                if config.stt.engine == "gemini":
                    await client.check_model(config.gemini.stt_model)
                if config.tts.engine == "gemini":
                    await client.check_model(config.gemini.tts_model)

        asyncio.run(probe())

    def check_deepgram() -> None:
        import aiohttp

        from .deepgram import DeepgramClient

        async def probe() -> None:
            async with aiohttp.ClientSession() as session:
                await DeepgramClient(config.deepgram, session).check_auth()

        asyncio.run(probe())

    def check_openhab() -> None:
        from .openhab import OpenHABClient, make_session

        async def ping() -> None:
            async with make_session(config.openhab) as session:
                await OpenHABClient(config.openhab, session).ping()

        asyncio.run(ping())

    step("audio devices", check_audio)
    step("wakeword model", check_wakeword)
    step("vad model", check_vad)
    step("whisper model (incl. warmup)", check_stt)
    if config.tts.engine == "piper":
        step("piper voices", check_piper)
    else:
        # kokoro is the engine or the cloud fallback
        step("kokoro voices", check_tts)
    if "gemini" in (config.stt.engine, config.tts.engine):
        step("gemini API", check_gemini)
    if "deepgram" in (config.stt.engine, config.tts.engine):
        step("deepgram API", check_deepgram)
    step("openHAB REST", check_openhab)

    print("all checks passed" if not failures else f"{failures} check(s) failed")
    return 1 if failures else 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="stt-proxy", description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--list-devices", action="store_true", help="list audio devices and exit")
    parser.add_argument("--check", action="store_true", help="run self-test and exit")
    args = parser.parse_args()

    if args.list_devices:
        _list_devices()
        return

    if args.check:
        sys.exit(_check(args.config))

    config = load_config(args.config)
    logging.basicConfig(
        level=config.logging.level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    # phonemizer warns about word-count mismatches (numbers expand to words);
    # irrelevant for TTS, which only uses the whole phoneme string
    logging.getLogger("phonemizer").setLevel(logging.ERROR)

    from .app import App

    app = App(config, base_dir=args.config.resolve().parent)
    try:
        asyncio.run(app.run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
