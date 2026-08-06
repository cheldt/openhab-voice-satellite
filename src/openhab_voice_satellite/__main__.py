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
    try:
        from .audio.gst_devices import list_audio_nodes

        nodes = list_audio_nodes()
    except Exception as exc:
        print(f"cannot list PipeWire nodes: {exc}")
        print("(requires PipeWire running and the GStreamer pipewire plugin, "
              "see deploy/install.md)")
        sys.exit(1)
    print("PipeWire audio nodes (audio.input_device / audio.output_device "
          "match a substring of name or description):")
    for node in nodes:
        label = "source" if node.media_class.startswith("Audio/Source") else "sink  "
        print(f"  {label}  {node.name:<55} {node.description}")
    if not nodes:
        print("  (none found — is PipeWire running in this session?)")


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

    print("openhab-voice-satellite self-test")

    def check_audio() -> None:
        from .audio.gst_devices import probe_capture, resolve_node

        input_node = resolve_node(config.audio.input_device, "input")
        resolve_node(config.audio.output_device, "output")
        # opens the real capture pipeline and requires one sample — also
        # catches a node WirePlumber cannot link (which stalls silently)
        probe_capture(input_node, config.audio.sample_rate)

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

        from .tts import make_onnx_session

        def resolve(p: str) -> Path:
            return Path(p) if Path(p).is_absolute() else base_dir / p

        for lang, vc in config.kokoro.voices.items():
            # same session construction as production (tts.Speaker)
            kokoro = Kokoro.from_session(
                make_onnx_session(str(resolve(vc.model)), config.kokoro.threads),
                str(resolve(vc.voices)),
            )
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
    parser = argparse.ArgumentParser(prog="openhab-voice-satellite", description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--list-devices", action="store_true", help="list audio devices and exit")
    parser.add_argument("--check", action="store_true", help="run self-test and exit")
    parser.add_argument(
        "--probe-mic", action="store_true",
        help="30s field diagnostic: per-second mic RMS + wakeword score, "
             "with earcon playback and stream-link verification",
    )
    args = parser.parse_args()

    if args.list_devices:
        _list_devices()
        return

    if args.check:
        sys.exit(_check(args.config))

    if args.probe_mic:
        logging.basicConfig(format="%(levelname)-7s %(name)s: %(message)s", level=logging.INFO)
        config = load_config(args.config)

        from .probe import probe_mic

        sys.exit(probe_mic(config, args.config.resolve().parent))

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
