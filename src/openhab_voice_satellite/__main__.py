"""CLI entry point: run the assistant, list audio devices, or self-test."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
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
        from .selftest import run_checks

        config = load_config(args.config)
        sys.exit(asyncio.run(run_checks(config, args.config.resolve().parent)))

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

    from .app import App

    app = App(config, base_dir=args.config.resolve().parent)
    try:
        asyncio.run(app.run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
