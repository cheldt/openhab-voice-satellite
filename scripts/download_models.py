#!/usr/bin/env python3
"""Download all models: openWakeWord, Piper voices, faster-whisper cache warmup.

Run from the repo root inside the venv:
    .venv/bin/python scripts/download_models.py [--config config.yaml]
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

PIPER_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
# voice name -> HF subpath
PIPER_VOICES = {
    "de_DE-thorsten-medium": "de/de_DE/thorsten/medium/de_DE-thorsten-medium",
    "en_US-lessac-medium": "en/en_US/lessac/medium/en_US-lessac-medium",
}


def download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  exists: {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  fetching {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)
    print(f"  saved:  {dest}")


def download_openwakeword() -> None:
    print("openWakeWord models:")
    import openwakeword.utils

    openwakeword.utils.download_models()
    print("  done (shared feature models + pretrained wakewords)")


def download_piper(models_dir: Path) -> None:
    print("Piper voices:")
    for name, subpath in PIPER_VOICES.items():
        download(f"{PIPER_BASE}/{subpath}.onnx", models_dir / "piper" / f"{name}.onnx")
        download(f"{PIPER_BASE}/{subpath}.onnx.json", models_dir / "piper" / f"{name}.onnx.json")


def warm_whisper(model: str, compute_type: str) -> None:
    print(f"faster-whisper {model} ({compute_type}) cache warmup:")
    from faster_whisper import WhisperModel

    WhisperModel(model, device="cpu", compute_type=compute_type)
    print("  done")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config.yaml")
    args = parser.parse_args()

    stt_model, compute_type = "small", "int8"
    if args.config.exists():
        from stt_proxy.config import load_config

        config = load_config(args.config)
        stt_model, compute_type = config.stt.model, config.stt.compute_type

    download_openwakeword()
    download_piper(REPO_ROOT / "models")
    warm_whisper(stt_model, compute_type)
    print("all models ready")


if __name__ == "__main__":
    main()
