#!/usr/bin/env python3
"""Download all models: openWakeWord, Piper TTS voices, faster-whisper cache warmup.

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

_PIPER_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
# voice file -> HF subpath; each .onnx needs its sidecar .onnx.json
_PIPER_VOICES = {
    "en_GB-alba-medium": "en/en_GB/alba/medium",
    "de_DE-thorsten-medium": "de/de_DE/thorsten/medium",
}
PIPER_FILES = {
    f"{name}{ext}": f"{_PIPER_BASE}/{subpath}/{name}{ext}"
    for name, subpath in _PIPER_VOICES.items()
    for ext in (".onnx", ".onnx.json")
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


def check_custom_wakeword(model: str) -> None:
    """Report on a custom .onnx; there is no registry to fetch one from.

    A custom head is trained on a workstation (ultiwake) and copied here, so
    the most this can do is say whether it arrived.
    """
    path = Path(model)
    state = "ready" if path.exists() else "MISSING, train and deploy it first"
    print(f"custom wakeword model:\n  {state}: {path}")


def download_piper(models_dir: Path) -> None:
    print("Piper TTS models:")
    for name, url in PIPER_FILES.items():
        download(url, models_dir / "piper" / name)


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
    wakeword = None
    if args.config.exists():
        from openhab_voice_satellite.config import load_config

        config = load_config(args.config)
        stt_model, compute_type = config.stt.model, config.stt.compute_type
        wakeword = config.wakeword

    # the shared feature models come down either way; only the head can be ours
    download_openwakeword()
    if wakeword is not None:
        for model in (wakeword.model, wakeword.stop_model):
            if model and model.endswith(".onnx"):
                check_custom_wakeword(model)
    download_piper(REPO_ROOT / "models")
    warm_whisper(stt_model, compute_type)
    print("all models ready")


if __name__ == "__main__":
    main()
