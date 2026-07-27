#!/usr/bin/env python3
"""Download all models: openWakeWord, Kokoro TTS models, faster-whisper cache warmup.

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

_KOKORO_RELEASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
_KOKORO_MARTIN = "https://huggingface.co/Godelaune/Kokoro-82M-ONNX-German-Martin/resolve/main"
# A smaller/faster English model exists at the same release tag:
# kokoro-v1.0.int8.onnx (92 MB) — point tts.voices.en.model at it if fp32 is too slow.
KOKORO_FILES = {
    "kokoro-v1.0.onnx": f"{_KOKORO_RELEASE}/kokoro-v1.0.onnx",
    "voices-v1.0.bin": f"{_KOKORO_RELEASE}/voices-v1.0.bin",
    "kokoro-martin.onnx": f"{_KOKORO_MARTIN}/kokoro-martin.onnx",
    "voices-martin.npz": f"{_KOKORO_MARTIN}/voices-martin.npz",
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


def download_kokoro(models_dir: Path) -> None:
    print("Kokoro TTS models:")
    for name, url in KOKORO_FILES.items():
        download(url, models_dir / "kokoro" / name)


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
    download_kokoro(REPO_ROOT / "models")
    warm_whisper(stt_model, compute_type)
    print("all models ready")


if __name__ == "__main__":
    main()
