#!/usr/bin/env python3
"""Download all models: openWakeWord, Piper TTS voices, faster-whisper cache warmup.

Run from the repo root inside the venv:
    .venv/bin/python scripts/download_models.py [--config config.yaml]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# Pinned to a revision, not to `main`. These .onnx files are parsed by
# onnxruntime — native code — inside a long-running always-listening service,
# so "whatever that branch points at today" is the wrong contract: a force-push
# or a compromised repo would be fetched and loaded with nothing noticing. The
# hashes below were verified against this revision (the .onnx against its LFS
# sha256, the sidecars against their git blob oids).
_PIPER_REVISION = "f5a6e9094787fd865d65cb024472f977f9c542b5"
_PIPER_BASE = f"https://huggingface.co/rhasspy/piper-voices/resolve/{_PIPER_REVISION}"
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
PIPER_SHA256 = {
    "en_GB-alba-medium.onnx":
        "401369c4a81d09fdd86c32c5c864440811dbdcc66466cde2d64f7133a66ad03b",
    "en_GB-alba-medium.onnx.json":
        "aa965a2f02ecced632c2694e1fc72bbff6d65f265fab567ca945918c73dd89f4",
    "de_DE-thorsten-medium.onnx":
        "7e64762d8e5118bb578f2eea6207e1a35a8e0c30595010b666f983fc87bb7819",
    "de_DE-thorsten-medium.onnx.json":
        "974adee790533adb273a1ac88f49027d2a1b8f0f2cf4905954a4791e79264e85",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, dest: Path, sha256: str | None = None) -> None:
    """Fetch `url` to `dest`, verifying `sha256` before it lands.

    Verified in the .part file and only then renamed, so a mismatch leaves
    nothing in place for the service to load on the next start.
    """
    if dest.exists():
        print(f"  exists: {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  fetching {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    if sha256 is not None:
        got = _sha256(tmp)
        if got != sha256:
            tmp.unlink()
            raise SystemExit(
                f"  CHECKSUM MISMATCH for {dest.name}\n"
                f"    expected sha256 {sha256}\n"
                f"    got      sha256 {got}\n"
                f"  Refusing to install it. This file is parsed by onnxruntime "
                f"inside the service, so a substituted model is native code "
                f"execution — do not retry until you know why it changed."
            )
    tmp.rename(dest)
    print(f"  saved:  {dest}")


def download_openwakeword() -> None:
    """Fetch openWakeWord's shared feature models via its own downloader.

    Unverified, and not fixable from here: the URLs and the fetching both live
    inside openwakeword.utils. Noted rather than hidden — these .onnx files are
    parsed by onnxruntime in the service too, so a compromised upstream release
    asset reaches native code. Provisioning them from a pinned, hashed mirror
    is the fix if that ever matters more than the convenience.
    """
    print("openWakeWord models:")
    import openwakeword.utils

    openwakeword.utils.download_models()
    print("  done (shared feature models + pretrained wakewords, unverified)")


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
        download(url, models_dir / "piper" / name, PIPER_SHA256[name])


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
