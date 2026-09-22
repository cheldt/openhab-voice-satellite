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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, dest: Path, sha256: str | None = None) -> None:
    """Fetch `url` to `dest`; with `sha256`, refuse a file that does not match."""
    if dest.exists():
        if sha256 and _sha256(dest) != sha256:
            sys.exit(
                f"  {dest} exists but its sha256 does not match the pinned digest; "
                "delete it to re-download, or update the pin if the model changed on purpose"
            )
        print(f"  exists: {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  fetching {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    if sha256:
        actual = _sha256(tmp)
        if actual != sha256:
            tmp.unlink()
            sys.exit(f"  sha256 mismatch for {url}: expected {sha256}, got {actual}")
    tmp.rename(dest)
    print(f"  saved:  {dest}")


def download_openwakeword() -> None:
    print("openWakeWord models:")
    import openwakeword.utils

    openwakeword.utils.download_models()
    print("  done (shared feature models + pretrained wakewords)")


# the pretrained phrase livekit-wakeword ships nothing of; the library
# bundles only the shared mel and embedding frontends. Kept in its own
# subdirectory because models/wakeword/*.onnx is openWakeWord's namespace —
# tooling there globs the directory flat.
#
# Pinned to a commit and a digest, not `main`: a branch name is as mutable
# as an action tag (see the SHA-pinning note in .github/workflows), and this
# file is fed straight into an ONNX Runtime session at every startup.
LIVEKIT_MODEL_URL = (
    "https://raw.githubusercontent.com/livekit-examples/hello-wakeword"
    "/d9a6c14bf86f822e31854f3c2df5012ff4d5dd8e/client/models/hey_livekit.onnx"
)
LIVEKIT_MODEL_SHA256 = "8bd634fb7acf1e52d06307fb8f460abf2c7a40e561fb4532fc56e087e0246f62"
DEFAULT_LIVEKIT_MODEL = REPO_ROOT / "models" / "wakeword" / "livekit" / "hey_livekit.onnx"


def download_livekit(model_path: Path) -> None:
    """Fetch the pretrained phrase to `model_path` — the configured wakeword.model.

    Only the file this script knows about is downloaded. A config that points
    at a custom classifier is reported present or missing instead of having
    hey_livekit.onnx written next to it and "all models ready" printed over a
    startup that will fail on the real path.
    """
    print("livekit wakeword model:")
    if model_path.name != Path(LIVEKIT_MODEL_URL).name:
        status = "present" if model_path.exists() else "MISSING — train/export it yourself"
        print(f"  custom model {model_path}: {status}")
        return
    download(LIVEKIT_MODEL_URL, model_path, sha256=LIVEKIT_MODEL_SHA256)


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
    wakeword_engine = "openwakeword"
    livekit_model = DEFAULT_LIVEKIT_MODEL
    if args.config.exists():
        from openhab_voice_satellite.config import load_config

        config = load_config(args.config)
        stt_model, compute_type = config.stt.model, config.stt.compute_type
        wakeword_engine = config.wakeword.engine
        if wakeword_engine == "livekit":
            # already absolute: load_config resolves it against the config dir
            livekit_model = Path(config.wakeword.model)

    download_openwakeword()
    if wakeword_engine == "livekit":
        download_livekit(livekit_model)
    else:
        print(f"livekit wakeword model: skipped (wakeword.engine is {wakeword_engine!r})")
    download_piper(REPO_ROOT / "models")
    warm_whisper(stt_model, compute_type)
    print("all models ready")


if __name__ == "__main__":
    main()
