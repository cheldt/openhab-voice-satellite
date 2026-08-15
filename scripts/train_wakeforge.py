#!/usr/bin/env python3
"""Train a wakeforge wakeword model, then print how to judge it.

Workstation-only. ww_trainer pulls PyTorch, torchaudio, librosa and several
gigabytes of corpora, none of which belong on the device that runs the
satellite — train here, copy the two .onnx files over. This wrapper exists so
the output lands where the config expects it and so training ends pointing at
the evaluation, rather than at a directory nobody scores.

The trainer is not on PyPI (both `wakeforge` and `ww-trainer` 404); it installs
from git, which is what this prints if it is missing.

    python scripts/train_wakeforge.py "showdaan listen" showdaan_v1

A model is not finished when training ends. Recall has to be measured on
positives recorded in the voice and the room that will use it — a TTS corpus is
only ever valid as the negative half. See README's --score-wav section.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "models" / "wakeword" / "wakeforge"

INSTALL_HINT = (
    'pip install "ww_trainer @ git+https://github.com/TigreGotico/wakeforge@dev"'
)


def require_trainer() -> str:
    """The trainer's console entry point, or a clear exit explaining the install.

    Only the console script is required, deliberately: ww_trainer drags in
    PyTorch, so it usually lives in a venv of its own with nothing but its
    `bin` on PATH. Checking that it is importable *here* would reject exactly
    that arrangement.
    """
    entry = shutil.which("ww_trainer-train")
    if entry:
        return entry
    print("ww_trainer is not installed in this environment.", file=sys.stderr)
    print(f"It is not published on PyPI; install it from git:\n\n  {INSTALL_HINT}\n",
          file=sys.stderr)
    print("Expect PyTorch and several GB of corpus downloads — workstation only.",
          file=sys.stderr)
    raise SystemExit(2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("phrase", help='the wake phrase, e.g. "showdaan listen"')
    parser.add_argument("name", help="output directory name under models/wakeword/wakeforge")
    parser.add_argument("--tier", default="small",
                        help="ww_trainer hardware tier (default: small)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--metadata", type=Path,
                        help="dataset CSV; omit to let ww_trainer generate one")
    args, extra = parser.parse_known_args()

    entry = require_trainer()
    out = MODEL_DIR / args.name
    out.mkdir(parents=True, exist_ok=True)

    command = [
        entry, "--wake-word", args.phrase, "--tier", args.tier,
        "--epochs", str(args.epochs), "--save-best", "--export-onnx",
        "--output-dir", str(out),
    ]
    if args.metadata:
        command += ["--metadata", str(args.metadata)]
    command += extra

    print(" ".join(command))
    result = subprocess.run(command)
    if result.returncode:
        return result.returncode

    relative = out.relative_to(REPO_ROOT)
    print(f"\ntrained: {relative}")
    print("Judge it against the model you are running now, on the same audio:\n")
    print(f"  openhab-voice-satellite --positives recordings/wake \\\n"
          f"      --negatives recordings/room --compare wakeforge:{relative}\n")
    print("Read the live lines, not the sweep, wherever a stage-2 verifier is "
          "configured. Positives must be recorded in the target voice and room.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
