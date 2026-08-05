#!/usr/bin/env bash
# Export kikiri-tts/kikiri-german-victoria to ONNX + voices npz, then smoke-test
# with the project's kokoro-onnx runtime.
#
# Usage: scripts/export_kokoro_victoria.sh
# Env:   BUILD_DIR  where the export toolchain is cloned (default: models/.build)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/models/.build}"
TOOL_DIR="$BUILD_DIR/kokoro-onnx-export"
OUT_DIR="$ROOT/models/kokoro"

mkdir -p "$BUILD_DIR"
if [ ! -d "$TOOL_DIR/.git" ]; then
    git clone --depth 1 https://github.com/adrianlyjak/kokoro-onnx-export "$TOOL_DIR"
fi

echo "==> syncing export tool environment"
(cd "$TOOL_DIR" && uv sync)

echo "==> exporting model to ONNX"
(cd "$TOOL_DIR" && uv run python "$ROOT/scripts/export_kokoro_victoria.py" --out-dir "$OUT_DIR")

echo "==> smoke test (project kokoro-onnx runtime, German)"
cd "$ROOT"
uv run python - <<'EOF'
import numpy as np
from kokoro_onnx import Kokoro

k = Kokoro("models/kokoro/kokoro-victoria.onnx", "models/kokoro/voices-victoria.npz")
samples, rate = k.create(
    "Hallo, ich bin Victoria. Dies ist ein Test der Sprachsynthese.",
    voice="victoria", speed=1.0, lang="de",
)
dur = len(samples) / rate
rms = float(np.sqrt(np.mean(samples**2)))
print(f"synthesized {dur:.2f}s @ {rate} Hz, rms={rms:.4f}")
assert dur > 1.0, "output suspiciously short"
assert rms > 0.01, "output near-silent"

import wave
pcm = (np.clip(samples, -1, 1) * 32767).astype(np.int16)
with wave.open("models/.build/victoria_smoke.wav", "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
    w.writeframes(pcm.tobytes())
print("wrote models/.build/victoria_smoke.wav")
EOF

echo "==> done: $OUT_DIR/kokoro-victoria.onnx + voices-victoria.npz"
