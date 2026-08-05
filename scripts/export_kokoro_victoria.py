"""Export kikiri-tts/kikiri-german-victoria (Kokoro-compatible StyleTTS2 fine-tune)
to ONNX + a kokoro-onnx voices .npz.

Run inside the kokoro-onnx-export tool environment (see export_kokoro_victoria.sh):
    uv run python scripts/export_kokoro_victoria.py --out-dir models/kokoro
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
import torch
from huggingface_hub import hf_hub_download
from kokoro.model import KModel
from kokoro_onnx.cli_export import KModelForONNXWithDuration
from onnxruntime.quantization import shape_inference
from onnxruntime.quantization.quant_utils import add_infer_metadata
from torch.nn import utils

HF_REPO = "kikiri-tts/kikiri-german-victoria"
CHECKPOINT = "kikiri_german_victoria_ep10.pth"
VOICEPACK = "voices/victoria.pt"
VOICE_NAME = "victoria"
OPSET = 20
STYLE_DIM = 256


def load_model(ckpt_path: Path) -> KModel:
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    # Unwrap common training-checkpoint nestings.
    for wrap in ("net", "model", "state_dict"):
        if isinstance(sd, dict) and wrap in sd and isinstance(sd[wrap], dict):
            sd = sd[wrap]
    print(f"checkpoint top-level keys: {sorted(sd.keys())}")

    # Default config + weights from hexgrad/Kokoro-82M, then overlay fine-tune.
    model = KModel(disable_complex=True).eval()
    for key, module_sd in sd.items():
        if not hasattr(model, key):
            print(f"  ! skipping unknown module {key!r}")
            continue
        try:
            getattr(model, key).load_state_dict(module_sd)
        except RuntimeError:
            module_sd = {k.removeprefix("module."): v for k, v in module_sd.items()}
            missing, unexpected = getattr(model, key).load_state_dict(
                module_sd, strict=False
            )
            if missing or unexpected:
                print(f"  ! {key}: missing={missing} unexpected={unexpected}")
        print(f"  loaded {key}")
    return model


def remove_weight_norm_recursive(module: torch.nn.Module) -> None:
    for child in module.children():
        if hasattr(child, "weight_v"):
            utils.remove_weight_norm(child)
        else:
            remove_weight_norm_recursive(child)


def export_onnx(model: KModel, output_path: Path) -> None:
    wrapped = KModelForONNXWithDuration(model).eval()
    remove_weight_norm_recursive(wrapped)

    input_ids = torch.zeros((1, 12), dtype=torch.long)
    input_ids[0, :] = torch.LongTensor([0] + [1] * 10 + [0])
    style = torch.randn(1, STYLE_DIM)
    speed = torch.tensor([0.95], dtype=torch.float32)

    print("exporting ONNX ...")
    torch.onnx.export(
        wrapped,
        (input_ids, style, speed),
        output_path,
        # Name the token input "tokens" (not "input_ids"): the kokoro-onnx
        # runtime feeds speed as int32 to models with an "input_ids" input,
        # but as float32 (what this export expects) to "tokens" models.
        input_names=["tokens", "style", "speed"],
        output_names=["waveform", "duration"],
        dynamic_axes={
            "tokens": {1: "sequence_length"},
            "waveform": {0: "num_samples"},
        },
        opset_version=OPSET,
        export_params=True,
        do_constant_folding=True,
    )

    print("quantization pre-processing ...")
    onnx_model = onnx.load(output_path)
    shape_inference.quant_pre_process(
        onnx_model, output_model_path=str(output_path), skip_symbolic_shape=True
    )
    onnx_model = onnx.load(output_path)
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
    add_infer_metadata(onnx_model)
    onnx.save_model(onnx_model, output_path)
    onnx.checker.check_model(onnx_model)
    print(f"wrote {output_path} ({output_path.stat().st_size / 1e6:.0f} MB)")


def convert_voicepack(voice_path: Path, output_path: Path) -> np.ndarray:
    pack = torch.load(voice_path, map_location="cpu", weights_only=True)
    if isinstance(pack, dict):
        # Single-voice repos sometimes store {"victoria": tensor}.
        pack = next(iter(pack.values()))
    arr = pack.float().numpy()
    print(f"voicepack shape: {arr.shape}")  # expect (510, 1, 256)
    np.savez(output_path, **{VOICE_NAME: arr.astype(np.float32)})
    print(f"wrote {output_path}")
    return arr


def score_difference(model: KModel, onnx_path: Path, style: np.ndarray) -> None:
    import onnxruntime as ort

    tokens = [0] + list(range(10, 40)) + [0]
    input_ids = torch.LongTensor([tokens])
    ref_s = torch.from_numpy(style[len(tokens) - 2, 0][None, :].astype(np.float32))

    wrapped = KModelForONNXWithDuration(model).eval()
    with torch.no_grad():
        torch_wave, _ = wrapped(input_ids, ref_s, 1.0)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    (onnx_wave, _) = sess.run(
        None,
        {
            "tokens": input_ids.numpy(),
            "style": ref_s.numpy(),
            "speed": np.array([1.0], dtype=np.float32),
        },
    )
    n = min(len(torch_wave.squeeze()), len(onnx_wave.squeeze()))
    t, o = torch_wave.squeeze().numpy()[:n], onnx_wave.squeeze()[:n]
    print(f"torch/onnx diff: max_abs={np.abs(t - o).max():.5f} mse={((t - o) ** 2).mean():.7f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = Path(hf_hub_download(HF_REPO, CHECKPOINT))
    voice = Path(hf_hub_download(HF_REPO, VOICEPACK))

    model = load_model(ckpt)
    onnx_path = args.out_dir / "kokoro-victoria.onnx"
    export_onnx(model, onnx_path)
    style = convert_voicepack(voice, args.out_dir / "voices-victoria.npz")
    score_difference(model, onnx_path, style)


if __name__ == "__main__":
    main()
