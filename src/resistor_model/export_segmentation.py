from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from resistor_model.segmentation import (
    ForegroundProbabilityWrapper,
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_lraspp_model,
)


def load_checkpoint_model(checkpoint_path: str | Path):
    payload = torch.load(checkpoint_path, map_location="cpu")
    architecture = payload.get("architecture")
    if architecture != "lraspp_mobilenet_v3_large":
        raise ValueError(f"Unsupported segmentation architecture: {architecture!r}")
    model = build_lraspp_model(
        num_classes=int(payload.get("num_classes", 2)),
        pretrained_backbone=False,
    )
    model.load_state_dict(payload["model"])
    model.eval()
    return model, payload


def metadata_for_checkpoint(payload: dict) -> dict:
    image_size = int(payload.get("image_size", 384))
    return {
        "architecture": "lraspp_mobilenet_v3_large",
        "task": "binary semantic segmentation",
        "input": {
            "name": "image",
            "shape": [1, 3, image_size, image_size],
            "dtype": "float32",
            "normalization": {
                "scale": "RGB uint8 / 255.0",
                "mean": list(IMAGENET_MEAN),
                "std": list(IMAGENET_STD),
            },
            "resize": "letterbox to square, black padding",
        },
        "output": {
            "name": "foreground_probability",
            "shape": [1, 1, image_size, image_size],
            "dtype": "float32",
            "meaning": "resistor foreground probability",
        },
        "threshold": 0.5,
        "category_names": payload.get("category_names"),
    }


def export_onnx(checkpoint_path: str | Path, output_path: str | Path, *, opset: int = 17) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    model, payload = load_checkpoint_model(checkpoint_path)
    image_size = int(payload.get("image_size", 384))
    wrapper = ForegroundProbabilityWrapper(model).eval()
    example = torch.zeros(1, 3, image_size, image_size, dtype=torch.float32)
    torch.onnx.export(
        wrapper,
        example,
        output,
        input_names=["image"],
        output_names=["foreground_probability"],
        dynamic_axes={
            "image": {0: "batch"},
            "foreground_probability": {0: "batch"},
        },
        opset_version=opset,
    )
    metadata_path = output.with_suffix(output.suffix + ".json")
    metadata_path.write_text(json.dumps(metadata_for_checkpoint(payload), indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the resistor LR-ASPP segmenter to ONNX")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()
    print(export_onnx(args.checkpoint, args.output, opset=args.opset))


if __name__ == "__main__":
    main()
