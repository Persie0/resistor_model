from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from resistor_model.detection import DetectionExportWrapper, build_ssdlite_model


ARCHITECTURE = "ssdlite320_mobilenet_v3_large"


def load_checkpoint_model(checkpoint_path: str | Path):
    payload = torch.load(checkpoint_path, map_location="cpu")
    architecture = payload.get("architecture")
    if architecture != ARCHITECTURE:
        raise ValueError(f"Unsupported detection architecture: {architecture!r}")
    model = build_ssdlite_model(
        num_classes=int(payload.get("num_classes", 2)),
        pretrained_backbone=False,
    )
    model.load_state_dict(payload["model"])
    model.eval()
    return model, payload


def metadata_for_checkpoint(payload: dict) -> dict:
    return {
        "architecture": ARCHITECTURE,
        "task": "single-class resistor object detection",
        "input": {
            "name": "image",
            "shape": [1, 3, 320, 320],
            "dtype": "float32",
            "format": "RGB",
            "normalization": "uint8 / 255.0; torchvision detector normalization is inside the model",
            "resize": "letterbox or resize to 320x320 before inference; map output boxes back to source image",
        },
        "outputs": {
            "boxes": {
                "shape": ["detections", 4],
                "dtype": "float32",
                "format": "xyxy in 320x320 input pixel coordinates",
            },
            "scores": {
                "shape": ["detections"],
                "dtype": "float32",
                "meaning": "resistor confidence",
            },
            "labels": {
                "shape": ["detections"],
                "dtype": "int64",
                "meaning": "1=resistor",
            },
        },
        "recommended_score_threshold": 0.35,
        "category_names": payload.get("category_names", ["resistor"]),
        "pretrained_backbone": bool(payload.get("pretrained_backbone", False)),
        "postprocessing": "torchvision SSD postprocessing including NMS is exported in the graph",
    }


def export_onnx(
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    opset: int = 17,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    model, payload = load_checkpoint_model(checkpoint_path)
    wrapper = DetectionExportWrapper(model).eval()
    example = torch.zeros(1, 3, 320, 320, dtype=torch.float32)
    torch.onnx.export(
        wrapper,
        example,
        output,
        input_names=["image"],
        output_names=["boxes", "scores", "labels"],
        dynamic_axes={
            "boxes": {0: "detections"},
            "scores": {0: "detections"},
            "labels": {0: "detections"},
        },
        opset_version=opset,
    )
    metadata_path = output.with_suffix(output.suffix + ".json")
    metadata_path.write_text(
        json.dumps(metadata_for_checkpoint(payload), indent=2),
        encoding="utf-8",
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the resistor SSDLite box detector to ONNX")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()
    print(export_onnx(args.checkpoint, args.output, opset=args.opset))


if __name__ == "__main__":
    main()
