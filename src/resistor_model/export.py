from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from resistor_model.runtime import build_model


class ExportWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor):
        out = self.model(image)
        return (
            out["slot_exist_logits"], out["slot_color_logits"], out["slot_center"],
            out["slot_width"], out["count_logits"], out["dense_logits"],
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Export ResistorBandNet checkpoint to ONNX")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--opset", type=int, default=18)
    args = ap.parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = build_model(cfg).eval()
    model.load_state_dict(ckpt["model_state"])
    h, w = cfg["data"]["output_size"]
    dummy = torch.zeros(1, 3, h, w)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        ExportWrapper(model), dummy, args.output,
        input_names=["image"],
        output_names=["slot_exist_logits", "slot_color_logits", "slot_center", "slot_width", "count_logits", "dense_logits"],
        dynamic_axes={"image": {0: "batch"}, "slot_exist_logits": {0: "batch"}, "slot_color_logits": {0: "batch"}, "slot_center": {0: "batch"}, "slot_width": {0: "batch"}, "count_logits": {0: "batch"}, "dense_logits": {0: "batch"}},
        opset_version=args.opset,
        dynamo=False,
    )
    metadata = {"config": cfg, "outputs": ["slot_exist_logits", "slot_color_logits", "slot_center", "slot_width", "count_logits", "dense_logits"]}
    args.output.with_suffix(args.output.suffix + ".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"exported {args.output}")


if __name__ == "__main__":
    main()
