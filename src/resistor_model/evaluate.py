from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from resistor_model.data.dataset import ResistorBandDataset
from resistor_model.runtime import build_model
from resistor_model.train import evaluate_loader


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a ResistorBandNet checkpoint")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--image-root", type=Path, default=None)
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--batch-size", type=int, default=None)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    manifest = args.manifest or Path(cfg["data"]["manifest"])
    root = args.image_root or Path(cfg["data"]["image_root"])
    ids = set(ckpt["splits"][args.split])
    ds = ResistorBandDataset(
        manifest, root, allowed_resistor_ids=ids,
        output_size=tuple(cfg["data"]["output_size"]), sequence_bins=cfg["data"]["sequence_bins"],
        max_bands=cfg["model"]["max_bands"], augment=False,
    )
    if not ds:
        raise ValueError(f"{args.split} split is empty")
    loader = DataLoader(ds, batch_size=args.batch_size or cfg["train"]["batch_size"], shuffle=False, num_workers=cfg["data"]["num_workers"])
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    metrics = evaluate_loader(model, loader, device, cfg)
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
