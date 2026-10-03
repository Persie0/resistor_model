from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from resistor_model.segmentation import (
    CocoResistorSegmentationDataset,
    build_lraspp_model,
    segmentation_loss,
)


def foreground_metrics(logits: torch.Tensor, target: torch.Tensor) -> dict[str, int]:
    prediction = logits.argmax(dim=1) == 1
    truth = target == 1
    intersection = int((prediction & truth).sum().item())
    predicted = int(prediction.sum().item())
    target_count = int(truth.sum().item())
    union = int((prediction | truth).sum().item())
    correct = int((prediction == truth).sum().item())
    pixels = int(truth.numel())
    return {
        "intersection": intersection,
        "predicted": predicted,
        "target": target_count,
        "union": union,
        "correct": correct,
        "pixels": pixels,
    }


def _finalize_metrics(counts: dict[str, float]) -> dict[str, float]:
    intersection = counts["intersection"]
    predicted = counts["predicted"]
    target = counts["target"]
    union = counts["union"]
    return {
        "loss": float(counts.get("loss", 0.0) / max(1.0, counts.get("batches", 1.0))),
        "iou": float(intersection / union) if union else 1.0,
        "dice": float((2.0 * intersection) / (predicted + target)) if (predicted + target) else 1.0,
        "pixel_accuracy": float(counts["correct"] / counts["pixels"]) if counts["pixels"] else 1.0,
    }


@torch.no_grad()
def evaluate_model(model, loader, device: torch.device) -> dict[str, float]:
    model.eval()
    totals = {
        "loss": 0.0,
        "batches": 0.0,
        "intersection": 0.0,
        "predicted": 0.0,
        "target": 0.0,
        "union": 0.0,
        "correct": 0.0,
        "pixels": 0.0,
    }
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = model(images)["out"]
        totals["loss"] += float(segmentation_loss(logits, masks).item())
        totals["batches"] += 1
        batch = foreground_metrics(logits, masks)
        for key, value in batch.items():
            totals[key] += value
    return _finalize_metrics(totals)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_loader(dataset, *, batch_size: int, shuffle: bool, num_workers: int, device: torch.device):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )


def _save_checkpoint(path: Path, *, model, optimizer, scheduler, epoch: int, best_dice: float, args) -> None:
    torch.save(
        {
            "architecture": "lraspp_mobilenet_v3_large",
            "num_classes": 2,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_dice": best_dice,
            "image_size": int(args.image_size),
            "category_names": list(args.category) if args.category else None,
        },
        path,
    )


def train(args: argparse.Namespace) -> dict[str, object]:
    _seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = CocoResistorSegmentationDataset(
        args.dataset_root,
        "train",
        image_size=args.image_size,
        augment=True,
        category_names=args.category,
    )
    val_dataset = CocoResistorSegmentationDataset(
        args.dataset_root,
        "valid",
        image_size=args.image_size,
        augment=False,
        category_names=args.category,
    )
    train_loader = _make_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        device=device,
    )
    val_loader = _make_loader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        device=device,
    )

    model = build_lraspp_model(
        num_classes=2,
        pretrained_backbone=args.pretrained_backbone,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler_enabled = args.amp and device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    except (AttributeError, TypeError):  # torch 2.2 compatibility
        scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
    start_epoch = 0
    best_dice = -1.0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_dice = float(checkpoint.get("best_dice", -1.0))

    metrics_path = output_dir / "metrics.jsonl"
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        batches = 0
        for images, masks in train_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
                logits = model(images)["out"]
                loss = segmentation_loss(logits, masks, foreground_weight=args.foreground_weight)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item())
            batches += 1
        scheduler.step()

        validation = evaluate_model(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": running_loss / max(1, batches),
            "lr": optimizer.param_groups[0]["lr"],
            "validation": validation,
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

        if validation["dice"] > best_dice:
            best_dice = validation["dice"]
            _save_checkpoint(
                output_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_dice=best_dice,
                args=args,
            )
        _save_checkpoint(
            output_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_dice=best_dice,
            args=args,
        )

    best_checkpoint = torch.load(output_dir / "best.pt", map_location="cpu")
    model.load_state_dict(best_checkpoint["model"])
    result: dict[str, object] = {
        "best_dice": best_dice,
        "validation": evaluate_model(model, val_loader, device),
    }
    try:
        test_dataset = CocoResistorSegmentationDataset(
            args.dataset_root,
            "test",
            image_size=args.image_size,
            augment=False,
            category_names=args.category,
        )
    except (FileNotFoundError, ValueError):
        test_dataset = None
    if test_dataset is not None:
        test_loader = _make_loader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            device=device,
        )
        result["test"] = evaluate_model(model, test_loader, device)
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train binary whole-resistor segmentation with LR-ASPP MobileNetV3-Large"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Extracted Roboflow COCO-segmentation root containing train/valid/test",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs/resistor_segmentation"))
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--foreground-weight", type=float, default=2.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--category",
        action="append",
        help=(
            "COCO category name to treat as foreground; repeat as needed. "
            "Default: resistor-named categories when present, otherwise all categories"
        ),
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--pretrained-backbone",
        action="store_true",
        help=(
            "Opt in to torchvision ImageNet weights. Torchvision warns pretrained weights may have "
            "dataset-derived terms; verify them for your use case."
        ),
    )
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(amp=True)
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = train(args)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
