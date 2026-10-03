from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

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


def format_training_progress(
    *,
    epoch: int,
    epochs: int,
    batch: int,
    batches: int,
    loss: float,
    lr: float,
    elapsed_s: float,
) -> str:
    percent = 100.0 * batch / max(1, batches)
    return (
        f"[train] epoch {epoch}/{epochs} | batch {batch}/{batches} ({percent:.1f}%) | "
        f"loss {loss:.4f} | lr {lr:.2e} | {elapsed_s:.1f}s"
    )


def should_save_periodic_checkpoint(completed_epoch: int, total_epochs: int, every: int) -> bool:
    if every <= 0:
        return False
    return completed_epoch % every == 0 or completed_epoch == total_epochs


@torch.no_grad()
def evaluate_model(
    model,
    loader,
    device: torch.device,
    *,
    progress_every: int = 0,
    label: str = "validation",
) -> dict[str, float]:
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
    total_batches = len(loader) if hasattr(loader, "__len__") else 0
    started = time.perf_counter()
    for batch_index, (images, masks) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = model(images)["out"]
        totals["loss"] += float(segmentation_loss(logits, masks).item())
        totals["batches"] += 1
        batch = foreground_metrics(logits, masks)
        for key, value in batch.items():
            totals[key] += value

        if progress_every > 0 and total_batches and (
            batch_index % progress_every == 0 or batch_index == total_batches
        ):
            current = _finalize_metrics(totals)
            elapsed = time.perf_counter() - started
            percent = 100.0 * batch_index / total_batches
            print(
                f"[{label}] batch {batch_index}/{total_batches} ({percent:.1f}%) | "
                f"loss {current['loss']:.4f} | dice {current['dice']:.4f} | "
                f"iou {current['iou']:.4f} | {elapsed:.1f}s",
                flush=True,
            )
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


def _save_checkpoint(
    path: Path,
    *,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    best_dice: float,
    args,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": "lraspp_mobilenet_v3_large",
            "num_classes": 2,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
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
    checkpoints_dir = output_dir / "checkpoints"

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
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_dice = float(checkpoint.get("best_dice", -1.0))
        print(
            f"[resume] {args.resume} | completed epoch {start_epoch}/{args.epochs} | "
            f"best dice {best_dice:.4f}",
            flush=True,
        )

    print(
        f"[setup] device={device} | train={len(train_dataset)} | validation={len(val_dataset)} | "
        f"image={args.image_size} | batch={args.batch_size} | epochs={args.epochs}",
        flush=True,
    )
    print(f"[setup] outputs={output_dir}", flush=True)
    print(
        f"[setup] progress every {max(0, args.progress_every)} batches | "
        f"numbered checkpoint every {max(0, args.checkpoint_every)} epochs",
        flush=True,
    )

    metrics_path = output_dir / "metrics.jsonl"
    for epoch in range(start_epoch, args.epochs):
        completed_epoch = epoch + 1
        print(f"\n[epoch] {completed_epoch}/{args.epochs} starting", flush=True)
        epoch_started = time.perf_counter()
        model.train()
        running_loss = 0.0
        batches = 0
        total_batches = len(train_loader)
        for batch_index, (images, masks) in enumerate(train_loader, start=1):
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

            if args.progress_every > 0 and (
                batch_index % args.progress_every == 0 or batch_index == total_batches
            ):
                print(
                    format_training_progress(
                        epoch=completed_epoch,
                        epochs=args.epochs,
                        batch=batch_index,
                        batches=total_batches,
                        loss=running_loss / max(1, batches),
                        lr=float(optimizer.param_groups[0]["lr"]),
                        elapsed_s=time.perf_counter() - epoch_started,
                    ),
                    flush=True,
                )

        validation = evaluate_model(
            model,
            val_loader,
            device,
            progress_every=max(0, args.progress_every),
            label="validation",
        )
        scheduler.step()
        epoch_seconds = time.perf_counter() - epoch_started
        row = {
            "epoch": epoch,
            "completed_epoch": completed_epoch,
            "train_loss": running_loss / max(1, batches),
            "lr": optimizer.param_groups[0]["lr"],
            "elapsed_seconds": epoch_seconds,
            "validation": validation,
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")

        improved = validation["dice"] > best_dice
        if improved:
            best_dice = validation["dice"]
            _save_checkpoint(
                output_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_dice=best_dice,
                args=args,
            )
            print(f"[checkpoint] best.pt updated | dice {best_dice:.4f}", flush=True)

        _save_checkpoint(
            output_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_dice=best_dice,
            args=args,
        )
        print(f"[checkpoint] last.pt saved after epoch {completed_epoch}", flush=True)

        if should_save_periodic_checkpoint(completed_epoch, args.epochs, args.checkpoint_every):
            periodic = checkpoints_dir / f"epoch_{completed_epoch:03d}.pt"
            _save_checkpoint(
                periodic,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_dice=best_dice,
                args=args,
            )
            print(f"[checkpoint] saved {periodic}", flush=True)

        print(
            f"[epoch] {completed_epoch}/{args.epochs} complete | "
            f"train loss {row['train_loss']:.4f} | val loss {validation['loss']:.4f} | "
            f"dice {validation['dice']:.4f} | iou {validation['iou']:.4f} | "
            f"pixel acc {validation['pixel_accuracy']:.4f} | {epoch_seconds:.1f}s",
            flush=True,
        )

    best_checkpoint = torch.load(output_dir / "best.pt", map_location="cpu")
    model.load_state_dict(best_checkpoint["model"])
    result: dict[str, object] = {
        "best_dice": best_dice,
        "validation": evaluate_model(
            model,
            val_loader,
            device,
            progress_every=max(0, args.progress_every),
            label="validation-best",
        ),
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
        print(f"[test] evaluating {len(test_dataset)} images with best.pt", flush=True)
        result["test"] = evaluate_model(
            model,
            test_loader,
            device,
            progress_every=max(0, args.progress_every),
            label="test",
        )
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[done] summary written to {output_dir / 'summary.json'}", flush=True)
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
        "--progress-every",
        type=int,
        default=10,
        help="Print live train/validation progress every N batches; 0 disables batch progress",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=5,
        help="Also save checkpoints/epoch_NNN.pt every N completed epochs; 0 disables",
    )
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
