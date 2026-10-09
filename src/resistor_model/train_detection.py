from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from resistor_model.detection import (
    CocoResistorDetectionDataset,
    build_ssdlite_model,
    collate_detection_batch,
)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def should_save_periodic_checkpoint(completed_epoch: int, total_epochs: int, every: int) -> bool:
    if every <= 0:
        return False
    return completed_epoch % every == 0 or completed_epoch == total_epochs


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


def _make_loader(
    dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
):
    if batch_size <= 0:
        raise ValueError("Detection batch size must be positive")
    if shuffle and len(dataset) < 2:
        raise ValueError("SSDLite training requires at least two images for BatchNorm")

    effective_batch_size = min(batch_size, len(dataset)) if shuffle else batch_size
    # SSDLite has BatchNorm on a 1x1 feature map, so its training batches
    # cannot contain only one sample. Drop just a singleton final batch.
    drop_last = bool(
        shuffle
        and len(dataset) > effective_batch_size
        and len(dataset) % effective_batch_size == 1
    )
    if drop_last:
        print(
            f"[loader] excluding singleton tail batch "
            f"({len(dataset)} samples, batch size {effective_batch_size})",
            flush=True,
        )
    return DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        collate_fn=collate_detection_batch,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )


def _target_to_device(target: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in target.items()}


def _float_metric(value) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        raise ValueError(f"Expected scalar metric tensor, got shape {tuple(value.shape)}")
    return float(value)


@torch.no_grad()
def evaluate_map(
    model,
    loader,
    device: torch.device,
    *,
    progress_every: int = 0,
    label: str = "validation",
) -> dict[str, float]:
    try:
        from torchmetrics.detection.mean_ap import MeanAveragePrecision
    except ImportError as exc:
        raise RuntimeError(
            "Detection evaluation requires torchmetrics and pycocotools. "
            "Install resistor-model[detection]."
        ) from exc

    metric = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        class_metrics=False,
    )
    model.eval()
    total_batches = len(loader)
    started = time.perf_counter()
    for batch_index, (images, targets) in enumerate(loader, start=1):
        device_images = [image.to(device, non_blocking=True) for image in images]
        predictions = model(device_images)
        cpu_predictions = [
            {
                "boxes": prediction["boxes"].detach().cpu(),
                "scores": prediction["scores"].detach().cpu(),
                "labels": prediction["labels"].detach().cpu(),
            }
            for prediction in predictions
        ]
        cpu_targets = [
            {
                "boxes": target["boxes"].detach().cpu(),
                "labels": target["labels"].detach().cpu(),
            }
            for target in targets
        ]
        metric.update(cpu_predictions, cpu_targets)

        if progress_every > 0 and (
            batch_index % progress_every == 0 or batch_index == total_batches
        ):
            elapsed = time.perf_counter() - started
            percent = 100.0 * batch_index / max(1, total_batches)
            print(
                f"[{label}] batch {batch_index}/{total_batches} ({percent:.1f}%) | "
                f"elapsed {elapsed:.1f}s",
                flush=True,
            )

    result = metric.compute()
    return {
        "map": _float_metric(result["map"]),
        "map_50": _float_metric(result["map_50"]),
        "map_75": _float_metric(result["map_75"]),
        "mar_100": _float_metric(result["mar_100"]),
    }


def _save_checkpoint(
    path: Path,
    *,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    best_map: float,
    args,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": "ssdlite320_mobilenet_v3_large",
            "task": "single-class object detection",
            "num_classes": 2,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": int(epoch),
            "best_map": float(best_map),
            "image_size": 320,
            "category_names": ["resistor"],
            "pretrained_backbone": bool(args.pretrained_backbone),
        },
        path,
    )


def train(args: argparse.Namespace) -> dict[str, object]:
    _seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"

    print(f"[startup] device={device} | output={output_dir}", flush=True)
    train_annotation = Path(args.dataset_root) / "train" / "_annotations.coco.json"
    val_annotation = Path(args.dataset_root) / "valid" / "_annotations.coco.json"
    print(
        f"[startup] loading train COCO bbox metadata | "
        f"{train_annotation.stat().st_size / 1024 / 1024:.1f} MiB",
        flush=True,
    )
    train_dataset = CocoResistorDetectionDataset(
        args.dataset_root,
        "train",
        augment=True,
        progress=True,
    )
    print(f"[startup] train dataset ready | {len(train_dataset)} images", flush=True)
    print(
        f"[startup] loading validation COCO bbox metadata | "
        f"{val_annotation.stat().st_size / 1024 / 1024:.1f} MiB",
        flush=True,
    )
    val_dataset = CocoResistorDetectionDataset(
        args.dataset_root,
        "valid",
        augment=False,
        progress=True,
    )
    print(f"[startup] validation dataset ready | {len(val_dataset)} images", flush=True)
    print("[startup] creating DataLoaders", flush=True)
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

    print("[startup] building SSDLite320 model on CPU", flush=True)
    model = build_ssdlite_model(
        num_classes=2,
        pretrained_backbone=args.pretrained_backbone,
    )
    print("[startup] model built; moving model to device", flush=True)
    model = model.to(device)
    print(f"[startup] model ready on {device}", flush=True)
    print("[startup] creating optimizer/scheduler/AMP scaler", flush=True)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs),
    )

    scaler_enabled = args.amp and device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

    start_epoch = 0
    best_map = -1.0
    if args.resume:
        resume_path = Path(args.resume)
        print(
            f"[startup] loading resume checkpoint {resume_path} "
            f"({resume_path.stat().st_size / 1024 / 1024:.1f} MiB)",
            flush=True,
        )
        checkpoint = torch.load(resume_path, map_location="cpu")
        print("[startup] resume checkpoint deserialized; restoring state", flush=True)
        if checkpoint.get("architecture") != "ssdlite320_mobilenet_v3_large":
            raise ValueError(
                f"Unsupported resume architecture: {checkpoint.get('architecture')!r}"
            )
        model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_map = float(checkpoint.get("best_map", -1.0))
        print("[startup] resume state restored", flush=True)
        print(
            f"[resume] {args.resume} | completed epoch {start_epoch}/{args.epochs} | "
            f"best mAP {best_map:.4f}",
            flush=True,
        )

    print(
        f"[setup] model=ssdlite320_mobilenet_v3_large | device={device} | "
        f"train={len(train_dataset)} | validation={len(val_dataset)} | "
        f"input=320x320 | batch={args.batch_size} | epochs={args.epochs}",
        flush=True,
    )
    print(
        f"[setup] pretrained backbone={args.pretrained_backbone} | outputs={output_dir}",
        flush=True,
    )
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

        for batch_index, (images, targets) in enumerate(train_loader, start=1):
            device_images = [image.to(device, non_blocking=True) for image in images]
            device_targets = [_target_to_device(target, device) for target in targets]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                enabled=args.amp and device.type == "cuda",
            ):
                loss_dict = model(device_images, device_targets)
                loss = sum(loss_dict.values())

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite detection loss at epoch {completed_epoch}, "
                    f"batch {batch_index}: {float(loss.detach().cpu())}"
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.detach().cpu().item())
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

        validation = evaluate_map(
            model,
            val_loader,
            device,
            progress_every=max(0, args.progress_every),
            label="validation",
        )
        scheduler.step()
        epoch_seconds = time.perf_counter() - epoch_started
        train_loss = running_loss / max(1, batches)
        row = {
            "epoch": epoch,
            "completed_epoch": completed_epoch,
            "train_loss": train_loss,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "elapsed_seconds": epoch_seconds,
            "validation": validation,
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")

        improved = validation["map"] > best_map
        if improved:
            best_map = validation["map"]
            _save_checkpoint(
                output_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_map=best_map,
                args=args,
            )
            print(
                f"[checkpoint] best.pt updated | mAP {best_map:.4f} | "
                f"mAP50 {validation['map_50']:.4f}",
                flush=True,
            )

        _save_checkpoint(
            output_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_map=best_map,
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
                best_map=best_map,
                args=args,
            )
            print(f"[checkpoint] saved {periodic}", flush=True)

        print(
            f"[epoch] {completed_epoch}/{args.epochs} complete | train loss {train_loss:.4f} | "
            f"mAP {validation['map']:.4f} | mAP50 {validation['map_50']:.4f} | "
            f"mAP75 {validation['map_75']:.4f} | AR100 {validation['mar_100']:.4f} | "
            f"{epoch_seconds:.1f}s",
            flush=True,
        )

    best_path = output_dir / "best.pt"
    if not best_path.is_file():
        raise RuntimeError(
            "No best.pt exists. If resuming, ensure requested epochs exceed the checkpoint epoch."
        )
    best_checkpoint = torch.load(best_path, map_location="cpu")
    model.load_state_dict(best_checkpoint["model"])
    result: dict[str, object] = {
        "best_map": best_map,
        "validation": evaluate_map(
            model,
            val_loader,
            device,
            progress_every=max(0, args.progress_every),
            label="validation-best",
        ),
    }

    try:
        test_dataset = CocoResistorDetectionDataset(args.dataset_root, "test", augment=False)
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
        result["test"] = evaluate_map(
            model,
            test_loader,
            device,
            progress_every=max(0, args.progress_every),
            label="test",
        )

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[done] summary written to {summary_path}", flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a one-class resistor box detector with SSDLite320 MobileNetV3-Large"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="COCO root containing train/valid/test and _annotations.coco.json files",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs/resistor_detector"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--pretrained-backbone", action="store_true")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
