from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from resistor_model.config import load_config
from resistor_model.constants import COLOR_TO_INDEX
from resistor_model.data.dataset import ResistorBandDataset
from resistor_model.losses import LossWeights, compute_loss
from resistor_model.metrics import MetricAccumulator
from resistor_model.runtime import build_model, resolve_split_ids


class ModelEMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        self.decay = float(decay)
        self.model = deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        src = model.state_dict()
        dst = self.model.state_dict()
        for key, value in dst.items():
            incoming = src[key].detach()
            if value.is_floating_point():
                value.mul_(self.decay).add_(incoming, alpha=1.0 - self.decay)
            else:
                value.copy_(incoming)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _seed_dataset_worker(worker_id: int) -> None:
    """Give every worker an independent deterministic augmentation RNG stream."""
    del worker_id
    info = torch.utils.data.get_worker_info()
    if info is not None and hasattr(info.dataset, "reseed_augmenters"):
        info.dataset.reseed_augmenters(int(info.seed))


def _selection_key(metrics: dict[str, float]) -> tuple[float, float]:
    """Rank checkpoints by whole-sequence accuracy, then band macro-F1."""
    return (float(metrics["exact_sequence_accuracy"]), float(metrics["macro_f1"]))


def _restore_scaler_state(scaler, checkpoint: dict) -> None:
    """Restore AMP state when present while remaining compatible with old checkpoints."""
    state = checkpoint.get("scaler_state")
    if state is not None:
        scaler.load_state_dict(state)


def _advance_scheduler_for_checkpoint(scheduler) -> dict:
    """Advance to the next-epoch LR and return the state that must be checkpointed."""
    scheduler.step()
    return scheduler.state_dict()


def _atomic_torch_save(obj, path: str | Path) -> None:
    """Serialize a checkpoint to a sibling temp file, then atomically replace it."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + ".tmp")
    try:
        torch.save(obj, temp)
        temp.replace(target)
    finally:
        if temp.exists():
            temp.unlink()


def _progress_interval(total_steps: int, updates: int = 10) -> int:
    """Return a batch interval that yields roughly ``updates`` progress lines."""
    return max(1, math.ceil(max(int(total_steps), 1) / max(int(updates), 1)))


def _format_duration(seconds: float) -> str:
    total = max(0, int(round(float(seconds))))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _format_progress(
    *,
    phase: str,
    epoch: int,
    epochs: int,
    step: int,
    total_steps: int,
    running_loss: float,
    lr: float | None,
    elapsed: float,
) -> str:
    total = max(int(total_steps), 1)
    done = min(max(int(step), 0), total)
    percent = 100.0 * done / total
    eta = float(elapsed) * max(total - done, 0) / max(done, 1)
    parts = [
        f"[{phase}]",
        f"epoch {epoch}/{epochs}",
        f"batch {done}/{total}",
        f"{percent:.1f}%",
        f"loss {float(running_loss):.4f}",
    ]
    if lr is not None:
        parts.append(f"lr {float(lr):.3e}")
    parts.extend((f"elapsed {_format_duration(elapsed)}", f"eta {_format_duration(eta)}"))
    return " | ".join(parts)


def _to_device(batch: dict, device: torch.device) -> dict:
    return {key: (value.to(device, non_blocking=True) if torch.is_tensor(value) else value) for key, value in batch.items()}


def _loss_weights(cfg: dict) -> LossWeights:
    raw = cfg["loss"]
    return LossWeights(**{key: float(raw.get(key, field.default)) for key, field in LossWeights.__dataclass_fields__.items()})


def _make_loaders(cfg: dict, split_ids: dict[str, set[str]]) -> tuple[DataLoader, DataLoader]:
    data_cfg = cfg["data"]
    common = dict(
        manifest=data_cfg["manifest"],
        image_root=data_cfg["image_root"],
        output_size=tuple(data_cfg["output_size"]),
        sequence_bins=int(data_cfg["sequence_bins"]),
        max_bands=int(cfg["model"]["max_bands"]),
    )
    two_views = float(cfg["loss"].get("consistency", 0.0)) > 0 or float(cfg["loss"].get("kl", 0.0)) > 0
    train_ds = ResistorBandDataset(
        **common,
        allowed_resistor_ids=split_ids["train"],
        augment=True,
        two_views=two_views,
        seed=int(cfg["seed"]),
        geometric_augment=bool(data_cfg.get("geometric_augment", False)),
        hflip_prob=float(data_cfg.get("hflip_prob", 0.5)),
        vflip_prob=float(data_cfg.get("vflip_prob", 0.5)),
        jitter_strength=float(data_cfg.get("jitter_strength", 1.0)),
    )
    val_ds = ResistorBandDataset(
        **common,
        allowed_resistor_ids=split_ids["val"],
        augment=False,
        two_views=False,
        seed=int(cfg["seed"]) + 1,
        geometric_augment=False,
    )
    if not train_ds:
        raise ValueError("training split is empty")
    if not val_ds:
        raise ValueError("validation split is empty; provide more resistor IDs or explicit splits")
    kwargs = dict(
        batch_size=int(cfg["train"]["batch_size"]),
        num_workers=int(data_cfg["num_workers"]),
        pin_memory=torch.cuda.is_available(),
    )
    generator = torch.Generator().manual_seed(int(cfg["seed"]))
    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        drop_last=False,
        generator=generator,
        worker_init_fn=_seed_dataset_worker,
        **kwargs,
    )
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **kwargs)
    return train_loader, val_loader


def _dense_weights(cfg: dict, device: torch.device) -> torch.Tensor:
    num_colors = int(cfg["model"]["num_colors"])
    weights = torch.ones(num_colors + 1, device=device)
    weights[-1] = float(cfg["loss"]["body_class_weight"])
    return weights


def _color_weights(dataset: ResistorBandDataset, cfg: dict, device: torch.device) -> torch.Tensor | None:
    mode = str(cfg["loss"].get("color_balance", "none"))
    if mode == "none":
        return None
    num_colors = int(cfg["model"]["num_colors"])
    counts = torch.zeros(num_colors, dtype=torch.float64)
    for sample in dataset.samples:
        for band in sample.resistor.bands:
            index = COLOR_TO_INDEX.get(band.color)
            if index is not None and index < num_colors:
                counts[index] += 1.0
    present = counts > 0
    if not bool(present.any()):
        return None
    total = counts[present].sum()
    weights = torch.ones(num_colors, dtype=torch.float64)
    weights[present] = torch.sqrt(total / (counts[present] * present.sum()))
    weights[present] /= weights[present].mean().clamp_min(1e-12)
    weights = weights.clamp(max=float(cfg["loss"].get("max_color_weight", 4.0)))
    return weights.to(device=device, dtype=torch.float32)


def train_one_epoch(
    model,
    ema,
    loader,
    optimizer,
    scaler,
    device,
    cfg,
    *,
    max_batches: int | None = None,
    epoch: int = 1,
    epochs: int = 1,
    show_progress: bool = True,
) -> dict[str, float]:
    model.train()
    weights = _loss_weights(cfg)
    dense_weights = _dense_weights(cfg, device)
    color_weights = _color_weights(loader.dataset, cfg, device)
    sums: dict[str, float] = {"total": 0.0}
    count = 0
    use_amp = bool(cfg["train"]["amp"]) and device.type == "cuda"
    optimizer.zero_grad(set_to_none=True)
    total_steps = len(loader)
    if max_batches is not None:
        total_steps = min(total_steps, max(int(max_batches), 0))
    interval = _progress_interval(total_steps)
    started = time.time()
    for step, batch in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        batch = _to_device(batch, device)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            out = model(batch["image"])
            out2 = model(batch["image_view2"]) if "image_view2" in batch else None
            loss = compute_loss(
                out,
                batch,
                weights,
                second_view=out2,
                dense_class_weights=dense_weights,
                color_class_weights=color_weights,
            )
        scaler.scale(loss.total).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"]["grad_clip"]))
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        ema.update(model)
        sums["total"] += float(loss.total.detach())
        for key, value in loss.parts.items():
            sums[key] = sums.get(key, 0.0) + float(value.detach())
        count += 1
        if show_progress and (count % interval == 0 or count == total_steps):
            print(
                _format_progress(
                    phase="train",
                    epoch=epoch,
                    epochs=epochs,
                    step=count,
                    total_steps=total_steps,
                    running_loss=sums["total"] / max(count, 1),
                    lr=float(optimizer.param_groups[0]["lr"]),
                    elapsed=time.time() - started,
                ),
                flush=True,
            )
    return {key: value / max(count, 1) for key, value in sums.items()}


@torch.no_grad()
def evaluate_loader(
    model,
    loader,
    device,
    cfg,
    *,
    max_batches: int | None = None,
    epoch: int = 1,
    epochs: int = 1,
    show_progress: bool = True,
) -> dict[str, float]:
    model.eval()
    eval_cfg = cfg.get("eval", {})
    acc = MetricAccumulator(
        int(cfg["model"]["num_colors"]),
        int(cfg["model"]["max_bands"]),
        extra_decoders=bool(eval_cfg.get("extra_decoders", False)),
        series_bonus=float(eval_cfg.get("series_bonus", 0.0)),
    )
    weights = _loss_weights(cfg)
    dense_weights = _dense_weights(cfg, device)
    color_weights = _color_weights(loader.dataset, cfg, device)
    loss_sum = 0.0
    count = 0
    total_steps = len(loader)
    if max_batches is not None:
        total_steps = min(total_steps, max(int(max_batches), 0))
    interval = _progress_interval(total_steps)
    started = time.time()
    for step, batch in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        batch = _to_device(batch, device)
        out = model(batch["image"])
        loss = compute_loss(
            out,
            batch,
            weights,
            dense_class_weights=dense_weights,
            color_class_weights=color_weights,
        )
        loss_sum += float(loss.total)
        count += 1
        acc.update(out, batch)
        if show_progress and (count % interval == 0 or count == total_steps):
            print(
                _format_progress(
                    phase="val",
                    epoch=epoch,
                    epochs=epochs,
                    step=count,
                    total_steps=total_steps,
                    running_loss=loss_sum / max(count, 1),
                    lr=None,
                    elapsed=time.time() - started,
                ),
                flush=True,
            )
    metrics = acc.compute()
    metrics["loss"] = loss_sum / max(count, 1)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ResistorBandNet")
    parser.add_argument("--config", type=Path, default=Path("configs/bandnet.yaml"))
    parser.add_argument("--smoke", action="store_true", help="Run only one train and validation batch")
    args = parser.parse_args()
    cfg = load_config(args.config)
    _seed_everything(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg["train"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    split_ids = resolve_split_ids(cfg["data"]["manifest"], cfg)
    (out_dir / "splits.json").write_text(
        json.dumps({key: sorted(value) for key, value in split_ids.items()}, indent=2),
        encoding="utf-8",
    )

    train_loader, val_loader = _make_loaders(cfg, split_ids)
    model = build_model(cfg).to(device)
    ema = ModelEMA(model, float(cfg["train"]["ema_decay"]))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    epochs = 1 if args.smoke else int(cfg["train"]["epochs"])
    warmup = min(int(cfg["train"]["warmup_epochs"]), max(epochs - 1, 0))

    def lr_lambda(epoch: int) -> float:
        if warmup > 0 and epoch < warmup:
            return (epoch + 1) / warmup
        span = max(epochs - warmup, 1)
        progress = min(max((epoch - warmup) / span, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=bool(cfg["train"]["amp"]) and device.type == "cuda"
    )
    start_epoch = 0
    best_key = (-1.0, -1.0)
    resume = cfg["train"].get("resume")
    if resume:
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint.get("raw_model_state", checkpoint["model_state"]))
        ema.model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        _restore_scaler_state(scaler, checkpoint)
        start_epoch = int(checkpoint["epoch"]) + 1
        best_key = (
            float(checkpoint.get("best_exact_sequence", -1.0)),
            float(checkpoint.get("best_macro_f1", -1.0)),
        )

    print(
        f"Training on {device} | epochs {start_epoch + 1}-{epochs} | "
        f"train batches {len(train_loader)} | val batches {len(val_loader)}",
        flush=True,
    )
    for epoch in range(start_epoch, epochs):
        started = time.time()
        lr_used = float(optimizer.param_groups[0]["lr"])
        print(
            f"[epoch {epoch + 1}/{epochs}] start | lr {lr_used:.3e} | "
            f"train batches {1 if args.smoke else len(train_loader)} | "
            f"val batches {1 if args.smoke else len(val_loader)}",
            flush=True,
        )
        train_metrics = train_one_epoch(
            model,
            ema,
            train_loader,
            optimizer,
            scaler,
            device,
            cfg,
            max_batches=1 if args.smoke else None,
            epoch=epoch + 1,
            epochs=epochs,
        )
        val_metrics = evaluate_loader(
            ema.model,
            val_loader,
            device,
            cfg,
            max_batches=1 if args.smoke else None,
            epoch=epoch + 1,
            epochs=epochs,
        )
        current_key = _selection_key(val_metrics)
        is_best = current_key > best_key
        if is_best:
            best_key = current_key

        scheduler_state = _advance_scheduler_for_checkpoint(scheduler)
        checkpoint = {
            "epoch": epoch,
            "config": cfg,
            "model_state": ema.model.state_dict(),
            "raw_model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler_state,
            "scaler_state": scaler.state_dict(),
            "best_exact_sequence": best_key[0],
            "best_macro_f1": best_key[1],
            "splits": {key: sorted(value) for key, value in split_ids.items()},
            "val_metrics": val_metrics,
        }
        _atomic_torch_save(checkpoint, out_dir / "last.pt")
        if is_best:
            _atomic_torch_save(checkpoint, out_dir / "best.pt")
        record = {
            "epoch": epoch,
            "seconds": time.time() - started,
            "lr": lr_used,
            "train": train_metrics,
            "val": val_metrics,
        }
        with (out_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
