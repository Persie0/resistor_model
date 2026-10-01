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
    del worker_id  # worker_info.seed already includes the worker id.
    info = torch.utils.data.get_worker_info()
    if info is not None and hasattr(info.dataset, "reseed_augmenters"):
        info.dataset.reseed_augmenters(int(info.seed))


def _to_device(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _loss_weights(cfg: dict) -> LossWeights:
    raw = cfg["loss"]
    return LossWeights(**{k: float(raw[k]) for k in LossWeights.__dataclass_fields__})


def _make_loaders(cfg: dict, split_ids: dict[str, set[str]]) -> tuple[DataLoader, DataLoader]:
    d = cfg["data"]
    manifest, root = d["manifest"], d["image_root"]
    common = dict(
        manifest=manifest,
        image_root=root,
        output_size=tuple(d["output_size"]),
        sequence_bins=int(d["sequence_bins"]),
        max_bands=int(cfg["model"]["max_bands"]),
    )
    consistency = float(cfg["loss"].get("consistency", 0.0)) > 0
    train_ds = ResistorBandDataset(**common, allowed_resistor_ids=split_ids["train"], augment=True, two_views=consistency, seed=int(cfg["seed"]))
    val_ds = ResistorBandDataset(**common, allowed_resistor_ids=split_ids["val"], augment=False, two_views=False, seed=int(cfg["seed"]) + 1)
    if not train_ds:
        raise ValueError("training split is empty")
    if not val_ds:
        raise ValueError("validation split is empty; provide more resistor IDs or explicit splits")
    kwargs = dict(batch_size=int(cfg["train"]["batch_size"]), num_workers=int(d["num_workers"]), pin_memory=torch.cuda.is_available())
    generator = torch.Generator().manual_seed(int(cfg["seed"]))
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=False, generator=generator, worker_init_fn=_seed_dataset_worker, **kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **kwargs)
    return train_loader, val_loader


def _dense_weights(cfg: dict, device: torch.device) -> torch.Tensor:
    n = int(cfg["model"]["num_colors"])
    w = torch.ones(n + 1, device=device)
    w[-1] = float(cfg["loss"]["body_class_weight"])
    return w


def train_one_epoch(model, ema, loader, optimizer, scaler, device, cfg, *, max_batches: int | None = None) -> dict[str, float]:
    model.train()
    weights = _loss_weights(cfg)
    dense_weights = _dense_weights(cfg, device)
    sums: dict[str, float] = {"total": 0.0}
    count = 0
    use_amp = bool(cfg["train"]["amp"]) and device.type == "cuda"
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        batch = _to_device(batch, device)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            out = model(batch["image"])
            out2 = model(batch["image_view2"]) if "image_view2" in batch else None
            loss = compute_loss(out, batch, weights, second_view=out2, dense_class_weights=dense_weights)
        scaler.scale(loss.total).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"]["grad_clip"]))
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        ema.update(model)
        sums["total"] += float(loss.total.detach())
        for k, v in loss.parts.items():
            sums[k] = sums.get(k, 0.0) + float(v.detach())
        count += 1
    return {k: v / max(count, 1) for k, v in sums.items()}


@torch.no_grad()
def evaluate_loader(model, loader, device, cfg, *, max_batches: int | None = None) -> dict[str, float]:
    model.eval()
    acc = MetricAccumulator(int(cfg["model"]["num_colors"]), int(cfg["model"]["max_bands"]))
    weights = _loss_weights(cfg)
    dense_weights = _dense_weights(cfg, device)
    loss_sum = 0.0
    count = 0
    for step, batch in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        batch = _to_device(batch, device)
        out = model(batch["image"])
        loss = compute_loss(out, batch, weights, dense_class_weights=dense_weights)
        loss_sum += float(loss.total)
        count += 1
        acc.update(out, batch)
    metrics = acc.compute()
    metrics["loss"] = loss_sum / max(count, 1)
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Train ResistorBandNet")
    ap.add_argument("--config", type=Path, default=Path("configs/bandnet.yaml"))
    ap.add_argument("--smoke", action="store_true", help="Run only one train and validation batch")
    args = ap.parse_args()
    cfg = load_config(args.config)
    _seed_everything(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg["train"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    split_ids = resolve_split_ids(cfg["data"]["manifest"], cfg)
    (out_dir / "splits.json").write_text(json.dumps({k: sorted(v) for k, v in split_ids.items()}, indent=2), encoding="utf-8")

    train_loader, val_loader = _make_loaders(cfg, split_ids)
    model = build_model(cfg).to(device)
    ema = ModelEMA(model, float(cfg["train"]["ema_decay"]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr"]), weight_decay=float(cfg["train"]["weight_decay"]))
    epochs = 1 if args.smoke else int(cfg["train"]["epochs"])
    warmup = min(int(cfg["train"]["warmup_epochs"]), max(epochs - 1, 0))

    def lr_lambda(epoch: int) -> float:
        if warmup > 0 and epoch < warmup:
            return (epoch + 1) / warmup
        span = max(epochs - warmup, 1)
        progress = min(max((epoch - warmup) / span, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg["train"]["amp"]) and device.type == "cuda")
    start_epoch, best = 0, -1.0
    resume = cfg["train"].get("resume")
    if resume:
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt.get("raw_model_state", ckpt["model_state"]))
        ema.model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = int(ckpt["epoch"]) + 1
        best = float(ckpt.get("best_exact_sequence", -1.0))

    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        train_metrics = train_one_epoch(model, ema, train_loader, optimizer, scaler, device, cfg, max_batches=1 if args.smoke else None)
        val_metrics = evaluate_loader(ema.model, val_loader, device, cfg, max_batches=1 if args.smoke else None)
        scheduler.step()
        score = val_metrics["exact_sequence_accuracy"]
        checkpoint = {
            "epoch": epoch,
            "config": cfg,
            "model_state": ema.model.state_dict(),
            "raw_model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_exact_sequence": max(best, score),
            "splits": {k: sorted(v) for k, v in split_ids.items()},
            "val_metrics": val_metrics,
        }
        torch.save(checkpoint, out_dir / "last.pt")
        if score >= best:
            best = score
            torch.save(checkpoint, out_dir / "best.pt")
        record = {"epoch": epoch, "seconds": time.time() - t0, "lr": optimizer.param_groups[0]["lr"], "train": train_metrics, "val": val_metrics}
        with (out_dir / "metrics.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
