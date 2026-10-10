from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 42,
    "data": {
        "manifest": "data/manifest.jsonl",
        "image_root": "data/images",
        "splits_file": None,
        "output_size": [128, 768],
        "sequence_bins": 256,
        "group_session": False,
        "split_ratios": [0.7, 0.15, 0.15],
        "num_workers": 4,
        "geometric_augment": False,
        "hflip_prob": 0.5,
        "vflip_prob": 0.5,
        "jitter_strength": 1.0,
    },
    "model": {
        "architecture": "v1",
        "backbone": "convnext_lite",
        "pretrained": False,
        "num_colors": 12,
        "max_bands": 6,
        "base_channels": 48,
        "d_model": 256,
        "transformer_layers": 4,
        "transformer_heads": 8,
        "slot_decoder_layers": 2,
        "dropout": 0.1,
        "drop_path": 0.1,
        "conv_kernel": 7,
        "use_chromatic_branch": True,
    },
    "train": {
        "epochs": 100,
        "batch_size": 32,
        "lr": 3e-4,
        "weight_decay": 0.05,
        "warmup_epochs": 5,
        "grad_clip": 1.0,
        "amp": True,
        "ema_decay": 0.999,
        "output_dir": "runs/bandnet",
        "resume": None,
    },
    "loss": {
        "dense": 1.0,
        "color": 2.0,
        "exist": 0.5,
        "center": 1.0,
        "width": 0.5,
        "order": 0.2,
        "count": 0.2,
        "consistency": 0.1,
        "ctc": 0.0,
        "kl": 0.0,
        "label_smoothing": 0.0,
        "body_class_weight": 0.25,
        "color_balance": "none",
        "max_color_weight": 4.0,
    },
    "eval": {
        "extra_decoders": False,
        "series_bonus": 0.0,
    },
    "distill": {
        "teacher_checkpoint": None,
        "weight": 0.0,
        "temperature": 3.0,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def validate_config(cfg: dict) -> None:
    h, w = cfg["data"]["output_size"]
    if h <= 0 or w <= 0:
        raise ValueError("data.output_size values must be positive")
    if cfg["data"]["sequence_bins"] <= 0:
        raise ValueError("data.sequence_bins must be positive")
    for key in ("hflip_prob", "vflip_prob"):
        value = float(cfg["data"].get(key, 0.0))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"data.{key} must be in [0, 1]")
    if float(cfg["data"].get("jitter_strength", 0.0)) < 0.0:
        raise ValueError("data.jitter_strength must be >= 0")

    model_cfg = cfg["model"]
    if model_cfg["max_bands"] < 1:
        raise ValueError("model.max_bands must be >= 1")
    if model_cfg["d_model"] % model_cfg["transformer_heads"] != 0:
        raise ValueError("model.d_model must be divisible by model.transformer_heads")
    if str(model_cfg.get("architecture", "v1")).lower() not in {"v1", "v2", "mobile1d"}:
        raise ValueError("model.architecture must be v1, v2 or mobile1d")
    if int(model_cfg.get("conv_kernel", 7)) < 1 or int(model_cfg.get("conv_kernel", 7)) % 2 == 0:
        raise ValueError("model.conv_kernel must be a positive odd integer")

    ratios = cfg["data"]["split_ratios"]
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("data.split_ratios must contain three values summing to 1")

    loss_cfg = cfg["loss"]
    for key in ("dense", "color", "exist", "center", "width", "order", "count", "consistency", "ctc", "kl"):
        if float(loss_cfg.get(key, 0.0)) < 0.0:
            raise ValueError(f"loss.{key} must be >= 0")
    smoothing = float(loss_cfg.get("label_smoothing", 0.0))
    if not 0.0 <= smoothing < 1.0:
        raise ValueError("loss.label_smoothing must be in [0, 1)")
    if str(loss_cfg.get("color_balance", "none")) not in {"none", "sqrt_inverse"}:
        raise ValueError("loss.color_balance must be none or sqrt_inverse")
    if float(loss_cfg.get("max_color_weight", 4.0)) < 1.0:
        raise ValueError("loss.max_color_weight must be >= 1")

    distill_cfg = cfg.get("distill", {})
    if float(distill_cfg.get("weight", 0.0)) < 0.0:
        raise ValueError("distill.weight must be >= 0")
    if float(distill_cfg.get("temperature", 3.0)) <= 0.0:
        raise ValueError("distill.temperature must be > 0")


def load_config(path: str | Path | None = None) -> dict:
    cfg = deepcopy(DEFAULT_CONFIG)
    if path is not None:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError("config root must be a mapping")
        cfg = _deep_merge(cfg, raw)
    validate_config(cfg)
    return cfg
