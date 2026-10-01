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
    },
    "model": {
        "num_colors": 12,
        "max_bands": 6,
        "base_channels": 48,
        "d_model": 256,
        "transformer_layers": 4,
        "transformer_heads": 8,
        "slot_decoder_layers": 2,
        "dropout": 0.1,
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
        "body_class_weight": 0.25,
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
    if cfg["model"]["max_bands"] < 1:
        raise ValueError("model.max_bands must be >= 1")
    if cfg["model"]["d_model"] % cfg["model"]["transformer_heads"] != 0:
        raise ValueError("model.d_model must be divisible by model.transformer_heads")
    ratios = cfg["data"]["split_ratios"]
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("data.split_ratios must contain three values summing to 1")


def load_config(path: str | Path | None = None) -> dict:
    cfg = deepcopy(DEFAULT_CONFIG)
    if path is not None:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError("config root must be a mapping")
        cfg = _deep_merge(cfg, raw)
    validate_config(cfg)
    return cfg
