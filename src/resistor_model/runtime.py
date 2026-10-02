from __future__ import annotations

import json
from pathlib import Path

from torch import nn

from resistor_model.data.schema import load_manifest
from resistor_model.data.split import grouped_split
from resistor_model.models.bandnet import ResistorBandNet
from resistor_model.models.bandnet_v2 import ResistorBandNetV2


_SPLIT_NAMES = ("train", "val", "test")


def build_model(cfg: dict) -> nn.Module:
    data_cfg, model_cfg = cfg["data"], cfg["model"]
    common = dict(
        num_colors=model_cfg["num_colors"],
        max_bands=model_cfg["max_bands"],
        sequence_bins=data_cfg["sequence_bins"],
        base_channels=model_cfg["base_channels"],
        d_model=model_cfg["d_model"],
        transformer_layers=model_cfg["transformer_layers"],
        transformer_heads=model_cfg["transformer_heads"],
        slot_decoder_layers=model_cfg["slot_decoder_layers"],
        dropout=model_cfg["dropout"],
        use_chromatic_branch=model_cfg["use_chromatic_branch"],
    )
    architecture = str(model_cfg.get("architecture", "v1")).lower()
    if architecture == "v1":
        return ResistorBandNet(**common)
    if architecture == "v2":
        return ResistorBandNetV2(
            **common,
            backbone=str(model_cfg.get("backbone", "convnext_lite")),
            pretrained=bool(model_cfg.get("pretrained", False)),
            drop_path=float(model_cfg.get("drop_path", 0.1)),
            conv_kernel=int(model_cfg.get("conv_kernel", 7)),
        )
    raise ValueError(f"unsupported model architecture: {architecture!r}")


def _validate_disjoint_split_ids(split_ids: dict[str, set[str]]) -> dict[str, set[str]]:
    normalized = {name: set(split_ids.get(name, set())) for name in _SPLIT_NAMES}
    for i, left in enumerate(_SPLIT_NAMES):
        for right in _SPLIT_NAMES[i + 1:]:
            overlap = normalized[left] & normalized[right]
            if overlap:
                sample = ", ".join(sorted(overlap)[:5])
                suffix = " ..." if len(overlap) > 5 else ""
                raise ValueError(
                    f"split resistor ID overlap between {left} and {right}: {sample}{suffix}"
                )
    return normalized


def resolve_split_ids(manifest_path: str | Path, cfg: dict) -> dict[str, set[str]]:
    split_file = cfg["data"].get("splits_file")
    if split_file:
        payload = json.loads(Path(split_file).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("split file root must be a mapping")
        return _validate_disjoint_split_ids({k: set(payload.get(k, [])) for k in _SPLIT_NAMES})

    rows = load_manifest(manifest_path)
    has_explicit = any(row.split is not None for row in rows)
    if has_explicit:
        if any(row.split not in _SPLIT_NAMES for row in rows):
            raise ValueError("when using explicit manifest splits, every row must have split=train|val|test")
        out = {"train": set(), "val": set(), "test": set()}
        for row in rows:
            for resistor in row.resistors:
                out[row.split].add(resistor.id)
        return _validate_disjoint_split_ids(out)

    records = [(row.image, resistor.id, row.session_id) for row in rows for resistor in row.resistors]
    ratios = tuple(float(value) for value in cfg["data"]["split_ratios"])
    split_rows = grouped_split(
        records,
        ratios=ratios,
        seed=int(cfg["seed"]),
        group_session=bool(cfg["data"]["group_session"]),
    )
    return _validate_disjoint_split_ids(
        {name: {record[1] for record in items} for name, items in split_rows.items()}
    )
