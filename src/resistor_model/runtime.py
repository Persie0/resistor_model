from __future__ import annotations

import json
from pathlib import Path

from resistor_model.data.schema import load_manifest
from resistor_model.data.split import grouped_split
from resistor_model.models.bandnet import ResistorBandNet


def build_model(cfg: dict) -> ResistorBandNet:
    data_cfg, model_cfg = cfg["data"], cfg["model"]
    return ResistorBandNet(
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


def resolve_split_ids(manifest_path: str | Path, cfg: dict) -> dict[str, set[str]]:
    split_file = cfg["data"].get("splits_file")
    if split_file:
        payload = json.loads(Path(split_file).read_text(encoding="utf-8"))
        return {k: set(payload.get(k, [])) for k in ("train", "val", "test")}

    rows = load_manifest(manifest_path)
    has_explicit = any(row.split is not None for row in rows)
    if has_explicit:
        if any(row.split not in {"train", "val", "test"} for row in rows):
            raise ValueError("when using explicit manifest splits, every row must have split=train|val|test")
        out = {"train": set(), "val": set(), "test": set()}
        for row in rows:
            for r in row.resistors:
                out[row.split].add(r.id)
        return out

    records = [(row.image, r.id, row.session_id) for row in rows for r in row.resistors]
    ratios = tuple(float(x) for x in cfg["data"]["split_ratios"])
    split_rows = grouped_split(records, ratios=ratios, seed=int(cfg["seed"]), group_session=bool(cfg["data"]["group_session"]))
    return {name: {r[1] for r in items} for name, items in split_rows.items()}
