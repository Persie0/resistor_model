import json
from pathlib import Path

import numpy as np

from resistor_model.data.schema import load_manifest
from resistor_model.data.split import grouped_split
from resistor_model.data.geometry import estimate_axis_angle, rectify_resistor
from resistor_model.decoder import decode_resistor


def test_load_manifest_supports_multiple_resistors(tmp_path: Path):
    p = tmp_path / "manifest.jsonl"
    row = {"image": "a.jpg", "session_id": "s1", "resistors": [{"id": "r1", "bands": [{"color": "brown", "bbox": [10, 10, 20, 30]}]}, {"id": "r2", "bands": [{"color": "red", "bbox": [30, 10, 40, 30]}]}]}
    p.write_text(json.dumps(row) + "\n", encoding="utf-8")
    rows = load_manifest(p)
    assert len(rows) == 1
    assert [r.id for r in rows[0].resistors] == ["r1", "r2"]
    assert rows[0].resistors[0].bands[0].color == "brown"


def test_grouped_split_never_leaks_resistor_ids():
    records = [(f"img_{i}.jpg", f"r{i // 3}", f"s{i // 6}") for i in range(60)]
    splits = grouped_split(records, ratios=(0.7, 0.15, 0.15), seed=9, group_session=False)
    ids = {name: {r[1] for r in rows} for name, rows in splits.items()}
    assert ids["train"].isdisjoint(ids["val"])
    assert ids["train"].isdisjoint(ids["test"])
    assert ids["val"].isdisjoint(ids["test"])
    assert sum(len(v) for v in splits.values()) == len(records)


def test_grouped_split_can_group_capture_session_too():
    records = [("a.jpg", "r1", "s1"), ("b.jpg", "r2", "s1"), ("c.jpg", "r3", "s2")]
    splits = grouped_split(records, ratios=(0.34, 0.33, 0.33), seed=1, group_session=True)
    where = {}
    for split, rows in splits.items():
        for _, _, session in rows:
            where.setdefault(session, set()).add(split)
    assert all(len(v) == 1 for v in where.values())


def test_axis_angle_is_undirected_and_handles_vertical():
    horizontal = np.array([[10, 20], [30, 20], [50, 20]], dtype=np.float32)
    vertical = np.array([[20, 10], [20, 30], [20, 50]], dtype=np.float32)
    a = estimate_axis_angle(horizontal)
    b = estimate_axis_angle(vertical)
    assert min(abs(a), abs(abs(a) - 180.0)) < 1e-3
    assert abs(abs(b) - 90.0) < 1e-3


def test_rectification_makes_vertical_bands_horizontal():
    image = np.zeros((120, 100, 3), dtype=np.uint8); image[:] = 180
    bands = [{"color": "brown", "bbox": [42, 20, 58, 30]}, {"color": "black", "bbox": [42, 45, 58, 55]}, {"color": "red", "bbox": [42, 70, 58, 80]}, {"color": "gold", "bbox": [42, 95, 58, 105]}]
    crop, transformed = rectify_resistor(image, bands, output_size=(128, 768))
    assert crop.shape == (128, 768, 3)
    centers = [((b["bbox"][0] + b["bbox"][2]) / 2, (b["bbox"][1] + b["bbox"][3]) / 2) for b in transformed]
    xs = [p[0] for p in centers]; ys = [p[1] for p in centers]
    assert max(xs) - min(xs) > 400
    assert max(ys) - min(ys) < 5


def test_decoder_handles_both_directions_and_invalid_sequences():
    forward = decode_resistor(["brown", "black", "red", "gold"])
    reverse = decode_resistor(["gold", "red", "black", "brown"])
    invalid = decode_resistor(["gold", "gold", "gold", "gold"])
    assert forward.valid and forward.ohms == 1000 and forward.tolerance_percent == 5
    assert reverse.valid and reverse.ohms == 1000 and reverse.colors == ["brown", "black", "red", "gold"]
    assert not invalid.valid
