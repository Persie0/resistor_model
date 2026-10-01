import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import yaml

from resistor_model.config import load_config
from resistor_model.metrics import MetricAccumulator
from resistor_model.runtime import resolve_split_ids
from resistor_model.tools.convert_yolo import convert_yolo_pair
from resistor_model.train import _advance_scheduler_for_checkpoint, _restore_scaler_state, _selection_key


def test_config_deep_merges_user_overrides(tmp_path: Path):
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump({"model": {"d_model": 96}, "train": {"epochs": 3}}), encoding="utf-8")
    cfg = load_config(p)
    assert cfg["model"]["d_model"] == 96
    assert cfg["model"]["max_bands"] == 6
    assert cfg["train"]["epochs"] == 3
    assert cfg["data"]["output_size"] == [128, 768]


def test_metric_accumulator_reports_exact_sequence_value_and_per_color_f1():
    acc = MetricAccumulator(num_colors=12, max_bands=6)
    slot_color = torch.full((2,6,12), -10.0)
    for b, seq in enumerate([[1,0,2,10],[1,1,2,10]]):
        for i, c in enumerate(seq):
            slot_color[b,i,c] = 10.0
    count_logits = torch.full((2,7), -5.0); count_logits[:,4] = 5.0
    outputs = {"slot_color_logits": slot_color, "count_logits": count_logits, "slot_center": torch.tensor([[.1,.3,.5,.8,0,0],[.1,.3,.5,.8,0,0]]), "dense_logits": torch.zeros((2,13,8))}
    outputs["dense_logits"][:,12,:] = 1.0
    targets = {"slot_colors": torch.tensor([[1,0,2,10,-100,-100],[1,0,2,10,-100,-100]]), "slot_exists": torch.tensor([[1,1,1,1,0,0],[1,1,1,1,0,0]], dtype=torch.float32), "slot_centers": torch.tensor([[.1,.3,.5,.8,0,0],[.1,.3,.5,.8,0,0]]), "count": torch.tensor([4,4]), "dense_target": torch.full((2,8),12,dtype=torch.long)}
    acc.update(outputs, targets)
    m = acc.compute()
    assert m["exact_sequence_accuracy"] == 0.5
    assert m["exact_value_accuracy"] == 0.5
    assert m["count_accuracy"] == 1.0
    assert m["position_mae"] == 0.0
    assert m["dense_accuracy"] == 1.0
    assert 0.0 <= m["macro_f1"] <= 1.0
    assert m["f1_brown"] == 0.8  # one black band is a brown false positive
    assert m["f1_red"] == 1.0
    assert m["f1_gold"] == 1.0
    assert abs(m["f1_black"] - (2.0 / 3.0)) < 1e-9


def test_band_f1_penalizes_missing_and_extra_predicted_bands():
    def run(pred_count: int, fifth_color: int = 2) -> dict[str, float]:
        acc = MetricAccumulator(num_colors=12, max_bands=6)
        slot_color = torch.full((1, 6, 12), -10.0)
        for i, c in enumerate([1, 0, 2, 10, fifth_color]):
            slot_color[0, i, c] = 10.0
        count_logits = torch.full((1, 7), -5.0); count_logits[0, pred_count] = 5.0
        outputs = {
            "slot_color_logits": slot_color,
            "count_logits": count_logits,
            "slot_center": torch.tensor([[.1, .3, .5, .8, .9, 0.0]]),
            "dense_logits": torch.zeros((1, 13, 8)),
        }
        outputs["dense_logits"][:, 12, :] = 1.0
        targets = {
            "slot_colors": torch.tensor([[1, 0, 2, 10, -100, -100]]),
            "slot_exists": torch.tensor([[1, 1, 1, 1, 0, 0]], dtype=torch.float32),
            "slot_centers": torch.tensor([[.1, .3, .5, .8, 0, 0]]),
            "count": torch.tensor([4]),
            "dense_target": torch.full((1, 8), 12, dtype=torch.long),
        }
        acc.update(outputs, targets)
        return acc.compute()

    missing = run(3)
    assert missing["f1_gold"] == 0.0
    assert missing["macro_f1"] < 1.0

    extra = run(5, fifth_color=2)
    assert abs(extra["f1_red"] - (2.0 / 3.0)) < 1e-9
    assert extra["macro_f1"] < 1.0


def test_selection_key_breaks_exact_sequence_ties_with_macro_f1():
    a = {"exact_sequence_accuracy": 0.8, "macro_f1": 0.55}
    b = {"exact_sequence_accuracy": 0.8, "macro_f1": 0.60}
    c = {"exact_sequence_accuracy": 0.81, "macro_f1": 0.10}
    assert _selection_key(b) > _selection_key(a)
    assert _selection_key(c) > _selection_key(b)


def test_restore_scaler_state_is_backward_compatible():
    class FakeScaler:
        def __init__(self):
            self.loaded = None

        def load_state_dict(self, state):
            self.loaded = state

    scaler = FakeScaler()
    _restore_scaler_state(scaler, {})
    assert scaler.loaded is None
    _restore_scaler_state(scaler, {"scaler_state": {"scale": 123.0}})
    assert scaler.loaded == {"scale": 123.0}


def test_scheduler_is_advanced_before_its_checkpoint_state_is_captured():
    class FakeScheduler:
        def __init__(self):
            self.steps = 0

        def step(self):
            self.steps += 1

        def state_dict(self):
            return {"steps": self.steps}

    scheduler = FakeScheduler()
    state = _advance_scheduler_for_checkpoint(scheduler)
    assert scheduler.steps == 1
    assert state == {"steps": 1}


def _band(color: str = "brown") -> dict:
    return {"color": color, "bbox": [1, 1, 2, 3]}


def test_explicit_manifest_splits_reject_overlapping_resistor_ids(tmp_path: Path):
    manifest = tmp_path / "manifest.jsonl"
    rows = [
        {"image": "train.jpg", "split": "train", "resistors": [{"id": "same", "bands": [_band()]}]},
        {"image": "val.jpg", "split": "val", "resistors": [{"id": "same", "bands": [_band("red")]}]},
        {"image": "test.jpg", "split": "test", "resistors": [{"id": "other", "bands": [_band("black")]}]},
    ]
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    cfg = load_config()
    cfg["data"]["splits_file"] = None
    with pytest.raises(ValueError, match="overlap"):
        resolve_split_ids(manifest, cfg)


def test_split_file_rejects_overlapping_resistor_ids(tmp_path: Path):
    split_file = tmp_path / "splits.json"
    split_file.write_text(json.dumps({"train": ["r1", "r2"], "val": ["r2"], "test": ["r3"]}), encoding="utf-8")
    cfg = load_config()
    cfg["data"]["splits_file"] = str(split_file)
    with pytest.raises(ValueError, match="overlap"):
        resolve_split_ids(tmp_path / "unused.jsonl", cfg)


def test_yolo_converter_creates_absolute_band_boxes(tmp_path: Path):
    image = np.zeros((100, 200, 3), np.uint8)
    image_path = tmp_path / "r.jpg"; label_path = tmp_path / "r.txt"
    cv2.imwrite(str(image_path), image)
    label_path.write_text("1 0.5 0.5 0.1 0.4\n", encoding="utf-8")
    row = convert_yolo_pair(image_path, label_path, ["black","brown","red"])
    band = row["resistors"][0]["bands"][0]
    assert band["color"] == "brown"
    assert np.allclose(band["bbox"], [90,30,110,70])
