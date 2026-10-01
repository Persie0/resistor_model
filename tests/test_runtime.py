from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from resistor_model.config import load_config
from resistor_model.metrics import MetricAccumulator
from resistor_model.tools.convert_yolo import convert_yolo_pair


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
    assert m["f1_brown"] == 1.0
    assert m["f1_red"] == 1.0
    assert m["f1_gold"] == 1.0
    assert m["f1_black"] < 1.0


def test_yolo_converter_creates_absolute_band_boxes(tmp_path: Path):
    image = np.zeros((100, 200, 3), np.uint8)
    image_path = tmp_path / "r.jpg"; label_path = tmp_path / "r.txt"
    cv2.imwrite(str(image_path), image)
    label_path.write_text("1 0.5 0.5 0.1 0.4\n", encoding="utf-8")
    row = convert_yolo_pair(image_path, label_path, ["black","brown","red"])
    band = row["resistors"][0]["bands"][0]
    assert band["color"] == "brown"
    assert np.allclose(band["bbox"], [90,30,110,70])
