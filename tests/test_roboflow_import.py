import json
from pathlib import Path

import cv2
import numpy as np

from resistor_model.tools.import_roboflow_yolo import (
    canonical_class_name,
    group_bands_by_body,
    import_dataset,
    parse_yolo_label_line,
    roboflow_source_id,
)


def test_parse_box_polygon_and_aliases():
    classes = ["grey", "resistor symbol"]
    band = parse_yolo_label_line("0 0.5 0.5 0.2 0.4", classes, 100, 200)
    assert band["name"] == "gray"
    assert band["bbox"] == [40.0, 60.0, 60.0, 140.0]

    body = parse_yolo_label_line("1 0.1 0.2 0.9 0.2 0.9 0.8 0.1 0.8", classes, 100, 200)
    assert body["name"] == "resistor symbol"
    assert body["bbox"] == [10.0, 40.0, 90.0, 160.0]


def test_group_bands_assigns_two_resistors_without_cross_talk():
    bodies = [[0, 0, 100, 40], [200, 100, 300, 140]]
    bands = []
    for x in (20, 40, 60, 80):
        bands.append({"color": "red", "bbox": [x - 3, 5, x + 3, 35]})
    for x in (220, 240, 260, 280):
        bands.append({"color": "blue", "bbox": [x - 3, 105, x + 3, 135]})

    groups, unassigned = group_bands_by_body(bodies, bands, min_bands=3, max_bands=6)
    assert unassigned == 0
    assert len(groups) == 2
    assert {b["color"] for b in groups[0][1]} == {"red"}
    assert {b["color"] for b in groups[1][1]} == {"blue"}


def test_group_bands_fallback_accepts_single_unboxed_resistor():
    bands = [{"color": "black", "bbox": [i * 10, 0, i * 10 + 4, 30]} for i in range(4)]
    groups, unassigned = group_bands_by_body([], bands, min_bands=3, max_bands=6)
    assert groups == [(None, bands)]
    assert unassigned == 0


def test_roboflow_source_id_strips_augmentation_hash():
    assert roboflow_source_id("100R_jpg.rf.0123456789abcdef.jpg") == "100R_jpg"
    assert roboflow_source_id("plain.jpg") == "plain"
    assert canonical_class_name("  GREY ") == "gray"


def _write_sample(root: Path, split: str, filename: str, body_polygon: bool = False) -> None:
    image_dir = root / split / "images"
    label_dir = root / split / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(image_dir / filename), np.zeros((100, 200, 3), dtype=np.uint8))
    stem = Path(filename).stem
    body = "2 0.1 0.2 0.9 0.2 0.9 0.8 0.1 0.8" if body_polygon else "2 0.5 0.5 0.8 0.6"
    labels = [
        body,
        "0 0.25 0.5 0.05 0.4",
        "1 0.40 0.5 0.05 0.4",
        "0 0.55 0.5 0.05 0.4",
        "1 0.70 0.5 0.05 0.4",
    ]
    (label_dir / f"{stem}.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")


def test_import_dataset_groups_roboflow_variants_to_same_split_id(tmp_path: Path):
    (tmp_path / "data.yaml").write_text(
        "nc: 3\nnames: ['black', 'grey', 'resistor symbol']\n", encoding="utf-8"
    )
    _write_sample(tmp_path, "train", "source_jpg.rf.aaaaaaaa.jpg")
    _write_sample(tmp_path, "valid", "source_jpg.rf.bbbbbbbb.jpg", body_polygon=True)

    output = tmp_path / "manifest.jsonl"
    report = import_dataset(tmp_path, output, min_bands=3, max_bands=6)
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    assert report["counts"]["images_accepted"] == 2
    assert report["counts"]["source_groups"] == 1
    assert report["counts"]["source_groups_with_multiple_exports"] == 1
    assert [row["resistors"][0]["id"] for row in rows] == ["source_jpg", "source_jpg"]
    assert all(row["split"] is None for row in rows)
    assert rows[0]["resistors"][0]["bands"][1]["color"] == "gray"
