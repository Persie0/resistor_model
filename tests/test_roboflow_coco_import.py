import json
from pathlib import Path

import cv2
import numpy as np

from resistor_model.tools.import_roboflow_coco import (
    import_datasets,
    normalize_category_name,
    source_group_id,
)


def _write_image(path: Path, *, width: int = 200, height: int = 100) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), np.zeros((height, width, 3), dtype=np.uint8))


def _write_coco(root: Path, split: str, *, categories: list[str], images: list[dict], annotations: list[dict]) -> None:
    split_dir = root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    for image in images:
        _write_image(split_dir / image["file_name"], width=int(image["width"]), height=int(image["height"]))
    payload = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": i, "name": name} for i, name in enumerate(categories)],
    }
    (split_dir / "_annotations.coco.json").write_text(json.dumps(payload), encoding="utf-8")


def _band(ann_id: int, image_id: int, category_id: int, x: float) -> dict:
    return {"id": ann_id, "image_id": image_id, "category_id": category_id, "bbox": [x, 30, 8, 40], "area": 320, "segmentation": []}


def test_normalizes_color_and_body_aliases():
    assert normalize_category_name(" GREY ") == "gray"
    assert normalize_category_name("resistor symbol") == "resistor"
    assert normalize_category_name("object-resistors") is None
    assert normalize_category_name("resistors") is None


def test_source_group_matches_original_name_across_roboflow_exports():
    assert source_group_id("same.jpg") == "same"
    assert source_group_id("same_jpg.rf.0123456789abcdef.jpg") == "same"
    assert source_group_id("same_png.rf.aaaaaaaaaaaaaaaa.png") == "same"


def test_imports_body_annotated_coco_and_preserves_polygon(tmp_path: Path):
    root = tmp_path / "body"
    cats = ["object-resistors", "black", "grey", "red", "gold", "resistor"]
    image = {"id": 7, "file_name": "sample_jpg.rf.aaaa.jpg", "width": 200, "height": 100, "extra": {"name": "sample.jpg"}}
    body = {
        "id": 1,
        "image_id": 7,
        "category_id": 5,
        "bbox": [10, 10, 180, 80],
        "area": 14400,
        "segmentation": [[10, 10, 190, 10, 190, 90, 10, 90]],
    }
    anns = [body, _band(2, 7, 1, 40), _band(3, 7, 2, 75), _band(4, 7, 3, 110), _band(5, 7, 4, 145)]
    _write_coco(root, "train", categories=cats, images=[image], annotations=anns)

    output = tmp_path / "manifest.jsonl"
    report = import_datasets([root], output, min_bands=3, max_bands=6)
    rows = [json.loads(line) for line in output.read_text().splitlines()]

    assert report["counts"]["resistors_accepted"] == 1
    assert len(rows) == 1
    resistor = rows[0]["resistors"][0]
    assert [b["color"] for b in resistor["bands"]] == ["black", "gray", "red", "gold"]
    assert resistor["body_annotation_type"] == "polygon"
    assert resistor["polygon"] == [[10.0, 10.0], [190.0, 10.0], [190.0, 90.0], [10.0, 90.0]]
    assert rows[0]["session_id"] == "sample"


def test_imports_single_resistor_without_body_annotation(tmp_path: Path):
    root = tmp_path / "nobody"
    cats = ["resistors", "black", "brown", "red", "gold"]
    image = {"id": 0, "file_name": "10k_jpg.rf.bbbb.jpg", "width": 200, "height": 100}
    anns = [_band(1, 0, 2, 30), _band(2, 0, 1, 65), _band(3, 0, 3, 100), _band(4, 0, 4, 135)]
    _write_coco(root, "train", categories=cats, images=[image], annotations=anns)

    output = tmp_path / "manifest.jsonl"
    report = import_datasets([root], output, min_bands=3, max_bands=6)
    row = json.loads(output.read_text().strip())

    assert report["counts"]["images_without_body"] == 1
    assert report["counts"]["resistors_accepted"] == 1
    assert row["resistors"][0]["bbox"] is None
    assert row["resistors"][0]["body_annotation_type"] is None
    assert len(row["resistors"][0]["bands"]) == 4


def test_rejects_ambiguous_no_body_image_with_more_than_max_bands(tmp_path: Path):
    root = tmp_path / "ambiguous"
    cats = ["resistors", "red"]
    image = {"id": 0, "file_name": "many.jpg", "width": 200, "height": 100}
    anns = [_band(i + 1, 0, 1, 10 + i * 20) for i in range(8)]
    _write_coco(root, "train", categories=cats, images=[image], annotations=anns)

    output = tmp_path / "manifest.jsonl"
    report = import_datasets([root], output, min_bands=3, max_bands=6)

    assert output.read_text() == ""
    assert report["counts"]["images_without_usable_resistor"] == 1


def test_same_original_source_across_datasets_gets_same_session_but_unique_ids(tmp_path: Path):
    roots = []
    for index, filename in enumerate(("same_jpg.rf.aaaaaaaa.jpg", "same_jpg.rf.bbbbbbbb.jpg")):
        root = tmp_path / f"dataset{index}"
        roots.append(root)
        cats = ["resistors", "black", "brown", "red", "gold"]
        image = {"id": 0, "file_name": filename, "width": 200, "height": 100}
        anns = [_band(1, 0, 1, 30), _band(2, 0, 2, 65), _band(3, 0, 3, 100), _band(4, 0, 4, 135)]
        _write_coco(root, "train", categories=cats, images=[image], annotations=anns)

    output = tmp_path / "manifest.jsonl"
    report = import_datasets(roots, output, min_bands=3, max_bands=6)
    rows = [json.loads(line) for line in output.read_text().splitlines()]

    assert [row["session_id"] for row in rows] == ["same", "same"]
    ids = [row["resistors"][0]["id"] for row in rows]
    assert len(set(ids)) == 2
    assert report["counts"]["cross_dataset_source_groups"] == 1
