import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from resistor_model.detection import (
    CocoResistorDetectionDataset,
    build_ssdlite_model,
    collate_detection_batch,
)


def _fixture(root: Path) -> None:
    split = root / "train"
    images = split / "images"
    images.mkdir(parents=True)
    Image.fromarray(np.full((12, 20, 3), 128, dtype=np.uint8)).save(images / "sample.jpg")
    coco = {
        "images": [{"id": 7, "file_name": "images/sample.jpg", "width": 20, "height": 12}],
        "categories": [{"id": 1, "name": "resistor"}],
        "annotations": [
            {
                "id": 11,
                "image_id": 7,
                "category_id": 1,
                "bbox": [2, 3, 10, 4],
                "area": 40,
                "iscrowd": 0,
            }
        ],
    }
    (split / "_annotations.coco.json").write_text(json.dumps(coco), encoding="utf-8")


def test_coco_detection_dataset_converts_xywh_to_xyxy(tmp_path: Path):
    _fixture(tmp_path)
    dataset = CocoResistorDetectionDataset(tmp_path, "train", augment=False)
    image, target = dataset[0]
    assert tuple(image.shape) == (3, 12, 20)
    assert image.dtype == torch.float32
    assert target["boxes"].tolist() == [[2.0, 3.0, 12.0, 7.0]]
    assert target["labels"].tolist() == [1]
    assert target["image_id"].tolist() == [7]
    assert target["area"].tolist() == [40.0]
    assert target["iscrowd"].tolist() == [0]


def test_detection_dataset_uses_single_foreground_class_even_if_coco_id_is_not_one(tmp_path: Path):
    _fixture(tmp_path)
    annotation_path = tmp_path / "train" / "_annotations.coco.json"
    coco = json.loads(annotation_path.read_text(encoding="utf-8"))
    coco["categories"][0]["id"] = 23
    coco["annotations"][0]["category_id"] = 23
    annotation_path.write_text(json.dumps(coco), encoding="utf-8")

    _, target = CocoResistorDetectionDataset(tmp_path, "train")[0]
    assert target["labels"].tolist() == [1]


def test_collate_detection_batch_preserves_variable_targets():
    batch = [
        (torch.zeros(3, 10, 20), {"boxes": torch.zeros(1, 4)}),
        (torch.zeros(3, 12, 18), {"boxes": torch.zeros(2, 4)}),
    ]
    images, targets = collate_detection_batch(batch)
    assert len(images) == 2
    assert len(targets) == 2
    assert targets[1]["boxes"].shape == (2, 4)


def test_ssdlite_builder_is_one_class_and_does_not_require_pretrained_weights():
    model = build_ssdlite_model(num_classes=2, pretrained_backbone=False)
    assert model.transform.min_size == (320,)
    assert model.transform.max_size == 320
