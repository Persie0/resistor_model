import inspect
import json
from pathlib import Path

from PIL import Image
import torch

from resistor_model.segmentation import (
    CocoResistorSegmentationDataset,
    ForegroundProbabilityWrapper,
    build_lraspp_model,
    segmentation_loss,
)


def _write_tiny_coco(root: Path) -> None:
    split = root / "train"
    split.mkdir(parents=True)
    Image.new("RGB", (20, 10), "white").save(split / "sample.jpg")
    payload = {
        "images": [{"id": 1, "file_name": "sample.jpg", "width": 20, "height": 10}],
        "categories": [{"id": 3, "name": "resistor"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 3,
                "segmentation": [[2, 2, 17, 2, 17, 7, 2, 7]],
                "bbox": [2, 2, 15, 5],
                "area": 75,
                "iscrowd": 0,
            }
        ],
    }
    (split / "_annotations.coco.json").write_text(json.dumps(payload), encoding="utf-8")


def test_coco_dataset_builds_binary_resistor_mask(tmp_path: Path):
    _write_tiny_coco(tmp_path)
    dataset = CocoResistorSegmentationDataset(tmp_path, "train", image_size=32, augment=False)
    image, mask = dataset[0]
    assert image.shape == (3, 32, 32)
    assert mask.shape == (32, 32)
    assert mask.dtype == torch.long
    assert int(mask.max()) == 1
    assert int(mask.sum()) > 0


def test_lraspp_library_default_does_not_load_pretrained_weights():
    default = inspect.signature(build_lraspp_model).parameters["pretrained_backbone"].default
    assert default is False


def test_lraspp_returns_two_class_logits_at_input_resolution():
    model = build_lraspp_model(num_classes=2, pretrained_backbone=False)
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 64, 64))["out"]
    assert out.shape == (1, 2, 64, 64)


def test_segmentation_loss_is_lower_for_correct_foreground_prediction():
    target = torch.zeros(1, 8, 8, dtype=torch.long)
    target[:, 2:6, 2:6] = 1

    correct = torch.full((1, 2, 8, 8), -4.0)
    correct[:, 0] = 4.0
    correct[:, 0, 2:6, 2:6] = -4.0
    correct[:, 1, 2:6, 2:6] = 4.0

    wrong = -correct
    assert segmentation_loss(correct, target).item() < segmentation_loss(wrong, target).item()


def test_probability_wrapper_returns_single_foreground_channel():
    model = build_lraspp_model(num_classes=2, pretrained_backbone=False)
    wrapped = ForegroundProbabilityWrapper(model).eval()
    with torch.no_grad():
        prob = wrapped(torch.zeros(1, 3, 64, 64))
    assert prob.shape == (1, 1, 64, 64)
    assert torch.all(prob >= 0)
    assert torch.all(prob <= 1)


def test_dataset_prefers_resistor_named_categories_over_other_annotations(tmp_path: Path):
    split = tmp_path / "train"
    split.mkdir(parents=True)
    Image.new("RGB", (20, 20), "white").save(split / "sample.jpg")
    payload = {
        "images": [{"id": 1, "file_name": "sample.jpg", "width": 20, "height": 20}],
        "categories": [{"id": 1, "name": "resistor"}, {"id": 2, "name": "label"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "segmentation": [[2, 2, 8, 2, 8, 8, 2, 8]],
                "bbox": [2, 2, 6, 6],
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 2,
                "segmentation": [[12, 12, 18, 12, 18, 18, 12, 18]],
                "bbox": [12, 12, 6, 6],
            },
        ],
    }
    (split / "_annotations.coco.json").write_text(json.dumps(payload), encoding="utf-8")
    dataset = CocoResistorSegmentationDataset(tmp_path, "train", image_size=20, augment=False)
    _, mask = dataset[0]
    assert int(mask[5, 5]) == 1
    assert int(mask[15, 15]) == 0



def test_v4_semantic_png_masks_override_coco_bbox_and_rle(tmp_path: Path):
    """The v4 training target must be the SAM binary silhouette, not its bbox."""
    import numpy as np

    split = tmp_path / "train"
    (split / "images").mkdir(parents=True)
    (split / "masks_semantic").mkdir()

    Image.new("RGB", (20, 10), "white").save(split / "images" / "000001_sha.jpg")
    binary = np.zeros((10, 20), dtype=np.uint8)
    binary[2:8, 8:12] = 255
    binary[4:6, 3:17] = 255
    Image.fromarray(binary).save(split / "masks_semantic" / "000001_sha.png")

    # Deliberately contains a bbox that covers background; the PNG is authoritative.
    payload = {
        "images": [{"id": 1, "file_name": "images/000001_sha.jpg", "width": 20, "height": 10}],
        "categories": [{"id": 1, "name": "resistor"}],
        "annotations": [{
            "image_id": 1, "category_id": 1,
            "bbox": [0, 0, 20, 10], "segmentation": {"size": [10, 20], "counts": "bad"}
        }],
    }
    (split / "_annotations.coco.json").write_text(json.dumps(payload), encoding="utf-8")

    dataset = CocoResistorSegmentationDataset(
        tmp_path, "train", image_size=40, augment=False, category_names=("resistor",)
    )
    assert len(dataset) == 1
    image, target = dataset[0]
    assert image.shape == (3, 40, 40)
    assert target.shape == (40, 40)
    # Source 20x10 letterboxes into 40x20 with top 10 rows black.
    assert int(target[14, 20]) == 1
    assert int(target[14, 4]) == 0
    assert int(target[2, 20]) == 0


def test_v4_png_pairing_fails_fast_when_mask_missing(tmp_path: Path):
    split = tmp_path / "train"
    (split / "images").mkdir(parents=True)
    (split / "masks_semantic").mkdir()
    Image.new("RGB", (20, 10), "white").save(split / "images" / "a.jpg")
    (split / "_annotations.coco.json").write_text(
        json.dumps({"images": [], "annotations": [], "categories": []}),
        encoding="utf-8",
    )
    import pytest
    with pytest.raises(ValueError, match="mask"):
        CocoResistorSegmentationDataset(tmp_path, "train", image_size=32)


def test_v4_png_masks_validate_dimensions(tmp_path: Path):
    split = tmp_path / "train"
    (split / "images").mkdir(parents=True)
    (split / "masks_semantic").mkdir()
    Image.new("RGB", (20, 10), "white").save(split / "images" / "a.png")
    Image.new("L", (8, 8), 255).save(split / "masks_semantic" / "a.png")
    (split / "_annotations.coco.json").write_text(
        json.dumps({"images": [], "annotations": [], "categories": []}),
        encoding="utf-8",
    )
    dataset = CocoResistorSegmentationDataset(tmp_path, "train", image_size=32)
    import pytest
    with pytest.raises(ValueError, match="shape|size|dimensions"):
        dataset[0]
