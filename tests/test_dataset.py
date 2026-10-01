import json
from pathlib import Path

import cv2
import numpy as np
import torch

from resistor_model.constants import BACKGROUND_INDEX, COLOR_TO_INDEX
from resistor_model.data.augment import PhotometricAugment, make_chromatic_channels
from resistor_model.data.dataset import ResistorBandDataset


def _make_fixture(tmp_path: Path):
    img = np.full((120, 220, 3), 190, np.uint8)
    bands = [("brown", 50), ("black", 85), ("red", 120), ("gold", 165)]
    anns = []
    bgr = {"brown": (25, 55, 95), "black": (15, 15, 15), "red": (20, 20, 220), "gold": (40, 170, 210)}
    for color, x in bands:
        cv2.rectangle(img, (x, 35), (x + 12, 85), bgr[color], -1)
        anns.append({"color": color, "bbox": [x, 35, x + 12, 85]})
    image_path = tmp_path / "r1.jpg"; cv2.imwrite(str(image_path), img)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"image": "r1.jpg", "session_id": "s1", "resistors": [{"id": "r1", "bbox": [30, 25, 195, 95], "bands": anns}]}) + "\n", encoding="utf-8")
    return manifest


def test_dataset_builds_dense_and_ordered_slot_targets(tmp_path: Path):
    manifest = _make_fixture(tmp_path)
    ds = ResistorBandDataset(manifest, tmp_path, output_size=(64, 384), sequence_bins=96, max_bands=6, augment=False)
    sample = ds[0]
    assert sample["image"].shape == (3, 64, 384)
    assert sample["dense_target"].shape == (96,)
    assert sample["slot_exists"].tolist() == [1, 1, 1, 1, 0, 0]
    assert sample["slot_colors"][:4].tolist() == [COLOR_TO_INDEX["brown"], COLOR_TO_INDEX["black"], COLOR_TO_INDEX["red"], COLOR_TO_INDEX["gold"]]
    assert sample["slot_colors"][4:].tolist() == [-100, -100]
    assert torch.all(sample["slot_centers"][:4][1:] > sample["slot_centers"][:4][:-1])
    assert sample["count"].item() == 4
    assert (sample["dense_target"] != BACKGROUND_INDEX).sum().item() > 0


def test_target_generation_drops_fully_clipped_bands_without_slot_holes(tmp_path: Path):
    manifest = _make_fixture(tmp_path)
    ds = ResistorBandDataset(manifest, tmp_path, output_size=(64, 384), sequence_bins=96, max_bands=6, augment=False)
    targets = ds._targets(
        [
            {"color": "brown", "bbox": [-30, 5, -10, 50]},
            {"color": "red", "bbox": [80, 5, 100, 50]},
            {"color": "gold", "bbox": [180, 5, 200, 50]},
        ],
        width=160,
    )
    assert targets["count"].item() == 1
    assert targets["slot_exists"].tolist() == [1, 0, 0, 0, 0, 0]
    assert targets["slot_colors"].tolist() == [COLOR_TO_INDEX["red"], -100, -100, -100, -100, -100]


def test_dataset_filter_by_resistor_id(tmp_path: Path):
    manifest = _make_fixture(tmp_path)
    assert len(ResistorBandDataset(manifest, tmp_path, allowed_resistor_ids={"other"}, augment=False)) == 0
    assert len(ResistorBandDataset(manifest, tmp_path, allowed_resistor_ids={"r1"}, augment=False)) == 1


def test_photometric_augment_identity_when_disabled():
    image = np.full((32, 64, 3), 0.4, dtype=np.float32)
    out = PhotometricAugment(probability=0.0, seed=1)(image)
    assert np.array_equal(out, image)


def test_photometric_augment_can_be_reseeded_reproducibly():
    image = np.linspace(0.05, 0.95, 32 * 64 * 3, dtype=np.float32).reshape(32, 64, 3)
    augment = PhotometricAugment(probability=1.0, seed=1)
    augment.reseed(12345)
    first = augment(image)
    augment.reseed(12345)
    repeated = augment(image)
    augment.reseed(54321)
    different = augment(image)
    assert np.array_equal(first, repeated)
    assert not np.array_equal(first, different)


def test_chromatic_channels_are_finite_for_black_pixels():
    rgb = torch.zeros(2, 3, 16, 32)
    chroma = make_chromatic_channels(rgb)
    assert chroma.shape == (2, 5, 16, 32)
    assert torch.isfinite(chroma).all()
