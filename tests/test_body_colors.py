import inspect
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from resistor_model.tools.eval_color_buckets import bucket
from resistor_model.tools.synthesize_body_colors import recolor_body, synthesize_split


def _tiny_pair(root: Path, stem: str = "r1") -> tuple[Path, Path]:
    images, masks = root / "train" / "images", root / "train" / "masks_semantic"
    images.mkdir(parents=True)
    masks.mkdir(parents=True)
    img = np.full((20, 40, 3), 200, np.uint8)  # light background
    img[6:14, 8:32] = (40, 90, 200)  # brown-ish body in BGR
    mask = np.zeros((20, 40), np.uint8)
    mask[6:14, 8:32] = 255
    img_path, mask_path = images / f"{stem}.jpg", masks / f"{stem}.png"
    cv2.imwrite(str(img_path), img)
    cv2.imwrite(str(mask_path), mask)
    return img_path, mask_path


def _fg_median_hue_ocv(img_bgr: np.ndarray, mask01: np.ndarray) -> float:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    return float(np.median(hsv[..., 0].astype(np.float32)[mask01 > 0]))


def test_synth_recolors_foreground_toward_blue_and_keeps_mask(tmp_path: Path):
    img_path, mask_path = _tiny_pair(tmp_path)
    img = cv2.imread(str(img_path))
    mask01 = (cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
    rng = np.random.default_rng(0)
    recol = recolor_body(img, mask01, rng, target_hue_ocv=108)
    assert recol.shape == img.shape
    # Background untouched.
    assert np.array_equal(recol[mask01 == 0], img[mask01 == 0])
    # Foreground hue moved to blue.
    assert abs(_fg_median_hue_ocv(recol, mask01) - 108) < 12


def test_synth_gray_variant_desaturates_foreground(tmp_path: Path):
    img_path, mask_path = _tiny_pair(tmp_path)
    img = cv2.imread(str(img_path))
    mask01 = (cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
    recol = recolor_body(img, mask01, np.random.default_rng(1), target_hue_ocv=None, gray=True)
    hsv = cv2.cvtColor(recol, cv2.COLOR_BGR2HSV)
    assert float(hsv[..., 1].astype(np.float32)[mask01 > 0].mean()) < 20


def test_synthesize_split_writes_pairs_and_log(tmp_path: Path):
    _tiny_pair(tmp_path)
    rng = np.random.default_rng(7)
    records = synthesize_split(tmp_path / "train", variants=["blue", "green"], rng=rng)
    assert len(records) == 2
    for row in records:
        out_mask = tmp_path / "train" / "masks_semantic" / row["mask"]
        src_mask = tmp_path / "train" / "masks_semantic" / "r1.png"
        assert out_mask.is_file()
        # Silhouette byte-identical: same body, new color.
        assert out_mask.read_bytes() == src_mask.read_bytes()
    # Second run without overwrite skips but still logs.
    again = synthesize_split(tmp_path / "train", variants=["blue", "green"], rng=rng)
    assert all(row.get("skipped") for row in again)


def test_bucket_maps_hues_and_low_saturation_to_gray():
    assert bucket(15.0, 120.0) == "brown"
    assert bucket(130.0, 120.0) == "green"
    assert bucket(215.0, 120.0) == "blue"
    assert bucket(0.0, 120.0) == "red"
    assert bucket(130.0, 10.0) == "gray"
    assert bucket(215.0, 5.0) == "gray"


def test_segmentation_dataset_accepts_stronger_color_aug():
    from resistor_model.segmentation import CocoResistorSegmentationDataset

    sig = inspect.signature(CocoResistorSegmentationDataset.__init__)
    assert sig.parameters["hue_jitter"].default == 0.12
    assert sig.parameters["saturation_jitter"].default == 0.4
    assert sig.parameters["gray_prob"].default == 0.10


def test_train_segmentation_parser_exposes_color_flags():
    from resistor_model.train_segmentation import build_parser

    args = build_parser().parse_args(["--dataset-root", "dummy"])
    assert args.hue_jitter == 0.12
    assert args.saturation_jitter == 0.4
    assert args.gray_prob == 0.10


def test_pp_liteseg_colab_targets_liteseg_512_and_synth_colors():
    from importlib.util import module_from_spec, spec_from_file_location

    script = Path(__file__).resolve().parents[1] / "colab" / "train_pp_liteseg_colab.py"
    spec = spec_from_file_location("train_pp_liteseg_colab", script)
    assert spec is not None and spec.loader is not None
    colab = module_from_spec(spec)
    spec.loader.exec_module(colab)
    assert colab.MODEL_TYPE_NEEDLE == "liteseg"
    assert colab.IMAGE_SIZE == 512
    assert "blue" in colab.SYNTH_VARIANTS and "green" in colab.SYNTH_VARIANTS
    assert colab.RUN_DIR.name == "segmentation-pp-liteseg"
    assert "Ultralytics" not in Path(script).read_text(encoding="utf-8")
