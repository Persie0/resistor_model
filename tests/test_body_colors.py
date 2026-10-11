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
    assert colab.PRECISION == "fp16"  # compute-bound T4: AMP is the real speedup lever
    assert colab.BATCH_SIZE == 48
    assert "blue" in colab.SYNTH_VARIANTS and "green" in colab.SYNTH_VARIANTS
    assert colab.RUN_DIR.name == "segmentation-pp-liteseg"
    assert "Ultralytics" not in Path(script).read_text(encoding="utf-8")


def _make_merge_fixture(root: Path) -> tuple[Path, Path]:
    import zipfile

    v4 = root / "v4"
    for split in ("train", "valid"):
        (v4 / split / "images").mkdir(parents=True)
        (v4 / split / "masks_semantic").mkdir()
    Image.new("RGB", (24, 12), "white").save(v4 / "train" / "images" / "v.jpg")
    Image.new("L", (24, 12), 0).save(v4 / "train" / "masks_semantic" / "v.png")

    extra = root / "extra"
    body = np.zeros((12, 24, 3), np.uint8)
    body[:] = (200, 30, 30)  # blue-ish BGR body
    ok, jpg = cv2.imencode(".jpg", body)
    assert ok
    mask = np.zeros((12, 24), np.uint8)
    mask[2:10, 4:20] = 255
    ok, png = cv2.imencode(".png", mask)
    assert ok
    archive = root / "blue.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("train/images/b.jpg", jpg.tobytes())
        zf.writestr("train/masks_semantic/b.png", png.tobytes())
        zf.writestr("train/images/lonely.jpg", jpg.tobytes())  # no mask -> ignored
        zf.writestr("valid/images/b.jpg", jpg.tobytes())
        zf.writestr("valid/masks_semantic/b.png", png.tobytes())
    return v4, archive


def test_merge_extra_masks_prefixes_and_preserves_pairs(tmp_path: Path):
    import hashlib

    from resistor_model.tools.merge_extra_masks import merge_archive

    v4, archive = _make_merge_fixture(tmp_path)
    sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    report = merge_archive(v4, archive, splits=["train", "valid"], prefix="blue", expected_sha256=sha)
    assert report["train"]["copied"] == 1
    assert report["valid"]["copied"] == 1
    img = v4 / "train" / "images" / "blue_b.jpg"
    mask = v4 / "train" / "masks_semantic" / "blue_b.png"
    assert img.is_file() and mask.is_file()
    with Image.open(mask) as m:
        assert m.size == (24, 12)
        assert np.asarray(m).sum() > 0
    # Original v4 pair untouched, idempotent rerun.
    assert (v4 / "train" / "images" / "v.jpg").is_file()
    rerun = merge_archive(v4, archive, splits=["train", "valid"], prefix="blue", expected_sha256=sha)
    assert rerun["train"]["copied"] == 0
    assert rerun["train"]["skipped_existing"] == 1


def test_merge_extra_masks_rejects_bad_checksum(tmp_path: Path):
    import pytest

    from resistor_model.tools.merge_extra_masks import merge_archive

    v4, archive = _make_merge_fixture(tmp_path)
    with pytest.raises(RuntimeError, match="checksum"):
        merge_archive(v4, archive, splits=["train"], prefix="blue", expected_sha256="0" * 64)


def test_segmentation_colabs_reference_blue_release():
    for name in ("train_resistor_segmentation_colab.py", "train_pp_liteseg_colab.py"):
        text = (Path(__file__).resolve().parents[1] / "colab" / name).read_text(encoding="utf-8")
        assert "blue_resistors_sam3.zip" in text
        assert "merge" in text.lower()


def test_pp_liteseg_colab_streams_progress_like_other_colabs():
    text = (Path(__file__).resolve().parents[1] / "colab" / "train_pp_liteseg_colab.py").read_text(encoding="utf-8")
    assert "def run_with_heartbeat" in text
    assert "def print_run_configuration" in text
    assert "run_with_heartbeat(command" in text


def test_pp_liteseg_colab_syncs_checkpoints_to_drive_periodically():
    text = (Path(__file__).resolve().parents[1] / "colab" / "train_pp_liteseg_colab.py").read_text(encoding="utf-8")
    assert "CHECKPOINT_SYNC_SECONDS" in text
    assert "def _start_drive_sync" in text
    assert "def _sync_dir_to_drive" in text
    assert "def download_with_progress" in text
    assert "_start_drive_sync()" in text


def test_pp_liteseg_colab_normalizes_masks_to_class_indices():
    text = (Path(__file__).resolve().parents[1] / "colab" / "train_pp_liteseg_colab.py").read_text(encoding="utf-8")
    # PaddleSeg reads labels as class indices (255 = ignore); 0/255 SAM masks
    # would silently train the foreground as background.
    assert "def normalize_masks_01" in text
    assert "normalize_masks_01(dataset_root)" in text


def test_pp_liteseg_colab_sets_pythonpath_for_paddleseg_tools():
    # tools/train.py does `from paddleseg...` but the script's own dir lands
    # on sys.path, not the repo root: without PYTHONPATH -> ModuleNotFoundError.
    text = (Path(__file__).resolve().parents[1] / "colab" / "train_pp_liteseg_colab.py").read_text(encoding="utf-8")
    assert "extra_env" in text
    assert 'extra_env={"PYTHONPATH": str(PADDLESEG_DIR)}' in text


def test_merge_and_synth_tools_accept_progress_every_flag():
    import os
    import subprocess
    import sys

    repo = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(repo / "src"))
    for tool in ("merge_extra_masks", "synthesize_body_colors"):
        proc = subprocess.run(
            [sys.executable, "-m", f"resistor_model.tools.{tool}", "--help"],
            capture_output=True, text=True, cwd=repo, env=env,
        )
        assert proc.returncode == 0, proc.stderr
        assert "--progress-every" in proc.stdout


def test_merge_and_synth_tools_report_incremental_progress():
    import inspect

    from resistor_model.tools.merge_extra_masks import merge_archive
    from resistor_model.tools.synthesize_body_colors import synthesize_split

    assert inspect.signature(merge_archive).parameters["progress_every"].default == 100
    assert inspect.signature(synthesize_split).parameters["progress_every"].default == 25


def test_pp_liteseg_colab_installs_paddle_from_official_index():
    # Regression test: `pip install paddlepaddle-gpu` from default PyPI has no
    # Python 3.13 wheel (exit 1 with -q hiding the reason). The GPU build must
    # come from Paddle's official index, with the pip tail printed on failure.
    import re

    text = (Path(__file__).resolve().parents[1] / "colab" / "train_pp_liteseg_colab.py").read_text(encoding="utf-8")
    assert "https://www.paddlepaddle.org.cn/packages/stable/cu126/" in text
    assert "PADDLE_GPU_PIN" in text
    # No bare PyPI paddle install left (only indexed attempts + error tail printer).
    bare = [line for line in text.splitlines()
            if "paddlepaddle-gpu" in line and "-i" not in line and "packages/stable" not in line
            and line.strip().startswith(("+", "run(", '"', "'"))]
    assert bare == []
    assert "opencv" in text and "requirements" in text  # pinned cv2 (no 3.13 wheel) is filtered out
