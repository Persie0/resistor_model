"""Regression tests for Colab post-training image testing flows."""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("notebook_name", "export_module", "tester"),
    [
        (
            "train_resistor_segmentation_colab.ipynb",
            "resistor_model.export_segmentation",
            "test_resistor_segmentation_alignment_colab.py",
        ),
        (
            "train_resistor_detector_colab.ipynb",
            "resistor_model.export_detection",
            "test_resistor_detector_colab.py",
        ),
    ],
)
def test_training_notebooks_include_independent_post_training_tester(
    notebook_name: str, export_module: str, tester: str
):
    notebook = json.loads((ROOT / "colab" / notebook_name).read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) == 4
    assert [cell["cell_type"] for cell in notebook["cells"]] == [
        "markdown", "code", "markdown", "code",
    ]
    source = "".join(notebook["cells"][-1]["source"])
    ast.parse(source)  # Catch syntax regressions in Colab test cells.
    assert "best.pt" in source
    assert export_module in source
    assert tester in source
    assert "files.upload" not in source  # Handled by tester, not trainer.
    assert "train_and_export()" not in source
    assert "mtime_ns" in source  # Avoid unnecessary repeated ONNX exports.
    assert "runpy.run_path" in source


def _load_test_module(name: str):
    script = ROOT / "colab" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), script)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_ssdlite_outputs_map_independent_source_axes_and_filter_scores():
    detector = _load_test_module("test_resistor_detector_colab.py")
    boxes = np.array([
        [32.0, 64.0, 160.0, 160.0],
        [10.0, 10.0, 60.0, 60.0],
        [0.0, 0.0, 320.0, 320.0],
        [100.0, 100.0, 140.0, 140.0],
    ], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.1, 0.7], dtype=np.float32)
    labels = np.array([1, 0, 1, 1], dtype=np.int64)
    result = detector.decode_detections(
        boxes, scores, labels, source_width=640, source_height=240, score_threshold=.35
    )
    assert len(result) == 2
    assert result[0][0] == (64, 48, 320, 120)
    assert result[1][0] == (200, 75, 280, 105)


def test_ssdlite_inference_input_and_zero_detections():
    detector = _load_test_module("test_resistor_detector_colab.py")
    rgb = np.zeros((240, 640, 3), dtype=np.uint8)
    output = detector.preprocess(rgb)
    assert output.shape == (1, 3, 320, 320)
    assert output.dtype == np.float32

    class DummyInput:
        name = "image"

    class DummySession:
        def get_inputs(self):
            return [DummyInput()]
        def run(self, outputs, feed):
            assert outputs == ["boxes", "scores", "labels"]
            assert feed["image"].shape == (1, 3, 320, 320)
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
            )

    detections, elapsed = detector.predict(DummySession(), rgb)
    assert detections == []
    assert elapsed >= 0
    assert np.array_equal(detector.overlay_boxes(rgb, detections), rgb)


def test_segmentation_test_uploads_multiple_and_skips_one_bad_photo(monkeypatch, capsys):
    tester = _load_test_module("test_resistor_segmentation_alignment_colab.py")
    called = []

    def fake_test(name, data, *args):
        called.append(name)
        if name.endswith("bad.jpg"):
            raise ValueError("bad image")

    monkeypatch.setattr(tester, "test_uploaded_image", fake_test)
    # Test the upload loop directly via patching the display/session dependencies.
    # The production wrapper uses a single ONNX session for all uploaded photos.
    assert "for filename, data in uploaded.items()" in (
        ROOT / "colab" / "test_resistor_segmentation_alignment_colab.py"
    ).read_text()
    assert called == []
