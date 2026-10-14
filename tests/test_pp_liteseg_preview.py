"""Unit checks for the Colab 8k training target and interactive inference previews.

No GPU, Google Drive, network, Paddle, or ONNX runtime needed.
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
from PIL import Image


def _load_colab():
    path = Path(__file__).resolve().parents[1] / "colab" / "train_pp_liteseg_colab.py"
    spec = spec_from_file_location("train_pp_liteseg_preview_test", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_target_is_8k_and_upload_preview_follows_model_zip():
    module = _load_colab()
    assert module.ITERS == 8000
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert source.index("    package_and_download()\n") < source.index(
        "        preview_uploaded_images(onnx_path)\n"
    )
    assert 'PP_LITESEG_PREVIEW_ONLY' in source


def test_letterbox_uses_paddleseg_normalize_not_imagenet():
    module = _load_colab()
    image = Image.new("RGB", (400, 200), (255, 0, 0))
    tensor = module._letterbox_preview_input(image)
    assert tensor.shape == (1, 3, 512, 512)
    assert tensor.dtype == np.float32
    np.testing.assert_allclose(tensor[0, :, 256, 256], [1.0, -1.0, -1.0], atol=1e-6)
    # Original aspect ratio preserved; unused band is padded with neutral 128.
    np.testing.assert_allclose(tensor[0, :, 0, 0], [128.0 / 127.5 - 1] * 3, atol=1e-6)


def test_mask_unletterbox_and_exif_rotated_dimensions():
    module = _load_colab()
    foreground = np.zeros((512, 512), dtype=bool)
    foreground[128:384, :] = True
    restored = module._unletterbox_preview_mask(
        foreground, original_size=(400, 200), display_size=(200, 100)
    )
    assert restored.shape == (100, 200)
    assert restored.dtype == bool
    assert restored.all()


def test_foreground_output_accepts_two_class_logits_and_binary_masks():
    module = _load_colab()
    logits = np.array([[
        [[0.8, 0.1], [0.1, 0.8]],
        [[0.2, 0.9], [0.9, 0.2]],
    ]], dtype=np.float32)
    np.testing.assert_array_equal(
        module._foreground_mask_from_output(logits),
        np.array([[False, True], [True, False]])
    )
    np.testing.assert_array_equal(
        module._foreground_mask_from_output(np.array([[[0, 1], [1, 0]]], dtype=np.uint8)),
        np.array([[False, True], [True, False]])
    )


def test_color_eval_uses_same_paddleseg_normalization():
    from resistor_model.tools.eval_color_buckets import PADDLESEG_MEAN, PADDLESEG_STD

    np.testing.assert_array_equal(PADDLESEG_MEAN, [0.5, 0.5, 0.5])
    np.testing.assert_array_equal(PADDLESEG_STD, [0.5, 0.5, 0.5])


def test_resume_keeps_better_existing_checkpoint(tmp_path: Path, monkeypatch):
    module = _load_colab()
    monkeypatch.setattr(module, "OUTPUT_LOCAL", tmp_path / "output")
    best = module.OUTPUT_LOCAL / "best_model"
    backup = tmp_path / "old-best"
    best.mkdir(parents=True)
    backup.mkdir()
    (best / "model.pdparams").write_bytes(b"worse-model")
    (backup / "model.pdparams").write_bytes(b"better-model")
    monkeypatch.setattr(module, "_checkpoint_miou", lambda path: 0.8930)
    module._keep_stronger_best_model(backup, previous_miou=0.8985)
    assert (best / "model.pdparams").read_bytes() == b"better-model"


def test_resume_keeps_improved_checkpoint(tmp_path: Path, monkeypatch):
    module = _load_colab()
    monkeypatch.setattr(module, "OUTPUT_LOCAL", tmp_path / "output")
    best = module.OUTPUT_LOCAL / "best_model"
    backup = tmp_path / "old-best"
    best.mkdir(parents=True)
    backup.mkdir()
    (best / "model.pdparams").write_bytes(b"better-new-model")
    (backup / "model.pdparams").write_bytes(b"old-model")
    monkeypatch.setattr(module, "_checkpoint_miou", lambda path: 0.9100)
    module._keep_stronger_best_model(backup, previous_miou=0.8985)
    assert (best / "model.pdparams").read_bytes() == b"better-new-model"
