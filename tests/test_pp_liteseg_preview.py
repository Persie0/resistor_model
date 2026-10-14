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


def test_converter_uses_isolated_py312_instead_of_colab_python313():
    module = _load_colab()
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "onnx-converter-py312" in source
    assert '"venv", "--python", "3.12"' in source
    assert '"paddle2onnx==2.1.0"' in source
    assert '"paddlepaddle==3.1.1"' in source
    assert 'def _inference_model_filename' in source
    assert 'if str(PADDLESEG_DIR) not in sys.path' in source
    assert '"PP_LITESEG_EXPORT_ONLY"' in source


def test_converter_model_filename_prefers_paddle3_json(tmp_path: Path):
    module = _load_colab()
    (tmp_path / "model.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.pdmodel").write_bytes(b"legacy")
    assert module._inference_model_filename(tmp_path) == "model.json"
    (tmp_path / "model.json").unlink()
    assert module._inference_model_filename(tmp_path) == "model.pdmodel"


def test_converter_fails_readably_if_exported_graph_absent(tmp_path: Path):
    import pytest

    module = _load_colab()
    (tmp_path / "deploy.yaml").write_text("Deploy: {}", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="neither model.json nor model.pdmodel"):
        module._inference_model_filename(tmp_path)


def test_paddle_export_is_persisted_and_packaged_even_without_onnx():
    module = _load_colab()
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "shutil.copytree(inference_dir, DRIVE_INFERENCE, dirs_exist_ok=True)" in source
    assert 'DRIVE_INFERENCE.rglob("*")' in source


def test_existing_converter_missing_packaging_is_repaired_without_reinstalling_paddle(tmp_path, monkeypatch):
    import subprocess
    from types import SimpleNamespace

    module = _load_colab()
    venv = tmp_path / "py312"
    binary = venv / "bin" / "paddle2onnx"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!fake", encoding="utf-8")
    monkeypatch.setattr(module, "CONVERTER_VENV", venv)
    installed = []
    commands = []
    monkeypatch.setattr(module, "_pip_install", lambda args, description: installed.append(args))
    monkeypatch.setattr(
        module, "run_with_heartbeat",
        lambda args, **kwargs: commands.append(args),
    )
    states = iter((
        SimpleNamespace(returncode=1, stdout="", stderr="No module named 'packaging'"),
        SimpleNamespace(returncode=0, stdout="paddle2onnx 2.1.0", stderr=""),
    ))
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: next(states))
    assert module._converter_executable() == binary
    assert installed == [["-q", "uv"]]
    assert len(commands) == 1
    assert "packaging>=23,<27" in commands[0]
    assert "paddlepaddle==3.1.1" not in commands[0]
    assert "paddle2onnx==2.1.0" not in commands[0]


def test_healthy_converter_reused_without_any_install(tmp_path, monkeypatch):
    from types import SimpleNamespace

    module = _load_colab()
    venv = tmp_path / "py312"
    binary = venv / "bin" / "paddle2onnx"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!fake", encoding="utf-8")
    monkeypatch.setattr(module, "CONVERTER_VENV", venv)
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=0, stdout="2.1.0", stderr=""))
    monkeypatch.setattr(module, "_pip_install", lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("a healthy converter must not reinstall dependencies")
    ))
    assert module._converter_executable() == binary


def test_paddleseg_checkpoint_preview_builds_model_via_segbuilder(tmp_path, monkeypatch):
    import sys
    import types
    from contextlib import nullcontext

    module = _load_colab()
    checkpoint = tmp_path / "output" / "best_model" / "model.pdparams"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"placeholder")
    config_path = tmp_path / "config.yml"
    config_path.write_text("model: {}", encoding="utf-8")
    source_dir = tmp_path / "PaddleSeg"
    source_dir.mkdir()
    monkeypatch.setattr(module, "OUTPUT_LOCAL", checkpoint.parent.parent)
    monkeypatch.setattr(module, "RUN_DIR", tmp_path)
    monkeypatch.setattr(module, "CUSTOM_CONFIG", config_path)
    monkeypatch.setattr(module, "PADDLESEG_DIR", source_dir)

    class FakeModel:
        def eval(self):
            return None

        def __call__(self, tensor):
            return [tensor]

    class FakeConfig:
        def __init__(self, path):
            self.path = path
            # Deliberately no .model: PaddleSeg 2.10 doesn't expose it.

    class FakeSegBuilder:
        def __init__(self, cfg):
            self.model = FakeModel()

    cvlibs = types.ModuleType("paddleseg.cvlibs")
    cvlibs.Config, cvlibs.SegBuilder = FakeConfig, FakeSegBuilder
    utils = types.ModuleType("paddleseg.utils")
    recorded = []
    utils.utils = types.SimpleNamespace(
        load_entire_model=lambda model, path: recorded.append(path)
    )
    paddle = types.ModuleType("paddle")
    paddle.no_grad = lambda: nullcontext()
    paddle.to_tensor = lambda arr: types.SimpleNamespace(numpy=lambda: arr)

    package = types.ModuleType("paddleseg")
    package.__path__ = [str(source_dir)]
    monkeypatch.setitem(sys.modules, "paddle", paddle)
    monkeypatch.setitem(sys.modules, "paddleseg", package)
    monkeypatch.setitem(sys.modules, "paddleseg.cvlibs", cvlibs)
    monkeypatch.setitem(sys.modules, "paddleseg.utils", utils)

    predictor = module._load_preview_predictor(None)
    assert callable(predictor)
    assert recorded == [str(checkpoint)]
    dummy = types.SimpleNamespace(numpy=lambda: np.ones((1, 2, 8, 8)))
    assert predictor(dummy).shape == (1, 2, 8, 8)
