import importlib.util
from pathlib import Path


COLAB = Path("colab/train_resistor_detector_colab.py")


def _load():
    spec = importlib.util.spec_from_file_location("detector_colab", COLAB)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_detector_colab_uses_v4_merged_coco_release():
    colab = _load()
    assert colab.DATASET_URL == (
        "https://github.com/Persie0/resistor_model/releases/download/v4/"
        "resistor_sam3_merged.zip"
    )
    assert colab.DATASET_SHA256 == (
        "be3a1bb3b952f07556906decf6393fa7a8c228665867f721914a4e8e64376c6f"
    )
    assert colab.MODEL_NAME == "ssdlite320_mobilenet_v3_large"


def test_detector_colab_training_command_targets_detection_module(tmp_path: Path):
    colab = _load()
    command = colab.training_command(tmp_path / "dataset", tmp_path / "run", None)
    assert "resistor_model.train_detection" in command
    assert "--epochs" in command
    assert "--batch-size" in command
    assert "--checkpoint-every" in command
    assert "--progress-every" in command


def test_detector_colab_outputs_distinct_detector_artifacts():
    colab = _load()
    assert colab.ONNX_PATH.name == "resistor_detector_ssdlite320.onnx"
    assert "detector-v4" in str(colab.RUN_DIR)
