from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "colab" / "train_resistor_segmentation_colab.py"


def _load():
    spec = spec_from_file_location("train_resistor_segmentation_colab", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_colab_recipe_uses_m2_segmentation_release_asset():
    colab = _load()
    assert colab.DATASET_URL.endswith("/releases/download/m2/detection_res.v1i.coco-segmentation.zip")
    assert colab.MODEL_NAME == "lraspp_mobilenet_v3_large"


def test_colab_training_command_uses_mobile_defaults(tmp_path: Path):
    colab = _load()
    command = colab.training_command(tmp_path / "dataset", tmp_path / "run", None)
    joined = " ".join(map(str, command))
    assert "--image-size 384" in joined
    assert "--epochs 60" in joined
    assert "--batch-size 16" in joined
    assert "--no-pretrained-backbone" in command
    assert "--resume" not in command


def test_colab_training_command_resumes_last_checkpoint(tmp_path: Path):
    colab = _load()
    resume = tmp_path / "last.pt"
    command = colab.training_command(tmp_path / "dataset", tmp_path / "run", resume)
    assert "--resume" in command
    assert str(resume) in command
