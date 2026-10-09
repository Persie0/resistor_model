from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "colab" / "train_resistor_segmentation_colab.py"


def _load():
    spec = spec_from_file_location("train_resistor_segmentation_colab", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_colab_recipe_uses_v4_sam_segmentation_release_asset():
    colab = _load()
    assert colab.DATASET_URL.endswith("/releases/download/v4/resistor_sam3_merged.zip")
    assert colab.DATASET_SHA256 == (
        "be3a1bb3b952f07556906decf6393fa7a8c228665867f721914a4e8e64376c6f"
    )
    assert colab.MODEL_NAME == "lraspp_mobilenet_v3_large"
    assert "v4" in colab.SEGMENTATION_COLAB_VERSION


def test_colab_training_command_uses_mobile_and_licensing_clean_defaults(tmp_path: Path):
    colab = _load()
    command = colab.training_command(tmp_path / "dataset", tmp_path / "run", None)
    joined = " ".join(map(str, command))
    assert "--image-size 384" in joined
    assert "--epochs 60" in joined
    assert "--batch-size 16" in joined
    assert "--progress-every 1" in joined
    assert "--checkpoint-every 5" in joined
    assert "--pretrained-backbone" not in command
    assert command.count("--category") == 1
    assert command[command.index("--category") + 1] == "resistor"
    assert "--resume" not in command


def test_colab_training_command_resumes_last_checkpoint(tmp_path: Path):
    colab = _load()
    resume = tmp_path / "last.pt"
    command = colab.training_command(tmp_path / "dataset", tmp_path / "run", resume)
    assert "--resume" in command
    assert str(resume) in command


def test_colab_stages_drive_resume_checkpoint_locally(tmp_path: Path):
    colab = _load()
    source = tmp_path / "drive" / "last.pt"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"checkpoint-data")
    local = tmp_path / "local-resume.pt"

    staged = colab.stage_resume_checkpoint(source, local)

    assert staged == local
    assert local.read_bytes() == b"checkpoint-data"


def test_colab_recipe_packages_persistent_outputs_for_download():
    colab = _load()
    assert colab.ZIP_PATH.name == "resistor-segmentation-v4-lraspp.zip"
    assert colab.RUN_DIR.name == "segmentation-v4-lraspp"
    assert callable(colab.package_and_download)
