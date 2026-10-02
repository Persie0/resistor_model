from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "colab" / "train_rres_v4_colab.py"
spec = spec_from_file_location("train_rres_v4_colab_r1", SCRIPT)
assert spec is not None and spec.loader is not None
colab = module_from_spec(spec)
spec.loader.exec_module(colab)


def test_r1_recipe_uses_exactly_three_release_coco_assets():
    assert colab.DATASET_ASSETS == (
        "rres.v4i.coco.zip",
        "resistor.value.training.v8i.coco.zip",
        "Deteksi.Nilai.Resistor.v1i.coco.zip",
    )
    assert colab.RELEASE_BASE.endswith("/Persie0/resistor_model/releases/download/r1")


def test_r1_recipe_uses_separate_persistent_v2_directory():
    assert colab.RUN_DIR.name == "r1-v2-colab"
    assert colab.DATA_ROOT.name == "r1_datasets"


def test_r1_v2_config_enables_accepted_improvements(tmp_path: Path):
    cfg = colab.build_config(tmp_path / "run", None)
    assert cfg["model"]["architecture"] == "v2"
    assert cfg["model"]["backbone"] == "convnext_lite"
    assert cfg["data"]["geometric_augment"] is True
    assert cfg["data"]["group_session"] is True
    assert cfg["loss"]["ctc"] > 0
    assert cfg["loss"]["kl"] > 0
    assert cfg["loss"]["label_smoothing"] > 0
    assert cfg["loss"]["color_balance"] == "sqrt_inverse"
    assert cfg["eval"]["extra_decoders"] is True
    assert cfg["eval"]["series_bonus"] == 0.0


def test_r1_recipe_manifest_and_splits_live_under_combined_dataset_root(tmp_path: Path):
    cfg = colab.build_config(tmp_path / "run", None)
    assert Path(cfg["data"]["manifest"]).name == "manifest.jsonl"
    assert Path(cfg["data"]["splits_file"]).name == "splits.json"
    assert Path(cfg["data"]["image_root"]) == colab.DATA_ROOT
