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
    assert "v5-diagnostic-startup" in colab.SEGMENTATION_COLAB_VERSION


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



def _make_v4_fixture(root: Path) -> None:
    import json

    summary = {"splits": {}}
    for split in ("train", "valid", "test"):
        base = root / split
        (base / "images").mkdir(parents=True)
        (base / "masks_semantic").mkdir()
        (base / "_annotations.coco.json").write_text("{}", encoding="utf-8")
        (base / "images" / "x.jpg").write_bytes(b"img")
        (base / "masks_semantic" / "x.png").write_bytes(b"mask")
        summary["splits"][split] = {"images": 1}
    (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


def test_v4_extraction_cache_requires_complete_mask_pairs(tmp_path: Path):
    colab = _load()
    _make_v4_fixture(tmp_path)
    assert colab.extracted_dataset_ready(tmp_path)
    (tmp_path / "train" / "masks_semantic" / "x.png").unlink()
    assert not colab.extracted_dataset_ready(tmp_path)


def test_v4_colab_retry_skips_verified_download_and_extraction(tmp_path: Path, monkeypatch):
    import hashlib

    colab = _load()
    extracted = tmp_path / "dataset"
    extracted.mkdir()
    _make_v4_fixture(extracted)
    archive = tmp_path / "dataset.zip"
    archive.write_bytes(b"already-verified-test-archive")
    monkeypatch.setattr(colab, "WORK", tmp_path)
    monkeypatch.setattr(colab, "ARCHIVE", archive)
    monkeypatch.setattr(colab, "DATASET_ROOT", extracted)
    monkeypatch.setattr(colab, "DATASET_SHA256", hashlib.sha256(archive.read_bytes()).hexdigest())

    def no_download(*args, **kwargs):
        raise AssertionError("Verified local v4 dataset must not be redownloaded")

    def no_extract(*args, **kwargs):
        raise AssertionError("Complete extracted v4 dataset must not be extracted again")

    monkeypatch.setattr(colab.urllib.request, "urlopen", no_download)
    monkeypatch.setattr(colab.shutil, "unpack_archive", no_extract)
    colab.download_dataset()
    assert colab.extracted_dataset_ready(extracted)


def test_v4_colab_early_startup_diagnostics_are_enabled():
    trainer = Path(__file__).resolve().parents[1] / "src" / "resistor_model" / "train_segmentation.py"
    source = trainer.read_text(encoding="utf-8")
    assert '[boot] trainer started;' in source
    assert 'dump_traceback_later(' in source
    assert '[startup] indexing training images' in source
    assert '[startup] constructing LR-ASPP model' in source
    assert '[train] awaiting first DataLoader batch' in source
