"""High-capacity Mobile1D XL stays separate and targets a 25 MB FP32 model."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import torch

from resistor_model.runtime import build_model

ROOT = Path(__file__).resolve().parents[1]
BASE_SCRIPT = ROOT / "colab" / "train_rres_mobile1d_colab.py"
XL_SCRIPT = ROOT / "colab" / "train_rres_mobile1d_xl_colab.py"


def _load_script(path: Path, name: str):
    spec = spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    obj = module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def test_xl_uses_isolated_checkpoints_but_shared_verified_dataset_cache():
    base = _load_script(BASE_SCRIPT, "mobile1d_base_xl_test")
    xl = _load_script(XL_SCRIPT, "mobile1d_xl_recipe_test")
    original_dir = base.RUN_DIR
    original_prepared_zip = base.DRIVE_PREPARED_ARCHIVE
    original_manifest = base.DRIVE_PREPARED_MANIFEST
    original_archive_root = base.DRIVE_ARCHIVE_ROOT
    xl.configure_xl(base)

    assert base.RUN_DIR != original_dir
    assert base.RUN_DIR.name == "r1-mobile1d-xl-colab"
    assert base.DRIVE_PREPARED_ARCHIVE == original_prepared_zip
    assert base.DRIVE_PREPARED_MANIFEST == original_manifest
    assert base.DRIVE_ARCHIVE_ROOT == original_archive_root
    assert original_dir.name == "r1-mobile1d-colab"
    assert "xl" in base.CONFIG_PATH.name
    assert "xl" in base.ZIP_PATH.name

    config = base.build_config(base.RUN_DIR, None, None)
    assert config["train"]["output_dir"] == str(base.RUN_DIR)
    assert config["train"]["resume"] is None
    assert config["model"]["architecture"] == "mobile1d"
    assert config["model"]["base_channels"] == 48
    assert config["model"]["d_model"] == 568
    assert config["data"]["output_size"] == [176, 512]
    assert config["train"]["epochs"] == 120
    assert config["train"]["batch_size"] == 8
    assert config["train"]["lr"] == 2.5e-4
    assert config["train"]["lr_initial_cycle_epochs"] is None


def test_xl_parameters_approximately_25mb():
    base = _load_script(BASE_SCRIPT, "mobile1d_base_capacity_test")
    xl = _load_script(XL_SCRIPT, "mobile1d_xl_capacity_test")
    old_config = base.build_config(base.RUN_DIR, None, None)
    old_model = build_model(old_config)
    old_params = sum(p.numel() for p in old_model.parameters())

    xl.configure_xl(base)
    xl_config = base.build_config(base.RUN_DIR, None, None)
    model = build_model(xl_config)
    xl_params = sum(p.numel() for p in model.parameters())
    assert old_params == 369_116
    assert xl_params == 6_333_985
    assert 24_500_000 < xl_params * 4 < 26_000_000


def test_xl_forward_contract_is_unchanged():
    base = _load_script(BASE_SCRIPT, "mobile1d_base_forward_test")
    xl = _load_script(XL_SCRIPT, "mobile1d_xl_forward_test")
    xl.configure_xl(base)
    model = build_model(base.build_config(base.RUN_DIR, None, None)).eval()
    with torch.inference_mode():
        result = model(torch.zeros(1, 3, 64, 128))
    assert result["dense_logits"].shape == (1, 13, 256)
    assert result["slot_color_logits"].shape == (1, 6, 12)
    assert result["count_logits"].shape == (1, 7)
    assert result["slot_center"].shape == (1, 6)
    assert result["embedding"].shape == (1, 640)


def test_compact_package_excludes_xl_snapshots(monkeypatch, tmp_path: Path):
    base = _load_script(BASE_SCRIPT, "mobile1d_package_xl_test")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "best.pt").write_bytes(b"best")
    (run_dir / "last.pt").write_bytes(b"last")
    (run_dir / "checkpoints").mkdir()
    (run_dir / "checkpoints" / "epoch_005.pt").write_bytes(b"huge-state")
    monkeypatch.setattr(base, "RUN_DIR", run_dir)
    monkeypatch.setattr(base, "ZIP_PATH", tmp_path / "xl.zip")
    base.package_and_download(download=False, include_recovery_snapshots=False)
    import zipfile

    with zipfile.ZipFile(tmp_path / "xl.zip") as archive:
        assert set(archive.namelist()) == {"best.pt", "last.pt"}
    assert (run_dir / "checkpoints" / "epoch_005.pt").is_file()
