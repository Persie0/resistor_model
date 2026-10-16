"""No-GPU smoke tests for the Drive-backed Mobile1D Colab dataset cache."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from zipfile import ZipFile

from resistor_model.tools import packed_dataset_cache

SCRIPT = Path(__file__).resolve().parents[1] / "colab" / "train_rres_mobile1d_colab.py"
spec = spec_from_file_location("mobile1d_colab_cache_under_test", SCRIPT)
assert spec is not None and spec.loader is not None
colab = module_from_spec(spec)
spec.loader.exec_module(colab)


def _fill_prepared_dataset():
    colab.DATA_ROOT.mkdir(parents=True, exist_ok=True)
    for name in ("manifest.jsonl", "splits.json", "import-report.json"):
        (colab.DATA_ROOT / name).write_text("test-data", encoding="utf-8")
    for path in colab.dataset_roots():
        path.mkdir(parents=True, exist_ok=True)


def test_cache_fingerprint_covers_immutable_sources_and_splits():
    assert set(colab.DATASET_SHAS) == set(colab.DATASET_ASSETS)
    assert all(len(sha) == 64 for sha in colab.DATASET_SHAS.values())
    assert colab.DATASET_FINGERPRINT["group_session"] is True
    assert colab.DATASET_FINGERPRINT["seed"] == colab.SEED


def test_prepared_drive_zip_restores_without_reimporting(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(colab, "DATA_ROOT", tmp_path / "dataset")
    monkeypatch.setattr(colab, "DRIVE_PREPARED_ARCHIVE", tmp_path / "prepared.zip")
    monkeypatch.setattr(colab, "DRIVE_PREPARED_MANIFEST", tmp_path / "prepared.json")
    monkeypatch.setattr(packed_dataset_cache, "prepared_usable_archive", lambda *args: True)

    def restore(*args):
        _fill_prepared_dataset()
        return True

    monkeypatch.setattr(packed_dataset_cache, "restore_prepared_archive", restore)
    monkeypatch.setattr(colab, "download_release_datasets", lambda: (_ for _ in ()).throw(
        AssertionError("valid Drive cache must skip network downloads")))
    monkeypatch.setattr(colab, "_create_manifest_and_splits", lambda: (_ for _ in ()).throw(
        AssertionError("valid Drive cache must skip preprocessing")))
    colab.prepare_dataset()
    assert colab._prepared_valid()


def test_invalid_drive_cache_rebuilds_and_persists(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(colab, "DATA_ROOT", tmp_path / "dataset")
    monkeypatch.setattr(colab, "DRIVE_PREPARED_ARCHIVE", tmp_path / "prepared.zip")
    monkeypatch.setattr(colab, "DRIVE_PREPARED_MANIFEST", tmp_path / "prepared.json")
    monkeypatch.setattr(packed_dataset_cache, "prepared_usable_archive", lambda *args: False)
    calls = []
    monkeypatch.setattr(colab, "download_release_datasets", lambda: calls.append("download"))
    monkeypatch.setattr(colab, "_create_manifest_and_splits", lambda: (
        calls.append("import"), _fill_prepared_dataset()))
    monkeypatch.setattr(packed_dataset_cache, "persist_prepared_archive", lambda *args:
        calls.append("persist"))
    colab.prepare_dataset()
    assert calls == ["download", "import", "persist"]


def test_result_zip_does_not_embed_drive_training_cache(monkeypatch, tmp_path: Path):
    root = tmp_path / "drive-run"
    root.mkdir()
    (root / "best.pt").write_bytes(b"weight")
    (root / "training_summary.json").write_text("{}")
    (root / "prepared_dataset.zip").write_bytes(b"large-dataset-cache")
    (root / "prepared_dataset.manifest.json").write_text("{}")
    (root / "archives").mkdir()
    (root / "archives" / "source.zip").write_bytes(b"large-source")
    zip_path = tmp_path / "model-results.zip"
    monkeypatch.setattr(colab, "RUN_DIR", root)
    monkeypatch.setattr(colab, "ZIP_PATH", zip_path)
    colab.package_and_download(download=False)
    with ZipFile(zip_path) as archive:
        assert set(archive.namelist()) == {"best.pt", "training_summary.json"}


def test_mobile1d_checkpoint_frequency_is_opt_in(tmp_path: Path):
    cfg = colab.build_config(tmp_path / "run", None, None)
    assert cfg["train"]["checkpoint_every"] == 5
    assert cfg["train"]["resume"] is None


def test_resume_prefers_last_then_latest_numbered_snapshot(tmp_path: Path):
    recovery = tmp_path / "checkpoints"
    recovery.mkdir()
    (recovery / "epoch_005.pt").write_bytes(b"old")
    (recovery / "epoch_010.pt").write_bytes(b"new")
    assert colab.find_resume_checkpoint(tmp_path) == recovery / "epoch_010.pt"
    (tmp_path / "last.pt").write_bytes(b"latest")
    assert colab.find_resume_checkpoint(tmp_path) == tmp_path / "last.pt"

def test_colab_editable_package_visible_without_kernel_restart(monkeypatch, tmp_path: Path):
    """pip editable .pth files need not be refreshed in a running Colab kernel."""
    import os
    import subprocess
    import sys

    source = tmp_path / "training-repo" / "src"
    stub_package = source / "resistor_model"
    stub_package.mkdir(parents=True)
    (stub_package / "__init__.py").write_text("VERSION = 'fresh-clone'\n")
    monkeypatch.setattr(colab, "MODEL_REPO", tmp_path / "training-repo")
    monkeypatch.setattr(sys, "path", [p for p in sys.path if p != str(source)])
    monkeypatch.setenv("PYTHONPATH", "")

    activated = colab.activate_local_package()
    assert activated == source.resolve()
    assert str(source.resolve()) == sys.path[0]
    assert str(source.resolve()) in os.environ["PYTHONPATH"].split(os.pathsep)
    # Simulate a new training subprocess: it must see the same package source.
    test = subprocess.run(
        [sys.executable, "-c", "import resistor_model; print(resistor_model.__file__)"],
        env=os.environ.copy(), capture_output=True, text=True, check=True,
    )
    assert str(source.resolve()) in test.stdout
    # Repeated activation must not accumulate duplicate PYTHONPATH entries.
    colab.activate_local_package()
    assert os.environ["PYTHONPATH"].split(os.pathsep).count(str(source.resolve())) == 1
