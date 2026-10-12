"""Portable cache tests: no Colab, Paddle, network or GPU required."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import zipfile

import pytest

from resistor_model.tools import packed_dataset_cache as cache


def _fixture(tmp_path):
    local = tmp_path / "local"
    (local / "train" / "images").mkdir(parents=True)
    (local / "train" / "masks_semantic").mkdir()
    (local / "train" / "images" / "one.jpg").write_bytes(b"image" * 1000)
    (local / "train" / "masks_semantic" / "one.png").write_bytes(b"mask" * 800)
    (local / "valid").mkdir()
    (local / "valid" / "a.txt").write_text("foo")
    archive = tmp_path / "drive" / "prepared_dataset.zip"
    manifest = tmp_path / "drive" / "prepared_dataset.manifest.json"
    fingerprint = {"recipe": "test", "variant": 1}
    return local, archive, manifest, fingerprint


def _trainer():
    script = Path(__file__).resolve().parents[1] / "colab" / "train_pp_liteseg_colab.py"
    spec = importlib.util.spec_from_file_location("pp_liteseg_test_cache", script)
    assert spec and spec.loader
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    return trainer


def test_roundtrip_and_one_archive(tmp_path):
    local, archive, manifest, fp = _fixture(tmp_path)
    cache.persist_prepared_archive(local, archive, manifest, fp)
    assert cache.prepared_usable_archive(archive, manifest, fp)
    assert not cache.prepared_usable_archive(archive, manifest, {**fp, "variant": 2})
    assert {p.name for p in archive.parent.iterdir()} == {
        "prepared_dataset.zip", "prepared_dataset.manifest.json",
    }
    restored = tmp_path / "restored"
    restored.mkdir()
    (restored / "stale.txt").write_text("stale")
    assert cache.restore_prepared_archive(archive, manifest, restored, fp)
    assert not (restored / "stale.txt").exists()
    for file in local.rglob("*"):
        if file.is_file():
            assert (restored / file.relative_to(local)).read_bytes() == file.read_bytes()


def test_corrupt_archive_rejected_without_deleting_local(tmp_path):
    local, archive, manifest, fp = _fixture(tmp_path)
    cache.persist_prepared_archive(local, archive, manifest, fp)
    with archive.open("r+b") as handle:
        handle.seek(120)
        handle.write(b"corrupt!")
    assert cache.prepared_usable_archive(archive, manifest, fp)  # metadata is still valid
    restored = tmp_path / "retained"
    restored.mkdir()
    (restored / "original.txt").write_text("safe")
    assert not cache.restore_prepared_archive(archive, manifest, restored, fp)
    assert (restored / "original.txt").read_text() == "safe"


def test_partial_manifest_is_cache_miss(tmp_path):
    local, archive, manifest, fp = _fixture(tmp_path)
    assert not cache.prepared_usable_archive(archive, manifest, fp)
    archive.parent.mkdir(parents=True)
    manifest.write_text("{not json}")
    assert not cache.prepared_usable_archive(archive, manifest, fp)
    assert not cache.restore_prepared_archive(archive, manifest, tmp_path / "other", fp)


def test_unsafe_zip_entry_rejected(tmp_path):
    local, archive, manifest, fp = _fixture(tmp_path)
    cache.persist_prepared_archive(local, archive, manifest, fp)
    with zipfile.ZipFile(archive, "a") as z:
        z.writestr("../escape.txt", "evil")
    metadata = json.loads(manifest.read_text())
    metadata["size_bytes"] = archive.stat().st_size
    metadata["sha256"] = cache._copy_progress(archive, tmp_path / "copy.zip", label="test")
    metadata["file_count"] += 1
    metadata["uncompressed_bytes"] += 4
    manifest.write_text(json.dumps(metadata))
    assert not cache.restore_prepared_archive(archive, manifest, tmp_path / "restored", fp)
    assert not (tmp_path / "escape.txt").exists()


def test_interrupted_upload_preserves_previous_cache(tmp_path, monkeypatch):
    local, archive, manifest, fp = _fixture(tmp_path)
    cache.persist_prepared_archive(local, archive, manifest, fp)
    old_manifest, old_archive = manifest.read_bytes(), archive.read_bytes()

    def fail_upload(src, dest, *, label, expected_sha=None):
        dest.write_bytes(b"partial")
        raise OSError("simulated Drive upload failure")

    monkeypatch.setattr(cache, "_copy_progress", fail_upload)
    with pytest.raises(OSError, match="simulated"):
        cache.persist_prepared_archive(local, archive, manifest, {**fp, "variant": 2})
    assert manifest.read_bytes() == old_manifest
    assert archive.read_bytes() == old_archive
    assert not list(archive.parent.glob("*.partial"))


def test_trainer_restores_from_archive_without_legacy_rsync(tmp_path):
    trainer = _trainer()
    local, archive, manifest, fp = _fixture(tmp_path)
    (local / "train" / "_annotations.coco.json").write_text("{}")
    cache.persist_prepared_archive(local, archive, manifest, fp)
    trainer.packed_dataset_cache = cache
    trainer.DRIVE_PREPARED_ARCHIVE = archive
    trainer.DRIVE_PREPARED_MANIFEST = manifest
    trainer.DATASET_ROOT = tmp_path / "restored"
    trainer.DATASET_FINGERPRINT = fp
    trainer.dataset_cache = None  # if old rsync path is reached, this fails
    root, origin = trainer.prepare_dataset()
    assert root == trainer.DATASET_ROOT
    assert origin == "Drive ZIP cache"
    assert (root / "train" / "images" / "one.jpg").is_file()


def test_result_package_excludes_legacy_and_new_caches(tmp_path, monkeypatch):
    trainer = _trainer()
    run = tmp_path / "drive"
    (run / "dataset" / "train").mkdir(parents=True)
    (run / "dataset" / "train" / "old.jpg").write_bytes(b"old" * 1000)
    (run / "archives").mkdir()
    (run / "archives" / "old.zip").write_bytes(b"old")
    (run / "prepared_dataset.zip").write_bytes(b"huge cached dataset")
    (run / "prepared_dataset.manifest.json").write_text("{}")
    (run / "output" / "best_model").mkdir(parents=True)
    (run / "output" / "best_model" / "model.pdparams").write_bytes(b"weights")
    (run / "resistor_segmenter_ppliteseg.onnx").write_bytes(b"onnx")
    (run / "training_summary.json").write_text("{}")
    trainer.RUN_DIR = run
    trainer.OUTPUT_DRIVE = run / "output"
    trainer.DRIVE_PREPARED_MANIFEST = run / "prepared_dataset.manifest.json"
    trainer.ZIP_PATH = tmp_path / "results.zip"
    trainer.package_and_download()
    with zipfile.ZipFile(trainer.ZIP_PATH) as output:
        assert set(output.namelist()) == {
            "output/best_model/model.pdparams",
            "resistor_segmenter_ppliteseg.onnx",
            "training_summary.json",
        }
