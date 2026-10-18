"""Tests for resolving GitHub-release segmenter without internet or huge assets."""
import hashlib
import io
import json
from pathlib import Path
import shutil
import zipfile

import pytest

from resistor_model.tools import segmentation_release as release


def _fake_asset(monkeypatch, tmp_path: Path, choice: str, members: dict):
    archive = tmp_path / "fixture-release.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for name, data in members.items():
            z.writestr(name, data)
    asset = dict(
        url="https://github.com/Persie0/resistor_model/releases/download/fake/fake.zip",
        sha256=release._sha256(archive),
        bytes=archive.stat().st_size,
        filename=("resistor_segmenter_lraspp.onnx" if choice=="lraspp"
                  else "resistor_segmenter_ppliteseg.onnx"),
        drive_folder="segmentation-"+choice,
    )
    monkeypatch.setitem(release.RELEASE_ASSETS, choice, asset)
    def fake_download(url, path, digest, byte_size):
        assert url == asset["url"]
        assert digest == asset["sha256"]
        assert byte_size == asset["bytes"]
        shutil.copyfile(archive, path)
    monkeypatch.setattr(release, "_download_and_verify", fake_download)
    return asset


@pytest.mark.parametrize("choice", ["lraspp", "ppliteseg"])
def test_extracts_only_export_and_metadata_then_reuses_drive(monkeypatch, tmp_path, choice):
    filename = ("resistor_segmenter_lraspp.onnx" if choice=="lraspp"
                else "resistor_segmenter_ppliteseg.onnx")
    asset = _fake_asset(monkeypatch, tmp_path, choice, {
        filename: b"fake-but-nonempty-onnx",
        filename+".json": b'{"architecture":"unit-test"}',
        "output/best_model/large.bin": b"not extracted",
        "output/checkpoints/checkpoint.pt": b"not extracted",
    })
    drive = tmp_path / "drive"
    model = release.ensure_segmentation_onnx(choice, drive)
    assert model == drive / asset["drive_folder"] / filename
    assert model.read_bytes() == b"fake-but-nonempty-onnx"
    assert model.with_suffix(".onnx.json").is_file()
    assert not (drive / asset["drive_folder"] / "output").exists()
    receipt = json.loads((model.parent/(filename+".release.json")).read_text())
    assert receipt["release_sha256"] == asset["sha256"]
    assert receipt["model_sha256"] == hashlib.sha256(model.read_bytes()).hexdigest()

    # Never download/overwrite a user-provided or already cached ONNX.
    monkeypatch.setattr(release, "_download_and_verify", lambda *a: pytest.fail("network should not be used"))
    assert release.ensure_segmentation_onnx(choice, drive) == model
    assert model.read_bytes() == b"fake-but-nonempty-onnx"


def test_fails_cleanly_if_release_has_no_ppliteseg_onnx(monkeypatch, tmp_path):
    _fake_asset(monkeypatch, tmp_path, "ppliteseg", {
        "output/best_model/model.pdparams": b"paddle-weights",
    })
    with pytest.raises(FileNotFoundError, match="Re-export"):
        release.ensure_segmentation_onnx("ppliteseg", tmp_path/"drive")


def test_rejects_archive_traversal(monkeypatch, tmp_path):
    _fake_asset(monkeypatch, tmp_path, "lraspp", {
        "resistor_segmenter_lraspp.onnx": b"onnx",
        "../unsafe-entry": b"bad",
    })
    with pytest.raises(ValueError, match="unsafe path"):
        release.ensure_segmentation_onnx("lraspp", tmp_path/"drive")


def test_download_rejects_wrong_hash_and_incomplete_transfer(monkeypatch, tmp_path):
    content = b"hello-untrusted-release"
    monkeypatch.setattr(
        release.urllib.request, "urlopen",
        lambda *_args, **_kw: io.BytesIO(content),
    )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        release._download_and_verify(
            "https://github.com/example", tmp_path/"output.zip",
            "0"*64, len(content),
        )
    assert not (tmp_path/"output.zip").exists()
    assert not (tmp_path/"output.zip.partial").exists()
    with pytest.raises(ValueError, match="size mismatch"):
        release._download_and_verify(
            "https://github.com/example", tmp_path/"output.zip",
            hashlib.sha256(content).hexdigest(), len(content)+1,
        )


def test_unknown_choice_fails_without_network(tmp_path):
    with pytest.raises(ValueError, match="Unknown segmenter"):
        release.ensure_segmentation_onnx("other", tmp_path)
