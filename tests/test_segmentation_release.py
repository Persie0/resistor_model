"""Regression tests for SHA-pinned ONNX release exports with external weights."""
import hashlib
import io
import json
from pathlib import Path
import shutil
import struct
import zipfile

import onnx
from onnx import TensorProto, helper
import pytest

from resistor_model.tools import segmentation_release as release


def _onnx_bytes(filename: str, *, external: bool = True) -> bytes:
    weight = TensorProto()
    weight.name = "weight"
    weight.data_type = TensorProto.FLOAT
    weight.dims.extend([1])
    if external:
        weight.data_location = TensorProto.EXTERNAL
        for key, value in (("location", filename + ".data"), ("offset", "0"), ("length", "4")):
            entry = weight.external_data.add()
            entry.key, entry.value = key, value
    else:
        weight.raw_data = struct.pack("<f", 2.0)

    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1])
    graph = helper.make_graph([helper.make_node("Add", ["X", "weight"], ["Y"])],
                              "fake-segmenter", [x], [y], [weight])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    return model.SerializeToString()


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
        assert (url, digest, byte_size) == (
            asset["url"], asset["sha256"], asset["bytes"])
        shutil.copyfile(archive, path)

    monkeypatch.setattr(release, "_download_and_verify", fake_download)
    return asset


@pytest.mark.parametrize("choice", ["lraspp", "ppliteseg"])
def test_extracts_graph_external_tensors_metadata_and_reuses_drive(monkeypatch, tmp_path, choice):
    filename = ("resistor_segmenter_lraspp.onnx" if choice=="lraspp"
                else "resistor_segmenter_ppliteseg.onnx")
    raw = _onnx_bytes(filename)
    weights = struct.pack("<f", 2.0)
    asset = _fake_asset(monkeypatch, tmp_path, choice, {
        filename: raw,
        filename + ".data": weights,
        filename + ".json": b'{"architecture":"unit-test"}',
        "output/best_model/large.bin": b"not extracted",
        "output/checkpoints/checkpoint.pt": b"not extracted",
    })
    drive = tmp_path / "drive"
    model = release.ensure_segmentation_onnx(choice, drive)
    assert model == drive / asset["drive_folder"] / filename
    assert model.read_bytes() == raw
    assert (model.parent/(filename+".data")).read_bytes() == weights
    assert model.with_suffix(".onnx.json").is_file()
    assert not (drive / asset["drive_folder"] / "output").exists()
    receipt = json.loads((model.parent/(filename+".release.json")).read_text())
    assert receipt["release_sha256"] == asset["sha256"]
    assert receipt["model_sha256"] == hashlib.sha256(raw).hexdigest()
    assert receipt["external_data_files"] == [{
        "location":filename+".data",
        "bytes":len(weights),
        "sha256":hashlib.sha256(weights).hexdigest(),
    }]
    onnx.load_model(str(model), load_external_data=True)

    monkeypatch.setattr(
        release, "_download_and_verify",
        lambda *a: pytest.fail("complete Drive ONNX should not redownload")
    )
    assert release.ensure_segmentation_onnx(choice, drive) == model


def test_repairs_old_onnx_only_cache_without_replacing_graph(monkeypatch, tmp_path):
    filename="resistor_segmenter_lraspp.onnx"
    raw=_onnx_bytes(filename)
    weights=struct.pack("<f",2.0)
    asset=_fake_asset(monkeypatch,tmp_path,"lraspp",{
        filename:raw,filename+".data":weights,"unrelated.bin":b"ignore"
    })
    drive=tmp_path/"drive"
    result=drive/asset["drive_folder"]/filename
    result.parent.mkdir(parents=True)
    result.write_bytes(raw)
    (result.parent/(filename+".release.json")).write_text(json.dumps({
        "choice":"lraspp",
        "release_sha256":asset["sha256"],
        "model_sha256":hashlib.sha256(raw).hexdigest(),
        "model_bytes":len(raw),
    }))
    assert release._missing_external_data(result)==[filename+".data"]
    repaired=release.ensure_segmentation_onnx("lraspp",drive)
    assert repaired == result
    assert result.read_bytes()==raw
    assert (result.parent/(filename+".data")).read_bytes()==weights
    assert not release._missing_external_data(result)
    onnx.load_model(str(result),load_external_data=True)

    monkeypatch.setattr(
        release, "_download_and_verify",
        lambda *a: pytest.fail("repaired release should be cached")
    )
    assert release.ensure_segmentation_onnx("lraspp",drive)==result


def test_does_not_overwrite_custom_incomplete_onnx(monkeypatch, tmp_path):
    filename="resistor_segmenter_lraspp.onnx"
    raw=_onnx_bytes(filename)
    asset=_fake_asset(monkeypatch,tmp_path,"lraspp",{
        filename:raw, filename+".data":struct.pack("<f",2.0),
    })
    dest=tmp_path/"drive"/asset["drive_folder"]/filename
    dest.parent.mkdir(parents=True)
    dest.write_bytes(raw)
    with pytest.raises(FileNotFoundError, match="custom ONNX"):
        release.ensure_segmentation_onnx("lraspp",tmp_path/"drive")
    assert dest.read_bytes()==raw


def test_missing_external_file_in_release_is_fatal(monkeypatch, tmp_path):
    filename="resistor_segmenter_lraspp.onnx"
    _fake_asset(monkeypatch,tmp_path,"lraspp",{filename:_onnx_bytes(filename)})
    with pytest.raises(FileNotFoundError,match="required external weights"):
        release.ensure_segmentation_onnx("lraspp",tmp_path/"drive")
    assert not (tmp_path/"drive"/"segmentation-lraspp"/filename).exists()


def test_self_contained_custom_model_does_not_need_release(monkeypatch, tmp_path):
    filename="resistor_segmenter_lraspp.onnx"
    dest=tmp_path/"drive"/"segmentation-v4-lraspp"/filename
    dest.parent.mkdir(parents=True)
    dest.write_bytes(_onnx_bytes(filename,external=False))
    monkeypatch.setattr(release, "_download_and_verify", lambda *a: pytest.fail("network"))
    assert release.ensure_segmentation_onnx("lraspp",tmp_path/"drive")==dest


def test_fails_cleanly_if_release_has_no_ppliteseg_onnx(monkeypatch, tmp_path):
    _fake_asset(monkeypatch, tmp_path, "ppliteseg", {
        "output/best_model/model.pdparams": b"paddle-weights",
    })
    with pytest.raises(FileNotFoundError, match="re-export"):
        release.ensure_segmentation_onnx("ppliteseg", tmp_path/"drive")


def test_rejects_archive_traversal(monkeypatch, tmp_path):
    filename="resistor_segmenter_lraspp.onnx"
    _fake_asset(monkeypatch, tmp_path, "lraspp", {
        filename: _onnx_bytes(filename),
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


def test_colab_preparation_cell_refreshes_stale_segmentation_imports():
    """Colab reuses sys.modules across reruns even when git clone is refreshed."""
    notebook = Path(__file__).resolve().parents[1] / "colab" / "prepare_mobile1d_segsynthetic_colab.ipynb"
    obj = json.loads(notebook.read_text(encoding="utf-8"))
    cells = [c for c in obj["cells"] if c["cell_type"] == "code"]
    source = "".join(cells[0]["source"])
    # Catch invalid notebook code before Colab spends minutes downloading data.
    compile(source, str(notebook) + ":main", "exec")
    eviction = source.index("del sys.modules[stale_name]")
    loader_import = source.index("import resistor_model.tools.segmentation_release as segmenter_release")
    assert eviction < loader_import
    assert "_missing_external_data" in source
    assert "expected_loader" in source
    assert "segmenter_release.ensure_segmentation_onnx" in source
