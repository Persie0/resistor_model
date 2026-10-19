"""Restore trained whole-resistor segmentation ONNX from verified GitHub Releases.

This is a Colab convenience layer, not a model downloader for arbitrary URLs:
each published artifact is pinned to the SHA-256 exposed by GitHub Releases.
Only the ONNX export, its external tensor data, and optional JSON metadata are copied to Drive;
large training checkpoints and dataset files are not extracted.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import time
import urllib.request
import zipfile

RELEASE_ASSETS = {
    "lraspp": {
        "url": "https://github.com/Persie0/resistor_model/releases/download/ee/resistor-segmentation-v4-lraspp.zip",
        "sha256": "bc0d550f240318371b40c08135142c1c00889f9e701a793dbc1ad3f8953d59ce",
        "bytes": 513301388,
        "filename": "resistor_segmenter_lraspp.onnx",
        "drive_folder": "segmentation-v4-lraspp",
    },
    "ppliteseg": {
        "url": "https://github.com/Persie0/resistor_model/releases/download/v5/resistor-segmentation-pp-liteseg.zip",
        "sha256": "2ebd7d34e1ffda5549634f78f119c18d9203abdbeab875252e07baa94e956476",
        "bytes": 966439779,
        "filename": "resistor_segmenter_ppliteseg.onnx",
        "drive_folder": "segmentation-pp-liteseg",
    },
}
MIB = 1024 * 1024


def _sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(4 * MIB), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _download_and_verify(url: str, dest: Path, digest: str, byte_size: int) -> None:
    """Download to ephemeral local SSD; refuse modified/truncated artifacts."""
    temporary = dest.with_name(dest.name + ".partial")
    dest.parent.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    sha = hashlib.sha256()
    start = last = time.monotonic()
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "resistor-model-Colab-segmentation-restore"}
        )
        with urllib.request.urlopen(req, timeout=120) as response, temporary.open("wb") as file:
            while chunk := response.read(4 * MIB):
                file.write(chunk)
                sha.update(chunk)
                downloaded += len(chunk)
                now = time.monotonic()
                if now - last >= 15 or downloaded == byte_size:
                    print(
                        f"[segmenter-release] download {downloaded/MIB:.1f}/"
                        f"{byte_size/MIB:.1f} MiB ({100*downloaded/max(byte_size,1):.1f}%) "
                        f"in {now-start:.0f}s", flush=True
                    )
                    last = now
        if downloaded != byte_size:
            raise ValueError(f"Release size mismatch: expected {byte_size}, got {downloaded}")
        actual = sha.hexdigest()
        if actual != digest:
            raise ValueError(f"Release SHA-256 mismatch: expected {digest}, got {actual}")
        os.replace(temporary, dest)
    finally:
        temporary.unlink(missing_ok=True)


def _safe_member(name: str) -> bool:
    path = PurePosixPath(name)
    return (bool(name) and not name.startswith("/") and "\\" not in name
            and ":" not in name and all(p not in ("", ".", "..") for p in path.parts))


def _extract_member(archive: zipfile.ZipFile, member: zipfile.ZipInfo, target: Path) -> None:
    if not _safe_member(member.filename):
        raise ValueError(f"Unsafe ZIP member: {member.filename}")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    try:
        with archive.open(member) as src, partial.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=4*MIB)
        os.replace(partial, target)
    finally:
        partial.unlink(missing_ok=True)


def _select_member(archive: zipfile.ZipFile, name: str) -> zipfile.ZipInfo | None:
    matches = [i for i in archive.infolist()
               if not i.is_dir() and PurePosixPath(i.filename).name == name
               and _safe_member(i.filename)]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous release ZIP entries for {name!r}")
    return matches[0] if matches else None


def _external_data_locations(model_path: Path) -> list[str]:
    """Read ONNX protobuf refs without loading the potentially large weights."""
    import onnx

    model = onnx.load_model(str(model_path), load_external_data=False)
    locations: set[str] = set()
    for tensor in model.graph.initializer:
        if tensor.data_location != onnx.TensorProto.EXTERNAL:
            continue
        attributes = {kv.key: kv.value for kv in tensor.external_data}
        location = attributes.get("location")
        if not location or not _safe_member(location):
            raise ValueError(f"Unsafe or missing ONNX external tensor location: {location!r}")
        locations.add(location)
    return sorted(locations)


def _external_path(model_path: Path, location: str) -> Path:
    if not _safe_member(location):
        raise ValueError(f"Unsafe ONNX external tensor location: {location!r}")
    return model_path.parent.joinpath(*PurePosixPath(location).parts)


def _missing_external_data(model_path: Path) -> list[str]:
    return [
        location for location in _external_data_locations(model_path)
        if not _external_path(model_path, location).is_file()
        or _external_path(model_path, location).stat().st_size == 0
    ]


def _required_external_member(
    archive: zipfile.ZipFile, exported: zipfile.ZipInfo, location: str,
) -> zipfile.ZipInfo:
    """Match external data relative to the ONNX's path inside the ZIP."""
    expected = (PurePosixPath(exported.filename).parent / location).as_posix()
    matches = [i for i in archive.infolist() if i.filename == expected and not i.is_dir()]
    if len(matches) != 1 or not _safe_member(expected):
        raise FileNotFoundError(
            f"Release contains {exported.filename!r} referencing {location!r}, "
            f"but the required external weights {expected!r} are missing."
        )
    return matches[0]


def _validate_complete_export(model_path: Path) -> list[str]:
    """Require every external data sidecar to exist and be ONNX-readable."""
    missing = _missing_external_data(model_path)
    if missing:
        raise FileNotFoundError(
            f"ONNX external weights missing next to {model_path}: {', '.join(missing)}"
        )
    # Loading with external_data=True also catches truncated or broken tensors.
    import onnx
    onnx.load_model(str(model_path), load_external_data=True)
    return _external_data_locations(model_path)


def _write_receipt(choice: str, asset: dict, result: Path, locations: list[str]) -> None:
    receipt = {
        "choice": choice,
        "release_url": asset["url"],
        "release_sha256": asset["sha256"],
        "model_sha256": _sha256(result),
        "model_bytes": result.stat().st_size,
        "external_data_files": [
            {
                "location": loc,
                "bytes": _external_path(result, loc).stat().st_size,
                "sha256": _sha256(_external_path(result, loc)),
            } for loc in locations
        ],
    }
    (result.parent / (result.name + ".release.json")).write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )


def ensure_segmentation_onnx(
    choice: str,
    drive_base: Path = Path("/content/drive/MyDrive/resistor_model"),
) -> Path:
    """Return complete ONNX + tensor sidecars; repair incomplete release caches.

    Custom models are retained if complete. A previous incomplete release
    extraction is safely repaired after checking its recorded model checksum.
    """
    if choice not in RELEASE_ASSETS:
        raise ValueError(f"Unknown segmenter {choice!r}; expected lraspp or ppliteseg")
    asset = RELEASE_ASSETS[choice]
    result = Path(drive_base) / asset["drive_folder"] / asset["filename"]
    receipt_path = result.parent / (result.name + ".release.json")
    have_model = result.is_file() and result.stat().st_size > 0
    missing: list[str] = []
    if have_model:
        missing = _missing_external_data(result)
        if not missing:
            print(f"[segmenter-release] reusing complete Drive ONNX: {result}", flush=True)
            return result
        if not receipt_path.is_file():
            raise FileNotFoundError(
                f"Existing custom ONNX at {result} requires external files {missing}, "
                "but has no verified release receipt. Restore its own sidecars "
                "or move it aside to install the published model."
            )
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (receipt.get("model_sha256") != _sha256(result)
                or receipt.get("release_sha256") != asset["sha256"]
                or receipt.get("choice") != choice):
            raise ValueError(
                "Incomplete ONNX differs from recorded release. "
                "Refusing to overwrite an unknown/custom model."
            )
        print(
            "[segmenter-release] incomplete cached ONNX; restoring missing "
            f"external weights: {missing}", flush=True
        )
    else:
        print(f"[segmenter-release] missing ONNX in Drive: {result}", flush=True)

    print(
        f"[segmenter-release] fetching {asset['url']} "
        f"({asset['bytes']/MIB:.0f} MiB, SHA-256 verified)",
        flush=True,
    )
    result.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="resistor-segmenter-",
        dir="/content" if Path("/content").is_dir() else None,
    ) as temp:
        tmp = Path(temp)
        archive_path = tmp / "release.zip"
        _download_and_verify(asset["url"], archive_path, asset["sha256"], asset["bytes"])
        with zipfile.ZipFile(archive_path) as archive:
            names = archive.namelist()
            if any(not _safe_member(name) for name in names):
                raise ValueError("GitHub Release ZIP includes an unsafe path")
            exported = _select_member(archive, asset["filename"])

            if exported is None:
                if have_model:
                    raise FileNotFoundError(
                        f"Cannot repair {result}: release no longer includes the ONNX export"
                    )
                if choice != "lraspp":
                    raise FileNotFoundError(
                        "Verified PP-LiteSeg release does not contain "
                        f"{asset['filename']}; re-export the trained model."
                    )
                checkpoint = _select_member(archive, "best.pt")
                if checkpoint is None:
                    raise FileNotFoundError(
                        "Verified LR-ASPP release contains neither ONNX nor best.pt."
                    )
                from resistor_model.export_segmentation import export_onnx
                checkpoint_path = tmp / "best.pt"
                _extract_member(archive, checkpoint, checkpoint_path)
                print("[segmenter-release] exporting packaged LR-ASPP best.pt...", flush=True)
                staged = tmp / asset["filename"]
                export_onnx(checkpoint_path, staged)
            else:
                staged = tmp / asset["filename"]
                _extract_member(archive, exported, staged)

            # Discover tensor locations from the actual ONNX graph, rather than
            # assuming a fixed .data basename or omitting external weights.
            all_locations = _external_data_locations(staged)
            needed = missing if have_model else all_locations
            for location in needed:
                if exported is None:
                    source = _external_path(staged, location)
                    if not source.is_file() or source.stat().st_size == 0:
                        raise FileNotFoundError(
                            f"Checkpoint export omitted ONNX tensor data {location}"
                        )
                else:
                    member = _required_external_member(archive, exported, location)
                    source = _external_path(staged, location)
                    _extract_member(archive, member, source)
                print(
                    f"[segmenter-release] restored external weights {location} "
                    f"({_external_path(staged, location).stat().st_size/MIB:.1f} MiB)",
                    flush=True,
                )

            if have_model:
                if all_locations != _external_data_locations(result):
                    raise ValueError("Release ONNX external data references differ from cached model")
                # Do not overwrite the existing ONNX bytes. The sidecars alone
                # were omitted by the previous downloader.
                for location in needed:
                    destination = _external_path(result, location)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    partial = destination.with_name(destination.name + ".partial")
                    try:
                        shutil.copyfile(_external_path(staged, location), partial)
                        os.replace(partial, destination)
                    finally:
                        partial.unlink(missing_ok=True)
            else:
                # Weights first, ONNX model last: a disconnect cannot expose a
                # new graph as complete before its external tensors are present.
                for location in all_locations:
                    destination = _external_path(result, location)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    partial = destination.with_name(destination.name + ".partial")
                    try:
                        shutil.copyfile(_external_path(staged, location), partial)
                        os.replace(partial, destination)
                    finally:
                        partial.unlink(missing_ok=True)
                partial = result.with_name(result.name + ".partial")
                try:
                    shutil.copyfile(staged, partial)
                    os.replace(partial, result)
                finally:
                    partial.unlink(missing_ok=True)

            if exported is not None:
                metadata = _select_member(archive, asset["filename"] + ".json")
                if metadata is not None:
                    _extract_member(
                        archive, metadata, result.with_suffix(result.suffix + ".json")
                    )
            else:
                local_meta = staged.with_suffix(staged.suffix + ".json")
                if local_meta.is_file():
                    shutil.copyfile(local_meta, result.with_suffix(result.suffix + ".json"))

    locations = _validate_complete_export(result)
    _write_receipt(choice, asset, result, locations)
    print(
        f"[segmenter-release] complete model + {len(locations)} external file(s) "
        f"saved to Drive: {result}", flush=True
    )
    return result
