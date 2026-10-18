"""Restore trained whole-resistor segmentation ONNX from verified GitHub Releases.

This is a Colab convenience layer, not a model downloader for arbitrary URLs:
each published artifact is pinned to the SHA-256 exposed by GitHub Releases.
Only the ONNX export (and its optional JSON metadata) is copied to Drive;
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


def ensure_segmentation_onnx(
    choice: str,
    drive_base: Path = Path("/content/drive/MyDrive/resistor_model"),
) -> Path:
    """Return the existing ONNX or install an SHA-verified release export.

    An already-present ONNX is always retained, including custom-trained ones.
    The first restore may download 500-970 MB; later runs use the Drive copy.
    For LR-ASPP only, a packaged best.pt is exported when the ZIP has no ONNX.
    """
    if choice not in RELEASE_ASSETS:
        raise ValueError(f"Unknown segmenter {choice!r}; expected lraspp or ppliteseg")
    asset = RELEASE_ASSETS[choice]
    result = Path(drive_base) / asset["drive_folder"] / asset["filename"]
    if result.is_file() and result.stat().st_size > 0:
        print(f"[segmenter-release] reusing existing Drive ONNX: {result}", flush=True)
        return result

    print(
        f"[segmenter-release] missing {result.name} in Drive; fetching {asset['url']} "
        f"({asset['bytes']/MIB:.0f} MiB, SHA-256 verified)", flush=True
    )
    result.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="resistor-segmenter-", dir="/content"
                                     if Path("/content").is_dir() else None) as temp:
        tmp = Path(temp)
        downloaded = tmp / "release.zip"
        _download_and_verify(asset["url"], downloaded, asset["sha256"], asset["bytes"])

        with zipfile.ZipFile(downloaded) as archive:
            names = archive.namelist()
            # Refuse ZIP traversal, even though we only extract selected members.
            if any(not _safe_member(name) for name in names):
                raise ValueError("GitHub Release ZIP includes an unsafe path")

            exported = _select_member(archive, asset["filename"])
            if exported is not None:
                _extract_member(archive, exported, result)
                print(
                    f"[segmenter-release] extracted {exported.filename} "
                    f"({result.stat().st_size/MIB:.1f} MiB) -> {result}", flush=True,
                )
                metadata = _select_member(archive, asset["filename"] + ".json")
                if metadata is not None:
                    _extract_member(archive, metadata, result.with_suffix(result.suffix+".json"))
            elif choice == "lraspp":
                checkpoint = _select_member(archive, "best.pt")
                if checkpoint is None:
                    raise FileNotFoundError(
                        "Verified LR-ASPP release contains neither resistor_segmenter_lraspp.onnx "
                        "nor best.pt; cannot build a segmentation model."
                    )
                checkpoint_path = tmp / "best.pt"
                _extract_member(archive, checkpoint, checkpoint_path)
                print(
                    "[segmenter-release] ONNX absent from release; exporting "
                    "packaged LR-ASPP best.pt checkpoint...", flush=True
                )
                from resistor_model.export_segmentation import export_onnx
                local_onnx = tmp / asset["filename"]
                export_onnx(checkpoint_path, local_onnx)
                if not local_onnx.is_file() or local_onnx.stat().st_size == 0:
                    raise RuntimeError("LR-ASPP checkpoint conversion did not produce an ONNX")
                partial = result.with_name(result.name + ".partial")
                try:
                    shutil.copyfile(local_onnx, partial)
                    os.replace(partial, result)
                finally:
                    partial.unlink(missing_ok=True)
                local_metadata = local_onnx.with_suffix(local_onnx.suffix + ".json")
                if local_metadata.exists():
                    shutil.copyfile(local_metadata, result.with_suffix(result.suffix+".json"))
            else:
                raise FileNotFoundError(
                    "Verified PP-LiteSeg v5 release has no "
                    f"{asset['filename']}. Re-export the trained model to ONNX first "
                    "or select SEGMENTER_CHOICE='lraspp'."
                )
    if not result.is_file() or result.stat().st_size == 0:
        raise RuntimeError("Segmentation ONNX was not installed")
    receipt = {
        "choice": choice,
        "release_url": asset["url"],
        "release_sha256": asset["sha256"],
        "model_sha256": _sha256(result),
        "model_bytes": result.stat().st_size,
    }
    (result.parent / (result.name + ".release.json")).write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[segmenter-release] saved and cached on Drive: {result}", flush=True)
    return result
