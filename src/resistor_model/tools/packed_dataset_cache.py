"""Single-file Drive cache for fully prepared Colab datasets (stdlib only).

Only one large archive is transferred through the Drive mount. All individual
images and masks are packed/extracted on the Colab VM's local filesystem.
A fingerprinted, checksummed manifest is published *last*, so interrupted
uploads cannot make a partial archive appear usable.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import time
import uuid
import zipfile

MARKER = "dataset.ready.json"
FORMAT_VERSION = 1
MIB = 1024 * 1024


def _log(message: str) -> None:
    print(f"[dataset-cache] {message}", flush=True)


def _referenced_archive(archive: Path, metadata: dict) -> Path:
    """Only allow immutable versioned siblings (or the original v1 ZIP)."""
    name = metadata.get("archive_name", archive.name)
    if (not isinstance(name, str) or Path(name).name != name or
            (name != archive.name and
             not (name.startswith(f"{archive.stem}.") and name.endswith(archive.suffix)))):
        raise ValueError("invalid prepared dataset archive name")
    return archive.with_name(name)


def _manifest(archive: Path, manifest: Path, fingerprint: dict) -> dict | None:
    """Cheap metadata check: never enumerate a Drive-backed dataset tree."""
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
        selected = _referenced_archive(archive, data)
        if (data.get("format") != FORMAT_VERSION or data.get("fingerprint") != fingerprint
                or not isinstance(data.get("sha256"), str)
                or len(data["sha256"]) != 64
                or not isinstance(data.get("size_bytes"), int)
                or data["size_bytes"] <= 0
                or not isinstance(data.get("file_count"), int)
                or data["file_count"] < 1
                or not isinstance(data.get("uncompressed_bytes"), int)
                or data["uncompressed_bytes"] < 0
                or selected.stat().st_size != data["size_bytes"]):
            return None
        return data
    except (OSError, ValueError, TypeError, AttributeError):
        return None

def prepared_usable_archive(archive: Path, manifest: Path, fingerprint: dict) -> bool:
    return _manifest(archive, manifest, fingerprint) is not None


def _copy_progress(src: Path, dst: Path, *, label: str, expected_sha: str | None = None) -> str:
    """Streaming copy, checksum and regular progress messages, no Drive file fanout."""
    total = src.stat().st_size
    digest = hashlib.sha256()
    done = 0
    started = last = time.monotonic()
    with src.open("rb") as reader, dst.open("wb") as writer:
        while True:
            chunk = reader.read(4 * MIB)
            if not chunk:
                break
            writer.write(chunk)
            digest.update(chunk)
            done += len(chunk)
            now = time.monotonic()
            if done == total or now - last >= 15:
                _log(f"{label}: {done / MIB:.0f}/{total / MIB:.0f} MiB "
                     f"({100 * done / total:.0f}%) in {now - started:.0f}s")
                last = now
    result = digest.hexdigest()
    if expected_sha and result != expected_sha:
        raise ValueError(f"{label}: SHA256 mismatch (cache damaged or incomplete)")
    return result


def persist_prepared_archive(local_root: Path, archive: Path, manifest: Path, fingerprint: dict) -> None:
    """Create ZIP locally and upload it as one file; publish manifest last."""
    local_root = Path(local_root)
    if not local_root.is_dir():
        raise RuntimeError(f"prepared dataset missing: {local_root}")
    (local_root / MARKER).write_text(json.dumps(fingerprint, indent=2), encoding="utf-8")
    files = sorted(p for p in local_root.rglob("*") if p.is_file())
    if not files:
        raise RuntimeError("prepared dataset is empty")
    if any(p.is_symlink() for p in files):
        raise RuntimeError("prepared dataset contains symlinks")
    raw_size = sum(p.stat().st_size for p in files)
    _log(f"packing {len(files):,} local files ({raw_size / MIB:.0f} MiB), "
         "then transferring one ZIP to Drive")
    local_zip = local_root.parent / f".prepared-dataset-{uuid.uuid4().hex}.zip"
    # Never overwrite an archive referenced by the currently published manifest.
    # A failed upload or interrupted manifest write must leave that cache usable.
    next_archive = archive.with_name(f"{archive.stem}.{uuid.uuid4().hex}{archive.suffix}")
    drive_tmp = next_archive.with_name(next_archive.name + ".partial")
    previous_archive = None
    try:
        previous_archive = _referenced_archive(
            archive, json.loads(manifest.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    try:
        started = last = time.monotonic()
        with zipfile.ZipFile(local_zip, "w", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=1, allowZip64=True) as output:
            for i, path in enumerate(files, 1):
                output.write(path, arcname=path.relative_to(local_root).as_posix())
                now = time.monotonic()
                if i == len(files) or i % 1000 == 0 or now - last >= 15:
                    _log(f"packing: {i:,}/{len(files):,} files ({100 * i / len(files):.0f}%) "
                         f"elapsed {now - started:.0f}s")
                    last = now
        archive.parent.mkdir(parents=True, exist_ok=True)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        _log(f"uploading {local_zip.stat().st_size / MIB:.0f} MiB cache archive to Drive")
        checksum = _copy_progress(local_zip, drive_tmp, label="upload")
        os.replace(drive_tmp, next_archive)
        metadata = {
            "archive_name": next_archive.name,
            "format": FORMAT_VERSION,
            "fingerprint": fingerprint,
            "sha256": checksum,
            "size_bytes": next_archive.stat().st_size,
            "file_count": len(files),
            "uncompressed_bytes": raw_size,
        }
        tmp_manifest = manifest.with_name(manifest.name + f".{uuid.uuid4().hex}.partial")
        try:
            tmp_manifest.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            os.replace(tmp_manifest, manifest)
        except BaseException:
            # The previous manifest still points to its own intact archive.
            next_archive.unlink(missing_ok=True)
            raise
        finally:
            tmp_manifest.unlink(missing_ok=True)
        if previous_archive and previous_archive != next_archive:
            try:
                previous_archive.unlink(missing_ok=True)
            except OSError as exc:
                _log(f"old cache cleanup skipped: {exc}")
        _log(f"archive cached in Drive; {len(files):,} files, "
             f"{metadata['size_bytes'] / MIB:.0f} MiB, sha256 {checksum[:16]}...")
    finally:
        local_zip.unlink(missing_ok=True)
        drive_tmp.unlink(missing_ok=True)


def _safe_member(name: str) -> bool:
    path = PurePosixPath(name)
    return (bool(name) and not name.startswith("/") and "\\" not in name
            and ":" not in name and all(p not in ("", ".", "..") for p in path.parts))


def restore_prepared_archive(archive: Path, manifest: Path, local_root: Path, fingerprint: dict) -> bool:
    """Verify and locally extract cache. Return False on corruption/cache miss."""
    metadata = _manifest(archive, manifest, fingerprint)
    if metadata is None:
        return False
    local_root.parent.mkdir(parents=True, exist_ok=True)
    local_zip = local_root.parent / f".restore-{uuid.uuid4().hex}.zip"
    staged = Path(tempfile.mkdtemp(prefix=".dataset-extract-", dir=local_root.parent))
    try:
        _log(f"restoring one archive from Drive ({metadata['size_bytes'] / MIB:.0f} MiB)")
        source_archive = _referenced_archive(archive, metadata)
        _copy_progress(source_archive, local_zip, label="download", expected_sha=metadata["sha256"])
        with zipfile.ZipFile(local_zip) as source:
            infos = source.infolist()
            members = [info for info in infos if not info.is_dir()]
            if (len(members) != metadata["file_count"] or
                    sum(i.file_size for i in members) != metadata["uncompressed_bytes"] or
                    len({i.filename for i in members}) != len(members)):
                raise ValueError("cache ZIP contents disagree with manifest")
            for info in infos:
                mode = info.external_attr >> 16
                if not _safe_member(info.filename) or (mode and stat.S_IFMT(mode)
                    not in (stat.S_IFREG, stat.S_IFDIR, 0)):
                    raise ValueError(f"unsafe ZIP entry: {info.filename}")
            started = last = time.monotonic()
            for i, info in enumerate(members, 1):
                target = staged.joinpath(*PurePosixPath(info.filename).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.open(info) as reader, target.open("wb") as writer:
                    shutil.copyfileobj(reader, writer, length=MIB)
                now = time.monotonic()
                if i == len(members) or i % 1000 == 0 or now - last >= 15:
                    _log(f"extracting locally: {i:,}/{len(members):,} files "
                         f"({100 * i / len(members):.0f}%) elapsed {now - started:.0f}s")
                    last = now
        marker = staged / MARKER
        if json.loads(marker.read_text(encoding="utf-8")) != fingerprint:
            raise ValueError("dataset marker does not match current recipe")
        if local_root.exists():
            shutil.rmtree(local_root)
        os.replace(staged, local_root)
        _log("prepared dataset restored to local SSD")
        return True
    except (OSError, ValueError, RuntimeError, EOFError, zipfile.BadZipFile, KeyError) as exc:
        _log(f"archive restore failed ({exc}); rebuilding the prepared dataset")
        return False
    finally:
        local_zip.unlink(missing_ok=True)
        if staged.exists():
            shutil.rmtree(staged)
