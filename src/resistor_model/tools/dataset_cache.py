"""Drive-backed cache for prepared segmentation datasets (stdlib only).

Rebuilding the training tree on every fresh Colab session is wasteful:
~870 MiB of archives re-downloaded, blue pairs re-merged, ~14k synth JPEGs
re-encoded and ~20k masks re-scanned. This module persists the *prepared*
tree (v4 + blue merge + synth + 0/1 masks) in Drive behind a fingerprint, so
reruns restore instead of rebuild:

* ``prepared_usable(drive_dir, fingerprint)``: True only if the Drive marker
  matches the current recipe exactly (archive SHAs, synth variants/seed...).
  Any recipe change automatically rebuilds.
* ``persist_prepared(local_root, drive_dir, fingerprint)``: writes the marker
  and mirrors local -> Drive (rsync --delete when available).
* ``restore_prepared(drive_dir, local_root)``: clears local and mirrors
  Drive -> local. Refuses when the marker is missing (never wipe Drive).
* ``fetch_cached_archive(...)``: local zip, else Drive backup, else download
  (with progress) + back up to Drive.

No torch/cv2/PIL: safe to import in any Colab env, including Paddle-only ones
(via ``sys.path`` pointing at the repo's ``src/``).
"""

from __future__ import annotations

from collections.abc import Callable
import hashlib
import json
import shutil
import subprocess
import urllib.request
from pathlib import Path

MARKER = "dataset.ready.json"


def sha_of(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def tree_size_mb(path: Path) -> float:
    total = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return total / 1024 / 1024


def prepared_usable(drive_dir: Path, fingerprint: dict) -> bool:
    marker = drive_dir / MARKER
    if not marker.is_file():
        return False
    try:
        saved = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return saved == fingerprint


def write_marker(local_root: Path, fingerprint: dict) -> None:
    (local_root / MARKER).write_text(json.dumps(fingerprint, indent=2), encoding="utf-8")


def sync_tree(src: Path, dst: Path, *, label: str) -> None:
    """Mirror src -> dst (rsync --delete when available, else copytree)."""
    if not (src / MARKER).is_file():
        raise RuntimeError(f"refusing to mirror {src}: no {MARKER} (source not prepared)")
    dst.mkdir(parents=True, exist_ok=True)
    rsync = shutil.which("rsync")
    if rsync:
        print(f"[{label}] rsync {src} -> {dst} ...", flush=True)
        subprocess.run(
            [rsync, "-a", "--delete", f"{src}/", f"{dst}/"],
            check=True,
        )
    else:
        print(f"[{label}] copytree {src} -> {dst} (no rsync) ...", flush=True)
        shutil.copytree(src, dst, dirs_exist_ok=True)
    print(f"[{label}] done | {tree_size_mb(dst):.0f} MiB at {dst}", flush=True)


def persist_prepared(local_root: Path, drive_dir: Path, fingerprint: dict) -> None:
    write_marker(local_root, fingerprint)
    print(f"[dataset-cache] persisting prepared tree ({tree_size_mb(local_root):.0f} MiB)...", flush=True)
    sync_tree(local_root, drive_dir, label="dataset-cache")
    print("[dataset-cache] prepared dataset safe in Drive; reruns will restore it.", flush=True)


def restore_prepared(drive_dir: Path, local_root: Path) -> None:
    if local_root.exists():
        shutil.rmtree(local_root)
    local_root.mkdir(parents=True, exist_ok=True)
    sync_tree(drive_dir, local_root, label="dataset-cache")


def download_with_progress(url: str, destination: Path, *, label: str) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "resistor-model-colab"})
    hasher = hashlib.sha256()
    downloaded = 0
    print(f"[{label}] downloading...", flush=True)
    with urllib.request.urlopen(request) as response, destination.open("wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        last_report = 0
        while True:
            chunk = response.read(8 * 1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            hasher.update(chunk)
            downloaded += len(chunk)
            if downloaded - last_report >= 32 * 1024 * 1024 or (total and downloaded >= total):
                progress = f" ({100 * downloaded / total:.0f}%)" if total else ""
                print(f"[{label}] downloaded {downloaded / 1024 / 1024:.1f} MiB{progress}", flush=True)
                last_report = downloaded
    print(f"[{label}] done | {downloaded / 1024 / 1024:.1f} MiB", flush=True)


def fetch_cached_archive(
    *,
    local: Path,
    drive_copy: Path,
    url: str,
    sha256: str,
    label: str,
    verbose: Callable[[str], None] | None = None,
) -> Path:
    """Return a verified local archive, preferring cache (local, then Drive)."""
    log = verbose or (lambda msg: print(msg, flush=True))
    for candidate, origin in ((local, "local"), (drive_copy, "Drive")):
        if candidate.is_file():
            log(f"[{label}] checking {origin} copy...")
            if sha_of(candidate) == sha256:
                if candidate != local:
                    log(f"[{label}] restoring {candidate.stat().st_size / 1024 / 1024:.1f} MiB from Drive...")
                    local.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(candidate, local)
                else:
                    log(f"[{label}] verified archive cached locally")
                return local
            log(f"[{label}] stale {origin} copy; discarding")
            candidate.unlink()
    download_with_progress(url, local, label=label)
    if sha_of(local) != sha256:
        local.unlink(missing_ok=True)
        raise RuntimeError(f"{label} archive checksum mismatch")
    drive_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(local, drive_copy)
    log(f"[{label}] checksum OK; backed up to Drive")
    return local
