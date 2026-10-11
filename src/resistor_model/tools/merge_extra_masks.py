"""Merge an extra SAM-mask segmentation archive (same v4 layout) into a dataset root.

Used for the real blue-resistor release (``blue_resistors_sam3.zip``): its
``train/valid/test`` image+``masks_semantic`` pairs are copied into the v4
tree so every trainer (PyTorch LR-ASPP, PaddleSeg PP-LiteSeg, ...) and the
hue-bucketed eval pick them up with zero loader changes.

* SHA-256 verified, overlays/ skipped (not needed for training).
* All copied stems get ``--prefix`` (default ``blue``) so collisions with v4
  names are impossible and provenance stays visible in file lists.
* Masks must be PNGs whose dimensions match their image; mismatches are
  skipped with a count (same rule as the training loader).
* Idempotent: existing destinations are skipped, so Colab retries are safe.

Dependencies: stdlib + Pillow only (no torch/cv2, safe in any Colab env).
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import zipfile

from PIL import Image

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def sha256_of(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _zip_split_prefix(names: list[str]) -> str:
    """Top-level wrapper dir inside the archive, or '' if splits are at root."""
    for split in ("train", "valid", "test"):
        for name in names:
            if name == f"{split}/masks_semantic/" or f"/{split}/masks_semantic/" in f"/{name}":
                head, _, _ = name.partition(f"{split}/masks_semantic/")
                return head
    raise RuntimeError("archive has no train/valid/test masks_semantic/ layout")


def merge_archive(
    dataset_root: Path,
    archive: Path,
    *,
    splits: list[str],
    prefix: str = "blue",
    expected_sha256: str | None = None,
    overwrite: bool = False,
    progress_every: int = 100,
) -> dict[str, dict[str, int]]:
    if expected_sha256 and sha256_of(archive) != expected_sha256:
        raise RuntimeError(f"checksum mismatch for {archive}")
    report: dict[str, dict[str, int]] = {}
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
        top = _zip_split_prefix(names)
        for split in splits:
            stats = {"copied": 0, "skipped_existing": 0, "skipped_mismatch": 0, "skipped_no_image": 0}
            mask_prefix = f"{top}{split}/masks_semantic/"
            img_prefix = f"{top}{split}/images/"
            masks = sorted(n for n in names if n.startswith(mask_prefix) and n.lower().endswith(".png"))
            by_stem: dict[str, str] = {}
            for n in names:
                if n.startswith(img_prefix) and Path(n).suffix.lower() in IMAGE_EXTENSIONS:
                    by_stem.setdefault(Path(n).stem, n)
            total = len(masks)
            for done, mask_name in enumerate(masks, start=1):
                if progress_every > 0 and (done % progress_every == 0 or done == total):
                    print(f"[merge] {split}: {done}/{total} pairs...", flush=True)
                stem = Path(mask_name).stem
                img_name = by_stem.get(stem)
                if img_name is None:
                    stats["skipped_no_image"] += 1
                    continue
                with zf.open(mask_name) as handle:
                    mask_img = Image.open(handle)
                    mask_img.load()
                    mask_size = mask_img.size
                with zf.open(img_name) as handle:
                    img_img = Image.open(handle)
                    img_img.load()
                    img_size = img_img.size
                if mask_size != img_size:
                    stats["skipped_mismatch"] += 1
                    continue
                new_stem = f"{prefix}_{stem}"
                img_ext = Path(img_name).suffix.lower()
                dest_img = dataset_root / split / "images" / f"{new_stem}{img_ext}"
                dest_mask = dataset_root / split / "masks_semantic" / f"{new_stem}.png"
                if dest_img.is_file() and dest_mask.is_file() and not overwrite:
                    stats["skipped_existing"] += 1
                    continue
                dest_img.parent.mkdir(parents=True, exist_ok=True)
                dest_mask.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(img_name) as handle:
                    dest_img.write_bytes(handle.read())
                mask_img.save(dest_mask)
                stats["copied"] += 1
            report[split] = stats
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge an extra SAM-mask archive into a v4 dataset root")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--extra-archive", type=Path, required=True)
    parser.add_argument("--extra-sha256", type=str, default=None)
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    parser.add_argument("--prefix", type=str, default="blue")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    report = merge_archive(
        args.dataset_root, args.extra_archive, splits=list(args.splits),
        prefix=str(args.prefix), expected_sha256=args.extra_sha256,
        overwrite=bool(args.overwrite),
    )
    for split, stats in report.items():
        print(f"[merge] {split}: " + ", ".join(f"{k}={v}" for k, v in stats.items()), flush=True)


if __name__ == "__main__":
    main()
