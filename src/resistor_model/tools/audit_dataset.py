from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import cv2

from resistor_model.data.schema import load_manifest


def _sha1(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def audit(manifest: Path, image_root: Path) -> dict:
    rows = load_manifest(manifest)
    colors = Counter()
    band_counts = Counter()
    issues: list[str] = []
    hashes: dict[str, list[str]] = {}
    resistor_ids = Counter()
    image_count = 0
    resistor_count = 0
    for row in rows:
        image_count += 1
        path = image_root / row.image
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            issues.append(f"missing/unreadable image: {path}")
            continue
        h, w = image.shape[:2]
        digest = _sha1(path)
        hashes.setdefault(digest, []).append(row.image)
        for resistor in row.resistors:
            resistor_count += 1
            resistor_ids[resistor.id] += 1
            band_counts[len(resistor.bands)] += 1
            for band in resistor.bands:
                colors[band.color] += 1
                x1, y1, x2, y2 = band.bbox
                if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                    issues.append(f"bbox outside image: {row.image} {resistor.id} {band.color} {band.bbox}")
    duplicates = [paths for paths in hashes.values() if len(paths) > 1]
    return {
        "images": image_count,
        "resistors": resistor_count,
        "unique_resistor_ids": len(resistor_ids),
        "colors": dict(sorted(colors.items())),
        "band_count_distribution": {str(k): v for k, v in sorted(band_counts.items())},
        "exact_duplicate_groups": duplicates,
        "issues": issues,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit resistor annotations and images")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--image-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    result = audit(args.manifest, args.image_root)
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
