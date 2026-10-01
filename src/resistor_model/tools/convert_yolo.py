from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from resistor_model.constants import BAND_COLORS


def convert_yolo_pair(image_path: Path, label_path: Path, class_names: list[str]) -> dict:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"cannot read image {image_path}")
    h, w = image.shape[:2]
    bands = []
    if label_path.exists():
        for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f"{label_path}:{line_no}: expected 5 YOLO fields")
            cls, xc, yc, bw, bh = map(float, parts)
            ci = int(cls)
            if ci < 0 or ci >= len(class_names):
                raise ValueError(f"{label_path}:{line_no}: class {ci} outside class list")
            x1 = (xc - bw / 2) * w
            y1 = (yc - bh / 2) * h
            x2 = (xc + bw / 2) * w
            y2 = (yc + bh / 2) * h
            bands.append({"color": class_names[ci], "bbox": [x1, y1, x2, y2]})
    if not bands:
        raise ValueError(f"no band annotations in {label_path}")
    return {"image": image_path.name, "resistors": [{"id": image_path.stem, "bands": bands}]}


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert YOLO band boxes to resistor JSONL manifest")
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--classes", default=",".join(BAND_COLORS), help="Comma-separated class names in YOLO class-id order")
    args = ap.parse_args()
    class_names = [x.strip().lower() for x in args.classes.split(",") if x.strip()]
    suffixes = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    images = sorted(p for p in args.images.iterdir() if p.suffix.lower() in suffixes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for image_path in images:
            label_path = args.labels / f"{image_path.stem}.txt"
            row = convert_yolo_pair(image_path, label_path, class_names)
            fh.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(f"wrote {len(images)} images to {args.output}")


if __name__ == "__main__":
    main()
