from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
from typing import Iterable

import cv2
import yaml

from resistor_model.constants import COLOR_TO_INDEX


_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_CLASS_ALIASES = {
    "grey": "gray",
    "resistor-symbol": "resistor symbol",
    "resistor_symbol": "resistor symbol",
}
_ROBOFLOW_SUFFIX = re.compile(r"\.rf\.[0-9a-f]+$", re.IGNORECASE)
_CAPTURE_STAMP = re.compile(
    r"^(?P<prefix>.+?)_(?P<timestamp>20\d{12})_(?P<tail>\d+)(?:_(?:jpg|jpeg|png|webp|bmp))?$",
    re.IGNORECASE,
)
_FORMAT_SUFFIX = re.compile(r"_(?:jpg|jpeg|png|webp|bmp)$", re.IGNORECASE)
_NUMBERED_CAPTURE = re.compile(r"^(?P<prefix>.+?)[_-]+(?P<number>\d+)[_-]*$")
_GENERIC_CAPTURE_PREFIXES = {"batch", "error"}
_GENERIC_NUMBERED_PREFIXES = {"image", "img", "download", "r", "rr", "rrr", "batch", "error"}


def canonical_class_name(name: str) -> str:
    value = " ".join(str(name).strip().lower().split())
    return _CLASS_ALIASES.get(value, value)


def roboflow_source_id(image_path: str | Path) -> str:
    stem = Path(image_path).stem
    return _ROBOFLOW_SUFFIX.sub("", stem)


def roboflow_capture_group_id(image_path: str | Path) -> str:
    """Return a conservative repeated-capture group for split isolation.

    rres.v4 contains both timestamped series such as
    ``1k-5-_20251215140233_4900_jpg`` and numbered series such as
    ``100R_1-4W_-60-_jpg``. Captures sharing those meaningful prefixes are
    kept in one split because they may show the same physical component or
    acquisition setup. Generic recorder/file names remain independent.
    """
    source_id = roboflow_source_id(image_path)

    stamped = _CAPTURE_STAMP.match(source_id)
    if stamped is not None:
        prefix = stamped.group("prefix").rstrip("_- ")
        if prefix and prefix.lower() not in _GENERIC_CAPTURE_PREFIXES:
            return prefix
        return source_id

    # Roboflow commonly embeds the original extension in the stem. Remove it
    # before detecting a trailing capture counter.
    cleaned = _FORMAT_SUFFIX.sub("", source_id).rstrip("_- ")
    numbered = _NUMBERED_CAPTURE.match(cleaned)
    if numbered is not None:
        prefix = numbered.group("prefix").rstrip("_- ")
        if prefix and prefix.lower() not in _GENERIC_NUMBERED_PREFIXES:
            return prefix
    return source_id


def _bbox_from_yolo_fields(fields: list[float], width: int, height: int) -> list[float]:
    if len(fields) == 4:
        xc, yc, bw, bh = fields
        return [
            (xc - bw / 2.0) * width,
            (yc - bh / 2.0) * height,
            (xc + bw / 2.0) * width,
            (yc + bh / 2.0) * height,
        ]
    if len(fields) >= 6 and len(fields) % 2 == 0:
        xs = fields[0::2]
        ys = fields[1::2]
        return [min(xs) * width, min(ys) * height, max(xs) * width, max(ys) * height]
    raise ValueError(f"expected YOLO box (4 values) or polygon (even >=6 values), got {len(fields)}")


def parse_yolo_label_line(line: str, class_names: list[str], width: int, height: int) -> dict:
    parts = line.split()
    if len(parts) < 5:
        raise ValueError(f"invalid YOLO annotation with {len(parts)} fields")
    class_id = int(float(parts[0]))
    if class_id < 0 or class_id >= len(class_names):
        raise ValueError(f"class id {class_id} outside class list of length {len(class_names)}")
    fields = [float(v) for v in parts[1:]]
    bbox = _bbox_from_yolo_fields(fields, width, height)
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"degenerate annotation bbox: {bbox}")
    if len(fields) == 4:
        polygon = None
        annotation_type = "box"
    else:
        polygon = [[fields[i] * width, fields[i + 1] * height] for i in range(0, len(fields), 2)]
        if len(polygon) > 3 and polygon[0] == polygon[-1]:
            polygon = polygon[:-1]
        annotation_type = "polygon"
    return {
        "name": canonical_class_name(class_names[class_id]),
        "bbox": bbox,
        "polygon": polygon,
        "annotation_type": annotation_type,
    }


def _body_bbox(body: dict | list[float]) -> list[float]:
    return body["bbox"] if isinstance(body, dict) else body


def _intersection_fraction(inner: list[float], outer: list[float]) -> float:
    x1 = max(inner[0], outer[0])
    y1 = max(inner[1], outer[1])
    x2 = min(inner[2], outer[2])
    y2 = min(inner[3], outer[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area = max((inner[2] - inner[0]) * (inner[3] - inner[1]), 1e-9)
    return inter / area


def _expanded_contains(box: list[float], x: float, y: float, margin: float = 0.18) -> bool:
    w = box[2] - box[0]
    h = box[3] - box[1]
    return box[0] - margin * w <= x <= box[2] + margin * w and box[1] - margin * h <= y <= box[3] + margin * h


def _normalized_distance_to_box_center(inner: list[float], outer: list[float]) -> float:
    ix = (inner[0] + inner[2]) * 0.5
    iy = (inner[1] + inner[3]) * 0.5
    ox = (outer[0] + outer[2]) * 0.5
    oy = (outer[1] + outer[3]) * 0.5
    diag = max(((outer[2] - outer[0]) ** 2 + (outer[3] - outer[1]) ** 2) ** 0.5, 1e-6)
    return (((ix - ox) ** 2 + (iy - oy) ** 2) ** 0.5) / diag


def _association_score(band_bbox: list[float], body: dict | list[float]) -> tuple[bool, float]:
    body_bbox = _body_bbox(body)
    overlap = _intersection_fraction(band_bbox, body_bbox)
    cx = (band_bbox[0] + band_bbox[2]) * 0.5
    cy = (band_bbox[1] + band_bbox[3]) * 0.5
    inside = _expanded_contains(body_bbox, cx, cy)
    distance = _normalized_distance_to_box_center(band_bbox, body_bbox)
    eligible = overlap >= 0.05 or inside or distance <= 0.18
    score = 3.0 * overlap + (1.0 if inside else 0.0) - distance
    return eligible, score


def group_bands_by_body(
    bodies: list[dict | list[float]], bands: list[dict], *, min_bands: int = 3, max_bands: int = 6
) -> tuple[list[tuple[dict | list[float] | None, list[dict]]], int]:
    """Assign bands to whole-resistor annotations.

    Body metadata is preserved in the returned groups. If no whole-resistor
    annotation exists, one unambiguous 3-6-band image is accepted as fallback.
    """
    if not bodies:
        if min_bands <= len(bands) <= max_bands:
            return [(None, bands)], 0
        return [], len(bands)

    ordered_bodies = sorted(bodies, key=lambda b: ((_body_bbox(b)[1] + _body_bbox(b)[3]) * 0.5, (_body_bbox(b)[0] + _body_bbox(b)[2]) * 0.5))
    assigned: list[list[dict]] = [[] for _ in ordered_bodies]
    unassigned = 0
    for band in bands:
        candidates: list[tuple[float, int]] = []
        for idx, body in enumerate(ordered_bodies):
            eligible, score = _association_score(band["bbox"], body)
            if eligible:
                candidates.append((score, idx))
        if not candidates:
            unassigned += 1
            continue
        _, best_idx = max(candidates)
        assigned[best_idx].append(band)

    accepted: list[tuple[dict | list[float] | None, list[dict]]] = []
    for body, body_bands in zip(ordered_bodies, assigned):
        if min_bands <= len(body_bands) <= max_bands:
            accepted.append((body, body_bands))
    return accepted, unassigned


def _load_class_names(dataset_root: Path) -> list[str]:
    yaml_path = dataset_root / "data.yaml"
    if not yaml_path.exists():
        candidates = list(dataset_root.glob("**/data.yaml")) + list(dataset_root.glob("**/dataset.yaml"))
        if not candidates:
            raise FileNotFoundError(f"no data.yaml/dataset.yaml under {dataset_root}")
        yaml_path = candidates[0]
    payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    names = payload.get("names")
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=lambda x: int(x))]
    if not isinstance(names, list) or not names:
        raise ValueError(f"{yaml_path}: expected non-empty names list/dict")
    return [canonical_class_name(str(x)) for x in names]


def _iter_split_images(dataset_root: Path) -> Iterable[tuple[str, Path, Path]]:
    for split in ("train", "valid", "val", "test"):
        base = dataset_root / split
        image_dir = base / "images"
        label_dir = base / "labels"
        if not image_dir.exists():
            continue
        for image_path in sorted(p for p in image_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTENSIONS):
            yield split, image_path, label_dir / f"{image_path.stem}.txt"


def import_dataset(
    dataset_root: Path,
    output: Path,
    *,
    body_class: str = "resistor symbol",
    min_bands: int = 3,
    max_bands: int = 6,
) -> dict:
    dataset_root = dataset_root.resolve()
    body_class = canonical_class_name(body_class)
    class_names = _load_class_names(dataset_root)
    unknown_classes = sorted({x for x in class_names if x != body_class and x not in COLOR_TO_INDEX})
    if unknown_classes:
        raise ValueError(f"unsupported non-body classes: {unknown_classes}")

    stats: Counter[str] = Counter()
    per_band_count: Counter[int] = Counter()
    source_occurrences: Counter[str] = Counter()
    capture_occurrences: Counter[str] = Counter()
    rows: list[dict] = []

    for original_split, image_path, label_path in _iter_split_images(dataset_root):
        stats["images_seen"] += 1
        if not label_path.exists():
            stats["images_missing_labels"] += 1
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            stats["images_unreadable"] += 1
            continue
        height, width = image.shape[:2]
        objects = []
        for line_no, raw in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
            if not raw.strip():
                continue
            try:
                objects.append(parse_yolo_label_line(raw, class_names, width, height))
            except Exception as exc:
                raise ValueError(f"{label_path}:{line_no}: {exc}") from exc

        bodies = [obj for obj in objects if obj["name"] == body_class]
        bands = [{"color": obj["name"], "bbox": obj["bbox"]} for obj in objects if obj["name"] != body_class]
        stats["body_annotations"] += len(bodies)
        stats["body_box_annotations"] += sum(obj["annotation_type"] == "box" for obj in bodies)
        stats["body_polygon_annotations"] += sum(obj["annotation_type"] == "polygon" for obj in bodies)
        stats["band_annotations"] += len(bands)
        if not bodies:
            stats["images_without_body"] += 1
        elif len(bodies) > 1:
            stats["images_with_multiple_bodies"] += 1

        groups, unassigned = group_bands_by_body(bodies, bands, min_bands=min_bands, max_bands=max_bands)
        stats["unassigned_bands"] += unassigned
        stats["skipped_body_groups"] += max(0, len(bodies) - len(groups))
        if not groups:
            stats["images_without_usable_resistor"] += 1
            continue

        source_id = roboflow_source_id(image_path)
        capture_group = roboflow_capture_group_id(image_path)
        source_occurrences[source_id] += 1
        capture_occurrences[capture_group] += 1
        resistors = []
        for body, group_bands in groups:
            body_bbox = _body_bbox(body) if body is not None else None
            polygon = body.get("polygon") if isinstance(body, dict) else None
            annotation_type = body.get("annotation_type") if isinstance(body, dict) else None
            resistors.append({
                "id": source_id,
                "bbox": body_bbox,
                "polygon": polygon,
                "body_annotation_type": annotation_type,
                "bands": group_bands,
            })
            per_band_count[len(group_bands)] += 1
            stats["resistors_accepted"] += 1

        rows.append({
            "image": image_path.relative_to(dataset_root).as_posix(),
            "session_id": capture_group,
            "camera_id": None,
            "split": None,
            "source_split": original_split,
            "resistors": resistors,
        })
        stats["images_accepted"] += 1

    stats["source_groups"] = len(source_occurrences)
    stats["source_groups_with_multiple_exports"] = sum(v > 1 for v in source_occurrences.values())
    stats["capture_groups"] = len(capture_occurrences)
    stats["capture_groups_with_multiple_images"] = sum(v > 1 for v in capture_occurrences.values())
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, separators=(",", ":")) + "\n")

    return {
        "class_names": class_names,
        "body_class": body_class,
        "counts": dict(sorted(stats.items())),
        "resistors_by_band_count": {str(k): v for k, v in sorted(per_band_count.items())},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Import a Roboflow YOLO resistor dataset into the canonical JSONL manifest")
    ap.add_argument("--root", type=Path, required=True, help="Extracted Roboflow dataset root containing data.yaml")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--body-class", default="resistor symbol")
    ap.add_argument("--min-bands", type=int, default=3)
    ap.add_argument("--max-bands", type=int, default=6)
    args = ap.parse_args()
    report = import_dataset(args.root, args.output, body_class=args.body_class, min_bands=args.min_bands, max_bands=args.max_bands)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
