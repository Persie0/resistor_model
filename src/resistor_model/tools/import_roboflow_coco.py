from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import re
from typing import Iterable

from resistor_model.constants import COLOR_TO_INDEX
from resistor_model.tools.import_roboflow_yolo import group_bands_by_body


_BODY_NAMES = {"resistor", "resistor symbol", "resistor-symbol", "resistor_symbol"}
_IGNORED_NAMES = {"r", "resistors", "object-resistors", "object resistor", "object resistors"}
_ROBOFLOW_SUFFIX = re.compile(r"\.rf\.[0-9a-f]+$", re.IGNORECASE)
_FORMAT_SUFFIX = re.compile(r"_(?:jpg|jpeg|png|webp|bmp)$", re.IGNORECASE)


def normalize_category_name(name: str) -> str | None:
    """Normalize the category names used by the three public r1 datasets."""
    value = " ".join(str(name).strip().lower().split())
    if value == "grey":
        return "gray"
    if value in _BODY_NAMES:
        return "resistor"
    if value in _IGNORED_NAMES:
        return None
    return value


def source_group_id(name: str | Path) -> str:
    """Return a stable original-image identity across Roboflow exports."""
    stem = Path(str(name)).stem
    stem = _ROBOFLOW_SUFFIX.sub("", stem)
    stem = _FORMAT_SUFFIX.sub("", stem)
    return stem.strip("_- ").casefold()


def _bbox_xyxy(annotation: dict) -> list[float]:
    raw = annotation.get("bbox")
    if not isinstance(raw, list) or len(raw) != 4:
        raise ValueError(f"invalid COCO bbox: {raw!r}")
    x, y, w, h = (float(v) for v in raw)
    if w <= 0 or h <= 0:
        raise ValueError(f"degenerate COCO bbox: {raw!r}")
    return [x, y, x + w, y + h]


def _polygon(annotation: dict) -> list[list[float]] | None:
    segmentation = annotation.get("segmentation")
    if not isinstance(segmentation, list) or not segmentation:
        return None
    candidates = [segment for segment in segmentation if isinstance(segment, list) and len(segment) >= 6 and len(segment) % 2 == 0]
    if not candidates:
        return None
    coords = max(candidates, key=len)
    points = [[float(coords[i]), float(coords[i + 1])] for i in range(0, len(coords), 2)]
    if len(points) > 3 and points[0] == points[-1]:
        points = points[:-1]
    return points if len(points) >= 3 else None


def _body_object(annotation: dict) -> dict:
    polygon = _polygon(annotation)
    return {
        "bbox": _bbox_xyxy(annotation),
        "polygon": polygon,
        "annotation_type": "polygon" if polygon is not None else "box",
    }


def _annotation_files(root: Path) -> Iterable[tuple[str, Path]]:
    seen: set[Path] = set()
    for split in ("train", "valid", "val", "test"):
        path = root / split / "_annotations.coco.json"
        if path.is_file():
            seen.add(path.resolve())
            yield split, path
    for path in sorted(root.rglob("_annotations.coco.json")):
        resolved = path.resolve()
        if resolved in seen:
            continue
        yield path.parent.name, path


def _manifest_image_path(image_path: Path, manifest_parent: Path) -> str:
    image_path = image_path.resolve()
    manifest_parent = manifest_parent.resolve()
    try:
        return image_path.relative_to(manifest_parent).as_posix()
    except ValueError:
        return os.fspath(image_path)


def _dataset_slug(root: Path, index: int) -> str:
    clean = re.sub(r"[^a-z0-9]+", "-", root.name.casefold()).strip("-") or "dataset"
    return f"{index}-{clean}"


def import_datasets(
    dataset_roots: Iterable[str | Path],
    output: str | Path,
    *,
    min_bands: int = 3,
    max_bands: int = 6,
    report_path: str | Path | None = None,
) -> dict:
    roots = [Path(root).resolve() for root in dataset_roots]
    if not roots:
        raise ValueError("at least one COCO dataset root is required")
    if min_bands < 1 or max_bands < min_bands:
        raise ValueError("invalid min/max band limits")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    stats: Counter[str] = Counter()
    by_band_count: Counter[int] = Counter()
    rows: list[dict] = []
    source_datasets: dict[str, set[str]] = defaultdict(set)
    dataset_reports: dict[str, dict] = {}

    for dataset_index, root in enumerate(roots):
        slug = _dataset_slug(root, dataset_index)
        dataset_stats: Counter[str] = Counter()
        annotation_files = list(_annotation_files(root))
        if not annotation_files:
            raise FileNotFoundError(f"no _annotations.coco.json under {root}")

        for original_split, annotation_path in annotation_files:
            payload = json.loads(annotation_path.read_text(encoding="utf-8"))
            categories = {
                int(category["id"]): normalize_category_name(category["name"])
                for category in payload.get("categories", [])
            }
            unknown = sorted({name for name in categories.values() if name is not None and name != "resistor" and name not in COLOR_TO_INDEX})
            if unknown:
                raise ValueError(f"{annotation_path}: unsupported categories: {unknown}")

            annotations_by_image: dict[int, list[dict]] = defaultdict(list)
            for annotation in payload.get("annotations", []):
                annotations_by_image[int(annotation["image_id"])].append(annotation)

            for image in payload.get("images", []):
                stats["images_seen"] += 1
                dataset_stats["images_seen"] += 1
                image_id = int(image["id"])
                image_path = annotation_path.parent / str(image["file_name"])
                if not image_path.is_file():
                    stats["images_missing_file"] += 1
                    dataset_stats["images_missing_file"] += 1
                    continue

                bodies: list[dict] = []
                bands: list[dict] = []
                for annotation in annotations_by_image.get(image_id, []):
                    name = categories.get(int(annotation.get("category_id", -1)))
                    if name is None:
                        continue
                    if name == "resistor":
                        bodies.append(_body_object(annotation))
                        continue
                    bands.append({"color": name, "bbox": _bbox_xyxy(annotation)})

                stats["body_annotations"] += len(bodies)
                dataset_stats["body_annotations"] += len(bodies)
                stats["body_polygon_annotations"] += sum(body["annotation_type"] == "polygon" for body in bodies)
                stats["body_box_annotations"] += sum(body["annotation_type"] == "box" for body in bodies)
                stats["band_annotations"] += len(bands)
                dataset_stats["band_annotations"] += len(bands)
                if not bodies:
                    stats["images_without_body"] += 1
                    dataset_stats["images_without_body"] += 1
                elif len(bodies) > 1:
                    stats["images_with_multiple_bodies"] += 1

                groups, unassigned = group_bands_by_body(
                    bodies, bands, min_bands=min_bands, max_bands=max_bands
                )
                stats["unassigned_bands"] += unassigned
                stats["skipped_body_groups"] += max(0, len(bodies) - len(groups))
                if not groups:
                    stats["images_without_usable_resistor"] += 1
                    dataset_stats["images_without_usable_resistor"] += 1
                    continue

                original_name = image.get("extra", {}).get("name") if isinstance(image.get("extra"), dict) else None
                source = source_group_id(original_name or image["file_name"])
                source_datasets[source].add(slug)
                resistors: list[dict] = []
                for group_index, (body, group_bands) in enumerate(groups):
                    body_bbox = body["bbox"] if isinstance(body, dict) else None
                    polygon = body.get("polygon") if isinstance(body, dict) else None
                    annotation_type = body.get("annotation_type") if isinstance(body, dict) else None
                    ordered_bands = sorted(
                        group_bands,
                        key=lambda band: ((band["bbox"][0] + band["bbox"][2]) * 0.5, (band["bbox"][1] + band["bbox"][3]) * 0.5),
                    )
                    resistors.append({
                        "id": f"{slug}:{source}:{image_id}:{group_index}",
                        "bbox": body_bbox,
                        "polygon": polygon,
                        "body_annotation_type": annotation_type,
                        "bands": ordered_bands,
                    })
                    stats["resistors_accepted"] += 1
                    dataset_stats["resistors_accepted"] += 1
                    by_band_count[len(group_bands)] += 1

                rows.append({
                    "image": _manifest_image_path(image_path, output.parent),
                    "session_id": source,
                    "camera_id": slug,
                    "split": None,
                    "source_split": original_split,
                    "resistors": resistors,
                })
                stats["images_accepted"] += 1
                dataset_stats["images_accepted"] += 1

        dataset_reports[slug] = dict(sorted(dataset_stats.items()))

    stats["source_groups"] = len(source_datasets)
    stats["cross_dataset_source_groups"] = sum(len(dataset_names) > 1 for dataset_names in source_datasets.values())
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")

    report = {
        "dataset_roots": [os.fspath(root) for root in roots],
        "datasets": dataset_reports,
        "counts": dict(sorted(stats.items())),
        "resistors_by_band_count": {str(k): v for k, v in sorted(by_band_count.items())},
    }
    if report_path is not None:
        report_file = Path(report_path)
        report_file.parent.mkdir(parents=True, exist_ok=True)
        report_file.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine Roboflow COCO resistor datasets into the canonical manifest")
    parser.add_argument("--root", type=Path, action="append", required=True, help="Extracted Roboflow COCO root; repeat for multiple datasets")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--min-bands", type=int, default=3)
    parser.add_argument("--max-bands", type=int, default=6)
    args = parser.parse_args()
    report = import_datasets(
        args.root,
        args.output,
        min_bands=args.min_bands,
        max_bands=args.max_bands,
        report_path=args.report,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
