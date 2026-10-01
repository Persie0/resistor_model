from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

from resistor_model.constants import COLOR_TO_INDEX


@dataclass(frozen=True)
class BandAnnotation:
    color: str
    bbox: tuple[float, float, float, float]

    @classmethod
    def from_dict(cls, data: dict) -> "BandAnnotation":
        color = str(data["color"]).lower().strip()
        if color not in COLOR_TO_INDEX:
            raise ValueError(f"unknown band color: {color}")
        bbox = tuple(float(v) for v in data["bbox"])
        if len(bbox) != 4 or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            raise ValueError(f"invalid band bbox: {bbox}")
        return cls(color=color, bbox=bbox)


@dataclass(frozen=True)
class ResistorAnnotation:
    id: str
    bands: tuple[BandAnnotation, ...]
    bbox: tuple[float, float, float, float] | None = None
    polygon: tuple[tuple[float, float], ...] | None = None
    body_annotation_type: str | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "ResistorAnnotation":
        bands = tuple(BandAnnotation.from_dict(x) for x in data.get("bands", []))
        if not bands:
            raise ValueError("resistor must contain at least one band")
        bbox = data.get("bbox")
        if bbox is not None:
            bbox = tuple(float(v) for v in bbox)
            if len(bbox) != 4 or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                raise ValueError(f"invalid resistor bbox: {bbox}")
        polygon_raw = data.get("polygon")
        polygon = None
        if polygon_raw is not None:
            polygon = tuple(tuple(float(v) for v in point) for point in polygon_raw)
            if len(polygon) < 3 or any(len(point) != 2 for point in polygon):
                raise ValueError(f"invalid resistor polygon: {polygon_raw}")
        annotation_type = data.get("body_annotation_type")
        if annotation_type is not None:
            annotation_type = str(annotation_type)
            if annotation_type not in {"box", "polygon"}:
                raise ValueError(f"invalid body_annotation_type: {annotation_type}")
        return cls(
            id=str(data["id"]), bands=bands, bbox=bbox,
            polygon=polygon, body_annotation_type=annotation_type,
        )


@dataclass(frozen=True)
class ImageAnnotation:
    image: str
    resistors: tuple[ResistorAnnotation, ...]
    session_id: str | None = None
    camera_id: str | None = None
    split: str | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "ImageAnnotation":
        resistors = tuple(ResistorAnnotation.from_dict(x) for x in data.get("resistors", []))
        if not resistors:
            raise ValueError("image annotation must contain at least one resistor")
        return cls(
            image=str(data["image"]),
            resistors=resistors,
            session_id=None if data.get("session_id") is None else str(data["session_id"]),
            camera_id=None if data.get("camera_id") is None else str(data["camera_id"]),
            split=None if data.get("split") is None else str(data["split"]),
        )


def load_manifest(path: str | Path) -> list[ImageAnnotation]:
    path = Path(path)
    rows: list[ImageAnnotation] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(ImageAnnotation.from_dict(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
    return rows


def write_manifest(rows: Iterable[ImageAnnotation], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            payload = {
                "image": row.image,
                "session_id": row.session_id,
                "camera_id": row.camera_id,
                "split": row.split,
                "resistors": [
                    {
                        "id": r.id,
                        "bbox": list(r.bbox) if r.bbox else None,
                        "polygon": [list(point) for point in r.polygon] if r.polygon else None,
                        "body_annotation_type": r.body_annotation_type,
                        "bands": [{"color": b.color, "bbox": list(b.bbox)} for b in r.bands],
                    }
                    for r in row.resistors
                ],
            }
            fh.write(json.dumps(payload, separators=(",", ":")) + "\n")
