from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
import json
from pathlib import Path
import random

import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.transforms import ColorJitter
from torchvision.transforms import functional as TF
from PIL import Image


_SCALAR_EVENTS = {"string", "number", "boolean", "null"}


def _compact_coco_payload(coco: dict) -> tuple[dict[int, str], list[dict], list[dict]]:
    """Discard segmentation/RLE payloads and keep only fields needed for bbox detection."""
    categories = {
        int(item["id"]): str(item.get("name", "")).strip().lower()
        for item in coco.get("categories", [])
    }
    images = [
        {
            "id": int(item["id"]),
            "file_name": str(item["file_name"]),
            "width": int(item.get("width", 0) or 0),
            "height": int(item.get("height", 0) or 0),
        }
        for item in coco.get("images", [])
    ]
    annotations = []
    for item in coco.get("annotations", []):
        bbox = item.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        annotations.append(
            {
                "image_id": int(item["image_id"]),
                "category_id": int(item["category_id"]),
                "bbox": [float(value) for value in bbox],
                "iscrowd": int(item.get("iscrowd", 0)),
            }
        )
    return categories, images, annotations


def load_coco_bbox_metadata(
    path: str | Path,
    *,
    progress_label: str | None = None,
) -> tuple[dict[int, str], list[dict], list[dict]]:
    """Load COCO metadata without retaining masks/segmentations.

    With the detection optional dependency installed this streams the JSON via ijson,
    which avoids materializing the large SAM RLE segmentation payloads. A small-file
    stdlib fallback remains available for minimal/dev installations.
    """
    annotation_path = Path(path)
    size = annotation_path.stat().st_size

    try:
        import ijson
    except ImportError:
        if progress_label:
            print(
                f"[coco] {progress_label}: ijson unavailable; using stdlib JSON "
                f"for {size / 1024 / 1024:.1f} MiB",
                flush=True,
            )
        coco = json.loads(annotation_path.read_text(encoding="utf-8"))
        result = _compact_coco_payload(coco)
        if progress_label:
            print(
                f"[coco] {progress_label}: ready | images={len(result[1])} | "
                f"boxes={len(result[2])}",
                flush=True,
            )
        return result

    categories: dict[int, str] = {}
    images: list[dict] = []
    annotations: list[dict] = []

    current_image: dict | None = None
    current_category: dict | None = None
    current_annotation: dict | None = None
    current_bbox: list[float] | None = None

    report_step = max(8 * 1024 * 1024, size // 20 if size else 8 * 1024 * 1024)
    next_report = report_step
    event_count = 0

    if progress_label:
        print(
            f"[coco] {progress_label}: streaming bbox metadata from "
            f"{annotation_path} ({size / 1024 / 1024:.1f} MiB)",
            flush=True,
        )

    with annotation_path.open("rb") as handle:
        for prefix, event, value in ijson.parse(handle):
            event_count += 1

            if prefix == "images.item" and event == "start_map":
                current_image = {}
            elif prefix == "images.item" and event == "end_map":
                if current_image is not None and "id" in current_image and "file_name" in current_image:
                    images.append(current_image)
                current_image = None
            elif current_image is not None and event in _SCALAR_EVENTS:
                if prefix == "images.item.id":
                    current_image["id"] = int(value)
                elif prefix == "images.item.file_name":
                    current_image["file_name"] = str(value)
                elif prefix == "images.item.width":
                    current_image["width"] = int(value or 0)
                elif prefix == "images.item.height":
                    current_image["height"] = int(value or 0)

            if prefix == "categories.item" and event == "start_map":
                current_category = {}
            elif prefix == "categories.item" and event == "end_map":
                if current_category is not None and "id" in current_category:
                    categories[int(current_category["id"])] = str(
                        current_category.get("name", "")
                    ).strip().lower()
                current_category = None
            elif current_category is not None and event in _SCALAR_EVENTS:
                if prefix == "categories.item.id":
                    current_category["id"] = int(value)
                elif prefix == "categories.item.name":
                    current_category["name"] = str(value)

            if prefix == "annotations.item" and event == "start_map":
                current_annotation = {}
            elif prefix == "annotations.item" and event == "end_map":
                if (
                    current_annotation is not None
                    and "image_id" in current_annotation
                    and "category_id" in current_annotation
                    and isinstance(current_annotation.get("bbox"), list)
                    and len(current_annotation["bbox"]) == 4
                ):
                    annotations.append(current_annotation)
                current_annotation = None
                current_bbox = None
            elif current_annotation is not None:
                if prefix == "annotations.item.image_id" and event in _SCALAR_EVENTS:
                    current_annotation["image_id"] = int(value)
                elif prefix == "annotations.item.category_id" and event in _SCALAR_EVENTS:
                    current_annotation["category_id"] = int(value)
                elif prefix == "annotations.item.iscrowd" and event in _SCALAR_EVENTS:
                    current_annotation["iscrowd"] = int(value or 0)
                elif prefix == "annotations.item.bbox" and event == "start_array":
                    current_bbox = []
                elif prefix == "annotations.item.bbox.item" and event == "number":
                    if current_bbox is not None:
                        current_bbox.append(float(value))
                elif prefix == "annotations.item.bbox" and event == "end_array":
                    if current_bbox is not None:
                        current_annotation["bbox"] = current_bbox
                    current_bbox = None

            if progress_label and event_count % 4096 == 0:
                position = handle.tell()
                if position >= next_report:
                    print(
                        f"[coco] {progress_label}: "
                        f"{position / 1024 / 1024:.1f}/{size / 1024 / 1024:.1f} MiB "
                        f"({100.0 * position / max(1, size):.0f}%)",
                        flush=True,
                    )
                    next_report = position + report_step

    if progress_label:
        print(
            f"[coco] {progress_label}: ready | images={len(images)} | "
            f"boxes={len(annotations)}",
            flush=True,
        )
    return categories, images, annotations


class CocoResistorDetectionDataset(Dataset):
    """COCO bbox dataset mapped to one foreground class: resistor=1, background=0."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        *,
        augment: bool = False,
        category_names: Sequence[str] = ("resistor",),
        progress: bool = False,
    ) -> None:
        self.root = Path(root)
        self.split = str(split)
        self.split_root = self.root / self.split
        self.annotation_path = self.split_root / "_annotations.coco.json"
        if not self.annotation_path.is_file():
            raise FileNotFoundError(f"COCO annotation file not found: {self.annotation_path}")

        categories, images, annotations = load_coco_bbox_metadata(
            self.annotation_path,
            progress_label=self.split if progress else None,
        )
        wanted = {str(name).strip().lower() for name in category_names}
        self.category_ids = {category_id for category_id, name in categories.items() if name in wanted}
        if not self.category_ids:
            raise ValueError(
                f"None of the requested categories {sorted(wanted)} exist in "
                f"{self.annotation_path}; found {sorted(set(categories.values()))}"
            )

        self.images = sorted(images, key=lambda item: int(item["id"]))
        self.annotations_by_image: dict[int, list[dict]] = defaultdict(list)
        for annotation in annotations:
            if int(annotation.get("category_id", -1)) in self.category_ids:
                self.annotations_by_image[int(annotation["image_id"])].append(
                    {
                        "bbox": annotation["bbox"],
                        "iscrowd": int(annotation.get("iscrowd", 0)),
                    }
                )

        if not self.images:
            raise ValueError(f"No images found in {self.annotation_path}")
        self.augment = bool(augment)
        self.color_jitter = ColorJitter(
            brightness=0.20,
            contrast=0.20,
            saturation=0.15,
            hue=0.02,
        )

    def __len__(self) -> int:
        return len(self.images)

    def _image_path(self, file_name: str) -> Path:
        path = self.split_root / file_name
        if path.is_file():
            return path
        fallback = self.split_root / "images" / Path(file_name).name
        if fallback.is_file():
            return fallback
        raise FileNotFoundError(f"Image not found for COCO file_name={file_name!r}")

    def __getitem__(self, index: int):
        image_meta = self.images[index]
        image_id = int(image_meta["id"])
        image = Image.open(self._image_path(str(image_meta["file_name"]))).convert("RGB")
        width, height = image.size

        boxes: list[list[float]] = []
        iscrowd: list[int] = []
        for annotation in self.annotations_by_image.get(image_id, []):
            bbox = annotation.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            x, y, box_width, box_height = (float(value) for value in bbox)
            x1 = min(max(0.0, x), float(width))
            y1 = min(max(0.0, y), float(height))
            x2 = min(max(0.0, x + box_width), float(width))
            y2 = min(max(0.0, y + box_height), float(height))
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
            iscrowd.append(int(annotation.get("iscrowd", 0)))

        boxes_tensor = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels = torch.ones((len(boxes),), dtype=torch.int64)
        iscrowd_tensor = torch.tensor(iscrowd, dtype=torch.int64)
        area = (
            (boxes_tensor[:, 2] - boxes_tensor[:, 0])
            * (boxes_tensor[:, 3] - boxes_tensor[:, 1])
            if len(boxes)
            else torch.zeros((0,), dtype=torch.float32)
        )

        if self.augment:
            if random.random() < 0.5:
                image = TF.hflip(image)
                if len(boxes_tensor):
                    old_x1 = boxes_tensor[:, 0].clone()
                    old_x2 = boxes_tensor[:, 2].clone()
                    boxes_tensor[:, 0] = width - old_x2
                    boxes_tensor[:, 2] = width - old_x1
            if random.random() < 0.5:
                image = TF.vflip(image)
                if len(boxes_tensor):
                    old_y1 = boxes_tensor[:, 1].clone()
                    old_y2 = boxes_tensor[:, 3].clone()
                    boxes_tensor[:, 1] = height - old_y2
                    boxes_tensor[:, 3] = height - old_y1
            image = self.color_jitter(image)

        tensor = TF.pil_to_tensor(image).to(dtype=torch.float32).div_(255.0)
        target = {
            "boxes": boxes_tensor,
            "labels": labels,
            "image_id": torch.tensor([image_id], dtype=torch.int64),
            "area": area,
            "iscrowd": iscrowd_tensor,
        }
        return tensor, target


def collate_detection_batch(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def build_ssdlite_model(*, num_classes: int = 2, pretrained_backbone: bool = False):
    """Build mobile-friendly SSDLite without downloading weights by default."""
    backbone_weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained_backbone else None
    return ssdlite320_mobilenet_v3_large(
        weights=None,
        weights_backbone=backbone_weights,
        num_classes=int(num_classes),
    )


class DetectionExportWrapper(nn.Module):
    """Fixed-batch=1 export wrapper returning post-NMS boxes, scores and labels."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor):
        result = self.model([image[0]])[0]
        return result["boxes"], result["scores"], result["labels"]
