from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import random
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.segmentation import lraspp_mobilenet_v3_large
from torchvision.transforms import ColorJitter
from torchvision.transforms import functional as TF

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _find_split_dir(root: Path, split: str) -> Path:
    candidates = [split]
    if split == "val":
        candidates.append("valid")
    elif split == "valid":
        candidates.append("val")
    for name in candidates:
        path = root / name
        if (path / "_annotations.coco.json").is_file():
            return path
    raise FileNotFoundError(f"No COCO annotations found for split {split!r} under {root}")


def _polygon_points(raw: Iterable[float]) -> list[tuple[float, float]]:
    coords = [float(v) for v in raw]
    if len(coords) < 6 or len(coords) % 2:
        return []
    return list(zip(coords[0::2], coords[1::2]))


def _mask_from_annotations(width: int, height: int, annotations: list[dict], category_ids: set[int]) -> Image.Image:
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for annotation in annotations:
        if int(annotation.get("category_id", -1)) not in category_ids:
            continue
        segmentation = annotation.get("segmentation")
        drew_polygon = False
        if isinstance(segmentation, list):
            for segment in segmentation:
                if not isinstance(segment, list):
                    continue
                points = _polygon_points(segment)
                if len(points) >= 3:
                    draw.polygon(points, fill=1)
                    drew_polygon = True
        if drew_polygon:
            continue
        bbox = annotation.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4:
            x, y, w, h = (float(v) for v in bbox)
            if w > 0 and h > 0:
                draw.rectangle((x, y, x + w, y + h), fill=1)
    return mask


def _letterbox_pair(image: Image.Image, mask: Image.Image, size: int) -> tuple[Image.Image, Image.Image]:
    if size <= 0:
        raise ValueError("image_size must be positive")
    width, height = image.size
    scale = min(size / width, size / height)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    image = image.resize(new_size, Image.Resampling.BILINEAR)
    mask = mask.resize(new_size, Image.Resampling.NEAREST)
    pad_w = size - new_size[0]
    pad_h = size - new_size[1]
    left = pad_w // 2
    top = pad_h // 2
    padding = (left, top, pad_w - left, pad_h - top)
    return ImageOps.expand(image, border=padding, fill=0), ImageOps.expand(mask, border=padding, fill=0)


class CocoResistorSegmentationDataset(Dataset):
    """Roboflow-style COCO segmentation dataset collapsed to resistor/background."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        *,
        image_size: int = 384,
        augment: bool = False,
        category_names: Iterable[str] | None = None,
    ) -> None:
        self.root = Path(root)
        self.split_dir = _find_split_dir(self.root, split)
        # The v4 SAM 3.1 release already ships exact binary body masks.
        # Read paired PNGs directly: COCO RLE parsing is expensive and older
        # Roboflow polygon/bbox fallback would destroy the SAM silhouettes.
        mask_dir = self.split_dir / "masks_semantic"
        if mask_dir.is_dir():
            wanted = {name.strip().casefold() for name in category_names} if category_names else None
            if wanted is not None and "resistor" not in wanted:
                raise ValueError("v4 semantic PNG masks represent the 'resistor' foreground class")
            image_dir = self.split_dir / "images"
            image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
            images = {
                path.stem: path
                for path in image_dir.iterdir()
                if path.is_file() and path.suffix.casefold() in image_extensions
            } if image_dir.is_dir() else {}
            masks = sorted(path for path in mask_dir.glob("*.png") if path.is_file())
            missing_images = [mask.name for mask in masks if mask.stem not in images]
            missing_masks = [stem for stem in images if not (mask_dir / f"{stem}.png").is_file()]
            if not masks or missing_images or missing_masks:
                raise ValueError(
                    f"Invalid v4 semantic mask/image pairs in {self.split_dir}: "
                    f"masks={len(masks)}, images={len(images)}, "
                    f"missing images={missing_images[:5]}, missing masks={missing_masks[:5]}"
                )
            self.samples = [(images[mask.stem], {}, mask) for mask in masks]
            self.category_ids = {1}
            self.image_size = int(image_size)
            self.augment = bool(augment)
            self.color_jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15, hue=0.03)
            return

        payload = json.loads((self.split_dir / "_annotations.coco.json").read_text(encoding="utf-8"))
        wanted = {name.strip().casefold() for name in category_names} if category_names else None
        categories = {
            int(row["id"]): str(row.get("name", "")).strip()
            for row in payload.get("categories", [])
        }
        if wanted is None:
            resistor_ids = {
                category_id
                for category_id, name in categories.items()
                if "resistor" in name.casefold()
            }
            self.category_ids = resistor_ids or set(categories)
        else:
            self.category_ids = {category_id for category_id, name in categories.items() if name.casefold() in wanted}
            if not self.category_ids:
                raise ValueError(f"None of categories {sorted(wanted)} exist; available={sorted(categories.values())}")
        annotations_by_image: dict[int, list[dict]] = defaultdict(list)
        for annotation in payload.get("annotations", []):
            annotations_by_image[int(annotation["image_id"])].append(annotation)
        self.samples = []
        for image in payload.get("images", []):
            image_id = int(image["id"])
            path = self.split_dir / str(image["file_name"])
            if not path.is_file():
                continue
            self.samples.append((path, image, annotations_by_image.get(image_id, [])))
        if not self.samples:
            raise ValueError(f"No readable images found in {self.split_dir}")
        self.image_size = int(image_size)
        self.augment = bool(augment)
        self.color_jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15, hue=0.03)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, _image_meta, annotations_or_mask = self.samples[index]
        with Image.open(path) as raw_image:
            image = raw_image.convert("RGB")
        width, height = image.size
        if isinstance(annotations_or_mask, Path):
            with Image.open(annotations_or_mask) as raw_mask:
                if raw_mask.size != (width, height):
                    raise ValueError(
                        f"v4 semantic mask dimensions {raw_mask.size} do not match "
                        f"image dimensions {(width, height)}: {annotations_or_mask}"
                    )
                # SAM masks use 0=background and 255=foreground.
                mask = raw_mask.convert("L").point(lambda value: 1 if value > 0 else 0)
        else:
            mask = _mask_from_annotations(width, height, annotations_or_mask, self.category_ids)
        image, mask = _letterbox_pair(image, mask, self.image_size)

        if self.augment:
            if random.random() < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            if random.random() < 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
            turns = random.randrange(4)
            if turns:
                angle = 90 * turns
                image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
                mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)
            image = self.color_jitter(image)

        image_tensor = TF.to_tensor(image)
        image_tensor = TF.normalize(image_tensor, IMAGENET_MEAN, IMAGENET_STD)
        mask_tensor = torch.from_numpy(np.array(mask, dtype=np.uint8, copy=True)).long()
        mask_tensor.clamp_(0, 1)
        return image_tensor, mask_tensor


def build_lraspp_model(*, num_classes: int = 2, pretrained_backbone: bool = False) -> nn.Module:
    weights_backbone = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained_backbone else None
    return lraspp_mobilenet_v3_large(
        weights=None,
        weights_backbone=weights_backbone,
        num_classes=num_classes,
    )


def segmentation_loss(logits: torch.Tensor, target: torch.Tensor, *, foreground_weight: float = 2.0) -> torch.Tensor:
    class_weights = logits.new_tensor([1.0, float(foreground_weight)])
    ce = F.cross_entropy(logits, target, weight=class_weights)
    foreground = torch.softmax(logits, dim=1)[:, 1]
    target_fg = target.float()
    intersection = (foreground * target_fg).sum(dim=(1, 2))
    denominator = foreground.sum(dim=(1, 2)) + target_fg.sum(dim=(1, 2))
    dice = (2.0 * intersection + 1.0) / (denominator + 1.0)
    return ce + (1.0 - dice.mean())


class ForegroundProbabilityWrapper(nn.Module):
    """Export-friendly wrapper returning [N,1,H,W] foreground probabilities."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        logits = self.model(image)["out"]
        return torch.softmax(logits, dim=1)[:, 1:2]
