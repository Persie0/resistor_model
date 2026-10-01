from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from resistor_model.constants import BACKGROUND_INDEX, COLOR_TO_INDEX
from .augment import PhotometricAugment
from .geometry import rectify_resistor
from .schema import ResistorAnnotation, load_manifest


@dataclass(frozen=True)
class SampleRef:
    image: str
    resistor: ResistorAnnotation
    session_id: str | None


class ResistorBandDataset(Dataset):
    def __init__(self, manifest: str | Path, image_root: str | Path | None = None, *, split: str | None = None, allowed_resistor_ids: set[str] | None = None, output_size: tuple[int, int] = (128, 768), sequence_bins: int = 256, max_bands: int = 6, augment: bool = False, two_views: bool = False, seed: int = 42) -> None:
        self.manifest_path = Path(manifest)
        self.image_root = Path(image_root) if image_root is not None else self.manifest_path.parent
        self.output_size = output_size; self.sequence_bins = int(sequence_bins); self.max_bands = int(max_bands); self.two_views = bool(two_views)
        self.augmenter = PhotometricAugment(seed=seed) if augment else PhotometricAugment(probability=0.0, seed=seed)
        self.augmenter2 = PhotometricAugment(seed=seed + 100003) if augment else PhotometricAugment(probability=0.0, seed=seed + 100003)
        self.samples: list[SampleRef] = []
        for row in load_manifest(self.manifest_path):
            if split is not None and row.split != split: continue
            for resistor in row.resistors:
                if allowed_resistor_ids is not None and resistor.id not in allowed_resistor_ids: continue
                self.samples.append(SampleRef(row.image, resistor, row.session_id))

    def __len__(self) -> int:
        return len(self.samples)

    def _targets(self, bands: list[dict], width: int) -> dict[str, torch.Tensor]:
        ordered = sorted(bands, key=lambda b: (b["bbox"][0] + b["bbox"][2]) * 0.5)
        if len(ordered) > self.max_bands: raise ValueError(f"sample has {len(ordered)} bands but max_bands={self.max_bands}")
        dense = torch.full((self.sequence_bins,), BACKGROUND_INDEX, dtype=torch.long)
        exists = torch.zeros(self.max_bands, dtype=torch.float32); colors = torch.full((self.max_bands,), -100, dtype=torch.long)
        centers = torch.zeros(self.max_bands, dtype=torch.float32); widths = torch.zeros(self.max_bands, dtype=torch.float32)
        for i, band in enumerate(ordered):
            x1, _, x2, _ = band["bbox"]; x1 = float(np.clip(x1, 0, width)); x2 = float(np.clip(x2, 0, width))
            if x2 <= x1: continue
            color_idx = COLOR_TO_INDEX[band["color"]]; exists[i] = 1.0; colors[i] = color_idx; centers[i] = ((x1 + x2) * 0.5) / width; widths[i] = (x2 - x1) / width
            b1 = max(0, min(self.sequence_bins - 1, int(np.floor(x1 / width * self.sequence_bins)))); b2 = max(b1 + 1, min(self.sequence_bins, int(np.ceil(x2 / width * self.sequence_bins))))
            dense[b1:b2] = color_idx
        return {"dense_target": dense, "slot_exists": exists, "slot_colors": colors, "slot_centers": centers, "slot_widths": widths, "count": torch.tensor(len(ordered), dtype=torch.long)}

    @staticmethod
    def _to_tensor(rgb: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float()

    def __getitem__(self, index: int) -> dict:
        ref = self.samples[index]; path = self.image_root / ref.image; bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None: raise FileNotFoundError(f"could not read image: {path}")
        bands = [{"color": b.color, "bbox": list(b.bbox)} for b in ref.resistor.bands]
        crop_bgr, transformed = rectify_resistor(
            bgr,
            bands,
            resistor_bbox=list(ref.resistor.bbox) if ref.resistor.bbox else None,
            resistor_polygon=[list(point) for point in ref.resistor.polygon] if ref.resistor.polygon else None,
            output_size=self.output_size,
        )
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0; targets = self._targets(transformed, self.output_size[1])
        out: dict = {"image": self._to_tensor(self.augmenter(rgb)), **targets, "resistor_id": ref.resistor.id, "image_path": str(path)}
        if self.two_views: out["image_view2"] = self._to_tensor(self.augmenter2(rgb))
        return out
