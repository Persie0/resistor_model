"""Geometric augmentation for rectified resistor crops.

Rectification is deterministic given the annotation, so without this every training
crop is identical in every epoch and never reflects the crop error a real detector
produces at inference time. The jitter is deliberately bounded to keep the annotated
bands inside the crop for normal body boxes.
"""

from __future__ import annotations

import numpy as np


def sample_rectify_jitter(
    rng: np.random.Generator,
    strength: float = 1.0,
    identity_prob: float = 0.2,
) -> dict[str, float] | None:
    """Return a bounded random crop perturbation or ``None`` for the canonical crop."""
    if strength <= 0.0 or rng.random() < identity_prob:
        return None
    s = float(strength)
    return {
        "angle": float(np.deg2rad(rng.uniform(-4.0, 4.0) * s)),
        "shift_u": float(rng.uniform(-0.03, 0.03) * s),
        "shift_v": float(rng.uniform(-0.05, 0.05) * s),
        "scale_u": float(1.0 + rng.uniform(-0.06, 0.10) * s),
        "scale_v": float(1.0 + rng.uniform(-0.08, 0.20) * s),
    }


def apply_flips(
    rgb: np.ndarray,
    bands: list[dict],
    rng: np.random.Generator,
    hflip_prob: float = 0.5,
    vflip_prob: float = 0.5,
) -> tuple[np.ndarray, list[dict]]:
    """Randomly flip an HxWx3 crop and mirror the band boxes accordingly."""
    height, width = rgb.shape[:2]
    out = bands
    if rng.random() < hflip_prob:
        rgb = np.ascontiguousarray(rgb[:, ::-1])
        out = [
            {
                **band,
                "bbox": [
                    width - band["bbox"][2],
                    band["bbox"][1],
                    width - band["bbox"][0],
                    band["bbox"][3],
                ],
            }
            for band in out
        ]
    if rng.random() < vflip_prob:
        rgb = np.ascontiguousarray(rgb[::-1])
        out = [
            {
                **band,
                "bbox": [
                    band["bbox"][0],
                    height - band["bbox"][3],
                    band["bbox"][2],
                    height - band["bbox"][1],
                ],
            }
            for band in out
        ]
    return rgb, out
