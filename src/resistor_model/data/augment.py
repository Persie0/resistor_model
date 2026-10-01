from __future__ import annotations

import cv2
import numpy as np
import torch


class PhotometricAugment:
    def __init__(self, probability: float = 0.85, exposure_ev: float = 1.5, gamma_range: tuple[float, float] = (0.7, 1.4), wb_range: tuple[float, float] = (0.78, 1.28), noise_std: float = 0.025, seed: int | None = None) -> None:
        self.probability = float(probability); self.exposure_ev = float(exposure_ev); self.gamma_range = gamma_range; self.wb_range = wb_range; self.noise_std = float(noise_std); self.rng = np.random.default_rng(seed)

    def reseed(self, seed: int) -> None:
        """Reset the RNG, e.g. from a PyTorch DataLoader worker seed."""
        self.rng = np.random.default_rng(int(seed))

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if self.probability <= 0 or self.rng.random() > self.probability:
            return image.copy()
        x = image.astype(np.float32, copy=True)
        lo, hi = self.wb_range
        gains = np.array([self.rng.uniform(lo, hi), self.rng.uniform(max(lo, 0.9), min(hi, 1.1)), self.rng.uniform(lo, hi)], dtype=np.float32)
        x *= gains[None, None, :]
        x *= 2.0 ** self.rng.uniform(-self.exposure_ev, self.exposure_ev)
        x = np.clip(x, 0.0, 1.0)
        x = np.power(x, self.rng.uniform(*self.gamma_range), dtype=np.float32)
        h, w = x.shape[:2]
        if self.rng.random() < 0.45:
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32); theta = self.rng.uniform(0, np.pi)
            coord = xx * np.cos(theta) + yy * np.sin(theta); coord = (coord - coord.min()) / max(float(np.ptp(coord)), 1e-6)
            center = self.rng.uniform(0.2, 0.8); softness = self.rng.uniform(0.10, 0.30)
            mask = 1.0 / (1.0 + np.exp(-(coord - center) / softness))
            if self.rng.random() < 0.5: mask = 1.0 - mask
            x *= (1.0 - self.rng.uniform(0.25, 0.7) * mask[..., None]).astype(np.float32)
        if self.rng.random() < 0.35:
            yy = np.arange(h, dtype=np.float32)[:, None]; center_y = self.rng.uniform(0.2 * h, 0.8 * h); sigma = self.rng.uniform(max(1.0, h * 0.025), max(2.0, h * 0.12))
            stripe = np.exp(-0.5 * ((yy - center_y) / sigma) ** 2)[:, :, None]
            x = x + stripe * self.rng.uniform(0.08, 0.35)
        if self.rng.random() < 0.25:
            k = int(self.rng.choice([3, 5])); x = cv2.GaussianBlur(x, (k, k), self.rng.uniform(0.2, 1.2))
        if self.rng.random() < 0.35 and self.noise_std > 0:
            x += self.rng.normal(0.0, self.rng.uniform(0.002, self.noise_std), x.shape).astype(np.float32)
        if self.rng.random() < 0.18:
            quality = int(self.rng.integers(45, 91)); jpeg_bgr = (cv2.cvtColor(np.clip(x, 0, 1), cv2.COLOR_RGB2BGR) * 255.0).round().astype(np.uint8)
            ok, encoded = cv2.imencode(".jpg", jpeg_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if ok:
                decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR); x = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return np.clip(x, 0.0, 1.0).astype(np.float32)


def _srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055).pow(2.4))


def make_chromatic_channels(rgb: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError("rgb must have shape [B,3,H,W]")
    linear = _srgb_to_linear(rgb.clamp(0, 1)).clamp_min(eps); log_rgb = linear.log(); r, g, b = log_rgb[:, 0:1], log_rgb[:, 1:2], log_rgb[:, 2:3]
    return torch.cat([log_rgb, r - g, b - g], dim=1)
