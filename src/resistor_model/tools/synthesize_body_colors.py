"""Offline body-color synthesis for color-invariant resistor segmentation.

Segmentation (body vs. background) is color-agnostic: hue never changes the
label. But a dataset of mostly brown/beige bodies teaches the shortcut
"brown blob = resistor", so blue/green bodies get missed. This tool breaks
that shortcut WITHOUT new labels by recoloring exact SAM body masks:

* hue-rotates ONLY foreground (mask) pixels to blue / green / random hues,
  preserving per-pixel shading, bands and background realism,
* optionally adds desaturated (gray) bodies to force shape reliance,
* writes new image+mask pairs in place, so every trainer (PyTorch LR-ASPP,
  PaddleSeg PP-LiteSeg, ...) picks them up via its normal directory glob,
* records ``colors.jsonl`` (target hue per image) for hue-bucketed evaluation.

Only the ``train`` split is synthesized by default; valid/test stay real so
metrics measure reality, not synthesis.

Dependencies: numpy + OpenCV only (both are base install deps).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

# OpenCV hue units are 0-179 (= degrees / 2).
VARIANT_HUES = {
    "blue": 108,   # ~216 deg
    "green": 65,   # ~130 deg
}
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def _find_image(images_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTENSIONS:
        for candidate in (images_dir / f"{stem}{ext}", images_dir / f"{stem}{ext.upper()}"):
            if candidate.is_file():
                return candidate
    return None


def _median_hue_deg(hsv: np.ndarray, fg: np.ndarray) -> float:
    hues = hsv[..., 0].astype(np.float32)[fg]
    if hues.size == 0:
        return 0.0
    # Hues of one body cluster; linear median is fine, convert to degrees.
    return float(np.median(hues)) * 2.0


def _feather_alpha(mask01: np.ndarray) -> np.ndarray:
    h, w = mask01.shape
    k = max(3, (min(h, w) // 60) * 2 + 1)
    if k % 2 == 0:
        k += 1
    alpha = cv2.GaussianBlur((mask01 * 255).astype(np.uint8), (k, k), 0).astype(np.float32) / 255.0
    return alpha[..., None]


def recolor_body(
    img_bgr: np.ndarray,
    mask01: np.ndarray,
    rng: np.random.Generator,
    *,
    target_hue_ocv: int | None,
    gray: bool = False,
) -> np.ndarray:
    """Return a copy with foreground pixels recolored; background untouched."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0].astype(np.float32), hsv[..., 1].astype(np.float32), hsv[..., 2].astype(np.float32)
    fg = mask01 > 0
    if gray:
        gray_v = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        recol_bgr = cv2.cvtColor(gray_v.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    else:
        assert target_hue_ocv is not None
        delta = float(target_hue_ocv) - float(np.median(h.astype(np.float32)[fg])) if fg.any() else 0.0
        h2 = (h + delta + rng.uniform(-3.0, 3.0, h.shape)) % 180.0
        s2 = np.clip(s * rng.uniform(0.9, 1.25), 0, 255)
        recol_hsv = np.stack([h2, s2, v], axis=-1).astype(np.uint8)
        recol_bgr = cv2.cvtColor(recol_hsv, cv2.COLOR_HSV2BGR)
    alpha = _feather_alpha(mask01)
    out = img_bgr.astype(np.float32) * (1.0 - alpha) + recol_bgr.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def synthesize_split(
    split_dir: Path,
    *,
    variants: list[str],
    rng: np.random.Generator,
    jpeg_quality: int = 95,
    overwrite: bool = False,
) -> list[dict]:
    images_dir = split_dir / "images"
    masks_dir = split_dir / "masks_semantic"
    if not images_dir.is_dir() or not masks_dir.is_dir():
        raise FileNotFoundError(f"expected v4 layout under {split_dir}: images/ + masks_semantic/")
    records: list[dict] = []
    masks = sorted(masks_dir.glob("*.png"))
    for mask_path in masks:
        stem = mask_path.stem
        if "_syn" in stem:
            continue  # never re-synthesize synthesis
        img_path = _find_image(images_dir, stem)
        if img_path is None:
            continue
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None or img.shape[:2] != mask.shape[:2]:
            continue
        mask01 = (mask > 0).astype(np.uint8)
        if mask01.sum() == 0:
            continue
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        source_hue = _median_hue_deg(hsv, mask01 > 0)
        for variant in variants:
            tag = f"syn{variant}"
            out_img = images_dir / f"{stem}_{tag}.jpg"
            out_mask = masks_dir / f"{stem}_{tag}.png"
            if out_img.is_file() and out_mask.is_file() and not overwrite:
                target = None if variant == "gray" else int(VARIANT_HUES.get(variant, rng.integers(0, 180)))
                records.append({
                    "image": out_img.name, "mask": out_mask.name, "split": split_dir.name,
                    "synth_from": img_path.name, "variant": variant,
                    "target_hue_deg": None if variant == "gray" else float(target) * 2.0,
                    "source_hue_deg": source_hue, "skipped": True,
                })
                continue
            if variant == "gray":
                recol = recolor_body(img, mask01, rng, target_hue_ocv=None, gray=True)
                target_deg: float | None = None
            elif variant in VARIANT_HUES:
                recol = recolor_body(img, mask01, rng, target_hue_ocv=VARIANT_HUES[variant])
                target_deg = float(VARIANT_HUES[variant]) * 2.0
            elif variant == "random":
                recol = recolor_body(img, mask01, rng, target_hue_ocv=int(rng.integers(0, 180)))
                target_deg = None  # logged below from actual output
                out_hsv = cv2.cvtColor(recol, cv2.COLOR_BGR2HSV)
                target_deg = _median_hue_deg(out_hsv, mask01 > 0)
            else:
                raise ValueError(f"unknown variant: {variant!r} (blue|green|random|gray)")
            cv2.imwrite(str(out_img), recol, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
            cv2.imwrite(str(out_mask), mask)  # mask bytes unchanged: same silhouette
            records.append({
                "image": out_img.name, "mask": out_mask.name, "split": split_dir.name,
                "synth_from": img_path.name, "variant": variant,
                "target_hue_deg": target_deg, "source_hue_deg": source_hue,
            })
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize blue/green/random/gray resistor bodies")
    parser.add_argument("--dataset-root", type=Path, required=True,
                        help="Extracted v4 root containing train[/valid/test] with images/ + masks_semantic/")
    parser.add_argument("--splits", nargs="+", default=["train"])
    parser.add_argument("--variants", nargs="+", default=["blue", "green"],
                        choices=["blue", "green", "random", "gray"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    rng = np.random.default_rng(int(args.seed))
    log_path = args.dataset_root / "colors.jsonl"
    existing: dict[str, dict] = {}
    if log_path.is_file():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                existing[str(row.get("image"))] = row
    total_new = 0
    for split in args.splits:
        records = synthesize_split(
            args.dataset_root / split, variants=list(args.variants), rng=rng,
            jpeg_quality=int(args.jpeg_quality), overwrite=bool(args.overwrite),
        )
        for row in records:
            existing[row["image"]] = row
            if not row.get("skipped"):
                total_new += 1
        print(f"[synth] split {split}: {len(records)} variants ({total_new} newly written)", flush=True)
    log_path.write_text(
        "\n".join(json.dumps(existing[k]) for k in sorted(existing)) + "\n", encoding="utf-8",
    )
    print(f"[synth] wrote {log_path} | {total_new} new pairs", flush=True)


if __name__ == "__main__":
    main()
