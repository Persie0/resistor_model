"""IoU per body-color bucket: proof the segmenter sees ANY body color.

Ground-truth masks don't record body color, so this buckets each test image by
the mean hue of its own foreground pixels (original resolution, no letterbox
needed for statistics) and reports IoU overall + per bucket:

  gray   saturation < 35 (shape-only bodies, incl. synthesized gray)
  red    hue < 10 or >= 345 deg
  brown  10-40 deg  (classic carbon-film beige/brown)
  yellow 40-65 deg
  green  65-165 deg (classic green/blue-green metal-film... and synth green)
  blue   165-255 deg
  violet 255-345 deg
  other  anything else / unparseable

Works with a PyTorch LR-ASPP ``best.pt`` (``--checkpoint``) or ANY exported
``--onnx`` (e.g. PaddleSeg PP-LiteSeg): single-channel float masks are
thresholded at 0.5, two-channel logits are argmaxed, integer label maps are
used directly. The PyTorch checkpoint path uses its dataset preprocessing. The ONNX path
assumes PaddleSeg PP-LiteSeg RGB letterboxing and stock Normalize() defaults
(mean/std=0.5), matching the PP-LiteSeg quick-start training configuration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PADDLESEG_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
PADDLESEG_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)


def bucket(h_deg: float, s: float) -> str:
    if s < 35:
        return "gray"
    h = float(h_deg) % 360.0
    if h < 10 or h >= 345:
        return "red"
    if h < 40:
        return "brown"
    if h < 65:
        return "yellow"
    if h < 165:
        return "green"
    if h < 255:
        return "blue"
    if h < 345:
        return "violet"
    return "other"


def letterbox_rgb(img: Image.Image, size: int) -> np.ndarray:
    w, h = img.size
    scale = min(size / w, size / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    img = img.resize((nw, nh), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size))
    canvas.paste(img, ((size - nw) // 2, (size - nh) // 2))
    return np.asarray(canvas)


def body_hue_stats(image_path: Path, mask_path: Path) -> tuple[float, float]:
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if bgr is None or mask is None:
        return 0.0, 0.0
    fg = mask > 0
    if not fg.any():
        return 0.0, 0.0
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h_deg = hsv[..., 0].astype(np.float32)[fg] * 2.0
    s = hsv[..., 1].astype(np.float32)[fg]
    # Circular mean for hue.
    ang = np.deg2rad(h_deg.astype(np.float64))
    mean_h = float(np.rad2deg(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()))) % 360.0
    return mean_h, float(s.mean())


def _pairs_v4(split_dir: Path) -> list[tuple[Path, Path]]:
    images, masks = split_dir / "images", split_dir / "masks_semantic"
    exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    pairs = []
    for mask in sorted(masks.glob("*.png")):
        img = next((images / f"{mask.stem}{e}" for e in exts if (images / f"{mask.stem}{e}").is_file()), None)
        if img is None:
            img = next((images / f"{mask.stem}{e.upper()}" for e in exts if (images / f"{mask.stem}{e.upper()}").is_file()), None)
        if img is not None:
            pairs.append((img, mask))
    return pairs


def _iou(pred01: np.ndarray, gt01: np.ndarray) -> float:
    inter = np.logical_and(pred01, gt01).sum()
    union = np.logical_or(pred01, gt01).sum()
    return 1.0 if union == 0 else float(inter) / float(union)


def _letterbox_mask(mask: np.ndarray, size: int) -> np.ndarray:
    h, w = mask.shape
    scale = min(size / w, size / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    small = Image.fromarray((mask > 0).astype(np.uint8) * 255).resize((nw, nh), Image.Resampling.NEAREST)
    canvas = np.zeros((size, size), dtype=bool)
    ox, oy = (size - nw) // 2, (size - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = np.asarray(small) > 0
    return canvas


def _to_pred01(output: np.ndarray) -> np.ndarray:
    arr = np.asarray(output)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] == 2:  # logits [2,H,W]
        return arr.argmax(axis=0).astype(bool)
    if arr.ndim == 3 and arr.shape[-1] == 2:
        return arr.argmax(axis=-1).astype(bool)
    if arr.ndim == 3 and arr.shape[0] == 1:  # prob [1,H,W]
        return (arr[0] > 0.5)
    if arr.ndim == 2:
        if arr.dtype == bool or set(np.unique(arr)).issubset({0, 1}):
            return arr.astype(bool)
        return (arr > 0.5)
    raise ValueError(f"unsupported model output shape: {np.asarray(output).shape}")


def evaluate_onnx(onnx_path: Path, pairs: list[tuple[Path, Path]], size: int) -> list[tuple[str, float]]:
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    out: list[tuple[str, float]] = []
    for img_path, mask_path in pairs:
        with Image.open(img_path) as raw:
            rgb = letterbox_rgb(raw.convert("RGB"), size).astype(np.float32) / 255.0
        tensor = ((rgb - PADDLESEG_MEAN) / PADDLESEG_STD).transpose(2, 0, 1)[None]
        pred = _to_pred01(session.run(None, {input_name: tensor})[0])
        if pred.shape != (size, size):
            pred = np.asarray(Image.fromarray(pred.astype(np.uint8) * 255).resize((size, size), Image.Resampling.NEAREST)) > 0
        gt = _letterbox_mask(cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE), size)
        h, s = body_hue_stats(img_path, mask_path)
        out.append((bucket(h, s), _iou(pred, gt)))
    return out


def evaluate_checkpoint(ckpt_path: Path, dataset_root: Path, split: str, size: int) -> list[tuple[str, float]]:
    import torch

    from resistor_model.segmentation import CocoResistorSegmentationDataset, build_lraspp_model

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    size = int(ckpt.get("image_size", size))
    model = build_lraspp_model(num_classes=int(ckpt.get("num_classes", 2))).eval()
    model.load_state_dict(ckpt["model"])
    ds = CocoResistorSegmentationDataset(dataset_root, split, image_size=size, augment=False)
    out: list[tuple[str, float]] = []
    with torch.no_grad():
        for i in range(len(ds)):
            image, gt = ds[i]
            logits = model(image[None])["out"][0].argmax(dim=0).cpu().numpy().astype(bool)
            gt_np = gt.cpu().numpy().astype(bool)
            path, _, mask_or_anns = ds.samples[i]
            if isinstance(mask_or_anns, Path):
                h, s = body_hue_stats(path, mask_or_anns)
            else:
                h, s = 0.0, 0.0
            out.append((bucket(h, s), _iou(logits, gt_np)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Report segmentation IoU per body-color bucket")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--checkpoint", type=Path, help="PyTorch lraspp best.pt")
    parser.add_argument("--onnx", type=Path, help="Exported model (any framework)")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if bool(args.checkpoint) == bool(args.onnx):
        raise ValueError("pass exactly one of --checkpoint or --onnx")

    if args.onnx:
        pairs = _pairs_v4(args.dataset_root / args.split)
        if not pairs:
            raise ValueError(f"no v4 image/mask pairs under {args.dataset_root / args.split}")
        scored = evaluate_onnx(args.onnx, pairs, int(args.image_size))
    else:
        scored = evaluate_checkpoint(args.checkpoint, args.dataset_root, args.split, int(args.image_size))

    buckets: dict[str, list[float]] = {}
    for name, iou in scored:
        buckets.setdefault(name, []).append(iou)
    report = {
        "n": len(scored),
        "overall_iou": float(np.mean([v for _, v in scored])) if scored else 0.0,
        "buckets": {
            name: {"n": len(v), "iou": float(np.mean(v))}
            for name, v in sorted(buckets.items())
        },
    }
    print(json.dumps(report, indent=2), flush=True)
    if args.output is not None:
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
