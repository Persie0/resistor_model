from __future__ import annotations

import cv2
import numpy as np


def _band_centers(bands: list[dict]) -> np.ndarray:
    return np.asarray([[(b["bbox"][0] + b["bbox"][2]) * 0.5, (b["bbox"][1] + b["bbox"][3]) * 0.5] for b in bands], dtype=np.float32)


def axis_unit(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if len(points) < 2:
        return np.array([1.0, 0.0], dtype=np.float32)
    centered = points - points.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    u = vt[0].astype(np.float32)
    if abs(float(u[0])) >= abs(float(u[1])):
        if u[0] < 0: u = -u
    elif u[1] < 0:
        u = -u
    return u / max(float(np.linalg.norm(u)), 1e-8)


def estimate_axis_angle(points: np.ndarray) -> float:
    u = axis_unit(points)
    return float(np.degrees(np.arctan2(u[1], u[0])))


def _corners(bbox: list[float] | tuple[float, ...]) -> np.ndarray:
    x1, y1, x2, y2 = map(float, bbox)
    return np.asarray([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)


def _transform_bbox(h: np.ndarray, bbox: list[float] | tuple[float, ...]) -> list[float]:
    pts = cv2.perspectiveTransform(_corners(bbox)[None, :, :], h)[0]
    return [float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())]


def rectify_resistor(image: np.ndarray, bands: list[dict], resistor_bbox: list[float] | tuple[float, ...] | None = None, output_size: tuple[int, int] = (128, 768), longitudinal_margin: float = 0.30, transverse_margin: float = 1.2) -> tuple[np.ndarray, list[dict]]:
    if not bands:
        raise ValueError("at least one band is required")
    out_h, out_w = output_size
    centers = _band_centers(bands)
    u = axis_unit(centers)
    v = np.array([-u[1], u[0]], dtype=np.float32)
    origin = centers.mean(axis=0)
    pts = _corners(resistor_bbox) if resistor_bbox is not None else np.concatenate([_corners(b["bbox"]) for b in bands], axis=0)
    rel = pts - origin
    pu, pv = rel @ u, rel @ v
    min_u, max_u = float(pu.min()), float(pu.max())
    min_v, max_v = float(pv.min()), float(pv.max())
    band_widths_u, band_heights_v = [], []
    for b in bands:
        bp = _corners(b["bbox"]) - origin
        bu, bv = bp @ u, bp @ v
        band_widths_u.append(float(bu.max() - bu.min()))
        band_heights_v.append(float(bv.max() - bv.min()))
    if resistor_bbox is None:
        span_u = max(max_u - min_u, np.median(band_widths_u) * 4.0, 1.0)
        body_half = max((max_v - min_v) * 0.5, float(np.median(band_heights_v)) * transverse_margin)
        min_u -= span_u * longitudinal_margin
        max_u += span_u * longitudinal_margin
        min_v, max_v = -body_half, body_half
    else:
        pad_u = max(1.0, 0.05 * (max_u - min_u)); pad_v = max(1.0, 0.08 * (max_v - min_v))
        min_u -= pad_u; max_u += pad_u; min_v -= pad_v; max_v += pad_v
    src = np.asarray([origin + u * min_u + v * min_v, origin + u * max_u + v * min_v, origin + u * max_u + v * max_v, origin + u * min_u + v * max_v], dtype=np.float32)
    dst = np.asarray([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
    h = cv2.getPerspectiveTransform(src, dst)
    crop = cv2.warpPerspective(image, h, (out_w, out_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    transformed = [{"color": b["color"], "bbox": _transform_bbox(h, b["bbox"])} for b in bands]
    return crop, transformed
