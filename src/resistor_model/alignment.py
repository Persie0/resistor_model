from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def normalize_axis_angle(angle: float) -> float:
    """Normalize an undirected line angle to [-90, 90)."""
    value = float(angle)
    while value >= 90.0:
        value -= 180.0
    while value < -90.0:
        value += 180.0
    return value


def angle_difference_180(a: float, b: float) -> float:
    """Return the smallest angular distance between two undirected axes."""
    return abs(normalize_axis_angle(float(a) - float(b)))


def consensus_axis_angle(angles: Sequence[float], weights: Sequence[float] | None = None) -> float:
    """Weighted circular mean for orientations where angle and angle+180 are equal."""
    if not angles:
        raise ValueError("angles must not be empty")
    values = np.asarray(angles, dtype=np.float64)
    if weights is None:
        weight_values = np.ones_like(values)
    else:
        weight_values = np.asarray(weights, dtype=np.float64)
        if weight_values.shape != values.shape:
            raise ValueError("weights must have the same length as angles")
    if np.any(weight_values < 0.0) or not np.any(weight_values > 0.0):
        raise ValueError("weights must be non-negative and at least one must be positive")

    doubled = np.radians(values * 2.0)
    x = float(np.sum(weight_values * np.cos(doubled)))
    y = float(np.sum(weight_values * np.sin(doubled)))
    if abs(x) < 1e-12 and abs(y) < 1e-12:
        raise ValueError("axis consensus is undefined for cancelling orientations")
    return normalize_axis_angle(np.degrees(np.arctan2(y, x)) / 2.0)


def rotation_to_vertical(axis_angle: float) -> float:
    """Return the minimal OpenCV rotation that maps an image-axis angle to vertical."""
    return normalize_axis_angle(float(axis_angle) - 90.0)
