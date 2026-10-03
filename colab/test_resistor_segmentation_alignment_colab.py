"""Interactive Colab tester for the exported whole-resistor segmenter.

Loads the ONNX model from Google Drive, lets the user upload a photo, finds the
largest resistor mask, estimates the long axis with several complementary
methods, rotates the resistor vertically, and displays the final tight crop.
It also produces a horizontal BandNet-ready crop.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort

from resistor_model.alignment import (
    angle_difference_180,
    consensus_axis_angle,
    normalize_axis_angle,
    rotation_to_vertical,
)

MODEL_PATH = Path(
    "/content/drive/MyDrive/resistor_model/segmentation-m2-lraspp/"
    "resistor_segmenter_lraspp.onnx"
)
IMAGE_SIZE = 384
THRESHOLD = 0.50
FINAL_PADDING = 0.20
ROI_PADDING = 0.60
OUTLIER_DEGREES = 25.0
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def estimate_border_color(image: np.ndarray) -> tuple[int, int, int]:
    border = np.concatenate(
        [image[0], image[-1], image[:, 0], image[:, -1]],
        axis=0,
    )
    return tuple(int(value) for value in np.median(border, axis=0))


def letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, int, int, int, int]:
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    left = (size - new_width) // 2
    top = (size - new_height) // 2
    canvas[top : top + new_height, left : left + new_width] = resized
    return canvas, left, top, new_width, new_height


def preprocess(image: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    canvas, left, top, new_width, new_height = letterbox(image, IMAGE_SIZE)
    tensor = canvas.astype(np.float32) / 255.0
    tensor = (tensor - MEAN) / STD
    tensor = np.transpose(tensor, (2, 0, 1))[None].astype(np.float32)
    return tensor, (left, top, new_width, new_height)


def restore_probability(
    probability: np.ndarray,
    letterbox_info: tuple[int, int, int, int],
    original_shape: tuple[int, int],
) -> np.ndarray:
    left, top, new_width, new_height = letterbox_info
    original_height, original_width = original_shape
    unpadded = probability[top : top + new_height, left : left + new_width]
    return cv2.resize(
        unpadded,
        (original_width, original_height),
        interpolation=cv2.INTER_LINEAR,
    )


def largest_clean_component(mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        raise RuntimeError("No resistor detected. Try THRESHOLD = 0.4 or 0.3.")
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    largest = np.zeros_like(mask, dtype=np.uint8)
    largest[labels == largest_label] = 255
    largest = cv2.morphologyEx(
        largest,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    )
    largest = cv2.morphologyEx(
        largest,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    return largest


def orientation_candidates(mask: np.ndarray) -> tuple[list[str], list[float], list[float]]:
    ys, xs = np.where(mask > 0)
    if len(xs) < 20:
        raise RuntimeError("Too few resistor pixels after mask cleanup.")
    points = np.column_stack((xs, ys)).astype(np.float32)

    fit = cv2.fitLine(points, cv2.DIST_WELSCH, 0, 0.01, 0.01).flatten()
    angle_fitline = normalize_axis_angle(np.degrees(np.arctan2(fit[1], fit[0])))

    _, eigenvectors = cv2.PCACompute(points, mean=None, maxComponents=2)
    angle_pca = normalize_axis_angle(
        np.degrees(np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0]))
    )

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise RuntimeError("No resistor contour found.")
    contour = max(contours, key=cv2.contourArea)

    box = cv2.boxPoints(cv2.minAreaRect(contour))
    rectangle_edges: list[tuple[float, float]] = []
    for index in range(4):
        vector = box[(index + 1) % 4] - box[index]
        rectangle_edges.append(
            (
                float(np.linalg.norm(vector)),
                normalize_axis_angle(np.degrees(np.arctan2(vector[1], vector[0]))),
            )
        )
    angle_rectangle = max(rectangle_edges, key=lambda item: item[0])[1]

    hull = cv2.convexHull(contour).reshape(-1, 2)
    max_distance_squared = -1.0
    best_pair: tuple[np.ndarray, np.ndarray] | None = None
    for index in range(len(hull)):
        remaining = hull[index + 1 :]
        if len(remaining) == 0:
            continue
        difference = remaining - hull[index]
        distances_squared = np.sum(difference.astype(np.float64) ** 2, axis=1)
        local_index = int(np.argmax(distances_squared))
        value = float(distances_squared[local_index])
        if value > max_distance_squared:
            max_distance_squared = value
            best_pair = (hull[index], remaining[local_index])
    if best_pair is None:
        raise RuntimeError("Could not determine longest hull chord.")
    long_vector = best_pair[1].astype(np.float32) - best_pair[0].astype(np.float32)
    angle_longest = normalize_axis_angle(
        np.degrees(np.arctan2(long_vector[1], long_vector[0]))
    )

    names = ["fitLine", "PCA", "minAreaRect", "longest chord"]
    angles = [angle_fitline, angle_pca, angle_rectangle, angle_longest]
    weights = [3.5, 3.0, 2.0, 1.0]
    return names, angles, weights


def robust_axis_angle(mask: np.ndarray) -> tuple[float, list[tuple[str, float, float, bool]]]:
    names, angles, weights = orientation_candidates(mask)
    initial = consensus_axis_angle(angles, weights)
    accepted_angles: list[float] = []
    accepted_weights: list[float] = []
    diagnostics: list[tuple[str, float, float, bool]] = []
    for name, angle, weight in zip(names, angles, weights):
        difference = angle_difference_180(angle, initial)
        accepted = difference <= OUTLIER_DEGREES
        diagnostics.append((name, angle, difference, accepted))
        if accepted:
            accepted_angles.append(angle)
            accepted_weights.append(weight)
    if not accepted_angles:
        raise RuntimeError("All orientation estimators were rejected.")
    return consensus_axis_angle(accepted_angles, accepted_weights), diagnostics


def contour_for_mask(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise RuntimeError("No resistor contour found.")
    return max(contours, key=cv2.contourArea)


def rotate_vertical_crop(
    image: np.ndarray,
    mask: np.ndarray,
    axis_angle: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    contour = contour_for_mask(mask)
    image_height, image_width = image.shape[:2]
    box_x, box_y, box_width, box_height = cv2.boundingRect(contour)
    roi_extra = int(max(box_width, box_height) * ROI_PADDING)
    x0 = max(0, box_x - roi_extra)
    y0 = max(0, box_y - roi_extra)
    x1 = min(image_width, box_x + box_width + roi_extra)
    y1 = min(image_height, box_y + box_height + roi_extra)
    roi_image = image[y0:y1, x0:x1]
    roi_mask = mask[y0:y1, x0:x1]
    roi_height, roi_width = roi_image.shape[:2]

    rotation_angle = rotation_to_vertical(axis_angle)
    center = (roi_width / 2.0, roi_height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    cosine = abs(matrix[0, 0])
    sine = abs(matrix[0, 1])
    rotated_width = int(roi_height * sine + roi_width * cosine)
    rotated_height = int(roi_height * cosine + roi_width * sine)
    matrix[0, 2] += rotated_width / 2 - center[0]
    matrix[1, 2] += rotated_height / 2 - center[1]

    rotated_image = cv2.warpAffine(
        roi_image,
        matrix,
        (rotated_width, rotated_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=estimate_border_color(roi_image),
    )
    rotated_mask = cv2.warpAffine(
        roi_mask,
        matrix,
        (rotated_width, rotated_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    ys, xs = np.where(rotated_mask > 0)
    if len(xs) == 0:
        raise RuntimeError("Rotated resistor mask is empty.")
    crop_x0, crop_x1 = int(xs.min()), int(xs.max())
    crop_y0, crop_y1 = int(ys.min()), int(ys.max())
    resistor_width = crop_x1 - crop_x0 + 1
    resistor_height = crop_y1 - crop_y0 + 1
    pad_x = int(resistor_width * FINAL_PADDING)
    pad_y = int(resistor_height * FINAL_PADDING)
    crop_x0 = max(0, crop_x0 - pad_x)
    crop_x1 = min(rotated_width - 1, crop_x1 + pad_x)
    crop_y0 = max(0, crop_y0 - pad_y)
    crop_y1 = min(rotated_height - 1, crop_y1 + pad_y)

    crop = rotated_image[crop_y0 : crop_y1 + 1, crop_x0 : crop_x1 + 1]
    crop_mask = rotated_mask[crop_y0 : crop_y1 + 1, crop_x0 : crop_x1 + 1]
    if crop.shape[1] > crop.shape[0]:
        crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
        crop_mask = cv2.rotate(crop_mask, cv2.ROTATE_90_CLOCKWISE)
    return crop, crop_mask, rotated_image, rotation_angle


def axis_debug_image(image: np.ndarray, mask: np.ndarray, axis_angle: float) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    center_x = float(xs.mean())
    center_y = float(ys.mean())
    theta = np.radians(axis_angle)
    length = max(image.shape[:2]) * 0.45
    dx = np.cos(theta) * length
    dy = np.sin(theta) * length
    start = (int(center_x - dx), int(center_y - dy))
    end = (int(center_x + dx), int(center_y + dy))
    debug = image.copy()
    cv2.line(debug, start, end, (255, 0, 0), max(2, round(max(image.shape[:2]) / 450)))
    return debug


def overlay_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    red = np.zeros_like(image)
    red[..., 0] = 255
    blended = (image.astype(np.float32) * 0.60 + red.astype(np.float32) * 0.40).astype(np.uint8)
    return np.where(mask[..., None] > 0, blended, image)


def load_session() -> ort.InferenceSession:
    available = ort.get_available_providers()
    providers = []
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    return ort.InferenceSession(str(MODEL_PATH), providers=providers)


def main() -> None:
    from google.colab import drive, files

    drive.mount("/content/drive", force_remount=False)
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"ONNX model not found: {MODEL_PATH}")

    session = load_session()
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]
    print(f"Model: {MODEL_PATH}")
    print(f"Providers: {session.get_providers()}")
    print(f"Input: {input_info.name} {input_info.shape}")
    print(f"Output: {output_info.name} {output_info.shape}")
    print("\nUpload a resistor photo:")
    uploaded = files.upload()
    if not uploaded:
        raise RuntimeError("No image uploaded.")
    filename = next(iter(uploaded))
    raw = np.frombuffer(uploaded[filename], dtype=np.uint8)
    bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Could not decode {filename}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    tensor, letterbox_info = preprocess(rgb)
    probability = session.run([output_info.name], {input_info.name: tensor})[0][0, 0]
    probability_original = restore_probability(probability, letterbox_info, rgb.shape[:2])
    mask = (probability_original >= THRESHOLD).astype(np.uint8)
    clean_mask = largest_clean_component(mask)

    axis_angle, diagnostics = robust_axis_angle(clean_mask)
    print("\nOrientation estimates:")
    for name, angle, difference, accepted in diagnostics:
        state = "accepted" if accepted else "REJECTED"
        print(f"  {name:15s}: {angle:7.2f}° | diff {difference:5.1f}° | {state}")
    print(f"Final resistor axis: {axis_angle:.2f}°")

    vertical_crop, vertical_mask, rotated_roi, rotation_angle = rotate_vertical_crop(
        rgb,
        clean_mask,
        axis_angle,
    )
    bandnet_crop = cv2.rotate(vertical_crop, cv2.ROTATE_90_CLOCKWISE)
    debug_axis = axis_debug_image(rgb, clean_mask, axis_angle)
    overlay = overlay_mask(rgb, clean_mask)

    final_height, final_width = vertical_crop.shape[:2]
    print(f"OpenCV rotation: {rotation_angle:.2f}°")
    print(f"Final vertical crop: {final_width}x{final_height}")
    print(f"BandNet-ready crop: {bandnet_crop.shape[1]}x{bandnet_crop.shape[0]}")

    plt.figure(figsize=(18, 11))
    panels = [
        (rgb, "Original", None),
        (probability_original, "Segmentation probability", "inferno"),
        (clean_mask, "Clean resistor mask", "gray"),
        (overlay, "Detected resistor", None),
        (debug_axis, f"Consensus long axis\n{axis_angle:.1f}°", None),
        (vertical_crop, f"Vertically aligned\n{final_width} × {final_height}", None),
    ]
    for index, (panel, title, cmap) in enumerate(panels, start=1):
        plt.subplot(2, 3, index)
        if cmap is None:
            plt.imshow(panel)
        else:
            plt.imshow(panel, cmap=cmap)
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 11))
    plt.imshow(vertical_crop)
    plt.title("FINAL VERTICALLY ALIGNED RESISTOR", fontsize=15)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 5))
    plt.imshow(bandnet_crop)
    plt.title("BANDNET-READY HORIZONTAL CROP", fontsize=15)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
