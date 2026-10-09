"""Interactive image tester for the exported v4 SSDLite resistor box detector.

Upload one or multiple photos in Colab and visualize mapped bounding boxes.
The ONNX graph includes torchvision NMS, and boxes are in 320x320 input pixels.
"""

from __future__ import annotations

from pathlib import Path
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort

MODEL_PATH = Path(
    "/content/drive/MyDrive/resistor_model/detector-v4-ssdlite320/"
    "resistor_detector_ssdlite320.onnx"
)
IMAGE_SIZE = 320
SCORE_THRESHOLD = 0.35
MAX_DISPLAY_BOXES = 30


def decode_detections(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    source_width: int,
    source_height: int,
    score_threshold: float = SCORE_THRESHOLD,
) -> list[tuple[tuple[int, int, int, int], float]]:
    """Filter post-NMS class-1 predictions and map stretched xyxy back to source."""
    if source_width <= 0 or source_height <= 0:
        raise ValueError("Image dimensions must be positive")
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if not (len(boxes) == len(scores) == len(labels)):
        raise ValueError("ONNX detection outputs have inconsistent lengths")

    detected = []
    for index in np.argsort(-scores):
        score = float(scores[index])
        if labels[index] != 1 or not np.isfinite(score) or score < score_threshold:
            continue
        x0, y0, x1, y1 = boxes[index]
        if not np.all(np.isfinite([x0, y0, x1, y1])):
            continue
        left = int(round(np.clip(x0 * source_width / IMAGE_SIZE, 0, source_width - 1)))
        top = int(round(np.clip(y0 * source_height / IMAGE_SIZE, 0, source_height - 1)))
        right = int(round(np.clip(x1 * source_width / IMAGE_SIZE, 0, source_width - 1)))
        bottom = int(round(np.clip(y1 * source_height / IMAGE_SIZE, 0, source_height - 1)))
        if right <= left or bottom <= top:
            continue
        detected.append(((left, top, right, bottom), score))
        if len(detected) >= MAX_DISPLAY_BOXES:
            break
    return detected


def preprocess(image_rgb: np.ndarray) -> np.ndarray:
    """Stretch to 320x320, RGB [0,1]; torchvision normalization is in ONNX."""
    square = cv2.resize(image_rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    return np.transpose(square.astype(np.float32) / 255.0, (2, 0, 1))[None]


def predict(
    session: ort.InferenceSession,
    image_rgb: np.ndarray,
    *,
    score_threshold: float = SCORE_THRESHOLD,
) -> tuple[list[tuple[tuple[int, int, int, int], float]], float]:
    input_name = session.get_inputs()[0].name
    start = time.perf_counter()
    boxes, scores, labels = session.run(
        ["boxes", "scores", "labels"], {input_name: preprocess(image_rgb)}
    )
    elapsed_ms = (time.perf_counter() - start) * 1000
    height, width = image_rgb.shape[:2]
    detections = decode_detections(
        boxes, scores, labels,
        source_width=width, source_height=height,
        score_threshold=score_threshold,
    )
    return detections, elapsed_ms


def overlay_boxes(
    image_rgb: np.ndarray,
    detections: list[tuple[tuple[int, int, int, int], float]],
) -> np.ndarray:
    output = image_rgb.copy()
    thickness = max(2, round(max(output.shape[:2]) / 450))
    for (x0, y0, x1, y1), score in detections:
        cv2.rectangle(output, (x0, y0), (x1, y1), (0, 220, 70), thickness)
        label = f"resistor {score:.2f}"
        label_y = max(15, y0 - 8)
        cv2.putText(output, label, (x0, label_y), cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.45, thickness / 3), (0, 220, 70), thickness, cv2.LINE_AA)
    return output


def main() -> None:
    from google.colab import drive, files

    drive.mount("/content/drive", force_remount=False)
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Detector ONNX model not found: {MODEL_PATH}")

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    providers = [provider for provider in providers if provider in ort.get_available_providers()]
    session = ort.InferenceSession(str(MODEL_PATH), providers=providers)
    print(f"Model: {MODEL_PATH}")
    print(f"Providers: {session.get_providers()}")
    print(f"Minimum confidence: {SCORE_THRESHOLD:.2f} (edit SCORE_THRESHOLD to adjust)")
    print("Upload one or more JPG/PNG photos:")
    uploaded = files.upload()
    if not uploaded:
        print("No photos uploaded; you can rerun this cell later.")
        return

    for filename, image_bytes in uploaded.items():
        raw = np.frombuffer(image_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"{filename}: unable to read image, skipped")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        detections, elapsed_ms = predict(session, rgb)
        print(f"{filename}: {len(detections)} resistor(s), {elapsed_ms:.1f} ms ONNX inference")
        if not detections:
            print("  No box above threshold; consider adjusting SCORE_THRESHOLD.")
        else:
            for index, (coordinates, score) in enumerate(detections, 1):
                print(f"  {index}. confidence={score:.3f} box={coordinates}")

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes[0].imshow(rgb)
        axes[0].set_title(f"Original: {filename}")
        axes[1].imshow(overlay_boxes(rgb, detections))
        axes[1].set_title(f"SSDLite detections: {len(detections)}")
        for axis in axes:
            axis.axis("off")
        plt.tight_layout()
        plt.show()
        plt.close(fig)


if __name__ == "__main__":
    main()
