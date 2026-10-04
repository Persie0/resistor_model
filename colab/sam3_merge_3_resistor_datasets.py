"""Colab workflow: rebuild three resistor COCO datasets with SAM 3.1 body masks.

The source COCO annotations are used only to associate detections with known resistor
instances. SAM 3.1 is run with the text prompt ``resistor body`` once per image. Each
source annotation receives at most one text-prompt candidate, and each SAM candidate
can be assigned only once. The selected SAM mask replaces the old segmentation.
Images with no accepted SAM body mask are removed from the final merged dataset.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import random
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

SAM3_COLAB_VERSION = "2026-10-04-v5-body-only"
BODY_TEXT_PROMPT = "resistor body"
NUM_DATASETS = 3
FINAL_CATEGORY_NAME = "resistor"
SOURCE_CATEGORY_NAMES = {"resistor", "resistor_body", "resistors", "res"}

SAM_PROCESSOR_THRESHOLD = 0.15
MIN_SAM_SCORE = 0.20
MIN_SAM_COVERAGE = 0.55
MIN_BBOX_CONTAINMENT = 0.60
MAX_NORMALIZED_CENTER_DISTANCE = 0.90
MAX_CANDIDATE_TO_OLD_BBOX_AREA = 1.35
MIN_IMAGE_AREA_FRACTION = 0.00015
MAX_IMAGE_AREA_FRACTION = 0.80
FINAL_INSTANCE_DUP_IOU = 0.85

TRAIN_FRACTION = 0.80
VALID_FRACTION = 0.10
SPLIT_SEED = 1337
PROGRESS_EVERY_IMAGES = 25
DOWNLOAD_CHUNK_BYTES = 8 * 1024 * 1024

WORK_ROOT = Path("/content/sam3_merge_work")
OUTPUT_ROOT = Path("/content/resistor_sam3_merged")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

RELEASE_BASE = "https://github.com/Persie0/resistor_model/releases/download/r1"
DATASET_ASSETS = (
    "rres.v4i.coco.zip",
    "resistor.value.training.v8i.coco.zip",
    "Deteksi.Nilai.Resistor.v1i.coco.zip",
)
DATASET_SHA256 = {
    "rres.v4i.coco.zip": "24b46548700156a5f553197193c153b42381472b3eddfa2f9cf19d849fc84d4a",
    "resistor.value.training.v8i.coco.zip": "dbf4b665d361a98d01024481970fc74be27d71e795b95dac5936af94dd4cf958",
    "Deteksi.Nilai.Resistor.v1i.coco.zip": "ee8db6b64e59dfe47050b313f215051d3da5932dbe5e1f7b7d94e6495449dc92",
}


def log(message=""):
    print(message, flush=True)


def stage(number, total, title):
    log("")
    log("=" * 72)
    log(f"[stage {number}/{total}] {title}")
    log("=" * 72)


def run(command, cwd=None):
    log("+ " + " ".join(map(str, command)))
    subprocess.run(
        list(map(str, command)),
        cwd=str(cwd) if cwd else None,
        check=True,
    )


def ensure_sam3_ready():
    """Reuse the notebook bootstrap model, or bootstrap SAM 3.1 when run standalone."""
    if globals().get("SAM3_BOOTSTRAPPED", False) and globals().get("processor") is not None:
        log("[sam3] reusing already-loaded SAM 3.1 image model")
        return

    log("[sam3] no loaded model found; running the shared SAM 3.1 bootstrap")
    bootstrap_url = (
        "https://raw.githubusercontent.com/Persie0/resistor_model/main/"
        "colab/sam3_test_bootstrap.py"
    )
    bootstrap_path = "/content/sam3_test_bootstrap.py"
    urllib.request.urlretrieve(bootstrap_url, bootstrap_path)
    source = Path(bootstrap_path).read_text(encoding="utf-8")
    exec(compile(source, bootstrap_path, "exec"), globals(), globals())


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_with_progress(url, destination, label):
    request = urllib.request.Request(url, headers={"User-Agent": "resistor-model-colab"})
    started = time.monotonic()
    with urllib.request.urlopen(request) as response, open(destination, "wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        copied = 0
        report_step = max(DOWNLOAD_CHUNK_BYTES, total // 10 if total else DOWNLOAD_CHUNK_BYTES)
        next_report = report_step
        log(
            f"[download] {label}: {total / 1024 / 1024:.1f} MiB"
            if total
            else f"[download] {label}: size unknown"
        )
        while True:
            chunk = response.read(DOWNLOAD_CHUNK_BYTES)
            if not chunk:
                break
            out.write(chunk)
            copied += len(chunk)
            if copied >= next_report or (total and copied >= total):
                elapsed = max(0.001, time.monotonic() - started)
                speed = copied / elapsed / 1024 / 1024
                if total:
                    log(
                        f"[download] {label}: {copied / 1024 / 1024:.1f}/"
                        f"{total / 1024 / 1024:.1f} MiB "
                        f"({100.0 * copied / total:.0f}%) | {speed:.1f} MiB/s"
                    )
                else:
                    log(f"[download] {label}: {copied / 1024 / 1024:.1f} MiB | {speed:.1f} MiB/s")
                next_report = copied + report_step
    log(
        f"[download] {label}: complete | {copied / 1024 / 1024:.1f} MiB in "
        f"{max(0.001, time.monotonic() - started):.1f}s"
    )


def looks_like_coco(path):
    try:
        with open(path, encoding="utf-8") as handle:
            obj = json.load(handle)
        return all(isinstance(obj.get(key), list) for key in ("images", "annotations", "categories"))
    except Exception:
        return False


def basename_index(root):
    output = defaultdict(list)
    for path in Path(root).rglob("*"):
        if path.is_file() and path.suffix.lower() in IMG_EXTS:
            output[path.name].append(path)
    return output


def resolve_image(root, json_path, file_name, index):
    file_name = str(file_name).replace("\\", "/")
    candidates = (
        Path(json_path).parent / file_name,
        Path(root) / file_name,
        Path(json_path).parent / Path(file_name).name,
    )
    for path in candidates:
        if path.is_file():
            return str(path)
    hits = index.get(Path(file_name).name, [])
    return str(hits[0]) if len(hits) == 1 else None


def write_manifest_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    preferred = [
        "source", "source_asset", "image", "old_instance", "accepted", "reason",
        "sam_candidate", "sam_score", "match_rank", "mask_iou", "old_coverage",
        "sam_coverage", "bbox_iou", "bbox_containment", "center_distance",
        "candidate_to_old_bbox_area",
    ]
    fields = [key for key in preferred if any(key in row for row in rows)]
    fields.extend(sorted({key for row in rows for key in row} - set(fields)))
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


stage(1, 8, "SAM 3.1 environment")
log(f"SAM 3.1 body-only COCO merge version: {SAM3_COLAB_VERSION}")
ensure_sam3_ready()
run([sys.executable, "-m", "pip", "install", "-q", "pycocotools"])

import numpy as np
import torch
from google.colab import files
from PIL import Image
from pycocotools import mask as mask_utils
from tqdm.auto import tqdm

if not torch.cuda.is_available():
    raise RuntimeError("No CUDA GPU detected. In Colab choose Runtime -> Change runtime type -> GPU.")
log(f"[sam3] prompt: {BODY_TEXT_PROMPT!r}")
log(f"[sam3] GPU: {torch.cuda.get_device_name(0)}")


stage(2, 8, "Download and extract the 3 source COCO datasets")
shutil.rmtree(WORK_ROOT, ignore_errors=True)
WORK_ROOT.mkdir(parents=True, exist_ok=True)
source_roots = []
for index, asset_name in enumerate(DATASET_ASSETS, 1):
    zip_path = WORK_ROOT / asset_name
    root = WORK_ROOT / f"source_{index}"
    log(f"[dataset {index}/{NUM_DATASETS}] {asset_name}")
    download_with_progress(f"{RELEASE_BASE}/{asset_name}", zip_path, asset_name)
    actual_sha = sha256_file(zip_path)
    expected_sha = DATASET_SHA256[asset_name]
    if actual_sha.lower() != expected_sha.lower():
        raise RuntimeError(f"SHA-256 mismatch for {asset_name}: expected {expected_sha}, got {actual_sha}")
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(root)
    source_roots.append(root)
    log(f"[dataset {index}/{NUM_DATASETS}] checksum OK; extracted to {root}")


stage(3, 8, "Index source annotations")
records = []
for source_index, root in enumerate(source_roots, 1):
    asset_name = DATASET_ASSETS[source_index - 1]
    json_paths = [str(path) for path in Path(root).rglob("*.json") if looks_like_coco(path)]
    if not json_paths:
        raise RuntimeError(f"No COCO JSON found in source {source_index}: {asset_name}")
    name_index = basename_index(root)
    before = len(records)
    annotation_count = 0
    for json_path in sorted(json_paths):
        with open(json_path, encoding="utf-8") as handle:
            coco = json.load(handle)
        categories = {
            category["id"]: str(category.get("name", "")).strip().lower()
            for category in coco["categories"]
        }
        allowed = {
            category_id for category_id, name in categories.items()
            if name in SOURCE_CATEGORY_NAMES
        } or set(categories)
        annotations_by_image = defaultdict(list)
        for annotation in coco["annotations"]:
            if annotation.get("category_id") in allowed:
                annotations_by_image[annotation["image_id"]].append(annotation)
        for image_meta in coco["images"]:
            annotations = annotations_by_image.get(image_meta["id"], [])
            if not annotations:
                continue
            image_path = resolve_image(root, json_path, image_meta["file_name"], name_index)
            if not image_path:
                continue
            annotation_count += len(annotations)
            records.append({
                "source": source_index,
                "source_asset": asset_name,
                "image_path": image_path,
                "file_name": image_meta["file_name"],
                "annotations": annotations,
            })
    log(
        f"[index {source_index}/{NUM_DATASETS}] {len(records) - before} annotated images | "
        f"{annotation_count} source annotations | {len(json_paths)} COCO JSON file(s)"
    )
log(f"[index] {len(records)} source image records total")


def box_from_mask(mask):
    ys, xs = np.where(mask > 0)
    if not len(xs):
        return np.array([0, 0, 0, 0], dtype=np.float32)
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)


def box_area(box):
    return max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))


def box_intersection_area(first, second):
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_iou(first, second):
    intersection = box_intersection_area(first, second)
    union = box_area(first) + box_area(second) - intersection
    return intersection / union if union else 0.0


def bbox_containment(candidate_box, old_box):
    area = box_area(candidate_box)
    return box_intersection_area(candidate_box, old_box) / area if area else 0.0


def normalized_center_distance(candidate_box, old_box):
    old_width = max(1.0, float(old_box[2] - old_box[0]))
    old_height = max(1.0, float(old_box[3] - old_box[1]))
    old_cx = (float(old_box[0]) + float(old_box[2])) / 2.0
    old_cy = (float(old_box[1]) + float(old_box[3])) / 2.0
    candidate_cx = (float(candidate_box[0]) + float(candidate_box[2])) / 2.0
    candidate_cy = (float(candidate_box[1]) + float(candidate_box[3])) / 2.0
    dx = (candidate_cx - old_cx) / old_width
    dy = (candidate_cy - old_cy) / old_height
    return float(np.sqrt(dx * dx + dy * dy))


def decode_old(annotation, height, width):
    segmentation = annotation.get("segmentation")
    if segmentation:
        try:
            if isinstance(segmentation, list):
                rle = mask_utils.merge(mask_utils.frPyObjects(segmentation, height, width))
            else:
                rle = dict(segmentation)
                if isinstance(rle.get("counts"), list):
                    rle = mask_utils.frPyObjects(rle, height, width)
                elif isinstance(rle.get("counts"), str):
                    rle["counts"] = rle["counts"].encode("ascii")
            mask = mask_utils.decode(rle)
            if mask.ndim == 3:
                mask = np.any(mask, axis=2)
            mask = (mask > 0).astype(np.uint8)
            if mask.any():
                return mask
        except Exception:
            pass
    mask = np.zeros((height, width), dtype=np.uint8)
    if "bbox" not in annotation:
        return mask
    x, y, box_width, box_height = map(float, annotation["bbox"])
    x1 = max(0, int(np.floor(x)))
    y1 = max(0, int(np.floor(y)))
    x2 = min(width, int(np.ceil(x + box_width)))
    y2 = min(height, int(np.ceil(y + box_height)))
    mask[y1:y2, x1:x2] = 1
    return mask


def comparison_metrics(old_mask, sam_mask):
    old = old_mask.astype(bool)
    sam = sam_mask.astype(bool)
    intersection = float(np.logical_and(old, sam).sum())
    old_area = float(old.sum())
    sam_area = float(sam.sum())
    union = old_area + sam_area - intersection
    old_box = box_from_mask(old_mask)
    sam_box = box_from_mask(sam_mask)
    old_box_area = max(1.0, box_area(old_box))
    return {
        "mask_iou": intersection / union if union else 0.0,
        "old_coverage": intersection / old_area if old_area else 0.0,
        "sam_coverage": intersection / sam_area if sam_area else 0.0,
        "bbox_iou": bbox_iou(old_box, sam_box),
        "bbox_containment": bbox_containment(sam_box, old_box),
        "center_distance": normalized_center_distance(sam_box, old_box),
        "candidate_to_old_bbox_area": box_area(sam_box) / old_box_area,
    }


def pair_is_plausible(metrics):
    contained = (
        metrics["sam_coverage"] >= MIN_SAM_COVERAGE
        or metrics["bbox_containment"] >= MIN_BBOX_CONTAINMENT
    )
    return (
        contained
        and metrics["center_distance"] <= MAX_NORMALIZED_CENTER_DISTANCE
        and metrics["candidate_to_old_bbox_area"] <= MAX_CANDIDATE_TO_OLD_BBOX_AREA
    )


def body_match_rank(metrics, score):
    containment = max(metrics["sam_coverage"], metrics["bbox_containment"])
    center_score = max(0.0, 1.0 - metrics["center_distance"])
    oversize = max(0.0, metrics["candidate_to_old_bbox_area"] - 1.0)
    return (
        0.65 * float(score)
        + 0.20 * containment
        + 0.10 * center_score
        + 0.05 * metrics["bbox_iou"]
        - 0.25 * oversize
    )


def mask_iou(first, second):
    first = first.astype(bool)
    second = second.astype(bool)
    intersection = np.logical_and(first, second).sum()
    union = np.logical_or(first, second).sum()
    return float(intersection / union) if union else 0.0


def dedup_instances(instances):
    kept = []
    for instance in sorted(instances, key=lambda item: item["rank"], reverse=True):
        if not any(mask_iou(instance["mask"], existing["mask"]) >= FINAL_INSTANCE_DUP_IOU for existing in kept):
            kept.append(instance)
    return kept


def select_body_assignments(old_entries, candidates):
    """Greedy one-to-one matching of text-prompt body masks to old COCO instances."""
    pairs = []
    for old_entry in old_entries:
        for candidate_index, candidate in enumerate(candidates):
            metrics = comparison_metrics(old_entry["mask"], candidate["mask"])
            if not pair_is_plausible(metrics):
                continue
            rank = body_match_rank(metrics, candidate["score"])
            pairs.append((rank, old_entry["old_index"], candidate_index, metrics))

    pairs.sort(key=lambda item: item[0], reverse=True)
    used_old_indices = set()
    used_candidate_indices = set()
    assignments = {}
    for rank, old_index, candidate_index, metrics in pairs:
        if old_index in used_old_indices or candidate_index in used_candidate_indices:
            continue
        used_old_indices.add(old_index)
        used_candidate_indices.add(candidate_index)
        candidate = candidates[candidate_index]
        assignments[old_index] = {
            **candidate,
            "candidate_index": candidate_index,
            "metrics": metrics,
            "rank": rank,
        }
    return assignments


def coco_rle(mask):
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


stage(4, 8, "Generate one SAM 3.1 resistor-body mask per source instance")
log(
    f"[sam3] processing {len(records)} images with text prompt {BODY_TEXT_PROMPT!r} | "
    f"explicit status every {PROGRESS_EVERY_IMAGES} images"
)
refined = []
manifest_rows = []
accepted_instances_total = 0
rejected_instances_total = 0
refine_started = time.monotonic()
progress = tqdm(total=len(records), desc="SAM 3.1 body refinement", unit="image", dynamic_ncols=True)

for record_index, record in enumerate(records, 1):
    image = Image.open(record["image_path"]).convert("RGB")
    width, height = image.size
    state = processor.set_image(image)
    output = processor.set_text_prompt(prompt=BODY_TEXT_PROMPT, state=state)

    masks_tensor = output["masks"]
    if masks_tensor.ndim == 4 and masks_tensor.shape[1] == 1:
        masks_tensor = masks_tensor.squeeze(1)
    masks = masks_tensor.detach().cpu().numpy().astype(np.uint8)
    scores = output["scores"].detach().float().cpu().numpy().astype(float)

    candidates = []
    for candidate_index, (sam_mask, sam_score) in enumerate(zip(masks, scores)):
        sam_score = float(sam_score)
        area_fraction = float(sam_mask.sum() / max(1, height * width))
        if (
            sam_score < MIN_SAM_SCORE
            or not sam_mask.any()
            or not (MIN_IMAGE_AREA_FRACTION <= area_fraction <= MAX_IMAGE_AREA_FRACTION)
        ):
            continue
        candidates.append({
            "mask": sam_mask,
            "score": sam_score,
            "text_detection_index": candidate_index,
        })

    old_entries = []
    for old_index, annotation in enumerate(record["annotations"]):
        old_mask = decode_old(annotation, height, width)
        if old_mask.any():
            old_entries.append({"old_index": old_index, "mask": old_mask})
        else:
            rejected_instances_total += 1
            manifest_rows.append({
                "source": record["source"],
                "source_asset": record["source_asset"],
                "image": record["file_name"],
                "old_instance": old_index,
                "accepted": False,
                "reason": "empty_source_annotation",
            })

    assignments = select_body_assignments(old_entries, candidates)
    accepted = []
    for old_entry in old_entries:
        old_index = old_entry["old_index"]
        best = assignments.get(old_index)
        if best is None:
            rejected_instances_total += 1
            manifest_rows.append({
                "source": record["source"],
                "source_asset": record["source_asset"],
                "image": record["file_name"],
                "old_instance": old_index,
                "accepted": False,
                "reason": "no_sam_body_match",
            })
            continue

        accepted.append(best)
        accepted_instances_total += 1
        manifest_rows.append({
            "source": record["source"],
            "source_asset": record["source_asset"],
            "image": record["file_name"],
            "old_instance": old_index,
            "accepted": True,
            "reason": "sam_body_selected",
            "sam_candidate": best["text_detection_index"],
            "sam_score": best["score"],
            "match_rank": best["rank"],
            **best["metrics"],
        })

    accepted = dedup_instances(accepted)
    if accepted:
        refined.append({
            **record,
            "hash": sha256_file(record["image_path"]),
            "instances": accepted,
        })

    del state, output
    progress.update(1)
    progress.set_postfix(
        kept_images=len(refined),
        accepted=accepted_instances_total,
        rejected=rejected_instances_total,
        refresh=False,
    )
    if record_index == 1 or record_index % PROGRESS_EVERY_IMAGES == 0 or record_index == len(records):
        elapsed = max(0.001, time.monotonic() - refine_started)
        rate = record_index / elapsed
        remaining = (len(records) - record_index) / rate if rate else 0.0
        log(
            f"[sam3] {record_index}/{len(records)} images "
            f"({100.0 * record_index / max(1, len(records)):.1f}%) | "
            f"kept {len(refined)} | accepted {accepted_instances_total} | "
            f"rejected {rejected_instances_total} | {rate:.2f} img/s | ETA {remaining / 60:.1f} min"
        )
    if record_index % 50 == 0:
        gc.collect()
        torch.cuda.empty_cache()

progress.close()
log(
    f"[sam3] complete | kept {len(refined)}/{len(records)} image records | "
    f"accepted {accepted_instances_total} body masks | rejected {rejected_instances_total} | "
    f"elapsed {(time.monotonic() - refine_started) / 60:.1f} min"
)


stage(5, 8, "Deduplicate images and split 80/10/10")
by_hash = defaultdict(list)
for record in refined:
    by_hash[record["hash"]].append(record)
unique = [
    max(group, key=lambda record: (len(record["instances"]), np.mean([item["rank"] for item in record["instances"]])))
    for group in by_hash.values()
]
if not unique:
    raise RuntimeError("No image survived SAM 3.1 body matching. Inspect refinement_manifest.csv before lowering thresholds.")
duplicates_removed = len(refined) - len(unique)
rng = random.Random(SPLIT_SEED)
rng.shuffle(unique)
count = len(unique)
train_count = min(count, int(round(count * TRAIN_FRACTION)))
valid_count = min(count - train_count, int(round(count * VALID_FRACTION)))
splits = {
    "train": unique[:train_count],
    "valid": unique[train_count:train_count + valid_count],
    "test": unique[train_count + valid_count:],
}
log(f"[dedup] {len(refined)} SAM-kept records -> {len(unique)} unique images | removed {duplicates_removed}")
for split_name, items in splits.items():
    log(f"[split] {split_name}: {len(items)} images | {sum(len(r['instances']) for r in items)} instances")


stage(6, 8, "Export merged COCO dataset, masks, and QA overlays")
shutil.rmtree(OUTPUT_ROOT, ignore_errors=True)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
summary = {}
for split_index, (split_name, items) in enumerate(splits.items(), 1):
    base = OUTPUT_ROOT / split_name
    image_dir = base / "images"
    mask_dir = base / "masks_semantic"
    overlay_dir = base / "overlays"
    for directory in (image_dir, mask_dir, overlay_dir):
        directory.mkdir(parents=True, exist_ok=True)

    coco_images = []
    coco_annotations = []
    annotation_id = 1
    export_progress = tqdm(items, total=len(items), desc=f"Export {split_name}", unit="image", dynamic_ncols=True)
    for image_id, record in enumerate(export_progress, 1):
        image = Image.open(record["image_path"]).convert("RGB")
        width, height = image.size
        suffix = Path(record["image_path"]).suffix.lower()
        extension = suffix if suffix in IMG_EXTS else ".jpg"
        output_name = f"{image_id:06d}_{record['hash'][:12]}{extension}"
        shutil.copy2(record["image_path"], image_dir / output_name)
        coco_images.append({
            "id": image_id,
            "file_name": f"images/{output_name}",
            "width": width,
            "height": height,
            "source_dataset": record["source"],
            "source_asset": record["source_asset"],
            "sha256": record["hash"],
        })
        semantic = np.zeros((height, width), dtype=np.uint8)
        overlay = np.array(image).copy()
        for instance in record["instances"]:
            mask = instance["mask"]
            semantic = np.maximum(semantic, mask)
            mask_bool = mask.astype(bool)
            overlay[mask_bool] = (0.55 * overlay[mask_bool] + 0.45 * 255).astype(np.uint8)
            rle = coco_rle(mask)
            binary_rle = {"size": rle["size"], "counts": rle["counts"].encode("ascii")}
            coco_annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,
                "segmentation": rle,
                "area": float(mask.sum()),
                "bbox": [float(value) for value in mask_utils.toBbox(binary_rle)],
                "iscrowd": 0,
                "sam3_prompt": BODY_TEXT_PROMPT,
                "sam3_score": instance["score"],
                "sam3_text_detection_index": instance["text_detection_index"],
                "old_sam_mask_iou": instance["metrics"]["mask_iou"],
                "old_coverage": instance["metrics"]["old_coverage"],
                "sam_coverage": instance["metrics"]["sam_coverage"],
                "old_sam_bbox_iou": instance["metrics"]["bbox_iou"],
                "bbox_containment": instance["metrics"]["bbox_containment"],
                "match_rank": instance["rank"],
            })
            annotation_id += 1
        stem = Path(output_name).stem
        Image.fromarray(semantic * 255).save(mask_dir / f"{stem}.png")
        Image.fromarray(overlay).save(overlay_dir / f"{stem}.jpg", quality=92)

    coco = {
        "info": {
            "description": (
                "Three r1 resistor COCO datasets merged after replacing source labels with "
                "one-to-one SAM 3.1 text-prompt 'resistor body' masks."
            ),
            "sam3_prompt": BODY_TEXT_PROMPT,
        },
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": [{"id": 1, "name": FINAL_CATEGORY_NAME, "supercategory": "electronic_component"}],
    }
    with open(base / "_annotations.coco.json", "w", encoding="utf-8") as handle:
        json.dump(coco, handle)
    summary[split_name] = {"images": len(coco_images), "instances": len(coco_annotations)}
    log(f"[export {split_index}/3] {split_name}: {len(coco_images)} images | {len(coco_annotations)} body masks")


stage(7, 8, "Write QA manifest and summary")
write_manifest_csv(OUTPUT_ROOT / "refinement_manifest.csv", manifest_rows)
with open(OUTPUT_ROOT / "summary.json", "w", encoding="utf-8") as handle:
    json.dump({
        "source_release": "r1",
        "source_assets": list(DATASET_ASSETS),
        "sam3_prompt": BODY_TEXT_PROMPT,
        "source_records": len(records),
        "survived_sam": len(refined),
        "final_unique_images": len(unique),
        "exact_duplicate_records_removed": duplicates_removed,
        "splits": summary,
        "selection": {
            "min_sam_score": MIN_SAM_SCORE,
            "min_sam_coverage": MIN_SAM_COVERAGE,
            "min_bbox_containment": MIN_BBOX_CONTAINMENT,
            "max_normalized_center_distance": MAX_NORMALIZED_CENTER_DISTANCE,
            "max_candidate_to_old_bbox_area": MAX_CANDIDATE_TO_OLD_BBOX_AREA,
            "one_candidate_per_source_instance": True,
            "one_source_instance_per_candidate": True,
        },
    }, handle, indent=2)
log("[export] summary")
log(json.dumps(summary, indent=2))


stage(8, 8, "Package and download")
zip_path = shutil.make_archive("/content/resistor_sam3_merged", "zip", OUTPUT_ROOT)
zip_size = Path(zip_path).stat().st_size / 1024 / 1024
log(f"[package] created {zip_path} ({zip_size:.1f} MiB)")
files.download(zip_path)
log("[done] body-only SAM 3.1 COCO merge finished")
