"""Colab workflow: refine three release COCO resistor datasets with SAM 3 and merge them.

Designed for current Google Colab Python 3.13 runtimes without changing the runtime's
NumPy installation. Upstream SAM 3 still constrains NumPy below 2, so we locally apply
its NumPy-2 compatibility change and install SAM 3 with --no-deps.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

SAM3_COLAB_VERSION = "2026-10-04-v3"
NUM_DATASETS = 3
FINAL_CATEGORY_NAME = "resistor"
SOURCE_CATEGORY_NAMES = {"resistor", "resistor_body", "resistors", "res"}

SAM_PROCESSOR_THRESHOLD = 0.15
MIN_SAM_SCORE = 0.20
MIN_MASK_IOU = 0.60
MIN_OLD_COVERAGE = 0.70
MIN_SAM_COVERAGE = 0.70
MIN_BBOX_IOU = 0.50
MIN_BOX_ONLY_BBOX_IOU = 0.55
FINAL_INSTANCE_DUP_IOU = 0.85
MIN_IMAGE_AREA_FRACTION = 0.00015
MAX_IMAGE_AREA_FRACTION = 0.80

TRAIN_FRACTION = 0.80
VALID_FRACTION = 0.10
TEST_FRACTION = 0.10
SPLIT_SEED = 1337
PROGRESS_EVERY_IMAGES = 25
DOWNLOAD_CHUNK_BYTES = 8 * 1024 * 1024

WORK_ROOT = Path("/content/sam3_merge_work")
OUTPUT_ROOT = Path("/content/resistor_sam3_merged")
SAM3_DIR = Path("/content/sam3")
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


def patch_sam3_numpy2_compat():
    """Keep Colab's NumPy 2.x instead of letting upstream SAM 3 downgrade it."""
    pyproject = SAM3_DIR / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    original = text
    text = text.replace('"numpy>=1.26,<2",', '"numpy>=1.26",')
    text = text.replace('"numpy==1.26",', '"numpy>=1.26",')
    if text != original:
        pyproject.write_text(text, encoding="utf-8")
        log("[setup] relaxed upstream NumPy <2 constraint for Colab Python 3.13")
    else:
        log("[setup] SAM 3 NumPy dependency already compatible; no constraint patch needed")

    visualizer = SAM3_DIR / "sam3" / "agent" / "helpers" / "visualizer.py"
    if visualizer.is_file():
        source = visualizer.read_text(encoding="utf-8")
        patched = re.sub(r"\bnp\.bool\b", "np.bool_", source)
        if patched != source:
            visualizer.write_text(patched, encoding="utf-8")
            log("[setup] patched deprecated np.bool usage to np.bool_")


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
        report_step = max(
            DOWNLOAD_CHUNK_BYTES,
            total // 10 if total else DOWNLOAD_CHUNK_BYTES,
        )
        next_report = report_step

        if total:
            log(f"[download] {label}: {total / 1024 / 1024:.1f} MiB")
        else:
            log(f"[download] {label}: size unknown")

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
                    percent = 100.0 * copied / total
                    log(
                        f"[download] {label}: "
                        f"{copied / 1024 / 1024:.1f}/{total / 1024 / 1024:.1f} MiB "
                        f"({percent:.0f}%) | {speed:.1f} MiB/s"
                    )
                else:
                    log(
                        f"[download] {label}: "
                        f"{copied / 1024 / 1024:.1f} MiB | {speed:.1f} MiB/s"
                    )
                next_report = copied + report_step

    elapsed = max(0.001, time.monotonic() - started)
    log(
        f"[download] {label}: complete | "
        f"{copied / 1024 / 1024:.1f} MiB in {elapsed:.1f}s"
    )


def looks_like_coco(path):
    try:
        with open(path, encoding="utf-8") as handle:
            obj = json.load(handle)
        return all(
            isinstance(obj.get(key), list)
            for key in ("images", "annotations", "categories")
        )
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
        "source",
        "source_asset",
        "image",
        "old_instance",
        "accepted",
        "reason",
        "sam_score",
        "mask_iou",
        "old_coverage",
        "sam_coverage",
        "bbox_iou",
        "match_rank",
    ]
    fields = [key for key in preferred if any(key in row for row in rows)]
    extras = sorted({key for row in rows for key in row} - set(fields))
    fields.extend(extras)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


stage(1, 8, "Environment and dependencies")
log(f"SAM 3 Colab version: {SAM3_COLAB_VERSION}")
log(f"[setup] Python: {sys.version.split()[0]}")
log("[setup] installing non-NumPy runtime dependencies")
run([
    sys.executable,
    "-m",
    "pip",
    "install",
    "-q",
    "huggingface_hub",
    "pycocotools",
    "tqdm",
    "timm>=1.0.17",
    "ftfy==6.1.1",
    "regex",
    "iopath>=0.1.10",
    "typing_extensions",
])

if not SAM3_DIR.is_dir():
    log("[setup] cloning official SAM 3 repository")
    run([
        "git",
        "clone",
        "--depth",
        "1",
        "https://github.com/facebookresearch/sam3.git",
        SAM3_DIR,
    ])
else:
    log("[setup] SAM 3 repository already present; updating")
    run(["git", "pull", "--ff-only"], cwd=SAM3_DIR)

log("[setup] applying NumPy-2 compatibility patch")
patch_sam3_numpy2_compat()
log("[setup] installing SAM 3 without dependency resolver touching NumPy")
run([
    sys.executable,
    "-m",
    "pip",
    "install",
    "-q",
    "-e",
    SAM3_DIR,
    "--no-deps",
])
if str(SAM3_DIR) not in sys.path:
    sys.path.insert(0, str(SAM3_DIR))

import numpy as np
import torch
from google.colab import files
from huggingface_hub import login, notebook_login
from PIL import Image
from pycocotools import mask as mask_utils
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model
from tqdm.auto import tqdm

log(f"[setup] NumPy kept at runtime version: {np.__version__}")
if not torch.cuda.is_available():
    raise RuntimeError(
        "No CUDA GPU detected. In Colab choose Runtime -> Change runtime type -> GPU."
    )

gpu_name = torch.cuda.get_device_name(0)
props = torch.cuda.get_device_properties(0)
log(f"[setup] GPU: {gpu_name} | VRAM: {props.total_memory / 1024**3:.1f} GiB")


stage(2, 8, "Hugging Face authentication")
token = None
try:
    from google.colab import userdata

    token = userdata.get("HF_TOKEN")
except Exception:
    token = None

if token:
    login(token=token, add_to_git_credential=False)
    log("[auth] authenticated from Colab secret HF_TOKEN")
else:
    log("[auth] HF_TOKEN secret not found; opening Hugging Face login")
    notebook_login()


stage(3, 8, "Download and extract the 3 release datasets")
shutil.rmtree(WORK_ROOT, ignore_errors=True)
WORK_ROOT.mkdir(parents=True, exist_ok=True)

if len(DATASET_ASSETS) != NUM_DATASETS:
    raise RuntimeError(
        f"Expected {NUM_DATASETS} release datasets, configured {len(DATASET_ASSETS)}."
    )

source_roots = []
for index, asset_name in enumerate(DATASET_ASSETS, 1):
    url = f"{RELEASE_BASE}/{asset_name}"
    zip_path = WORK_ROOT / asset_name
    root = WORK_ROOT / f"source_{index}"

    log(f"[dataset {index}/{NUM_DATASETS}] {asset_name}")
    download_with_progress(url, zip_path, asset_name)

    log(f"[dataset {index}/{NUM_DATASETS}] verifying SHA-256")
    actual_sha = sha256_file(zip_path)
    expected_sha = DATASET_SHA256[asset_name]
    if actual_sha.lower() != expected_sha.lower():
        raise RuntimeError(
            f"SHA-256 mismatch for {asset_name}: "
            f"expected {expected_sha}, got {actual_sha}"
        )
    log(f"[dataset {index}/{NUM_DATASETS}] checksum OK")

    root.mkdir(parents=True, exist_ok=True)
    log(f"[dataset {index}/{NUM_DATASETS}] extracting")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(root)
    source_roots.append(root)
    log(f"[dataset {index}/{NUM_DATASETS}] ready: {root}")

log("[datasets] all 3 release datasets are ready")


stage(4, 8, "Index annotated COCO images")
records = []
for source_index, root in enumerate(source_roots, 1):
    asset_name = DATASET_ASSETS[source_index - 1]
    log(f"[index {source_index}/{NUM_DATASETS}] scanning {asset_name}")
    json_paths = [
        str(path)
        for path in Path(root).rglob("*.json")
        if looks_like_coco(path)
    ]
    if not json_paths:
        raise RuntimeError(f"No COCO JSON found in source {source_index}.")

    name_index = basename_index(root)
    before = len(records)
    source_annotations = 0

    for json_path in sorted(json_paths):
        with open(json_path, encoding="utf-8") as handle:
            coco = json.load(handle)

        categories = {
            category["id"]: str(category.get("name", "")).strip().lower()
            for category in coco["categories"]
        }
        allowed = {
            category_id
            for category_id, name in categories.items()
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
            image_path = resolve_image(
                root,
                json_path,
                image_meta["file_name"],
                name_index,
            )
            if image_path:
                source_annotations += len(annotations)
                records.append({
                    "source": source_index,
                    "source_asset": asset_name,
                    "image_path": image_path,
                    "file_name": image_meta["file_name"],
                    "annotations": annotations,
                })

    source_images = len(records) - before
    log(
        f"[index {source_index}/{NUM_DATASETS}] "
        f"{source_images} annotated images | "
        f"{source_annotations} source annotations | "
        f"{len(json_paths)} COCO JSON file(s)"
    )

log(
    f"[index] complete | {len(records)} annotated source image records | "
    f"{sum(len(record['annotations']) for record in records)} total source annotations"
)


def box_from_mask(mask):
    ys, xs = np.where(mask > 0)
    if not len(xs):
        return np.array([0, 0, 0, 0], dtype=np.float32)
    return np.array(
        [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1],
        dtype=np.float32,
    )


def bbox_iou(first, second):
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, float(first[2] - first[0])) * max(
        0.0, float(first[3] - first[1])
    )
    second_area = max(0.0, float(second[2] - second[0])) * max(
        0.0, float(second[3] - second[1])
    )
    union = first_area + second_area - intersection
    return float(intersection / union) if union else 0.0


def decode_old(annotation, height, width):
    segmentation = annotation.get("segmentation")
    if segmentation:
        try:
            if isinstance(segmentation, list):
                rle = mask_utils.merge(
                    mask_utils.frPyObjects(segmentation, height, width)
                )
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
                return mask, False
        except Exception:
            pass

    mask = np.zeros((height, width), dtype=np.uint8)
    if "bbox" not in annotation:
        return mask, True

    x, y, box_width, box_height = map(float, annotation["bbox"])
    x1 = max(0, int(np.floor(x)))
    y1 = max(0, int(np.floor(y)))
    x2 = min(width, int(np.ceil(x + box_width)))
    y2 = min(height, int(np.ceil(y + box_height)))
    mask[y1:y2, x1:x2] = 1
    return mask, True


def comparison_metrics(old_mask, sam_mask):
    old = old_mask.astype(bool)
    sam = sam_mask.astype(bool)
    intersection = float(np.logical_and(old, sam).sum())
    old_area = float(old.sum())
    sam_area = float(sam.sum())
    union = old_area + sam_area - intersection
    return {
        "mask_iou": intersection / union if union else 0.0,
        "old_coverage": intersection / old_area if old_area else 0.0,
        "sam_coverage": intersection / sam_area if sam_area else 0.0,
        "bbox_iou": bbox_iou(box_from_mask(old_mask), box_from_mask(sam_mask)),
    }


def candidate_matches(metrics, bbox_only):
    if bbox_only:
        return metrics["bbox_iou"] >= MIN_BOX_ONLY_BBOX_IOU
    return (
        metrics["mask_iou"] >= MIN_MASK_IOU
        or (
            metrics["old_coverage"] >= MIN_OLD_COVERAGE
            and metrics["sam_coverage"] >= MIN_SAM_COVERAGE
            and metrics["bbox_iou"] >= MIN_BBOX_IOU
        )
    )


def candidate_rank(metrics, score, bbox_only):
    if bbox_only:
        return 0.8 * metrics["bbox_iou"] + 0.2 * score
    return (
        0.5 * metrics["mask_iou"]
        + 0.2 * min(metrics["old_coverage"], metrics["sam_coverage"])
        + 0.15 * metrics["bbox_iou"]
        + 0.15 * score
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
        if not any(
            mask_iou(instance["mask"], existing["mask"]) >= FINAL_INSTANCE_DUP_IOU
            for existing in kept
        ):
            kept.append(instance)
    return kept


def coco_rle(mask):
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


stage(5, 8, "Load SAM 3 and refine annotations")
log("[sam3] building image model; this can be quiet for a while")
model_started = time.monotonic()
model = build_sam3_image_model()
model.eval()
processor = Sam3Processor(
    model=model,
    device="cuda",
    confidence_threshold=SAM_PROCESSOR_THRESHOLD,
)
log(f"[sam3] model ready in {time.monotonic() - model_started:.1f}s")
log(
    f"[sam3] processing {len(records)} images | "
    f"explicit status every {PROGRESS_EVERY_IMAGES} images"
)

refined = []
manifest_rows = []
accepted_instances_total = 0
rejected_instances_total = 0
refine_started = time.monotonic()

progress = tqdm(
    total=len(records),
    desc="SAM 3 refinement",
    unit="image",
    dynamic_ncols=True,
)

for record_index, record in enumerate(records, 1):
    image = Image.open(record["image_path"]).convert("RGB")
    width, height = image.size
    state = processor.set_image(image)
    accepted = []

    for old_index, annotation in enumerate(record["annotations"]):
        old_mask, bbox_only = decode_old(annotation, height, width)
        if not old_mask.any():
            rejected_instances_total += 1
            manifest_rows.append({
                "source": record["source"],
                "source_asset": record["source_asset"],
                "image": record["file_name"],
                "old_instance": old_index,
                "accepted": False,
                "reason": "empty_source_annotation",
            })
            continue

        x1, y1, x2, y2 = box_from_mask(old_mask)
        prompt_box = [
            ((x1 + x2) / 2) / width,
            ((y1 + y2) / 2) / height,
            max(1.0, x2 - x1) / width,
            max(1.0, y2 - y1) / height,
        ]

        processor.reset_all_prompts(state)
        output = processor.add_geometric_prompt(
            box=prompt_box,
            label=True,
            state=state,
        )

        masks = (
            output["masks"]
            .squeeze(1)
            .detach()
            .cpu()
            .numpy()
            .astype(np.uint8)
        )
        scores = output["scores"].detach().cpu().numpy().astype(float)

        candidates = []
        for sam_mask, sam_score in zip(masks, scores):
            sam_score = float(sam_score)
            area_fraction = float(sam_mask.sum() / max(1, height * width))
            if (
                sam_score < MIN_SAM_SCORE
                or not sam_mask.any()
                or not (
                    MIN_IMAGE_AREA_FRACTION
                    <= area_fraction
                    <= MAX_IMAGE_AREA_FRACTION
                )
            ):
                continue

            metrics = comparison_metrics(old_mask, sam_mask)
            candidates.append({
                "mask": sam_mask,
                "score": sam_score,
                "metrics": metrics,
                "rank": candidate_rank(metrics, sam_score, bbox_only),
            })

        matches = [
            candidate
            for candidate in candidates
            if candidate_matches(candidate["metrics"], bbox_only)
        ]

        if matches:
            best = max(matches, key=lambda candidate: candidate["rank"])
            accepted.append(best)
            accepted_instances_total += 1
            manifest_rows.append({
                "source": record["source"],
                "source_asset": record["source_asset"],
                "image": record["file_name"],
                "old_instance": old_index,
                "accepted": True,
                **best["metrics"],
                "sam_score": best["score"],
                "match_rank": best["rank"],
            })
        else:
            rejected_instances_total += 1
            best = max(
                candidates,
                key=lambda candidate: candidate["rank"],
                default=None,
            )
            row = {
                "source": record["source"],
                "source_asset": record["source_asset"],
                "image": record["file_name"],
                "old_instance": old_index,
                "accepted": False,
                "reason": "no_sam_match",
            }
            if best:
                row.update(best["metrics"])
                row.update(
                    sam_score=best["score"],
                    match_rank=best["rank"],
                )
            manifest_rows.append(row)

    accepted = dedup_instances(accepted)
    if accepted:
        refined.append({
            **record,
            "hash": sha256_file(record["image_path"]),
            "instances": accepted,
        })

    del state
    progress.update(1)
    progress.set_postfix(
        kept_images=len(refined),
        accepted=accepted_instances_total,
        rejected=rejected_instances_total,
        refresh=False,
    )

    if (
        record_index == 1
        or record_index % PROGRESS_EVERY_IMAGES == 0
        or record_index == len(records)
    ):
        elapsed = max(0.001, time.monotonic() - refine_started)
        rate = record_index / elapsed
        remaining = (len(records) - record_index) / rate if rate else 0.0
        log(
            f"[sam3] {record_index}/{len(records)} images "
            f"({100.0 * record_index / max(1, len(records)):.1f}%) "
            f"| kept images {len(refined)} "
            f"| accepted instances {accepted_instances_total} "
            f"| rejected instances {rejected_instances_total} "
            f"| {rate:.2f} img/s | ETA {remaining / 60:.1f} min"
        )

    if record_index % 50 == 0:
        gc.collect()
        torch.cuda.empty_cache()

progress.close()
log(
    f"[sam3] complete | kept {len(refined)}/{len(records)} image records | "
    f"accepted {accepted_instances_total} instances | "
    f"rejected {rejected_instances_total} instances | "
    f"elapsed {(time.monotonic() - refine_started) / 60:.1f} min"
)


stage(6, 8, "Deduplicate and split final images")
by_hash = defaultdict(list)
for record in refined:
    by_hash[record["hash"]].append(record)

unique = []
for group in by_hash.values():
    unique.append(
        max(
            group,
            key=lambda record: (
                len(record["instances"]),
                np.mean([item["rank"] for item in record["instances"]]),
            ),
        )
    )

if not unique:
    raise RuntimeError(
        "No image survived. Lower thresholds only after checking why matching failed."
    )

duplicates_removed = len(refined) - len(unique)
log(
    f"[dedup] {len(refined)} SAM-kept records -> {len(unique)} unique images "
    f"| removed {duplicates_removed} exact duplicate record(s)"
)

rng = random.Random(SPLIT_SEED)
rng.shuffle(unique)
count = len(unique)
train_count = min(count, int(round(count * TRAIN_FRACTION)))
valid_count = min(
    count - train_count,
    int(round(count * VALID_FRACTION)),
)
splits = {
    "train": unique[:train_count],
    "valid": unique[train_count:train_count + valid_count],
    "test": unique[train_count + valid_count:],
}

for split_name, items in splits.items():
    log(
        f"[split] {split_name}: {len(items)} images | "
        f"{sum(len(record['instances']) for record in items)} instances"
    )


stage(7, 8, "Export merged COCO dataset, masks, and QA overlays")
shutil.rmtree(OUTPUT_ROOT, ignore_errors=True)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
summary = {}

for split_index, (split_name, items) in enumerate(splits.items(), 1):
    log(f"[export {split_index}/3] {split_name}: starting {len(items)} images")
    base = OUTPUT_ROOT / split_name
    image_dir = base / "images"
    mask_dir = base / "masks_semantic"
    overlay_dir = base / "overlays"
    for directory in (image_dir, mask_dir, overlay_dir):
        directory.mkdir(parents=True, exist_ok=True)

    coco_images = []
    coco_annotations = []
    annotation_id = 1

    export_progress = tqdm(
        items,
        total=len(items),
        desc=f"Export {split_name}",
        unit="image",
        dynamic_ncols=True,
    )

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
            overlay[mask_bool] = (
                0.55 * overlay[mask_bool] + 0.45 * 255
            ).astype(np.uint8)

            rle = coco_rle(mask)
            binary_rle = {
                "size": rle["size"],
                "counts": rle["counts"].encode("ascii"),
            }
            coco_annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,
                "segmentation": rle,
                "area": float(mask.sum()),
                "bbox": [
                    float(value)
                    for value in mask_utils.toBbox(binary_rle)
                ],
                "iscrowd": 0,
                "sam3_score": instance["score"],
                "old_sam_mask_iou": instance["metrics"]["mask_iou"],
                "old_coverage": instance["metrics"]["old_coverage"],
                "sam_coverage": instance["metrics"]["sam_coverage"],
                "old_sam_bbox_iou": instance["metrics"]["bbox_iou"],
                "match_rank": instance["rank"],
            })
            annotation_id += 1

        stem = Path(output_name).stem
        Image.fromarray(semantic * 255).save(mask_dir / f"{stem}.png")
        Image.fromarray(overlay).save(
            overlay_dir / f"{stem}.jpg",
            quality=92,
        )

    coco = {
        "info": {
            "description": (
                "3 r1 release resistor datasets refined/merged with SAM 3; "
                "final labels are accepted SAM 3 masks only"
            )
        },
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": [{
            "id": 1,
            "name": FINAL_CATEGORY_NAME,
            "supercategory": "electronic_component",
        }],
    }
    with open(base / "_annotations.coco.json", "w", encoding="utf-8") as handle:
        json.dump(coco, handle)

    summary[split_name] = {
        "images": len(coco_images),
        "instances": len(coco_annotations),
    }
    log(
        f"[export {split_index}/3] {split_name}: complete | "
        f"{len(coco_images)} images | {len(coco_annotations)} instances"
    )

write_manifest_csv(
    OUTPUT_ROOT / "refinement_manifest.csv",
    manifest_rows,
)
with open(OUTPUT_ROOT / "summary.json", "w", encoding="utf-8") as handle:
    json.dump({
        "source_release": "r1",
        "source_assets": list(DATASET_ASSETS),
        "source_records": len(records),
        "survived_sam": len(refined),
        "final_unique_images": len(unique),
        "exact_duplicate_records_removed": duplicates_removed,
        "splits": summary,
        "thresholds": {
            "mask_iou": MIN_MASK_IOU,
            "old_coverage": MIN_OLD_COVERAGE,
            "sam_coverage": MIN_SAM_COVERAGE,
            "bbox_iou": MIN_BBOX_IOU,
        },
    }, handle, indent=2)

log("[export] summary")
log(json.dumps(summary, indent=2))


stage(8, 8, "Package and download")
zip_path = shutil.make_archive(
    "/content/resistor_sam3_merged",
    "zip",
    OUTPUT_ROOT,
)
zip_size = Path(zip_path).stat().st_size / 1024 / 1024
log(f"[package] created {zip_path} ({zip_size:.1f} MiB)")
log("[package] starting browser download")
files.download(zip_path)
log("[done] SAM 3 refinement + merge workflow finished")
