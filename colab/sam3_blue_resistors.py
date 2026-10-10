"""Colab workflow: rebuild the Blue Resistors r1 COCO dataset with SAM 3.1.

The source COCO JSON files are used only to locate images. Existing labels, boxes,
and segmentations are ignored. SAM 3.1 runs the default text prompt
``blue axial resistor body`` once per image, selecting the highest-confidence
nonempty mask if the result reaches the configurable score threshold (60% by
default). At most one resistor-body mask is retained per image.

Inference is checkpointed to an independent Google Drive folder every 25 images,
and resumed automatically. The output is a new 80/10/10 COCO segmentation dataset,
per-image PNG masks, QA overlays, a manifest, and a ZIP copied to Drive.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

SAM3_COLAB_VERSION = "2026-10-10-v1-blue-sam3"
BODY_TEXT_PROMPT = os.environ.get("SAM3_BLUE_PROMPT", "blue axial resistor body").strip()
MIN_ACCEPT_SCORE = float(os.environ.get("SAM3_BLUE_MIN_SCORE", "0.60"))
if not BODY_TEXT_PROMPT or not (0.0 <= MIN_ACCEPT_SCORE <= 1.0):
    raise ValueError("SAM3_BLUE_PROMPT must be nonempty and SAM3_BLUE_MIN_SCORE must be in [0, 1]")
NUM_DATASETS = 1
FINAL_CATEGORY_NAME = "resistor"

TRAIN_FRACTION = 0.80
VALID_FRACTION = 0.10
SPLIT_SEED = 1337
PROGRESS_EVERY_IMAGES = 25
CHECKPOINT_EVERY_IMAGES = 25
DOWNLOAD_CHUNK_BYTES = 8 * 1024 * 1024

WORK_ROOT = Path("/content/sam3_blue_resistors_work")
OUTPUT_ROOT = Path("/content/blue_resistors_sam3")
LOCAL_CHECKPOINT_MASK_DIR = Path("/content/sam3_blue_checkpoint_masks")
DRIVE_CHECKPOINT_DIR = Path("/content/drive/MyDrive/resistor_sam3_blue_checkpoint")
DRIVE_CHECKPOINT_MASK_DIR = DRIVE_CHECKPOINT_DIR / "checkpoint_masks"
DRIVE_CHECKPOINT_JSON = DRIVE_CHECKPOINT_DIR / "checkpoint.json"
DRIVE_FINAL_ZIP = DRIVE_CHECKPOINT_DIR / "blue_resistors_sam3.zip"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

RELEASE_BASE = "https://github.com/Persie0/resistor_model/releases/download/r1"
DATASET_ASSETS = (
    "Blue.Resistors.v1i.coco.zip",
)
DATASET_SHA256 = {
    "Blue.Resistors.v1i.coco.zip": "3ad8126de1308b3109734414f9b1836b427f75eae1a62bf770c19c61a44c551c",
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
    """Reuse the notebook bootstrap model, or bootstrap SAM 3.1 standalone."""
    if globals().get("SAM3_BOOTSTRAPPED", False) and globals().get("processor") is not None:
        log("[sam3] reusing already-loaded SAM 3.1 image model")
        return

    log("[sam3] no loaded model found; running shared SAM 3.1 bootstrap")
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


def stable_key_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


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
        return isinstance(obj.get("images"), list)
    except Exception:
        return False


def basename_index(root):
    output = defaultdict(list)
    for path in Path(root).rglob("*"):
        if path.is_file() and path.suffix.lower() in IMG_EXTS:
            output[path.name.lower()].append(path)
    return output


def resolve_image(root, json_path, file_name, index):
    raw_name = str(file_name).replace("\\", "/")
    candidates = (
        Path(json_path).parent / raw_name,
        Path(root) / raw_name,
        Path(json_path).parent / Path(raw_name).name,
    )
    for path in candidates:
        if path.is_file():
            return str(path.resolve())

    hits = index.get(Path(raw_name).name.lower(), [])
    if len(hits) == 1:
        return str(hits[0].resolve())

    target = raw_name.lower().lstrip("./")
    suffix_hits = []
    for path in hits:
        try:
            rel = path.relative_to(root).as_posix().lower()
        except ValueError:
            rel = path.as_posix().lower()
        if rel.endswith(target):
            suffix_hits.append(path)
    if len(suffix_hits) == 1:
        return str(suffix_hits[0].resolve())
    return None


def coco_rle(mask):
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def checkpoint_mask_path(mask_filename):
    local = LOCAL_CHECKPOINT_MASK_DIR / mask_filename
    if local.is_file():
        return local
    drive_path = DRIVE_CHECKPOINT_MASK_DIR / mask_filename
    if drive_path.is_file():
        return drive_path
    raise FileNotFoundError(f"Checkpoint mask missing: {mask_filename}")


def load_checkpoint(valid_keys):
    if not DRIVE_CHECKPOINT_JSON.is_file():
        log("[checkpoint] no Drive checkpoint found; starting fresh")
        return set(), [], 0

    try:
        state = json.loads(DRIVE_CHECKPOINT_JSON.read_text(encoding="utf-8"))
    except Exception as exc:
        log(f"[checkpoint] could not read checkpoint; starting fresh: {exc}")
        return set(), [], 0

    if (
        state.get("workflow_version") != SAM3_COLAB_VERSION
        or state.get("prompt") != BODY_TEXT_PROMPT
        or state.get("min_accept_score") != MIN_ACCEPT_SCORE
    ):
        log("[checkpoint] existing checkpoint uses different prompt/threshold/version; starting fresh")
        return set(), [], 0

    processed_keys = set(state.get("processed_keys", [])) & valid_keys
    accepted_records = []
    accepted_keys = set()
    for item in state.get("accepted_records", []):
        key = item.get("image_key")
        mask_filename = item.get("mask_filename")
        if key not in valid_keys or not mask_filename:
            continue
        if (DRIVE_CHECKPOINT_MASK_DIR / mask_filename).is_file():
            accepted_records.append(item)
            accepted_keys.add(key)
        else:
            processed_keys.discard(key)

    rejected_count = max(0, len(processed_keys - accepted_keys))
    log(
        f"[checkpoint] resumed {len(processed_keys)} processed images | "
        f"{len(accepted_records)} accepted | {rejected_count} rejected"
    )
    return processed_keys, accepted_records, rejected_count


def save_checkpoint(processed_keys, accepted_records, pending_mask_names):
    DRIVE_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    DRIVE_CHECKPOINT_MASK_DIR.mkdir(parents=True, exist_ok=True)

    for mask_filename in sorted(pending_mask_names):
        source = LOCAL_CHECKPOINT_MASK_DIR / mask_filename
        destination = DRIVE_CHECKPOINT_MASK_DIR / mask_filename
        if source.is_file():
            shutil.copy2(source, destination)

    state = {
        "workflow_version": SAM3_COLAB_VERSION,
        "prompt": BODY_TEXT_PROMPT,
        "min_accept_score": MIN_ACCEPT_SCORE,
        "processed_keys": sorted(processed_keys),
        "accepted_records": accepted_records,
        "processed_count": len(processed_keys),
        "accepted_count": len(accepted_records),
        "rejected_count": len(processed_keys) - len(accepted_records),
        "saved_at_unix": time.time(),
    }
    temp_path = DRIVE_CHECKPOINT_DIR / "checkpoint.tmp.json"
    temp_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    temp_path.replace(DRIVE_CHECKPOINT_JSON)
    pending_mask_names.clear()
    log(
        f"[checkpoint] saved to Drive | processed {len(processed_keys)} | "
        f"accepted {len(accepted_records)} | rejected {state['rejected_count']}"
    )


def write_manifest_csv(path, records, processed_keys):
    rows = []
    accepted_by_key = {record["image_key"]: record for record in records}
    for image_key in sorted(processed_keys):
        accepted = accepted_by_key.get(image_key)
        if accepted:
            rows.append({
                "image_key": image_key,
                "source": accepted["source"],
                "source_asset": accepted["source_asset"],
                "image": accepted["file_name"],
                "accepted": True,
                "reason": "sam_top_result",
                "sam_candidate": accepted["text_detection_index"],
                "sam_score": accepted["score"],
            })
        else:
            rows.append({
                "image_key": image_key,
                "accepted": False,
                "reason": "below_threshold_or_no_sam_result",
            })
    fields = [
        "image_key", "source", "source_asset", "image", "accepted", "reason",
        "sam_candidate", "sam_score",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


stage(1, 8, "SAM 3.1 environment and Google Drive checkpoint")
log(f"SAM 3.1 Blue Resistors COCO rebuild version: {SAM3_COLAB_VERSION}")
ensure_sam3_ready()
run([sys.executable, "-m", "pip", "install", "-q", "pycocotools"])

import numpy as np
import torch
from google.colab import drive, files
from PIL import Image
from pycocotools import mask as mask_utils
from tqdm.auto import tqdm

if not torch.cuda.is_available():
    raise RuntimeError("No CUDA GPU detected. In Colab choose Runtime -> Change runtime type -> GPU.")

log("[checkpoint] mounting Google Drive")
drive.mount("/content/drive")
DRIVE_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
DRIVE_CHECKPOINT_MASK_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_CHECKPOINT_MASK_DIR.mkdir(parents=True, exist_ok=True)
log(f"[checkpoint] Drive folder: {DRIVE_CHECKPOINT_DIR}")
log(f"[checkpoint] cadence: every {CHECKPOINT_EVERY_IMAGES} processed images")
log(f"[sam3] prompt: {BODY_TEXT_PROMPT!r}")
log(f"[sam3] minimum acceptance score: {MIN_ACCEPT_SCORE:.0%}")
log(f"[sam3] GPU: {torch.cuda.get_device_name(0)}")


stage(2, 8, "Download and verify the Blue Resistors COCO ZIP")
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


stage(3, 8, "Index source COCO images (annotations ignored)")
records = []
seen_keys = set()
for source_index, root in enumerate(source_roots, 1):
    asset_name = DATASET_ASSETS[source_index - 1]
    json_paths = [str(path) for path in Path(root).rglob("*.json") if looks_like_coco(path)]
    if not json_paths:
        raise RuntimeError(f"No COCO JSON with images found in source {source_index}: {asset_name}")

    name_index = basename_index(root)
    before = len(records)
    unresolved = 0
    duplicate_refs = 0
    for json_path in sorted(json_paths):
        with open(json_path, encoding="utf-8") as handle:
            coco = json.load(handle)
        for image_meta in coco["images"]:
            image_path = resolve_image(root, json_path, image_meta["file_name"], name_index)
            if not image_path:
                unresolved += 1
                continue
            try:
                relative_path = Path(image_path).relative_to(root.resolve()).as_posix()
            except ValueError:
                relative_path = Path(image_path).name
            image_key = f"{source_index}:{relative_path}"
            if image_key in seen_keys:
                duplicate_refs += 1
                continue
            seen_keys.add(image_key)
            records.append({
                "image_key": image_key,
                "source": source_index,
                "source_asset": asset_name,
                "image_path": image_path,
                "file_name": image_meta["file_name"],
            })
    log(
        f"[index {source_index}/{NUM_DATASETS}] {len(records) - before} unique images | "
        f"annotations ignored | unresolved {unresolved} | duplicate refs {duplicate_refs} | "
        f"{len(json_paths)} COCO JSON file(s)"
    )

if not records:
    raise RuntimeError("No source images could be resolved from the Blue Resistors COCO dataset.")
log(f"[index] {len(records)} unique source images total")

record_by_key = {record["image_key"]: record for record in records}
valid_keys = set(record_by_key)
processed_keys, accepted_records, rejected_count = load_checkpoint(valid_keys)
accepted_by_key = {record["image_key"]: record for record in accepted_records}

for image_key, accepted in list(accepted_by_key.items()):
    current = record_by_key[image_key]
    accepted["image_path"] = current["image_path"]
    accepted["file_name"] = current["file_name"]
    accepted["source"] = current["source"]
    accepted["source_asset"] = current["source_asset"]


stage(4, 8, "Infer the best blue resistor body mask per image")
log(
    f"[sam3] source annotations/categories are ignored | {len(records)} images | "
    f"{len(processed_keys)} already checkpointed"
)
refine_started = time.monotonic()
pending_mask_names = set()
processed_this_run = 0
progress = tqdm(total=len(records), initial=len(processed_keys), desc="SAM 3.1 axial body mask", unit="image", dynamic_ncols=True)

for record_index, record in enumerate(records, 1):
    image_key = record["image_key"]
    if image_key in processed_keys:
        continue

    image = Image.open(record["image_path"]).convert("RGB")
    with torch.inference_mode():
        state = processor.set_image(image)
        output = processor.set_text_prompt(prompt=BODY_TEXT_PROMPT, state=state)

    masks_tensor = output["masks"]
    if masks_tensor.ndim == 4 and masks_tensor.shape[1] == 1:
        masks_tensor = masks_tensor.squeeze(1)
    masks = masks_tensor.detach().cpu().numpy().astype(np.uint8)
    scores = output["scores"].detach().float().cpu().numpy().astype(float)

    accepted = None
    if len(scores) > 0:
        # Empty masks cannot be exported, even when their confidence is highest.
        nonempty = masks.reshape(len(scores), -1).any(axis=1)
        ranked_scores = np.where(nonempty, scores, -np.inf)
        best_candidate_index = int(np.argmax(ranked_scores))
        best_score = float(ranked_scores[best_candidate_index])
        best_mask = masks[best_candidate_index]
        if best_score >= MIN_ACCEPT_SCORE and best_mask.any():
            mask_filename = f"{stable_key_hash(image_key)}.png"
            Image.fromarray(best_mask * 255).save(LOCAL_CHECKPOINT_MASK_DIR / mask_filename)
            pending_mask_names.add(mask_filename)
            accepted = {
                "image_key": image_key,
                "source": record["source"],
                "source_asset": record["source_asset"],
                "image_path": record["image_path"],
                "file_name": record["file_name"],
                "mask_filename": mask_filename,
                "score": best_score,
                "text_detection_index": best_candidate_index,
                "reason": "sam_top_result",
            }
            accepted_records.append(accepted)
            accepted_by_key[image_key] = accepted
        else:
            rejected_count += 1
    else:
        rejected_count += 1

    processed_keys.add(image_key)
    processed_this_run += 1
    del state, output
    progress.update(1)
    progress.set_postfix(
        accepted=len(accepted_records),
        rejected=rejected_count,
        checkpointed=len(processed_keys) - processed_this_run % CHECKPOINT_EVERY_IMAGES,
        refresh=False,
    )

    if processed_this_run % CHECKPOINT_EVERY_IMAGES == 0:
        save_checkpoint(processed_keys, accepted_records, pending_mask_names)

    total_done = len(processed_keys)
    if (
        processed_this_run == 1
        or processed_this_run % PROGRESS_EVERY_IMAGES == 0
        or total_done == len(records)
    ):
        elapsed = max(0.001, time.monotonic() - refine_started)
        rate = processed_this_run / elapsed if processed_this_run else 0.0
        remaining_new = len(records) - total_done
        eta = remaining_new / rate if rate else 0.0
        log(
            f"[sam3] {total_done}/{len(records)} images "
            f"({100.0 * total_done / len(records):.1f}%) | accepted {len(accepted_records)} | "
            f"rejected {rejected_count} | {rate:.2f} new img/s | ETA {eta / 60:.1f} min"
        )

save_checkpoint(processed_keys, accepted_records, pending_mask_names)
progress.close()
log(
    f"[sam3] inference complete | processed {len(processed_keys)}/{len(records)} | "
    f"accepted {len(accepted_records)} | rejected {rejected_count}"
)


stage(5, 8, "Deduplicate accepted images and split 80/10/10")
by_hash = defaultdict(list)
for record in accepted_records:
    if "hash" not in record:
        record["hash"] = sha256_file(record["image_path"])
    by_hash[record["hash"]].append(record)

unique = [max(group, key=lambda item: item["score"]) for group in by_hash.values()]
if not unique:
    raise RuntimeError(f"No SAM 3.1 blue resistor body mask met the {MIN_ACCEPT_SCORE:.0%} threshold.")

duplicates_removed = len(accepted_records) - len(unique)
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
log(f"[dedup] {len(accepted_records)} accepted -> {len(unique)} unique images | removed {duplicates_removed}")
for split_name, items in splits.items():
    log(f"[split] {split_name}: {len(items)} images / masks")


stage(6, 8, "Export Blue Resistors COCO dataset, masks, and QA overlays")
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
    export_progress = tqdm(items, total=len(items), desc=f"Export {split_name}", unit="image", dynamic_ncols=True)
    for image_id, record in enumerate(export_progress, 1):
        image = Image.open(record["image_path"]).convert("RGB")
        width, height = image.size
        mask = (np.array(Image.open(checkpoint_mask_path(record["mask_filename"])).convert("L")) > 0).astype(np.uint8)
        if mask.shape != (height, width):
            raise RuntimeError(
                f"Checkpoint mask shape {mask.shape} does not match image {(height, width)} for {record['image_key']}"
            )

        suffix = Path(record["image_path"]).suffix.lower()
        extension = suffix if suffix in IMG_EXTS else ".jpg"
        output_name = f"{image_id:06d}_{record['hash'][:12]}{extension}"
        shutil.copy2(record["image_path"], image_dir / output_name)
        stem = Path(output_name).stem
        Image.fromarray(mask * 255).save(mask_dir / f"{stem}.png")

        overlay = np.array(image).copy()
        mask_bool = mask.astype(bool)
        overlay[mask_bool] = (0.55 * overlay[mask_bool] + 0.45 * 255).astype(np.uint8)
        Image.fromarray(overlay).save(overlay_dir / f"{stem}.jpg", quality=92)

        coco_images.append({
            "id": image_id,
            "file_name": f"images/{output_name}",
            "width": width,
            "height": height,
            "source_dataset": record["source"],
            "source_asset": record["source_asset"],
            "source_annotations_ignored": True,
            "sha256": record["hash"],
        })

        rle = coco_rle(mask)
        binary_rle = {"size": rle["size"], "counts": rle["counts"].encode("ascii")}
        coco_annotations.append({
            "id": image_id,
            "image_id": image_id,
            "category_id": 1,
            "segmentation": rle,
            "area": float(mask.sum()),
            "bbox": [float(value) for value in mask_utils.toBbox(binary_rle)],
            "iscrowd": 0,
            "sam3_prompt": BODY_TEXT_PROMPT,
            "sam3_score": record["score"],
            "sam3_text_detection_index": record["text_detection_index"],
        })

    coco = {
        "info": {
            "description": (
                "Blue Resistors r1 COCO images rebuilt with the highest-confidence "
                "SAM 3.1 blue axial resistor-body mask above the configured threshold. "
                "All source annotations and categories were ignored."
            ),
            "sam3_prompt": BODY_TEXT_PROMPT,
            "min_accept_score": MIN_ACCEPT_SCORE,
            "source_annotations_ignored": True,
        },
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": [{"id": 1, "name": FINAL_CATEGORY_NAME, "supercategory": "electronic_component"}],
    }
    with open(base / "_annotations.coco.json", "w", encoding="utf-8") as handle:
        json.dump(coco, handle)

    summary[split_name] = {"images": len(coco_images), "instances": len(coco_annotations)}
    log(f"[export {split_index}/3] {split_name}: {len(coco_images)} images | {len(coco_annotations)} masks")


stage(7, 8, "Write QA manifest, summary, and final checkpoint")
write_manifest_csv(OUTPUT_ROOT / "refinement_manifest.csv", accepted_records, processed_keys)
with open(OUTPUT_ROOT / "summary.json", "w", encoding="utf-8") as handle:
    json.dump({
        "workflow_version": SAM3_COLAB_VERSION,
        "source_release": "r1",
        "source_assets": list(DATASET_ASSETS),
        "sam3_prompt": BODY_TEXT_PROMPT,
        "min_accept_score": MIN_ACCEPT_SCORE,
        "source_annotations_ignored": True,
        "source_images": len(records),
        "processed_images": len(processed_keys),
        "accepted_images": len(accepted_records),
        "rejected_images": rejected_count,
        "final_unique_images": len(unique),
        "exact_duplicate_records_removed": duplicates_removed,
        "splits": summary,
        "checkpoint_every_images": CHECKPOINT_EVERY_IMAGES,
        "checkpoint_dir": str(DRIVE_CHECKPOINT_DIR),
    }, handle, indent=2)
save_checkpoint(processed_keys, accepted_records, pending_mask_names)
log("[export] summary")
log(json.dumps(summary, indent=2))


stage(8, 8, "Package, copy final ZIP to Drive, and download")
zip_path = shutil.make_archive("/content/blue_resistors_sam3", "zip", OUTPUT_ROOT)
zip_size = Path(zip_path).stat().st_size / 1024 / 1024
shutil.copy2(zip_path, DRIVE_FINAL_ZIP)
log(f"[package] created {zip_path} ({zip_size:.1f} MiB)")
log(f"[package] copied final ZIP to Google Drive: {DRIVE_FINAL_ZIP}")
files.download(zip_path)
log("[done] Blue Resistors SAM 3.1 COCO rebuild finished")
