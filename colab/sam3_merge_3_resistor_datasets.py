"""Colab workflow: refine three COCO resistor datasets with SAM 3 and merge them.

Old labels are used only as prompts/comparison. Accepted SAM 3 masks become the new
truth. Images with no accepted SAM 3 masks are removed. Exact duplicate images are
removed before an 80/10/10 split.
"""

import gc
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import zipfile
from collections import defaultdict
from pathlib import Path


def run(cmd, cwd=None):
    print("+", " ".join(map(str, cmd)))
    subprocess.run(list(map(str, cmd)), cwd=cwd, check=True)


# ---- Configuration -------------------------------------------------------
NUM_DATASETS = 3
FINAL_CATEGORY_NAME = "resistor"
SOURCE_CATEGORY_NAMES = {"resistor", "resistor_body", "resistors"}
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
TRAIN_FRACTION, VALID_FRACTION, TEST_FRACTION = 0.80, 0.10, 0.10
SPLIT_SEED = 1337
WORK_ROOT = "/content/sam3_merge_work"
OUTPUT_ROOT = "/content/resistor_sam3_merged"
SAM3_DIR = "/content/sam3"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# ---- Install/load dependencies ------------------------------------------
run([
    sys.executable, "-m", "pip", "install", "-q", "-U",
    "huggingface_hub", "pycocotools", "opencv-python-headless",
    "pandas", "tqdm", "matplotlib",
])
if not os.path.isdir(SAM3_DIR):
    run(["git", "clone", "--depth", "1", "https://github.com/facebookresearch/sam3.git", SAM3_DIR])
else:
    run(["git", "pull", "--ff-only"], cwd=SAM3_DIR)
run([sys.executable, "-m", "pip", "install", "-q", "-e", SAM3_DIR])
if SAM3_DIR not in sys.path:
    sys.path.insert(0, SAM3_DIR)

import cv2
import numpy as np
import pandas as pd
import torch
from google.colab import files
from huggingface_hub import login, notebook_login
from PIL import Image, ImageDraw
from pycocotools import mask as mask_utils
from tqdm.auto import tqdm
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

if not torch.cuda.is_available():
    raise RuntimeError("Switch Colab to a GPU runtime and rerun.")

# HF_TOKEN secret is preferred; interactive login is the fallback.
token = None
try:
    from google.colab import userdata
    token = userdata.get("HF_TOKEN")
except Exception:
    pass
if token:
    login(token=token, add_to_git_credential=False)
else:
    notebook_login()


# ---- Upload/extract the three source datasets ----------------------------
shutil.rmtree(WORK_ROOT, ignore_errors=True)
os.makedirs(WORK_ROOT, exist_ok=True)
uploaded = files.upload()
zip_items = [(name, data) for name, data in uploaded.items() if name.lower().endswith(".zip")]
if len(zip_items) != NUM_DATASETS:
    raise RuntimeError(f"Upload exactly {NUM_DATASETS} dataset ZIPs; got {len(zip_items)}.")

source_roots = []
for i, (name, data) in enumerate(zip_items, 1):
    zp = os.path.join(WORK_ROOT, f"source_{i}.zip")
    root = os.path.join(WORK_ROOT, f"source_{i}")
    with open(zp, "wb") as f:
        f.write(data)
    os.makedirs(root, exist_ok=True)
    with zipfile.ZipFile(zp) as zf:
        zf.extractall(root)
    source_roots.append(root)
    print(f"[{i}] {name}")


def looks_like_coco(path):
    try:
        obj = json.load(open(path, encoding="utf-8"))
        return all(isinstance(obj.get(k), list) for k in ("images", "annotations", "categories"))
    except Exception:
        return False


def basename_index(root):
    out = defaultdict(list)
    for p in Path(root).rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            out[p.name].append(p)
    return out


def resolve_image(root, json_path, file_name, idx):
    fn = str(file_name).replace("\\", "/")
    for p in (Path(json_path).parent / fn, Path(root) / fn, Path(json_path).parent / Path(fn).name):
        if p.is_file():
            return str(p)
    hits = idx.get(Path(fn).name, [])
    return str(hits[0]) if len(hits) == 1 else None


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ---- Index all annotated images -----------------------------------------
records = []
for source_index, root in enumerate(source_roots, 1):
    json_paths = [str(p) for p in Path(root).rglob("*.json") if looks_like_coco(p)]
    if not json_paths:
        raise RuntimeError(f"No COCO JSON found in source {source_index}.")
    idx = basename_index(root)
    for json_path in sorted(json_paths):
        coco = json.load(open(json_path, encoding="utf-8"))
        cats = {c["id"]: str(c.get("name", "")).lower() for c in coco["categories"]}
        allowed = {cid for cid, name in cats.items() if name in SOURCE_CATEGORY_NAMES} or set(cats)
        anns_by_image = defaultdict(list)
        for ann in coco["annotations"]:
            if ann.get("category_id") in allowed:
                anns_by_image[ann["image_id"]].append(ann)
        for im in coco["images"]:
            anns = anns_by_image.get(im["id"], [])
            if not anns:
                continue
            image_path = resolve_image(root, json_path, im["file_name"], idx)
            if image_path:
                records.append({
                    "source": source_index,
                    "image_path": image_path,
                    "file_name": im["file_name"],
                    "annotations": anns,
                })
print("Annotated source image records:", len(records))


# ---- Geometry/mask helpers ----------------------------------------------
def box_from_mask(mask):
    ys, xs = np.where(mask > 0)
    if not len(xs):
        return np.array([0, 0, 0, 0], np.float32)
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], np.float32)


def bbox_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    aa = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    ab = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return float(inter / (aa + ab - inter)) if aa + ab - inter else 0.0


def decode_old(ann, h, w):
    seg = ann.get("segmentation")
    if seg:
        try:
            if isinstance(seg, list):
                rle = mask_utils.merge(mask_utils.frPyObjects(seg, h, w))
            else:
                rle = dict(seg)
                if isinstance(rle.get("counts"), list):
                    rle = mask_utils.frPyObjects(rle, h, w)
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
    mask = np.zeros((h, w), np.uint8)
    if "bbox" not in ann:
        return mask, True
    x, y, bw, bh = map(float, ann["bbox"])
    x1, y1 = max(0, int(x)), max(0, int(y))
    x2, y2 = min(w, int(np.ceil(x + bw))), min(h, int(np.ceil(y + bh)))
    mask[y1:y2, x1:x2] = 1
    return mask, True


def metrics(old, sam):
    a, b = old.astype(bool), sam.astype(bool)
    inter = float(np.logical_and(a, b).sum())
    aa, bb = float(a.sum()), float(b.sum())
    return {
        "mask_iou": inter / (aa + bb - inter) if aa + bb - inter else 0.0,
        "old_coverage": inter / aa if aa else 0.0,
        "sam_coverage": inter / bb if bb else 0.0,
        "bbox_iou": bbox_iou(box_from_mask(old), box_from_mask(sam)),
    }


def matches(m, bbox_only):
    if bbox_only:
        return m["bbox_iou"] >= MIN_BOX_ONLY_BBOX_IOU
    return m["mask_iou"] >= MIN_MASK_IOU or (
        m["old_coverage"] >= MIN_OLD_COVERAGE
        and m["sam_coverage"] >= MIN_SAM_COVERAGE
        and m["bbox_iou"] >= MIN_BBOX_IOU
    )


def rank(m, score, bbox_only):
    if bbox_only:
        return 0.8 * m["bbox_iou"] + 0.2 * score
    return 0.5 * m["mask_iou"] + 0.2 * min(m["old_coverage"], m["sam_coverage"]) + 0.15 * m["bbox_iou"] + 0.15 * score


def mask_iou(a, b):
    a, b = a.astype(bool), b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def dedup_instances(instances):
    kept = []
    for inst in sorted(instances, key=lambda x: x["rank"], reverse=True):
        if not any(mask_iou(inst["mask"], k["mask"]) >= FINAL_INSTANCE_DUP_IOU for k in kept):
            kept.append(inst)
    return kept


def coco_rle(mask):
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


# ---- SAM 3 refinement ----------------------------------------------------
model = build_sam3_image_model()
model.eval()
processor = Sam3Processor(model=model, device="cuda", confidence_threshold=SAM_PROCESSOR_THRESHOLD)
refined, manifest_rows = [], []

for ri, rec in enumerate(tqdm(records, desc="SAM 3 refinement")):
    image = Image.open(rec["image_path"]).convert("RGB")
    w, h = image.size
    state = processor.set_image(image)
    accepted = []
    for old_i, ann in enumerate(rec["annotations"]):
        old, bbox_only = decode_old(ann, h, w)
        if not old.any():
            continue
        x1, y1, x2, y2 = box_from_mask(old)
        prompt_box = [((x1 + x2) / 2) / w, ((y1 + y2) / 2) / h, max(1, x2 - x1) / w, max(1, y2 - y1) / h]
        processor.reset_all_prompts(state)
        out = processor.add_geometric_prompt(box=prompt_box, label=True, state=state)
        masks = out["masks"].squeeze(1).detach().cpu().numpy().astype(np.uint8)
        scores = out["scores"].detach().cpu().numpy().astype(float)
        cands = []
        for mask, score in zip(masks, scores):
            score = float(score)
            frac = float(mask.sum() / (h * w))
            if score < MIN_SAM_SCORE or not mask.any() or not (MIN_IMAGE_AREA_FRACTION <= frac <= MAX_IMAGE_AREA_FRACTION):
                continue
            m = metrics(old, mask)
            cands.append({"mask": mask, "score": score, "metrics": m, "rank": rank(m, score, bbox_only)})
        good = [c for c in cands if matches(c["metrics"], bbox_only)]
        if good:
            best = max(good, key=lambda c: c["rank"])
            accepted.append(best)
            manifest_rows.append({"source": rec["source"], "image": rec["file_name"], "old_instance": old_i, "accepted": True, **best["metrics"], "sam_score": best["score"], "match_rank": best["rank"]})
        else:
            best = max(cands, key=lambda c: c["rank"], default=None)
            row = {"source": rec["source"], "image": rec["file_name"], "old_instance": old_i, "accepted": False}
            if best:
                row.update(best["metrics"])
                row.update(sam_score=best["score"], match_rank=best["rank"])
            manifest_rows.append(row)
    accepted = dedup_instances(accepted)
    if accepted:
        refined.append({**rec, "hash": sha256_file(rec["image_path"]), "instances": accepted})
    del state
    if ri % 50 == 0:
        gc.collect()
        torch.cuda.empty_cache()

manifest = pd.DataFrame(manifest_rows)
print("Records surviving SAM 3:", len(refined), "/", len(records))


# ---- Deduplicate exact images across the 3 datasets ----------------------
by_hash = defaultdict(list)
for rec in refined:
    by_hash[rec["hash"]].append(rec)
unique = []
for group in by_hash.values():
    unique.append(max(group, key=lambda r: (len(r["instances"]), np.mean([i["rank"] for i in r["instances"]]))))
if not unique:
    raise RuntimeError("No image survived. Lower thresholds only after checking why matching failed.")
print("Unique final images:", len(unique))


# ---- Deterministic split -------------------------------------------------
rng = random.Random(SPLIT_SEED)
rng.shuffle(unique)
n = len(unique)
n_train = min(n, int(round(n * TRAIN_FRACTION)))
n_valid = min(n - n_train, int(round(n * VALID_FRACTION)))
splits = {
    "train": unique[:n_train],
    "valid": unique[n_train:n_train + n_valid],
    "test": unique[n_train + n_valid:],
}


# ---- Export --------------------------------------------------------------
shutil.rmtree(OUTPUT_ROOT, ignore_errors=True)
os.makedirs(OUTPUT_ROOT, exist_ok=True)
summary = {}
for split, items in splits.items():
    base = Path(OUTPUT_ROOT) / split
    image_dir, mask_dir, overlay_dir = base / "images", base / "masks_semantic", base / "overlays"
    for d in (image_dir, mask_dir, overlay_dir):
        d.mkdir(parents=True, exist_ok=True)
    coco_images, coco_anns, ann_id = [], [], 1
    for image_id, rec in enumerate(items, 1):
        image = Image.open(rec["image_path"]).convert("RGB")
        w, h = image.size
        ext = Path(rec["image_path"]).suffix.lower() if Path(rec["image_path"]).suffix.lower() in IMG_EXTS else ".jpg"
        name = f"{image_id:06d}_{rec['hash'][:12]}{ext}"
        shutil.copy2(rec["image_path"], image_dir / name)
        coco_images.append({"id": image_id, "file_name": f"images/{name}", "width": w, "height": h, "source_dataset": rec["source"], "sha256": rec["hash"]})
        semantic = np.zeros((h, w), np.uint8)
        overlay = np.array(image).copy()
        for inst in rec["instances"]:
            mask = inst["mask"]
            semantic = np.maximum(semantic, mask)
            overlay[mask.astype(bool)] = (0.55 * overlay[mask.astype(bool)] + 0.45 * 255).astype(np.uint8)
            rle = coco_rle(mask)
            rr = {"size": rle["size"], "counts": rle["counts"].encode("ascii")}
            coco_anns.append({
                "id": ann_id, "image_id": image_id, "category_id": 1,
                "segmentation": rle, "area": float(mask.sum()),
                "bbox": [float(x) for x in mask_utils.toBbox(rr)], "iscrowd": 0,
                "sam3_score": inst["score"], "old_sam_mask_iou": inst["metrics"]["mask_iou"],
                "old_coverage": inst["metrics"]["old_coverage"], "sam_coverage": inst["metrics"]["sam_coverage"],
                "old_sam_bbox_iou": inst["metrics"]["bbox_iou"], "match_rank": inst["rank"],
            })
            ann_id += 1
        stem = Path(name).stem
        Image.fromarray(semantic * 255).save(mask_dir / f"{stem}.png")
        Image.fromarray(overlay).save(overlay_dir / f"{stem}.jpg", quality=92)
    coco = {
        "info": {"description": "3 resistor datasets refined/merged with SAM 3; final labels are accepted SAM 3 masks only"},
        "images": coco_images, "annotations": coco_anns,
        "categories": [{"id": 1, "name": FINAL_CATEGORY_NAME, "supercategory": "electronic_component"}],
    }
    json.dump(coco, open(base / "_annotations.coco.json", "w", encoding="utf-8"))
    summary[split] = {"images": len(coco_images), "instances": len(coco_anns)}

manifest.to_csv(Path(OUTPUT_ROOT) / "refinement_manifest.csv", index=False)
json.dump({
    "source_records": len(records), "survived_sam": len(refined), "final_unique_images": len(unique),
    "splits": summary,
    "thresholds": {"mask_iou": MIN_MASK_IOU, "old_coverage": MIN_OLD_COVERAGE, "sam_coverage": MIN_SAM_COVERAGE, "bbox_iou": MIN_BBOX_IOU},
}, open(Path(OUTPUT_ROOT) / "summary.json", "w"), indent=2)
print(json.dumps(summary, indent=2))

zip_path = shutil.make_archive("/content/resistor_sam3_merged", "zip", OUTPUT_ROOT)
print("Created:", zip_path)
files.download(zip_path)
