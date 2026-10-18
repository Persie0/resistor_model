"""Segmentation-in-the-loop synthesis for the Mobile1D reader.

Sources are labeled REAL resistor photographs. The trained LR-ASPP segmenter
first locates the physical body. Only TRAIN sources get synthetic, electrically
valid band colors. Rotated/sheared variants are segmented AGAIN and cropped by
the predicted mask rather than by ground-truth geometry. Every band bounding
box follows the exact warp/crop homographies. No generated sample enters val/test.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import tempfile
import time
import zipfile

import cv2
import numpy as np

from resistor_model.data.schema import load_manifest
from resistor_model.decoder import _decode_one

TARGET_HW = (176, 512)
COLORS = {
    "black": (34, 33, 31), "brown": (45, 67, 104), "red": (35, 45, 177),
    "orange": (16, 120, 220), "yellow": (35, 210, 235),
    "green": (60, 132, 40), "blue": (193, 94, 30),
    "violet": (124, 60, 123), "gray": (128, 128, 128),
    "white": (224, 224, 224), "gold": (55, 168, 196),
    "silver": (178, 176, 171),
}
DIGITS = ("black", "brown", "red", "orange", "yellow", "green", "blue", "violet", "gray", "white")
TOLERANCES = ("brown", "red", "green", "blue", "violet", "gray", "gold", "silver")
TEMPCOS = ("brown", "red", "orange", "yellow", "blue", "violet")
SHARD_SIZE = 120
SYNTH_PER_TRAIN = 3
MIN_MASK_IOU = 0.42
RECIPE = "r1-segsynthetic-lraspp-donor-bands-v2"
INPUT_ROOT = Path("/content/resistor_training/r1_datasets")
WORK_ROOT = Path("/content/mobile1d_segsynthetic")
DRIVE_ROOT = Path("/content/drive/MyDrive/resistor_model/mobile1d-segsynthetic-data")
# Override to the exported PP-LiteSeg ONNX when that is the trained segmenter.
# Example: MOBILE1D_SEGMENTER_ONNX=/content/drive/MyDrive/resistor_model/segmentation-pp-liteseg/resistor_segmenter_ppliteseg.onnx
SEGMENTER = Path(os.environ.get("MOBILE1D_SEGMENTER_ONNX",
    "/content/drive/MyDrive/resistor_model/segmentation-v4-lraspp/resistor_segmenter_lraspp.onnx"))
MEAN = np.asarray([0.485, 0.456, 0.406], np.float32)
STD = np.asarray([0.229, 0.224, 0.225], np.float32)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _foreground_from_output(raw: np.ndarray) -> np.ndarray:
    """Convert LR-ASPP probabilities or PaddleSeg class logits to a binary mask."""
    arr = np.asarray(raw)
    if arr.ndim == 4:
        if arr.shape[1] == 2:
            return (arr[0].argmax(axis=0) == 1).astype(np.uint8)
        if arr.shape[-1] == 2:
            return (arr[0].argmax(axis=-1) == 1).astype(np.uint8)
        if arr.shape[1] == 1:
            arr = arr[0, 0]
        else:
            raise ValueError(f"Unsupported segmentation channels: {arr.shape}")
    elif arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Unsupported segmentation output: {arr.shape}")
    if np.issubdtype(arr.dtype, np.integer) or arr.dtype == np.bool_:
        return (arr > 0).astype(np.uint8)
    return (arr >= (0.5 if np.nanmin(arr) >= 0.0 and np.nanmax(arr) <= 1.0 else 0.0)).astype(np.uint8)


class Segmenter:
    def __init__(self, model_path: Path):
        import onnxruntime as ort
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Trained LR-ASPP segmentation ONNX not found: {model_path}. "
                "Train/export segmentation-v4-lraspp first."
            )
        self.path = model_path
        self.sha = sha256(model_path)
        self.kind = "paddleseg" if "ppliteseg" in model_path.stem.lower() else "lraspp"
        metadata = model_path.with_suffix(model_path.suffix + ".json")
        if metadata.is_file():
            meta = json.loads(metadata.read_text(encoding="utf-8"))
            if self.kind == "lraspp" and meta.get("architecture") != "lraspp_mobilenet_v3_large":
                raise ValueError(f"Unexpected segmentation architecture {meta.get('architecture')!r}")
        self.mean, self.std = (
            (np.full(3, 0.5, np.float32), np.full(3, 0.5, np.float32))
            if self.kind == "paddleseg" else (MEAN, STD)
        )
        self.pad_value = 128 if self.kind == "paddleseg" else 0
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        print(f"[segmenter] ONNX Runtime providers: {self.session.get_providers()}", flush=True)
        inp = self.session.get_inputs()[0]
        self.input = inp.name
        shape = inp.shape
        if len(shape) != 4 or shape[1] != 3 or not isinstance(shape[2], int) or shape[2] != shape[3]:
            raise ValueError(f"Expected LR-ASPP static NCHW square input, got {shape}")
        self.size = int(shape[2])
        self.output = self.session.get_outputs()[0].name
        print(f"[segmenter] {self.kind} {self.size}x{self.size}, sha256 {self.sha[:16]}...")

    def mask(self, bgr: np.ndarray) -> np.ndarray:
        height, width = bgr.shape[:2]
        scale = min(self.size / width, self.size / height)
        nw, nh = max(1, round(width * scale)), max(1, round(height * scale))
        left, top = (self.size - nw) // 2, (self.size - nh) // 2
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        canvas = np.full((self.size, self.size, 3), self.pad_value, np.uint8)
        canvas[top:top+nh, left:left+nw] = cv2.resize(rgb, (nw, nh))
        x = (canvas.astype(np.float32) / 255.0 - self.mean) / self.std
        x = x.transpose(2, 0, 1)[None]
        raw = np.asarray(self.session.run([self.output], {self.input: x})[0])
        binary_square = _foreground_from_output(raw)
        if binary_square.shape != (self.size, self.size):
            raise ValueError(f"Unexpected segmentation output: {raw.shape}")
        binary = cv2.resize(
            binary_square[top:top+nh, left:left+nw].astype(np.uint8),
            (width, height), interpolation=cv2.INTER_NEAREST,
        )
        n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if n < 2:
            raise ValueError("segmenter found no resistor")
        selected = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        output = (labels == selected).astype(np.uint8)
        if output.sum() < 100:
            raise ValueError("segmented component is too small")
        return output


def _xyxy(bbox) -> np.ndarray:
    x1,y1,x2,y2 = map(float,bbox)
    return np.asarray([[x1,y1],[x2,y1],[x2,y2],[x1,y2]],np.float32)


def transform_box(bbox, h: np.ndarray, width: int, height: int) -> list[float]:
    points = cv2.perspectiveTransform(_xyxy(bbox)[None], h.astype(np.float32))[0]
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    return [
        max(0., min(float(width-1), float(lo[0]))),
        max(0., min(float(height-1), float(lo[1]))),
        max(1., min(float(width), float(hi[0]))),
        max(1., min(float(height), float(hi[1]))),
    ]


def _source_roi(image: np.ndarray, annotation):
    height, width = image.shape[:2]
    if annotation.bbox is not None:
        box = annotation.bbox
    elif annotation.polygon is not None:
        poly = np.asarray(annotation.polygon)
        box = (poly[:,0].min(), poly[:,1].min(), poly[:,0].max(), poly[:,1].max())
    else:
        xs = [n for b in annotation.bands for n in (b.bbox[0], b.bbox[2])]
        ys = [n for b in annotation.bands for n in (b.bbox[1], b.bbox[3])]
        box = (min(xs), min(ys), max(xs), max(ys))
    x1,y1,x2,y2 = map(float,box)
    pad_x,pad_y = (x2-x1)*0.7,(y2-y1)*1.0
    ix0,iy0 = max(0,int(x1-pad_x)),max(0,int(y1-pad_y))
    ix1,iy1 = min(width,int(math.ceil(x2+pad_x))),min(height,int(math.ceil(y2+pad_y)))
    if ix1-ix0 < 32 or iy1-iy0 < 24:
        raise ValueError("source resistor ROI too small")
    roi = image[iy0:iy1,ix0:ix1].copy()
    boxes = [[b.bbox[0]-ix0,b.bbox[1]-iy0,b.bbox[2]-ix0,b.bbox[3]-iy0] for b in annotation.bands]
    return roi, boxes


def _long_axis(boxes):
    centers = np.array([[(b[0]+b[2])/2,(b[1]+b[3])/2] for b in boxes],np.float32)
    if len(centers)>1:
        _,_,vt=np.linalg.svd(centers-centers.mean(axis=0),full_matrices=False)
        u=vt[0]
    else:
        u=np.array([1.,0.],np.float32)
    if u[0]<0 or (abs(u[0])<0.01 and u[1]<0):
        u=-u
    return u.astype(np.float32)


def new_colors(rng: np.random.Generator, n: int) -> list[str]:
    """Draw plausible manufactured E-series values, rather than random digits."""
    from resistor_model.decoding import _E24, _E96

    if n not in (3, 4, 5, 6):
        raise ValueError(f"unsupported resistor band count {n}")
    series = _E24 if n in (3, 4) else _E96
    mantissa = int(rng.choice(series))
    significant = f"{mantissa:02d}" if n <= 4 else f"{mantissa:03d}"
    color_digits = [DIGITS[int(v)] for v in significant]
    multiplier = str(rng.choice([
        "gold", "black", "brown", "red", "orange", "yellow", "green"
    ], p=[0.08, 0.12, 0.18, 0.27, 0.20, 0.10, 0.05]))
    result = color_digits + [multiplier]
    if n > 3:
        result.append(str(rng.choice(
            ["brown", "red", "green", "blue", "violet", "gold", "silver"],
            p=[0.13, 0.10, 0.03, 0.02, 0.02, 0.60, 0.10]
        )))
    if n == 6:
        result.append(str(rng.choice(TEMPCOS)))
    if not _decode_one(result).valid:
        raise AssertionError(f"synthetic band grammar invalid: {result}")
    return result


def collect_real_band_donors(records, *, max_per_color: int = 48):
    """Collect real color-band textures from TRAIN photos only.

    These are not extra synthetic image labels: they're texture patches from
    annotated training bands, never val/test. Their hue variation/specular
    structure is pasted onto other training resistor bodies and blended with
    the palette-based stripe simulation.
    """
    donors = {color: [] for color in COLORS}
    scanned = 0
    for rel, annotation, split in records:
        if split != "train":
            continue
        scanned += 1
        if scanned > 1500 or all(len(donors[k]) >= max_per_color for k in COLORS):
            break
        image = cv2.imread(str(INPUT_ROOT / rel), cv2.IMREAD_COLOR)
        if image is None:
            continue
        h, w = image.shape[:2]
        for band in annotation.bands:
            if band.color not in donors or len(donors[band.color]) >= max_per_color:
                continue
            x1,y1,x2,y2 = [int(round(v)) for v in band.bbox]
            x1,x2 = max(0,x1), min(w,x2)
            y1,y2 = max(0,y1), min(h,y2)
            if x2-x1 < 3 or y2-y1 < 3:
                continue
            patch = image[y1:y2,x1:x2].copy()
            # Remove annotated-box edge pixels, which may contain body color.
            ph,pw = patch.shape[:2]
            patch = patch[max(0,ph//8):max(1,ph-ph//8),
                          max(0,pw//8):max(1,pw-pw//8)]
            if patch.size:
                donors[band.color].append(patch)
    print("[donors] real TRAIN band textures: " +
          ", ".join(f"{c}={len(v)}" for c,v in donors.items()),flush=True)
    return donors


def repaint_bands(image: np.ndarray, mask: np.ndarray, boxes: list, names: list[str],
                  rng: np.random.Generator, donor_bank=None) -> tuple[np.ndarray, list]:
    """Remove source bands, paint *plausible textured* replacements inside the body."""
    h,w=image.shape[:2]
    u=_long_axis(boxes)
    yy,xx=np.mgrid[:h,:w]
    projected=xx.astype(np.float32)*u[0]+yy.astype(np.float32)*u[1]
    orig_centers=[((b[0]+b[2])/2)*u[0]+((b[1]+b[3])/2)*u[1] for b in boxes]
    orig_widths=[max(2., abs((b[2]-b[0])*u[0])+abs((b[3]-b[1])*u[1])) for b in boxes]
    span=max(orig_centers)-min(orig_centers) if len(orig_centers)>1 else w*.45
    span=max(span, float(np.median(orig_widths))*5)
    middle=(min(orig_centers)+max(orig_centers))/2
    # Erase only the original stripes, keeping body texture and specular shading.
    erase=np.zeros((h,w),np.uint8)
    for center,band_w in zip(orig_centers,orig_widths):
        erase[(mask>0)&(abs(projected-center)<0.68*band_w)]=255
    cleaned=cv2.inpaint(image,erase,3,cv2.INPAINT_TELEA)
    output=cleaned.copy()
    n=len(names)
    low=middle-span*0.57
    high=middle+span*0.57
    positions=np.linspace(low,high,n)
    stripe_width=min(float(np.median(orig_widths))*0.95, (high-low)/max(n+0.7,4))
    stripe_width=max(2.,stripe_width)
    labels=[]
    for name, center in zip(names, positions):
        region=(mask>0)&(np.abs(projected-center)<stripe_width*0.45)
        if int(region.sum())<24:
            raise ValueError("synthetic stripe has insufficient visible area")
        base=np.asarray(COLORS[name],np.float32)
        old=cleaned.astype(np.float32)
        # Preserve original local lightness, highlights and sensor texture,
        # modifying chroma only within a feathered stripe.
        gray=cv2.cvtColor(cleaned,cv2.COLOR_BGR2GRAY).astype(np.float32)
        local=float(np.median(gray[region]))
        luminance=np.clip(0.50+0.50*gray/(local+1),0.35,1.6)
        target=np.clip(base[None,None,:]*luminance[...,None],0,255)
        # A donor band of this color brings actual photographic stripe texture,
        # while the palette's luminance adjustment retains target-body shading.
        candidates=(donor_bank or {}).get(name,[])
        if candidates:
            patch=candidates[int(rng.integers(len(candidates)))]
            ypix,xpix=np.where(region)
            x0,x1=int(xpix.min()),int(xpix.max()+1)
            y0,y1=int(ypix.min()),int(ypix.max()+1)
            donor=cv2.resize(patch,(x1-x0,y1-y0),interpolation=cv2.INTER_LINEAR)
            donor=donor.astype(np.float32)
            donor_lum=cv2.cvtColor(donor.astype(np.uint8),cv2.COLOR_BGR2GRAY)
            adjustment=np.clip(float(np.median(gray[region]))/
                               max(float(np.median(donor_lum)),1.0),0.55,1.65)
            blended=np.clip(donor*adjustment,0,255)
            target[y0:y1,x0:x1] = (
                target[y0:y1,x0:x1]*.30+blended*.70
            )
        feather=np.clip((stripe_width*.5-np.abs(projected-center))/max(stripe_width*.14,1),0,1)
        alpha=(region*feather*rng.uniform(.87,.99)).astype(np.float32)[...,None]
        output=np.clip(output.astype(np.float32)*(1-alpha)+target*alpha,0,255).astype(np.uint8)
        y,x=np.where(region)
        labels.append({"color":name,"bbox":[float(x.min()),float(y.min()),float(x.max()+1),float(y.max()+1)]})
    return output,labels


def _warp(image:np.ndarray, mask:np.ndarray, boxes:list, rng:np.random.Generator, active:bool):
    h,w=image.shape[:2]
    if not active:
        return image,mask,boxes
    cx,cy=w/2,h/2
    angle=float(rng.uniform(-38,38))
    shear=float(rng.uniform(-0.18,0.18))
    scale=float(rng.uniform(.82,1.12))
    matrix=cv2.getRotationMatrix2D((cx,cy),angle,scale)
    h3=np.vstack([matrix,[0.,0.,1.]]) @ np.asarray([
        [1.,shear,-shear*cy],[0.,1.,0.],[0.,0.,1.]
    ])
    offset=np.asarray([[1.,0.,w*.15],[0.,1.,h*.15],[0.,0.,1.]])
    h3=offset@h3
    size=(int(w*1.3),int(h*1.3))
    warped=cv2.warpPerspective(image,h3,size,borderMode=cv2.BORDER_REFLECT_101)
    m=cv2.warpPerspective(mask,h3,size,flags=cv2.INTER_NEAREST)
    converted=[transform_box(b,h3,*size) for b in boxes]
    return warped,m,converted


def _horizontal_crop(image:np.ndarray,mask:np.ndarray,boxes:list):
    """Rectify with SEGMENTATION geometry, not ground-truth band annotation."""
    ys,xs=np.where(mask>0)
    if len(xs)<100:
        raise ValueError("insufficient segmented pixels to align")
    pts=np.stack((xs,ys),axis=1).astype(np.float32)
    _,_,vt=np.linalg.svd(pts-pts.mean(0),full_matrices=False)
    u=vt[0]
    if u[0]<0: u=-u
    v=np.asarray([-u[1],u[0]],np.float32)
    o=pts.mean(0)
    pu=(pts-o)@u;pv=(pts-o)@v
    x0,x1=np.percentile(pu,[0.3,99.7]);y0,y1=np.percentile(pv,[0.3,99.7])
    padx=max((x1-x0)*.08,2);pady=max((y1-y0)*.13,2)
    src=np.asarray([o+u*(x0-padx)+v*(y0-pady),
                    o+u*(x1+padx)+v*(y0-pady),
                    o+u*(x1+padx)+v*(y1+pady),
                    o+u*(x0-padx)+v*(y1+pady)],np.float32)
    H,W=TARGET_HW
    dst=np.asarray([[0,0],[W-1,0],[W-1,H-1],[0,H-1]],np.float32)
    h3=cv2.getPerspectiveTransform(src,dst)
    crop=cv2.warpPerspective(image,h3,(W,H),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    converted=[]
    for box in boxes:
        b=transform_box(box,h3,W,H)
        if b[2]-b[0]<2 or b[3]-b[1]<2:
            raise ValueError("band lost during segmenter cropping")
        converted.append(b)
    if len(converted)>1:
        xs=[(b[0]+b[2])/2 for b in converted]
        if min(np.diff(sorted(xs)))<2:
            raise ValueError("band centers collapsed during rectification")
    return crop,converted


def process_variant(segmenter:Segmenter,source:np.ndarray,source_boxes:list,
                    source_names:list[str],rng:np.random.Generator,synthetic:bool,
                    count:int|None=None,donor_bank=None):
    initial=segmenter.mask(source)
    # Source annotation and selected segmentation must overlap; reject
    # multi-resistor crops where segmenter picked the wrong object.
    annotated=np.zeros(initial.shape,np.uint8)
    for b in source_boxes:
        p=np.round(b).astype(int)
        cv2.rectangle(annotated,(max(0,p[0]),max(0,p[1])),(min(source.shape[1]-1,p[2]),min(source.shape[0]-1,p[3])),1,-1)
    overlap=int((initial & annotated).sum())/max(int(annotated.sum()),1)
    if overlap<0.22:
        raise ValueError(f"wrong initial segmentation (band overlap={overlap:.2f})")
    image=source
    boxes=source_boxes
    names=source_names
    if synthetic:
        names=new_colors(rng,int(count))
        image, labels=repaint_bands(source,initial,boxes,names,rng,donor_bank=donor_bank)
        boxes=[b["bbox"] for b in labels]
    image,expected,boxes=_warp(image,initial,boxes,rng,active=synthetic)
    predicted=segmenter.mask(image)
    intersection=int((expected & predicted).sum())
    union=int((expected | predicted).sum())
    iou=intersection/max(union,1)
    if iou<MIN_MASK_IOU:
        raise ValueError(f"post-transform segmentation IoU {iou:.3f} < {MIN_MASK_IOU}")
    crop,boxes=_horizontal_crop(image,predicted,boxes)
    if not cv2.imencode(".jpg",crop,[cv2.IMWRITE_JPEG_QUALITY,94])[0]:
        raise ValueError("failed JPEG encode")
    # Canonical image coords; image is already segmented and horizontal.
    bands=sorted([{"color":n,"bbox":b} for n,b in zip(names,boxes)],
                 key=lambda x:(x["bbox"][0]+x["bbox"][2])/2)
    return crop,bands,iou


def read_sources(manifest:Path,splits:Path):
    by_split=json.loads(splits.read_text())
    reverse={}
    for split in ("train","val","test"):
        for rid in by_split[split]:
            if rid in reverse: raise ValueError(f"split leakage: {rid}")
            reverse[rid]=split
    records=[]
    for row in load_manifest(manifest):
        for ann in row.resistors:
            split=reverse.get(ann.id)
            if split:
                records.append((row.image,ann,split))
    records.sort(key=lambda r:(r[0],r[1].id))
    return records


def process_shard(segmenter, records, shard_index, shard_dir:Path, seed=42, donor_bank=None):
    shard_dir.mkdir(parents=True,exist_ok=True)
    image_root=shard_dir/"images";image_root.mkdir(exist_ok=True)
    rng=np.random.default_rng(seed+shard_index)
    labels=[];failed=[];observed={"train":0,"val":0,"test":0}
    for index,(rel,ann,split) in enumerate(records):
        observed[split]+=1
        path=INPUT_ROOT/rel
        source=cv2.imread(str(path),cv2.IMREAD_COLOR)
        if source is None:
            failed.append({"id":ann.id,"split":split,"reason":"image missing"})
            continue
        try:
            roi,boxes=_source_roi(source,ann)
        except ValueError as exc:
            failed.append({"id":ann.id,"split":split,"reason":str(exc)})
            continue
        originals=[b.color for b in ann.bands]
        variants=[("real",None)]
        if split=="train":
            variants += [(f"syn{i}",6 if i<2 else int(rng.choice([3,4,5,6]))) for i in range(SYNTH_PER_TRAIN)]
        for tag,n in variants:
            name=hashlib.sha256(f"{ann.id}|{rel}|{tag}".encode()).hexdigest()[:24]
            try:
                crop,bands,iou=process_variant(
                    segmenter,roi,boxes,originals,rng,synthetic=n is not None,count=n,
                    donor_bank=donor_bank,
                )
                rel_file=f"images/{name}.jpg"
                cv2.imwrite(str(shard_dir/rel_file),crop,[cv2.IMWRITE_JPEG_QUALITY,94])
                labels.append({
                    "image":rel_file, "split":split,
                    "session_id":ann.id, "camera_id":"segsynthetic-lraspp",
                    "resistors":[{
                        "id":f"{ann.id}--{name}", "bbox":[0,0,TARGET_HW[1],TARGET_HW[0]],
                        "bands":bands,
                    }],
                    "source_resistor_id":ann.id,
                    "synthetic": n is not None,
                    "segmentation_iou":round(iou,4),
                })
            except (ValueError,cv2.error) as exc:
                failed.append({"id":ann.id,"split":split,"tag":tag,"reason":str(exc)})
        if (index+1)%20==0 or index+1==len(records):
            print(f"[shard {shard_index:03d}] {index+1}/{len(records)} sources "
                  f"| {len(labels)} crops | {len(failed)} failures",flush=True)
    (shard_dir/"rows.jsonl").write_text("".join(json.dumps(row)+"\n" for row in labels))
    (shard_dir/"failures.jsonl").write_text("".join(json.dumps(row)+"\n" for row in failed))
    (shard_dir/"counts.json").write_text(json.dumps({"observed":observed,"crops":len(labels),
                                                   "failed":len(failed)}))
    return len(labels)


def archive_shard(src:Path,dst:Path,recipe:dict) -> None:
    """Versioned checksum receipt last; interrupted uploads never count as complete."""
    dst.parent.mkdir(parents=True,exist_ok=True)
    temporary=dst.with_suffix(".zip.partial")
    local=src.with_suffix(".zip")
    with zipfile.ZipFile(local,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for path in sorted(src.rglob("*")):
            if path.is_file(): z.write(path,path.relative_to(src).as_posix())
    try:
        shutil.copyfile(local,temporary)
        os.replace(temporary,dst)
        receipt={"sha256":sha256(dst),"bytes":dst.stat().st_size,"recipe":recipe}
        metadata=dst.with_suffix(".json")
        tmp_meta=metadata.with_suffix(".json.partial")
        tmp_meta.write_text(json.dumps(receipt,indent=2))
        os.replace(tmp_meta,metadata)
    finally:
        local.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)


def valid_shard(dst:Path,recipe:dict)->bool:
    receipt=dst.with_suffix(".json")
    if not dst.is_file() or not receipt.is_file():return False
    try:
        meta=json.loads(receipt.read_text())
        return (meta["recipe"]==recipe and dst.stat().st_size==meta["bytes"]
                and sha256(dst)==meta["sha256"])
    except (OSError,ValueError,KeyError):return False


def prepare(segmenter:Segmenter,max_sources:int|None=None):
    data_manifest=INPUT_ROOT/"manifest.jsonl"
    data_splits=INPUT_ROOT/"splits.json"
    source=read_sources(data_manifest,data_splits)
    if max_sources is not None:
        source=source[:max_sources]
    recipe={"version":RECIPE,"segmenter_sha256":segmenter.sha,
            "source_manifest_sha256":sha256(data_manifest),
            "source_splits_sha256":sha256(data_splits),
            "seed":42,"variants":SYNTH_PER_TRAIN,"shard_size":SHARD_SIZE,
            "max_sources":max_sources}
    remote=DRIVE_ROOT/"shards";remote.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    # Optional train-only donor textures; val/test cannot become donor sources.
    donor_bank=collect_real_band_donors(source)
    for index,at in enumerate(range(0,len(source),SHARD_SIZE)):
        dst=remote/f"shard_{index:04d}.zip"
        if valid_shard(dst,recipe):
            print(f"[shard {index:04d}] verified Drive cache hit; skipping",flush=True)
            continue
        root=WORK_ROOT/f"shard_{index:04d}"
        if root.exists():shutil.rmtree(root)
        print(f"[shard {index:04d}] generating {min(SHARD_SIZE,len(source)-at)} sources",flush=True)
        process_shard(segmenter,source[at:at+SHARD_SIZE],index,root,donor_bank=donor_bank)
        archive_shard(root,dst,recipe)
        shutil.rmtree(root)
        print(f"[shard {index:04d}] saved {dst.stat().st_size/1048576:.1f} MiB "
              f"| elapsed {(time.monotonic()-started)/60:.1f} min",flush=True)
    (DRIVE_ROOT/"recipe.json").write_text(json.dumps({"recipe":recipe,
                             "n_shards":math.ceil(len(source)/SHARD_SIZE)},indent=2))
    print(f"[synth] complete: {len(source)} source resistors, "
          f"{math.ceil(len(source)/SHARD_SIZE)} verified ZIP shards",flush=True)


def restore(prepared_root:Path=Path("/content/mobile1d_segsynthetic_prepared"))->dict:
    """Restore all finished shards locally; publish a train-only-synthetic dataset."""
    manifest=json.loads((DRIVE_ROOT/"recipe.json").read_text())
    recipe=manifest["recipe"]
    prepared_root.mkdir(parents=True,exist_ok=True)
    images=prepared_root/"images";images.mkdir(exist_ok=True)
    rows=[];fail=[];observed={"train":0,"val":0,"test":0}
    for index in range(manifest["n_shards"]):
        archive=DRIVE_ROOT/"shards"/f"shard_{index:04d}.zip"
        if not valid_shard(archive,recipe):
            raise RuntimeError(f"Missing/invalid shard {index}; finish synthesis first")
        with zipfile.ZipFile(archive) as z:
            names=z.namelist()
            if any(p.startswith("/") or ".." in Path(p).parts for p in names):
                raise ValueError("unsafe archive member")
            for item in names:
                if item.startswith("images/"):
                    target=prepared_root/item
                    if not target.exists():
                        with z.open(item) as src,target.open("wb") as dst:
                            shutil.copyfileobj(src,dst)
            rows.extend(json.loads(l) for l in z.read("rows.jsonl").decode().splitlines() if l)
            fail.extend(json.loads(l) for l in z.read("failures.jsonl").decode().splitlines() if l)
            counts=json.loads(z.read("counts.json"))
            for split,v in counts["observed"].items():observed[split]+=v
        print(f"[restore] {index+1}/{manifest['n_shards']} shards | {len(rows)} crops",flush=True)
    splits={"train":[],"val":[],"test":[]}
    seen=set()
    for row in rows:
        uid=row["resistors"][0]["id"]
        if uid in seen:raise ValueError(f"duplicate ID {uid}")
        seen.add(uid)
        if row["synthetic"] and row["split"]!="train":
            raise AssertionError("synthetic leakage into val/test")
        splits[row["split"]].append(uid)
    (prepared_root/"manifest.jsonl").write_text(
        "".join(json.dumps({k:v for k,v in r.items() if k not in (
            "source_resistor_id","synthetic","segmentation_iou")})+"\n" for r in rows))
    (prepared_root/"splits.json").write_text(json.dumps(splits,indent=2))
    (prepared_root/"sources.jsonl").write_text(
        "".join(json.dumps({"image":r["image"],"split":r["split"],
            "source_resistor_id":r["source_resistor_id"],"synthetic":r["synthetic"],
            "segmentation_iou":r["segmentation_iou"]})+"\n" for r in rows))
    (prepared_root/"failures.jsonl").write_text("".join(json.dumps(r)+"\n" for r in fail))
    coverage={
        split:sum(r["split"]==split and not r["synthetic"] for r in rows)/max(observed[split],1)
        for split in splits
    }
    summary={"images":len(rows),"split_counts":{k:len(v) for k,v in splits.items()},
             "real_source_coverage":coverage,"source_counts":observed,
             "failed_variants":len(fail),"segmenter_sha256":recipe["segmenter_sha256"]}
    (prepared_root/"summary.json").write_text(json.dumps(summary,indent=2))
    print("[restore] dataset summary: "+json.dumps(summary,indent=2),flush=True)
    if not splits["train"] or not splits["val"] or not splits["test"]:
        raise ValueError("one of train/validation/test splits is empty")
    return summary
