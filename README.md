# resistor_model

PyTorch training stack for **structured resistor color-band recognition**.

Instead of treating each narrow band as an unrelated 2-D object, the pipeline rectifies each resistor to a horizontal crop and learns the bands as an ordered 1-D sequence.

## Architecture

1. Estimate the resistor long axis from the whole-resistor polygon when available, otherwise the whole-resistor box long axis, and only fall back to annotated band centers when no body annotation exists.
2. Rectify the resistor to `128×768` by default.
3. Encode RGB with a compact ConvNeXt-style branch plus a log-RGB/log-chromaticity branch.
4. Preserve horizontal resolution, pool vertically, and run a 1-D Transformer.
5. Predict a dense band map, up to six ordered band slots `(exists, color, center, width)`, and the band count.
6. Train with structured losses, monotonic-order regularization, EMA, and optional two-view illumination consistency.
7. Decode 3/4/5/6-band codes in both spatial directions with resistor-code rules. Standard 3-band codes use the implicit 20% tolerance. If both directions are electrically valid but imply different values, decoding is reported as **ambiguous** instead of silently choosing one direction.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

For ONNX export:

```bash
pip install -e '.[export]'
```

## Canonical annotation format

Use JSONL. Each line is one image and may contain multiple physical resistors:

```json
{
  "image": "img_001.jpg",
  "session_id": "pixel8_daylight_01",
  "camera_id": "pixel8",
  "resistors": [
    {
      "id": "physical_resistor_0123",
      "bbox": [110, 180, 790, 420],
      "polygon": [[120, 220], [760, 180], [780, 380], [140, 420]],
      "body_annotation_type": "polygon",
      "bands": [
        {"color": "brown", "bbox": [250, 205, 285, 395]},
        {"color": "black", "bbox": [350, 205, 385, 395]},
        {"color": "red",   "bbox": [455, 205, 490, 395]},
        {"color": "gold",  "bbox": [650, 205, 685, 395]}
      ]
    }
  ]
}
```

Whole-resistor `bbox` and `polygon` are optional. Polygon geometry is preferred for crop orientation, then the box long axis; if neither exists, the rectifier derives orientation/ROI from the band annotations. Band boxes use absolute pixel coordinates `[x1,y1,x2,y2]`.

**Important:** for a new dataset, `id` should identify the physical resistor rather than the photograph. Every image of the same component should use the same ID so train/test leakage is impossible. `session_id` can additionally isolate a capture session, setup, or conservative family of likely-related captures.

See `examples/manifest.jsonl`.

## rres.v4 annotation audit

The supplied Roboflow `rres.v4` export contains a real whole-resistor class named `resistor symbol` (class 8). It is **not** treated as a band/color class.

Audit of the YOLOv8 and COCO exports:

- 792 images and 3,668 color-band annotations.
- 746 whole-resistor annotations: 616 boxes and 130 true polygon segmentations; COCO independently confirms the polygon annotations.
- 289 images have no whole-resistor annotation, so body supervision remains optional.
- In single-resistor images with 3–6 bands, 99.79% of band centers fall inside the whole-resistor region and 99.1% of images have every band center inside it.
- 95.8% of those images have every band box at least 95% contained by the whole-resistor box.
- The whole-resistor region extends a median 1.46× beyond the union of the band boxes along the resistor axis, confirming it represents the resistor body rather than a duplicate band label.

The importer therefore preserves the body box/polygon and its annotation type. Images without body labels continue to use the band-derived fallback and must not be interpreted as negative examples for a body detector.

A separate geometry audit compared body-derived orientation against PCA of the accepted band centers. Polygon orientation had a median error of about 1.58° and p95 about 3.22°. Axis-aligned body-box orientation had a median error of about 1.86° and p95 about 7.97%; only about 0.2% exceeded 20°. This supports using body geometry for rectification instead of throwing those annotations away.

The cleaned importer accepts **829 resistor instances from 689 images**. Their annotated band counts are 69 three-band, 497 four-band, 259 five-band, and only 4 six-band resistors. There are no silver-band examples. The latter two facts are important dataset limitations: six-band and silver performance cannot be considered well validated from this dataset alone.

### Leakage protection for the supplied export

The Roboflow filenames contain several repeated capture families, including both timestamped names such as `1k-5-_202512...` and numbered series such as `100R_1-4W_-60-`. The filenames do not prove that every image in a family is the exact same physical component, but scattering likely-related captures independently across train/validation/test would be scientifically unsafe.

`resistor-import-roboflow` therefore:

- keeps the original source ID for each accepted source image,
- derives a conservative `session_id` for meaningful timestamped and numbered capture families,
- does **not** collapse generic recorder/file names such as `batch`, `Error`, `image`, `img`, or `download`,
- and the supplied `rres_v4` configs use `group_session: true` so connected IDs/sessions stay in one partition.

On the accepted dataset this produces **568 capture groups**, of which **60 contain multiple images**. The final validation asserts that no source ID or capture group crosses train/validation/test and that common annotated colors remain represented in both held-out splits.

## Convert YOLO band annotations

For normalized YOLO `class x_center y_center width height` labels where each image contains one resistor:

```bash
resistor-convert-yolo \
  --images /path/to/images \
  --labels /path/to/labels \
  --output data/manifest.jsonl \
  --classes black,brown,red,orange,yellow,green,blue,violet,gray,white,gold,silver
```

Pass the exact class order used by your dataset.

For the supplied mixed Roboflow export, use `resistor-import-roboflow`; it separates the `resistor symbol` body class from the band colors, preserves box/polygon metadata, groups bands to the correct resistor in multi-resistor images, and derives conservative capture groups for leakage-safe splitting.

## Audit the dataset

```bash
resistor-audit \
  --manifest data/manifest.jsonl \
  --image-root data/images \
  --output runs/audit.json
```

The audit reports unreadable images, out-of-bounds boxes, color distribution, band-count distribution and exact duplicate image groups.

## Create fixed leakage-proof splits

```bash
resistor-split \
  --manifest data/manifest.jsonl \
  --output data/splits.json \
  --ratios 0.70 0.15 0.15 \
  --seed 42
```

To additionally keep each capture session in one partition:

```bash
resistor-split ... --group-session
```

Then set `data.splits_file: data/splits.json` in the config. The supplied `configs/rres_v4*.yaml` enable session grouping.

## Train

Edit dataset paths in `configs/bandnet.yaml`:

```bash
resistor-train --config configs/bandnet.yaml
```

Pipeline smoke test:

```bash
resistor-train --config configs/bandnet.yaml --smoke
```

Outputs under `train.output_dir`:

- `best.pt` — EMA model selected first by validation exact-sequence accuracy and then by macro-F1 as the tie-breaker.
- `last.pt` — latest model, raw model, optimizer, scheduler and AMP scaler state.
- `splits.json` — exact source/resistor IDs used for each split.
- `metrics.jsonl` — epoch metrics; `lr` is the learning rate actually used for that epoch.

Resume by setting `train.resume` to `last.pt`. Checkpoints serialize the scheduler after advancing to the next epoch, so resumed and uninterrupted LR schedules match. New checkpoints restore AMP scaler state; older checkpoints without scaler state remain loadable.

Training DataLoader workers receive independent deterministic augmentation RNG streams, avoiding duplicated/repeated NumPy augmentation sequences across workers while retaining reproducibility.

## Evaluate

```bash
resistor-evaluate \
  --checkpoint runs/bandnet/best.pt \
  --split test
```

Metrics include:

- band macro-F1,
- per-color F1 (`f1_black`, `f1_brown`, …),
- exact spatial-sequence accuracy,
- exact decoded resistor-value accuracy on the **electrically unambiguous subset**,
- `value_decode_coverage` — fraction of samples with a unique valid electrical interpretation,
- `value_ambiguity_rate` — fraction whose color sequence is valid in both directions with different electrical values,
- band-count accuracy,
- normalized band-center MAE,
- dense 1-D accuracy.

Exact spatial-sequence accuracy is the primary vision metric. Do not interpret `exact_value_accuracy` without its coverage: color order alone is sometimes insufficient to choose electrical reading direction. In the supplied dataset audit, 217 of 784 electrically decodable rectified samples had two valid directions with different values, so silently preferring left-to-right would produce a misleading product metric.

Per-color F1 is also important because the supplied data is imbalanced (for example white is much rarer than black/brown, silver is absent).

## Export ONNX

```bash
resistor-export \
  --checkpoint runs/bandnet/best.pt \
  --output exports/resistor_bandnet.onnx
```

The exporter also writes an adjacent JSON metadata file with the model configuration and output names.

## Train the v4 resistor box detector

The public release asset `v4/resistor_sam3_merged.zip` is the detector dataset. It contains COCO `train/valid/test` splits with a single `resistor` category; each accepted SAM 3.1 body mask also has a COCO bounding box.

Use `colab/train_resistor_detector_colab.ipynb` for the one-click GPU workflow. It trains torchvision **SSDLite320 + MobileNetV3-Large** as a one-class box detector, evaluates COCO bbox mAP, saves resumable Drive checkpoints, and exports `resistor_detector_ssdlite320.onnx`. The default recipe does not use Ultralytics or download pretrained ImageNet/COCO weights.

See `docs/resistor_detection.md` for local training/export commands and model I/O.

## Test whole-resistor segmentation and alignment in Colab

Use `colab/test_resistor_segmentation_alignment_colab.ipynb` after exporting `resistor_segmenter_lraspp.onnx` to the default Drive path. The notebook uploads a photo, runs the whole-resistor segmenter, and estimates the resistor long axis from a weighted consensus of robust `fitLine`, PCA, `minAreaRect`, and the longest convex-hull chord. Angle outliers are rejected before rotation.

It displays the segmentation probability, cleaned mask, detected consensus axis, the final **vertically aligned** crop, and a **horizontal BandNet-ready** crop. The reusable undirected-angle and OpenCV rotation helpers live in `src/resistor_model/alignment.py` and are covered by unit tests.

## Why the augmentation is different

No arbitrary HSV hue rotation is used because hue is the label. Training instead simulates:

- exposure and gamma changes,
- correlated white-balance shifts,
- soft local shadows,
- curved-body specular highlights,
- blur,
- sensor-like noise,
- JPEG degradation.

With `loss.consistency > 0`, two independently relit views of the same resistor are trained to have similar high-level embeddings while band colors remain directly supervised.

## Recommended ablations

Keep the same persisted split for every experiment:

1. RGB only: `model.use_chromatic_branch: false`.
2. RGB + chromatic branch: default.
3. No consistency loss: `loss.consistency: 0`.
4. No order regularization: `loss.order: 0`.
5. Crop sizes around `96×512`, `128×768`, and `160×1024`.
6. Mobile student model: e.g. `base_channels: 24`, `d_model: 128`, two encoder layers.

Evaluate all variants on the normal held-out test set and a manually curated OOD test set containing unseen cameras, lighting, backgrounds and resistor instances.

## Tests

```bash
pytest -q
```

The suite covers manifest parsing, whole-resistor box/polygon metadata, leakage-proof grouped splitting, body-driven rectification, clipped-band target compaction, 3/4/5/6-band bidirectional/ambiguous resistor decoding, dense/slot targets, deterministic augmentation reseeding, chromatic transforms, model output shapes, monotonic loss, backpropagation, consistency loss, per-color/scalar metrics, electrical decode coverage, checkpoint-selection helpers, resume compatibility, configuration merging, YOLO conversion, and resistor-axis alignment helpers.
