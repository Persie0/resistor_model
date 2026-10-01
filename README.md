# resistor_model

PyTorch training stack for **structured resistor color-band recognition**.

Instead of treating each narrow band as an unrelated 2-D object, the pipeline rectifies each resistor to a horizontal crop and learns the bands as an ordered 1-D sequence.

## Architecture

1. Estimate the resistor long axis from annotated band centers with PCA.
2. Rectify the resistor to `128×768` by default.
3. Encode RGB with a compact ConvNeXt-style branch plus a log-RGB/log-chromaticity branch.
4. Preserve horizontal resolution, pool vertically, and run a 1-D Transformer.
5. Predict a dense band map, up to six ordered band slots `(exists, color, center, width)`, and the band count.
6. Train with structured losses, monotonic-order regularization, EMA, and optional two-view illumination consistency.
7. Decode both spatial directions with resistor-code rules without silently overriding visual predictions.

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

`bbox` for the whole resistor is optional. If missing, the rectifier derives an ROI from band boxes. Band boxes use absolute pixel coordinates `[x1,y1,x2,y2]`.

**Important:** `id` must identify the physical resistor, not the photograph. Every image of the same component should use the same ID so train/test leakage is impossible.

See `examples/manifest.jsonl`.

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

Then set `data.splits_file: data/splits.json` in `configs/bandnet.yaml`.

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

- `best.pt` — EMA model selected by validation exact-sequence accuracy.
- `last.pt` — latest model, raw weights and optimizer/scheduler state.
- `splits.json` — exact physical resistor IDs used for each split.
- `metrics.jsonl` — epoch metrics.

Resume by setting `train.resume` to `last.pt`.

## Evaluate

```bash
resistor-evaluate \
  --checkpoint runs/bandnet/best.pt \
  --split test
```

Metrics:

- per-band macro F1,
- exact spatial sequence accuracy,
- exact decoded resistor-value accuracy,
- band-count accuracy,
- normalized band-center MAE,
- dense 1-D accuracy.

Exact sequence/value accuracy should be the main model-selection metric; one wrong band normally means one wrong resistance value.

## Export ONNX

```bash
resistor-export \
  --checkpoint runs/bandnet/best.pt \
  --output exports/resistor_bandnet.onnx
```

The exporter also writes an adjacent JSON metadata file with the model configuration and output names.

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

The suite covers manifest parsing, leakage-proof splitting, vertical rectification, bidirectional resistor decoding, dense/slot targets, chromatic transforms, model output shapes, monotonic loss, backpropagation, consistency loss, metrics, configuration merging and YOLO conversion.

Local verification for this implementation: **17 tests passed**, `python -m compileall -q src` passed, and a synthetic end-to-end `--smoke` training run completed forward/backward, EMA update, validation and checkpoint writing.
