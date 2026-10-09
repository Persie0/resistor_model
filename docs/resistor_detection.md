# Resistor box detector

This detector localizes the whole axial resistor before the BandNet crop/recognition stage.

## Dataset

Training uses the public release asset:

- tag: `v4`
- asset: `resistor_sam3_merged.zip`
- SHA-256: `be3a1bb3b952f07556906decf6393fa7a8c228665867f721914a4e8e64376c6f`

The archive was rebuilt from the three source image sets using SAM 3.1 whole-body masks. Source annotations were ignored. The final dataset has `train`, `valid`, and `test` COCO splits with one category named `resistor`. The COCO `bbox` field is derived from each accepted SAM mask.

## Model

The default detector is torchvision `ssdlite320_mobilenet_v3_large`:

- input: RGB float tensor in `[0, 1]`
- detector input size: 320×320
- preprocessing geometry: resize/stretch the source image directly to 320×320 (no letterbox), then map predicted X/Y coordinates back with independent width/height scales
- classes: background + resistor
- training target: COCO bounding boxes
- validation selection: COCO bbox mAP over IoU 0.50:0.95
- additional metrics: mAP@0.50, mAP@0.75, AR@100
- default pretrained weights: none

The code is torchvision-based and does not use Ultralytics.

## Colab

Open:

`colab/train_resistor_detector_colab.ipynb`

The notebook:

1. mounts Google Drive,
2. downloads the exact v4 release asset,
3. verifies the release SHA-256,
4. extracts the COCO dataset,
5. trains SSDLite320,
6. saves `best.pt`, `last.pt`, and numbered checkpoints,
7. automatically resumes from Drive `last.pt`,
8. evaluates validation and test COCO bbox metrics,
9. exports ONNX,
10. packages the run outputs for download.

Persistent outputs are written to:

`MyDrive/resistor_model/detector-v4-ssdlite320`

The ONNX file is:

`resistor_detector_ssdlite320.onnx`

## Local training

Install detection dependencies:

```bash
pip install -e '.[detection,export]'
```

Extract the v4 ZIP so the root contains `train`, `valid`, and `test`, then run:

```bash
resistor-train-detection \
  --dataset-root /path/to/resistor_sam3_merged \
  --output-dir runs/resistor_detector_v4 \
  --epochs 80 \
  --batch-size 16 \
  --checkpoint-every 5 \
  --progress-every 10
```

Resume:

```bash
resistor-train-detection \
  --dataset-root /path/to/resistor_sam3_merged \
  --output-dir runs/resistor_detector_v4 \
  --epochs 80 \
  --resume runs/resistor_detector_v4/last.pt
```

If you explicitly want torchvision's ImageNet MobileNetV3 backbone weights, add `--pretrained-backbone`. The default remains off.

## ONNX export

```bash
resistor-export-detection \
  --checkpoint runs/resistor_detector_v4/best.pt \
  --output exports/resistor_detector_ssdlite320.onnx
```

An adjacent `.onnx.json` file documents the input/output contract.

The graph outputs post-NMS:

- `boxes`: `[N,4]` xyxy coordinates in 320×320 input pixels,
- `scores`: `[N]` resistor confidence,
- `labels`: `[N]`, where 1 means resistor.

A starting application confidence threshold of 0.35 is written to metadata. Tune it on real phone-camera validation data before release.
