# Whole-resistor segmentation

This repository includes a separate binary semantic-segmentation pipeline for locating the whole resistor before ResistorBandNet runs.

## Model

The default model is `LR-ASPP + MobileNetV3-Large` from torchvision with two output classes: background and resistor. It is deliberately a semantic segmenter rather than an Ultralytics detector/segmenter because the Android pipeline only needs a clean resistor mask, long-axis orientation and crop.

Expected mobile pipeline:

```text
camera frame
  -> resistor_segmenter_lraspp.onnx
  -> foreground probability mask
  -> threshold + connected components
  -> long-axis/oriented crop
  -> ResistorBandNet
  -> electrical decoder
```

The ONNX export returns one `float32` tensor named `foreground_probability` with shape `[N, 1, H, W]`.

## Google Colab

Open `colab/train_resistor_segmentation_colab.ipynb`, choose a GPU runtime and run all cells.

The recipe automatically downloads:

```text
https://github.com/Persie0/resistor_model/releases/download/v4/resistor_sam3_merged.zip
```

The v4 dataset contains one `resistor` foreground class and the original SAM 3.1 semantic PNG masks under `train/valid/test/masks_semantic/`. Training reads those binary masks directly, preserving the detailed silhouette without COCO RLE parsing or bounding-box fallback. The archive is verified against its release SHA-256 (`be3a1bb3b952f07556906decf6393fa7a8c228665867f721914a4e8e64376c6f`). Legacy COCO polygon datasets remain supported by the loader.

Defaults:

- input: `384 x 384` letterboxed RGB
- epochs: `60`
- batch size: `16`
- optimizer: AdamW
- loss: foreground-weighted cross entropy + soft Dice
- augmentation: flips, 90-degree rotations and conservative color jitter
- live train and validation progress every `1` batch in Colab (plus startup heartbeats)
- validation metrics: foreground IoU, Dice and pixel accuracy
- checkpoint selection: best validation Dice
- `last.pt` after every completed epoch
- `best.pt` whenever validation Dice improves
- `checkpoints/epoch_NNN.pt` every `5` epochs and on the final epoch
- checkpoints include model, optimizer, scheduler and AMP scaler state
- append-only `metrics.jsonl` plus final `summary.json`
- persistence: `MyDrive/resistor_model/segmentation-v4-lraspp`
- automatic resume from `last.pt`
- automatic ONNX export of `best.pt`
- final ZIP package of the complete persistent run directory for download

### Test on personal photos after training

The training notebook includes a separate final cell **Test the trained model on your photos**. Run that cell after training, or run it independently once `best.pt` is available in Google Drive; it does not retrain. If the ONNX export is missing or older than the best checkpoint, it exports `best.pt` automatically. Upload one or more JPG/PNG resistor images to see the foreground probability, cleaned mask, detection overlay, long-axis estimate, rotated crop and horizontal BandNet crop. You can rerun only the test cell with different images. The same visual tester is also available in `colab/test_resistor_segmentation_alignment_colab.ipynb`.

If the Colab/browser download fails, all checkpoints and metrics remain in Google Drive.

## Local training

Install the segmentation dependencies:

```bash
pip install -e '.[segmentation,export]'
```

Then run:

```bash
resistor-train-segmentation \
  --dataset-root /path/to/resistor_sam3_merged \
  --category resistor \
  --output-dir runs/resistor_segmentation
```

The standalone CLI defaults print progress every 10 batches (Colab overrides to every batch) and write a numbered checkpoint every 5 epochs. Both can be changed or disabled:

```bash
resistor-train-segmentation ... \
  --progress-every 5 \
  --checkpoint-every 2
```

Use `0` for either option to disable that behavior.

Export:

```bash
resistor-export-segmentation \
  --checkpoint runs/resistor_segmentation/best.pt \
  --output exports/resistor_segmenter_lraspp.onnx
```

An adjacent `.onnx.json` file describes preprocessing, input/output names and the default `0.5` mask threshold.

## Licensing choice

The architecture and implementation come from torchvision, which is BSD-3-Clause licensed. The default training path does **not** load torchvision's pretrained ImageNet weights. This is intentional: torchvision documents that pretrained models may have separate or dataset-derived terms that users must evaluate for their use case.

To opt into the torchvision ImageNet MobileNetV3 backbone anyway, pass:

```bash
--pretrained-backbone
```

or set `USE_PRETRAINED_BACKBONE = True` in the Colab script after reviewing those terms.

The training-code license does not grant additional rights to the training images or annotations. Verify the license/provenance of any dataset you use and preserve any attribution requirements that apply to it.

No Ultralytics package, model implementation or pretrained weight is required by this segmentation path.
