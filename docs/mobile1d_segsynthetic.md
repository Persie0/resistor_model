# Segmentation-in-the-loop Mobile1D XL (~25 MB) experiment

Two Google Colab notebooks implement the full **data synthesis → deployed segmentation → reader training** pipeline without changing or overwriting the existing BandNet Mobile1D checkpoints.

1. [Prepare synthetic/segmented dataset](../colab/prepare_mobile1d_segsynthetic_colab.ipynb)
2. [Train Mobile1D XL from segmented crops](../colab/train_rres_mobile1d_xl_segsynthetic_colab.ipynb)

## Inputs

- The original three COCO datasets are restored from the checksummed cache under `MyDrive/resistor_model/r1-mobile1d-colab/`.
- A **trained segmentation ONNX** is required in Google Drive. The default is `segmentation-v4-lraspp/resistor_segmenter_lraspp.onnx` (384×384, ImageNet normalization, one-channel probability).
- To use the trained **PP-LiteSeg** export instead, set `os.environ["MOBILE1D_SEGMENTER_ONNX"] = "/content/drive/MyDrive/resistor_model/segmentation-pp-liteseg/resistor_segmenter_ppliteseg.onnx"` in notebook 1 **before** its `from resistor_model.tools.mobile1d_segsynthetic import ...` import. This uses PaddleSeg normalization (mean/std 0.5), gray padding and two-channel logits decoding, rather than LR-ASPP preprocessing. The exact ONNX SHA is recorded in every shard's recipe; changing segmenters invalidates/rebuilds the synthesis cache.
- Original resistor-band colors, boxes and grouped split assignments must be available from the source manifest and `splits.json`.

## Synthesis

- TRAIN only: two six-band and one random 3–6-band synthetic variant per source, plus its real input.
- Simulated resistor values draw legal E24/E96 significant digits, multipliers, tolerance and TCR colors; visual stripes are placed along the photographed resistor axis.
- The original colored stripes are inpainted; replacement stripes retain target body brightness, receive local shading and may blend **actual photographed band textures taken exclusively from training resistors of the same color**.
- Rotated/sheared/scale-jittered synthetic scenes are processed by the SAME trained segmenter as real images. A segmenter mask's inferred long axis determines the final horizontal crop; transformed band annotations are carried into that 176×512 crop.
- The segmenter is also run on each original photograph. Grossly mismatched segmentation masks are rejected and counted instead of labeled as if they were correct.
- Real validation/test photos **never get synthetic variants**. Source physical resistor IDs and capture groups retain their original split.
- **Count-balanced epoch sampling:** although two six-band variants are generated per training source, the reader training sampler targets 4% three-band, 65% four-band, 18% five-band and 13% six-band examples. This prevents the synthesized six-band majority from dominating band-count predictions.

## Caching / resumability

`MyDrive/resistor_model/mobile1d-segsynthetic-data/` stores SHA-256 verified, independently resumable ZIP shards. Every cache receipt is associated with:

- the segmentation ONNX SHA-256;
- the original manifest/split SHA-256;
- the synthesis recipe version and seed;
- the requested number of input sources.

If Colab disconnects, run notebook 1 again: verified shards are skipped, unfinished shards are regenerated. Notebook 2 restores all shards and assembles a local pre-rectified manifest with `train/val/test` lists. The dataset loader's `pre_rectified=true` option prevents the default geometry rectifier from inadvertently rewarping the processed crops.

## Training

`MyDrive/resistor_model/r1-mobile1d-xl-segsynthetic-colab/` is an **independent** run. It never reads or replaces `best.pt` or `last.pt` from the smaller Mobile1D or the prior XL run.

| Setting | Value |
| --- | --- |
| Band reader | Mobile1D XL |
| FP32 parameters | 6,333,985 |
| FP32 weights (decimal) | ~25.34 MB |
| Feature width | 568 |
| Base convolution width | 48 |
| Crop | 176×512 RGB |
| Batch size | 8 |
| Epochs | 120 |
| Initial learning rate | 0.00025 |
| Warmup | 5 epochs |
| Schedule | Single cosine decay |
| Recovery | `last.pt` every epoch; numbered snapshots every 5 |
| Exports | ONNX; best-effort TFLite |

The estimated size is **FP32 weights only**, not a guarantee of exact `.onnx`/`.tflite` sizes. INT8/FP16 exports change it; CPU/NNAPI performance and Play-services LiteRT operator compatibility require separate measurements.

## Evaluation caveats

The held-out split contains almost no true six-band examples. Synthetic six-band training can improve learning but **cannot replace an independent real six-band test set**. Reader test accuracy computed only on successfully segmented examples can also hide segmentation failures. The notebook therefore writes `real_source_coverage` per train/val/test split and refuses to train below 70% coverage. Evaluate the entire detector + segmenter + reader stack on untouched photographs before shipping.

Failed masks, possible color artifacts from inpainting, donor-patch seams, and incorrectly projected band annotations all warrant visual review in notebook 1's preview. The synthetic data changes are an experiment, **not a guaranteed accuracy improvement**.
