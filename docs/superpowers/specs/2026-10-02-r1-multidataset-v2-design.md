# r1 Multi-Dataset V2 Training Design

## Goal
Train ResistorBandNet from all three `r1` COCO release datasets while preserving leakage-safe grouping, and integrate the uploaded model/data/decoder improvements only where they improve robustness without breaking the current v1 path.

## Dataset ingestion
Add a Roboflow COCO importer that accepts one or more extracted dataset roots. Normalize `grey` to `gray`, ignore project/root categories such as `r`, `resistors`, and `object-resistors`, and treat `resistor symbol` or `resistor` as body annotations. When a dataset has no body annotations, accept images only when their usable color-band annotations form one 3-6 band resistor; infer the resistor ROI from the union of those bands with margin. Prefix image paths by dataset root but derive a cross-dataset source-group identity from `extra.name` when available, otherwise from the filename before `.rf.`, so repeated Roboflow exports remain in the same split.

Generate one canonical JSONL manifest and one grouped 70/15/15 split across all accepted samples. Keep supplied Roboflow train/valid/test folders as metadata only; do not trust them as the final split because the same physical/source image may occur across datasets.

## Accepted uploaded improvements
- Add bounded rectification jitter and label-safe horizontal/vertical flips for training only.
- Add optional CTC, prediction-KL consistency, label smoothing, and slot-color class weights. Defaults remain zero/off for backward compatibility; the r1 GPU config enables conservative nonzero values.
- Add grammar-aware constrained decoding and optional extra-decoder evaluation metrics. E-series preference is soft and evaluation-only by default.
- Add `ResistorBandNetV2` as an optional architecture. Keep v1 selectable and keep geometry-based reading-direction resolution disabled by default until separately validated.
- Add a stronger chromatic branch and learned vertical pooling in V2.

## Training integration
Update the Colab script to download the three public `r1` release assets, import/combine them locally, create the unified grouped split, and train the configurable V2 GPU model while saving outputs/checkpoints to Drive. The dataset remains on local Colab storage. Resume continues from Drive `last.pt`.

## Verification
Use test-first coverage for COCO normalization/grouping, inferred ROI, geometric transforms, auxiliary losses, constrained decoding, V2 model construction, and Colab config/download list. Run the complete pytest suite, compileall, then a GitHub Actions gate that downloads all three real release datasets, imports them, checks split disjointness/source leakage, and performs a one-batch train/validation smoke on the real combined data.
