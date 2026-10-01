# Resistor Band Training Stack Design

## Goal
Build a complete, reproducible PyTorch training stack for resistor color-band recognition that exploits resistor geometry: rectify each resistor to a horizontal crop, preserve horizontal resolution, decode bands as an ordered 1-D sequence, and evaluate exact sequence/value accuracy under leakage-proof grouped splits.

## Data
Canonical input is JSONL. Each image may contain multiple resistors. Each resistor has a stable physical `id`, optional whole-resistor `bbox`, and color-band boxes. Splits group by physical resistor ID; capture session can be grouped too.

## Geometry and targets
Band centers define the long axis with PCA. The crop is rectified to 128x768 by default. Training targets include a dense 1-D band map plus up to six ordered slots containing existence, color, normalized center, and width.

## Model
`ResistorBandNet` uses a compact ConvNeXt-style RGB encoder plus an optional log-RGB/log-chromaticity branch, asymmetric downsampling, vertical pooling, a Transformer encoder, ordered query decoder, dense sequence head, and band-count head.

## Training
AdamW, cosine decay, AMP on CUDA, gradient clipping, EMA, weighted dense CE, slot classification/localization losses, monotonic-order loss, count loss, and optional two-view illumination-consistency loss. Augmentation changes exposure, gamma, white balance, shadows, highlights, blur, noise and compression without arbitrary hue rotation.

## Evaluation
Primary metrics are exact band-sequence accuracy and exact decoded resistor-value accuracy. Also report band macro-F1, count accuracy, center MAE and dense accuracy.

## Tooling
Repository provides dataset audit, YOLO conversion, grouped split generation, train/evaluate/export CLIs, configs, tests, and documentation.
