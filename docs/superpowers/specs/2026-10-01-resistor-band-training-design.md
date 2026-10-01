# Resistor Band Training Stack Design

## Goal
Build a reproducible PyTorch training stack for resistor color-band recognition that exploits resistor geometry: rectify each resistor to a horizontal crop, preserve horizontal resolution, decode bands as an ordered 1-D sequence, and evaluate exact sequence/value accuracy under leakage-safe grouped splits.

## Data
Canonical input is JSONL. Each image may contain multiple resistors. Each resistor has a stable `id`, optional whole-resistor `bbox`, optional polygon and color-band boxes. For new data, `id` should identify the physical resistor. `session_id` can additionally group a capture session or conservative family of likely-related captures.

The supplied rres.v4 importer treats `resistor symbol` as body metadata rather than a band class, preserves its box/polygon geometry, groups bands to bodies in multi-resistor images and derives conservative session groups for repeated timestamped named captures. Generic `batch`/`Error` captures are not collapsed. Train/validation/test resistor-ID overlap is rejected even for hand-written manifests or external split files.

## Geometry and targets
Rectification prefers a whole-resistor polygon long axis, then the long axis of the whole-resistor box, and only falls back to PCA of annotated band centers when body geometry is unavailable. The supplied dataset audit supports this hierarchy: polygon orientation closely matches band PCA and box-axis orientation is also reliable for the accepted samples.

The crop is rectified to 128×768 by default. Training targets include a dense 1-D band map plus up to six ordered slots containing existence, color, normalized center and width.

## Model
`ResistorBandNet` uses a compact ConvNeXt-style RGB encoder plus an optional log-RGB/log-chromaticity branch, asymmetric downsampling, vertical pooling, a Transformer encoder, ordered query decoder, dense sequence head and band-count head.

## Training
AdamW, cosine decay, AMP on CUDA, gradient clipping, EMA, weighted dense cross-entropy, slot classification/localization losses, monotonic-order loss, count loss and optional two-view illumination-consistency loss. Augmentation changes exposure, gamma, white balance, shadows, highlights, blur, noise and compression without arbitrary hue rotation. Each DataLoader worker receives an independent deterministic augmentation RNG stream.

Best checkpoints are selected lexicographically by exact spatial-sequence accuracy and then macro-F1. Checkpoints retain raw and EMA model state, optimizer, scheduler and AMP scaler state; resume remains compatible with older checkpoints that lack scaler state.

## Decoding
The structured decoder supports standard 3/4/5/6-band codes in either spatial direction. Three-band resistors use the standard implicit 20% tolerance. Visual spatial-sequence confidence and electrical decoding should be interpreted separately because color-only direction can be ambiguous for some otherwise valid codes.

## Evaluation
Primary vision/product metrics are exact spatial band-sequence accuracy and exact decoded resistor-value accuracy. Also report band macro-F1, per-color F1, band-count accuracy, normalized center MAE and dense 1-D accuracy. Per-color metrics are required because the supplied dataset is strongly imbalanced across colors.

## Tooling and validation
The repository provides dataset audit/import/conversion, grouped split generation, train/evaluate/export CLIs, configs, tests and a Colab GPU runner. Final real-data validation uses the public `Persie0/Playground` Actions runner to test the private rres.v4 data while avoiding private-repository Actions usage. The gate checks the full test suite, compileability, annotation integrity, split/session isolation, held-out color coverage, orientation quality and a real-data training smoke path, plus evaluator/export execution.
