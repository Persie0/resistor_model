# Resistor Band Training Implementation Plan

**Goal:** Create a complete, reproducible training/evaluation/export repository for a rectified 1-D resistor color-band recognizer and validate it against the supplied `rres.v4` annotations.

**Architecture:** Canonical JSONL annotations feed leakage-safe grouped splitting and body-aware geometric rectification. Rectified crops produce dense and ordered-slot targets consumed by a dual-branch ConvNeXt-style + Transformer network, trained with structured losses and evaluated with exact sequence/value and per-color metrics.

**Tech Stack:** Python 3.10+, PyTorch, NumPy, OpenCV, PyYAML, pytest, optional ONNX/onnxscript export dependencies.

**Spec:** `docs/superpowers/specs/2026-10-01-resistor-band-training-design.md`

## Tasks

- [x] Canonical data contract with multiple resistors and optional whole-resistor box/polygon metadata.
- [x] Roboflow importer that separates the body class from band colors and preserves body geometry.
- [x] Leakage-safe ID/session grouping, including conservative grouping of repeated named rres.v4 captures.
- [x] Body-driven rectification with polygon → box-axis → band-PCA fallback.
- [x] Dense 1-D and ordered-slot targets plus physically plausible color-safe augmentation.
- [x] Independent deterministic augmentation RNG streams per DataLoader worker.
- [x] ResistorBandNet, chromatic branch, ordered queries and structured losses.
- [x] 3/4/5/6-band bidirectional resistor decoder.
- [x] Exact sequence/value/count, localization, dense, macro-F1 and per-color F1 metrics.
- [x] AdamW training with EMA, AMP, cosine scheduling, robust resume state and deterministic best-checkpoint selection.
- [x] Evaluation and ONNX export CLIs.
- [x] Dataset/body/orientation/leakage audits and real-data smoke validation using the public Playground runner.

## Verification strategy

Before merge, the public `Persie0/Playground` validation checks out the feature branch and runs the complete unit suite plus `compileall`, imports/audits the private `rres.v4` dataset, verifies disjoint source/capture groups and held-out color coverage, audits body-orientation quality, executes a real-data train/validation smoke batch, and exercises evaluation/export paths. Exact final test counts and run results are recorded in PR #1 after the final gate completes.
