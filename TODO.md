# TODO — resistor_model

Last reviewed: **2026-10-23**  
Companion: [Android scanner TODO](https://github.com/Persie0/resistor-scanner-android/blob/main/TODO.md)

## Current state (confirmed from source)

- The Android application currently ships **BandNet V2** (ConvNeXt-Lite/chromatic branch + 1-D attention/Transformer, 128×768, six exported prediction tensors) rather than Mobile1D.
- This repository has a separate **Mobile1D** implementation (`src/resistor_model/models/bandnet_mobile.py`) and Colab training/export workflow (`colab/train_rres_mobile1d_colab.py`, 176×512). Mobile1D uses attention-based vertical pooling, a 1-D TCN and a small slot reader; it is **not** the exact CNN + three-layer Transformer + center-heatmap architecture in the research proposal.
- Bidirectional grammar-aware 3/4/5/6-band decoding, top-k *color-combination* search, an optional soft E-series bonus, session-aware train/test splits, color metrics and geometric augmentation already exist. Do not rebuild these from scratch.
- The inspected GitHub release list contains a published V2 checkpoint/export but **no publicly released trained Mobile1D artifact**. The Colab workflow writes checkpoints to Drive and attempts ONNX/TFLite export, but its TFLite step is best effort, not yet a verified Android release.
- Known `rres.v4` limitations documented in README: only four six-band examples and no silver bands. Avoid extrapolating accuracy to those cases.

## P0 — establish a reliable next Android model

- [ ] **Freeze a deployed V2 baseline.** Record exact checkpoint/release SHA, source dataset and split, input preprocessing and output contract. Evaluate exact *spatial* band sequence, valid electrical value accuracy **with decode coverage and ambiguity rate**, band-count accuracy, per-color F1 (especially brown/red/orange, yellow/gold and white/gray/silver) and confidence calibration.
  - Done when reproducible reports and a fixed held-out manifest are published in the repo, with V2 results clearly labeled as the shipped baseline.
- [ ] **Train/evaluate Mobile1D against V2 fairly.** Run `colab/train_rres_mobile1d_colab.py` (resume from Drive if available) and evaluate both against the **same resistor/session-disjoint IDs and actual Android-produced rectified crops**. Compare accuracy by band count/color, ambiguity and latency/model size; distinguish 128×768 versus 176×512 effects.
  - Done when a side-by-side table indicates whether Mobile1D is truly better, not just smaller. Do not claim Mobile1D is deployed until the Android repo switches artifacts.
- [ ] **Make Mobile1D conversion production-grade.** Replace best-effort/first-found `.tflite` selection with an explicit expected filename, I/O tensor contracts, operator compatibility verification and numerical PyTorch → ONNX → LiteRT parity on real and synthetic crops. Verify all six outputs and any layout changes; reject unsupported Flex/custom ops.
  - Done when an immutable release contains the verified TFLite, metadata/output-index map, SHA256 checksums and conversion report usable by the Android model importer.
- [ ] **Build a real-crop regression set at the boundary with Android.** Save opt-in, privacy-safe full photo + segmenter mask + rectified crop + physical-resistor ID + ground-truth colors/direction/value. Match the actual Android rectangle, background fill, resampling and color order; test V2/Mobile1D against it.
  - Done when Python and Android can evaluate the same fixtures without hidden training/inference preprocessing differences.
- [ ] **Close the rare-class data gap.** Acquire verified six-band, silver, unusual tolerance/tempco, 0-ohm, low-contrast and repeated-adjacent-color resistors across multiple phone cameras and lighting/capture sessions. Inspect label quality and keep physical instances/sessions separated.
  - Done when per-class and per-count test coverage is explicitly reported; do not inflate metrics with synthetically duplicated test photos.

## P1 — improve recognition quality (ablate; do not rewrite blindly)

- [ ] **Add ranked electrical-value hypotheses with meaningful posteriors.** Extend `src/resistor_model/decoding.py` beyond the single best top-k color sequence: enumerate legal count/color/orientation interpretations, merge equivalent electrical results using log-sum-exp, expose top 3 *distinct* resistance/tolerance outcomes and uncertainty/abstention. Preserve cases genuinely ambiguous under reversal, including six-band tempco and valid rare/non-E-series codes. Tune soft E-series/tolerance priors on validation only, never use an absolute E-series filter.
- [ ] **Calibrate decoder confidence.** Test temperature scaling or another validated method; report reliability diagrams, ECE, selective accuracy/coverage and wrong-but-high-confidence cases, stratified by phone and lighting. A normalized candidate score is not by itself a calibrated probability.
- [ ] **Improve missing/extra/same-color-band robustness.** Compare current slot centers + dense logits with a small 1-D center-heatmap/peak head (and optional weak-peak insertion/deletion). Measure effects on consecutive equal bands and 3/4/5/6 count errors; retain the simpler head unless a held-out improvement is demonstrated.
- [ ] **Measure direction cues explicitly.** Compare the current bidirectional ambiguous decoder + center spacing to a learned left/right reading-direction head; never force a confident direction when visual evidence is insufficient. Test 180° reversals and symmetric spacing.
- [ ] **Ablate vertical/context mechanisms.** Compare V2 height attention + Transformer, Mobile1D height pooling + TCN, and (only if needed) the proposed asymmetric CNN + multihead vertical attention + small column Transformer at matched compute. Keep a plain CRNN/CTC baseline. Report accuracy, params, MACs, ONNX/LiteRT operator support, real-device speed, and memory; do **not** assume the proposed 0.7M/0.3 GMAC estimates are achieved.
- [ ] **Target color-constancy failures.** Run controlled tests for log-chroma inputs, illumination consistency/distillation, body-relative color features, specular reflection/shadow and motion/defocus/JPEG augmentations. Avoid aggressive hue jitter, grayscale, MixUp or CutMix that invalidate band hue labels.

## P2 — operational improvements

- [ ] Test float32 versus float16 and representative quantization/delegates **only if** brown/red/orange and gold/yellow confusion, exact value accuracy and real-device latency stay acceptable. Keep an FP32 reference artifact.
- [ ] Version a model manifest: architecture, input size/layout/normalization, color order, six tensor names and shapes, output-index map, segmenter pairing, dataset/split provenance, calibrated decoder parameters, checksums and license.
- [ ] Add opt-in human correction ingestion with anonymization/consent, deduplication and label review before corrected results become training data. Keep this outside validation/test sets.
- [ ] Maintain a concise error gallery and pre/post improvement comparison; focus on real scanner failures rather than training-loss-only optimization.

## Cross-repo handoff / sequencing

1. **Model repo:** fix benchmark and production-crop dataset → produce and verify candidate weights/export → publish pinned artifacts and metadata.
2. **Android repo:** import into a separate candidate build → compare end-to-end exact value/latency/memory on-device → switch the default only after it wins.
3. **Both repos:** share decoding fixtures, confidence semantics, orientation behavior, input normalization and release version IDs. Android's current top-k color decoder and best-of-frame voting should not be confused with a calibrated top-3 electrical posterior.

Research inspiration: supplied “A 1-D band reader for resistor color codes” proposal; many parts are already implemented in V2/Mobile1D. Tasks above identify **measurable gaps**, not a wholesale mandatory replacement.
