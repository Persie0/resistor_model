# TODO — Proposed 1-D resistor band reader

Implement **only** the uploaded proposal “A 1-D band reader for resistor color codes”. This checklist follows its sections 1–8, including options explicitly mentioned there. The Android-side tasks are in [resistor-scanner-android/TODO.md](https://github.com/Persie0/resistor-scanner-android/blob/main/TODO.md).

**Pipeline:** rectified resistor body crop → asymmetric CNN → vertical attention pooling → tiny Transformer over columns → dense column and band-centre heads + global direction/count heads → constrained decoder → top-3 electrical values and confidence → multi-frame fusion in Android.

## 1. Architecture (proposal §§1–3)

- [ ] **M01 — 64×256 input.** Accept an already rectified RGB resistor-body crop, shape `[B,3,64,256]`.
- [ ] **M02 — Exact asymmetric CNN.** Start with 3→24 conv (3×3, stride 2), BatchNorm and Hardswish; implement inverted-residual blocks with expansion factor 3, depthwise **3×5** kernels, pointwise projection and residual connection when shapes match. Follow the proposed sequence: `IR(24,32,(2,1))`, `IR(32,32)`, `IR(32,64,(2,1))`, `IR(64,64)×2`, `IR(64,128,(2,1))`, `IR(128,128)×2`. Preserve width at **128 columns** and reduce height to **4 rows**; resulting features `[B,128,4,128]`.
- [ ] **M03 — Four-head vertical attention pooling.** Apply a 1×1 conv for **4 head scores** per column; softmax over height; weight and sum each head's channel group, producing **128 column tokens of dimension 128**. Allow heads to use or ignore glare/edge shadows.
- [ ] **M04 — Context token and Transformer.** Prepend a learned global context token, add learned positional embeddings to all **129 tokens** and apply **3 Transformer encoder layers** (dimension **128**, **4 attention heads**, feedforward dimension **256**, dropout **0.1**, GELU, pre-norm), followed by final LayerNorm.
- [ ] **M05 — Dense heads.** Per column output `seg [B,128,14]`: **outside/lead**, **body**, plus **12 resistor colors**, and a separate `ctr [B,128]` 1-D band-centre heatmap (logits).
- [ ] **M06 — Global heads.** From the context token output `dir [B,2]` for first-band left/right and `cnt [B,4]` for **3, 4, 5 or 6 bands**.
- [ ] **M07 — Model size and compute.** Measure actual parameter count, MACs and inference runtime; compare to the proposal's approximate **0.7M parameters / 0.3 GMACs** target.

## 2. Labels and training (proposal §5)

- [ ] **M08 — Dense labels.** Rasterize annotated band x-ranges into **128 columns** with outside/lead, body and 12 band colors; mark **±1 column at band edges as ignore**.
- [ ] **M09 — Centre/direction/count labels.** Place Gaussian peaks (**σ ≈ 1 column**) at true band centres; annotate which end is the first band and whether there are 3/4/5/6 bands. Reverse spatial labels and first-band direction under horizontal flips.
- [ ] **M10 — Classification and centre losses.** Class-weighted column cross-entropy with label smoothing **0.05**; **CenterNet focal loss** on band-centre heatmap.
- [ ] **M11 — Global and consistency losses.** Add **0.5× direction cross-entropy**, **0.2× count cross-entropy**, and **0.5× KL consistency** between two independently photometrically augmented views of the same resistor.
- [ ] **M12 — Color-preserving photometric augmentation.** Apply random correlated white-balance/warm–cool gains and slight tint in **linear RGB**, exposure **±1.5 EV**, gamma and moderate saturation adjustments. Include defocus/motion blur, downscale→upscale, JPEG artifacts, sensor noise, synthetic glare stripes and soft shadow gradients. **Never use grayscale or cross-image MixUp/CutMix**; no hue jitter by default (at most **±3°** if tested).
- [ ] **M13 — Geometry augmentation.** Horizontal/vertical flips; **±4°** rotations, **0.8–1.25×** horizontal stretch and end-crop jitter, with transformed band annotations/heatmap targets. Horizontal flip must also transform direction label.
- [ ] **M14 — Optimize from scratch.** Train without pretrained weights using **AdamW**, initial LR approximately **2e-3**, weight decay **0.05**, cosine LR scheduling, EMA, **150–300 epochs**, and **effective batch size ≥256**.
- [ ] **M15 — Leakage-safe validation.** Split by **physical resistor and capture session**, not by random image/frame; include multiple phone brands and lighting conditions.
- [ ] **M16 — Report proposed metrics.** Exact decoded resistor-value accuracy, per-color confusion matrix, direction accuracy and confidence calibration; also inspect exact band sequences, centre/count accuracy and ambiguous results.
- [ ] **M17 — Optional augmentations.** Test swapping digit-band stripe regions **within the same image** (keeping lighting consistent and correcting labels), and test **3 optional additional channels** of `log(RGB) − log(median RGB of crop)` alongside normal RGB.

## 3. Electrical decoder (proposal §4)

- [ ] **M18 — Locate bands from the centre heatmap.** Sigmoid centre logits, find peaks above approximately **0.30**, then sum the 12 color probabilities over **±2 columns** around each peak and normalize.
- [ ] **M19 — Recover weak/missing peaks.** Also test weak peaks in the **0.15–0.30** range and optional peak omission/insertion; score alternative counts using the count-head distribution. Handle adjacent same-colored bands.
- [ ] **M20 — Enumerate both directions.** At each candidate band consider its **top 3 colors**, enumerate combinations in left-to-right and right-to-left order, and score with log color probabilities, learned `dir` probability, count probability and a soft prior.
- [ ] **M21 — Resistor-code constraints.** Support **3/4/5/6 bands**, correct digit/multiplier/tolerance behavior and **temperature coefficient for 6 bands**; disallow gold/silver as significant digits. Support **single black band = 0 Ω** as the proposal's special case.
- [ ] **M22 — Soft E-series prior.** Use **E24** for 3/4-band, **E24 ∪ E96 ∪ E192** for 5/6-band candidates, tolerance-color frequency priors (gold/silver in many 4-band codes, brown/red in many 5-band codes), optionally dataset value frequencies. Keep non-E-series readings possible with a nonzero floor; tune weight on validation.
- [ ] **M23 — Top-3 posterior outputs.** Log-sum-exp merge probabilities from different color sequences/directions that resolve to the same electrical result; normalize and return **top 3 electrical values and probabilities/confidence**.
- [ ] **M24 — Decoder coverage.** Test reverse direction, shared-value hypotheses, ambiguous readings, repeated same colors, missing/extra bands, gold/silver, 0 Ω, 3/4/5/6 bands and correct temperature coefficient handling.

## 4. Export, deployment assets and evaluation (proposal §§5–8)

- [ ] **M25 — Baseline from the proposal.** Train a **plain CRNN+CTC baseline** on the same data, then evaluate whether the proposed attention, heatmap and decoder prior each improve performance.
- [ ] **M26 — ONNX and mobile export.** Export PyTorch → ONNX → **LiteRT/TFLite** for Android; preserve four output tensors and their ordering, input dimensions and normalization. Use FP32/FP16; if `nn.TransformerEncoder` does not export, re-express the **same three layers** with Linear + softmax. Test conversion and outputs.
- [ ] **M27 — Quantization investigation (conditional).** If using INT8, reassess narrow color boundaries, particularly brown/red/orange. Retain an accurate FP32/FP16 path.
- [ ] **M28 — Real-world evaluation.** Use unseen physical resistors, capture sessions, multiple phone brands, unusual white balance, lighting, reflections, blur, shadows and backgrounds. Report exact-value accuracy, per-color confusion, direction accuracy and calibrated confidence.
- [ ] **M29 — License compliance.** Verify permissive/commercial-use-compatible components and dataset provenance; check any upstream segmentation model, particularly if YOLO/Ultralytics is used.
- [ ] **M30 — CTC-only fallback if annotations lack band positions.** If only an ordered color sequence is available, replace dense positional heads with a CTC head using body as blank. Keep positional supervision when x-ranges exist.
- [ ] **M31 — Support user-corrected examples.** Define how a user-selected correct top-3 reading can be collected as a labeled hard example for subsequent training.

## Proposal completion criterion

The new **64×256, 128-column, four-head** network, **four output heads**, proposed labels/losses/augmentation/training, **bidirectional E-series posterior top-3 decoder** and mobile export are implemented and connected to the Android tasks. Optional items remain optional exactly where the proposal calls them optional.
