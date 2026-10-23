# TODO — exact 1:1 implementation of the proposed 1-D resistor band reader

Status audited against `main`: **2026-10-23**. Source specification: user-provided **“A 1-D band reader for resistor color codes”**, sections **1–8** (pipeline, architecture, decoder, training, Android deployment, licensing and CTC fallback).  
Companion Android checklist: [resistor-scanner-android/TODO.md](https://github.com/Persie0/resistor-scanner-android/blob/main/TODO.md).

## Decision: implement the proposal, not an approximation

**Target:** `rectified 64×256 RGB body crop → asymmetric CNN → 4-head vertical attention pooling → 3-layer column Transformer + global context token → 14-class column logits + 1-D center heatmap + reading-direction head + 3/4/5/6-count head → bidirectional grammar/E-series posterior decoder → top-3 values → temporal probability fusion on Android`.

This is **not** a request merely to optimize BandNet V2 or Mobile1D. Both preexisting models should remain reproducible baselines, but **neither fulfills the proposed architecture**. Do not silently substitute the Mobile1D TCN, old six-slot output contract or Android color voting for the requested components.

Legend: `[x]` means present in existing code; `[ ]` means not yet complete **for the 1:1 target**. “Partial” means existing related work is reusable, not that the requirement is done.

## 1. Exact network architecture (proposal §§1–3)

- [ ] **M01 — Separate canonical model implementation.** Add a new selectable architecture (e.g. `bandnet_1d_proposed`) alongside V2 and Mobile1D; register it in `runtime.py`, configs, exporter and tests. Its canonical **FP32 RGB input is [B,3,64,256]**; keep the trained input contract explicit.
- [ ] **M02 — Asymmetric MobileNetV2-style convolution stack, matching the supplied model.** Initial `Conv2d(3→24,3,stride=2)`, BN + Hardswish; inverted residuals `24→32 (2,1)`, 32→32, `32→64 (2,1)`, 64→64 ×2, `64→128 (2,1)`, 128→128 ×2. Inverted blocks: expansion 3, depthwise kernel **3×5**, BN/Hardswish, linear 1×1 projection, skip where possible. Result **[B,128,4,128]**, conserving horizontal detail while shrinking height by 16 and width by 2.
- [ ] **M03 — Four-head vertical attention pooling.** `vattn=Conv2d(128,4,1)`; softmax **over four vertical rows** for each head/column; attend to grouped channels and reshape into **128 column tokens of 128 dimensions**. Do not replace with the single learned/mean height-pooling module in V2/Mobile1D.
- [ ] **M04 — Global context + column Transformer.** Learned `[1,1,128]` context token (lighting/body reference), learned position embeddings `[1,129,128]`; concatenate 1 context + 128 columns. **Three** pre-norm Transformer encoder layers: d_model=128, **four** self-attention heads, FFN=256, dropout=0.1, GELU and final LayerNorm. Every column must attend across the resistor.
- [ ] **M05 — Precisely four proposed output heads, *not* old six slot heads.** For columns: `seg [B,128,14]` representing **outside/lead (0), body (1), and 12 colors (2–13)**; `ctr [B,128]` band-center logits. From context: `dir [B,2]` (“band #1 on left/right”) and `cnt [B,4]` (3/4/5/6 bands). Return/document exact names, order, activation/logit conventions and geometry. Optional compatibility adapters must not redefine the canonical model.
- [ ] **M06 — Quantify model size/compute.** Report actual learnable parameters, MACs, ONNX/LiteRT ops, CPU/GPU memory and runtime. Proposal's **~0.7 million parameters, ~0.3 GMACs** are estimates to measure, not guaranteed outcomes. Keep architecture exact while assessing this target.

## 2. Labels and target generation (proposal §5)

- [x] **Existing foundation:** JSONL resistor/band bounding boxes, optional body polygons, geometric rectification and resistor/session-grouped splitting already exist in `data/schema.py`, `data/dataset.py` and `data/split.py`.
- [ ] **M07 — New 128-column segmentation labels.** Rasterize outside/lead vs true body vs 12 band colors; set **±1 column around each band edge to ignore index**. Existing `dense_target` is **13 classes (12 colors + background)** and does *not* distinguish body from outside. Source body annotations are optional, so define a principled mask/weak-label/ignore strategy; never hallucinate outside/body ground truth from band boxes alone.
- [ ] **M08 — Band-center heatmap.** Gaussian target **σ≈1 output column** at every actual band center, including adjacent identical-colored bands. Validate peak-center alignment after crop transforms.
- [ ] **M09 — Global direction/count targets.** Encode left/right **first significant band**, transform labels under horizontal flips, preserve vertical flips, and map **3→0, 4→1, 5→2, 6→3**. Do not infer physical reading direction from arbitrary crop order alone: obtain reliable annotation or mark genuinely undetermined cases as ignored for direction loss.
- [ ] **M10 — Correct color and edge supervision.** Explicit order `black,brown,red,orange,yellow,green,blue,violet,gray,white,gold,silver`; preserve silver/gold material cues and avoid treating masked black backgrounds as black bands. Include realistic 3/4/5/6 and repeated-same-color cases.

## 3. Training recipe (proposal §5)

- [ ] **M11 — Match required losses.** Class-weighted column cross-entropy with **0.05 label smoothing** and ignored ±1 edges; **CenterNet-style focal loss** on the center heatmap; **0.5 × direction CE**, **0.2 × count CE**; **0.5 × KL-consistency** for two independently photometrically augmented views. Old slot-existence/width/order/CTC and embedding-cosine objectives are *not* the canonical proposed loss.
- [ ] **M12 — Match exact photometric augmentation intent.** Correlated white balance/warm–cool shifts and slight tint **in linear RGB**, exposure **±1.5 EV**, gamma and moderate saturation. Include defocus blur, **motion blur**, downsample→upsample, JPEG artifacts, sensor noise, synthetic specular glare stripe and soft shadow gradients. No grayscale, no cross-image MixUp/CutMix, and **no hue jitter** by default (absolute maximum **±3°** in a controlled experiment). Existing `PhotometricAugment` covers several but not all and uses channel gains in its current working space.
- [ ] **M13 — Match geometry augmentation.** Horizontal flip with mirrored *ordered labels and direction*, vertical flip, **±4° rotation**, **horizontal stretch 0.8–1.25** and independent end-crop jitter, always remapping segmentation and Gaussian center targets; avoid lost-band target corruption.
- [ ] **M14 — Optimizer and training duration.** Train **from scratch** (no ImageNet/pretrained weights), **AdamW ~2e-3**, weight decay **0.05**, cosine LR decay, EMA, **150–300 epochs** and **effective batch ≥256** (gradient accumulation if a single device cannot fit that batch). Make run reproducible and resumable. Existing trainer's AdamW/cosine/EMA can be reused but Mobile1D `epochs=60, batch=32, lr=5e-4` is not this recipe.
- [ ] **M15 — Optional proposal augmentations, explicitly separated.** A guarded within-image swap of *digit* stripe regions (preserve lighting, update labels) and an optional **three-channel** body-relative `log(RGB) − log(median RGB)` input ablation. Canonical model stays three-channel RGB until evidence supports a versioned six-channel variant. Never accidentally use invalid digit/tolerance placement.
- [ ] **M16 — Data split and device variation.** Group by *physical resistor* AND capture session/family; use multiple camera brands, lighting, body colors and capture conditions; freeze splits before tuning. Current grouping logic is reusable, but prove every new evaluation set is leakage-free.
- [ ] **M17 — Class coverage and annotation audit.** Expand real annotated data for **six-band and silver** (README documents just four six-band examples and no silver in rres.v4), plus rare whites, tolerance/tempco, black/adjacent equal stripes, glare, shadows and varying phone AWB. Report evidence/limits; never claim general performance from synthetic-only test examples.

## 4. Proposed constrained probabilistic decoder (proposal §4)

- [x] **Existing foundation:** both-direction resistor grammar, ambiguity reporting, top-k *slot-color* search and E24/E96 soft preference exist in `decoder.py`/`decoding.py`. This is **not** yet the requested **heatmap-based, per-value posterior decoder**.
- [ ] **M18 — Peak finding and band distributions.** Sigmoid center heatmap; locate 1-D local maxima at **threshold ~0.30** and form each candidate band's 12-color distribution from **±2 columns** around the peak of the 14-way column softmax (exclude outside/body before renormalizing). Add weak-peak alternatives for **0.15–0.30** and missing/phantom peak handling weighted by `cnt`.
- [ ] **M19 — Enumerate candidate codes in *both* directions.** Per detected peak take up to **top three** colors and enumerate legal sequences/counts; score with sum of log color probabilities, the corresponding learned direction softmax, band-count probabilities and soft prior. Preserve explicitly ambiguous directions when evidence is insufficient.
- [ ] **M20 — Electrical grammar exactly, including edge cases.** Digits 0–9, gold/silver as multipliers (not significant digits); valid 3/4/5/6-band tolerance and **sixth-band temperature coefficient**; standard **3-band 20% implicit tolerance**; prohibit invalid digit/tolerance positions; add the proposal's **single-black-band 0 Ω** exceptional decoder case. **Do not copy the proposal's short `to_value` snippet literally:** it omits sixth-band tempco validation and rejects 0 Ω by design; implement these explicit surrounding requirements correctly.
- [ ] **M21 — Soft, non-exclusive electrical priors.** For 3/4 bands use **E24**; for 5/6 bands **E24 ∪ E96 ∪ E192**. Add tolerance-color frequencies by band count (gold/silver often 4-band, brown/red often 5-band) and optionally observed dataset-value frequencies. Non-E-series/rare valid readings get a **nonzero floor**. Tune prior weight on validation rather than forcing common values.
- [ ] **M22 — Posterior aggregation and top three distinct resistor values.** Merge identical electrical candidates from different color/direction paths with **log-sum-exp**, normalize over candidate outcomes and return **top 3 values with posterior/confidence** (keep tolerance and tempco to avoid falsely merging electrically distinct outcomes). Also expose uncertainty/abstain and coverage. No “best single slot sequence masquerading as top-3 values.”
- [ ] **M23 — Decoder tests and calibration.** Gold/silver vs yellow/gray, brown/red/orange, identical adjacent bands, both orientations, weak/missed/spurious peaks, 3–6 bands, six-band tempco, single black 0 Ω, invalid grammar and non-E-series values. Evaluate **exact value, per-color confusion, direction accuracy, calibration, top-3 recall, ambiguous/abstained fraction and coverage**.

## 5. Baselines, export and mobile contract (proposal §§5–8)

- [ ] **M24 — Plain CRNN+CTC baseline.** Train/evaluate on the *same fixed splits* before judging added attention, heatmap or priors; run ablations showing what each proposed part adds. **CTC-only as the primary model is the fallback only when band positions are unavailable**; current data already has band boxes.
- [ ] **M25 — Robust held-out and OOD evaluation.** Exact resistance plus tolerance/tempco, exact band sequence, per-color confusion matrix, direction, count, calibration, per-phone and real-world glare performance. Include multiple phone brands; track predictions vs human-confirmed ground truth, not just loss. Report selective accuracy vs coverage.
- [ ] **M26 — Export exact four-head network.** PyTorch→ONNX→**LiteRT/TFLite** for the current Android runtime, FP32 reference and optionally FP16 after accuracy checks. If `TransformerEncoder` export fails, implement **mathematically equivalent explicit Linear/softmax layers**; do not silently replace the specified Transformer with a TCN/dilated conv. Reject Flex/custom ops in the production artifact.
- [ ] **M27 — Release reproducible artifacts.** Include `*.tflite`, ONNX/reference output, exact `[1,3,64,256]` input normalization, four tensors' shapes/names/output indices, 12-color order, decoder/prior version, dataset/split provenance, training config, hashes and numerical parity report. Android currently expects **six V2 outputs**; give it a new versioned contract rather than reusing V2 names.
- [ ] **M28 — Licensing review.** Train weights from scratch, record source-data licenses, verify segmenter and all dependencies permit commercial closed-source deployment. **Do not introduce Ultralytics AGPL weights/code without a compatible commercial license**. Existing PP-LiteSeg must be independently checked for model/weight provenance.
- [ ] **M29 — Opt-in hard-example loop.** Define import format for user corrections of top-3 predictions (with camera/illumination/session metadata only when permitted), quality checks and deduplication, so examples can be retrained without contaminating fixed test splits.
- [ ] **M30 — End-to-end signoff with Android.** Real-device LiteRT-vs-PyTorch parity for all four heads and final decoded values; compare real-phone accuracy, startup memory and scan speed before promoting this model. Proposal's “tens of fps” expectation is **not measured**. Coordinate with Android checklist A01–A17.

## Source-to-work crosswalk (complete)

| Uploaded proposal section | Required tasks |
| --- | --- |
| §1 Pipeline | M01–M05, M18–M22, Android A01–A09 |
| §2 Why CNN/attention/context/direction | M02–M04, M09–M10, M17, M23–M25 |
| §3 PyTorch architecture/code | M01–M06 |
| §4 Decoder, prior, robustness and 0 Ω | M18–M23 |
| §5 Labels, losses, augmentation, optimizer, validation | M07–M17, M23–M25 |
| §6 Android export/runtime, fusion, cropping, AWB/torch, corrections | M26–M30, Android A01–A17 |
| §7 License constraints | M28 and Android A16 |
| §8 CTC fallback and CRNN baseline | M24 |

### Important implementation caveats

1. The supplied proposal's **illustrative code is not an authoritative verified implementation**. Keep the architecture and objectives faithful while correcting coding mistakes and incomplete electrical rules (especially 6-band tempco and 0 Ω).
2. The proposed network has a **3/4/5/6 count head**, so a one-black-band 0 Ω reading requires an explicit **decoder exception** outside ordinary count-head inference.
3. Outside/body pixel labels require body geometry or reliable segmentation/weak supervision; the current band-only box annotations alone cannot always establish them. Do not train fabricated labels.
4. A confidence above **0.90** should be interpreted as a **validated/calibrated posterior**, not the previous app's per-band heuristic score.
5. No checkbox above is marked complete merely because V2 or Mobile1D contains a vaguely similar capability.
