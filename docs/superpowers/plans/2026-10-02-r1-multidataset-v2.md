# r1 Multi-Dataset V2 Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train from all three r1 COCO datasets with leakage-safe grouping and integrate the validated V2 training/decoding improvements.

**Architecture:** Convert each COCO archive into the existing canonical manifest schema, merge at the manifest layer, and perform one grouped split across normalized source identities. Keep V1 behavior compatible while adding optional V2 augmentation/loss/model/decoder capabilities controlled by config.

**Tech Stack:** Python 3.12, PyTorch, OpenCV, torchvision (optional V2 ResNet backbone), pytest, GitHub Actions, Google Colab.

**Spec:** `docs/superpowers/specs/2026-10-02-r1-multidataset-v2-design.md`

## Global Constraints
- Preserve current v1 configuration behavior by default.
- Final r1 training source is exactly the three release COCO archives.
- Final train/val/test split must be source-group disjoint across datasets.
- Geometry direction heuristic remains disabled by default.
- Dataset images remain local in Colab; persistent outputs remain on Drive.

## Review Focus
- Same source image exported into multiple datasets must not cross splits.
- COCO datasets without resistor-body annotations must reject ambiguous multi-resistor images.
- Horizontal flips must update band geometry/order consistently.
- Repeated color bands must remain valid under CTC.
- Old checkpoints/configs must still construct V1 and load without new required fields.

---

### Task 1: Unified COCO importer
**Files:** create `src/resistor_model/tools/import_roboflow_coco.py`; create `tests/test_roboflow_coco_import.py`.
**Interfaces:** CLI accepts multiple `--root` values and one `--output`; produces canonical JSONL plus optional report JSON.
- [ ] Write tests for class normalization, body aliases, no-body single-resistor inference, cross-dataset source grouping, and ambiguous-image rejection.
- [ ] Run the tests and verify they fail because the importer is absent.
- [ ] Implement the importer using the existing schema.
- [ ] Run importer tests and full pytest.

### Task 2: Geometric augmentation
**Files:** create `src/resistor_model/data/geom_aug.py`; modify `src/resistor_model/data/geometry.py`, `src/resistor_model/data/dataset.py`; extend dataset tests.
- [ ] Write failing tests for jitter identity/bounds, bbox transforms under flips, and dataset config propagation.
- [ ] Implement jitter/flips and training-only dataset hooks.
- [ ] Run targeted and full pytest.

### Task 3: Auxiliary losses and decoder metrics
**Files:** modify `src/resistor_model/losses.py`, `src/resistor_model/metrics.py`; create `src/resistor_model/decoding.py`; add tests.
- [ ] Write failing tests for CTC, KL consistency, label smoothing compatibility, constrained valid decoding, count marginalization, and extra metric keys.
- [ ] Implement uploaded loss/decoder/metric improvements with all new behavior opt-in.
- [ ] Run targeted and full pytest.

### Task 4: Optional V2 model/runtime
**Files:** create `src/resistor_model/models/bandnet_v2.py`; modify `runtime.py`, `config.py`, `pyproject.toml`; extend model/runtime tests.
- [ ] Write failing tests for V1 default, V2 construction, V2 output shapes, and invalid backbone/config handling.
- [ ] Implement V2 and optional torchvision extra/dependency.
- [ ] Run targeted and full pytest.

### Task 5: Training and Colab integration
**Files:** modify `train.py`, `colab/train_rres_v4_colab.py`, notebook docs/config; extend Colab/train tests.
- [ ] Write failing tests for geometric config, new loss keys, three r1 asset URLs, combined importer command, and V2 optimizer/backbone LR handling if enabled.
- [ ] Implement the r1 multi-dataset training config and Drive-resume-compatible workflow.
- [ ] Run targeted and full pytest plus compileall.

### Task 6: Real-data gate
**Files:** update public Playground validation workflow only.
- [ ] Download all three r1 release assets.
- [ ] Import them with the new COCO importer and create one grouped split.
- [ ] Assert no source-group leakage between splits and print dataset statistics.
- [ ] Run a real one-batch train/validation smoke with the new config.
- [ ] Review final branch diff and merge only if the exact head passes the full gate.
