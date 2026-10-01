# Resistor Band Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a complete, testable training/evaluation/export repository for the rectified 1-D ResistorBandNet architecture.

**Architecture:** Canonical JSONL annotations feed grouped splitting and geometric rectification. Rectified crops produce dense and ordered-slot targets consumed by a dual-branch ConvNeXt-style + Transformer network, trained with structured losses and evaluated with sequence/value metrics.

**Tech Stack:** Python 3.10+, PyTorch, torchvision, NumPy, OpenCV, Pillow, PyYAML, pytest.

**Spec:** `docs/superpowers/specs/2026-10-01-resistor-band-training-design.md`

## Tasks

- [x] Data contract, grouped splitting, geometry and deterministic resistor decoder.
- [x] Rectified dataset targets and physically plausible color-safe augmentation.
- [x] ResistorBandNet, chromatic branch, ordered queries and structured losses.
- [x] Metrics, config, training with EMA/checkpoints, evaluation and ONNX export.
- [x] Dataset audit, split and YOLO-conversion utilities.
- [x] Unit tests, synthetic end-to-end smoke training, example config and documentation.

## Verification

Local verification completed with `pytest -q` (17 passing tests), `python -m compileall -q src`, and a synthetic `resistor_model.train --smoke` run that performed rectification, augmentation, forward/backward, EMA update, validation and checkpoint writing.
