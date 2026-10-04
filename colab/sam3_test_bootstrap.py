"""Minimal Colab bootstrap for testing SAM 3 on one resistor image.

This intentionally does not download or process any resistor datasets. Execute it in a
notebook cell, then run the image-upload test cell. The full dataset workflow can be
started later in a separate cell.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

SAM3_DIR = Path("/content/sam3")
SAM3_BOOTSTRAPPED = False


def log(message=""):
    print(message, flush=True)


def run(command, cwd=None):
    log("+ " + " ".join(map(str, command)))
    subprocess.run(
        list(map(str, command)),
        cwd=str(cwd) if cwd else None,
        check=True,
    )


log("=" * 72)
log("SAM 3 single-image test bootstrap")
log("=" * 72)

log("[bootstrap 1/4] Installing test dependencies")
run([
    sys.executable,
    "-m",
    "pip",
    "install",
    "-q",
    "-U",
    "huggingface_hub",
    "matplotlib",
    "pillow",
    "numpy",
])

log("[bootstrap 2/4] Preparing official SAM 3 repository")
if not SAM3_DIR.is_dir():
    run([
        "git",
        "clone",
        "--depth",
        "1",
        "https://github.com/facebookresearch/sam3.git",
        str(SAM3_DIR),
    ])
else:
    log("[bootstrap] SAM 3 repository already exists; updating")
    run(["git", "pull", "--ff-only"], cwd=SAM3_DIR)

run([
    sys.executable,
    "-m",
    "pip",
    "install",
    "-q",
    "-e",
    str(SAM3_DIR),
])
if str(SAM3_DIR) not in sys.path:
    sys.path.insert(0, str(SAM3_DIR))

import numpy as np
import torch
from google.colab import files
from huggingface_hub import login, notebook_login
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

if not torch.cuda.is_available():
    raise RuntimeError(
        "No CUDA GPU detected. In Colab choose Runtime -> Change runtime type -> GPU."
    )

log(
    f"[bootstrap] GPU: {torch.cuda.get_device_name(0)} | "
    f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB"
)

log("[bootstrap 3/4] Authenticating with Hugging Face")
token = None
try:
    from google.colab import userdata

    token = userdata.get("HF_TOKEN")
except Exception:
    token = None

if token:
    login(token=token, add_to_git_credential=False)
    log("[bootstrap] authenticated from Colab secret HF_TOKEN")
else:
    log("[bootstrap] HF_TOKEN secret not found; opening Hugging Face login")
    notebook_login()

log("[bootstrap 4/4] Loading SAM 3 image model")
started = time.monotonic()
model = build_sam3_image_model()
model.eval()
processor = Sam3Processor(
    model=model,
    device="cuda",
    confidence_threshold=0.15,
)
SAM3_BOOTSTRAPPED = True

log(f"[bootstrap] SAM 3 ready in {time.monotonic() - started:.1f}s")
log("[bootstrap] You can now run the single-image upload test cell.")
