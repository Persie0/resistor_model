"""Minimal Colab bootstrap for testing SAM 3.1 on one resistor image.

Uses the open AEmotionStudio/sam3.1 mirror instead of gated facebook/sam3 weights.
The mirror provides the upstream SAM 3.1 multiplex checkpoint; SAM 3.1 keeps the
image detector unchanged, so the detector weights can be loaded by
build_sam3_image_model().

Keeps Colab's existing NumPy installation untouched. Upstream SAM 3 currently has a
NumPy <2 dependency constraint, which can downgrade NumPy in Python 3.13 Colab and
break already-installed binary wheels. We apply the NumPy-2 compatibility change
locally and install SAM 3 with --no-deps.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

SAM3_DIR = Path("/content/sam3")
SAM3_REPO_ID = "AEmotionStudio/sam3.1"
SAM3_CHECKPOINT_FILE = "sam3.1_multiplex.pt"
SAM3_MIRROR_URL = "https://huggingface.co/AEmotionStudio/sam3.1"
SAM3_DEFAULT_CHECKPOINT = Path("/content/sam3.1_multiplex.pt")
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


def patch_sam3_numpy2_compat():
    """Apply the small upstream NumPy-2 compatibility change to the clone."""
    pyproject = SAM3_DIR / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    original = text
    text = text.replace('"numpy>=1.26,<2",', '"numpy>=1.26",')
    text = text.replace('"numpy==1.26",', '"numpy>=1.26",')
    if text != original:
        pyproject.write_text(text, encoding="utf-8")
        log("[bootstrap] relaxed upstream NumPy <2 constraint for Colab Python 3.13")
    else:
        log("[bootstrap] SAM 3 NumPy dependency already compatible; no constraint patch needed")

    visualizer = SAM3_DIR / "sam3" / "agent" / "helpers" / "visualizer.py"
    if visualizer.is_file():
        source = visualizer.read_text(encoding="utf-8")
        patched = re.sub(r"\bnp\.bool\b", "np.bool_", source)
        if patched != source:
            visualizer.write_text(patched, encoding="utf-8")
            log("[bootstrap] patched deprecated np.bool usage to np.bool_")


def resolve_sam3_checkpoint():
    """Return a local SAM 3.1 multiplex checkpoint path from the open mirror."""
    configured = os.environ.get("SAM3_CHECKPOINT_PATH", "").strip()
    if configured:
        checkpoint = Path(configured).expanduser()
        if not checkpoint.is_file():
            raise RuntimeError(
                f"SAM3_CHECKPOINT_PATH points to a missing file: {checkpoint}"
            )
        log(f"[checkpoint] using SAM3_CHECKPOINT_PATH: {checkpoint}")
        return str(checkpoint)

    if SAM3_DEFAULT_CHECKPOINT.is_file():
        log(f"[checkpoint] using local checkpoint: {SAM3_DEFAULT_CHECKPOINT}")
        return str(SAM3_DEFAULT_CHECKPOINT)

    log(
        f"[checkpoint] downloading {SAM3_CHECKPOINT_FILE} from open mirror "
        f"{SAM3_REPO_ID} (~3.5 GB)"
    )
    try:
        checkpoint = hf_hub_download(
            repo_id=SAM3_REPO_ID,
            filename=SAM3_CHECKPOINT_FILE,
        )
    except HfHubHTTPError as exc:
        raise RuntimeError(
            f"Could not download {SAM3_CHECKPOINT_FILE} from {SAM3_REPO_ID}. "
            f"Mirror: {SAM3_MIRROR_URL}. Original error: {exc}"
        ) from exc

    log(f"[checkpoint] SAM 3.1 checkpoint ready: {checkpoint}")
    return checkpoint


log("=" * 72)
log("SAM 3.1 single-image test bootstrap")
log("=" * 72)
log(f"[bootstrap] Python: {sys.version.split()[0]}")
log(f"[bootstrap] checkpoint source: {SAM3_REPO_ID}/{SAM3_CHECKPOINT_FILE}")

log("[bootstrap 1/5] Installing non-NumPy runtime dependencies")
run([
    sys.executable,
    "-m",
    "pip",
    "install",
    "-q",
    "huggingface_hub",
    "hf_xet",
    "matplotlib",
    "pillow",
    "timm>=1.0.17",
    "tqdm",
    "ftfy==6.1.1",
    "regex",
    "iopath>=0.1.10",
    "typing_extensions",
])

log("[bootstrap 2/5] Preparing official SAM 3 code")
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
    log("[bootstrap] SAM 3 repository already exists; resetting to latest main")
    run(["git", "fetch", "--depth", "1", "origin", "main"], cwd=SAM3_DIR)
    run(["git", "reset", "--hard", "origin/main"], cwd=SAM3_DIR)

log("[bootstrap 3/5] Applying Colab NumPy-2 compatibility patch")
patch_sam3_numpy2_compat()
run([
    sys.executable,
    "-m",
    "pip",
    "install",
    "-q",
    "-e",
    str(SAM3_DIR),
    "--no-deps",
])
if str(SAM3_DIR) not in sys.path:
    sys.path.insert(0, str(SAM3_DIR))

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError
from PIL import Image
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model

log(f"[bootstrap] NumPy kept at runtime version: {np.__version__}")
if not torch.cuda.is_available():
    raise RuntimeError(
        "No CUDA GPU detected. In Colab choose Runtime -> Change runtime type -> GPU."
    )

log(
    f"[bootstrap] GPU: {torch.cuda.get_device_name(0)} | "
    f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB"
)

log("[bootstrap 4/5] Resolving open SAM 3.1 checkpoint")
sam3_checkpoint = resolve_sam3_checkpoint()

log("[bootstrap 5/5] Loading SAM 3.1 detector as image model")
started = time.monotonic()
model = build_sam3_image_model(
    checkpoint_path=sam3_checkpoint,
    load_from_HF=False,
)
model.eval()
processor = Sam3Processor(
    model=model,
    device="cuda",
    confidence_threshold=0.15,
)
SAM3_BOOTSTRAPPED = True

log(f"[bootstrap] SAM 3.1 image model ready in {time.monotonic() - started:.1f}s")
log("[bootstrap] You can now run the single-image upload test cell.")
