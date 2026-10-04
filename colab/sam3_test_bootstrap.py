"""Minimal Colab bootstrap for testing SAM 3 on one resistor image.

Keeps Colab's existing NumPy installation untouched. Upstream SAM 3 currently has a
NumPy <2 dependency constraint, which can downgrade NumPy in Python 3.13 Colab and
break already-installed binary wheels. We apply the upstream NumPy-2 compatibility
change locally, install SAM 3 with --no-deps, and install only its non-NumPy runtime
dependencies.

The official SAM 3 checkpoint is gated on Hugging Face. This bootstrap resolves the
checkpoint explicitly so authorization errors are clear, and also supports an official
local checkpoint via /content/sam3.pt or the SAM3_CHECKPOINT_PATH environment variable.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

SAM3_DIR = Path("/content/sam3")
SAM3_REPO_ID = "facebook/sam3"
SAM3_ACCESS_URL = "https://huggingface.co/facebook/sam3"
SAM3_DEFAULT_CHECKPOINT = Path("/content/sam3.pt")
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
    """Return an official SAM 3 checkpoint path or raise an actionable access error."""
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

    log(f"[checkpoint] checking Hugging Face access to {SAM3_REPO_ID}")
    try:
        checkpoint = hf_hub_download(
            repo_id=SAM3_REPO_ID,
            filename="sam3.pt",
        )
    except GatedRepoError as exc:
        raise RuntimeError(
            "SAM 3 checkpoint access is not enabled for the currently logged-in "
            "Hugging Face account. Open the model page while logged into the same "
            f"account and accept/request access:\n  {SAM3_ACCESS_URL}\n\n"
            "After Hugging Face/Meta grants access, rerun this bootstrap cell. "
            "You do not need to reinstall or restart Colab.\n\n"
            "Alternative if you already have the official checkpoint: upload "
            "sam3.pt to /content/sam3.pt, or set SAM3_CHECKPOINT_PATH to its path."
        ) from exc
    except HfHubHTTPError as exc:
        raise RuntimeError(
            f"Could not download the official SAM 3 checkpoint from {SAM3_REPO_ID}. "
            f"Check Hugging Face login/access at {SAM3_ACCESS_URL}. Original error: {exc}"
        ) from exc

    log(f"[checkpoint] official SAM 3 checkpoint ready: {checkpoint}")
    return checkpoint


log("=" * 72)
log("SAM 3 single-image test bootstrap")
log("=" * 72)
log(f"[bootstrap] Python: {sys.version.split()[0]}")

log("[bootstrap 1/6] Installing non-NumPy runtime dependencies")
run([
    sys.executable,
    "-m",
    "pip",
    "install",
    "-q",
    "huggingface_hub",
    "matplotlib",
    "pillow",
    "timm>=1.0.17",
    "tqdm",
    "ftfy==6.1.1",
    "regex",
    "iopath>=0.1.10",
    "typing_extensions",
])

log("[bootstrap 2/6] Preparing official SAM 3 repository")
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

log("[bootstrap 3/6] Applying Colab NumPy-2 compatibility patch")
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
from huggingface_hub import hf_hub_download, login, notebook_login
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

log(f"[bootstrap] NumPy kept at runtime version: {np.__version__}")
if not torch.cuda.is_available():
    raise RuntimeError(
        "No CUDA GPU detected. In Colab choose Runtime -> Change runtime type -> GPU."
    )

log(
    f"[bootstrap] GPU: {torch.cuda.get_device_name(0)} | "
    f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB"
)

log("[bootstrap 4/6] Authenticating with Hugging Face")
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

log("[bootstrap 5/6] Resolving official SAM 3 checkpoint")
sam3_checkpoint = resolve_sam3_checkpoint()

log("[bootstrap 6/6] Loading SAM 3 image model")
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

log(f"[bootstrap] SAM 3 ready in {time.monotonic() - started:.1f}s")
log("[bootstrap] You can now run the single-image upload test cell.")
