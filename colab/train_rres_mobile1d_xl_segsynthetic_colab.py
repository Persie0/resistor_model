"""Train ~25 MB Mobile1D XL on real+synthetic SEGMENTER-output crops.

Requires running prepare_mobile1d_segsynthetic_colab.ipynb first.
Separate XL experiment; never touches the existing small or standard XL checkpoints.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Python Colab bootstrap missing {path}")
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def configure_segmented_training(base, xl):
    xl.configure_xl(base)
    # This is a THIRD independent training run, so old XL weights and metrics
    # remain intact for an unbiased performance comparison.
    base.RUN_DIR = base.DRIVE_MOUNT / "MyDrive" / "resistor_model" / "r1-mobile1d-xl-segsynthetic-colab"
    base.CONFIG_PATH = base.WORK / "r1-mobile1d-xl-segsynthetic-colab.yaml"
    base.ZIP_PATH = base.WORK / "resistor-bandnet-mobile1d-xl-segsynthetic.zip"
    base.DATA_ROOT = Path("/content/mobile1d_segsynthetic_prepared")
    base.MOBILE1D_COLAB_VERSION = "2026-10-16-v1-xl-segsynthetic-25mb"
    # Preserve the independent run's filename/epoch semantics.
    original_cfg = base.build_config

    def segmented_config(run_dir, resume, teacher):
        config = original_cfg(run_dir, resume, teacher)
        config["data"]["pre_rectified"] = True
        # Allow only label-consistent flips at loader time. Synthetic crops
        # already underwent rotation/shear BEFORE their segmenter pass.
        config["data"]["geometric_augment"] = True
        config["data"]["jitter_strength"] = 0.0
        config["train"]["data_recipe"] = "r1-segsynthetic-lraspp-bandlabels-v1"
        return config

    base.build_config = segmented_config

    def restore_prepared_segmentation_data():
        from resistor_model.tools.mobile1d_segsynthetic import restore
        summary = restore(base.DATA_ROOT)
        min_coverage = min(summary["real_source_coverage"].values())
        if min_coverage < 0.7:
            raise RuntimeError(
                f"Real segmentation coverage is only {min_coverage:.1%}. "
                "Investigate low-mask-IoU failures rather than train on "
                "a strongly selected validation/test subset."
            )
        print(
            "[segsynthetic] pre-rectified dataset ready. "
            "Real held-out images have NO synthetic siblings.",
            flush=True,
        )

    base.prepare_dataset = restore_prepared_segmentation_data

    # Segmentation pass was already done in the first notebook; this training
    # notebook runs only the exact prepared crops, no duplicate detection.
    original_train = base.train_and_evaluate

    def train_with_loader_tests():
        base.run([
            base.sys.executable, "-m", "pytest", "-q",
            "tests/test_mobile1d_segsynthetic.py",
        ], cwd=base.MODEL_REPO, heartbeat_seconds=base.HEARTBEAT_SECONDS)
        original_train()

    base.train_and_evaluate = train_with_loader_tests


def main():
    here = Path(__file__).resolve().parent
    base = _load_file(here / "train_rres_mobile1d_colab.py", "mobile1d_train_base")
    xl = _load_file(here / "train_rres_mobile1d_xl_colab.py", "mobile1d_train_xl")
    configure_segmented_training(base, xl)
    print("[xl-segsynthetic] standalone 120-epoch run; "
          "model ~25 MB FP32; output folder "+str(base.RUN_DIR),flush=True)
    base.main()


if __name__ == "__main__":
    main()
