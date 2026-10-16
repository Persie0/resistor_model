"""Train a >20x-capacity BandNet Mobile1D XL without overwriting the small model.

Run this file from the companion Colab notebook with a Tesla T4 or better.
The original Mobile1D run stays under r1-mobile1d-colab, including all weights.
The prepared r1 dataset ZIP and original downloaded source archives are SHARED,
and only a separate XL checkpoint/model-output directory is created.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SMALL_CHANNELS = 24
SMALL_FEATURES = 128
XL_CHANNELS = 48
XL_FEATURES = 640
XL_BATCH_SIZE = 8  # conservative starting point for 15 GiB Tesla T4
XL_EPOCHS = 120
XL_LR = 2.5e-4
MIN_PARAMETER_RATIO = 20.0
XL_RUN_NAME = "r1-mobile1d-xl-colab"


def configure_xl(base) -> None:
    """Set a separate model recipe on the shared Colab training pipeline."""
    cache_dir = base.RUN_DIR
    base.RUN_DIR = cache_dir.parent / XL_RUN_NAME
    base.CONFIG_PATH = base.WORK / "r1-mobile1d-xl-colab.yaml"
    base.ZIP_PATH = base.WORK / "resistor-bandnet-mobile1d-xl-colab.zip"
    # Keep the verified input data in its original shared Drive location.
    base.DRIVE_PREPARED_ARCHIVE = cache_dir / "prepared_dataset.zip"
    base.DRIVE_PREPARED_MANIFEST = cache_dir / "prepared_dataset.manifest.json"
    base.DRIVE_ARCHIVE_ROOT = cache_dir / "archives"

    base.MOBILE1D_COLAB_VERSION = "2026-10-16-v1-mobile1d-xl-21x"
    base.BASE_CHANNELS = XL_CHANNELS
    base.D_MODEL = XL_FEATURES
    base.BATCH_SIZE = XL_BATCH_SIZE
    base.EPOCHS = XL_EPOCHS
    # XL starts from fresh weights: use a single 120-epoch cosine cycle.
    base.LR_INITIAL_CYCLE_EPOCHS = None
    base.LR_EXTENSION_PEAK_FACTOR = 0.2
    base.DROPOUT = 0.15

    original_config = base.build_config

    def build_xl_config(run_dir, resume, teacher):
        config = original_config(run_dir, resume, teacher)
        config["train"]["lr"] = XL_LR
        config["train"]["warmup_epochs"] = 5
        config["train"]["effective_model_size"] = "mobile1d-xl"
        return config

    base.build_config = build_xl_config

    original_install = base.install_dependencies

    def install_and_check_capacity():
        original_install()
        from resistor_model.runtime import build_model

        small_cfg = build_xl_config(base.RUN_DIR, None, None)
        small_cfg["model"]["base_channels"] = SMALL_CHANNELS
        small_cfg["model"]["d_model"] = SMALL_FEATURES
        small_cfg["model"]["dropout"] = 0.1

        xl_cfg = build_xl_config(base.RUN_DIR, None, None)
        small = build_model(small_cfg)
        large = build_model(xl_cfg)
        small_params = sum(parameter.numel() for parameter in small.parameters())
        xl_params = sum(parameter.numel() for parameter in large.parameters())
        ratio = xl_params / small_params
        if ratio < MIN_PARAMETER_RATIO:
            raise RuntimeError(
                f"XL capacity below the {MIN_PARAMETER_RATIO:g}x target: {ratio:.2f}x"
            )
        print(
            f"[xl-capacity] small {small_params:,} params; "
            f"XL {xl_params:,} params; {ratio:.2f}x; "
            f"FP32 weights about {4 * xl_params / (1024**2):.2f} MiB",
            flush=True,
        )
        base.RUN_DIR.mkdir(parents=True, exist_ok=True)
        (base.RUN_DIR / "model_capacity.json").write_text(
            json.dumps({
                "base_parameters": small_params,
                "xl_parameters": xl_params,
                "ratio": ratio,
                "xl_fp32_weight_mib": 4 * xl_params / (1024**2),
                "baseline_run_dir": str(cache_dir),
                "xl_run_dir": str(base.RUN_DIR),
            }, indent=2),
            encoding="utf-8",
        )

    base.install_dependencies = install_and_check_capacity

    # Recovery snapshots remain on Drive, but the browser ZIP excludes them
    # to avoid packaging gigabytes of redundant XL optimizer/EMA states.
    original_package = base.package_and_download

    def package_xl():
        original_package(download=False, include_recovery_snapshots=False)
        print(
            f"[xl] Results ZIP saved locally at {base.ZIP_PATH}; "
            f"Drive checkpoints/exports persist at {base.RUN_DIR}",
            flush=True,
        )

    base.package_and_download = package_xl


def main() -> None:
    script = Path(__file__).with_name("train_rres_mobile1d_colab.py")
    spec = importlib.util.spec_from_file_location("mobile1d_xl_training_base", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load shared Mobile1D Colab script: {script}")
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    configure_xl(base)
    print(
        "[mobile1d-xl] separate high-capacity run: 120 epochs, "
        "48 base channels, 640 feature channels, batch 8",
        flush=True,
    )
    base.main()


if __name__ == "__main__":
    main()
