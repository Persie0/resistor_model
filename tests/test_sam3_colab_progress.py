import json
from pathlib import Path


NOTEBOOK = Path("colab/sam3_merge_3_resistor_datasets_colab.ipynb")
WORKFLOW = Path("colab/sam3_merge_3_resistor_datasets.py")
BOOTSTRAP = Path("colab/sam3_test_bootstrap.py")


def test_workflow_has_visible_progress():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "[download]" in text
    assert "[sam3]" in text
    assert "ETA" in text
    assert "[export" in text
    assert "flush=True" in text


def test_bootstrap_loads_sam3_without_running_dataset_workflow():
    text = BOOTSTRAP.read_text(encoding="utf-8")
    assert "build_sam3_image_model" in text
    assert "Sam3Processor" in text
    assert "SAM3_BOOTSTRAPPED = True" in text
    assert "DATASET_ASSETS" not in text


def test_single_image_test_comes_before_full_dataset_workflow():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = ["".join(cell.get("source", [])) for cell in notebook["cells"]]

    bootstrap_index = next(i for i, src in enumerate(code) if "sam3_test_bootstrap.py" in src)
    test_index = next(i for i, src in enumerate(code) if "uploaded_test = files.upload()" in src)
    workflow_index = next(
        i for i, src in enumerate(code)
        if "sam3_merge_3_resistor_datasets.py" in src and "sam3_test_bootstrap.py" not in src
    )

    assert bootstrap_index < test_index < workflow_index


def test_colab_setup_keeps_existing_numpy_abi():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    bootstrap = BOOTSTRAP.read_text(encoding="utf-8")

    for text in (workflow, bootstrap):
        assert "patch_sam3_numpy2_compat" in text
        assert '"--no-deps"' in text
        assert '"numpy"' not in text

    assert "import pandas" not in workflow
    assert "import cv2" not in workflow
    assert '"pandas"' not in workflow
    assert '"opencv-python-headless"' not in workflow


def test_bootstrap_uses_open_sam31_mirror_and_supports_local_checkpoint():
    text = BOOTSTRAP.read_text(encoding="utf-8")
    assert 'SAM3_REPO_ID = "AEmotionStudio/sam3.1"' in text
    assert 'SAM3_CHECKPOINT_FILE = "sam3.1_multiplex.pt"' in text
    assert "SAM3_CHECKPOINT_PATH" in text
    assert "resolve_sam3_checkpoint" in text
    assert "hf_hub_download" in text
    assert "GatedRepoError" not in text
    assert "notebook_login" not in text
    assert "load_from_HF=False" in text
    assert "checkpoint_path=sam3_checkpoint" in text


def test_notebook_reuses_open_sam31_checkpoint_for_full_workflow():
    notebook = NOTEBOOK.read_text(encoding="utf-8")
    assert "AEmotionStudio/sam3.1" in notebook
    assert "sam3.1_multiplex.pt" in notebook
    assert "SAM3_CHECKPOINT_PATH" in notebook
    assert "sam3_checkpoint_for_full" in notebook
    assert "download_ckpt_from_hf" in notebook
    assert "gated" not in notebook.lower()
