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
