from pathlib import Path


def test_colab_launcher_uses_progress_wrapper():
    notebook = Path("colab/sam3_merge_3_resistor_datasets_colab.ipynb").read_text(encoding="utf-8")
    assert "sam3_merge_3_resistor_datasets_progress.py" in notebook


def test_progress_wrapper_has_download_heartbeat_and_export_progress():
    text = Path("colab/sam3_merge_3_resistor_datasets_progress.py").read_text(encoding="utf-8")
    assert "[download]" in text
    assert "[heartbeat]" in text
    assert "[export]" in text
