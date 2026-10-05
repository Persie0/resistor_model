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

    assert "patch_sam3_numpy2_compat" in bootstrap
    assert '"--no-deps"' in bootstrap
    assert '"numpy"' not in bootstrap
    assert "ensure_sam3_ready" in workflow
    assert "sam3_test_bootstrap.py" in workflow
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
    assert "notebook_login = lambda" in notebook


def test_sam3_processor_methods_are_bfloat16_autocast_safe():
    bootstrap = BOOTSTRAP.read_text(encoding="utf-8")
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    notebook_code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )

    autocast = 'torch.autocast(device_type="cuda", dtype=torch.bfloat16)'
    assert "patch_sam3_processor_autocast" in bootstrap
    assert '"set_image"' in bootstrap
    assert '"set_text_prompt"' in bootstrap
    assert '"add_geometric_prompt"' in bootstrap
    assert autocast in bootstrap
    assert autocast in notebook_code


def test_prompt_metadata_is_cast_to_float32_before_numpy_use():
    bootstrap = BOOTSTRAP.read_text(encoding="utf-8")
    assert 'for key in ("scores", "boxes")' in bootstrap
    assert "value.dtype == torch.bfloat16" in bootstrap
    assert "result[key] = value.float()" in bootstrap


def test_processor_patch_upgrades_existing_colab_runtime():
    bootstrap = BOOTSTRAP.read_text(encoding="utf-8")
    assert 'SAM3_PROCESSOR_PATCH_VERSION = 2' in bootstrap
    assert '_resistor_model_bf16_autocast_patch_version' in bootstrap
    assert 'current_patch_version == SAM3_PROCESSOR_PATCH_VERSION' in bootstrap


def test_bulk_rebuild_ignores_old_annotations_and_accepts_top_sam_result_per_image():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert 'BODY_TEXT_PROMPT = "ceramic axial resistor body, not smd resistor"' in workflow
    assert "processor.set_text_prompt(" in workflow
    assert "prompt=BODY_TEXT_PROMPT" in workflow
    assert 'stage(3, 8, "Index source COCO images (annotations ignored)")' in workflow
    assert '"source_annotations_ignored": True' in workflow
    assert "best_candidate_index = int(np.argmax(scores))" in workflow
    assert '"reason": "sam_top_result"' in workflow
    assert "annotations_by_image" not in workflow
    assert "SOURCE_CATEGORY_NAMES" not in workflow
    assert "select_body_assignments" not in workflow
    assert "comparison_metrics" not in workflow
    assert "pair_is_plausible" not in workflow
    assert "processor.add_geometric_prompt(" not in workflow


def test_bulk_rebuild_requires_60_percent_confidence_for_acceptance():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "MIN_ACCEPT_SCORE = 0.60" in workflow
    assert "best_score = float(scores[best_candidate_index])" in workflow
    assert "best_score >= MIN_ACCEPT_SCORE" in workflow
    assert '"min_accept_score": MIN_ACCEPT_SCORE' in workflow


def test_bulk_rebuild_indexes_coco_images_without_requiring_annotations():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert 'for image_meta in coco["images"]:' in workflow
    assert '"image_path": image_path' in workflow
    assert '"file_name": image_meta["file_name"]' in workflow
    assert '"annotations": annotations' not in workflow
    assert "if not annotations" not in workflow


def test_bulk_rebuild_has_google_drive_resume_checkpoints_and_final_zip_copy():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "from google.colab import drive" in workflow
    assert 'DRIVE_CHECKPOINT_DIR = Path("/content/drive/MyDrive/resistor_sam3_merge_checkpoint")' in workflow
    assert "CHECKPOINT_EVERY_IMAGES = 25" in workflow
    assert "drive.mount(\"/content/drive\")" in workflow
    assert "load_checkpoint" in workflow
    assert "save_checkpoint" in workflow
    assert '"processed_keys"' in workflow
    assert '"accepted_records"' in workflow
    assert "checkpoint_masks" in workflow
    assert "if image_key in processed_keys:" in workflow
    assert "processed_this_run % CHECKPOINT_EVERY_IMAGES == 0" in workflow
    assert "shutil.copy2(zip_path, DRIVE_FINAL_ZIP)" in workflow


def test_single_image_preview_renders_only_selected_highest_score_body_mask():
    notebook = NOTEBOOK.read_text(encoding="utf-8")
    assert "test_best_index = int(np.argmax(test_scores))" in notebook
    assert "test_best_mask = test_masks[test_best_index]" in notebook
    assert "test_best_box = test_boxes[test_best_index]" in notebook
    assert "selected body candidate" in notebook
    assert "for i, (mask, box, score) in enumerate(zip(test_masks" not in notebook
