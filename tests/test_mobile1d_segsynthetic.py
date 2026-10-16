"""CPU-only tests for segmenter-in-the-loop color-band dataset preparation."""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from resistor_model.data.dataset import ResistorBandDataset
from resistor_model.decoder import _decode_one
from resistor_model.tools import mobile1d_segsynthetic as syn


@pytest.fixture
def fake_photo():
    image = np.full((128, 370, 3), 25, np.uint8)
    cv2.rectangle(image, (40, 35), (335, 92), (130, 185, 205), -1)
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[35:93, 40:336] = 1
    boxes = []
    for x, color in zip((96, 157, 220, 280), ("brown", "black", "red", "gold")):
        cv2.rectangle(image, (x, 35), (x + 9, 92), syn.COLORS[color], -1)
        boxes.append([float(x), 35., float(x+9), 92.])
    return image,mask,boxes


@pytest.mark.parametrize("count", [3, 4, 5, 6])
def test_synthesized_band_sequences_are_electrically_valid(count):
    rng=np.random.default_rng(42+count)
    for _ in range(200):
        colors=syn.new_colors(rng,count)
        assert len(colors)==count
        assert _decode_one(colors).valid


def test_six_band_repainting_creates_six_annotated_stripes(fake_photo):
    image,mask,boxes=fake_photo
    rng=np.random.default_rng(5)
    colors=["brown","black","black","red","gold","brown"]
    out,bands=syn.repaint_bands(image,mask,boxes,colors,rng)
    assert out.shape == image.shape
    assert len(bands)==6
    assert [b["color"] for b in bands]==colors
    assert sum(np.abs(out.astype(int)-image.astype(int)).ravel())>0
    centers=[(b["bbox"][0]+b["bbox"][2])/2 for b in bands]
    assert centers==sorted(centers)
    assert all(b["bbox"][2]>b["bbox"][0] for b in bands)


def test_geometry_warp_preserves_labels(fake_photo):
    image,mask,boxes=fake_photo
    warped,predicted,modified=syn._warp(image,mask,boxes,np.random.default_rng(17),True)
    assert warped.shape[:2]==predicted.shape
    assert len(modified)==len(boxes)
    assert predicted.sum()>0
    assert all(b[2]>b[0] and b[3]>b[1] for b in modified)


def test_horizontal_segmentation_crop_preserves_band_order(fake_photo):
    image,mask,boxes=fake_photo
    crop,result=syn._horizontal_crop(image,mask,boxes)
    assert crop.shape[:2]==syn.TARGET_HW
    assert len(result)==4
    centers=[(b[0]+b[2])/2 for b in result]
    assert centers==sorted(centers)


def test_pre_rectified_dataset_does_not_rectify_a_second_time(tmp_path:Path):
    width,height=512,176
    image=np.zeros((height,width,3),np.uint8)
    image[:]=(30,100,230)
    cv2.imwrite(str(tmp_path/"sample.jpg"),image)
    row={"image":"sample.jpg","session_id":"real-0","split":"train",
         "resistors":[{"id":"real-0","bbox":[0,0,width,height],
         "bands":[{"color":"brown","bbox":[100,8,120,166]},
                  {"color":"black","bbox":[200,8,220,166]},
                  {"color":"red","bbox":[300,8,320,166]}]}]}
    manifest=tmp_path/"manifest.jsonl"
    manifest.write_text(json.dumps(row)+"\n")
    dataset=ResistorBandDataset(
        manifest,tmp_path,output_size=(height,width),sequence_bins=256,
        pre_rectified=True,augment=False,geometric_augment=False,
    )
    sample=dataset[0]
    assert sample["image"].shape==(3,height,width)
    assert sample["count"].item()==3
    positions=sample["slot_centers"].numpy()[:3]
    assert positions==pytest.approx(np.asarray([110,210,310])/width,abs=.003)


def test_shards_verified_and_no_synthetic_data_in_heldout(monkeypatch,tmp_path:Path):
    remote=tmp_path/"remote"
    remote.mkdir()
    monkeypatch.setattr(syn,"DRIVE_ROOT",remote)
    recipe={"version":"test-only","segmenter_sha256":"mock-segmenter"}
    counts={"observed":{"train":1,"val":1,"test":1},"crops":4,"failed":0}
    stage=tmp_path/"build"
    (stage/"images").mkdir(parents=True)
    rows=[]
    for split,tag in (("train","real"),("train","synthetic"),
                      ("val","real"),("test","real")):
        ident=f"{split}-{tag}"
        (stage/"images"/f"{ident}.jpg").write_bytes(b"placeholder")
        rows.append({"image":f"images/{ident}.jpg","split":split,
                     "session_id":split,
                     "source_resistor_id":split,"synthetic":tag=="synthetic",
                     "segmentation_iou":0.8,
                     "resistors":[{"id":ident,"bands":[
                         {"color":"brown","bbox":[20,20,30,150]}]}]})
    (stage/"rows.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    (stage/"failures.jsonl").write_text("")
    (stage/"counts.json").write_text(json.dumps(counts))
    archive=remote/"shards"/"shard_0000.zip"
    syn.archive_shard(stage,archive,recipe)
    assert syn.valid_shard(archive,recipe)
    (remote/"recipe.json").write_text(json.dumps({"recipe":recipe,"n_shards":1}))
    summary=syn.restore(tmp_path/"output")
    assert summary["split_counts"]=={"train":2,"val":1,"test":1}
    assert summary["real_source_coverage"]=={"train":1.0,"val":1.0,"test":1.0}
    assert not any(r["synthetic"] and r["split"]!="train" for r in rows)
    archive.with_suffix(".json").write_text("{}")
    assert not syn.valid_shard(archive,recipe)


def test_segsynthetic_xl_training_config_is_independent():
    import importlib.util
    path=Path(__file__).resolve().parents[1]/"colab"
    def load(fname,name):
        spec=importlib.util.spec_from_file_location(name,path/fname)
        obj=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(obj)
        return obj
    base=load("train_rres_mobile1d_colab.py","segsynthetic_test_base")
    xl=load("train_rres_mobile1d_xl_colab.py","segsynthetic_test_xl")
    train=load("train_rres_mobile1d_xl_segsynthetic_colab.py","segsynthetic_test_train")
    train.configure_segmented_training(base,xl)
    config=base.build_config(base.RUN_DIR,None,None)
    assert "segsynthetic" in base.RUN_DIR.name
    assert config["data"]["pre_rectified"] is True
    assert config["model"]["d_model"]==568
    assert config["train"]["epochs"]==120
    assert config["train"]["resume"] is None


def test_six_band_variant_survives_segmentation_and_alignment(fake_photo):
    """Exercise generation -> geometric warp -> mask inference -> crop -> labels."""
    image, mask, boxes = fake_photo

    class SimulatedSegmenter:
        def mask(self, bgr):
            luminance = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            binary = (luminance > 42).astype(np.uint8)
            binary = cv2.morphologyEx(
                binary, cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_RECT, (19, 19)),
            )
            count, labels, stats, _ = cv2.connectedComponentsWithStats(
                binary, connectivity=8,
            )
            if count < 2:
                raise ValueError("simulated segmenter failed")
            biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            return (labels == biggest).astype(np.uint8)

    crop, labels, quality = syn.process_variant(
        SimulatedSegmenter(), image, boxes,
        ["brown", "black", "red", "gold"],
        np.random.default_rng(42), synthetic=True, count=6,
    )
    assert crop.shape[:2] == syn.TARGET_HW
    assert len(labels) == 6
    assert 0.42 <= quality <= 1.0
    assert _decode_one([b["color"] for b in labels]).valid
    assert all(
        0 <= b["bbox"][0] < b["bbox"][2] <= syn.TARGET_HW[1] and
        0 <= b["bbox"][1] < b["bbox"][3] <= syn.TARGET_HW[0]
        for b in labels
    )
