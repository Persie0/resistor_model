import numpy as np

from resistor_model.data.geom_aug import apply_flips, sample_rectify_jitter
from resistor_model.data.geometry import rectify_resistor


def test_rectify_jitter_can_be_disabled_and_stays_bounded():
    rng = np.random.default_rng(123)
    assert sample_rectify_jitter(rng, strength=0.0) is None
    jitter = sample_rectify_jitter(np.random.default_rng(5), strength=1.0, identity_prob=0.0)
    assert jitter is not None
    assert abs(jitter["angle"]) <= np.deg2rad(4.0) + 1e-9
    assert -0.03 <= jitter["shift_u"] <= 0.03
    assert -0.05 <= jitter["shift_v"] <= 0.05
    assert 0.94 <= jitter["scale_u"] <= 1.10
    assert 0.92 <= jitter["scale_v"] <= 1.20


def test_horizontal_flip_mirrors_band_boxes_and_image():
    image = np.arange(2 * 10 * 3, dtype=np.float32).reshape(2, 10, 3)
    bands = [{"color": "red", "bbox": [2.0, 0.0, 4.0, 2.0]}]
    flipped, out = apply_flips(image, bands, np.random.default_rng(1), hflip_prob=1.0, vflip_prob=0.0)
    assert np.array_equal(flipped, image[:, ::-1])
    assert out[0]["bbox"] == [6.0, 0.0, 8.0, 2.0]


def test_vertical_flip_mirrors_band_boxes():
    image = np.zeros((10, 20, 3), dtype=np.float32)
    bands = [{"color": "blue", "bbox": [2.0, 1.0, 4.0, 4.0]}]
    _, out = apply_flips(image, bands, np.random.default_rng(2), hflip_prob=0.0, vflip_prob=1.0)
    assert out[0]["bbox"] == [2.0, 6.0, 4.0, 9.0]


def test_rectification_accepts_training_jitter_and_preserves_shape():
    image = np.zeros((100, 220, 3), dtype=np.uint8)
    bands = [
        {"color": "brown", "bbox": [50, 30, 60, 70]},
        {"color": "black", "bbox": [85, 30, 95, 70]},
        {"color": "red", "bbox": [120, 30, 130, 70]},
        {"color": "gold", "bbox": [155, 30, 165, 70]},
    ]
    canonical, boxes0 = rectify_resistor(image, bands, resistor_bbox=[30, 20, 190, 80], output_size=(64, 384))
    jittered, boxes1 = rectify_resistor(
        image,
        bands,
        resistor_bbox=[30, 20, 190, 80],
        output_size=(64, 384),
        jitter={"angle": 0.02, "shift_u": 0.02, "shift_v": -0.01, "scale_u": 1.05, "scale_v": 1.10},
    )
    assert canonical.shape == jittered.shape == (64, 384, 3)
    assert boxes0 != boxes1
