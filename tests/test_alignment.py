import math

from resistor_model.alignment import (
    angle_difference_180,
    consensus_axis_angle,
    normalize_axis_angle,
    rotation_to_vertical,
)


def test_normalize_axis_angle_treats_180_degrees_as_same_axis():
    assert normalize_axis_angle(100.0) == -80.0
    assert normalize_axis_angle(-100.0) == 80.0
    assert normalize_axis_angle(180.0) == 0.0


def test_angle_difference_uses_undirected_axis_distance():
    assert angle_difference_180(85.0, -85.0) == 10.0
    assert angle_difference_180(10.0, 190.0) == 0.0


def test_consensus_axis_angle_handles_wraparound_near_vertical():
    angle = consensus_axis_angle([88.0, -89.0, 87.0], [1.0, 1.0, 1.0])
    assert abs(abs(angle) - 88.666) < 0.5


def test_weighted_consensus_downweights_outlier():
    angle = consensus_axis_angle([20.0, 22.0, 19.0, 70.0], [3.5, 3.0, 2.0, 1.0])
    assert 18.0 <= angle <= 25.0


def test_rotation_to_vertical_matches_opencv_image_coordinates():
    assert rotation_to_vertical(-62.9) == 27.1
    assert rotation_to_vertical(0.0) == -90.0
    assert rotation_to_vertical(90.0) == 0.0


def test_rotation_to_vertical_makes_axis_vertical_modulo_180():
    for axis in (-89.0, -62.9, -20.0, 0.0, 33.0, 89.0):
        rotation = rotation_to_vertical(axis)
        # In OpenCV image coordinates, rotating by r changes the measured
        # line angle to axis-r. The result must be vertical modulo 180.
        result = normalize_axis_angle(axis - rotation)
        assert math.isclose(abs(result), 90.0, abs_tol=1e-9) or math.isclose(result, -90.0, abs_tol=1e-9)
