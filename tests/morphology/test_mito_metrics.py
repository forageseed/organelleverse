"""Mitochondrial shape/network metrics against synthetic objects of known geometry and topology."""

from __future__ import annotations

import math

import numpy as np
import pytest
from skimage.draw import disk, line, polygon

from organelleverse.morphology.measure import measure
from organelleverse.morphology.mito_metrics import min_area_rect, network_summary, object_network, skeleton_raster


def _canvas(h: int = 80, w: int = 80) -> np.ndarray:
    return np.zeros((h, w), dtype=bool)


def _thin(points: list[tuple[int, int, int, int]], shape: tuple[int, int] = (80, 80)) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r0, c0, r1, c1 in points:
        rr, cc = line(r0, c0, r1, c1)
        mask[rr, cc] = True
    return mask


def test_rectangle_length_and_width_follow_the_minimum_bounding_rectangle() -> None:
    mask = _canvas()
    mask[10:15, 10:30] = True  # 20 x 5 px
    length, width, angle = min_area_rect(mask)
    assert (length, width) == (20.0, 5.0)
    assert abs(angle) < 1e-6


def test_rotated_rectangle_is_measured_along_its_own_axis() -> None:
    theta = math.radians(30)
    centre = np.array([40.0, 40.0])
    half = np.array([[20, 4], [-20, 4], [-20, -4], [20, -4]], dtype=float)  # 40 x 8
    rot = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    corners = half @ rot.T + centre  # (x, y)
    mask = _canvas()
    rr, cc = polygon(corners[:, 1], corners[:, 0], mask.shape)
    mask[rr, cc] = True
    length, width, angle = min_area_rect(mask)
    assert length == pytest.approx(40, abs=2)
    assert width == pytest.approx(8, abs=2)
    assert angle == pytest.approx(30, abs=3)


def test_straight_rod_has_two_ends_one_tubule_and_its_pixel_length() -> None:
    net = object_network(np.pad(_thin([(20, 10, 20, 40)]), 1))
    assert (net["ends"], net["junctions"], net["tubules"], net["morphology_class"]) == (2, 0, 1, "rod")
    assert net["skeleton_length"] == pytest.approx(30)


def test_diagonal_steps_count_root_two_without_double_counting() -> None:
    net = object_network(np.pad(_thin([(10, 10, 30, 30)]), 1))
    assert net["skeleton_length"] == pytest.approx(20 * math.sqrt(2))
    assert net["ends"] == 2


def test_branched_objects_report_junctions_ends_and_tubules() -> None:
    y_shape = _thin([(40, 40, 10, 40), (40, 40, 60, 20), (40, 40, 60, 60)])
    net = object_network(np.pad(y_shape, 1))
    assert (net["junctions"], net["ends"], net["tubules"], net["morphology_class"]) == (1, 3, 3, "network")
    plus = _thin([(40, 10, 40, 70), (10, 40, 70, 40)])
    net = object_network(np.pad(plus, 1))
    assert (net["junctions"], net["ends"], net["tubules"]) == (1, 4, 4)
    assert net["skeleton_length"] == pytest.approx(120, abs=2)


def test_punctate_and_ring_edge_cases() -> None:
    dot = _canvas(); dot[5, 5] = True
    assert object_network(np.pad(dot, 1))["morphology_class"] == "punctate"
    assert (object_network(np.pad(dot, 1))["ends"], object_network(np.pad(dot, 1))["tubules"]) == (2, 1)
    ring = _canvas()
    rr, cc = disk((40, 40), 15); ring[rr, cc] = True
    rr, cc = disk((40, 40), 12); ring[rr, cc] = False
    net = object_network(np.pad(ring, 1))
    assert (net["ends"], net["junctions"], net["tubules"]) == (0, 0, 1)


def test_measure_adds_metrics_and_ouellet_connectivity_summary() -> None:
    labels = np.zeros((80, 80), dtype=np.int32)
    labels[_thin([(5, 5, 5, 35)])] = 1  # rod: 2 ends, 1 tubule
    y_shape = _thin([(40, 40, 10, 40), (40, 40, 60, 20), (40, 40, 60, 60)])
    labels[y_shape] = 2  # network: 3 ends, 3 tubules, 1 junction
    result = measure(labels, pixel_size_um=0.5, min_area=1, instance_map=True,
                     class_names=("background", "object"))
    rows = {row["label"]: row for row in result["per_object"]}
    assert rows[1]["morphology_class"] == "rod" and rows[2]["morphology_class"] == "network"
    assert rows[1]["skeleton_length_um"] == pytest.approx(15)
    assert rows[1]["length_rect_um"] == pytest.approx(15.5)  # 31 px including pixel extents
    assert "form_factor" in rows[2]
    net = result["network"]
    assert (net["ends"], net["tubules"], net["junctions"], net["clusters"]) == (5, 4, 1, 2)
    assert net["junctions_per_end"] == pytest.approx(0.2)
    assert net["elements_per_cluster"] == pytest.approx(5.0)  # (5 + 4 + 1) / 2


def test_isolated_mitochondria_have_three_elements_per_cluster() -> None:
    rows = [{"skeleton_length": 10.0, "ends": 2, "junctions": 0, "tubules": 1, "morphology_class": "rod"}] * 4
    assert network_summary(rows, pixel_size_um=None)["elements_per_cluster"] == 3.0


def test_skeleton_overlay_marks_ends_and_junctions_per_object() -> None:
    labels = np.zeros((80, 80), dtype=np.int32)
    labels[_thin([(40, 40, 10, 40), (40, 40, 60, 20), (40, 40, 60, 60)])] = 1
    overlay = skeleton_raster(labels)
    assert int((overlay == 2).sum()) == 3
    assert int((overlay == 3).sum()) >= 1
    assert set(np.unique(overlay)) <= {0, 1, 2, 3}
