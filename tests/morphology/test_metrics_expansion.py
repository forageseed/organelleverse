"""T-C1: ImageJ-scale metric set with explicit definitions and unit discipline."""

from __future__ import annotations

import numpy as np

from organelleverse.morphology.measure import measure


def _rectangle() -> np.ndarray:
    lab = np.zeros((40, 60), dtype=np.int32)
    lab[5:15, 10:40] = 1  # 30 wide x 10 high rectangle, area 300
    return lab


def test_imagej_scale_metrics_present_with_expected_values() -> None:
    row = measure(_rectangle(), min_area=10)["per_object"][0]
    # scikit-image-direct geometric metrics
    assert row["area_px"] == 300.0
    assert row["solidity"] == 1.0  # rectangle == its convex hull
    assert row["extent"] == 1.0  # rectangle fills its bbox
    assert row["convex_area"] == 300.0
    assert row["feret_diameter_max"] == 31.32  # ~ diagonal sqrt(30^2+10^2)
    assert row["major_axis_length"] > row["minor_axis_length"]
    assert row["bbox_width"] == 30 and row["bbox_height"] == 10
    assert row["centroid_y"] == 9.5 and row["centroid_x"] == 24.5
    assert abs(row["orientation"]) > 1.0  # horizontal bar: near ±pi/2


def test_uncalibrated_physical_columns_are_empty_never_fake() -> None:
    row = measure(_rectangle(), min_area=10)["per_object"][0]
    for key in ("area_um2", "perimeter_um", "equivalent_diameter_um", "feret_max_um"):
        assert row[key] is None, f"{key} must be empty without calibration, not a fake 1.0"
    assert row["pixel_size_um"] is None


def test_calibrated_physical_columns_are_filled() -> None:
    row = measure(_rectangle(), min_area=10, pixel_size_um=0.1)["per_object"][0]
    assert row["area_um2"] == 3.0  # 300 px * 0.01 um^2
    assert row["perimeter_um"] == 7.6  # 76 px * 0.1
    assert row["pixel_size_um"] == 0.1


def test_intensity_metrics_with_source_image() -> None:
    lab = _rectangle()
    img = np.zeros((40, 60), dtype=np.float64)
    img[5:15, 10:40] = 100.0  # uniform intensity inside the object
    row = measure(lab, intensity_image=img, min_area=10)["per_object"][0]
    assert row["mean_intensity"] == 100.0
    assert row["intensity_min"] == 100.0 and row["intensity_max"] == 100.0
    assert row["intensity_std"] == 0.0
    assert row["intensity_median"] == 100.0
    assert row["intensity_skew"] is None  # undefined for constant intensity, never NaN
    assert row["integrated_density"] == 30000.0  # area x mean (ImageJ IntDen)
    assert row["centroid_weighted_y"] == 9.5


def test_intensity_columns_absent_without_intensity_image() -> None:
    row = measure(_rectangle(), min_area=10)["per_object"][0]
    assert "integrated_density" not in row
    assert "intensity_min" not in row


def test_perimeter_definition_matches_skimage_not_imagej() -> None:
    """Perimeter is skimage's pixel-boundary length, NOT ImageJ's — the
    metrics document records this divergence; the test pins the source."""
    row = measure(_rectangle(), min_area=10)["per_object"][0]
    # 30x10 rectangle: skimage perimeter (Crofton-style) = 76, not the
    # polygon perimeter 2*(30+10)=80
    assert row["perimeter"] == 76.0


def test_electron_density_ratio_against_background() -> None:
    """OrgSegNet-style mu_o/mu_b: object mean over BACKGROUND mean (the whole
    label map's background, not per-class leftovers)."""
    lab = _rectangle()
    img = np.full((40, 60), 200.0)
    img[5:15, 10:40] = 100.0
    row = measure(lab, intensity_image=img, min_area=10)["per_object"][0]
    assert row["electron_density_ratio"] == 0.5  # 100 / 200


def test_spatial_summary_area_fraction_and_density() -> None:
    lab = _rectangle()  # one object, area 300 of 2400 px
    result = measure(lab, min_area=10, pixel_size_um=0.1)
    assert result["image_area_px"] == 2400.0
    assert result["area_fraction_pct"]["Chloroplast"] == 12.5  # 300/2400
    # image area = 2400 * 0.01 um^2 = 24 um^2 -> 1/24 per um^2 -> 41666.7/mm^2
    assert result["number_density_per_mm2"]["Chloroplast"] == 41666.667
    assert result["background_mean_intensity"] is None  # no intensity image


def test_nearest_neighbor_distances() -> None:
    lab = np.zeros((40, 60), dtype=np.int32)
    lab[5:15, 5:15] = 1
    lab[5:15, 35:45] = 1  # same class, two squares, centroid distance 30
    result = measure(lab, min_area=10, pixel_size_um=0.5)
    rows = result["per_object"]
    assert len(rows) == 2
    assert all(row["nearest_neighbor_px"] == 30.0 for row in rows)
    assert all(row["nearest_neighbor_um"] == 15.0 for row in rows)
    assert result["nearest_neighbor_px"]["mean_px"] == 30.0


def test_single_object_has_no_nearest_neighbor() -> None:
    row = measure(_rectangle(), min_area=10)["per_object"][0]
    assert row["nearest_neighbor_px"] is None
    assert row["nearest_neighbor_um"] is None


def test_min_feret_and_imagej_aspect_and_roundness() -> None:
    row = measure(_rectangle(), min_area=10)["per_object"][0]
    assert row["feret_diameter_min"] == 10.0  # min caliper of a 30x10 rectangle
    # ImageJ AR (major/minor) is the reciprocal convention of ours
    assert abs(row["aspect_ratio_imagej"] - 1.0 / row["aspect_ratio"]) < 0.01
    # ImageJ Round = 4A / (pi * major^2)
    expected = 4.0 * 300.0 / (3.141592653589793 * row["major_axis_length"] ** 2)
    assert row["roundness"] == round(expected, 4)


def test_shape_complexity_convex_higher_than_concave() -> None:
    square = np.zeros((50, 50), dtype=np.int32)
    square[10:40, 10:40] = 1
    l_shape = np.zeros((50, 50), dtype=np.int32)
    l_shape[10:40, 10:20] = 1  # vertical bar
    l_shape[30:40, 10:40] = 1  # horizontal foot -> concave corner
    convex = measure(square, min_area=10)["per_object"][0]["shape_complexity"]
    concave = measure(l_shape, min_area=10)["per_object"][0]["shape_complexity"]
    # convex close to 1.0 — sub-1 because chords sampled on a rasterized
    # contour occasionally rint onto background pixels at the boundary
    assert convex is not None and convex > 0.85
    assert concave is not None and concave < convex


def test_imagej_raw_intden_and_mode() -> None:
    lab = _rectangle()  # 300 px
    img = np.full((40, 60), 50.0)
    img[5:15, 10:40] = 100.0
    row = measure(lab, intensity_image=img, min_area=10)["per_object"][0]
    assert row["raw_integrated_density"] == 30000.0  # 300 px x 100
    assert row["mode_intensity"] == 100.0
    # uniform object: Gaussian fit degenerate -> mean fallback keeps the ratio
    assert row["electron_density_ratio"] == 2.0  # 100 / 50
    assert row["electron_density_ratio_mean"] == 2.0


def test_electron_density_uses_gaussian_peaks_when_fittable() -> None:
    rng = np.random.default_rng(7)
    lab = _rectangle()
    img = np.full((40, 60), 0.0)
    img[5:15, 10:40] = rng.normal(120.0, 5.0, size=(10, 30))  # object ~120
    img[lab == 0] = rng.normal(200.0, 5.0, size=int((lab == 0).sum()))  # bg ~200
    row = measure(lab, intensity_image=img, min_area=10)["per_object"][0]
    # peak-based ratio should be near the mean-based one for Gaussian data
    assert abs(row["electron_density_ratio"] - 0.6) < 0.03
    assert abs(row["electron_density_ratio_mean"] - 0.6) < 0.03
