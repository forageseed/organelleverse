"""Chloroplast bodies and stromule candidates: synthetic scene with known stromules."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage as ndi
from skimage.draw import ellipse, line_aa

from organelleverse.morphology.chloro_metrics import (
    chloroplast_bodies,
    detect_stromules,
    finalize_stromules,
    stromule_summary,
)
from organelleverse.morphology.image_job import classical_instances, run_image_morphology

CENTRES = [(30, 30), (30, 100), (30, 170), (90, 30), (90, 100), (90, 170)]
DRAWN = 15  # stromule length beyond the body edge, px


def chloro_scene(seed: int = 3, noise: float = 25.0) -> np.ndarray:
    """Six ellipses (20×14 px, 1000) on background 100; stromules on bodies 0, 1 (bright) and 5 (dim, 350).

    Also a detached line far from any body and a 3-px bump on body 3; neither may count.
    """
    rng = np.random.default_rng(seed)
    image = np.full((140, 200), 100.0)
    for cy, cx in CENTRES:
        rr, cc = ellipse(cy, cx, 7, 10, shape=image.shape)
        image[rr, cc] = 1000
    image[29:31, 38:56] = 1000    # body 0, bright, starts inside the body
    image[29:31, 108:126] = 1000  # body 1, bright
    image[98:113, 169:171] = 350  # body 5, dim: well below the Otsu threshold, above background
    image[125:127, 5:20] = 1000   # detached line
    image[82:85, 30:33] = 1000    # bump
    image = ndi.gaussian_filter(image, 0.7)
    return image + rng.normal(0, noise, image.shape)


def _ellipse_area() -> int:
    rr, _ = ellipse(30, 30, 7, 10, shape=(140, 200))
    return int(rr.size)


def _segment(image: np.ndarray) -> np.ndarray:
    labels, _ = classical_instances(image, min_area=50)
    assert int(labels.max()) == 6
    return labels


def test_bodies_remove_thin_protrusions_and_keep_tiny_objects() -> None:
    labels = np.zeros((40, 60), dtype=np.uint16)
    labels[10:25, 10:25] = 1
    labels[17:19, 25:40] = 1          # 2-px stromule
    labels[30:32, 5:7] = 2            # smaller than the opening: keeps its pixels
    labels[10:25, 45:55] = 3          # plain rectangle: only its corners are rounded
    bodies = chloroplast_bodies(labels, radius_px=2)
    assert 210 <= int((bodies == 1).sum()) <= 225      # square with rounded corners, stromule gone
    assert not (bodies[17:19, 26:40] > 0).any()
    assert int((bodies == 2).sum()) == 4
    assert 130 <= int((bodies == 3).sum()) <= 150
    assert np.array_equal(bodies, chloroplast_bodies(labels, radius_px=2))


def test_bodies_keep_largest_piece_and_do_not_erode_at_image_border() -> None:
    labels = np.zeros((30, 50), dtype=np.uint16)
    labels[5:20, 0:15] = 1             # touches the left border
    labels[11:14, 15:22] = 1           # thin neck
    labels[8:17, 22:27] = 1            # small second blob (45 px)
    bodies = chloroplast_bodies(labels, radius_px=2)
    assert 215 <= int((bodies == 1).sum()) <= 225
    assert not (bodies[:, 18:] > 0).any()              # neck and the small second blob are gone
    assert bodies[5, 0] == 1 and bodies[19, 0] == 1    # the border row is kept


def test_stromules_found_with_known_length_and_bodies_keep_their_area() -> None:
    image = chloro_scene()
    labels = _segment(image)
    bodies, result = detect_stromules(image, labels, pixel_size_um=None)

    assert result["summary"]["chloroplasts"] == 6
    assert result["summary"]["with_stromules"] == 3
    assert result["summary"]["frequency"] == pytest.approx(3 / 6)
    assert result["summary"]["stromules"] == 3
    assert [c["id"] for c in result["candidates"]] == [1, 2, 3]
    expected = {int(bodies[30, 30]), int(bodies[30, 100]), int(bodies[90, 170])}
    assert {c["label"] for c in result["candidates"]} == expected
    for candidate in result["candidates"]:
        assert abs(candidate["length"] - DRAWN) <= 3
        assert candidate["mean_width"] <= 4
        assert len(candidate["skeleton"]) >= 5
        assert candidate["bbox"][0] <= candidate["skeleton"][0][0] <= candidate["bbox"][2]
    # the dim stromule is invisible to the Otsu segmentation: it comes from the ridge filter
    dim = next(c for c in result["candidates"] if c["label"] == int(bodies[90, 170]))
    assert (labels[100:112, 168:172] == 0).all()
    assert dim["bbox"][0] >= 98 and dim["bbox"][2] <= 113

    ellipse_area = _ellipse_area()
    for number in range(1, 7):
        assert abs(int((bodies == number).sum()) - ellipse_area) <= 0.1 * ellipse_area
    # stromule pixels are no longer part of the mask
    assert int((labels > 0).sum()) > int((bodies > 0).sum())
    assert bodies[29:31, 45:55].sum() == 0
    # no calibration, no µm keys
    assert all("length_um" not in c and "mean_width_um" not in c for c in result["candidates"])
    assert "total_length_um" not in result["summary"]
    assert result["parameters"]["body_radius_px"] == 2 and result["parameters"]["ridge_threshold"] > 0
    assert "Sato" in result["definition"]


def test_calibrated_result_has_um_fields_and_scaled_defaults() -> None:
    image = chloro_scene()
    labels = _segment(image)
    _, result = detect_stromules(image, labels, pixel_size_um=0.25)
    assert result["parameters"]["body_radius_px"] == 2          # round(0.5 µm / 0.25)
    assert result["parameters"]["min_length_px"] == 4.0         # 1 µm
    assert result["parameters"]["max_width_px"] == 4.0
    for c in result["candidates"]:
        assert c["length_um"] == pytest.approx(c["length"] * 0.25, abs=1e-3)
        assert c["mean_width_um"] == pytest.approx(c["mean_width"] * 0.25, abs=1e-3)
    assert result["summary"]["total_length_um"] == pytest.approx(result["summary"]["total_length"] * 0.25, abs=1e-2)
    assert result["summary"]["with_stromules"] == 3


def test_detection_is_deterministic_and_noise_alone_gives_no_stromules() -> None:
    image = chloro_scene()
    labels = _segment(image)
    first = detect_stromules(image, labels, pixel_size_um=0.5)
    second = detect_stromules(image.copy(), labels.copy(), pixel_size_um=0.5)
    assert np.array_equal(first[0], second[0])
    assert json.dumps(first[1], sort_keys=True) == json.dumps(second[1], sort_keys=True)

    plain = np.full((140, 200), 100.0)
    for cy, cx in CENTRES:
        rr, cc = ellipse(cy, cx, 7, 10, shape=plain.shape)
        plain[rr, cc] = 1000
    plain = ndi.gaussian_filter(plain, 0.7) + np.random.default_rng(11).normal(0, 40, plain.shape)
    _, result = detect_stromules(plain, _segment(plain), pixel_size_um=None)
    assert result["summary"]["stromules"] == 0 and result["candidates"] == []


def test_degenerate_inputs() -> None:
    image = np.full((20, 20), 5.0)
    bodies, result = detect_stromules(image, np.zeros((20, 20), dtype=np.uint16), pixel_size_um=None)
    assert int(bodies.max()) == 0 and result["summary"]["frequency"] is None
    square = np.zeros((20, 20), dtype=np.uint16)
    square[5:15, 5:15] = 1
    _, flat = detect_stromules(image, square, pixel_size_um=None)  # no contrast: nothing to find
    assert flat["candidates"] == [] and flat["summary"]["chloroplasts"] == 1


def test_finalize_drops_candidates_of_unmeasured_bodies_and_renumbers() -> None:
    candidates = [{"id": i, "label": label, "length": 10.0} for i, label in enumerate((1, 2, 2, 3), start=1)]
    result = {"definition": "d", "parameters": {}, "candidates": candidates,
              "summary": stromule_summary(candidates, chloroplasts=3, pixel_size_um=None)}
    final = finalize_stromules(result, {2, 3}, pixel_size_um=None)
    assert [(c["id"], c["label"]) for c in final["candidates"]] == [(1, 2), (2, 2), (3, 3)]
    assert final["summary"] == {"chloroplasts": 2, "with_stromules": 2, "frequency": 1.0, "stromules": 3,
                                "total_length": 30.0}


def _write_tiff(path: Path, image: np.ndarray) -> Path:
    import tifffile

    tifffile.imwrite(path, np.clip(image, 0, 65535).round().astype(np.uint16))
    return path


def test_image_job_with_stromules_writes_body_mask_rows_and_top_level_block(tmp_path: Path) -> None:
    import tifffile

    source = _write_tiff(tmp_path / "chloro.tif", chloro_scene() * 40)
    out = tmp_path / "on"
    result = run_image_morphology({"image": str(source), "engine": "classical", "min_area": 50,
                                   "pixel_size_um": 0.25, "stromules": True}, output_dir=out, log=lambda _: None)
    objects = json.loads(Path(result["output_files"]["objects"]).read_text(encoding="utf-8"))
    assert len(objects["objects"]) == 6
    for row in objects["objects"]:
        assert {"stromules", "stromule_length", "stromule_length_um"} <= set(row)
    assert sum(row["stromules"] for row in objects["objects"]) == 3
    block = objects["stromules"]
    assert block["summary"]["with_stromules"] == 3 and block["summary"]["frequency"] == pytest.approx(0.5)
    assert [c["id"] for c in block["candidates"]] == [1, 2, 3]
    by_label = {row["label"]: row for row in objects["objects"]}
    for c in block["candidates"]:
        assert by_label[c["label"]]["stromule_length"] == pytest.approx(c["length"], abs=0.01)

    labels = tifffile.imread(result["output_files"]["labels"])
    assert int((labels > 0).sum()) == sum(row["area_px"] for row in objects["objects"])
    assert labels[30, 45] == 0                      # stromule pixels are out of the body mask
    evidence = json.loads(Path(result["output_files"]["evidence"]).read_text(encoding="utf-8"))
    assert evidence["parameters"]["stromules"] is True
    assert "3 stromules on 3 of 6" in result["summary_text"]
    with open(result["output_files"]["objects_csv"], newline="") as handle:
        header = next(csv.reader(handle))
    assert {"stromules", "stromule_length", "stromule_length_um"} <= set(header)


def test_image_job_without_flag_is_unchanged(tmp_path: Path) -> None:
    source = _write_tiff(tmp_path / "chloro.tif", chloro_scene() * 40)
    result = run_image_morphology({"image": str(source), "engine": "classical", "min_area": 50},
                                  output_dir=tmp_path / "off", log=lambda _: None)
    objects = json.loads(Path(result["output_files"]["objects"]).read_text(encoding="utf-8"))
    assert "stromules" not in objects
    assert all(not any(key.startswith("stromule") for key in row) for row in objects["objects"])
    evidence = json.loads(Path(result["output_files"]["evidence"]).read_text(encoding="utf-8"))
    assert "stromules" not in evidence["parameters"]
    assert "stromule" not in result["summary_text"]
    with open(result["output_files"]["objects_csv"], newline="") as handle:
        assert not any(name.startswith("stromule") for name in next(csv.reader(handle)))


# --- 512 x 512 acceptance scene: 12 rotated lens-shaped chloroplasts, 5 stromules, 1 free filament, 2 bumps ---
LENS_CENTRES = [(70, 80), (70, 200), (70, 330), (70, 440), (190, 120), (190, 380),
                (330, 80), (330, 200), (330, 330), (330, 440), (450, 140), (450, 380)]
# (chloroplast, polyline (row, col), peak). Polylines start in the body centre so each line is connected;
# the same lines drawn from a point outside the body (anchored=False) leave gaps of 5 and 9 px.
LENS_STROMULES = [(0, [(70, 94), (70, 120)], 900), (2, [(84, 330), (110, 330)], 380),
                  (4, [(205, 120), (215, 130), (220, 145), (222, 160)], 900),
                  (8, [(320, 345), (300, 365), (290, 380)], 400), (10, [(450, 155), (450, 200)], 1000)]


def lens_scene(*, anchored: bool = True) -> tuple[np.ndarray, list[np.ndarray], list[tuple[int, list[tuple[int, int]]]]]:
    rng = np.random.default_rng(20261009)
    image = np.full((512, 512), 120.0)
    masks = []
    for number, (y, x) in enumerate(LENS_CENTRES):
        rr, cc = ellipse(y, x, 9, 14, shape=image.shape, rotation=math.radians((number * 37) % 180))
        image[rr, cc] = 1500
        mask = np.zeros(image.shape, dtype=bool)
        mask[rr, cc] = True
        masks.append(mask)

    def tube(points: list[tuple[int, int]], value: float) -> None:
        for (y0, x0), (y1, x1) in zip(points, points[1:]):
            for dy in range(2):
                rr, cc, weight = line_aa(y0 + dy, x0, y1 + dy, x1)
                image[rr, cc] = np.maximum(image[rr, cc], 120 + (value - 120) * weight)

    lines = []
    for number, points, value in LENS_STROMULES:
        points = ([LENS_CENTRES[number]] if anchored else []) + points
        tube(points, value)
        lines.append((number, points))
    tube([(400, 250), (400, 290)], 900)             # free filament
    for y, x in [(330, 214), (190, 394)]:           # 3-px bumps on bodies 7 and 5
        image[y - 1:y + 2, x:x + 3] = 1500
    image += rng.normal(0, 30, image.shape)
    return np.clip(image, 0, 65535).astype(np.uint16), masks, lines


def _outside_length(points: list[tuple[int, int]], mask: np.ndarray) -> float:
    """Length of the drawn polyline that lies outside the true ellipse (sampled every 0.05 px)."""
    total = 0.0
    for (y0, x0), (y1, x1) in zip(points, points[1:]):
        steps = max(int(math.dist((y0, x0), (y1, x1)) / 0.05), 1)
        t = np.linspace(0, 1, steps + 1)
        inside = mask[np.round(y0 + (y1 - y0) * t).astype(int), np.round(x0 + (x1 - x0) * t).astype(int)]
        total += float((~inside[1:]).sum()) * math.dist((y0, x0), (y1, x1)) / steps
    return total


@pytest.mark.parametrize("pixel_size_um", [None, 0.1])
def test_lens_scene_counts_chloroplasts_stromules_and_free_filaments(pixel_size_um: float | None) -> None:
    image, masks, lines = lens_scene()
    labels, _ = classical_instances(image, min_area=30)
    assert int(labels.max()) == 13                  # 12 bodies + the free filament
    bodies, result = detect_stromules(image, labels, pixel_size_um=pixel_size_um)

    summary = result["summary"]
    assert summary["chloroplasts"] == 12 and summary["stromules"] == 5 and summary["with_stromules"] == 5
    assert summary["free_filaments"] == 1
    assert len(np.unique(bodies[bodies > 0])) == 12            # filaments are not in the body map
    owners = {number: int(bodies[LENS_CENTRES[number]]) for number in range(12)}
    assert all(label > 0 for label in owners.values())
    assert sorted(c["label"] for c in result["candidates"]) == sorted(owners[n] for n in (0, 2, 4, 8, 10))
    for number, points in lines:
        candidate = next(c for c in result["candidates"] if c["label"] == owners[number])
        assert abs(candidate["length"] - _outside_length(points, masks[number])) <= 3, (number, candidate["length"])
    # the free filament (row 400) and both bumps are not counted
    assert all(not (390 <= c["bbox"][0] <= 410) for c in result["candidates"])
    assert not {owners[5], owners[7]} & {c["label"] for c in result["candidates"]}

    again = detect_stromules(image.copy(), labels.copy(), pixel_size_um=pixel_size_um)
    assert np.array_equal(bodies, again[0])
    assert json.dumps(result, sort_keys=True) == json.dumps(again[1], sort_keys=True)


def test_stromules_that_do_not_touch_the_body_are_not_attached() -> None:
    """Lines drawn from a point outside the body leave 5 and 9 px gaps: not stromules by the attach rule.

    The long line has a 2 px gap and was segmented as a separate thin object: it still attaches.
    """
    image, _, _ = lens_scene(anchored=False)
    labels, _ = classical_instances(image, min_area=30)
    assert int(labels.max()) == 14                  # the long line, 2 px from its body, is its own object
    bodies, result = detect_stromules(image, labels, pixel_size_um=0.1)
    assert result["summary"]["chloroplasts"] == 12 and result["summary"]["free_filaments"] == 1
    assert sorted(c["label"] for c in result["candidates"]) == sorted(
        int(bodies[LENS_CENTRES[n]]) for n in (0, 2, 10))


def test_small_round_object_stays_a_chloroplast_but_thin_line_does_not() -> None:
    image = np.full((60, 80), 100.0)
    labels = np.zeros((60, 80), dtype=np.uint16)
    image[10:30, 10:30] = 1000
    labels[10:30, 10:30] = 1
    image[40:47, 10:17] = 1000                     # 7 x 7: erased by the opening, but not a filament
    labels[40:47, 10:17] = 2
    image[45:47, 30:70] = 1000                     # 2 x 40 line far from every body
    labels[45:47, 30:70] = 3
    bodies, result = detect_stromules(image, labels, pixel_size_um=0.1)
    assert int((bodies == 2).sum()) == 49 and int((bodies == 3).sum()) == 0
    assert result["summary"]["chloroplasts"] == 2 and result["summary"]["free_filaments"] == 1
    assert result["candidates"] == []


def test_image_job_excludes_filaments_from_mask_and_rows(tmp_path: Path) -> None:
    import tifffile

    image, _, _ = lens_scene()
    source = tmp_path / "lens.tif"
    tifffile.imwrite(source, image)
    result = run_image_morphology({"image": str(source), "engine": "classical", "min_area": 30,
                                   "pixel_size_um": 0.1, "stromules": True}, output_dir=tmp_path / "out",
                                  log=lambda _: None)
    objects = json.loads(Path(result["output_files"]["objects"]).read_text(encoding="utf-8"))
    assert len(objects["objects"]) == 12
    assert objects["stromules"]["summary"]["stromules"] == 5
    assert objects["stromules"]["summary"]["free_filaments"] == 1
    labels = tifffile.imread(result["output_files"]["labels"])
    assert len(np.unique(labels[labels > 0])) == 12
    assert labels[400, 270] == 0                    # free filament
    assert sum(row["stromules"] for row in objects["objects"]) == 5
