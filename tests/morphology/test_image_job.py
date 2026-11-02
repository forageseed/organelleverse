"""Image-workbench job runners: deterministic, bit-depth preserving, auditable."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from organelleverse.core.errors import OrganelleInputError
from organelleverse.morphology.image_job import (
    classical_instances,
    load_analysis_image,
    run_image_engine_probe,
    run_image_morphology,
)


def _scene(dtype: type = np.uint16) -> np.ndarray:
    """Two bright rectangles (100 px each) and one 4-px speck on a dark field."""
    image = np.full((60, 80), 1000, dtype=dtype)
    image[10:20, 10:20] = 40000
    image[30:35, 40:60] = 40000
    image[50:52, 70:72] = 40000
    return image


def _write_tiff(path: Path, array: np.ndarray) -> Path:
    import tifffile

    tifffile.imwrite(path, array)
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_classical_instances_are_sequential_and_drop_small_objects() -> None:
    labels, threshold = classical_instances(_scene(), min_area=10)
    assert labels.dtype == np.uint16
    assert 1000 < threshold < 40000
    assert sorted(np.unique(labels).tolist()) == [0, 1, 2]
    assert sorted(int((labels == i).sum()) for i in (1, 2)) == [100, 100]


def test_constant_image_has_no_objects() -> None:
    labels, _ = classical_instances(np.full((8, 8), 7, dtype=np.uint16), min_area=1)
    assert int(labels.max()) == 0


def test_morphology_job_reads_16_bit_original_and_writes_auditable_outputs(tmp_path: Path) -> None:
    source = _write_tiff(tmp_path / "scene.tif", _scene())
    out = tmp_path / "out"
    result = run_image_morphology({"image": str(source), "engine": "classical", "min_area": 10,
                                   "pixel_size_um": 0.5}, output_dir=out, log=lambda _: None)

    evidence = json.loads(Path(result["output_files"]["evidence"]).read_text(encoding="utf-8"))
    assert evidence["input"] == {"name": "scene.tif", "sha256": _sha(source), "width": 80, "height": 60,
                                 "stored_dtype": "uint16", "stored_shape": [60, 80]}
    assert evidence["parameters"] == {"engine": "classical", "min_area": 10, "pixel_size_um": 0.5}
    assert evidence["outputs"]["labels_sha256"] == _sha(out / "labels.tif")
    assert evidence["outputs"]["objects_sha256"] == _sha(out / "objects.json")
    assert evidence["engine"]["method"] == "otsu_threshold+connected_components"

    labels = load_analysis_image(out / "labels.tif")
    assert labels.shape == (60, 80) and labels.dtype == np.uint16
    objects = json.loads((out / "objects.json").read_text(encoding="utf-8"))
    assert (objects["image_width"], objects["image_height"]) == (80, 60)
    assert sorted(o["area_px"] for o in objects["objects"]) == [100, 100]
    # 100 px at 0.5 µm/px = 25 µm²: area scales with the square of the pixel size.
    assert sorted(o["area_um2"] for o in objects["objects"]) == [25.0, 25.0]
    assert (out / "objects.csv").is_file()


def test_morphology_job_is_byte_identical_when_repeated(tmp_path: Path) -> None:
    source = _write_tiff(tmp_path / "scene.tif", _scene())
    digests = []
    for name in ("a", "b"):
        run_image_morphology({"image": str(source), "engine": "classical", "min_area": 10},
                             output_dir=tmp_path / name, log=lambda _: None)
        digests.append((_sha(tmp_path / name / "labels.tif"), _sha(tmp_path / name / "objects.json")))
    assert digests[0] == digests[1]


def test_rgb_input_is_reduced_to_luminance_and_recorded(tmp_path: Path) -> None:
    from PIL import Image

    rgb = np.stack([(_scene() // 160).astype(np.uint8)] * 3, axis=-1)
    source = tmp_path / "scene.png"
    Image.fromarray(rgb).save(source)
    out = tmp_path / "out"
    run_image_morphology({"image": str(source), "min_area": 10}, output_dir=out, log=lambda _: None)
    evidence = json.loads((out / "evidence.json").read_text(encoding="utf-8"))
    assert evidence["input"]["rgb_to_luminance"] == "BT.601"
    assert len(json.loads((out / "objects.json").read_text(encoding="utf-8"))["objects"]) == 2


def test_stacks_and_bad_parameters_are_rejected(tmp_path: Path) -> None:
    stack = _write_tiff(tmp_path / "stack.tif", np.stack([_scene(), _scene()]))
    with pytest.raises(OrganelleInputError) as error:
        run_image_morphology({"image": str(stack)}, output_dir=tmp_path / "s", log=lambda _: None)
    assert error.value.code == "image_job.not_2d"
    source = _write_tiff(tmp_path / "scene.tif", _scene())
    with pytest.raises(OrganelleInputError) as error:
        run_image_morphology({"image": str(source), "engine": "magic"}, output_dir=tmp_path / "e",
                             log=lambda _: None)
    assert error.value.code == "image_job.engine_unknown"
    with pytest.raises(OrganelleInputError) as error:
        run_image_morphology({"image": str(source), "pixel_size_um": -1}, output_dir=tmp_path / "p",
                             log=lambda _: None)
    assert error.value.code == "image_job.pixel_size_invalid"


def test_engine_probe_reports_without_installing(tmp_path: Path) -> None:
    result = run_image_engine_probe({}, output_dir=tmp_path, log=lambda _: None)
    report = json.loads(Path(result["output_files"]["engines"]).read_text(encoding="utf-8"))
    assert report["schema"] == "organelleverse.image-engines.v1"
    assert report["engines"]["classical"]["available"] is True
    assert set(report["engines"]["micro_sam"]) >= {"available", "reason", "gpu"}
    assert report["host"]["python"]
