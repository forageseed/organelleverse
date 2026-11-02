"""Pixel size from TIFF metadata and raw-frame selection in camera TIFFs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from organelleverse.core.errors import OrganelleInputError
from organelleverse.morphology.image_job import load_analysis_image
from organelleverse.morphology.tiff_scale import read_tiff_scale, scale_from_sis


def _camera_tiff(path: Path) -> np.ndarray:
    """EMSIS-like layout: page 0 an 8-bit RGB preview, page 1 the 16-bit raw frame."""
    raw = (np.arange(40 * 60, dtype=np.uint16).reshape(40, 60) * 7) % 60000
    with tifffile.TiffWriter(path) as writer:
        writer.write(np.zeros((40, 60, 3), dtype=np.uint8), photometric="rgb")
        writer.write(raw, photometric="minisblack")
    return raw


def test_camera_tiff_reads_the_raw_frame_not_the_preview(tmp_path: Path) -> None:
    raw = _camera_tiff(tmp_path / "camera.tif")
    notes: dict = {}
    array = load_analysis_image(tmp_path / "camera.tif", notes=notes)
    assert array.dtype == np.uint16 and np.array_equal(array, raw)
    assert notes["tiff_series"]["chosen"] == 1


def test_two_equal_planes_are_a_stack_and_are_refused(tmp_path: Path) -> None:
    path = tmp_path / "stack.tif"
    with tifffile.TiffWriter(path) as writer:
        writer.write(np.zeros((8, 8), dtype=np.uint16))
        writer.write(np.ones((8, 8), dtype=np.uint16))
    with pytest.raises(OrganelleInputError) as error:
        load_analysis_image(path)
    assert error.value.code in {"image_job.multipage_unsupported", "image_job.not_2d"}


def test_scale_sources(tmp_path: Path) -> None:
    sis = scale_from_sis({"pixelsizex": 3.91992432593241e-09, "pixelsizey": 3.91992432593241e-09, "magnification": 2500.0})
    assert sis["um_per_px_x"] == pytest.approx(0.0039199, rel=1e-5)
    assert sis["magnification"] == 2500.0 and "33560" in sis["source"]
    assert scale_from_sis({"pixelsizex": 0}) is None

    imagej = tmp_path / "imagej.tif"
    tifffile.imwrite(imagej, np.zeros((10, 10), np.uint16), imagej=True, resolution=(1 / 0.05, 1 / 0.05),
                     metadata={"unit": "um"})
    found = read_tiff_scale(imagej)
    assert found["um_per_px_x"] == pytest.approx(0.05) and found["source"].startswith("ImageJ")

    ome = tmp_path / "ome.tif"
    tifffile.imwrite(ome, np.zeros((10, 10), np.uint16), ome=True,
                     metadata={"PhysicalSizeX": 0.12, "PhysicalSizeXUnit": "µm", "PhysicalSizeY": 0.12, "PhysicalSizeYUnit": "µm"})
    found = read_tiff_scale(ome)
    assert found["um_per_px_x"] == pytest.approx(0.12) and found["source"].startswith("OME")

    placeholder = tmp_path / "dpi.tif"  # 72 dpi is a placeholder, never a physical scale
    tifffile.imwrite(placeholder, np.zeros((10, 10), np.uint8), resolution=(72, 72))
    assert read_tiff_scale(placeholder) is None


def test_camera_tiff_is_staged_as_the_raw_frame_for_external_engines(tmp_path: Path) -> None:
    from organelleverse.morphology.image_job import _stage_tiff

    raw = _camera_tiff(tmp_path / "camera.tif")
    _stage_tiff(tmp_path / "camera.tif", tmp_path / "staged.tif")
    with tifffile.TiffFile(tmp_path / "staged.tif") as tif:
        assert len(tif.series) == 1
        assert np.array_equal(tif.series[0].asarray(), raw)
