"""Physical pixel size and analysis page from TIFF metadata.

Scale sources, in order (the first one present wins; every result names its source):

* Olympus SIS private tag 33560 (EMSIS/Olympus TEM cameras, e.g. Xarosa): ``pixelsizex`` /
  ``pixelsizey`` in metres, plus ``magnification``. The tag may sit on any page.
* OME-XML ``PhysicalSizeX`` / ``PhysicalSizeY`` with their unit (default µm).
* ImageJ: the ``unit`` in the ImageJ description together with X/YResolution (pixels per unit).

Baseline X/YResolution alone is never used: TEM and many camera TIFFs write a 72 dpi placeholder
there, which is not a physical scale. The result is a suggestion for the user to confirm.

Page choice: EMSIS files carry an 8-bit RGB JPEG preview as page 0 and the 16-bit raw frame as page 1.
:func:`raw_plane_series` picks the single-channel, highest bit-depth plane in that situation, so
analysis reads the raw frame instead of the lossy preview. A genuine stack (several planes of the
same kind) is left to the caller to reject.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

__all__ = ["raw_plane_series", "read_tiff_scale", "scale_from_sis"]

_UNIT_TO_UM = {"m": 1e6, "mm": 1e3, "cm": 1e4, "um": 1.0, "µm": 1.0, "micron": 1.0, "microns": 1.0,
               "micrometer": 1.0, "nm": 1e-3, "nanometer": 1e-3, "pm": 1e-6, "Å": 1e-4, "a": 1e-4}


def _unit_factor(unit: str | None) -> float | None:
    key = (unit or "").strip().replace("μ", "µ")  # Greek mu -> micro sign
    return _UNIT_TO_UM.get(key) or _UNIT_TO_UM.get(key.lower())


def scale_from_sis(value: dict[str, Any]) -> dict[str, Any] | None:
    """Olympus SIS tag dict (pixel sizes in metres) -> µm/px."""
    x, y = value.get("pixelsizex"), value.get("pixelsizey")
    if not isinstance(x, (int, float)) or not x > 0:
        return None
    y = y if isinstance(y, (int, float)) and y > 0 else x
    out: dict[str, Any] = {"um_per_px_x": float(x) * 1e6, "um_per_px_y": float(y) * 1e6,
                           "source": "Olympus SIS TIFF tag 33560"}
    magnification = value.get("magnification")
    if isinstance(magnification, (int, float)) and magnification > 0:
        out["magnification"] = float(magnification)
    return out


def _scale_from_ome(xml: str) -> dict[str, Any] | None:
    match = re.search(r"<Pixels\b[^>]*>", xml)
    if not match:
        return None
    attrs = dict(re.findall(r'(\w+)="([^"]*)"', match.group(0)))
    try:
        x = float(attrs["PhysicalSizeX"])
    except (KeyError, ValueError):
        return None
    try:
        y = float(attrs.get("PhysicalSizeY", x))
    except ValueError:
        y = x
    fx = _unit_factor(attrs.get("PhysicalSizeXUnit", "µm"))
    fy = _unit_factor(attrs.get("PhysicalSizeYUnit", attrs.get("PhysicalSizeXUnit", "µm")))
    if not fx or not fy or not x > 0 or not y > 0:
        return None
    return {"um_per_px_x": x * fx, "um_per_px_y": y * fy, "source": "OME-XML PhysicalSizeX/Y"}


def _scale_from_imagej(tif: Any) -> dict[str, Any] | None:
    meta = tif.imagej_metadata or {}
    factor = _unit_factor(meta.get("unit"))
    page = tif.pages[0]
    xres, yres = page.tags.get("XResolution"), page.tags.get("YResolution")
    if not factor or xres is None:
        return None

    def per_pixel(tag: Any) -> float | None:
        num, den = tag.value
        return den / num * factor if num else None

    x = per_pixel(xres)
    y = per_pixel(yres) if yres is not None else x
    if not x or not y:
        return None
    return {"um_per_px_x": x, "um_per_px_y": y, "source": "ImageJ unit and resolution"}


def read_tiff_scale(path: str | Path) -> dict[str, Any] | None:
    """µm/px suggested by the file's own metadata, or None when it carries no physical scale."""
    import tifffile

    try:
        with tifffile.TiffFile(path) as tif:
            for page in tif.pages:
                tag = page.tags.get("OlympusSIS") or page.tags.get(33560)
                if tag is not None and isinstance(tag.value, dict):
                    found = scale_from_sis(tag.value)
                    if found:
                        return found
            if tif.ome_metadata:
                found = _scale_from_ome(tif.ome_metadata)
                if found:
                    return found
            if tif.imagej_metadata:
                return _scale_from_imagej(tif)
    except (OSError, ValueError, tifffile.TiffFileError):
        return None
    return None


def raw_plane_series(tif: Any) -> int | None:
    """Index of the raw single-channel series when the others are only RGB previews, else None."""
    series = list(tif.series)
    if len(series) < 2:
        return None
    planes = [i for i, s in enumerate(series) if len(s.shape) == 2]
    previews = [i for i, s in enumerate(series) if len(s.shape) == 3 and s.shape[-1] in (3, 4)]
    if not planes or len(planes) + len(previews) != len(series):
        return None
    deepest = max(planes, key=lambda i: series[i].dtype.itemsize)
    others = [i for i in planes if i != deepest]
    # Several equally deep planes are a stack, not "preview + raw frame".
    if any(series[i].dtype.itemsize == series[deepest].dtype.itemsize for i in others):
        return None
    return deepest
