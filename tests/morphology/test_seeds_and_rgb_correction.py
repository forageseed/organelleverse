"""Every separate object gets a watershed seed; RGB label corrections are seen on every channel."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from organelleverse.morphology.correction import correct
from organelleverse.morphology.measure import ORGSEG_PALETTE, _split_touching


def _disc(mask: np.ndarray, cy: int, cx: int, r: int) -> None:
    yy, xx = np.ogrid[: mask.shape[0], : mask.shape[1]]
    mask[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = True


def test_small_objects_next_to_a_large_one_or_the_edge_keep_their_seed() -> None:
    mask = np.zeros((200, 200), dtype=bool)
    _disc(mask, 100, 100, 30)  # large
    _disc(mask, 100, 145, 6)  # small, 9 px from the large one
    _disc(mask, 100, 5, 6)  # small, at the image edge
    _disc(mask, 170, 170, 6)  # small, isolated
    labels = _split_touching(mask)
    assert len(np.unique(labels[labels > 0])) == 4
    # no foreground pixel is left unlabelled
    assert not np.any(mask & (labels == 0))


def _rgb(path: Path, classes: np.ndarray) -> Path:
    image = np.zeros((*classes.shape, 3), dtype=np.uint8)
    for class_id, colour in ORGSEG_PALETTE.items():
        image[classes == class_id] = colour
    Image.fromarray(image).save(path)
    return path


def test_a_correction_that_changes_only_green_and_blue_is_recorded(tmp_path: Path) -> None:
    before = np.zeros((40, 40), dtype=np.int64)
    before[5:15, 5:15] = 2  # Mitochondria (0,128,0)
    after = before.copy()
    after[5:15, 5:15] = 4  # Nucleus (0,0,128): red channel unchanged
    result = correct(
        _rgb(tmp_path / "before.png", before),
        _rgb(tmp_path / "after.png", after),
        correction_type="relabel",
        reason="nucleus mislabelled as mitochondrion",
        operator="tester",
    )
    assert result.status == "ok", result.summary_text
    region = dict(result.metrics)["region"] if "region" in result.metrics else None
    text = str(result.model_dump())
    assert "2->4" in text, region
