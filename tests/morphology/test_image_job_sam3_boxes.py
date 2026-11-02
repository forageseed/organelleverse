"""``sam3_boxes``: candidate boxes (from a vision model or a person) become SAM 3 masks, against a FAKE SAM 3."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from organelleverse.core.errors import OrganelleInputError
from organelleverse.morphology import lab_mito_sam3, sam3_boxes, sam3_infer
from organelleverse.morphology.image_job import load_analysis_image, run_image_engine_probe, run_image_morphology

SHAPE = (120, 160)


def _sam3_root(root: Path) -> None:
    for rel, content in {lab_mito_sam3.SAM3_PYTHON_REL: b"#!/bin/sh\n", lab_mito_sam3.SAM3_CHECKPOINT_REL: b"sam3",
                         lab_mito_sam3.SAM3_BPE_REL: b"bpe"}.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        if rel.endswith("bin/python"):
            path.chmod(0o755)


class _Sam3:
    """Answers refine like SAM: per candidate a good disc (score .9) and a poor full-box mask (score .3)."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, cmd, cwd, timeout):
        self.calls.append(cmd)
        candidates = np.load(cmd[cmd.index("--candidates") + 1])
        props = []
        from scipy import ndimage as ndi
        for k, sl in enumerate(ndi.find_objects(candidates), 1):
            y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
            yy, xx = np.ogrid[y0:y1, x0:x1]
            disc = ((yy - (y0 + y1) / 2) ** 2 + (xx - (x0 + x1) / 2) ** 2) <= (min(y1 - y0, x1 - x0) / 2.5) ** 2
            props.append(sam3_infer.Proposal((y0, x0, y1, x1), disc, 0.9, "refine0", k))
            # A hypothesis spilling far outside the box: clipped back to it if it were chosen.
            wide = np.ones((min(SHAPE[0], y1 + 30) - y0, min(SHAPE[1], x1 + 30) - x0), bool)
            props.append(sam3_infer.Proposal((y0, x0, y0 + wide.shape[0], x0 + wide.shape[1]), wide, 0.3, "refine1", k))
        out = Path(cmd[cmd.index("--out") + 1])
        modes = cmd[cmd.index("--mode") + 1].split(",")
        if len(modes) == 1:
            np.savez(out, **sam3_infer.pack(props), mode="refine")
            return subprocess.CompletedProcess(cmd, 0, "", "")
        np.savez(out.with_name(f"{out.stem}.refine.npz"), **sam3_infer.pack(props), mode="refine")
        # Text mode: one object the boxes missed (kept), the first boxed object again (overlap, dropped),
        # a weak one (score, dropped) and a huge one (size, dropped).
        def disc(cy, cx, r):
            yy, xx = np.ogrid[cy - r:cy + r, cx - r:cx + r]
            return ((cy - r, cx - r, cy + r, cx + r), (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r)
        text = [sam3_infer.Proposal(*disc(90, 30, 12), 0.8, "text", 0), sam3_infer.Proposal(*disc(30, 30, 12), 0.95, "text", 0),
                sam3_infer.Proposal(*disc(20, 120, 10), 0.1, "text", 0), sam3_infer.Proposal(*disc(60, 80, 55), 0.9, "text", 0)]
        np.savez(out.with_name(f"{out.stem}.text.npz"), **sam3_infer.pack(text), mode="text")
        out.with_suffix(".json").write_text(json.dumps({"text": "membrane-bound organelle", "text_phrase_verified": True}),
                                            encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")


@pytest.fixture
def setup(tmp_path, monkeypatch):
    root = tmp_path / "root"
    _sam3_root(root)
    monkeypatch.setenv("ORGANELLEVERSE_FUSION3_ROOT", str(root))
    sam = _Sam3()
    monkeypatch.setattr(sam3_boxes, "_run_sam3", sam)
    import tifffile
    image = np.full(SHAPE, 120, np.uint8)
    path = tmp_path / "em.tif"
    tifffile.imwrite(path, image)
    return path, sam


def test_each_box_becomes_its_best_sam_mask_inside_the_box(tmp_path, setup) -> None:
    path, sam = setup
    boxes = [{"box": [10, 10, 50, 50], "call": "mitochondrion"}, {"box": [80, 40, 140, 100], "call": "uncertain"},
             {"box": [159, 0, 200, 5], "call": "mitochondrion"}]   # 1 px wide once clipped: dropped
    out = tmp_path / "out"
    run_image_morphology({"image": str(path), "engine": "sam3_boxes", "boxes": boxes, "min_area": 5,
                          "box_source": "vision model glm-5.3-flash + skill organelle-segmentation-measurement"},
                         output_dir=out, log=lambda _m: None)
    detail = json.loads((out / "evidence.json").read_text(encoding="utf-8"))["engine"]
    assert detail["method"] == "sam3_box_prompts" and detail["boxes"] == 2 and detail["objects"] == 2
    assert sorted(detail["calls"].values()) == ["mitochondrion", "uncertain"]
    assert "glm-5.3-flash" in detail["box_source"]
    labels = load_analysis_image(out / "labels.tif")
    assert int(labels.max()) == 2
    objects = json.loads((out / "objects.json").read_text(encoding="utf-8"))["objects"]
    assert [o["confidence"] for o in objects] == [0.9, 0.9]
    # The .9 disc won over the .3 box-filling hypothesis, and nothing lies outside its box.
    first = labels == int(next(k for k, v in detail["calls"].items() if v == "mitochondrion"))
    assert first[30, 30] and not first[11, 11] and not first[60:, :].any()
    assert "--mode" in sam.calls[0] and sam.calls[0][sam.calls[0].index("--mode") + 1] == "refine"


def test_boxes_are_required_and_validated(tmp_path, setup) -> None:
    path, _ = setup
    with pytest.raises(OrganelleInputError) as missing:
        run_image_morphology({"image": str(path), "engine": "sam3_boxes"}, output_dir=tmp_path / "a", log=lambda _m: None)
    assert missing.value.code == "image_job.boxes_missing"
    with pytest.raises(OrganelleInputError) as bad:
        sam3_boxes.parse_boxes([{"box": [1, 2, 3]}], SHAPE)
    assert bad.value.code == "image_job.box_invalid"
    assert sam3_boxes.parse_boxes(json.dumps([[5, 5, 20, 30]]), SHAPE) == [{"box": [5, 5, 20, 30], "call": "candidate"}]


def test_probe_reports_sam3_boxes(tmp_path, setup) -> None:
    report = run_image_engine_probe({}, output_dir=tmp_path / "probe", log=lambda _m: None)
    engines = report["engines"] if "engines" in report else json.loads(
        next((tmp_path / "probe").rglob("*.json")).read_text(encoding="utf-8"))["engines"]
    assert engines["sam3_boxes"]["available"] is True


def test_sam3_text_proposals_fill_in_objects_the_boxes_missed(tmp_path, setup) -> None:
    path, sam = setup
    out = tmp_path / "out"
    run_image_morphology({"image": str(path), "engine": "sam3_boxes", "min_area": 5,
                          "boxes": [{"box": [10, 10, 50, 50], "call": "mitochondrion"}],
                          "text_target": "mitochondria", "modality": "em", "width_px": [10, 40]},
                         output_dir=out, log=lambda _m: None)
    cmd = sam.calls[0]
    assert cmd[cmd.index("--mode") + 1] == "refine,text" and cmd[cmd.index("--target") + 1] == "mitochondria"
    detail = json.loads((out / "evidence.json").read_text(encoding="utf-8"))["engine"]
    assert detail["boxed_objects"] == 1 and detail["text_proposals"] == 4 and detail["text_added"] == 1
    assert detail["text_phrase"] == "membrane-bound organelle" and detail["text_phrase_verified"] is True
    assert sorted(detail["calls"].values()) == ["mitochondrion", "sam3_text"]
    labels = load_analysis_image(out / "labels.tif")
    assert int(labels.max()) == 2 and labels[90, 30] == int(next(k for k, v in detail["calls"].items() if v == "sam3_text"))
