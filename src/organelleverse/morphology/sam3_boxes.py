"""SAM 3 masks for given candidate boxes (the ``sam3_boxes`` image-job engine).

The boxes come from elsewhere: a vision model guided by the organelle segmentation skill's prompt
card, or a person. Each is ``{"box": [x0, y0, x1, y1], "call": "mitochondrion" | "uncertain" | ...}``
in source-image pixels. Every box becomes one candidate (an ellipse inscribed in it) for SAM 3's
``refine`` mode, which prompts it with the box and its centre on a context crop; the best of the
three hypotheses, kept inside the box, is the object. What the object is remains the box source's
claim: the objects are candidates for review, never counted organelles.

A weak box source misses objects. With ``text_target`` SAM 3 also runs its concept (text) mode with
the phrase measured for that organelle (``sam3_infer.concept_phrase``); a text proposal that a box
did not already cover, that scores at least ``TEXT_MIN_SCORE`` and whose size fits ``width_px``,
is added with the call ``sam3_text``. Such a phrase finds organelle-like objects, not one organelle
specifically, so these objects need the per-object identity check that follows.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Callable

import numpy as np

from organelleverse.core.errors import OrganelleInputError

SAM3_BOXES_TIMEOUT_S = 900
#: Extra time per box: one SAM 3 forward on a context crop each, ~14 s on a laptop CPU (2026-10-31).
SAM3_BOXES_PER_BOX_S = 30
#: Pixels a mask may extend past its box (SAM traces the membrane, the box may be tight).
BOX_MARGIN = 2
#: Text proposals: lowest SAM score kept (the skill's measured operating point on lab EM) and the
#: largest share of a proposal that may already belong to a boxed object.
TEXT_MIN_SCORE = 0.3
TEXT_MAX_OVERLAP = 0.2


def parse_boxes(raw: Any, shape: tuple[int, int]) -> list[dict[str, Any]]:
    """Validated boxes clipped to the image; refuses an empty or malformed list."""
    height, width = shape
    if isinstance(raw, str):
        raw = json.loads(raw)
    if not isinstance(raw, list) or not raw:
        raise OrganelleInputError(code="image_job.boxes_missing", message="the sam3_boxes engine needs candidate boxes",
                                  details={})
    boxes: list[dict[str, Any]] = []
    for item in raw:
        values = item.get("box") if isinstance(item, dict) else item
        if not isinstance(values, (list, tuple)) or len(values) != 4:
            raise OrganelleInputError(code="image_job.box_invalid", message="each box is [x0, y0, x1, y1]",
                                      details={"box": values})
        x0, y0, x1, y1 = (int(round(float(v))) for v in values)
        x0, x1 = sorted((max(0, min(width, x0)), max(0, min(width, x1))))
        y0, y1 = sorted((max(0, min(height, y0)), max(0, min(height, y1))))
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        call = str(item.get("call") or "candidate") if isinstance(item, dict) else "candidate"
        boxes.append({"box": [x0, y0, x1, y1], "call": call[:40]})
    if not boxes:
        raise OrganelleInputError(code="image_job.boxes_missing", message="no usable candidate box inside the image",
                                  details={"given": len(raw)})
    return boxes


def candidates_from_boxes(boxes: list[dict[str, Any]], shape: tuple[int, int]) -> np.ndarray:
    """Instance map: box k becomes label k as the ellipse inscribed in it (overlaps keep the earlier box)."""
    candidates = np.zeros(shape, np.uint16)
    for k, item in enumerate(boxes, 1):
        x0, y0, x1, y1 = item["box"]
        yy, xx = np.ogrid[y0:y1, x0:x1]
        cy, cx, ry, rx = (y0 + y1 - 1) / 2, (x0 + x1 - 1) / 2, max((y1 - y0) / 2, 1), max((x1 - x0) / 2, 1)
        inside = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1
        region = candidates[y0:y1, x0:x1]
        region[inside & (region == 0)] = k
    return candidates


def _run_sam3(cmd: list[str], cwd: str, timeout: float) -> subprocess.CompletedProcess[str]:
    """Run the SAM 3 inference script on this machine (indirection point for tests)."""
    # UTF-8 both ways: on Windows the child would otherwise write and be read in the console code page.
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                          env=env, timeout=timeout, check=False)


def _fits_width(mask: np.ndarray, width_px: tuple[float, float] | None) -> bool:
    """Equivalent diameter within half the smallest and 1.5 x the largest typical width."""
    if not width_px:
        return True
    diameter = 2 * float(np.sqrt(mask.sum() / np.pi))
    return 0.5 * width_px[0] <= diameter <= 1.5 * width_px[1]


def boxes_instances(array: np.ndarray, work: Path, raw_boxes: Any, *, root: str, min_area: int,
                    log: Callable[[str], None], text_target: str | None = None, modality: str = "em",
                    width_px: tuple[float, float] | None = None
                    ) -> tuple[np.ndarray, dict[str, Any], dict[int, float]]:
    """Labels, engine detail and per-object SAM score for the candidate boxes on ``array``."""
    from PIL import Image

    from . import sam3_infer
    from .lab_mito_sam3 import (SAM3_BPE_REL, SAM3_CHECKPOINT_REL, ensure_memory_headroom, failure_message,
                                sam3_availability)

    availability = sam3_availability(root)
    if not availability["available"]:
        raise OrganelleInputError(code="image_job.sam3_unavailable",
                                  message="SAM 3 is not installed on this compute target",
                                  details={"root": availability["root"], "missing": availability["missing"]})
    root = availability["root"]
    shape = (int(array.shape[0]), int(array.shape[1]))
    boxes = parse_boxes(raw_boxes, shape)
    # SAM 3 runs with its install as the working directory, so its file arguments are absolute.
    work = Path(work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    image_png = work / "image8.png"
    Image.fromarray(sam3_infer.stretch_to_uint8(np.asarray(array))).save(image_png)
    candidates_path = work / "box_candidates.npy"
    np.save(candidates_path, candidates_from_boxes(boxes, shape))
    out = work / "sam3_boxes.npz"
    script = Path(__file__).with_name("sam3_infer.py")
    cmd = [availability["python"], str(script), "--checkpoint", str(Path(root) / SAM3_CHECKPOINT_REL),
           "--bpe", str(Path(root) / SAM3_BPE_REL), "--image", str(image_png),
           "--mode", "refine,text" if text_target else "refine",
           "--candidates", str(candidates_path), "--out", str(out), "--context", "1.0"]
    if text_target:
        cmd += ["--target", text_target, "--modality", modality]
    # With two modes the script writes one file per mode next to ``out``.
    refine_out = out.with_name(f"{out.stem}.refine.npz") if text_target else out
    text_out = out.with_name(f"{out.stem}.text.npz")
    log(f"SAM 3 masks for {len(boxes)} candidate boxes" + (f" and its own '{text_target}' proposals" if text_target else ""))
    timeout = max(SAM3_BOXES_TIMEOUT_S, 120 + SAM3_BOXES_PER_BOX_S * len(boxes))
    ensure_memory_headroom(cmd[0])
    try:
        run = _run_sam3(cmd, root, timeout)
    except subprocess.TimeoutExpired as error:
        raise OrganelleInputError(code="image_job.sam3_failed", message=f"SAM 3 timed out after {timeout} s",
                                  details={"root": root}, retryable=True) from error
    except OSError as error:
        raise OrganelleInputError(code="image_job.sam3_failed", message=f"SAM 3 could not be started: {error}",
                                  details={"root": root}) from error
    if run.returncode != 0 or not refine_out.is_file():
        raise OrganelleInputError(code="image_job.sam3_failed", message=failure_message(run),
                                  details={"stderr_tail": (run.stderr or "")[-2000:]})
    with np.load(refine_out, allow_pickle=False) as data:
        hypotheses = sam3_infer.unpack(data)
    best: dict[int, Any] = {}
    for proposal in hypotheses:
        if proposal.ref not in best or proposal.score > best[proposal.ref].score:
            best[proposal.ref] = proposal
    labels = np.zeros(shape, np.uint16)
    confidence: dict[int, float] = {}
    calls: dict[int, str] = {}
    dropped = 0
    # Most confident first, so where two masks meet the surer one keeps the pixels.
    for ref in sorted(best, key=lambda r: -best[r].score):
        proposal = best[ref]
        x0, y0, x1, y1 = boxes[ref - 1]["box"]
        full = np.zeros(shape, bool)
        py0, px0, py1, px1 = proposal.box
        full[py0:py1, px0:px1] = proposal.mask
        keep = np.zeros(shape, bool)
        keep[max(0, y0 - BOX_MARGIN):y1 + BOX_MARGIN, max(0, x0 - BOX_MARGIN):x1 + BOX_MARGIN] = True
        full &= keep & (labels == 0)
        if int(full.sum()) < min_area:
            dropped += 1
            continue
        label = len(confidence) + 1
        labels[full] = label
        confidence[label] = round(float(proposal.score), 4)
        calls[label] = boxes[ref - 1]["call"]
    boxed = len(confidence)
    info_path = out.with_suffix(".json")
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.is_file() else {}
    text_detail: dict[str, Any] = {}
    if text_target:
        text_props: list[Any] = []
        if text_out.is_file():
            with np.load(text_out, allow_pickle=False) as data:
                text_props = sam3_infer.unpack(data)
        added = 0
        for proposal in sorted(text_props, key=lambda p: -p.score):
            if proposal.score < TEXT_MIN_SCORE:
                continue
            full = np.zeros(shape, bool)
            py0, px0, py1, px1 = proposal.box
            full[py0:py1, px0:px1] = proposal.mask
            area = int(full.sum())
            if area < min_area or not _fits_width(full, width_px):
                continue
            if int((full & (labels > 0)).sum()) > TEXT_MAX_OVERLAP * area:
                continue
            full &= labels == 0
            if int(full.sum()) < min_area:
                continue
            label = len(confidence) + 1
            labels[full] = label
            confidence[label] = round(float(proposal.score), 4)
            calls[label] = "sam3_text"
            added += 1
        text_detail = {"text_target": text_target, "text_phrase": info.get("text"),
                       "text_phrase_verified": info.get("text_phrase_verified"),
                       "text_proposals": len(text_props), "text_added": added}
    detail = {"method": "sam3_box_prompts", "boxes": len(boxes), "objects": len(confidence), "boxed_objects": boxed,
              "dropped_small_or_empty": dropped + (len(boxes) - len(best)), "calls": calls, **text_detail,
              # where and how SAM 3 ran (device chosen by the SAM 3 interpreter, see sam3_infer --device auto)
              "inference": {key: info.get(key) for key in ("device", "torch", "weights_read", "load_s",
                                                            "mode_seconds", "total_s")},
              "note": "object identity is the box source's claim (sam3_text: SAM 3's own proposal, identity "
                      "unchecked); review before counting"}
    return labels, detail, confidence
