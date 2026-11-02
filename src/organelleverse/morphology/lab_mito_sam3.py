"""SAM 3 assistance for the lab_mito engine (image-job option ``sam3_recover``).

Meta's SAM 3 runs in its own interpreter (``<root>/data/sam3/env``, Python 3.12 + torch + ``sam3``)
through :mod:`.sam3_infer`; the fusion itself is :mod:`.sam3_fusion`. The lab U-Net probability stays the
judge of *what* is a mitochondrion -- SAM 3 zero-shot text prompts find no mitochondria on the lab EM
images. SAM 3 is prompted with the confident lab_mito objects of the same image as exemplars, and its
proposals that no object covers yet are added when the U-Net gives them a low but non-zero probability:
mitochondria the U-Net saw but its threshold cut off.

Redrawing every object's outline with SAM 3 (``refine``) is implemented but off: on the lab VAL groups it
lowered traced-ROI recall and IoU (SAM's outline does not follow the traced membrane).

Everything here works on the lab model's 6 nm/px grid, the grid the fusion rule was chosen on.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from ..core.errors import OrganelleInputError

__all__ = ["SAM3_RULE", "probability_objects", "assist_with_sam3", "sam3_availability", "sam3_python",
           "sam3_roots", "commit_headroom_gb", "ensure_memory_headroom", "failure_message"]

SAM3_DIR_REL = "data/sam3"
SAM3_PYTHON_REL = "data/sam3/env/bin/python"
#: Interpreter locations tried in order: POSIX env, Windows venv, Windows embeddable Python (CPU workstation).
SAM3_PYTHON_RELS = (SAM3_PYTHON_REL, "data/sam3/env/Scripts/python.exe", "data/sam3/env/python.exe")
SAM3_CHECKPOINT_REL = "data/sam3/weights/sam3.pt"
SAM3_BPE_REL = "data/sam3/weights/bpe_simple_vocab_16e6.txt.gz"
SAM3_TIMEOUT_S = 3600
_RUN_JSON_CAP = 65536
#: Fusion rule on the 6 nm/px grid, chosen on the lab VAL groups (see SAM3_RULE_SOURCE).
#: refine: redraw the U-Net objects with SAM 3 (loose_threshold: U-Net probability that makes a candidate;
#: refine_min_iou: the SAM outline must still overlap it this much; min_confidence: mean U-Net probability inside
#: the final outline); off = keep the production lab_mito objects. recover: None or (min SAM score, min mean U-Net
#: probability) for exemplar proposals no object covers yet.
SAM3_RULE: dict[str, Any] = {"refine": False, "loose_threshold": 0.5, "refine_min_iou": 0.7, "min_confidence": 0.5,
                             "recover": (0.5, 0.2), "context": 1.0, "detector_threshold": 0.05, "max_exemplars": 4}
SAM3_RULE_SOURCE = (
    "chosen on the lab VAL groups (2026-10-27, bench_rules.py, 28 images / 584 traced ROIs): most traced-ROI hits "
    "(IoU >= 0.5) among rules with ring false-positive pixels <= 1.10 x lab_mito and elongated recall >= lab_mito "
    "- 0.02, ties by mean ROI IoU. VAL: recall 0.884 -> 0.902, mean IoU 0.821 -> 0.838, ring FP +6 %, "
    "objects +12 %, elongated recall unchanged (0.816); refine-only rules were all worse (recall <= 0.866). "
    "TEST, read once (42 images / 713 ROIs; lab_mito reproduces its recorded 0.930): recall 0.930 -> 0.935 "
    "(+4 ROIs, none lost), mean IoU 0.823 -> 0.827, ring FP +5 %, objects +10 %, elongated unchanged (0.805)")
SAM3_SCOPE = ("SAM 3 adds objects similar to the confident lab_mito objects of the same image; which objects are "
              "mitochondria is still decided by the lab U-Net probability (SAM 3 zero-shot text prompts find no "
              "mitochondria on the lab EM images)")
#: Exemplar objects: the production lab_mito rule (kept in sync with image_job's _LAB_MITO_* constants).
_EXEMPLAR_THRESHOLD = 0.6
_EXEMPLAR_MIN_CONFIDENCE = 0.8
_OPENING_PX = 2


def sam3_python(root: str | Path) -> Path | None:
    """The SAM 3 interpreter under ``root``: a POSIX env, a Windows venv or a Windows embeddable Python."""
    for rel in SAM3_PYTHON_RELS:
        path = Path(root) / rel
        if path.is_file() and os.access(path, os.X_OK):
            return path
    return None


def sam3_roots(root: str | Path, environ: dict[str, str] | None = None) -> list[Path]:
    """Where SAM 3 is looked for: ``ORGANELLEVERSE_SAM3_ROOT`` alone when set; otherwise the model root of
    this target and then the OrganelleVerse home (``~/.organelleverse``, where a workstation keeps it)."""
    env = os.environ if environ is None else environ
    if env.get("ORGANELLEVERSE_SAM3_ROOT"):
        return [Path(env["ORGANELLEVERSE_SAM3_ROOT"])]
    home = Path(env.get("ORGANELLEVERSE_HOME") or Path.home() / ".organelleverse")
    return [Path(root)] if Path(root) == home else [Path(root), home]


def sam3_availability(root: str | Path, environ: dict[str, str] | None = None) -> dict[str, Any]:
    """Check (filesystem only, no run) whether the SAM 3 interpreter and weights are installed.

    The first of :func:`sam3_roots` that holds ``data/sam3`` is the install, complete or not; ``root`` in
    the result is that directory and ``python`` its interpreter.
    """
    roots = sam3_roots(root, environ)
    chosen = next((r for r in roots if (r / SAM3_DIR_REL).is_dir()), roots[0])
    missing = [rel for rel in (SAM3_CHECKPOINT_REL, SAM3_BPE_REL) if not (chosen / rel).is_file()]
    python = sam3_python(chosen)
    if python is None:
        missing.insert(0, SAM3_PYTHON_REL)
    return {"available": not missing, "root": str(chosen), "python": None if python is None else str(python),
            "missing": missing, "searched": [str(r) for r in roots]}


def _run_sam3(cmd: list[str], cwd: str, timeout: float) -> subprocess.CompletedProcess[str]:
    """Run the SAM 3 inference script on this machine (indirection point for tests)."""
    # UTF-8 both ways: on Windows the child would otherwise write and be read in the console code page.
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                          env=env, timeout=timeout, check=False)


#: Free commit charge (RAM + page file) a Windows CPU run needs: its measured peak was 7.3 GB (fp32 weights,
#: bf16 autocast, demo field 610 x 458 with 10 boxes + text proposals, 2026-10-31), plus a margin.
SAM3_WINDOWS_COMMIT_GB = 7.5
_ACCESS_VIOLATION = 0xC0000005


def commit_headroom_gb() -> float | None:
    """Memory this computer can still commit (RAM + page file) in GB; None off Windows."""
    if os.name != "nt":
        return None
    import ctypes

    class _MemoryStatus(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                    *((name, ctypes.c_ulonglong) for name in ("total_phys", "avail_phys", "total_commit",
                                                              "avail_commit", "total_virtual", "avail_virtual",
                                                              "avail_extended"))]

    status = _MemoryStatus()
    status.length = ctypes.sizeof(_MemoryStatus)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return status.avail_commit / 2**30


def ensure_memory_headroom(python: str, headroom_gb: Callable[[], float | None] = commit_headroom_gb) -> None:
    """Refuse a Windows SAM 3 run up front when the computer cannot commit its peak memory.

    Short of commit charge the run dies inside a native allocation (access violation) minutes in.
    """
    if not str(python).lower().endswith(".exe"):
        return
    free = headroom_gb()
    if free is not None and free < SAM3_WINDOWS_COMMIT_GB:
        raise OrganelleInputError(
            code="image_job.sam3_memory",
            message=(f"SAM 3 needs about {SAM3_WINDOWS_COMMIT_GB:g} GB of free memory on this computer and "
                     f"{free:.1f} GB is free; close other programs or run it on a server"),
            details={"needed_gb": SAM3_WINDOWS_COMMIT_GB, "free_gb": round(free, 2)}, retryable=True)


def failure_message(run: subprocess.CompletedProcess[str]) -> str:
    """What a failed SAM 3 run tells the user; memory exhaustion is named as such."""
    stderr = run.stderr or ""
    if (run.returncode & 0xFFFFFFFF) == _ACCESS_VIOLATION or "os error 1455" in stderr:
        return (f"SAM 3 ran out of memory on this computer (exit {run.returncode}); close other programs "
                "or use a server")
    return f"SAM 3 failed (exit {run.returncode})"


def _disk(radius: int) -> np.ndarray:
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    return xx * xx + yy * yy <= radius * radius


def probability_objects(prob6: np.ndarray, threshold: float, min_confidence: float,
                        min_area: int) -> tuple[np.ndarray, np.ndarray]:
    """Objects of a uint8 probability map: p >= threshold, fill + open r2, area and mean-probability floors.

    Returns the sequential label map and each label's mean probability (index = label, entry 0 unused).
    """
    from scipy import ndimage as ndi

    mask = ndi.binary_opening(ndi.binary_fill_holes(prob6 >= round(threshold * 255)), structure=_disk(_OPENING_PX))
    components, n = ndi.label(mask, structure=np.ones((3, 3), int))
    if n == 0:
        return components.astype(np.int32), np.zeros(1)
    areas = np.bincount(components.ravel(), minlength=n + 1)
    means = np.asarray(ndi.mean(prob6.astype(np.float32) / 255.0, components, index=np.arange(n + 1)))
    keep = (areas >= min_area) & (means >= min_confidence)
    keep[0] = False
    remap = np.zeros(n + 1, np.int32)
    remap[keep] = np.arange(1, int(keep.sum()) + 1)
    return remap[components], np.concatenate([[0.0], means[keep]])


def _failed(message: str, **details: Any) -> OrganelleInputError:
    return OrganelleInputError(code="image_job.sam3_failed", message=message, details=details)


def assist_with_sam3(work: Path, root: str, image6_path: Path, prob6: np.ndarray, *, min_area6: int,
                     log: Callable[[str], None], rule: dict[str, Any] | None = None,
                     ) -> tuple[np.ndarray, dict[int, float], dict[str, Any]]:
    """SAM 3 outlines for the lab_mito candidates (+ recovered objects) on the 6 nm grid.

    Returns the instance map (labels in priority order), each label's mean U-Net probability and the detail.
    """
    from scipy import ndimage as ndi

    from . import sam3_fusion, sam3_infer

    rule = dict(SAM3_RULE if rule is None else rule)
    availability = sam3_availability(root)
    if not availability["available"]:
        raise OrganelleInputError(
            code="image_job.engine_unavailable", message="SAM 3 is not installed on this compute target",
            details={"engine": "sam3", "root": availability["root"], "missing": availability["missing"]})
    root = availability["root"]
    if not rule["refine"] and rule["recover"] is None:
        raise ValueError("SAM 3 rule does nothing: neither refine nor recover is set")
    # SAM 3 runs with its install as the working directory, so its file arguments are absolute.
    work = Path(work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    # The production lab_mito objects: kept as they are without refine, and the exemplars of recover.
    confident, conf = probability_objects(prob6, _EXEMPLAR_THRESHOLD, _EXEMPLAR_MIN_CONFIDENCE, min_area6)
    modes: list[str] = []
    extra: list[str] = []
    loose = None
    if rule["refine"]:
        loose, _ = probability_objects(prob6, rule["loose_threshold"], 0.0, min_area6)
        cand_path = work / "candidates6nm.npy"
        np.save(cand_path, loose)
        modes.append("refine")
        extra += ["--candidates", str(cand_path)]
    if rule["recover"] is not None:
        slices = ndi.find_objects(confident)
        boxes = []
        for k in (np.argsort(-conf[1:]) + 1).tolist():
            s = slices[k - 1]
            if s is not None:
                boxes.append([s[0].start, s[1].start, s[0].stop, s[1].stop])
        ex_path = work / "exemplars6nm.json"
        ex_path.write_text(json.dumps(boxes), encoding="utf-8")
        modes.append("exemplar")
        extra += ["--exemplars", str(ex_path)]
    out = work / "sam3.npz"
    script = Path(__file__).with_name("sam3_infer.py")
    cmd = [availability["python"], str(script), "--checkpoint", str(Path(root) / SAM3_CHECKPOINT_REL),
           "--bpe", str(Path(root) / SAM3_BPE_REL), "--image", str(Path(image6_path).resolve()),
           "--mode", ",".join(modes),
           "--out", str(out), "--threshold", repr(rule["detector_threshold"]), "--context", repr(rule["context"]),
           *extra]
    log(f"SAM 3 {' + '.join(modes)} around {int(confident.max())} lab_mito objects")
    ensure_memory_headroom(cmd[0])
    try:
        run = _run_sam3(cmd, root, SAM3_TIMEOUT_S)
    except subprocess.TimeoutExpired as error:
        raise OrganelleInputError(code="image_job.sam3_failed", message=f"SAM 3 timed out after {SAM3_TIMEOUT_S} s",
                                  details={"root": root, "timeout_s": SAM3_TIMEOUT_S}, retryable=True) from error
    except OSError as error:
        raise _failed(f"SAM 3 could not be started: {error}", root=root, command=cmd) from error
    outputs = {m: (out if len(modes) == 1 else out.with_name(f"{out.stem}.{m}.npz")) for m in modes}
    missing = [str(p) for p in outputs.values() if not p.is_file()]
    if run.returncode != 0 or missing:
        raise _failed(failure_message(run), root=root, exit_code=run.returncode, missing=missing,
                      stdout_tail=(run.stdout or "")[-2000:], stderr_tail=(run.stderr or "")[-2000:])
    if loose is not None:
        with np.load(outputs["refine"], allow_pickle=False) as data:
            hypotheses = sam3_infer.unpack(data)
        kept = sam3_fusion.refine_objects(sam3_fusion.label_proposals(loose), hypotheses, prob6,
                                          min_iou=rule["refine_min_iou"], min_confidence=rule["min_confidence"],
                                          min_area=min_area6)
    else:
        kept = [sam3_infer.Proposal(p.box, p.mask, float(conf[p.ref]), "model", p.ref)
                for p in sam3_fusion.label_proposals(confident)]
    refined = sum(p.source == "sam3_refined" for p in kept)
    recovered = 0
    if "exemplar" in outputs:
        with np.load(outputs["exemplar"], allow_pickle=False) as data:
            pool = sam3_infer.unpack(data)
        min_score, min_conf = rule["recover"]
        before = len(kept)
        kept = sam3_fusion.add_recovered(kept, pool, prob6, prob6.shape, min_score=min_score,
                                         min_confidence=min_conf, min_area=min_area6)
        recovered = len(kept) - before
    labels6 = sam3_infer.paint(kept, prob6.shape)
    confidence = {k: round(p.score, 4) for k, p in enumerate(kept, 1)}
    info: dict[str, Any] | None = None
    info_path = out.with_suffix(".json")
    if info_path.is_file():
        if info_path.stat().st_size > _RUN_JSON_CAP:
            info = {"truncated": True, "size_bytes": info_path.stat().st_size}
        else:
            try:
                info = json.loads(info_path.read_text(encoding="utf-8"))
            except ValueError:
                info = {"unparseable": True}
    detail = {"rule": {k: (list(v) if isinstance(v, tuple) else v) for k, v in rule.items()},
              "rule_source": SAM3_RULE_SOURCE, "model_objects": int(confident.max()),
              "candidates": int(loose.max()) if loose is not None else None, "refined": int(refined),
              "kept_as_model": int(len(kept) - refined - recovered), "recovered": int(recovered),
              "command": [Path(cmd[0]).name, Path(cmd[1]).name, *cmd[2:]], "inference": info, "scope": SAM3_SCOPE}
    return labels6, confidence, detail
