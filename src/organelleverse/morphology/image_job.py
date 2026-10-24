"""Image-workbench jobs that run unchanged on any compute target.

Two built-in tool runners execute on whichever machine the plugin run lands
on — the Windows desktop service itself, a WSL distribution, or an SSH server
through the Linux provider — so the same code and the same output contract
apply everywhere:

``image_morphology``
    Segment one image and measure its objects. ``engine="classical"`` is a
    deterministic Otsu threshold → connected components → morphometrics
    pipeline that must give identical tables on every target. ``engine=
    "micro_sam"`` runs micro-SAM automatic instance segmentation where it is
    installed. ``engine="fusion3"`` calls the locally installed fusion3-v1
    delivery (no SSH: the job already runs on the compute target) and turns
    one class (chloroplast or mitochondria) of its class map into instances.
    ``engine="lab_mito"`` runs the lab's own fine-tuned mitochondria U-Net
    (6 nm/px model, needs a calibrated pixel size) through the same delivery
    interpreter and turns its mask into instances.
    Outputs: ``labels.tif`` (uint16 instance map on the source
    pixel grid), ``objects.json`` (the measure payload), ``objects.csv`` and
    ``evidence.json`` (host, environment, input/output digests, parameters).

``image_engine_probe``
    Report which engines this machine can run (imports, versions, GPU). It
    never installs or downloads anything.

The analysis pixels are read from the uploaded original with its native bit
depth; display previews never enter these runners.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..core.errors import OrganelleInputError

__all__ = [
    "IMAGE_JOB_RUNNERS",
    "classical_instances",
    "fusion3_availability",
    "lab_mito_availability",
    "load_analysis_image",
    "run_image_engine_probe",
    "run_image_morphology",
]

ENGINES = ("classical", "micro_sam", "fusion3", "lab_mito")
FUSION3_ORGANELLES = {"chloroplast": 1, "mitochondria": 2}
#: Interpreters the delivery needs, relative to its project root.
_FUSION3_PYTHON_REL = "data/orgsegnet/training-venv/bin/python"
_FUSION3_BASELINE_PYTHON_REL = "data/orgsegnet/orgsegnet-baseline-venv/bin/python"
_FUSION3_TIMEOUT_S = 3600
_FUSION3_RUN_JSON_CAP = 65536
_FUSION3_SCOPE = ("validated on OrgSegNet V6-corpus pixel scale only; "
                  "touching chloroplasts are split by a distance watershed; touching mitochondria merge into one instance")
#: lab_mito: the lab fine-tuned mitochondria U-Net, run by the delivery interpreter via lab_mito_infer.py.
_LAB_MITO_CHECKPOINT_REL = "experiments/plant-sam/lab-finetune-v1-s1234/best.pt"
_LAB_MITO_MODEL_REL = "scripts/plant-sam/organelle_unet.py"
_LAB_MITO_PREDICT_REL = "scripts/plant-sam/predict_organelle_unet.py"
_LAB_MITO_TIMEOUT_S = 3600
_LAB_MITO_NM_PER_PX = 6.0
_LAB_MITO_MIN_AREA_UM2 = 0.05  # the lab labels' own floor
# Decision rule on the 6 nm/px grid (chosen on the lab VAL groups, see _LAB_MITO_RULE_SOURCE):
# None = fused argmax, else mean(p_rel1, p_rel2) >= threshold; opening radius in 6 nm pixels after
# filling holes (0 = none); components whose mean probability is below the floor are dropped.
# Applied only when the inference script wrote the probability map; otherwise the fused argmax is
# kept unchanged (no opening, no floor).
_LAB_MITO_THRESHOLD: float | None = 0.6
_LAB_MITO_OPENING_PX = 2
_LAB_MITO_MIN_CONFIDENCE = 0.8
_LAB_MITO_RULE_SOURCE = (
    "chosen on the lab VAL groups (2026-10-11, select_lab_rule.py): fewest ring false-positive pixels among "
    "rules keeping traced-ROI recall >= argmax - 0.01 and elongated-ROI recall >= argmax - 0.02. "
    "TEST (42 images, 713 ROIs) vs argmax: ROI recall 0.930 vs 0.917, ring FP pixels -35%, objects -56%, "
    "elongated recall 0.805 vs 0.844 (62 vs 65 of 77)")
_LAB_MITO_SCOPE = ("fine-tuned on the user's alfalfa EM (roi-labels-v1 train split); test split: Dice 0.903, "
                   "traced-mitochondria recall 0.919 (42 images); mitochondria only; "
                   "whole-image precision not measured (partial labels)")
_PACKAGES = ("numpy", "scipy", "scikit-image", "tifffile", "pillow", "organelleverse")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_versions(names: tuple[str, ...] = _PACKAGES) -> dict[str, str | None]:
    from importlib import metadata

    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _host_facts() -> dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "packages": _package_versions(),
    }


def load_analysis_image(path: str | Path, *, notes: dict[str, Any] | None = None) -> np.ndarray:
    """Read the stored original as a 2-D array with its native dtype.

    TIFF goes through tifffile when available (keeps 16/32-bit and float
    planes); other formats through Pillow. A multi-page/stack file is
    rejected rather than silently reduced to its first page. RGB(A) images
    are combined to luminance with the ITU-R BT.601 weights, a documented
    choice recorded in the evidence; fluorescence channels must be chosen
    upstream (the microscopy reader exports one channel plane).
    """
    source = Path(path)
    array: np.ndarray
    if source.suffix.lower() in {".tif", ".tiff"}:
        try:
            import tifffile

            from .tiff_scale import raw_plane_series

            with tifffile.TiffFile(source) as tif:
                # Camera TIFFs (e.g. EMSIS Xarosa) store an 8-bit RGB JPEG preview first and the
                # raw 16-bit frame second; analyse the raw frame, never the lossy preview.
                raw = raw_plane_series(tif)
                if raw is not None:
                    array = np.asarray(tif.series[raw].asarray())
                    if notes is not None:
                        notes["tiff_series"] = {"chosen": raw, "series": len(tif.series),
                                                "reason": "raw single-channel frame next to RGB preview"}
                else:
                    if len(tif.series) > 1:
                        raise OrganelleInputError(
                            code="image_job.multipage_unsupported",
                            message="multi-series TIFF needs an explicit page/channel choice",
                            details={"path": source.name, "series": len(tif.series)},
                        )
                    array = np.asarray(tif.series[0].asarray())
        except ImportError:
            from PIL import Image

            with Image.open(source) as image:
                if getattr(image, "n_frames", 1) > 1:
                    raise OrganelleInputError(
                        code="image_job.multipage_unsupported",
                        message="multi-page TIFF needs an explicit page/channel choice",
                        details={"path": source.name, "pages": image.n_frames},
                    ) from None
                array = np.asarray(image)
    else:
        from PIL import Image

        with Image.open(source) as image:
            array = np.asarray(image)
    if notes is not None:
        notes["stored_dtype"] = str(array.dtype)
        notes["stored_shape"] = list(array.shape)
    if array.ndim == 3 and array.shape[-1] in (3, 4):
        rgb = array[..., :3].astype(np.float64)
        array = rgb @ np.array([0.299, 0.587, 0.114])
        if notes is not None:
            notes["rgb_to_luminance"] = "BT.601"
    if array.ndim != 2:
        raise OrganelleInputError(
            code="image_job.not_2d",
            message="image analysis jobs take one 2-D plane",
            details={"shape": list(array.shape)},
        )
    return array


def classical_instances(array: np.ndarray, *, min_area: int) -> tuple[np.ndarray, float]:
    """Otsu threshold → 8-connected components → drop objects below min_area.

    Returns a uint16 instance map (0 = background, 1..N sequential) and the
    threshold used. Deterministic: identical input and parameters give an
    identical map on every platform.
    """
    from skimage.filters import threshold_otsu
    from skimage.measure import label
    from skimage.segmentation import relabel_sequential

    values = array.astype(np.float64, copy=False)
    if float(values.max()) == float(values.min()):
        return np.zeros(values.shape, dtype=np.uint16), float(values.min())
    threshold = float(threshold_otsu(values))
    components = label(values > threshold, connectivity=2)
    # Keep components with area >= min_area. Done explicitly because
    # scikit-image changed remove_small_objects' boundary rule in 0.26, and
    # targets with different versions must give the same map.
    areas = np.bincount(components.ravel())
    keep = areas >= min_area
    keep[0] = False
    labels, _, _ = relabel_sequential(np.where(keep[components], components, 0))
    if int(labels.max()) > np.iinfo(np.uint16).max:
        raise OrganelleInputError(
            code="image_job.too_many_objects",
            message="more objects than a uint16 instance map can hold",
            details={"objects": int(labels.max())},
        )
    return labels.astype(np.uint16), threshold


def _micro_sam_instances(image_path: Path, work: Path, *, model_type: str, device: str) -> np.ndarray:
    """micro-SAM automatic instance segmentation on this machine."""
    from .microsam import default_micro_sam_executor

    work.mkdir(parents=True, exist_ok=True)
    produced = [Path(p) for p in default_micro_sam_executor(model_type, device, [str(image_path)], work)]
    if not produced:
        raise OrganelleInputError(
            code="image_job.micro_sam_no_output",
            message="micro-SAM finished without writing a label map",
            details={"work_dir": str(work)},
        )
    labels = load_analysis_image(produced[0])
    return labels.astype(np.uint16)


def fusion3_availability(environ: dict[str, str] | None = None) -> dict[str, Any]:
    """Check (filesystem only, no run) whether the fusion3-v1 delivery is on this machine."""
    from .fusion3 import FUSION3_CONFIG_REL, FUSION3_ENTRY_REL, fusion3_env

    root = Path(fusion3_env(environ)["root"])
    missing = [rel for rel in (FUSION3_CONFIG_REL, FUSION3_ENTRY_REL)
               if not (root / rel).is_file()]
    missing += [rel for rel in (_FUSION3_PYTHON_REL, _FUSION3_BASELINE_PYTHON_REL)
                if not ((root / rel).is_file() and os.access(root / rel, os.X_OK))]
    return {"available": not missing, "root": str(root), "missing": missing}


def lab_mito_availability(environ: dict[str, str] | None = None) -> dict[str, Any]:
    """Check (filesystem only, no run) whether the lab mitochondria model is on this machine."""
    from .fusion3 import fusion3_env

    root = Path(fusion3_env(environ)["root"])
    missing = [rel for rel in (_LAB_MITO_CHECKPOINT_REL, _LAB_MITO_MODEL_REL, _LAB_MITO_PREDICT_REL)
               if not (root / rel).is_file()]
    if not ((root / _FUSION3_PYTHON_REL).is_file() and os.access(root / _FUSION3_PYTHON_REL, os.X_OK)):
        missing.insert(0, _FUSION3_PYTHON_REL)
    return {"available": not missing, "root": str(root), "missing": missing}


def _run_delivery(cmd: list[str], cwd: str, timeout: float) -> subprocess.CompletedProcess[str]:
    """Run the delivery command on this machine (indirection point for tests)."""
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False)


def _stage_tiff(source: Path, target: Path) -> None:
    """Stage the input as a TIFF with its pixel values unchanged.

    A camera TIFF holding an RGB preview next to the raw frame (EMSIS) is staged as the raw frame
    alone: the external scripts read the first series, which would otherwise be the lossy preview.
    """
    if source.suffix.lower() in {".tif", ".tiff"}:
        try:
            import tifffile

            from .tiff_scale import raw_plane_series

            with tifffile.TiffFile(source) as tif:
                raw = raw_plane_series(tif)
                if raw is not None:
                    tifffile.imwrite(target, np.asarray(tif.series[raw].asarray()), compression=None)
                    return
        except ImportError:
            pass
        shutil.copyfile(source, target)
        return
    from PIL import Image

    with Image.open(source) as image:
        pixels = np.asarray(image)
    try:
        import tifffile

        tifffile.imwrite(target, pixels, compression=None)
    except ImportError:
        Image.fromarray(pixels).save(target, format="TIFF")


def _run_lab_mito(cmd: list[str], cwd: str, timeout: float) -> subprocess.CompletedProcess[str]:
    """Run the lab_mito inference script on this machine (indirection point for tests)."""
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False)


#: Touching chloroplasts merge into one class-map component. Splitting them with a distance-transform
#: watershed (peaks at least ``min_distance`` apart, computed at 1/``downsample`` resolution) was chosen on the
#: OrgSegNet V6 val split (instance F1 0.817 -> 0.834) and confirmed once on test; mitochondria gained nothing.
_CHLOROPLAST_SPLIT = {"downsample": 4, "min_distance": 140}


def _split_touching(mask: np.ndarray, *, downsample: int, min_distance: int) -> np.ndarray:
    """Instance labels for a binary mask, splitting touching objects by a distance-transform watershed."""
    from scipy import ndimage as ndi
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed

    small = mask[::downsample, ::downsample]
    components, _ = ndi.label(small, structure=np.ones((3, 3), int))
    distance = ndi.distance_transform_edt(small)
    peaks = peak_local_max(distance, min_distance=min_distance, labels=components, exclude_border=False)
    markers = np.zeros(small.shape, np.int32)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    # A component without a peak keeps one marker so it is never dropped.
    for k in np.setdiff1d(np.unique(components[components > 0]), np.unique(components[markers > 0])):
        ys, xs = np.nonzero(components == k)
        markers[ys[0], xs[0]] = markers.max() + 1
    coarse = watershed(-distance, markers, mask=small)
    # Back to full resolution inside the full-resolution mask. Leftover pixels take the nearest label of
    # their own full-resolution component; components the coarse grid never sampled stay separate objects.
    full = np.repeat(np.repeat(coarse, downsample, 0), downsample, 1)[:mask.shape[0], :mask.shape[1]]
    full = np.where(mask, full, 0).astype(np.int64)
    fine, _ = ndi.label(mask, structure=np.ones((3, 3), int))
    missing = mask & (full == 0)
    if missing.any() and full.any():
        _, (iy, ix) = ndi.distance_transform_edt(full == 0, return_indices=True)
        same = fine[iy, ix] == fine
        full = np.where(missing & same, full[iy, ix], full)
    left = mask & (full == 0)
    if left.any():
        full[left] = full.max() + fine[left]
    return full


def _fusion3_instances(image_path: Path, work: Path, *, organelle: str, min_area: int,
                       log: Callable[[str], None]) -> tuple[np.ndarray, dict[str, Any]]:
    """Run the fusion3-v1 delivery locally and turn one class into instance labels."""
    from skimage.measure import label
    from skimage.segmentation import relabel_sequential

    from .fusion3 import (
        FUSION3_CLASS_NAMES,
        FUSION3_CONFIG_REL,
        FUSION3_DELIVERY_NAME,
        FUSION3_ENTRY_REL,
    )

    availability = fusion3_availability()
    root = availability["root"]
    if not availability["available"]:
        raise OrganelleInputError(
            code="image_job.engine_unavailable",
            message=f"{FUSION3_DELIVERY_NAME} delivery is not installed on this compute target",
            details={"engine": "fusion3", "root": root, "missing": availability["missing"]})
    class_id = FUSION3_ORGANELLES[organelle]
    config_path = Path(root) / FUSION3_CONFIG_REL
    stem = "source"
    staged_dir, out_dir = work / "in", work / "out"
    staged_dir.mkdir(parents=True, exist_ok=True)
    _stage_tiff(image_path, staged_dir / f"{stem}.tif")
    cmd = [str(Path(root) / _FUSION3_PYTHON_REL), FUSION3_ENTRY_REL, "--image-dir", str(staged_dir),
           "--output", str(out_dir), "--names", stem, "--clean"]
    log(f"{FUSION3_DELIVERY_NAME} delivery ({organelle}, class {class_id})")
    try:
        run = _run_delivery(cmd, root, _FUSION3_TIMEOUT_S)
    except subprocess.TimeoutExpired as error:
        raise OrganelleInputError(
            code="image_job.fusion3_failed",
            message=f"{FUSION3_DELIVERY_NAME} delivery timed out after {_FUSION3_TIMEOUT_S} s",
            details={"root": root, "timeout_s": _FUSION3_TIMEOUT_S},
            retryable=True) from error
    except OSError as error:
        raise OrganelleInputError(
            code="image_job.fusion3_failed",
            message=f"{FUSION3_DELIVERY_NAME} delivery could not be started: {error}",
            details={"root": root, "command": cmd}) from error
    if run.returncode != 0:
        raise OrganelleInputError(
            code="image_job.fusion3_failed",
            message=f"{FUSION3_DELIVERY_NAME} delivery failed (exit {run.returncode})",
            details={"root": root, "exit_code": run.returncode,
                     "stdout_tail": (run.stdout or "")[-2000:], "stderr_tail": (run.stderr or "")[-2000:]})
    class_map_path = out_dir / f"{stem}.class.tif"
    if not class_map_path.is_file():
        raise OrganelleInputError(
            code="image_job.fusion3_failed",
            message=f"{FUSION3_DELIVERY_NAME} delivery wrote no class map",
            details={"root": root, "expected": str(class_map_path),
                     "stdout_tail": (run.stdout or "")[-2000:]})
    class_map = load_analysis_image(class_map_path)
    split = _CHLOROPLAST_SPLIT if organelle == "chloroplast" else None
    components = (_split_touching(class_map == class_id, **split) if split
                  else label(class_map == class_id, connectivity=2))
    # Keep components with area >= min_area (explicit, see classical_instances).
    areas = np.bincount(components.ravel())
    keep = areas >= min_area
    keep[0] = False
    labels, _, _ = relabel_sequential(np.where(keep[components], components, 0))
    if int(labels.max()) > np.iinfo(np.uint16).max:
        raise OrganelleInputError(
            code="image_job.too_many_objects",
            message="more objects than a uint16 instance map can hold",
            details={"objects": int(labels.max())})
    delivery_run: dict[str, Any] | None = None
    run_json = out_dir / "delivery-run.json"
    if run_json.is_file():
        if run_json.stat().st_size > _FUSION3_RUN_JSON_CAP:
            delivery_run = {"truncated": True, "size_bytes": run_json.stat().st_size}
        else:
            try:
                delivery_run = json.loads(run_json.read_text(encoding="utf-8"))
            except ValueError:
                delivery_run = {"unparseable": True}
    detail = {
        "method": "fusion3_semantic_segmentation+connected_components",
        "delivery": FUSION3_DELIVERY_NAME, "root": root, "config_sha256": _sha256(config_path),
        "organelle": organelle, "class_id": class_id, "class_name": FUSION3_CLASS_NAMES[class_id],
        "connectivity": 8, "split_touching": ({"method": "distance_watershed", **split} if split else None),
        "command": [Path(cmd[0]).name, *cmd[1:]], "delivery_run": delivery_run,
        "scope": _FUSION3_SCOPE,
    }
    return labels.astype(np.uint16), detail


def _lab_mito_instances(image_path: Path, work: Path, *, shape: tuple[int, int], pixel_size_um: float,
                       min_area: int, log: Callable[[str], None]) -> tuple[np.ndarray, dict[str, Any]]:
    """Run the lab fine-tuned mitochondria U-Net locally and turn its mask into instance labels."""
    import math

    from skimage.measure import label
    from skimage.segmentation import relabel_sequential

    availability = lab_mito_availability()
    root = availability["root"]
    if not availability["available"]:
        raise OrganelleInputError(
            code="image_job.engine_unavailable",
            message="the lab mitochondria model is not installed on this compute target",
            details={"engine": "lab_mito", "root": root, "missing": availability["missing"]})
    nm = pixel_size_um * 1000.0
    checkpoint = Path(root) / _LAB_MITO_CHECKPOINT_REL
    staged_dir = work / "in"
    staged_dir.mkdir(parents=True, exist_ok=True)
    staged = staged_dir / "source.tif"
    _stage_tiff(image_path, staged)
    mask_path = work / "out" / "mask6nm.npy"
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).with_name("lab_mito_infer.py")
    cmd = [str(Path(root) / _FUSION3_PYTHON_REL), str(script), "--root", root, "--checkpoint", str(checkpoint),
           "--image", str(staged), "--nm", repr(nm), "--out", str(mask_path)]
    log(f"lab mitochondria model at {nm:g} nm/px (resample x{nm / _LAB_MITO_NM_PER_PX:.4g} to 6 nm/px)")
    try:
        run = _run_lab_mito(cmd, root, _LAB_MITO_TIMEOUT_S)
    except subprocess.TimeoutExpired as error:
        raise OrganelleInputError(
            code="image_job.lab_mito_failed",
            message=f"lab mitochondria model timed out after {_LAB_MITO_TIMEOUT_S} s",
            details={"root": root, "timeout_s": _LAB_MITO_TIMEOUT_S}, retryable=True) from error
    except OSError as error:
        raise OrganelleInputError(
            code="image_job.lab_mito_failed",
            message=f"lab mitochondria model could not be started: {error}",
            details={"root": root, "command": cmd}) from error
    if run.returncode != 0:
        raise OrganelleInputError(
            code="image_job.lab_mito_failed",
            message=f"lab mitochondria model failed (exit {run.returncode})",
            details={"root": root, "exit_code": run.returncode,
                     "stdout_tail": (run.stdout or "")[-2000:], "stderr_tail": (run.stderr or "")[-2000:]})
    if not mask_path.is_file():
        raise OrganelleInputError(
            code="image_job.lab_mito_failed",
            message="lab mitochondria model wrote no mask",
            details={"root": root, "expected": str(mask_path), "stdout_tail": (run.stdout or "")[-2000:]})
    mask6 = np.load(mask_path, allow_pickle=False).astype(bool)
    if mask6.ndim != 2 or mask6.size == 0:
        raise OrganelleInputError(
            code="image_job.lab_mito_failed", message="lab mitochondria model wrote an unusable mask",
            details={"root": root, "mask_shape": list(mask6.shape)})
    prob_path = mask_path.with_name(mask_path.stem + ".prob.npy")
    prob6: np.ndarray | None = None
    if prob_path.is_file():
        prob6 = np.load(prob_path, allow_pickle=False)
        if prob6.shape != mask6.shape:
            raise OrganelleInputError(
                code="image_job.lab_mito_failed", message="lab mitochondria probability and mask differ in size",
                details={"root": root, "mask_shape": list(mask6.shape), "prob_shape": list(prob6.shape)})
    if prob6 is not None and _LAB_MITO_THRESHOLD is not None:
        mask6 = prob6 >= round(_LAB_MITO_THRESHOLD * 255)
    if prob6 is not None and _LAB_MITO_OPENING_PX > 0:
        from scipy import ndimage as ndi

        radius = _LAB_MITO_OPENING_PX
        yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
        mask6 = ndi.binary_opening(ndi.binary_fill_holes(mask6), structure=xx * xx + yy * yy <= radius * radius)
    # Nearest-neighbour back to the original pixel grid (pixel centres), exactly the image shape.
    rows = np.minimum(((np.arange(shape[0]) + 0.5) * mask6.shape[0] / shape[0]).astype(np.int64),
                      mask6.shape[0] - 1)
    cols = np.minimum(((np.arange(shape[1]) + 0.5) * mask6.shape[1] / shape[1]).astype(np.int64),
                      mask6.shape[1] - 1)
    mask = mask6[np.ix_(rows, cols)]
    components = label(mask, connectivity=2)
    floor = max(min_area, math.ceil(round(_LAB_MITO_MIN_AREA_UM2 / (pixel_size_um ** 2), 6)))
    areas = np.bincount(components.ravel())
    keep = areas >= floor
    keep[0] = False
    prob = prob6[np.ix_(rows, cols)].astype(np.float32) / 255.0 if prob6 is not None else None
    if prob is not None and components.max():
        from scipy import ndimage as ndi

        means = np.asarray(ndi.mean(prob, components, index=np.arange(components.max() + 1)))
        if _LAB_MITO_MIN_CONFIDENCE > 0:
            keep &= means >= _LAB_MITO_MIN_CONFIDENCE
    labels, forward, _ = relabel_sequential(np.where(keep[components], components, 0))
    # Mean mitochondrion probability of every kept object, keyed by its final label.
    confidence: dict[int, float] = {}
    if prob is not None and components.max():
        for old in np.flatnonzero(keep):
            confidence[int(forward[old])] = round(float(means[old]), 4)
    if int(labels.max()) > np.iinfo(np.uint16).max:
        raise OrganelleInputError(
            code="image_job.too_many_objects",
            message="more objects than a uint16 instance map can hold",
            details={"objects": int(labels.max())})
    inference: dict[str, Any] | None = None
    info_path = mask_path.with_suffix(".json")
    if info_path.is_file():
        if info_path.stat().st_size > _FUSION3_RUN_JSON_CAP:
            inference = {"truncated": True, "size_bytes": info_path.stat().st_size}
        else:
            try:
                inference = json.loads(info_path.read_text(encoding="utf-8"))
            except ValueError:
                inference = {"unparseable": True}
    info = inference if isinstance(inference, dict) else {}
    detail = {
        "method": "lab_finetuned_unet_mitochondria+connected_components",
        "root": root, "checkpoint": str(checkpoint),
        "checkpoint_sha256": info.get("checkpoint_sha256") or _sha256(checkpoint),
        "model_nm_per_px": _LAB_MITO_NM_PER_PX, "input_nm_per_px": nm,
        "resample_factor": nm / _LAB_MITO_NM_PER_PX,
        "scales": info.get("scales"), "weights": info.get("weights"),
        "connectivity": 8, "min_area_um2": _LAB_MITO_MIN_AREA_UM2, "min_area_px_applied": floor,
        "rule": {"threshold": _LAB_MITO_THRESHOLD if prob6 is not None else None,
                 "opening_px_at_6nm": _LAB_MITO_OPENING_PX if prob6 is not None else 0, "min_confidence": _LAB_MITO_MIN_CONFIDENCE if prob6 is not None else 0.0,
                 "probability_available": prob6 is not None, "source": _LAB_MITO_RULE_SOURCE},
        "object_confidence": confidence,
        "command": [Path(cmd[0]).name, Path(cmd[1]).name, *cmd[2:]], "inference": inference,
        "scope": _LAB_MITO_SCOPE,
    }
    return labels.astype(np.uint16), detail


def _write_labels(labels: np.ndarray, path: Path) -> None:
    try:
        import tifffile

        tifffile.imwrite(path, labels, compression=None)
    except ImportError:
        from PIL import Image

        Image.fromarray(labels).save(path, format="TIFF")


def run_image_morphology(params: dict[str, Any], *, output_dir: Path, log: Callable[[str], None]) -> dict[str, Any]:
    """Segment + measure one image (see module docstring for the contract)."""
    from .measure import measure, summarize, write_csv

    image_path = Path(str(params.get("image") or ""))
    if not image_path.is_file():
        raise OrganelleInputError(code="image_job.image_missing", message="no image file was staged",
                                  details={"image": str(image_path)})
    engine = str(params.get("engine") or "classical")
    if engine not in ENGINES:
        raise OrganelleInputError(code="image_job.engine_unknown", message=f"unknown engine {engine!r}",
                                  details={"engine": engine, "known": list(ENGINES)})
    organelle = str(params.get("organelle") or "")
    if engine == "fusion3" and organelle not in FUSION3_ORGANELLES:
        raise OrganelleInputError(code="image_job.organelle_required",
                                  message="the fusion3 engine needs an organelle: chloroplast or mitochondria",
                                  details={"organelle": organelle or None, "known": list(FUSION3_ORGANELLES)})
    min_area = int(params.get("min_area") or 10)
    raw_pixel = params.get("pixel_size_um")
    pixel_size_um = float(raw_pixel) if raw_pixel not in (None, "", 0, 0.0) else None
    if pixel_size_um is not None and not (pixel_size_um > 0):
        raise OrganelleInputError(code="image_job.pixel_size_invalid", message="pixel size must be positive",
                                  details={"pixel_size_um": raw_pixel})
    if engine == "lab_mito" and pixel_size_um is None:
        raise OrganelleInputError(
            code="image_job.calibration_required",
            message="the lab mitochondria model only works at 6 nm/px: a calibrated pixel size is required",
            details={"engine": "lab_mito", "model_nm_per_px": _LAB_MITO_NM_PER_PX})

    output_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(UTC)
    log(f"Reading {image_path.name}")
    input_notes: dict[str, Any] = {}
    array = load_analysis_image(image_path, notes=input_notes)
    engine_detail: dict[str, Any] = {"engine": engine}
    object_confidence: dict[int, float] = {}
    if engine == "classical":
        log("Otsu threshold and connected components")
        labels, threshold = classical_instances(array, min_area=min_area)
        engine_detail.update(method="otsu_threshold+connected_components", connectivity=8,
                             threshold=threshold, foreground="above_threshold")
    elif engine == "fusion3":
        labels, fusion_detail = _fusion3_instances(image_path, output_dir / "fusion3", organelle=organelle,
                                                   min_area=min_area, log=log)
        engine_detail.update(fusion_detail)
    elif engine == "lab_mito":
        assert pixel_size_um is not None
        labels, lab_detail = _lab_mito_instances(
            image_path, output_dir / "lab_mito", shape=(int(array.shape[0]), int(array.shape[1])),
            pixel_size_um=pixel_size_um, min_area=min_area, log=log)
        object_confidence = lab_detail.pop("object_confidence", {})
        engine_detail.update(lab_detail)
    else:
        model_type = str(params.get("micro_sam_model_type") or "vit_l_lm")
        device = str(params.get("device") or "auto")
        log(f"micro-SAM {model_type} on {device}")
        labels = _micro_sam_instances(image_path, output_dir / "micro_sam", model_type=model_type, device=device)
        engine_detail.update(method="micro_sam_automatic_instance_segmentation", model_type=model_type,
                             device=device, micro_sam=_package_versions(("micro_sam", "torch")))
    if labels.shape != array.shape:
        raise OrganelleInputError(code="image_job.mask_dimension_mismatch",
                                  message="label map does not match the source pixel grid",
                                  details={"labels": list(labels.shape), "image": list(array.shape)})

    stromule_result: dict[str, Any] | None = None
    if params.get("stromules"):
        from .chloro_metrics import detect_stromules

        log("Chloroplast bodies and stromule candidates")
        labels, stromule_result = detect_stromules(array, labels, pixel_size_um=pixel_size_um)
    log(f"Measuring {int(labels.max())} objects")
    result = measure(labels, intensity_image=array, pixel_size_um=pixel_size_um, min_area=min_area,
                     instance_map=True, label_source="model", class_names=("background", "object"))
    if object_confidence:
        for row in result.get("per_object", []):
            row["confidence"] = object_confidence.get(int(row["label"]))
    csv_fields: tuple[str, ...] | None = None
    if stromule_result is not None:
        from .chloro_metrics import annotate_stromule_rows, finalize_stromules

        rows = result.get("per_object", [])
        # Objects below min_area are not measured; their candidates are dropped with them.
        stromule_result = finalize_stromules(stromule_result, {int(r["label"]) for r in rows},
                                             pixel_size_um=pixel_size_um)
        annotate_stromule_rows(rows, stromule_result["candidates"], pixel_size_um=pixel_size_um)
        csv_fields = (*inspect.signature(write_csv).parameters["fields"].default,
                      "stromules", "stromule_length", "stromule_length_um")
    labels_path = output_dir / "labels.tif"
    _write_labels(labels, labels_path)
    objects = {
        "image_width": int(array.shape[1]),
        "image_height": int(array.shape[0]),
        "backend": engine,
        "pixel_size_um": pixel_size_um,
        "objects": result.get("per_object", []),
        "counts": result.get("counts", {}),
        "summary": summarize(result),
        "network": result.get("network"),
    }
    if stromule_result is not None:
        objects["stromules"] = stromule_result
    objects_path = output_dir / "objects.json"
    objects_path.write_text(json.dumps(objects, sort_keys=True, default=float), encoding="utf-8")
    csv_path = write_csv(result, output_dir / "objects.csv", **({"fields": csv_fields} if csv_fields else {}))
    evidence = {
        "schema": "organelleverse.image-job-evidence.v1",
        "job": "image_morphology",
        "started_at": started.isoformat(),
        "completed_at": datetime.now(UTC).isoformat(),
        "host": _host_facts(),
        "input": {"name": image_path.name, "sha256": _sha256(image_path),
                  "width": int(array.shape[1]), "height": int(array.shape[0]), **input_notes},
        "parameters": {"engine": engine, "min_area": min_area, "pixel_size_um": pixel_size_um,
                       **({"organelle": organelle} if engine == "fusion3" else {}),
                       **({"stromules": True} if stromule_result is not None else {})},
        "engine": engine_detail,
        "outputs": {"labels_sha256": _sha256(labels_path), "objects_sha256": _sha256(objects_path),
                    "objects": int(labels.max())},
    }
    evidence_path = output_dir / "evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True, default=float), encoding="utf-8")
    count = len(objects["objects"])
    unit = f"{pixel_size_um} µm/px" if pixel_size_um else "pixel units"
    note = ""
    if stromule_result is not None:
        found = stromule_result["summary"]
        note = f" {found['stromules']} stromules on {found['with_stromules']} of {found['chloroplasts']} chloroplasts."
    return {
        "summary_text": f"{engine}: {count} objects measured on {array.shape[1]}×{array.shape[0]} px ({unit}).{note}",
        "output_files": {"labels": str(labels_path), "objects": str(objects_path),
                         "objects_csv": str(csv_path), "evidence": str(evidence_path)},
    }


def run_image_engine_probe(params: dict[str, Any], *, output_dir: Path, log: Callable[[str], None]) -> dict[str, Any]:
    """Report the engines this machine can run; never installs anything."""
    import importlib.util

    del params
    output_dir.mkdir(parents=True, exist_ok=True)
    log("Checking image analysis engines")
    classical_ready = all(importlib.util.find_spec(name) is not None for name in ("skimage", "scipy"))
    engines: dict[str, dict[str, Any]] = {
        "classical": {"available": classical_ready,
                      "reason": None if classical_ready else "scikit-image or scipy is not installed"},
    }
    micro = importlib.util.find_spec("micro_sam") is not None
    micro_runtime: str | None = "this Python" if micro else None
    if not micro:
        # The service may drive micro-SAM from a separate environment.
        from .install import check_backend

        runtime = check_backend("micro_sam", scan_envs=True)
        micro = bool(runtime.get("installed") and runtime.get("env"))
        micro_runtime = str(runtime.get("env")) if micro else None
    gpu: dict[str, Any] = {"cuda": False}
    if importlib.util.find_spec("torch") is not None:
        try:
            import torch

            gpu = {"cuda": bool(torch.cuda.is_available()),
                   "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
        except Exception as error:  # a broken torch install must not fail the probe
            gpu = {"cuda": False, "error": f"{type(error).__name__}: {error}"[:200]}
    engines["micro_sam"] = {"available": micro, "reason": None if micro else "micro-sam is not installed",
                            "runtime": micro_runtime, "gpu": gpu}
    fusion = fusion3_availability()
    engines["fusion3"] = {
        "available": fusion["available"],
        "reason": None if fusion["available"] else
        f"fusion3-v1 delivery is not installed at {fusion['root']} (missing: {', '.join(fusion['missing'])})",
        "root": fusion["root"], "organelles": list(FUSION3_ORGANELLES),
    }
    lab = lab_mito_availability()
    engines["lab_mito"] = {
        "available": lab["available"],
        "reason": None if lab["available"] else
        f"lab mitochondria model is not installed at {lab['root']} (missing: {', '.join(lab['missing'])})",
        "root": lab["root"],
    }
    report = {"schema": "organelleverse.image-engines.v1", "checked_at": datetime.now(UTC).isoformat(),
              "host": _host_facts() | {"packages": _package_versions(_PACKAGES + ("micro_sam", "torch"))},
              "engines": engines}
    path = output_dir / "engines.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    ready = [name for name, info in engines.items() if info["available"]]
    return {"summary_text": f"Engines available: {', '.join(ready) or 'none'}.",
            "output_files": {"engines": str(path)}}


IMAGE_JOB_RUNNERS: dict[str, Callable[..., dict[str, Any]]] = {
    "image_morphology": run_image_morphology,
    "image_engine_probe": run_image_engine_probe,
}
