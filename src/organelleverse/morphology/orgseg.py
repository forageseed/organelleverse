"""OrgSegNet (Plantorganelle Hunter) segmentation wrapper.

OrgSegNet is a PSPNet-R50-d8 with a custom ``OrgSeg_Head`` decoder that
performs semantic segmentation of plant electron-microscopy (TEM) images into
5 classes (background, Chloroplast, Mitochondria, Vacuole, Nucleus). It is the
model behind *Plantorganelle Hunter* (Feng et al., *Nat. Plants* 2023).

OrgSegNet is a fork of MMSegmentation v1.0.0 and requires an old stack
(Python 3.8, torch 1.13.1, mmcv 2.0.0rc4, mmseg 1.0.0) that is **not** pip-
installable into modern environments, and its trained checkpoint is released
under a DTU-style academic license (non-commercial, no redistribution). This
module therefore does NOT bundle any model weights or the OrgSegNet source --
it wraps an installed OrgSegNet via an executor hook and returns a plan when
the tool is absent, matching the population/pangenome/localization backend
pattern.

Two call modes
--------------
- **Python-API (recommended)**: ``executor`` is a callable taking
  ``(config, checkpoint, device, image_paths, out_dir)`` and returning a list
  of label-map paths (one per input image). The canonical inference code is::

      from mmseg.apis import init_model, inference_model

      model = init_model(config, checkpoint, device)
      result = inference_model(model, img_path)
      label_map = result.pred_sem_seg.data[0].cpu().numpy()  # int32, 0..4

- **CLI**: ``executor`` runs ``python tools/test.py <config> <checkpoint>
  --show-dir <out>`` (standard MMSeg); saved label PNGs are then collected.

References
----------
- Feng, X., Yu, Z., Fang, H. et al. (2023) *Plantorganelle Hunter.* Nat. Plants
  9:1705-1717. doi:10.1038/s41477-023-01527-5
- OrgSegNet repo: https://github.com/yzy0102/OrgSegNet
- Checkpoint: OrgSegNet_iter_Version1.pth, Zenodo DOI 10.5281/zenodo.8419877
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..core import ErrorDetail, Finding, OrganelleResult
from ._contract import (
    MORPHOLOGY_OPERATION_VERSION,
    MORPHOLOGY_SCOPE,
    collect_artifacts,
    make_provenance,
    parameters_hash,
    utc_now,
)
from .install import ORGSEG_CHECKPOINT_NAME
from .measure import measure as measure_objs
from .microsam import DEFAULT_EM_ORGANELLE_MODEL

__all__ = ["segment"]

OPERATION_ID = "morphology.segment"

#: Registered segmentation backends (``auto`` routes between them).
BACKENDS = ("orgseg", "micro_sam")

# Defaults for the released OrgSegNet checkpoint (Nat. Plants 2023).
DEFAULT_CONFIG = "configs/OrgSegNet/OrgSeg_PlantCell_768x512.py"
DEFAULT_CHECKPOINT = "OrgSegNet_iter_Version1.pth"
ZENODO_URL = "https://zenodo.org/records/8419877"
ORGSEGNET_REPO = "https://github.com/yzy0102/OrgSegNet"


def _route_backend(backend: str, *, is_3d_stack: bool) -> str | None:
    """Deterministically route ``backend="auto"`` on declared input metadata.

    The routing rule is a pure function of caller-declared metadata — never
    of which backend happens to be installed:

    - 3D stacks route to ``micro_sam`` (OrgSegNet is a 2D thin-section
      model; micro-SAM supports 3D natively).
    - 2D plant EM images route to ``orgseg`` (the plant-organelle
      specialist, Nat. Plants 2023).

    Returns ``None`` when ``backend`` names no registered backend.
    """
    if backend == "auto":
        return "micro_sam" if is_3d_stack else "orgseg"
    if backend in BACKENDS:
        return backend
    return None


def segment(
    images: str | Path | list[str | Path],
    *,
    executor: Callable[..., Any] | None = None,
    output_dir: str | Path | None = None,
    device: str = "auto",
    config: str | Path | None = None,
    checkpoint: str | Path | None = None,
    orgseg_root: str | Path | None = None,
    measure_objects: bool = True,
    intensity_image: str | Path | None = None,
    pixel_size_um: float | None = None,
    backend: str = "auto",
    is_3d_stack: bool = False,
    micro_sam_model_type: str = DEFAULT_EM_ORGANELLE_MODEL,
) -> OrganelleResult:
    """Segment plant EM images with OrgSegNet and (optionally) quantify.

    Parameters
    ----------
    images
        One image path or a list of paths (TIF/PNG/JPG).
    executor
        Callable that runs the selected backend's inference. Signature for
        ``backend="orgseg"``:
        ``executor(config, checkpoint, device, image_paths, out_dir) -> list[Path]``;
        for ``backend="micro_sam"``:
        ``executor(model_type, device, image_paths, out_dir) -> list[Path]``.
        Both return one label-map path per input image. When ``None``
        (default), the call returns a plan (``flags=("morph_planned",)``)
        instead of running.
    output_dir
        Directory for label maps / overlays.
    device
        Compute device. ``"auto"`` (default) resolves deterministically from
        detected hardware + the selected backend's declared support
        (:mod:`.device`); an explicit device (``"cuda:0"``, ``"cpu"``,
        ``"mps"``, ...) that is unavailable — or unsupported by the selected
        backend — fails closed with the feasible list. There is no silent
        fallback to CPU.
    config
        MMSeg config path. Defaults to the released
        ``configs/OrgSegNet/OrgSeg_PlantCell_768x512.py`` (resolved under
        ``orgseg_root`` if given).
    checkpoint
        ``.pth`` checkpoint path. Defaults to ``OrgSegNet_iter_Version1.pth``.
    orgseg_root
        Path to a checked-out OrgSegNet repo (must be `pip install -v -e .`-ed
        so the custom ``OrgSeg_Head`` / ``PlantCellDataset`` register with
        mmseg). Used to resolve default config/checkpoint paths.
    measure_objects
        When True (default) and segmentation ran, run :func:`.measure` on each
        produced label map to get per-object morphometrics + counts.
    intensity_image
        Optional source image for mean-intensity measurements (only used when
        ``measure_objects`` is True and a single image is segmented).
    pixel_size_um
        Physical pixel size (µm/pixel) for area conversion in measurements.
    backend
        ``"auto"`` (default) routes deterministically on declared input
        metadata (see ``is_3d_stack``); an explicit ``"orgseg"`` /
        ``"micro_sam"`` pins the backend. An explicitly pinned backend that
        is not installed fails the call — the wrapper never silently falls
        back to the other backend.
    is_3d_stack
        Declared input metadata: the images are a 3D stack. Routes
        ``backend="auto"`` to micro-SAM (OrgSegNet is 2D-only).
    micro_sam_model_type
        micro-SAM model series used when the micro-SAM backend runs
        (default ``vit_l_em_organelles``, the EM-organelle generalist).

    Returns
    -------
    OrganelleResult
        Canonical immutable result, ``operation_id="morphology.segment"``,
        ``scope="mixed"`` (OrgSegNet labels four organelle classes at once).
        Status is ``"ok"`` when segmentation ran, ``"warning"`` when only a plan
        could be produced (no executor / backend absent), and ``"failed"`` when
        the executor raised or produced nothing. On the success path the
        produced label maps are attached as content-addressed ``artifacts``.
    """
    from .install import check_backend, install_hint

    started_at = utc_now()

    # Normalize inputs.
    image_list = [Path(images)] if isinstance(images, (str, Path)) else [Path(p) for p in images]
    n_img = len(image_list)

    # Resolve the backend before anything runs. ``auto`` routes on declared
    # input metadata only; an unknown explicit name fails closed.
    selected = _route_backend(backend, is_3d_stack=is_3d_stack)
    if selected is None:
        return OrganelleResult(
            operation_id=OPERATION_ID,
            operation_version=MORPHOLOGY_OPERATION_VERSION,
            scope=MORPHOLOGY_SCOPE,
            status="failed",
            summary_text=f"Unknown morphology backend: {backend!r}.",
            metrics={"n_images": n_img, "ran": False, "backend": backend},
            flags=("morph_failed",),
            provenance=make_provenance(
                OPERATION_ID,
                params_hash=parameters_hash({"backend": backend}),
                started_at=started_at,
                finished_at=utc_now(),
            ),
            errors=(
                ErrorDetail(
                    code="morphology.unknown_backend",
                    message=f"Unknown morphology backend: {backend!r}",
                    details={"backend": backend, "registered": list(BACKENDS)},
                    retryable=False,
                ),
            ),
        )

    # Resolve the compute device against the selected backend's declared
    # support: deterministic for auto, fail-closed for explicit requests.
    from .device import resolve_device

    resolution = resolve_device(device, backend=selected)
    if not resolution.ok:
        return OrganelleResult(
            operation_id=OPERATION_ID,
            operation_version=MORPHOLOGY_OPERATION_VERSION,
            scope=MORPHOLOGY_SCOPE,
            status="failed",
            summary_text=f"Device {device!r} cannot be used: {resolution.error_message}",
            metrics={
                "n_images": n_img,
                "backend": selected,
                "ran": False,
                "requested_device": device,
            },
            flags=("morph_failed", f"backend_{selected}"),
            provenance=make_provenance(
                OPERATION_ID,
                params_hash=parameters_hash({"backend": backend, "device": device}),
                attempted_backends=(selected,),
                started_at=started_at,
                finished_at=utc_now(),
            ),
            errors=(
                ErrorDetail(
                    code=resolution.error_code or "morphology.device_unavailable",
                    message=resolution.error_message or "device resolution failed",
                    details={
                        "requested_device": device,
                        "backend": selected,
                        "feasible_devices": list(resolution.feasible),
                    },
                    retryable=False,
                ),
            ),
        )
    assert resolution.device is not None
    torch_device = resolution.device.torch_device
    device_record = resolution.device.provenance_record()

    cfg = Path(config) if config else Path(orgseg_root or ".") / DEFAULT_CONFIG
    ckpt = Path(checkpoint) if checkpoint else Path(orgseg_root or ".") / DEFAULT_CHECKPOINT
    outdir = Path(output_dir) if output_dir else image_list[0].parent / "morph_out"
    if executor is not None:
        outdir.mkdir(parents=True, exist_ok=True)

    # Probe for the selected backend.
    loc = check_backend(selected, scan_envs=True)
    backend_found = bool(loc.get("installed"))
    warning = "" if backend_found else "\n  ⚠ " + install_hint(selected)

    # An explicitly pinned backend that is missing fails closed: never plan,
    # never fall back to the other backend.
    if backend != "auto" and not backend_found and executor is not None:
        return OrganelleResult(
            operation_id=OPERATION_ID,
            operation_version=MORPHOLOGY_OPERATION_VERSION,
            scope=MORPHOLOGY_SCOPE,
            status="failed",
            summary_text=(
                f"Requested backend {selected!r} is not installed; refusing to "
                f"substitute another backend." + warning
            ),
            metrics={
                "n_images": n_img,
                "backend": selected,
                "ran": False,
                "backend_found": False,
            },
            flags=("morph_failed", f"backend_{selected}"),
            provenance=make_provenance(
                OPERATION_ID,
                params_hash=parameters_hash({"backend": backend}),
                attempted_backends=(selected,),
                started_at=started_at,
                finished_at=utc_now(),
            ),
            errors=(
                ErrorDetail(
                    code="morphology.backend_unavailable",
                    message=f"Requested backend {selected!r} is not installed.",
                    details={
                        "backend": selected,
                        "next_step": install_hint(selected),
                    },
                    retryable=False,
                ),
            ),
        )

    argv = ["python", "tools/test.py", str(cfg), str(ckpt), "--show-dir", str(outdir)]

    params_hash = parameters_hash(
        {
            "config": str(cfg),
            "checkpoint": str(ckpt),
            "device": torch_device,
            "requested_device": device,
            "orgseg_root": str(orgseg_root) if orgseg_root else None,
            "image_paths": [str(p) for p in image_list],
            "measure_objects": measure_objects,
            "intensity_image": str(intensity_image) if intensity_image else None,
            "pixel_size_um": pixel_size_um,
            "backend": backend,
            "is_3d_stack": is_3d_stack,
            "micro_sam_model_type": micro_sam_model_type,
        }
    )

    # ── Plan-only path: no executor, or tool not found ───────────────
    if executor is None or not backend_found:
        plan = {
            "backend": selected,
            "config": str(cfg),
            "checkpoint": str(ckpt),
            "device": torch_device,
            "requested_device": device,
            "orgseg_root": str(orgseg_root) if orgseg_root else None,
            "micro_sam_model_type": micro_sam_model_type if selected == "micro_sam" else None,
            "n_images": n_img,
            "argv": argv,
            "image_paths": [str(p) for p in image_list],
        }
        return OrganelleResult(
            operation_id=OPERATION_ID,
            operation_version=MORPHOLOGY_OPERATION_VERSION,
            scope=MORPHOLOGY_SCOPE,
            # Nothing was computed: a plan is not an "ok" segmentation.
            status="warning",
            summary_text=(
                f"{selected} segmentation planned for {n_img} image(s) "
                f"(not run: {'no executor' if executor is None else 'tool not installed'})."
                + warning
            ),
            metrics={
                "n_images": n_img,
                "backend": selected,
                "ran": False,
                "backend_found": backend_found,
                "plan": plan,
            },
            findings=(
                Finding(code="morphology.plan", metric="config", value=str(cfg)),
                Finding(code="morphology.plan", metric="checkpoint", value=str(ckpt)),
            ),
            flags=("morph_planned", f"backend_{selected}"),
            provenance=make_provenance(
                OPERATION_ID,
                params_hash=params_hash,
                argv=tuple(argv),
                attempted_backends=(selected,),
                started_at=started_at,
                finished_at=utc_now(),
            ),
        )

    # ── Run path: call the executor with the selected backend's convention ─
    # OrgSegNet's official checkpoint is license-restricted: the app never
    # downloads it. Missing or corrupt -> explicit failure with the download
    # guidance; NEVER a silent switch to micro-SAM (that would silently
    # degrade semantic segmentation to classless instance segmentation).
    if selected == "orgseg" and ckpt.name == ORGSEG_CHECKPOINT_NAME:
        from .install import verify_orgseg_checkpoint

        check = verify_orgseg_checkpoint(ckpt)
        if not check["ok"]:
            guidance = install_hint("orgseg")
            return OrganelleResult(
                operation_id=OPERATION_ID,
                operation_version=MORPHOLOGY_OPERATION_VERSION,
                scope=MORPHOLOGY_SCOPE,
                status="failed",
                summary_text=(
                    f"OrgSegNet checkpoint verification failed "
                    f"({check['reason']}): {ckpt}. Download it yourself from "
                    f"the official page (license acceptance is yours) and place "
                    f"it at this path." + warning
                ),
                metrics={
                    "n_images": n_img,
                    "backend": selected,
                    "ran": False,
                    "checkpoint_check": check["reason"],
                },
                flags=("morph_failed", f"backend_{selected}"),
                provenance=make_provenance(
                    OPERATION_ID,
                    params_hash=params_hash,
                    argv=tuple(argv),
                    attempted_backends=(selected,),
                    started_at=started_at,
                    finished_at=utc_now(),
                ),
                errors=(
                    ErrorDetail(
                        code=f"morphology.weights_{check['reason']}",
                        message=(f"OrgSegNet checkpoint {check['reason']}: {ckpt}"),
                        details={
                            "checkpoint": str(ckpt),
                            "check": check["reason"],
                            "next_step": guidance,
                        },
                        retryable=False,
                    ),
                ),
            )

    label_paths: list[Path] = []
    run_err = ""
    try:
        if selected == "micro_sam":
            result = executor(
                micro_sam_model_type, torch_device, [str(p) for p in image_list], str(outdir)
            )
        else:
            result = executor(cfg, ckpt, torch_device, [str(p) for p in image_list], str(outdir))
        label_paths = [Path(p) for p in (result or [])]
    except Exception as exc:
        run_err = str(exc)

    if not label_paths or run_err:
        message = run_err or f"{selected} executor returned no label maps."
        return OrganelleResult(
            operation_id=OPERATION_ID,
            operation_version=MORPHOLOGY_OPERATION_VERSION,
            scope=MORPHOLOGY_SCOPE,
            status="failed",
            summary_text=f"{selected} run failed: {message}",
            metrics={
                "n_images": n_img,
                "backend": selected,
                "ran": False,
                "backend_found": backend_found,
                "error": run_err,
            },
            flags=("morph_failed", f"backend_{selected}"),
            provenance=make_provenance(
                OPERATION_ID,
                params_hash=params_hash,
                argv=tuple(argv),
                actual_backend=selected,
                attempted_backends=(selected,),
                started_at=started_at,
                finished_at=utc_now(),
            ),
            errors=(
                ErrorDetail(
                    code="morphology.segment_failed",
                    message=message,
                    details={
                        "backend": selected,
                        "config": str(cfg),
                        "checkpoint": str(ckpt),
                        "device": torch_device,
                        "requested_device": device,
                        "n_images": n_img,
                    },
                    retryable=False,
                ),
            ),
        )

    # ── Success: collect counts + optional morphometrics ─────────────
    per_image: list[dict[str, Any]] = []
    for i, lp in enumerate(label_paths):
        entry: dict[str, Any] = {
            "image": str(image_list[i]) if i < n_img else None,
            "label_map": str(lp),
        }
        if measure_objects:
            inten = intensity_image if (i == 0 and intensity_image) else None
            if selected == "micro_sam":
                # micro-SAM AMG yields instance ids without semantic classes.
                m = measure_objs(
                    lp,
                    intensity_image=inten,
                    pixel_size_um=pixel_size_um,
                    class_names=("background", "object"),
                    instance_map=True,
                    label_source="model",
                )
            else:
                m = measure_objs(
                    lp,
                    intensity_image=inten,
                    pixel_size_um=pixel_size_um,
                    label_source="model",
                )
            entry["counts"] = m["counts"]
            entry["n_objects"] = m["n_total"]
            entry["per_object"] = m["per_object"]
            # envelope-level spatial summary (area fraction, NND, ...) rides
            # along so callers can persist it with the objects table
            for key in (
                "image_area_px",
                "area_fraction_pct",
                "number_density_per_mm2",
                "nearest_neighbor_px",
                "background_mean_intensity",
            ):
                if key in m:
                    entry[key] = m[key]
        per_image.append(entry)

    total_counts: dict[str, int] = {}
    for e in per_image:
        for k, v in e.get("counts", {}).items():
            total_counts[k] = total_counts.get(k, 0) + v

    label_artifacts = collect_artifacts(
        label_paths,
        kind="organelle_label_map",
        format="png",
        media_type="image/png",
    )
    findings = tuple(
        Finding(
            code="morphology.object_count",
            metric=name,
            value=int(total),
            unit="objects",
            evidence_artifact_ids=tuple(item.object_id for item in label_artifacts),
        )
        for name, total in sorted(total_counts.items())
        if total > 0
    ) or (Finding(code="morphology.object_count", metric="none", value=0, unit="objects"),)

    return OrganelleResult(
        operation_id=OPERATION_ID,
        operation_version=MORPHOLOGY_OPERATION_VERSION,
        scope=MORPHOLOGY_SCOPE,
        status="ok",
        summary_text=(
            f"{selected} segmented {n_img} image(s); objects found: {total_counts or 'none'}."
        ),
        metrics={
            "n_images": n_img,
            "backend": selected,
            "ran": True,
            "backend_found": backend_found,
            "device": torch_device,
            "requested_device": device,
            "per_image": per_image,
            "total_counts": total_counts,
        },
        findings=findings,
        flags=("morph_ran", f"backend_{selected}"),
        artifacts=label_artifacts,
        provenance=make_provenance(
            OPERATION_ID,
            params_hash=params_hash,
            argv=tuple(argv),
            actual_backend=selected,
            attempted_backends=(selected,),
            software_versions={
                **({selected: loc["version"]} if loc.get("version") else {}),
                "device": device_record,
            },
            started_at=started_at,
            finished_at=utc_now(),
        ),
    )


def default_executor(
    config: str | Path,
    checkpoint: str | Path,
    device: str,
    image_paths: list[str],
    out_dir: str | Path,
) -> list[Path]:
    """Default OrgSegNet inference executor using the real mmseg API.

    Implements the canonical inference flow from OrgSegNet's
    ``demo/inference_demo.ipynb``::

        from mmseg.apis import init_model, inference_model

        model = init_model(config, checkpoint, device)
        result = inference_model(model, img)
        label_map = result.pred_sem_seg.data[0].cpu().numpy()  # int32, 0..4

    This must run inside the ``orgseg`` conda env (with OrgSegNet pip-installed
    so the custom ``OrgSeg_Head`` registers). Saves each label map as a PNG and
    returns the list of paths.
    """
    import numpy as np  # local import: only needed when actually running

    from ._mmcv_compat import install_mmcv_ops_shim
    install_mmcv_ops_shim()  # compiled mmcv ops absent on this stack
    # Register OrgSegNet's custom head from the vendored checkout; its
    # relative imports are rewritten to the installed mmseg package (the
    # repo is not pip-installed on this stack).
    import types as _types

    import organelleverse as _ov

    _head = (
        Path(_ov.__file__).parents[2]
        / "external_tools" / "OrgSegNet"
        / "mmseg" / "models" / "decode_heads" / "orgseg_head.py"
    )
    if _head.is_file():
        _s = _head.read_text()
        for _a, _b in (
            ("from ..builder import HEADS", "from mmseg.models.builder import HEADS"),
            ("from ..utils import resize", "from mmseg.models.utils import resize"),
            ("from .decode_head import BaseDecodeHead",
             "from mmseg.models.decode_heads.decode_head import BaseDecodeHead"),
        ):
            _s = _s.replace(_a, _b)
        exec(compile(_s, "orgseg_head.py", "exec"), _types.ModuleType("orgseg_head").__dict__)
    from mmseg.apis import inference_model, init_model  # type: ignore
    from PIL import Image

    model = init_model(str(config), str(checkpoint), device)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    label_paths: list[Path] = []
    for img_path in image_paths:
        result = inference_model(model, img_path)
        label_map = result.pred_sem_seg.data[0].cpu().numpy().astype(np.uint8)
        p = out / (Path(img_path).stem + "_label.png")
        Image.fromarray(label_map).save(str(p))
        label_paths.append(p)
    return label_paths
