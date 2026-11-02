"""Destination-free public entry points for morphology workflows."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Annotated
from annotated_types import Gt
from uuid import uuid4

from organelleverse._public_validation import validated_public

from ..core.result import OrganelleResult
from ..inputs import Images
from ..runtime import managed_run_path
from ..sequences import as_path
from .microsam import DEFAULT_EM_ORGANELLE_MODEL

Micrometers = Annotated[float, Gt(0)]
VoxelSize = tuple[Micrometers, Micrometers, Micrometers]


@validated_public('morphology.segment', python_only=('executor',))
def segment(
    images: str | Path | list[str | Path] | Images,
    *,
    executor: Callable[..., Any] | None = None,
    device: str = "auto",
    config: str | Path | None = None,
    checkpoint: str | Path | None = None,
    orgseg_root: str | Path | None = None,
    measure_objects: bool = True,
    intensity_image: str | Path | Images | None = None,
    pixel_size_um: Micrometers | None = None,
    voxel_size_um: VoxelSize | None = None,
    model_id: str | None = None,
    model_dir: str | Path | None = None,
    backend: str = "auto",
    is_3d_stack: bool = False,
    micro_sam_model_type: str = DEFAULT_EM_ORGANELLE_MODEL,
) -> OrganelleResult:
    """Segment microscopy images and measure their masks in a managed run.

    model_id selects a pinned built-in profile from ``morphology.list_models()``.
    model_dir is an offline weight bundle (or ORGANELLEVERSE_MODEL_DIR). This
    path executes the native model when executor is omitted; it never downloads
    weights, substitutes a model, or assigns semantic classes to instance IDs.
    Without model_id, the existing backend/executor plan interface is retained.
    executor remains a Python-only callable injection hook.
    pixel_size_um is positive micrometers per pixel for 2D measurements.
    voxel_size_um is positive (z,y,x) spacing for 3D volume/surface measurements.
    is_3d_stack explicitly declares a single-channel (Z,Y,X) volume. RGB is 2D.
    Generic Cellpose cell/object models require organelle-specific validation;
    micro-SAM EM profiles produce organelle instances without semantic classes.
    """
    images = as_path(images)
    intensity_image = as_path(intensity_image)
    from .orgseg import segment as _segment

    available = None
    model_record = None
    if model_id is not None:
        from .models import _profile, make_executor
        profile = _profile(model_id)
        if backend not in ("auto", profile["backend"]):
            raise ValueError("model_id and backend select different methods")
        if checkpoint is not None or config is not None:
            raise ValueError("model_id uses its pinned weight bundle; omit config/checkpoint")
        backend = profile["backend"]
        micro_sam_model_type = model_id
        model_record = {"id": model_id, "revision": profile["revision"], "mask_kind": profile["mask_kind"],
                        "expected_runtime": profile["runtime_version"], "execution": "injected" if executor is not None else "native"}
        if executor is None:
            executor = make_executor(model_id, model_dir, is_3d_stack=is_3d_stack, voxel_size_um=voxel_size_um)
            from importlib.metadata import version
            model_record["runtime_version"] = version(profile["package"])
        available = True
    elif model_dir is not None:
        raise ValueError("model_dir requires model_id")

    return _segment(
        images,
        executor=executor,
        output_dir=managed_run_path("morphology.segment", uuid4().hex),
        device=device,
        config=config,
        checkpoint=checkpoint,
        orgseg_root=orgseg_root,
        measure_objects=measure_objects,
        intensity_image=intensity_image,
        pixel_size_um=pixel_size_um,
        voxel_size_um=voxel_size_um,
        backend_available=available,
        model_record=model_record,
        backend=backend,
        is_3d_stack=is_3d_stack,
        micro_sam_model_type=micro_sam_model_type,
    )


@validated_public('morphology.train', python_only=('executor',))
def train(
    data_root: str | Path,
    *,
    executor: Callable[[dict[str, Any]], None] | None = None,
    config: str | Path | None = None,
    load_from: str | Path | None = None,
    max_iters: int = 40000,
    batch_size: int = 2,
    device: str = "auto",
    orgseg_root: str | Path | None = None,
) -> OrganelleResult:
    """Fine-tune OrgSegNet with checkpoints in a platform-managed run."""
    from .train import train as _train

    return _train(
        data_root,
        executor=executor,
        config=config,
        load_from=load_from,
        work_dir=managed_run_path("morphology.train", uuid4().hex),
        max_iters=max_iters,
        batch_size=batch_size,
        device=device,
        orgseg_root=orgseg_root,
    )
