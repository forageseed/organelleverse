"""Destination-free public entry points for morphology workflows."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..core.result import OrganelleResult
from ..runtime import managed_run_path
from .microsam import DEFAULT_EM_ORGANELLE_MODEL


def segment(
    images: str | Path | list[str | Path],
    *,
    executor: Callable[..., Any] | None = None,
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
    """Segment images with intermediates in a platform-managed run."""
    from .orgseg import segment as _segment

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
        backend=backend,
        is_3d_stack=is_3d_stack,
        micro_sam_model_type=micro_sam_model_type,
    )


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
