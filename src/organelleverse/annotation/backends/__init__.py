"""Catalog of annotation backends that passed the released service boundary."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import AnnotationBackend, AnnotationRequest, AnnotationStage, BackendRun

if TYPE_CHECKING:
    from .mitochondrion import MitochondrionBackend
    from .plastome import PlastomeBackend

    RELEASED_BACKENDS: dict[str, AnnotationBackend]

__all__ = [
    "RELEASED_BACKENDS",
    "AnnotationBackend",
    "AnnotationRequest",
    "AnnotationStage",
    "BackendRun",
    "MitochondrionBackend",
    "PlastomeBackend",
]


def __getattr__(name: str) -> object:
    """Load executable backends only when execution requests them."""

    if name == "MitochondrionBackend":
        from .mitochondrion import MitochondrionBackend

        globals()[name] = MitochondrionBackend
        return MitochondrionBackend
    if name == "PlastomeBackend":
        from .plastome import PlastomeBackend

        globals()[name] = PlastomeBackend
        return PlastomeBackend
    if name == "RELEASED_BACKENDS":
        from .mitochondrion import MitochondrionBackend
        from .plastome import PlastomeBackend

        released: dict[str, AnnotationBackend] = {
            "mitochondrion": MitochondrionBackend(),
            # "plastid" is the canonical organelle token driving auto-selection;
            # "plastome" is the explicit backend name accepted by request.
            "plastid": PlastomeBackend(),
            "plastome": PlastomeBackend(),
        }
        globals()[name] = released
        return released
    raise AttributeError(name)
