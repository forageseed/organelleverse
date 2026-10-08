"""Human correction of segmentation label maps as a first-class provenance object.

A segmentation always returns a mask, and a wrong mask looks exactly as
confident as a right one. Biologists fix errors interactively (micro-SAM's
napari annotator makes this easy), but downstream morphometrics could never
tell which numbers a human had touched. This module makes the correction
itself a contract-level, hash-addressed provenance record that travels with
the corrected label map.

The correction record captures: both endpoint hashes, the affected region
(bounding box + per-class pixel transition counts), the correction type, a
free-text reason, the operator, and a timestamp. The natural-language reason
is **evidence, not mechanism**: it records *why* the change was made and is
never used to perform or infer any segmentation. This module makes no LLM or
network calls.

Downstream contract: a corrected label map is measured with
``measure(..., label_source="human_corrected")`` so every morphometric row
carries its origin (:data:`LABEL_SOURCE` is the token to pass).
"""

from __future__ import annotations

import hashlib
from datetime import datetime
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

__all__ = ["CORRECTION_TYPES", "LABEL_SOURCE", "correct"]

OPERATION_ID = "morphology.correct"

#: The correction categories this contract distinguishes.
CORRECTION_TYPES = ("merge", "split", "relabel", "delete", "add")

#: The label-source token consumers pass to ``measure(..., label_source=...)``
#: for any label map that went through human correction.
LABEL_SOURCE = "human_corrected"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _failed(
    code: str,
    message: str,
    *,
    details: dict[str, Any],
    started_at: datetime,
    params_hash: str,
) -> OrganelleResult:
    return OrganelleResult(
        operation_id=OPERATION_ID,
        operation_version=MORPHOLOGY_OPERATION_VERSION,
        scope=MORPHOLOGY_SCOPE,
        status="failed",
        summary_text=f"Correction rejected: {message}",
        metrics={"recorded": False},
        flags=("morph_correction_failed",),
        provenance=make_provenance(
            OPERATION_ID,
            params_hash=params_hash,
            started_at=started_at,
            finished_at=utc_now(),
        ),
        errors=(ErrorDetail(code=code, message=message, details=details, retryable=False),),
    )


def correct(
    original_label_map: str | Path,
    corrected_label_map: str | Path,
    *,
    correction_type: str,
    reason: str,
    operator: str,
    corrected_at: datetime | None = None,
) -> OrganelleResult:
    """Record a human correction of a segmentation label map.

    Parameters
    ----------
    original_label_map
        The model-produced label map before correction.
    corrected_label_map
        The label map after the human edit. Attached to the result as a
        content-addressed artifact.
    correction_type
        One of :data:`CORRECTION_TYPES` (``merge`` / ``split`` / ``relabel`` /
        ``delete`` / ``add``).
    reason
        Free-text explanation of *why* the correction was made. Evidence,
        not mechanism: never parsed, never executed.
    operator
        Identifier of the human who made the correction.
    corrected_at
        When the correction happened; defaults to now (UTC).

    Returns
    -------
    OrganelleResult
        ``operation_id="morphology.correct"``, ``stage=transform`` semantics
        (label map in, corrected label map + correction record out). Status
        is ``"failed"`` — never a partial record — when inputs are missing,
        shapes differ, metadata is blank, the type is unknown, or the two
        maps are pixel-identical (a correction that changes nothing is not a
        correction).
    """
    import numpy as np

    from .measure import _load_image

    started_at = utc_now()
    original = Path(original_label_map)
    corrected = Path(corrected_label_map)
    params_hash = parameters_hash(
        {
            "original_label_map": str(original),
            "corrected_label_map": str(corrected),
            "correction_type": correction_type,
            "reason": reason,
            "operator": operator,
        }
    )

    if not original.is_file() or not corrected.is_file():
        missing = str(original if not original.is_file() else corrected)
        return _failed(
            "morphology.correct_input_missing",
            f"label map does not exist: {missing}",
            details={"missing": missing},
            started_at=started_at,
            params_hash=params_hash,
        )
    if correction_type not in CORRECTION_TYPES:
        return _failed(
            "morphology.correct_invalid_type",
            f"unknown correction type: {correction_type!r}",
            details={"correction_type": correction_type, "allowed": list(CORRECTION_TYPES)},
            started_at=started_at,
            params_hash=params_hash,
        )
    if not reason.strip() or not operator.strip():
        return _failed(
            "morphology.correct_missing_metadata",
            "correction requires a non-blank reason and operator",
            details={"reason_blank": not reason.strip(), "operator_blank": not operator.strip()},
            started_at=started_at,
            params_hash=params_hash,
        )

    before = _load_image(original)
    after = _load_image(corrected)
    if before.ndim == 3:
        before = before[..., 0]
    if after.ndim == 3:
        after = after[..., 0]
    if before.shape != after.shape:
        return _failed(
            "morphology.correct_shape_mismatch",
            f"label map shapes differ: {before.shape} vs {after.shape}",
            details={"original_shape": list(before.shape), "corrected_shape": list(after.shape)},
            started_at=started_at,
            params_hash=params_hash,
        )

    diff = before != after
    n_changed = int(diff.sum())
    if n_changed == 0:
        return _failed(
            "morphology.correct_no_op",
            "original and corrected label maps are pixel-identical",
            details={"original_label_map": str(original), "corrected_label_map": str(corrected)},
            started_at=started_at,
            params_hash=params_hash,
        )

    rows, cols = np.nonzero(diff)
    bbox = (int(rows.min()), int(cols.min()), int(rows.max()) + 1, int(cols.max()) + 1)
    transitions: dict[str, int] = {}
    changed_before = before[diff]
    changed_after = after[diff]
    for from_value, to_value in zip(changed_before.tolist(), changed_after.tolist(), strict=True):
        key = f"{from_value}->{to_value}"
        transitions[key] = transitions.get(key, 0) + 1

    original_sha256 = _sha256(original)
    corrected_artifacts = collect_artifacts(
        [corrected], kind="corrected_label_map", format="png", media_type="image/png"
    )
    corrected_sha256 = corrected_artifacts[0].sha256
    timestamp = corrected_at if corrected_at is not None else utc_now()

    record: dict[str, Any] = {
        "correction_type": correction_type,
        "reason": reason,
        "operator": operator,
        "corrected_at": timestamp.isoformat(),
        "original_label_map": str(original),
        "corrected_label_map": str(corrected),
        "original_sha256": original_sha256,
        "corrected_sha256": corrected_sha256,
        "region": {
            "bbox": bbox,
            "n_changed_pixels": n_changed,
            "n_pixels": int(before.size),
            "transitions": transitions,
        },
        "label_source": LABEL_SOURCE,
    }

    return OrganelleResult(
        operation_id=OPERATION_ID,
        operation_version=MORPHOLOGY_OPERATION_VERSION,
        scope=MORPHOLOGY_SCOPE,
        status="ok",
        summary_text=(
            f"Human correction recorded ({correction_type} by {operator}: "
            f"{n_changed} pixels across {len(transitions)} transition kinds)."
        ),
        metrics={"recorded": True, "correction": record},
        findings=(
            Finding(
                code="morphology.corrected",
                metric="n_changed_pixels",
                value=n_changed,
                unit="pixels",
                evidence_artifact_ids=tuple(item.object_id for item in corrected_artifacts),
            ),
        ),
        flags=("morph_corrected",),
        artifacts=corrected_artifacts,
        provenance=make_provenance(
            OPERATION_ID,
            params_hash=params_hash,
            started_at=started_at,
            finished_at=utc_now(),
        ),
    )
