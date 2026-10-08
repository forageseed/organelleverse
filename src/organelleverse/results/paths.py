"""Canonical output-path grammar for the desktop state (spec §1).

Every durable result byte the desktop owns lives under ``history_root`` at an
address this module can build and validate. Content identity stays sha256
(R3) — an address is where bytes live, never who they are.
"""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath

_SEGMENT = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_RECORD_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_SUFFIX = re.compile(r"^[a-z0-9][a-z0-9_-]{0,15}$")
_HEX8 = re.compile(r"^[0-9a-f]{8}$")

# Producer-kind id prefixes (R2): one registry so validators and builders
# cannot drift apart.
PREFIXED_IDS: tuple[tuple[str, str], ...] = (
    ("revisions", "rev-"),
    ("images", "img-"),
    ("conversations", "conv-"),
)


class PathSpecError(ValueError):
    """A path or path component violates the output path specification."""


def validate_segment(segment: str) -> str:
    """One path segment per R2: charset, length, and no dot-names."""
    if not _SEGMENT.match(segment):
        raise PathSpecError(f"path segment violates the output path spec: {segment!r}")
    return segment


def validate_record_id(record_id: str) -> str:
    if not _RECORD_ID.match(record_id):
        raise PathSpecError(f"record id violates the output path spec: {record_id!r}")
    return record_id


def validate_image_id(image_id: str) -> str:
    if not image_id.startswith("img-") or not _RECORD_ID.match(image_id):
        raise PathSpecError(f"image artifact id must be img-prefixed: {image_id!r}")
    return image_id


def validate_hex8(value: str) -> str:
    if not _HEX8.match(value):
        raise PathSpecError(f"child entropy must be 8 lowercase hex chars: {value!r}")
    return value


def validate_suffix(suffix: str) -> str:
    if not _SUFFIX.match(suffix):
        raise PathSpecError(f"file suffix violates the output path spec: {suffix!r}")
    return suffix


# -- builders (one per producer kind; R1: each owns exactly its subtree) ----


def run_receipt(run_id: str) -> PurePosixPath:
    return PurePosixPath("runs") / f"{validate_record_id(run_id)}.json"


def run_artifact(run_id: str, name: str) -> PurePosixPath:
    """Canonical materialized run output address (spec S3 target)."""
    return PurePosixPath("runs") / validate_record_id(run_id) / "artifacts" / validate_segment(name)


def experiment_record(experiment_id: str) -> PurePosixPath:
    return PurePosixPath("experiments") / f"{validate_record_id(experiment_id)}.json"


def revision_record(revision_id: str) -> PurePosixPath:
    return PurePosixPath("revisions") / f"{validate_record_id(revision_id)}.json"


def image_artifact(image_id: str, suffix: str) -> PurePosixPath:
    return PurePosixPath("images") / f"{validate_image_id(image_id)}.{validate_suffix(suffix)}"


def image_child(image_id: str, hex8: str, suffix: str) -> PurePosixPath:
    return PurePosixPath("images") / f"{image_id}-{validate_hex8(hex8)}.{validate_suffix(suffix)}"


def notebook_record(record_id: str) -> PurePosixPath:
    return PurePosixPath("notebooks") / f"{validate_record_id(record_id)}.json"


def conversation_record(conversation_id: str) -> PurePosixPath:
    return PurePosixPath("conversations") / f"{validate_record_id(conversation_id)}.json"


_OWNED_ROOTS = frozenset(
    {
        "runs",
        "experiments",
        "revisions",
        "images",
        "notebooks",
        "conversations",
        "comments",
        "revision-proposals",
        "skills",
        "memory",
    }
)


def is_canonical(relative: str | PurePosixPath) -> bool:
    """Whether ``relative`` sits inside a producer-owned subtree at all.

    Segment-level validity is the builders' job (they construct, never parse);
    this answers the ownership question for lint and conformance checks.
    """
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts:
        return False
    return len(path.parts) >= 2 and path.parts[0] in _OWNED_ROOTS


def relative_address(path: str | Path, history_root: Path) -> str | None:
    """The R7 browser-facing address: relative when the bytes live under the
    state root, ``None`` for anything outside it (never a host path)."""
    try:
        candidate = Path(path)
        relative = candidate.resolve().relative_to(history_root.resolve())
    except (OSError, ValueError):
        return None
    posix = PurePosixPath(relative)
    if ".." in posix.parts:
        return None
    return posix.as_posix()


__all__ = [
    "PathSpecError",
    "conversation_record",
    "experiment_record",
    "image_artifact",
    "image_child",
    "is_canonical",
    "notebook_record",
    "relative_address",
    "revision_record",
    "run_artifact",
    "run_receipt",
    "validate_hex8",
    "validate_image_id",
    "validate_record_id",
    "validate_segment",
    "validate_suffix",
]
