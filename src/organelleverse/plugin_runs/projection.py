"""Closed projections for result JSON crossing UI and Agent boundaries."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path, PureWindowsPath
from typing import cast
from urllib.parse import urlsplit

__all__ = ["browser_safe_diagnostic_details", "browser_safe_json_object"]

_OMITTED = object()
_PRIVATE_EXECUTION_FIELDS = frozenset(
    {"argv", "plugin_root", "staging_root", "worker_payload"}
)
_PRIVATE_URI_SCHEMES = frozenset({"file", "managed"})


def browser_safe_json_object(value: Mapping[str, object]) -> dict[str, object]:
    """Project scientific JSON while omitting private execution vocabulary."""

    projected = _project(value)
    assert isinstance(projected, dict)
    return cast(dict[str, object], projected)


def browser_safe_diagnostic_details(
    value: Mapping[str, object],
) -> dict[str, object]:
    """Expose stable diagnostic fields, never a worker's private raw reason."""

    return browser_safe_json_object(
        {str(name): item for name, item in value.items() if name != "reason"}
    )


def _project(value: object) -> object:
    if isinstance(value, str):
        if Path(value).is_absolute() or PureWindowsPath(value).is_absolute():
            return _OMITTED
        if urlsplit(value).scheme.casefold() in _PRIVATE_URI_SCHEMES:
            return _OMITTED
        return value
    if isinstance(value, Mapping):
        projected: dict[str, object] = {}
        for name, item in cast(Mapping[object, object], value).items():
            if str(name) in _PRIVATE_EXECUTION_FIELDS:
                continue
            safe_item = _project(item)
            if safe_item is not _OMITTED:
                projected[str(name)] = safe_item
        return projected
    if isinstance(value, (list, tuple)):
        return [
            safe_item
            for item in cast(list[object] | tuple[object, ...], value)
            if (safe_item := _project(item)) is not _OMITTED
        ]
    return value
