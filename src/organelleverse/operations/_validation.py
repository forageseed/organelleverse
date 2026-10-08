"""Stable JSON-safe summaries for Pydantic validation failures."""

from collections.abc import Iterable

from pydantic import ValidationError


def validation_summaries(error: ValidationError) -> list[dict[str, object]]:
    """Convert all Pydantic errors to the operations error-envelope shape."""
    return [
        validation_summary(detail["loc"], str(detail["msg"]), str(detail["type"]))
        for detail in error.errors()
    ]


def validation_summary(loc: Iterable[object], message: str, type_: str) -> dict[str, object]:
    """Build one validation summary with stable keys and JSON-safe values."""
    return {
        "loc": [str(part) for part in loc],
        "message": message,
        "type": type_,
    }
