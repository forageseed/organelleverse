"""Stable identity for one knowledge asset (L2).

An asset is a reference library, profile set, covariance model, trained model,
or database that a scientific method depends on but does not compute. The
locator is the stable name; the content hash in :mod:`.manifest` is the identity
that admission decisions are made against.
"""

from __future__ import annotations

import re
from typing import Self

from pydantic import BaseModel, ConfigDict, Field

from organelleverse.core.errors import OrganelleInputError

__all__ = ["ASSET_LOCATOR_PATTERN", "AssetLocator"]

ASSET_LOCATOR_PATTERN = (
    r"^ov-asset:"
    r"(?P<namespace>[a-z][a-z0-9_]*)"
    r"/(?P<name>[a-z][a-z0-9_]*)"
    r"@(?P<version>[A-Za-z0-9][A-Za-z0-9._-]*)$"
)

_LOCATOR_RE = re.compile(ASSET_LOCATOR_PATTERN)


class AssetLocator(BaseModel):
    """One asset's stable name, independent of where it is stored.

    The locator never encodes a filesystem path. Resolution from locator to a
    concrete directory is the resolver's job, so the same declared dependency
    works for a packaged asset, a user-supplied one, and a downloaded one.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    namespace: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    version: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

    @classmethod
    def parse(cls, text: str) -> Self:
        """Parse ``ov-asset:<namespace>/<name>@<version>``.

        Arguments:
            text: The locator string to parse.

        Returns:
            The parsed locator.

        Raises:
            OrganelleInputError: The string is not a well-formed asset locator.
        """
        match = _LOCATOR_RE.match(text)
        if match is None:
            raise OrganelleInputError(
                code="asset.locator_invalid",
                message=f"Not a well-formed asset locator: {text!r}",
                details={"locator": text, "expected_pattern": ASSET_LOCATOR_PATTERN},
            )
        return cls(**match.groupdict())

    def __str__(self) -> str:
        return f"ov-asset:{self.namespace}/{self.name}@{self.version}"
