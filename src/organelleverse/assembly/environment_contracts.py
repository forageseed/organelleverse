"""Immutable environment provider contracts for PMAT2 assembly.

Defines closed capability identity, provider resolution records,
and the diagnostic types used by the environment resolver.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Annotated, Any, Literal, cast

from pydantic import Field, model_validator

from organelleverse.assembly.profiles import AssemblyProfile
from organelleverse.compute.contracts import ResolvedComputeTarget
from organelleverse.operations.spec import StrictSpecModel

# ---------------------------------------------------------------------------
# Literal type aliases
# ---------------------------------------------------------------------------

EnvironmentSource = Literal["auto", "existing", "managed"]

BackendVersionSelector = Annotated[
    str,
    Field(
        pattern=r"^(?:tested|latest|(?:0|[1-9][0-9]*)(?:\.(?:0|[1-9][0-9]*)){2,3})$",
    ),
]

# valid discovery source values
_DiscoverySource = Literal["agent_hint", "registry", "active_conda", "path", "managed"]

_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _str_field(component: object, field: str) -> str:
    if isinstance(component, dict):
        d: dict[object, object] = cast(dict[object, object], component)
        val = d.get(field, "")
        return str(val) if val is not None else ""
    val = getattr(component, field, "")
    return str(val) if val is not None else ""


def _version_field(component: object) -> str | None:
    """Extract version field for digest: None stays None (encoded as JSON null)."""
    if isinstance(component, dict):
        d: dict[object, object] = cast(dict[object, object], component)
        val = d.get("version")
        if val is None:
            return None
        return str(val) if val != "" else None
    val = getattr(component, "version", None)
    if val is None:
        return None
    s = str(val)
    return s if s else None


# ---------------------------------------------------------------------------
# Hint / capability / identity contracts
# ---------------------------------------------------------------------------


class EnvironmentHint(StrictSpecModel):
    """Optional user-supplied hint for locating an environment provider."""

    executable: Path | None = None
    prefix: Path | None = None
    expected_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _reject_relative_paths(self) -> EnvironmentHint:
        if self.executable is not None and not self.executable.is_absolute():
            raise ValueError("environment hint executable must be an absolute path")
        if self.prefix is not None and not self.prefix.is_absolute():
            raise ValueError("environment hint prefix must be an absolute path")
        return self

    @model_validator(mode="after")
    def _reject_both_paths(self) -> EnvironmentHint:
        if self.executable is not None and self.prefix is not None:
            raise ValueError("environment hint must not provide both executable and prefix")
        return self


class CapabilityItem(StrictSpecModel):
    """One required component of a backend environment."""

    role: str
    kind: Literal["executable", "resource", "host_provider"]
    safe_names: tuple[str, ...]
    version_argv: tuple[str, ...] = ()
    required_version_text: str | None = None
    version_exit_codes: tuple[int, ...] = (0,)
    version_specifier: str | None = None
    trusted_sha256: tuple[str, ...] = ()
    required_profiles: tuple[AssemblyProfile, ...] = ()
    required_parameter: str | None = None
    host_scope: Literal["prefix", "path"] = "prefix"
    version_style: Literal["semver", "major_minor", "strict_four_part"] = "semver"

    @model_validator(mode="after")
    def _validate_safe_names_nonempty_unique(self) -> CapabilityItem:
        if len(self.safe_names) == 0:
            raise ValueError("safe_names must be non-empty")
        if len(set(self.safe_names)) != len(self.safe_names):
            raise ValueError("safe_names must be unique")
        return self

    @model_validator(mode="after")
    def _validate_sha256_format(self) -> CapabilityItem:
        for sha in self.trusted_sha256:
            if not _SHA256_HEX.fullmatch(sha):
                raise ValueError(f"trusted_sha256 entries must be 64 hex chars, got {sha!r}")
        return self

    @model_validator(mode="after")
    def _validate_host_scope(self) -> CapabilityItem:
        if self.host_scope == "path" and self.kind != "host_provider":
            raise ValueError("host_scope='path' is valid only for host providers")
        return self


class BackendCapabilityContract(StrictSpecModel):
    """Closed specification of what a backend needs from its environment."""

    schema_version: Literal["organelleverse.backend-capabilities.v1"]
    backend_id: str
    items: tuple[CapabilityItem, ...]

    @model_validator(mode="after")
    def _validate_unique_roles(self) -> BackendCapabilityContract:
        roles = [item.role for item in self.items]
        if len(set(roles)) != len(roles):
            raise ValueError("BackendCapabilityContract items must have unique roles")
        return self

    @model_validator(mode="after")
    def _validate_unique_safe_names_across_items(self) -> BackendCapabilityContract:
        seen: dict[str, str] = {}
        for item in self.items:
            for name in item.safe_names:
                if name in seen:
                    raise ValueError(
                        f"safe_name {name!r} appears in both "
                        f"role {seen[name]!r} and role {item.role!r}"
                    )
                seen[name] = item.role
        return self


class ProviderComponentIdentity(StrictSpecModel):
    """Verified identity of a single component within a resolved provider."""

    role: str
    kind: Literal["executable", "resource", "host_provider"]
    path: Path
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    version: str | None = None


# ---------------------------------------------------------------------------
# Resolution records
# ---------------------------------------------------------------------------


class ResolvedProvider(StrictSpecModel):
    """A provider whose identity follows capability and component bytes, not discovery path."""

    requested_source: EnvironmentSource
    discovery_source: _DiscoverySource
    carrier: Literal["conda"]
    platform: str
    prefix: Path | None
    capability_contract_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    provider_digest: str = Field(default="sha256:" + "0" * 64, pattern=r"^sha256:[0-9a-f]{64}$")
    components: tuple[ProviderComponentIdentity, ...]

    @model_validator(mode="before")
    @classmethod
    def _compute_provider_digest(cls, data: Any) -> Any:
        """Compute provider_digest from canonical payload.

        Does NOT mutate the caller's dict.  version=None is encoded as JSON null.
        """
        if not isinstance(data, dict):
            return data
        # Work on a shallow copy to never mutate the caller's dict
        d: dict[str, object] = dict(cast(dict[str, object], data))
        components_raw = d.get("components", ())
        sorted_components = sorted(
            [
                {
                    "role": _str_field(c, "role"),
                    "kind": _str_field(c, "kind"),
                    "sha256": _str_field(c, "sha256"),
                    "version": _version_field(c),
                }
                for c in cast(list[object], list(components_raw))  # type: ignore[union-attr]
            ],
            key=lambda item: (
                str(item["role"]),
                str(item["kind"]),
                str(item["sha256"]),
                "" if item.get("version") is None else str(item["version"]),
            ),
        )
        payload: dict[str, object] = {
            "capability_contract_digest": str(d.get("capability_contract_digest", "")),
            "carrier": str(d.get("carrier", "conda")),
            "platform": str(d.get("platform", "")),
            "components": sorted_components,
        }
        digest = "sha256:" + hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
        d["provider_digest"] = digest
        return d


class ManagedProviderPlan(StrictSpecModel):
    """Placeholder for a future managed-environment materialization plan."""

    backend_id: str
    carrier: Literal["conda"]
    platform: str


class ProviderRejection(StrictSpecModel):
    """Diagnostic record for a provider candidate that was rejected."""

    backend_id: str
    reason_code: str
    detail: str = ""


class EnvironmentResolution(StrictSpecModel):
    """Result of environment resolution: exactly one of selected_provider,
    managed_plan, or compute_target, plus ordered rejections.

    The ``compute_target`` branch (spec §7.3) selects a verified external target
    such as WSL; it is chosen only after explicit configuration and
    compatibility verification. The two existing local branches keep their
    names and values byte-for-byte compatible.
    """

    selected_provider: ResolvedProvider | None = None
    managed_plan: ManagedProviderPlan | None = None
    compute_target: ResolvedComputeTarget | None = None
    rejections: tuple[ProviderRejection, ...] = ()

    @model_validator(mode="after")
    def _require_exactly_one(self) -> EnvironmentResolution:
        set_count = sum(
            x is not None
            for x in (self.selected_provider, self.managed_plan, self.compute_target)
        )
        if set_count != 1:
            raise ValueError(
                "EnvironmentResolution must have exactly one of selected_provider, "
                "managed_plan, or compute_target"
            )
        return self


class ResolvedBackendVersion(StrictSpecModel):
    """Exact canonical version resolved by the routing layer.

    Only strict canonical numeric releases are accepted: a bare 3- or 4-component
    ``X.Y.Z`` / ``W.X.Y.Z`` with no ``v`` prefix, prerelease, build metadata,
    abbreviation, or leading zeros. ``"tested"``, ``"latest"``, and abbreviated
    versions are rejected; a backend may impose a stricter policy (e.g. PMAT
    accepts only stable ``2.x.y``) at the routing layer.
    """

    backend_id: str
    version: str = Field(pattern=r"^(?:0|[1-9][0-9]*)(?:\.(?:0|[1-9][0-9]*)){2,3}$")
    requested_selector: str = "tested"
    tag: str | None = None
    commit: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")
    release_metadata_uri: str | None = None
    source_uri: str | None = None
    source_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
