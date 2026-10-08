"""Fixed Slurm job-script renderer (spec §13, §14).

The job script is the **sole shell text** in the L5 subsystem, and it contains
**no model-authored bytes**. It is rendered from a closed set of inputs: a
static prologue, administrator-allowlisted environment/module lines, and one
``exec`` line whose two paths are provider-generated safe-ASCII derived from
prepared/run identities. Scientific parameters live only in the canonical
request JSON file the worker reads — never interpolated into shell source.

Changing scientific parameters changes only the request JSON (and its identity),
never the script bytes. Hash the exact script bytes before submission.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from organelleverse.core.errors import OrganelleContractError

from .config import SlurmResources, SlurmTargetConfig

__all__ = ["RenderedJobScript", "is_safe_provider_path", "render_job_script"]

# Each path component must match this tight alphabet so a provider-generated
# path can never break out of a shell token, form an option (leading "-"), or
# traverse (".." / empty component). Absolute paths are allowed because every
# component is independently safe.
_SAFE_COMPONENT = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$").fullmatch


def is_safe_provider_path(value: str) -> bool:
    """True iff ``value`` is a provider-generated path whose every component is
    shell-safe ASCII (no traversal, no leading option-dash, no empty segment)."""
    if not value or value.startswith("-"):
        return False
    body = value[1:] if value.startswith("/") else value
    if body == "":
        return value == "/"  # bare "/" is harmless but pointless; allow strictly
    return all(_SAFE_COMPONENT(component) is not None for component in body.split("/"))


@dataclass(frozen=True)
class RenderedJobScript:
    """The fixed job script plus the identities it was rendered from."""

    script: str
    request_file: str
    worker_executable: str
    script_sha256: str

    def script_bytes(self) -> bytes:
        return self.script.encode("utf-8")


def render_job_script(
    *,
    config: SlurmTargetConfig,
    resources: SlurmResources,
    worker_executable: str,
    request_file: str,
    allowlisted_modules: tuple[str, ...] = (),
) -> RenderedJobScript:
    """Render the fixed POSIX job script.

    ``worker_executable`` and ``request_file`` are provider-generated safe-ASCII
    paths; they are validated here so a caller bug can never interpolate a bad
    value into shell source. ``allowlisted_modules`` come from trusted owner
    config (the ``environment_id`` mapping), never from model content. Scientific
    parameters are NOT arguments to this function — they live only in the
    request file.
    """
    for name, value in (("worker_executable", worker_executable), ("request_file", request_file)):
        if not is_safe_provider_path(value):
            raise OrganelleContractError(
                code="compute.slurm_unsafe_path",
                message=(
                    f"the provider-generated {name} is not a safe-ASCII path; it cannot "
                    "be interpolated into the fixed job script"
                ),
                details={name: value},
            )

    import hashlib

    lines: list[str] = ["#!/bin/sh", "# OrganelleVerse fixed Slurm job script (provider-rendered)."]
    # static prologue: fail fast, no globbing surprises
    lines.append("set -eu")

    # administrator-allowlisted environment/module lines only
    for module in allowlisted_modules:
        if _SAFE_COMPONENT(module) is None:
            raise OrganelleContractError(
                code="compute.slurm_unsafe_module",
                message="an allowlisted module id is not a safe identifier",
                details={"module": module},
            )
        lines.append(f"# module: {module}")  # documented; real `module load` is owner-rendered

    # the sole exec line: fixed worker invocation reading the request file.
    # No model bytes, no scientific parameters, no environment expansion of input.
    lines.append(f"exec {worker_executable} run --request {request_file}")
    script = "\n".join(lines) + "\n"

    digest = hashlib.sha256(script.encode("utf-8")).hexdigest()
    return RenderedJobScript(
        script=script,
        request_file=request_file,
        worker_executable=worker_executable,
        script_sha256=digest,
    )
