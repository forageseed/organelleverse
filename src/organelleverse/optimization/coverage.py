"""Audited built-in optimization eligibility, independent of capability admission.

Eligibility never grants execution or invents an objective. Only an admitted,
identity-matched enabled v3 profile can replace a scientific eligibility row.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from importlib.resources import files

from organelleverse.capabilities.index import CapabilityIndex, CapabilityStatus

from .contracts_v3 import OptimizationProfileV3
from .profiles_v3 import identity_from_capability_entry


def builtin_classifications() -> dict[str, dict[str, str]]:
    return json.loads(files(__package__).joinpath("builtin_coverage.json").read_text())[
        "capabilities"
    ]


def builtin_profiles(index: CapabilityIndex) -> dict[str, OptimizationProfileV3]:
    """Project admitted native identities to honest, initially non-enabled profiles."""
    declarations = builtin_classifications()
    return {
        entry.capability_id: OptimizationProfileV3(
            target=identity_from_capability_entry(entry), **declarations[entry.capability_id]
        )
        for entry in index.entries
        if entry.status is CapabilityStatus.ADMITTED
        and entry.capability_id in declarations
        and all(origin.channel == "core" for origin in entry.origins)
    }


def coverage_matrix(
    index: CapabilityIndex, profiles: Mapping[str, OptimizationProfileV3] | None = None
) -> dict:
    """Read-only complete inventory, including dependencies that are not admitted."""
    declarations = builtin_classifications()
    supplied = profiles or {}
    rows = []
    for entry in sorted(index.entries, key=lambda item: item.capability_id):
        if not all(origin.channel == "core" for origin in entry.origins):
            continue
        classification = declarations.get(
            entry.capability_id,
            {"status": "eligible", "reason_code": "optimization.onboarding_review_required"},
        )
        profile = supplied.get(entry.capability_id)
        if profile is not None:
            if (
                entry.status is not CapabilityStatus.ADMITTED
                or profile.target != identity_from_capability_entry(entry)
            ):
                raise ValueError(
                    "Optimization profile does not match the currently admitted capability"
                )
            if classification["status"] == "not_applicable" and profile.status == "enabled":
                raise ValueError(
                    "Enabling an inapplicable capability requires an explicit coverage policy revision"
                )
            classification = {"status": profile.status, "reason_code": profile.reason_code}
        rows.append(
            {
                "capability_id": entry.capability_id,
                "suite": entry.capability_id.split(".")[0],
                "admission_status": entry.status.value,
                **classification,
                "contract": profile.contract.model_dump(mode="json")
                if profile and profile.contract
                else None,
            }
        )
    return {
        "schema_version": "organelleverse.optimization.coverage.v1",
        "total_capabilities": len(rows),
        "total_suites": len({row["suite"] for row in rows}),
        "counts": {
            status: sum(row["status"] == status for row in rows)
            for status in ("enabled", "eligible", "not_applicable")
        },
        "rows": rows,
    }
