"""Immutable annotation-QC policy and fixed plant mitochondrial gene profile.

The policy pins the exact expected and variable plant mitochondrial
protein-coding gene (PCG) name sets and records their content-addressed digest.
It carries no numeric pass/fail threshold. Variable PCGs never enter the
expected-profile denominator and never produce a review signal when absent.
Adding a profile or changing a decision rule requires a new policy version.
"""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import Field, field_validator, model_validator

from organelleverse.core.base import StrictFrozenModel

#: The 24 expected plant mitochondrial core PCGs. Missing any of these is a
#: biological review signal, never an assertion of genome incompleteness.
EXPECTED_PCG_NAMES: tuple[str, ...] = (
    "atp1", "atp4", "atp6", "atp8", "atp9", "ccmB", "ccmC", "ccmFC",
    "ccmFN", "cob", "cox1", "cox2", "cox3", "matR", "mttB", "nad1",
    "nad2", "nad3", "nad4", "nad4L", "nad5", "nad6", "nad7", "nad9",
)

#: The 18 variable plant mitochondrial PCGs. Observed or absent, they never
#: contribute to failure or to the expected-profile denominator.
VARIABLE_PCG_NAMES: tuple[str, ...] = (
    "rpl2", "rpl5", "rpl6", "rpl10", "rpl16", "rps1", "rps2", "rps3",
    "rps4", "rps7", "rps10", "rps11", "rps12", "rps13", "rps14",
    "rps19", "sdh3", "sdh4",
)

_PROFILE_ID = "organelleverse.plant-mito-pcg.v1"
_PROFILE_VERSION = "v1"
_PROFILE_SCOPE = "plant_mitochondrial_core_protein_coding_genes"


def _profile_digest(
    *,
    profile_id: str,
    profile_version: str,
    scope: str,
    expected_pcg_names: tuple[str, ...],
    variable_pcg_names: tuple[str, ...],
) -> str:
    payload = {
        "profile_id": profile_id,
        "profile_version": profile_version,
        "scope": scope,
        "expected_pcg_names": list(expected_pcg_names),
        "variable_pcg_names": list(variable_pcg_names),
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


class AnnotationQcPolicy(StrictFrozenModel[Literal["annotation_qc_policy"]]):
    """Frozen annotation-QC policy with the pinned reference gene profile."""

    kind: Literal["annotation_qc_policy"] = "annotation_qc_policy"
    policy_version: str
    profile_id: str
    profile_version: str
    scope: str
    expected_pcg_names: tuple[str, ...]
    variable_pcg_names: tuple[str, ...]
    profile_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("expected_pcg_names", "variable_pcg_names")
    @classmethod
    def _reject_empty_name_set(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("annotation-QC policy name sets must be non-empty")
        return value

    @model_validator(mode="after")
    def _validate_profile_digest(self) -> AnnotationQcPolicy:
        expected = set(name.casefold() for name in self.expected_pcg_names)
        variable = set(name.casefold() for name in self.variable_pcg_names)
        overlap = expected & variable
        if overlap:
            raise ValueError(
                "expected and variable PCG name sets must not overlap: "
                f"{sorted(overlap)!r}"
            )
        digest = _profile_digest(
            profile_id=self.profile_id,
            profile_version=self.profile_version,
            scope=self.scope,
            expected_pcg_names=self.expected_pcg_names,
            variable_pcg_names=self.variable_pcg_names,
        )
        if digest != self.profile_sha256:
            raise ValueError("profile_sha256 does not match the pinned profile digest")
        return self

    @property
    def schema_version(self) -> str:
        return self.policy_version


ANNOTATION_QC_POLICY_V1 = AnnotationQcPolicy(
    policy_version="organelleverse.annotation-qc-policy.v1",
    profile_id=_PROFILE_ID,
    profile_version=_PROFILE_VERSION,
    scope=_PROFILE_SCOPE,
    expected_pcg_names=EXPECTED_PCG_NAMES,
    variable_pcg_names=VARIABLE_PCG_NAMES,
    profile_sha256=_profile_digest(
        profile_id=_PROFILE_ID,
        profile_version=_PROFILE_VERSION,
        scope=_PROFILE_SCOPE,
        expected_pcg_names=EXPECTED_PCG_NAMES,
        variable_pcg_names=VARIABLE_PCG_NAMES,
    ),
)


__all__ = [
    "ANNOTATION_QC_POLICY_V1",
    "EXPECTED_PCG_NAMES",
    "VARIABLE_PCG_NAMES",
    "AnnotationQcPolicy",
]
