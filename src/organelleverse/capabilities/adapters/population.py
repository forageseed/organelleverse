"""Adapter decisions for restored population capabilities.

The three public tool workflows use platform-managed output locations. Agent calls provide input paths and receive plans; output directories and prefixes are not Agent parameters. Python calls with an executor may create managed intermediate files. Other bindings below retain their established codecs.
"""

from __future__ import annotations

from dataclasses import dataclass

from organelleverse.operations.spec import ParameterCodec, ResultCodec, SideEffect


@dataclass(frozen=True)
class ParameterOverride:
    """One named, Agent-facing parameter the generator could not derive alone."""

    name: str
    codec: ParameterCodec
    path_role: str | None = None


@dataclass(frozen=True)
class NamedParameterOverride:
    """The full ``named_parameters`` binding plan for one restored capability."""

    parameters: tuple[ParameterOverride, ...]
    result_codec: ResultCodec
    result_key: str | None
    side_effects: tuple[SideEffect, ...]


OVERRIDES: dict[str, NamedParameterOverride] = {
    "population.call_variants": NamedParameterOverride(
        parameters=(
            ParameterOverride(name="bam_dir", codec=ParameterCodec.DIRECTORY, path_role="input"),
        ),
        result_codec=ResultCodec.CANONICAL,
        result_key=None,
        side_effects=(SideEffect.SUBPROCESS,),
    ),
    "population.prepare_gemma_input": NamedParameterOverride(
        parameters=(
            ParameterOverride(name="vcf_path", codec=ParameterCodec.PATH),
            ParameterOverride(name="phenotype", codec=ParameterCodec.PATH),
        ),
        result_codec=ResultCodec.CANONICAL,
        result_key=None,
        side_effects=(SideEffect.SUBPROCESS, SideEffect.WRITE_FILES),
    ),
    "population.check_backend": NamedParameterOverride(
        parameters=(ParameterOverride(name="name", codec=ParameterCodec.JSON),),
        result_codec=ResultCodec.JSON_METRIC,
        result_key="backend_status",
        side_effects=(SideEffect.SUBPROCESS,),
    ),
    "population.check_all_backends": NamedParameterOverride(
        parameters=(),
        result_codec=ResultCodec.JSON_METRIC,
        result_key="backend_status",
        side_effects=(SideEffect.SUBPROCESS,),
    ),
    "population.install_hint": NamedParameterOverride(
        parameters=(ParameterOverride(name="name", codec=ParameterCodec.JSON),),
        result_codec=ResultCodec.JSON_METRIC,
        result_key="hint_text",
        side_effects=(),
    ),
    "population.compute_fst": NamedParameterOverride(
        parameters=(
            ParameterOverride(name="vcf_path", codec=ParameterCodec.PATH),
            ParameterOverride(name="pop_assignments", codec=ParameterCodec.JSON),
        ),
        result_codec=ResultCodec.JSON_METRIC,
        result_key="fst",
        side_effects=(SideEffect.READ_FILES,),
    ),
    "population.fst_scan": NamedParameterOverride(
        parameters=(
            ParameterOverride(name="vcf_path", codec=ParameterCodec.PATH),
            ParameterOverride(name="pop_assignments", codec=ParameterCodec.JSON),
        ),
        result_codec=ResultCodec.CANONICAL,
        result_key=None,
        side_effects=(SideEffect.READ_FILES,),
    ),
    "population.detect_numt": NamedParameterOverride(
        parameters=(
            ParameterOverride(name="nuclear_fasta", codec=ParameterCodec.PATH),
            ParameterOverride(name="mito_fasta", codec=ParameterCodec.PATH),
        ),
        result_codec=ResultCodec.CANONICAL,
        result_key=None,
        side_effects=(SideEffect.READ_FILES,),
    ),
    "population.cytonuclear_gwas": NamedParameterOverride(
        parameters=(
            ParameterOverride(name="vcf_path", codec=ParameterCodec.PATH),
            ParameterOverride(name="phenotype", codec=ParameterCodec.PATH),
        ),
        result_codec=ResultCodec.CANONICAL,
        result_key=None,
        side_effects=(SideEffect.WRITE_FILES,),
    ),
}
"""Every restored id in this domain resolves - none is deliberately absent."""

__all__ = ["OVERRIDES", "NamedParameterOverride", "ParameterOverride"]
