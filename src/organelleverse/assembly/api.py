"""Canonical released assembly API."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from annotated_types import Ge, Le

from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.data import OrganelleData
from organelleverse.core.errors import OrganelleInputError
from organelleverse.core.result import OrganelleResult
from organelleverse.operations.parameters import OperationParameterModel

from .contracts import AssemblyMethod, AssemblyRequest, ReleasedAssemblyBackendParameters
from .data_contract import validate_released_assembly_data
from .environment_contracts import EnvironmentHint, EnvironmentSource

__all__ = ["assemble", "pmat_continue", "pmat_graph_build", "write"]


class AssemblyEnvironmentHint(OperationParameterModel):
    """Closed environment hint for Agent-facing assembly operations."""

    executable: Path | None = None
    prefix: Path | None = None
    expected_sha256: str | None = None


PmatGraphEnvironmentHint = AssemblyEnvironmentHint


def assemble(
    data: OrganelleData,
    *,
    organelle: Literal["mitochondrion", "plastid"],
    method: AssemblyMethod | Literal["auto"] = "auto",
    backend_parameters: ReleasedAssemblyBackendParameters | None = None,
    threads: Annotated[int, Ge(1), Le(256)] = 4,
    memory_gb: Annotated[int, Ge(1)] | None = None,
    timeout_seconds: Annotated[int, Ge(1)] | None = None,
    taxon_group: Literal["plant", "animal", "fungi"] = "plant",
    environment_source: EnvironmentSource = "auto",
    backend_version: str = "tested",
    environment_hint: AssemblyEnvironmentHint | None = None,
) -> OrganelleResult:
    """Assemble one organelle from a released assembly backend profile."""
    validate_released_assembly_data(data)
    request = AssemblyRequest(
        data=data,
        organelle=organelle,
        method=method,
        backend_parameters=backend_parameters,
        threads=threads,
        memory_gb=memory_gb,
        timeout_seconds=timeout_seconds,
        taxon_group=taxon_group,
        environment_source=environment_source,
        backend_version=backend_version,
        environment_hint=(
            None
            if environment_hint is None
            else EnvironmentHint.model_validate(environment_hint.model_dump())
        ),
    )
    from .service import execute_assembly

    return execute_assembly(request)


def write(result: OrganelleResult, *, output: Path) -> OrganelleResult:
    """Promote a managed assembly Result to a user-selected destination."""

    from .writer import materialize_result

    return materialize_result(result, output)


def pmat_continue(
    result: OrganelleResult,
    *,
    organelle: Literal["mitochondrion", "plastid"] | None = None,
    taxon_group: Literal["plant", "animal", "fungi"] = "plant",
    depth: Annotated[float, Ge(0)] | None = None,
    seeds: tuple[int, ...] | None = None,
    threads: Annotated[int, Ge(1), Le(256)] = 8,
    timeout_seconds: Annotated[int, Ge(1)] | None = None,
    environment_source: EnvironmentSource = "auto",
    backend_version: str = "tested",
    environment_hint: PmatGraphEnvironmentHint | None = None,
) -> OrganelleResult:
    """Continue a successful PMAT autoMito ``assemble`` into ``PMAT graphBuild``.

    Stage one (``assemble(method="pmat")``) normalizes its continuation
    outputs into the run directory; this bridge rebinds exactly those
    artifacts into the closed ``pmat_graph_input`` data contract, so callers
    never assemble the five-role payload by hand.
    """
    if result.status != "ok":
        raise OrganelleInputError(
            code="assembly.invalid_continuation",
            message="only a successful PMAT autoMito result can continue into graphBuild",
            details={"status": result.status},
        )
    resolved_organelle = organelle or _scope_organelle(result)
    by_name = {Path(artifact.uri).name: artifact for artifact in result.artifacts}
    roles: dict[str, str] = {
        "subsample_manifest": "pmat_subsample.manifest.json",
        "assembly_result_manifest": "pmat_assembly_result.manifest.json",
        "pmat_subsample": "pmat.subsample.fasta",
        "pmat_all_contigs": "pmat.all_contigs.fasta",
        "pmat_contig_graph": "pmat.contig_graph.txt",
    }
    missing = sorted(role for role, name in roles.items() if name not in by_name)
    if missing:
        raise OrganelleInputError(
            code="assembly.invalid_continuation",
            message=(
                "result carries no PMAT autoMito continuation artifacts "
                "(was assemble run with method='pmat'?)"
            ),
            details={"missing_roles": missing},
        )
    data = OrganelleData.model_validate(
        {
            "modality": "pmat_graph_input",
            "artifacts": {role: by_name[name] for role, name in roles.items()},
            "payload": {
                "contract_version": "organelleverse.pmat-graph-input.v1",
                "subsample_manifest_artifact": "subsample_manifest",
                "assembly_result_manifest_artifact": "assembly_result_manifest",
                "file_artifact_roles": [
                    "pmat_subsample",
                    "pmat_all_contigs",
                    "pmat_contig_graph",
                ],
            },
        }
    )
    return pmat_graph_build(
        data,
        organelle=resolved_organelle,
        taxon_group=taxon_group,
        depth=depth,
        seeds=seeds,
        threads=threads,
        timeout_seconds=timeout_seconds,
        environment_source=environment_source,
        backend_version=backend_version,
        environment_hint=environment_hint,
    )


def _scope_organelle(result: OrganelleResult) -> Literal["mitochondrion", "plastid"]:
    scope = str(result.scope)
    if scope in ("mitochondrion", "plastid"):
        return scope  # type: ignore[return-value]
    raise OrganelleInputError(
        code="assembly.invalid_continuation",
        message="cannot infer the organelle from the assemble result; pass organelle explicitly",
        details={"scope": scope},
    )


def pmat_graph_build(
    data: OrganelleData,
    *,
    organelle: Literal["mitochondrion", "plastid"],
    taxon_group: Literal["plant", "animal", "fungi"] = "plant",
    depth: Annotated[float, Ge(0)] | None = None,
    seeds: tuple[int, ...] | None = None,
    threads: Annotated[int, Ge(1), Le(256)] = 8,
    timeout_seconds: Annotated[int, Ge(1)] | None = None,
    environment_source: EnvironmentSource = "auto",
    backend_version: str = "tested",
    environment_hint: PmatGraphEnvironmentHint | None = None,
) -> OrganelleResult:
    """Run PMAT2 graphBuild from verified autoMito continuation artifacts."""
    from .pmat_graph import execute_pmat_graph_build

    return execute_pmat_graph_build(
        data,
        organelle=organelle,
        taxon_group=taxon_group,
        depth=depth,
        seeds=seeds,
        threads=threads,
        timeout_seconds=timeout_seconds,
        environment_source=environment_source,
        backend_version=backend_version,
        environment_hint=(
            EnvironmentHint.model_validate(environment_hint.model_dump(mode="python"))
            if environment_hint is not None
            else None
        ),
    )
