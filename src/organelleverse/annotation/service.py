"""Canonical annotation compute service for humans and Agents."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import replace
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic
from typing import cast
from uuid import uuid4

from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.errors import OrganelleError
from organelleverse.core.frozen import FrozenMap
from organelleverse.core.genome import OrganelleGenome
from organelleverse.core.provenance import ResultProvenance
from organelleverse.core.result import ErrorDetail, OrganelleResult
from organelleverse.runtime import publish_run

from .backends.base import AnnotationBackend, AnnotationRequest, BackendRun
from .genbank import extract_feature_records, parse_genbank
from .validation import AnnotationValidationReport, validate_document
from .writer import load_document_json, write_document_json


def _package_version() -> str:
    try:
        return version("organelleverse")
    except PackageNotFoundError:
        return "0.0.1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _request_payload(request: AnnotationRequest) -> dict[str, object]:
    payload = {
        "backend": request.backend,
        "threads": request.threads,
        "stages": list(request.stages),
    }
    if request.call_orfs:
        payload.update(
            call_orfs=True, orf_min_aa=request.orf_min_aa, orf_circular=request.orf_circular
        )
    return payload


def _error_result(
    genome: OrganelleGenome,
    *,
    code: str,
    message: str,
    details: Mapping[str, object] | None = None,
    suggested_action: Mapping[str, object] | None = None,
    retryable: bool = False,
) -> OrganelleResult:
    return OrganelleResult(
        operation_id="annotation.annotate",
        scope=genome.organelle,
        status="failed",
        summary_text=message,
        errors=(
            ErrorDetail(
                code=code,
                message=message,
                details=FrozenMap({} if details is None else details),
                suggested_action=FrozenMap({} if suggested_action is None else suggested_action),
                retryable=retryable,
            ),
        ),
    )


def _error_from_exception(genome: OrganelleGenome, error: OrganelleError) -> OrganelleResult:
    payload = error.as_dict()
    return _error_result(
        genome,
        code=error.code,
        message=error.message,
        details=cast(Mapping[str, object], payload["details"]),
        suggested_action=cast(Mapping[str, object], payload["suggested_action"]),
        retryable=error.retryable,
    )


def _validate_artifact(
    artifact: ArtifactRef,
    *,
    missing_code: str,
    changed_code: str,
) -> Path:
    path = artifact.resolve()
    if not path.is_file():
        raise ValueError(missing_code)
    if path.stat().st_size != artifact.size_bytes or _sha256_file(path) != artifact.sha256:
        raise ValueError(changed_code)
    return path


def _active_adapters(
    adapters: Mapping[str, AnnotationBackend] | None,
) -> Mapping[str, AnnotationBackend]:
    if adapters is not None:
        return adapters
    from .backends import RELEASED_BACKENDS

    return RELEASED_BACKENDS


def _select_backend(
    genome: OrganelleGenome,
    request: AnnotationRequest,
    adapters: Mapping[str, AnnotationBackend],
) -> tuple[str, AnnotationBackend] | None:
    selected_name = genome.organelle if request.backend == "auto" else request.backend
    adapter = adapters.get(selected_name)
    if adapter is None or genome.organelle not in adapter.organelle_types:
        return None
    return selected_name, adapter


def _semantic_run_payload(
    genome: OrganelleGenome,
    request: AnnotationRequest,
    backend_name: str,
    backend_run: BackendRun,
) -> dict[str, object]:
    return {
        "schema_version": "organelleverse.annotation-run.v1",
        "operation_id": "annotation.annotate",
        "input_object_id": genome.object_id,
        "input_artifact_hash": genome.sequence.sha256 if genome.sequence is not None else "",
        "parameters_hash": _sha256_json(_request_payload(request)),
        "backend": backend_name,
        "annotation_object_id": backend_run.document.object_id,
        "software_versions": dict(sorted(backend_run.software_versions.items())),
        "database_hashes": dict(sorted(backend_run.database_hashes.items())),
        "argv": [list(argv) for argv in _semantic_argv(backend_run)],
    }


def _semantic_argv(backend_run: BackendRun) -> tuple[tuple[str, ...], ...]:
    normalized: list[tuple[str, ...]] = []
    for command in backend_run.commands:
        arguments: list[str] = []
        for index, argument in enumerate(command.argv):
            path = Path(argument).expanduser()
            if index == 0:
                arguments.append(path.name)
            elif path.is_absolute():
                arguments.append(f"<path>/{path.name}")
            else:
                arguments.append(argument)
        normalized.append(tuple(arguments))
    return tuple(normalized)


def _existing_auxiliary_paths(run_dir: Path) -> tuple[Path, ...]:
    paths = [run_dir / "run_manifest.json", run_dir / "commands.jsonl"]
    logs_dir = run_dir / "logs"
    if logs_dir.is_dir():
        paths.extend(sorted(path for path in logs_dir.iterdir() if path.is_file()))
    return tuple(path for path in paths if path.is_file())


def _persist_backend_run(
    genome: OrganelleGenome,
    request: AnnotationRequest,
    backend_name: str,
    backend_run: BackendRun,
) -> tuple[Path, str, tuple[Path, ...]]:
    semantic = _semantic_run_payload(genome, request, backend_name, backend_run)
    semantic_hash = _sha256_json(semantic)
    run_manifest_id = f"annotation-run:sha256:{semantic_hash}"
    run_dir = request.workspace / f"sha256-{semantic_hash}"
    annotation_path = run_dir / "annotation.json"
    if run_dir.exists():
        if (
            not annotation_path.is_file()
            or load_document_json(annotation_path) != backend_run.document
        ):
            raise ValueError("existing content-addressed annotation run is inconsistent")
        return annotation_path, run_manifest_id, _existing_auxiliary_paths(run_dir)

    request.workspace.mkdir(parents=True, exist_ok=True)
    temporary = request.workspace / f".{run_dir.name}.tmp-{uuid4().hex}"
    try:
        temporary.mkdir()
        write_document_json(backend_run.document, temporary / "annotation.json")
        command_path = temporary / "commands.jsonl"
        command_path.write_text(
            "".join(
                json.dumps(
                    command.model_dump(mode="json"),
                    sort_keys=True,
                    allow_nan=False,
                    ensure_ascii=False,
                )
                + "\n"
                for command in backend_run.commands
            )
        )
        logs_dir = temporary / "logs"
        copied_logs: list[str] = []
        if backend_run.logs:
            logs_dir.mkdir()
        for index, source in enumerate(backend_run.logs, start=1):
            if not source.is_file():
                raise ValueError(f"backend log is missing: {source}")
            target_name = f"{index:03d}-{source.name}"
            shutil.copy2(source, logs_dir / target_name)
            copied_logs.append(target_name)
        manifest = {
            **semantic,
            "run_manifest_id": run_manifest_id,
            "commands_file": command_path.name,
            "logs": copied_logs,
        }
        (temporary / "run_manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, allow_nan=False, ensure_ascii=False, indent=2)
            + "\n"
        )
        if load_document_json(temporary / "annotation.json") != backend_run.document:
            raise ValueError("persisted canonical annotation did not round-trip")
        publish_run(temporary, run_dir)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return annotation_path, run_manifest_id, _existing_auxiliary_paths(run_dir)


def _validation_failure(
    genome: OrganelleGenome,
    report: AnnotationValidationReport,
) -> OrganelleResult:
    return _error_result(
        genome,
        code="annotation_validation_failed",
        message="backend annotation failed canonical validation",
        details={
            "issue_codes": [issue.code for issue in report.errors],
            "issues": [issue.model_dump(mode="json") for issue in report.errors],
        },
    )


def _execute_validate_and_persist(
    genome: OrganelleGenome,
    request: AnnotationRequest,
    backend_name: str,
    adapter: AnnotationBackend,
) -> (
    tuple[
        BackendRun,
        AnnotationValidationReport,
        Path,
        str,
        tuple[Path, ...],
    ]
    | OrganelleResult
):
    scratch_root = request.workspace / ".scratch"
    scratch_root.mkdir(exist_ok=True)
    with TemporaryDirectory(prefix="annotation-", dir=scratch_root) as temporary:
        try:
            backend_run = adapter.run(genome, request, Path(temporary))
        except OrganelleError as error:
            return _error_from_exception(genome, error)
        except Exception as error:
            return _error_result(
                genome,
                code="backend_execution_failed",
                message="annotation backend raised an unexpected exception",
                details={"backend": backend_name, "exception_type": type(error).__name__},
            )

        try:
            report = validate_document(backend_run.document, request.stages)
        except Exception as error:
            return _error_result(
                genome,
                code="backend_output_invalid",
                message="annotation backend output could not be validated",
                details={"backend": backend_name, "exception_type": type(error).__name__},
            )
        if not report.valid:
            return _validation_failure(genome, report)

        if request.call_orfs:
            from .orf_annotation import add_orf_features

            try:
                document = add_orf_features(
                    backend_run.document,
                    scratch=Path(temporary),
                    organelle=genome.organelle,
                    genetic_code=genome.metadata.genetic_code,
                    min_aa=request.orf_min_aa,
                    circular=request.orf_circular,
                )
                report = validate_document(document, request.stages)
                if not report.valid:
                    return _validation_failure(genome, report)
                backend_run = replace(
                    backend_run,
                    document=document,
                    software_versions={
                        **backend_run.software_versions,
                        "orfipy": version("orfipy"),
                    },
                )
            except OrganelleError as error:
                return _error_from_exception(genome, error)

        try:
            annotation_path, manifest_id, auxiliary_paths = _persist_backend_run(
                genome, request, backend_name, backend_run
            )
        except Exception as error:
            return _error_result(
                genome,
                code="backend_output_invalid",
                message="validated backend output could not be persisted",
                details={"backend": backend_name, "exception_type": type(error).__name__},
            )
        return backend_run, report, annotation_path, manifest_id, auxiliary_paths


def run_annotation(
    genome: OrganelleGenome,
    request: AnnotationRequest,
    *,
    adapters: Mapping[str, AnnotationBackend] | None = None,
) -> OrganelleResult:
    """Execute one released backend and persist only validated canonical output."""

    if genome.sequence is None:
        return _error_result(
            genome,
            code="input.missing_sequence_artifact",
            message="annotation requires a sequence artifact",
        )
    try:
        _validate_artifact(
            genome.sequence,
            missing_code="input.missing_sequence_artifact",
            changed_code="input.sequence_artifact_changed",
        )
    except ValueError as error:
        return _error_result(genome, code=str(error), message="sequence artifact is unavailable")

    selected = _select_backend(genome, request, _active_adapters(adapters))
    if selected is None:
        return _error_result(
            genome,
            code="unsupported_annotation_scope",
            message="no released annotation backend supports this request",
            details={
                "organelle": genome.organelle,
                "requested_backend": request.backend,
            },
            suggested_action={"choose_supported_backend": ["mitochondrion"]},
        )
    backend_name, adapter = selected

    request.workspace.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(UTC)
    started = monotonic()
    outcome = _execute_validate_and_persist(genome, request, backend_name, adapter)
    if isinstance(outcome, OrganelleResult):
        return outcome
    backend_run, report, annotation_path, manifest_id, auxiliary_paths = outcome

    artifacts = [
        ArtifactRef.from_path(
            annotation_path,
            kind="annotation",
            format="json",
            media_type="application/json",
        )
    ]
    for path in auxiliary_paths:
        if path.name == "run_manifest.json":
            artifacts.append(
                ArtifactRef.from_path(
                    path,
                    kind="annotation_manifest",
                    format="json",
                    media_type="application/json",
                )
            )
        elif path.name != "commands.jsonl":
            artifacts.append(ArtifactRef.from_path(path, kind="annotation_log", format="text"))

    finished_at = datetime.now(UTC)
    package_version = _package_version()
    provenance = ResultProvenance(
        operation_id="annotation.annotate",
        operation_version="1.0",
        package_version=package_version,
        git_commit=os.environ.get("ORG_VERSE_GIT_COMMIT", ""),
        input_object_ids=(genome.object_id,),
        input_artifact_hashes=(genome.sequence.sha256,),
        parameters_hash=_sha256_json(_request_payload(request)),
        requested_backend=backend_name,
        actual_backend=backend_name,
        attempted_backends=(backend_name,),
        software_versions=FrozenMap(
            {
                "organelleverse": package_version,
                **dict(backend_run.software_versions),
            }
        ),
        database_hashes=FrozenMap(dict(backend_run.database_hashes)),
        argv=tuple(argument for command in _semantic_argv(backend_run) for argument in command),
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=monotonic() - started,
        run_manifest_id=manifest_id,
    )
    feature_counts: dict[str, int] = {}
    for feature_type, count in report.feature_counts.items():
        if not isinstance(count, int) or isinstance(count, bool):
            return _error_result(
                genome,
                code="backend_output_invalid",
                message="annotation validation returned invalid feature counts",
            )
        feature_counts[feature_type] = count
    feature_count = sum(feature_counts.values())
    source_metadata = backend_run.document.source_metadata
    rejected_value = source_metadata.get("rejected_cds_candidates", ())
    rejected_candidates = rejected_value if isinstance(rejected_value, tuple) else ()
    missing_value = source_metadata.get("missing_core_genes", ())
    missing_core_genes = (
        tuple(item for item in missing_value if isinstance(item, str))
        if isinstance(missing_value, tuple)
        else ()
    )
    rejected_count = len(rejected_candidates)
    summary = f"Annotated {feature_count} features with {backend_name}."
    if rejected_count:
        noun = "candidate" if rejected_count == 1 else "candidates"
        summary += f" {rejected_count} invalid CDS {noun} rejected."
    if missing_core_genes:
        summary += f" Core genes not recovered: {', '.join(missing_core_genes)}."
    flags = ["annotation_validated", "annotation_persisted"]
    if rejected_count:
        flags.append("cds_candidates_rejected")
    if missing_core_genes:
        flags.append("missing_core_genes")
    orf_metrics = source_metadata.get("orf_candidates") if request.call_orfs else None
    if request.call_orfs:
        flags.append("sequence_only_orf_candidates")
    return OrganelleResult(
        operation_id="annotation.annotate",
        scope=genome.organelle,
        status="warning" if rejected_count or missing_core_genes else "ok",
        summary_text=summary,
        metrics=FrozenMap(
            {
                "backend": backend_name,
                "requested_stages": backend_run.document.requested_stages,
                "completed_stages": backend_run.document.completed_stages,
                "feature_count": feature_count,
                "feature_counts": feature_counts,
                "command_count": len(backend_run.commands),
                "rejected_cds_count": rejected_count,
                "rejected_cds_candidates": rejected_candidates,
                "missing_core_genes": missing_core_genes,
                **({"orf_candidates": orf_metrics} if request.call_orfs else {}),
            }
        ),
        flags=tuple(flags),
        artifacts=tuple(artifacts),
        provenance=provenance,
    )


def _safe_feature_key(feature_type: str) -> str:
    aliases = {"cds": "cds", "protein": "protein", "trna": "trna", "rrna": "rrna"}
    normalized = feature_type.casefold()
    return aliases.get(normalized, re.sub(r"[^a-z0-9]+", "_", normalized).strip("_"))


def run_extraction(
    genome: OrganelleGenome,
    workspace: str | Path,
    feature_types: tuple[str, ...],
) -> OrganelleResult:
    """Extract FASTA from canonical annotation without rerunning a backend."""

    if genome.annotation is None:
        return _error_result(
            genome,
            code="input.missing_annotation_artifact",
            message="annotation extraction requires an annotation artifact",
        ).evolve(operation_id="annotation.extract")
    try:
        source = _validate_artifact(
            genome.annotation,
            missing_code="input.missing_annotation_artifact",
            changed_code="input.annotation_artifact_changed",
        )
        if genome.annotation.format.casefold() == "json":
            document = load_document_json(source)
        elif genome.annotation.format.casefold() in {"genbank", "gb", "gbk"}:
            document = parse_genbank(source)
        else:
            raise ValueError("input.unsupported_annotation_format")
    except ValueError as error:
        return _error_result(
            genome,
            code=str(error),
            message="annotation artifact cannot be read",
        ).evolve(operation_id="annotation.extract")
    except Exception as error:
        return _error_result(
            genome,
            code="backend_output_invalid",
            message="annotation artifact is malformed",
            details={"exception_type": type(error).__name__},
        ).evolve(operation_id="annotation.extract")

    report = validate_document(document)
    if not report.valid:
        return _validation_failure(genome, report).evolve(operation_id="annotation.extract")
    if not feature_types or len(set(feature_types)) != len(feature_types):
        return _error_result(
            genome,
            code="input.invalid_feature_types",
            message="feature_types must be non-empty and unique",
        ).evolve(operation_id="annotation.extract")

    output_root = Path(workspace)
    semantic_hash = _sha256_json(
        {"annotation_object_id": document.object_id, "feature_types": list(feature_types)}
    )
    output_dir = output_root / f"sha256-{semantic_hash}"
    temporary = output_root / f".{output_dir.name}.tmp-{uuid4().hex}"
    try:
        if not output_dir.exists():
            output_root.mkdir(parents=True, exist_ok=True)
            temporary.mkdir()
            for feature_type in feature_types:
                key = _safe_feature_key(feature_type)
                records = extract_feature_records(document, feature_type)
                lines = [line for name, sequence in records for line in (f">{name}", sequence)]
                (temporary / f"{key}.fasta").write_text("\n".join(lines) + ("\n" if lines else ""))
            publish_run(temporary, output_dir)
    except Exception as error:
        if temporary.exists():
            shutil.rmtree(temporary)
        return _error_result(
            genome,
            code="backend_output_invalid",
            message="annotation extraction could not be persisted",
            details={"exception_type": type(error).__name__},
        ).evolve(operation_id="annotation.extract")
    finally:
        with suppress(OSError):
            if temporary.exists():
                shutil.rmtree(temporary)

    artifacts = tuple(
        ArtifactRef.from_path(
            output_dir / f"{_safe_feature_key(feature_type)}.fasta",
            kind=f"annotation_{_safe_feature_key(feature_type)}",
            format="fasta",
            media_type="text/plain",
        )
        for feature_type in feature_types
    )
    return OrganelleResult(
        operation_id="annotation.extract",
        scope=genome.organelle,
        status="ok",
        summary_text=f"Extracted {len(feature_types)} annotation feature sets.",
        metrics=FrozenMap(
            {
                "feature_types": feature_types,
                "artifact_count": len(artifacts),
            }
        ),
        flags=("annotation_extracted",),
        artifacts=artifacts,
    )
