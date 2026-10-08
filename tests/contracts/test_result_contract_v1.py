from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.errors import OrganelleExecutionError
from organelleverse.core.frozen import FrozenMap
from organelleverse.core.provenance import ResultProvenance
from organelleverse.core.result import (
    ErrorDetail,
    Finding,
    OperationSuggestion,
    OrganelleResult,
)


def _provenance(**changes: object) -> ResultProvenance:
    values: dict[str, object] = {
        "operation_id": "localization.predict",
        "operation_version": "1.0",
        "package_version": "0.1.0",
        "git_commit": "abc1234",
        "input_object_ids": ("genome:sha256:abc",),
        "parameters_hash": "b" * 64,
        "requested_backend": "deeploc",
        "actual_backend": "heuristic",
        "attempted_backends": ("deeploc", "heuristic"),
    }
    values.update(changes)
    return ResultProvenance.model_validate(values)


def test_result_status_is_strict() -> None:
    with pytest.raises(ValidationError):
        OrganelleResult(
            operation_id="quality_control.assess",
            scope="mitochondrion",
            status="banana",  # type: ignore[arg-type]
        )


def test_failed_result_requires_structured_error() -> None:
    with pytest.raises(ValidationError, match="failed result requires at least one error"):
        OrganelleResult(
            operation_id="annotation.annotate",
            scope="mitochondrion",
            status="failed",
        )


def test_ok_result_cannot_contain_errors() -> None:
    with pytest.raises(ValidationError, match="ok result cannot contain errors"):
        OrganelleResult(
            operation_id="annotation.annotate",
            scope="mitochondrion",
            status="ok",
            errors=(ErrorDetail(code="execution.nonzero_exit", message="blastn failed"),),
        )


def test_result_summary_and_failure_raise_are_human_friendly() -> None:
    error = ErrorDetail(
        code="execution.nonzero_exit",
        message="blastn exited with code 2",
        retryable=True,
    )
    result = OrganelleResult(
        operation_id="annotation.annotate",
        scope="mitochondrion",
        status="failed",
        errors=(error,),
    )
    assert "annotation.annotate" in result.summary()
    with pytest.raises(OrganelleExecutionError, match="blastn exited with code 2") as raised:
        result.raise_for_failure()
    assert raised.value.code == error.code
    assert raised.value.retryable is True


def test_provenance_records_actual_backend_and_attempts() -> None:
    provenance = _provenance()

    assert provenance.actual_backend == "heuristic"
    assert provenance.attempted_backends == ("deeploc", "heuristic")


def test_empty_upstream_manifests_preserve_legacy_provenance_identity() -> None:
    provenance = _provenance()

    assert provenance.upstream_run_manifest_ids == ()
    assert provenance.object_id == (
        "provenance:sha256:124bf0086d8451cd86c9eb0fd0e638d2b7ca96556e4463b466feb8a17df514a8"
    )


def test_callable_locator_is_optional_and_absent_from_legacy_identity() -> None:
    provenance = _provenance()
    assert provenance.callable_locator is None
    # Omitted entirely from _identity_payload when unset - the hardcoded hash
    # in test_empty_upstream_manifests_preserve_legacy_provenance_identity
    # above must stay reachable by callers that never set this new field.
    assert provenance.object_id == (
        "provenance:sha256:124bf0086d8451cd86c9eb0fd0e638d2b7ca96556e4463b466feb8a17df514a8"
    )


def test_callable_locator_participates_in_provenance_identity_when_set() -> None:
    with_locator = _provenance(callable_locator="organelleverse.annotation.api:annotate")
    assert with_locator.callable_locator == "organelleverse.annotation.api:annotate"
    assert with_locator.object_id != _provenance().object_id

    other_locator = _provenance(callable_locator="organelleverse.annotation.api:extract")
    assert other_locator.object_id != with_locator.object_id


def test_callable_locator_must_match_the_python_locator_pattern() -> None:
    with pytest.raises(ValidationError, match="callable_locator"):
        _provenance(callable_locator="not a locator")


def test_upstream_manifest_ids_are_unique_stable_and_not_current() -> None:
    run_id = "assembly-run:sha256:" + "a" * 64
    assert _provenance(upstream_run_manifest_ids=(run_id,)).object_id != _provenance().object_id

    with pytest.raises(ValidationError, match="unique"):
        _provenance(upstream_run_manifest_ids=(run_id, run_id))
    with pytest.raises(ValidationError, match="stable content ID"):
        _provenance(upstream_run_manifest_ids=("run.json",))
    with pytest.raises(ValidationError, match="current run_manifest_id"):
        _provenance(run_manifest_id=run_id, upstream_run_manifest_ids=(run_id,))


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"requested_backend": "deeploc", "attempted_backends": ("heuristic",)}, "requested"),
        ({"actual_backend": "heuristic", "attempted_backends": ("heuristic", "deeploc")}, "final"),
        ({"attempted_backends": ("",)}, "non-empty"),
    ],
)
def test_provenance_rejects_inconsistent_backend_attempts(
    changes: dict[str, object], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        _provenance(**changes)


def test_provenance_allows_no_backend_or_attempts() -> None:
    provenance = _provenance(
        requested_backend="",
        actual_backend="",
        attempted_backends=(),
    )

    assert provenance.requested_backend == ""
    assert provenance.actual_backend == ""
    assert provenance.attempted_backends == ()


def test_requested_backend_fallback_requires_warning_status() -> None:
    with pytest.raises(ValidationError, match="warning"):
        OrganelleResult(
            operation_id="localization.predict",
            scope="mitochondrion",
            status="ok",
            provenance=_provenance(),
        )

    result = OrganelleResult(
        operation_id="localization.predict",
        scope="mitochondrion",
        status="warning",
        provenance=_provenance(),
    )

    assert result.status == "warning"


def test_provenance_freezes_json_mappings_and_guards_time_order() -> None:
    versions = {"deeploc": {"version": "1.0"}}
    provenance = _provenance(software_versions=versions)
    versions["deeploc"]["version"] = "changed"

    version = provenance.software_versions["deeploc"]
    assert isinstance(version, FrozenMap)
    assert version["version"] == "1.0"
    with pytest.raises(TypeError):
        version["version"] = "changed"  # type: ignore[index]
    with pytest.raises(ValidationError, match="finished_at cannot be earlier"):
        _provenance(
            started_at=datetime(2026, 7, 10, 12, tzinfo=UTC),
            finished_at=datetime(2026, 7, 10, 11, tzinfo=UTC),
        )


def test_result_identity_uses_artifact_and_provenance_object_ids() -> None:
    first_artifact = ArtifactRef(
        kind="report", uri="first/result.json", format="json", sha256="a" * 64, size_bytes=4
    )
    second_artifact = ArtifactRef(
        kind="report", uri="second/result.json", format="json", sha256="a" * 64, size_bytes=4
    )
    first_provenance = _provenance(started_at=datetime(2026, 7, 10, tzinfo=UTC))
    second_provenance = _provenance(started_at=datetime(2026, 7, 11, tzinfo=UTC))

    first = OrganelleResult(
        operation_id="localization.predict",
        scope="mitochondrion",
        status="warning",
        artifacts=(first_artifact,),
        provenance=first_provenance,
    )
    second = OrganelleResult(
        operation_id="localization.predict",
        scope="mitochondrion",
        status="warning",
        artifacts=(second_artifact,),
        provenance=second_provenance,
    )

    assert first_artifact.uri != second_artifact.uri
    assert first_artifact.object_id == second_artifact.object_id
    assert first_provenance.object_id == second_provenance.object_id
    assert first.object_id == second.object_id


def test_result_json_round_trip_preserves_nested_contracts_and_rejects_extras() -> None:
    artifact = ArtifactRef(
        kind="report",
        uri="result.json",
        format="json",
        sha256="a" * 64,
        size_bytes=4,
    )
    result = OrganelleResult(
        operation_id="quality_control.assess",
        scope="mitochondrion",
        status="warning",
        metrics=FrozenMap.from_json({"coverage": [42.0]}),
        findings=(
            Finding(
                code="quality.low_coverage",
                metric="coverage",
                value=42.0,
                evidence_artifact_ids=(artifact.object_id,),
            ),
        ),
        artifacts=(artifact,),
        provenance=_provenance(software_versions=FrozenMap.from_json({"tool": {"version": "1.0"}})),
        errors=(
            ErrorDetail(
                code="quality.low_coverage",
                message="coverage is below target",
                details=FrozenMap.from_json({"coverage": 42.0}),
                suggested_action=FrozenMap.from_json({"operation_id": "quality_control.assess"}),
            ),
        ),
        suggested_operations=(
            OperationSuggestion(
                operation_id="quality_control.assess",
                reason_code="quality.low_coverage",
                parameter_changes=FrozenMap.from_json({"minimum_coverage": 50}),
            ),
        ),
    )

    dumped = result.model_dump(mode="json", exclude={"object_id"})
    assert "object_id" not in dumped["artifacts"][0]
    assert "object_id" not in dumped["provenance"]
    assert "object_id" not in dumped["findings"][0]
    restored = OrganelleResult.model_validate_json(result.model_dump_json(exclude={"object_id"}))

    assert restored == result
    assert restored.object_id == result.object_id
    with pytest.raises(ValidationError):
        OrganelleResult.model_validate({**dumped, "unexpected": True})
    with pytest.raises(ValidationError):
        OrganelleResult.model_validate(
            {**dumped, "findings": [{**dumped["findings"][0], "unexpected": True}]}
        )


def test_default_result_dumps_restore_nested_contracts_without_losing_json_object_ids() -> None:
    artifact = ArtifactRef(
        kind="report",
        uri="result.json",
        format="json",
        sha256="a" * 64,
        size_bytes=4,
    )
    result = OrganelleResult(
        operation_id="quality_control.assess",
        scope="mitochondrion",
        status="warning",
        metrics=FrozenMap.from_json({"object_id": "user-metric-id", "coverage": [42.0]}),
        findings=(Finding(code="quality.low_coverage", value=42.0),),
        artifacts=(artifact,),
        provenance=_provenance(
            software_versions=FrozenMap.from_json({"object_id": "user-version-id"})
        ),
        errors=(
            ErrorDetail(
                code="quality.low_coverage",
                message="coverage is below target",
                details=FrozenMap.from_json({"object_id": "user-detail-id"}),
            ),
        ),
        suggested_operations=(
            OperationSuggestion(
                operation_id="quality_control.assess",
                reason_code="quality.low_coverage",
                parameter_changes=FrozenMap.from_json({"object_id": "user-suggestion-id"}),
            ),
        ),
    )

    for restored in (
        OrganelleResult.model_validate(result.model_dump()),
        OrganelleResult.model_validate_json(result.model_dump_json()),
    ):
        assert restored == result
        assert restored.object_id == result.object_id
        assert restored.metrics["object_id"] == "user-metric-id"
        assert restored.errors[0].details["object_id"] == "user-detail-id"

    with pytest.raises(ValidationError):
        Finding(code="quality.low_coverage", object_id="forged")  # type: ignore[call-arg]


def test_public_deserializers_validate_root_and_nested_computed_object_ids() -> None:
    finding = Finding(code="quality.low_coverage", value=42.0)
    result = OrganelleResult(
        operation_id="quality_control.assess",
        scope="mitochondrion",
        status="warning",
        findings=(finding,),
    )
    payload = result.model_dump()

    assert OrganelleResult.model_validate(payload) == result
    assert OrganelleResult.model_validate_json(result.model_dump_json()) == result

    forged_root = {**payload, "object_id": "result:sha256:" + "0" * 64}
    with pytest.raises(ValidationError, match="object_id"):
        OrganelleResult.model_validate(forged_root)

    forged_nested = {
        **payload,
        "findings": [{**finding.model_dump(), "object_id": "finding:sha256:" + "0" * 64}],
    }
    with pytest.raises(ValidationError, match="object_id"):
        OrganelleResult.model_validate_json(json.dumps(forged_nested))


def test_model_copy_revalidates_updates_and_freezes_mapping_values() -> None:
    result = OrganelleResult(
        operation_id="quality_control.assess",
        scope="mitochondrion",
        status="ok",
        metrics=FrozenMap.from_json({"coverage": 42.0}),
    )
    metrics = {"coverage": [42.0]}

    with pytest.raises(ValidationError, match="failed result requires at least one error"):
        result.model_copy(update={"status": "failed"})
    with pytest.raises(ValidationError, match="failed result requires at least one error"):
        result.copy(update={"status": "failed"})

    copied = result.model_copy(update={"metrics": metrics})
    original_hash = hash(copied)
    metrics["coverage"].append(0.0)

    assert copied.metrics["coverage"] == (42.0,)
    assert hash(copied) == original_hash


@pytest.mark.parametrize("value", (float("nan"), float("inf"), float("-inf")))
def test_finding_value_rejects_non_finite_numbers(value: float) -> None:
    with pytest.raises(ValidationError, match="finite"):
        Finding(code="quality.low_coverage", value=value)


def test_provenance_rejects_mixed_naive_and_aware_timestamps() -> None:
    with pytest.raises(ValidationError, match="timezone"):
        _provenance(
            started_at=datetime(2026, 7, 10, 12),
            finished_at=datetime(2026, 7, 10, 13, tzinfo=UTC),
        )
