"""The four result codecs: canonical, legacy_result, json_metric, artifact.

``str(value)`` must never appear anywhere in ``capabilities/codecs.py`` as a
fallback; every codec either encodes a genuinely supported shape or raises a
clear ``OrganelleContractError``.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from organelleverse.capabilities.codecs import CapabilityExecutionContext, encode_capability_result
from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.data import OrganelleData
from organelleverse.core.errors import OrganelleContractError
from organelleverse.core.frozen import FrozenMap, thaw_json
from organelleverse.core.genome import OrganelleGenome
from organelleverse.core.provenance import ResultProvenance
from organelleverse.core.result import OrganelleResult
from organelleverse.operations.spec import CoreKind, PythonBindingSpec, ResultCodec


def _context(operation_id: str = "demo.probe") -> CapabilityExecutionContext:
    return CapabilityExecutionContext(
        operation_id=operation_id,
        operation_version="1.0",
        callable_locator="demo.api:probe",
        run_id="test-run-0001",
        parameters_hash="a" * 64,
    )


def test_context_carries_full_provenance_identity_fields() -> None:
    context = CapabilityExecutionContext(
        operation_id="demo.probe",
        operation_version="1.0",
        callable_locator="demo.api:probe",
        run_id="test-run-0001",
        input_object_ids=("genome:sha256:" + "a" * 64,),
        input_artifact_hashes=("b" * 64,),
        parameters_hash="c" * 64,
    )
    assert context.operation_version == "1.0"
    assert context.callable_locator == "demo.api:probe"
    assert context.input_object_ids == ("genome:sha256:" + "a" * 64,)
    assert context.input_artifact_hashes == ("b" * 64,)
    assert context.parameters_hash == "c" * 64


def test_context_defaults_input_identity_fields_to_empty() -> None:
    context = _context()
    assert context.input_object_ids == ()
    assert context.input_artifact_hashes == ()


def test_context_requires_a_well_formed_parameters_hash() -> None:
    with pytest.raises(ValidationError):
        CapabilityExecutionContext(
            operation_id="demo.probe",
            operation_version="1.0",
            callable_locator="demo.api:probe",
            run_id="test-run-0001",
            parameters_hash="not-a-hash",
        )


def test_context_requires_a_well_formed_callable_locator() -> None:
    with pytest.raises(ValidationError):
        CapabilityExecutionContext(
            operation_id="demo.probe",
            operation_version="1.0",
            callable_locator="not a locator",
            run_id="test-run-0001",
            parameters_hash="a" * 64,
        )


# --- canonical ---------------------------------------------------------------


def test_canonical_codec_revalidates_an_existing_result() -> None:
    binding = PythonBindingSpec(result_codec=ResultCodec.CANONICAL)
    source = OrganelleResult(operation_id="demo.probe", scope="none", status="ok")

    encoded = encode_capability_result(source, binding, _context())
    assert encoded == source


def test_canonical_codec_rejects_a_non_result_value() -> None:
    binding = PythonBindingSpec(result_codec=ResultCodec.CANONICAL)
    with pytest.raises(OrganelleContractError):
        encode_capability_result({"not": "a result"}, binding, _context())


@pytest.mark.parametrize(
    "result_codec",
    [ResultCodec.CANONICAL, ResultCodec.LEGACY_RESULT],
)
def test_in_process_result_codecs_preserve_trusted_identity_and_provenance(
    result_codec: ResultCodec,
) -> None:
    trusted_provenance = ResultProvenance(
        operation_id="demo.probe",
        operation_version="9.9",
        package_version="trusted-package",
        git_commit="trusted-commit",
        parameters_hash="f" * 64,
        callable_locator="trusted.impl:probe",
    )
    trusted = OrganelleResult(
        operation_id="demo.probe",
        operation_version="9.9",
        scope="none",
        status="ok",
        provenance=trusted_provenance,
    )
    context = CapabilityExecutionContext(
        operation_id="demo.probe",
        operation_version="2.0",
        callable_locator="parent.context:probe",
        run_id="in-process-preservation",
        parameters_hash="a" * 64,
    )

    encoded = encode_capability_result(
        trusted,
        PythonBindingSpec(result_codec=result_codec),
        context,
    )

    assert encoded == trusted
    assert encoded.object_id == trusted.object_id
    assert encoded.operation_version == "9.9"
    assert encoded.provenance == trusted_provenance


@pytest.mark.parametrize(
    ("output_kind", "source"),
    [
        (
            CoreKind.GENOME,
            OrganelleGenome(
                organelle="mitochondrion",
                sequence=ArtifactRef(
                    kind="sequence",
                    uri="sequence.fa",
                    format="fasta",
                    sha256="1" * 64,
                    size_bytes=10,
                ),
            ),
        ),
        (
            CoreKind.DATA,
            OrganelleData(
                modality="matrix",
                payload=FrozenMap.from_json({"value": 1}),
            ),
        ),
        (
            CoreKind.RESULT,
            OrganelleResult(
                operation_id="demo.probe",
                scope="none",
                status="ok",
                metrics=FrozenMap.from_json({"value": 1}),
            ),
        ),
    ],
)
def test_canonical_json_reconstructs_exactly_the_declared_parent_core_kind(
    output_kind: CoreKind,
    source: OrganelleGenome | OrganelleData | OrganelleResult,
) -> None:
    binding = PythonBindingSpec(result_codec=ResultCodec.CANONICAL_JSON)

    encoded = encode_capability_result(
        source.model_dump(mode="json"),
        binding,
        _context(),
        output_kind=output_kind,
    )

    assert type(encoded) is type(source)
    assert encoded == source


def test_canonical_json_rejects_missing_or_non_core_output_kinds() -> None:
    binding = PythonBindingSpec(result_codec=ResultCodec.CANONICAL_JSON)
    value = {"kind": "result"}

    with pytest.raises(OrganelleContractError):
        encode_capability_result(value, binding, _context())
    with pytest.raises(OrganelleContractError):
        encode_capability_result(
            value,
            binding,
            _context(),
            output_kind=CoreKind.NONE,
        )


def test_canonical_json_result_rejects_worker_operation_id_mismatch() -> None:
    binding = PythonBindingSpec(result_codec=ResultCodec.CANONICAL_JSON)
    value = OrganelleResult(
        operation_id="other.operation",
        scope="none",
        status="ok",
    ).model_dump(mode="json")

    with pytest.raises(OrganelleContractError) as captured:
        encode_capability_result(
            value,
            binding,
            _context("demo.probe"),
            output_kind=CoreKind.RESULT,
        )

    assert captured.value.code == "capability.result_codec_invalid"


def test_canonical_json_codec_decodes_before_worker_boundary_normalization() -> None:
    binding = PythonBindingSpec(result_codec=ResultCodec.CANONICAL_JSON)
    supplied_provenance = ResultProvenance(
        operation_id="demo.probe",
        operation_version="9.9",
        package_version="worker-package",
        git_commit="worker-commit",
        parameters_hash="f" * 64,
        callable_locator="worker.spoof:run",
    )
    source = OrganelleResult(
        operation_id="demo.probe",
        operation_version="9.9",
        scope="none",
        status="ok",
        provenance=supplied_provenance,
    )
    context = CapabilityExecutionContext(
        operation_id="demo.probe",
        operation_version="2.0",
        callable_locator="demo.api:probe",
        run_id="test-run-decode-only",
        parameters_hash="3" * 64,
    )

    encoded = encode_capability_result(
        source.model_dump(mode="json"),
        binding,
        context,
        output_kind=CoreKind.RESULT,
    )

    assert isinstance(encoded, OrganelleResult)
    assert encoded == source
    assert encoded.operation_version == "9.9"
    assert encoded.provenance == supplied_provenance


# --- legacy_result -------------------------------------------------------------
#
# legacy_result never imports or restores core.legacy: these tests simulate an
# archived pre-v1 result with a LOCAL, test-only dataclass. The codec must
# convert it field-by-field (suite+op -> operation_id, organelle -> scope,
# observed_metrics -> metrics, key_findings -> Finding, output_paths ->
# real-hash ArtifactRef, anomalies -> ErrorDetail, provenance -> a real
# ResultProvenance) - never `OrganelleResult.model_validate(dict(old))`
# wholesale, which would silently accept an old shape as if it were already
# canonical instead of actually converting it.


@dataclasses.dataclass
class _ArchivedLegacyResult:
    suite: str
    op: str
    organelle: str
    status: str
    output_paths: list[str]
    observed_metrics: dict[str, object]
    key_findings: list[dict[str, object]]
    flags: list[str]
    anomalies: list[dict[str, object]]
    summary_text: str
    provenance: dict[str, object]


def _legacy_payload(**changes: object) -> dict[str, object]:
    base = dataclasses.asdict(
        _ArchivedLegacyResult(
            suite="demo",
            op="probe",
            organelle="mitochondrion",
            status="ok",
            output_paths=[],
            observed_metrics={"count": 3},
            key_findings=[{"code": "demo.finding", "metric": "count", "value": 3}],
            flags=["reviewed"],
            anomalies=[],
            summary_text="legacy archived result",
            provenance={
                "operation_version": "2.0",
                "package_version": "0.9.0",
                "git_commit": "cafefeed",
                "parameters_hash": "f" * 64,
            },
        )
    )
    base.update(changes)
    return base


def test_legacy_result_codec_accepts_an_already_canonical_result() -> None:
    binding = PythonBindingSpec(result_codec=ResultCodec.LEGACY_RESULT)
    source = OrganelleResult(operation_id="demo.probe", scope="mitochondrion", status="ok")

    encoded = encode_capability_result(source, binding, _context())
    assert encoded == source


def test_legacy_result_codec_converts_a_well_formed_archived_mapping() -> None:
    binding = PythonBindingSpec(result_codec=ResultCodec.LEGACY_RESULT)

    encoded = encode_capability_result(_legacy_payload(), binding, _context())
    assert isinstance(encoded, OrganelleResult)
    assert encoded.operation_id == "demo.probe"
    assert encoded.scope == "mitochondrion"
    assert encoded.status == "ok"
    assert encoded.summary_text == "legacy archived result"
    assert encoded.flags == ("reviewed",)
    assert encoded.metrics["count"] == 3
    assert len(encoded.findings) == 1
    assert encoded.findings[0].code == "demo.finding"
    assert encoded.findings[0].value == 3


def test_legacy_result_codec_converts_provenance_field_by_field() -> None:
    binding = PythonBindingSpec(result_codec=ResultCodec.LEGACY_RESULT)
    context = _context()

    encoded = encode_capability_result(_legacy_payload(), binding, context)
    assert encoded.provenance is not None
    # operation_id comes from context (matching suite+op), never whatever a
    # wholesale dict-validate might have picked up from the legacy payload.
    assert encoded.provenance.operation_id == context.operation_id
    assert encoded.provenance.operation_version == "2.0"
    assert encoded.provenance.package_version == "0.9.0"
    assert encoded.provenance.git_commit == "cafefeed"
    assert encoded.provenance.parameters_hash == "f" * 64


def test_legacy_result_codec_converts_output_paths_into_real_hash_artifacts(tmp_path: Path) -> None:
    binding = PythonBindingSpec(result_codec=ResultCodec.LEGACY_RESULT)
    written = tmp_path / "legacy_output.txt"
    written.write_text("legacy artifact content", encoding="utf-8")

    encoded = encode_capability_result(
        _legacy_payload(output_paths=[str(written)]), binding, _context()
    )
    assert len(encoded.artifacts) == 1
    assert encoded.artifacts[0].sha256 == _sha256_of(written)


def test_legacy_result_codec_converts_failed_status_anomalies_into_error_details() -> None:
    binding = PythonBindingSpec(result_codec=ResultCodec.LEGACY_RESULT)
    payload = _legacy_payload(
        status="failed",
        anomalies=[{"code": "demo.anomaly", "message": "legacy failure", "retryable": True}],
    )

    encoded = encode_capability_result(payload, binding, _context())
    assert encoded.status == "failed"
    assert len(encoded.errors) == 1
    assert encoded.errors[0].code == "demo.anomaly"
    assert encoded.errors[0].message == "legacy failure"
    assert encoded.errors[0].retryable is True


def test_legacy_result_codec_requires_suite_and_op_to_match_the_context_operation_id() -> None:
    binding = PythonBindingSpec(result_codec=ResultCodec.LEGACY_RESULT)
    payload = _legacy_payload(suite="other", op="mismatch")

    with pytest.raises(OrganelleContractError):
        encode_capability_result(payload, binding, _context())


def test_legacy_result_codec_rejects_a_missing_required_legacy_field() -> None:
    binding = PythonBindingSpec(result_codec=ResultCodec.LEGACY_RESULT)
    payload = _legacy_payload()
    del payload["anomalies"]

    with pytest.raises(OrganelleContractError):
        encode_capability_result(payload, binding, _context())


def test_legacy_result_codec_rejects_an_invalid_organelle_value() -> None:
    binding = PythonBindingSpec(result_codec=ResultCodec.LEGACY_RESULT)
    payload = _legacy_payload(organelle="not-a-real-scope")

    with pytest.raises(OrganelleContractError):
        encode_capability_result(payload, binding, _context())


def test_legacy_result_codec_rejects_a_malformed_key_finding() -> None:
    binding = PythonBindingSpec(result_codec=ResultCodec.LEGACY_RESULT)
    payload = _legacy_payload(key_findings=[{"metric": "count", "value": 3}])  # missing "code"

    with pytest.raises(OrganelleContractError):
        encode_capability_result(payload, binding, _context())


def test_legacy_result_codec_requires_anomalies_when_status_is_failed() -> None:
    binding = PythonBindingSpec(result_codec=ResultCodec.LEGACY_RESULT)
    payload = _legacy_payload(status="failed", anomalies=[])

    with pytest.raises(OrganelleContractError):
        encode_capability_result(payload, binding, _context())


def test_legacy_result_codec_rejects_a_result_shaped_mapping_that_is_not_the_legacy_shape() -> None:
    """The pre-v1 fake conversion (validating an OrganelleResult-shaped dict as
    if it already were one) is gone: legacy_result now requires the real
    archived shape (suite/op/organelle/...), not today's field names."""
    binding = PythonBindingSpec(result_codec=ResultCodec.LEGACY_RESULT)
    payload = {
        "operation_id": "demo.probe",
        "scope": "mitochondrion",
        "status": "ok",
        "summary_text": "looks canonical but is not the archived legacy shape",
    }

    with pytest.raises(OrganelleContractError):
        encode_capability_result(payload, binding, _context())


def test_legacy_result_codec_rejects_an_unsupported_shape_without_a_str_fallback() -> None:
    binding = PythonBindingSpec(result_codec=ResultCodec.LEGACY_RESULT)

    class ArbitraryLegacyObject:
        pass

    with pytest.raises(OrganelleContractError):
        encode_capability_result(ArbitraryLegacyObject(), binding, _context())


# --- json_metric ---------------------------------------------------------------


def test_json_metric_codec_stores_the_value_under_result_key() -> None:
    binding = PythonBindingSpec(result_codec=ResultCodec.JSON_METRIC, result_key="gc")
    value = {"gc_content": 0.42, "windows": [{"start": 1, "end": 500, "gc_skew": 0.1}]}

    encoded = encode_capability_result(value, binding, _context())
    assert encoded.status == "ok"
    assert thaw_json(encoded.metrics["gc"]) == value


@pytest.mark.parametrize(
    "invalid_value",
    [
        float("nan"),
        float("inf"),
        {"bad": float("nan")},
        {1, 2, 3},
        object(),
        (x for x in range(3)),
    ],
)
def test_json_metric_codec_rejects_non_finite_or_non_json_values(invalid_value: object) -> None:
    binding = PythonBindingSpec(result_codec=ResultCodec.JSON_METRIC, result_key="value")
    with pytest.raises(OrganelleContractError):
        encode_capability_result(invalid_value, binding, _context())


def test_json_metric_codec_accepts_nested_finite_structures() -> None:
    binding = PythonBindingSpec(result_codec=ResultCodec.JSON_METRIC, result_key="value")
    value = {"a": [1, 2.5, "x", None, True], "b": {"c": [1, 2]}}
    encoded = encode_capability_result(value, binding, _context())
    assert thaw_json(encoded.metrics["value"]) == value


def test_json_metric_codec_populates_provenance_from_the_context() -> None:
    binding = PythonBindingSpec(result_codec=ResultCodec.JSON_METRIC, result_key="value")
    context = CapabilityExecutionContext(
        operation_id="demo.probe",
        operation_version="1.0",
        callable_locator="demo.api:probe",
        run_id="test-run-0001",
        input_object_ids=("genome:sha256:" + "a" * 64,),
        input_artifact_hashes=("b" * 64,),
        parameters_hash="c" * 64,
    )
    encoded = encode_capability_result({"x": 1}, binding, context)
    assert encoded.provenance is not None
    assert encoded.provenance.operation_id == "demo.probe"
    assert encoded.provenance.operation_version == "1.0"
    assert encoded.provenance.callable_locator == "demo.api:probe"
    assert encoded.provenance.input_object_ids == ("genome:sha256:" + "a" * 64,)
    assert encoded.provenance.input_artifact_hashes == ("b" * 64,)
    assert encoded.provenance.parameters_hash == "c" * 64


# --- artifact --------------------------------------------------------------------


def test_artifact_codec_encodes_a_single_path(tmp_path: Path) -> None:
    binding = PythonBindingSpec(result_codec=ResultCodec.ARTIFACT)
    written = tmp_path / "output.txt"
    written.write_text("hello", encoding="utf-8")

    encoded = encode_capability_result(written, binding, _context())
    assert len(encoded.artifacts) == 1
    assert encoded.artifacts[0].sha256 == _sha256_of(written)


def test_artifact_codec_encodes_a_sequence_of_paths(tmp_path: Path) -> None:
    binding = PythonBindingSpec(result_codec=ResultCodec.ARTIFACT)
    first = tmp_path / "a.txt"
    second = tmp_path / "b.txt"
    first.write_text("a", encoding="utf-8")
    second.write_text("b", encoding="utf-8")

    encoded = encode_capability_result([first, second], binding, _context())
    assert len(encoded.artifacts) == 2
    for artifact in encoded.artifacts:
        assert artifact.sha256 == _sha256_of(Path(artifact.uri))


def test_artifact_codec_multiple_paths_with_the_same_basename_do_not_collide(
    tmp_path: Path,
) -> None:
    binding = PythonBindingSpec(result_codec=ResultCodec.ARTIFACT)
    first_dir = tmp_path / "chr1"
    second_dir = tmp_path / "chr2"
    first_dir.mkdir()
    second_dir.mkdir()
    first = first_dir / "summary.txt"
    second = second_dir / "summary.txt"
    first.write_text("chromosome one summary", encoding="utf-8")
    second.write_text("chromosome two summary", encoding="utf-8")

    encoded = encode_capability_result([first, second], binding, _context())
    assert len(encoded.artifacts) == 2

    uris = [artifact.uri for artifact in encoded.artifacts]
    assert len(set(uris)) == 2, "colliding basenames must get distinct destination paths"

    # Each artifact's declared digest must match what is *currently* at its
    # own uri - never a value computed before a same-named sibling
    # overwrote the destination.
    for artifact in encoded.artifacts:
        assert artifact.sha256 == _sha256_of(Path(artifact.uri))
    contents = {Path(artifact.uri).read_text(encoding="utf-8") for artifact in encoded.artifacts}
    assert contents == {"chromosome one summary", "chromosome two summary"}


def test_artifact_codec_destination_naming_is_deterministic(tmp_path: Path) -> None:
    """Same input order -> same destination names, every time."""
    binding = PythonBindingSpec(result_codec=ResultCodec.ARTIFACT)
    first = tmp_path / "a" / "output.txt"
    second = tmp_path / "b" / "output.txt"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text("first", encoding="utf-8")
    second.write_text("second", encoding="utf-8")

    first_run = encode_capability_result(
        [first, second], binding, _context(operation_id="demo.probe")
    )
    names_first_run = tuple(Path(artifact.uri).name for artifact in first_run.artifacts)

    # A second, independent operation_id/run_id proves the naming rule
    # itself is a pure function of call order, not of any hidden state.
    second_run = encode_capability_result(
        [first, second], binding, _context(operation_id="demo.numpy_probe")
    )
    names_second_run = tuple(Path(artifact.uri).name for artifact in second_run.artifacts)
    assert names_first_run == names_second_run


def test_artifact_codec_single_path_keeps_its_plain_basename(tmp_path: Path) -> None:
    """A single-path result has no possible collision; the destination name
    is exactly the source basename, unchanged from before this fix."""
    binding = PythonBindingSpec(result_codec=ResultCodec.ARTIFACT)
    written = tmp_path / "output.txt"
    written.write_text("hello", encoding="utf-8")

    encoded = encode_capability_result(written, binding, _context())
    assert Path(encoded.artifacts[0].uri).name == "output.txt"


def test_artifact_codec_source_already_inside_the_managed_run_is_not_recopied(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A single artifact already sitting at its own managed-run destination
    must not be copied onto itself (shutil.copy2(x, x) would fail)."""
    import shutil

    from organelleverse import runtime

    monkeypatch.setattr(runtime, "cache_root", lambda: tmp_path)
    binding = PythonBindingSpec(result_codec=ResultCodec.ARTIFACT)
    context = _context()
    run_root = runtime.managed_run_path(context.operation_id, context.run_id)
    run_root.mkdir(parents=True)
    already_there = run_root / "output.txt"
    already_there.write_text("already managed", encoding="utf-8")

    def _forbidden_copy(*args: object, **kwargs: object) -> object:
        raise AssertionError("a source already at its destination must never be copied")

    monkeypatch.setattr(shutil, "copy2", _forbidden_copy)

    encoded = encode_capability_result(already_there, binding, context)
    assert encoded.artifacts[0].uri == str(already_there)
    assert encoded.artifacts[0].sha256 == _sha256_of(already_there)


def test_artifact_codec_rejects_a_missing_path(tmp_path: Path) -> None:
    binding = PythonBindingSpec(result_codec=ResultCodec.ARTIFACT)
    with pytest.raises(OrganelleContractError):
        encode_capability_result(tmp_path / "does-not-exist.txt", binding, _context())


def test_artifact_codec_encodes_numpy_data() -> None:
    binding = PythonBindingSpec(result_codec=ResultCodec.ARTIFACT)
    array = np.array([[1, 2], [3, 4]])

    encoded = encode_capability_result(array, binding, _context(operation_id="demo.numpy_probe"))
    assert len(encoded.artifacts) == 1
    assert encoded.artifacts[0].format == "npy"
    restored = np.load(encoded.artifacts[0].uri)
    assert np.array_equal(restored, array)


def test_artifact_codec_rejects_an_unsupported_type() -> None:
    binding = PythonBindingSpec(result_codec=ResultCodec.ARTIFACT)
    with pytest.raises(OrganelleContractError):
        encode_capability_result(object(), binding, _context())


def test_artifact_codec_populates_provenance_from_the_context(tmp_path: Path) -> None:
    binding = PythonBindingSpec(result_codec=ResultCodec.ARTIFACT)
    written = tmp_path / "output.txt"
    written.write_text("hello", encoding="utf-8")
    context = CapabilityExecutionContext(
        operation_id="demo.probe",
        operation_version="2.0",
        callable_locator="demo.api:make_output",
        run_id="test-run-0002",
        parameters_hash="d" * 64,
    )

    encoded = encode_capability_result(written, binding, context)
    assert encoded.provenance is not None
    assert encoded.provenance.operation_version == "2.0"
    assert encoded.provenance.callable_locator == "demo.api:make_output"
    assert encoded.provenance.parameters_hash == "d" * 64


def test_canonical_codec_preserves_the_original_provenance_unchanged() -> None:
    from organelleverse.core.provenance import ResultProvenance

    original_provenance = ResultProvenance(
        operation_id="demo.probe",
        operation_version="9.9",
        package_version="0.0.1",
        git_commit="deadbee",
        parameters_hash="e" * 64,
    )
    binding = PythonBindingSpec(result_codec=ResultCodec.CANONICAL)
    source = OrganelleResult(
        operation_id="demo.probe", scope="none", status="ok", provenance=original_provenance
    )

    encoded = encode_capability_result(source, binding, _context())
    assert encoded.provenance == original_provenance


def _sha256_of(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_no_str_fallback_anywhere_in_codecs_module() -> None:
    """Structural guard: the forbidden ``return str(value)`` pattern must not exist."""
    import inspect

    from organelleverse.capabilities import codecs

    source = inspect.getsource(codecs)
    assert "return str(value)" not in source
