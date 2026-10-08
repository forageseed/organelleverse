from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, cast

import pytest

from organelleverse.capabilities.admission import (
    ReferenceIdentity,
    admit_capabilities,
    evaluate_admission,
)
from organelleverse.capabilities.code_identity import ExecutionIdentity, inspect_bundle_code
from organelleverse.capabilities.hashing import hash_bundle, hash_fixture_dataset
from organelleverse.capabilities.index import (
    CapabilityConflict,
    CapabilityConflictMember,
    CapabilityDiagnostic,
    CapabilityEntry,
    CapabilityIndex,
    CapabilityOrigin,
)
from organelleverse.capabilities.models import CapabilityBundle, FixtureSpec
from organelleverse.capabilities.trust import TrustStore, trust
from organelleverse.capabilities.verification import (
    CandidateEvidence,
    EnvironmentSnapshot,
    EquivalenceEvidence,
    PlatformIdentity,
    ReferenceEvidence,
    VerificationRecord,
    VerificationStore,
    compute_environment_key,
    current_platform_identity,
    verify_capability,
)
from organelleverse.capabilities.worker import BundleWorkerExecutor
from organelleverse.capabilities.worker_contracts import (
    WorkerInspection,
    WorkerParameter,
    WorkerResult,
)
from organelleverse.core.data import OrganelleData
from organelleverse.core.errors import (
    OrganelleContractError,
    OrganelleDependencyError,
    OrganelleInputError,
    OrganellePermissionError,
)
from organelleverse.core.frozen import FrozenMap
from organelleverse.core.result import OperationSuggestion, OrganelleResult
from organelleverse.operations.data_contracts import DataContract
from organelleverse.operations.python_binding import (
    bind_python_capability,
    worker_parameter_schema,
)
from organelleverse.operations.registry import CoreObject, OperationRegistry
from organelleverse.operations.schemas import parameter_schema
from organelleverse.operations.spec import (
    ArgumentMode,
    CoreKind,
    ParameterBindingSpec,
    ParameterCodec,
    PythonBindingSpec,
    ResultCodec,
)

_BUNDLE_HASH = "sha256:" + "1" * 64
_FIXTURE_HASH = "sha256:" + "2" * 64
_PRODUCED_HASH = "sha256:" + "3" * 64
_REFERENCE_HASH = "sha256:" + "4" * 64
_CODE_HASH = "sha256:" + "5" * 64


def _execution_identity(
    *,
    capability_id: str = "demo.capability",
    bundle_hash: str = _BUNDLE_HASH,
    code_hash: str = _CODE_HASH,
    callable_locator: str = "demo.impl:run",
) -> ExecutionIdentity:
    interpreter = f"{sys.implementation.name}-{sys.version_info.major}.{sys.version_info.minor}"
    payload = {
        "kind": "bundle-local-python-v1",
        "capability_id": capability_id,
        "bundle_content_hash": bundle_hash,
        "code_tree_hash": code_hash,
        "callable_locator": callable_locator,
        "interpreter": interpreter,
        "worker_protocol": "organelleverse.bundle-worker.v1",
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return ExecutionIdentity(
        capability_id=capability_id,
        bundle_content_hash=bundle_hash,
        code_tree_hash=code_hash,
        callable_locator=callable_locator,
        interpreter=interpreter,
        digest=f"sha256:{hashlib.sha256(canonical).hexdigest()}",
    )


_EXECUTION_DIGEST = _execution_identity().digest


def _bundle(
    *, implementation: str = "native", callable_locator: str = "demo.impl:run"
) -> CapabilityBundle:
    contract: dict[str, object] = {
        "contract_version": "1.0",
        "operation_id": "demo.capability",
        "title": "Demo capability",
        "description": "A complete admission fixture capability contract.",
        "keywords": ("admission", "capability", "demo"),
        "execution_mode": "inline",
        "stage": "analyze",
        "input_kind": "none",
        "output_kind": "result",
        "organelle_types": ("mitochondrion",),
        "callable_locator": callable_locator if implementation != "composite" else None,
        "side_effects": (),
        "deterministic": True,
        "idempotent": True,
        "cacheable": True,
        "references": (),
        "binding": {"argument_mode": "named_parameters", "result_codec": "canonical"},
        "fallback": {"allowed": False},
    }
    payload: dict[str, object] = {
        "schema": "organelleverse.capability.v1",
        "capability": {
            "id": "demo.capability",
            "bundle_version": "1.0.0",
            "implementation": implementation,
        },
        "contract": contract,
    }
    if implementation == "composite":
        payload["composite"] = {
            "steps": (
                {
                    "id": "first",
                    "invocation": {"capability_id": "demo.downstream"},
                },
            )
        }
    return CapabilityBundle.model_validate(payload)


def _bundle_with_fixture(*, method: str = "exact") -> CapabilityBundle:
    payload = _bundle().model_dump(mode="python", by_alias=True)
    contract = cast(dict[str, object], payload["contract"])
    binding = cast(dict[str, object], contract["binding"])
    binding["result_codec"] = "canonical_json"
    payload["fixture"] = (
        {
            "case": "fixture",
            "input": {"path": "input.txt"},
            "expect": "expected.json",
            "equivalence": method,
        },
    )
    return CapabilityBundle.model_validate(payload)


def _entry(
    tmp_path: Path,
    *,
    implementation: str = "native",
    bundle: CapabilityBundle | None = None,
) -> CapabilityEntry:
    selected_bundle = bundle or _bundle(implementation=implementation)
    execution_identity = (
        _execution_identity(
            capability_id=selected_bundle.capability.id,
            bundle_hash=_BUNDLE_HASH,
            code_hash=_CODE_HASH,
            callable_locator=cast(str, selected_bundle.contract.callable_locator),
        )
        if selected_bundle.capability.implementation.value == "native"
        else None
    )
    return CapabilityEntry(
        capability_id="demo.capability",
        content_hash=_BUNDLE_HASH,
        bundle_root=tmp_path,
        bundle=selected_bundle,
        origins=(
            CapabilityOrigin(
                channel="local",
                source_path=str(tmp_path / "capability.toml"),
                search_root=str(tmp_path),
            ),
        ),
        execution_identity=execution_identity,
    )


def test_entry_rejects_execution_identity_locator_mismatch(tmp_path: Path) -> None:
    entry = _entry(tmp_path)
    mismatched_bundle = entry.bundle.model_copy(
        update={
            "contract": entry.bundle.contract.model_copy(
                update={"callable_locator": "different.impl:run"}
            )
        }
    )

    with pytest.raises(ValueError, match="callable locator"):
        entry.model_copy(update={"bundle": mismatched_bundle})


def _platform() -> PlatformIdentity:
    return PlatformIdentity(os="linux", arch="x86_64", libc="glibc-2.39", python="3.11.9")


def _environment() -> EnvironmentSnapshot:
    platform = _platform()
    return EnvironmentSnapshot(platform=platform, observed_environment={})


def _equivalence(
    *, verdict: Literal["pass", "fail"] = "pass", reference: bool = False
) -> EquivalenceEvidence:
    environment_key = compute_environment_key(_environment())
    return EquivalenceEvidence(
        case="fixture",
        method="exact",
        evaluator_version="1.0",
        verdict=verdict,
        fixture_dataset_hash=_FIXTURE_HASH,
        candidate=CandidateEvidence(
            bundle_content_hash=_BUNDLE_HASH,
            environment_key=environment_key,
            produced_hash=_PRODUCED_HASH,
        ),
        reference=(
            ReferenceEvidence(
                capability_id="reference.capability",
                bundle_content_hash=_REFERENCE_HASH,
                environment_key=environment_key,
                produced_hash=_PRODUCED_HASH,
            )
            if reference
            else None
        ),
    )


def _record(*, equivalence: tuple[EquivalenceEvidence, ...] | None = None) -> VerificationRecord:
    environment = _environment()
    return VerificationRecord(
        protocol_version="1.0",
        capability_id="demo.capability",
        bundle_content_hash=_BUNDLE_HASH,
        execution_identity=_execution_identity(),
        fixture_dataset_hash=_FIXTURE_HASH,
        adapter_requirements={},
        observed_environment=environment.observed_environment,
        environment_key=compute_environment_key(environment),
        equivalence=(_equivalence(),) if equivalence is None else equivalence,
        platform=environment.platform,
        worker_parameters=(),
        parameter_schema={
            "type": "object",
            "properties": {},
            "title": "demo_capability_WorkerParameters",
            "additionalProperties": False,
        },
        verified_at="2026-07-31T00:00:00Z",
        organelleverse_version="0.0.1",
    )


@pytest.mark.parametrize(
    ("record", "environment", "expected"),
    [
        (None, _environment(), "capability.verification_missing"),
        (
            _record().model_copy(update={"bundle_content_hash": "sha256:" + "9" * 64}),
            _environment(),
            "capability.verification_stale",
        ),
        (
            _record().model_copy(update={"fixture_dataset_hash": "sha256:" + "8" * 64}),
            _environment(),
            "capability.verification_stale",
        ),
        (
            _record().model_copy(update={"protocol_version": "9.0"}),
            _environment(),
            "capability.protocol_unsupported",
        ),
        (
            _record(),
            EnvironmentSnapshot(
                platform=_platform().model_copy(update={"arch": "aarch64"}),
                observed_environment={},
            ),
            "capability.platform_mismatch",
        ),
        (
            _record().model_copy(update={"environment_key": "sha256:" + "7" * 64}),
            _environment(),
            "capability.environment_drift",
        ),
        (
            _record(equivalence=(_equivalence(verdict="fail"),)),
            _environment(),
            "capability.equivalence_failed",
        ),
    ],
)
def test_each_non_reference_admission_rule_has_an_independent_error_code(
    tmp_path: Path,
    record: VerificationRecord | None,
    environment: EnvironmentSnapshot,
    expected: str,
) -> None:
    decision = evaluate_admission(
        _entry(tmp_path),
        record=record,
        current_environment=environment,
        current_fixture_hash=_FIXTURE_HASH,
        reference_identities={},
    )
    assert decision.admitted is False
    assert decision.diagnostic is not None
    assert decision.diagnostic.code == expected


@pytest.mark.parametrize(
    ("recorded_identity", "expected_recorded"),
    [
        (
            _execution_identity(code_hash="sha256:" + "7" * 64),
            "sha256:" + "7" * 64,
        ),
        (
            _execution_identity(callable_locator="demo.other:run"),
            _CODE_HASH,
        ),
    ],
)
def test_code_or_locator_execution_identity_drift_is_verification_stale(
    tmp_path: Path,
    recorded_identity: ExecutionIdentity,
    expected_recorded: str,
) -> None:
    record = _record().model_copy(update={"execution_identity": recorded_identity})

    decision = evaluate_admission(
        _entry(tmp_path),
        record=record,
        current_environment=_environment(),
        current_fixture_hash=_FIXTURE_HASH,
        reference_identities={},
    )

    assert decision.admitted is False
    assert decision.diagnostic is not None
    assert decision.diagnostic.code == "capability.verification_stale"
    assert decision.diagnostic.details["recorded_code_hash"] == expected_recorded
    assert decision.diagnostic.details["current_code_hash"] == _CODE_HASH
    assert decision.diagnostic.details["recorded_execution_digest"] == recorded_identity.digest
    assert decision.diagnostic.details["current_execution_digest"] == _EXECUTION_DIGEST


def test_reference_drift_is_distinct_from_candidate_environment_drift(tmp_path: Path) -> None:
    record = _record(equivalence=(_equivalence(reference=True),))
    decision = evaluate_admission(
        _entry(tmp_path),
        record=record,
        current_environment=_environment(),
        current_fixture_hash=_FIXTURE_HASH,
        reference_identities={
            "reference.capability": ReferenceIdentity(
                capability_id="reference.capability",
                bundle_content_hash="sha256:" + "5" * 64,
                environment_key=compute_environment_key(_environment()),
                evaluator_versions=("1.0",),
            )
        },
    )
    assert decision.diagnostic is not None
    assert decision.diagnostic.code == "capability.reference_drift"


def test_composite_is_describable_but_not_admitted_in_v1(tmp_path: Path) -> None:
    decision = evaluate_admission(
        _entry(tmp_path, implementation="composite"),
        record=None,
        current_environment=_environment(),
        current_fixture_hash=_FIXTURE_HASH,
        reference_identities={},
    )
    assert decision.diagnostic is not None
    assert decision.diagnostic.code == "capability.implementation_not_supported"


def test_a_fully_matching_record_is_admitted(tmp_path: Path) -> None:
    decision = evaluate_admission(
        _entry(tmp_path),
        record=_record(),
        current_environment=_environment(),
        current_fixture_hash=_FIXTURE_HASH,
        reference_identities={},
    )
    assert decision.admitted is True
    assert decision.diagnostic is None


def test_environment_key_ignores_provenance_only_realpath_and_fast_key() -> None:
    # The empty environment makes this invariant explicit without conflating it
    # with executable-record construction; executable-specific coverage lives
    # in test_probes and verification tests.
    first = _environment()
    second = _environment()
    assert compute_environment_key(first) == compute_environment_key(second)


class _VerificationExecutor:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.inspection_calls = 0
        self.invocation_calls = 0

    def inspect(self, entry: CapabilityEntry) -> WorkerInspection:
        self.events.append("inspect")
        self.inspection_calls += 1
        assert entry.execution_identity is not None
        return WorkerInspection(
            request_id="transport-request-id",
            execution_identity=entry.execution_identity,
            parameters=(),
        )

    def invoke(
        self,
        entry: CapabilityEntry,
        *,
        input: CoreObject | None,
        parameters: Mapping[str, object],
        run_id: str,
        staging_root: Path,
    ) -> WorkerResult:
        self.invocation_calls += 1
        raise AssertionError("fixture evaluation owns invocation in this test environment")


class _VerificationEnvironment:
    def __init__(
        self,
        index: CapabilityIndex,
        *,
        verdict: Literal["pass", "fail"] = "pass",
        fail_observation: bool = False,
    ) -> None:
        self._index = index
        self.verdict: Literal["pass", "fail"] = verdict
        self.fail_observation = fail_observation
        self.events: list[str] = []
        self._executor = _VerificationExecutor(self.events)
        self.fixture_calls = 0

    @property
    def index(self) -> CapabilityIndex:
        return self._index

    @property
    def executor(self) -> _VerificationExecutor:
        return self._executor

    def observe_environment(self, entry: CapabilityEntry) -> EnvironmentSnapshot:
        self.events.append("observe")
        if self.fail_observation:
            raise OrganelleDependencyError(
                code="capability.asset_unresolvable",
                message="asset unavailable",
                details={"reason": "asset.unresolvable"},
            )
        return _environment()

    def evaluate_fixture(
        self,
        entry: CapabilityEntry,
        fixture: FixtureSpec,
        *,
        executor: BundleWorkerExecutor,
        environment_key: str,
        fixture_dataset_hash: str,
    ) -> EquivalenceEvidence:
        self.events.append("fixture")
        self.fixture_calls += 1
        assert executor is self.executor
        assert entry.worker_parameters == ()
        assert entry.parameter_schema is not None
        return EquivalenceEvidence(
            case=fixture.case,
            method="exact",
            evaluator_version="1.0",
            verdict=self.verdict,
            fixture_dataset_hash=fixture_dataset_hash,
            candidate=CandidateEvidence(
                bundle_content_hash=entry.content_hash,
                environment_key=environment_key,
                produced_hash=_PRODUCED_HASH,
            ),
        )


def test_verification_store_uses_windows_safe_hash_components_and_round_trips(
    tmp_path: Path,
) -> None:
    store = VerificationStore(tmp_path / "verifications")
    path = store.write(_record())

    assert ":" not in path.relative_to(store.root).as_posix()
    assert path.parts[-3:-1] == ("sha256", "1" * 64)
    assert store.read("demo.capability", _BUNDLE_HASH, _record().environment_key) == _record()
    assert json.loads(path.read_text(encoding="utf-8"))["schema"] == (
        "organelleverse.verification.v2"
    )


def test_v1_verification_record_is_ignored_without_migration(tmp_path: Path) -> None:
    store = VerificationStore(tmp_path / "verifications")
    record = _record()
    path = store.path_for(record.capability_id, record.bundle_content_hash, record.environment_key)
    path.parent.mkdir(parents=True)
    payload = record.model_dump(mode="json", by_alias=True)
    payload["schema"] = "organelleverse.verification.v1"
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert (
        store.read(record.capability_id, record.bundle_content_hash, record.environment_key) is None
    )
    assert store.records_for(record.capability_id) == ()


def test_unknown_verification_record_schema_is_a_miss_not_a_crash(tmp_path: Path) -> None:
    store = VerificationStore(tmp_path / "verifications")
    path = store.write(_record())
    path.write_text('{"schema":"organelleverse.verification.v99"}', encoding="utf-8")

    assert store.read("demo.capability", _BUNDLE_HASH, _record().environment_key) is None


def test_verify_writes_a_failed_scientific_verdict_as_first_class_evidence(
    tmp_path: Path,
) -> None:
    entry = _entry(tmp_path, bundle=_bundle_with_fixture())
    index = CapabilityIndex(entries=(entry,))
    environment = _VerificationEnvironment(index, verdict="fail")
    store = VerificationStore(tmp_path / "records")

    record = verify_capability("demo.capability", store=store, environment=environment)

    assert record.equivalence[0].verdict == "fail"
    assert store.read(entry.capability_id, entry.content_hash, record.environment_key) == record
    assert record.worker_parameters == ()
    assert "transport-request-id" not in json.dumps(record.model_dump(mode="json"))
    assert environment.executor.inspection_calls == 1
    assert environment.events.index("inspect") < environment.events.index("fixture")
    assert environment.fixture_calls == 1


def test_unsupported_equivalence_writes_no_record_and_runs_nothing(tmp_path: Path) -> None:
    entry = _entry(tmp_path, bundle=_bundle_with_fixture(method="set_membership"))
    index = CapabilityIndex(entries=(entry,))
    environment = _VerificationEnvironment(index)
    store = VerificationStore(tmp_path / "records")

    with pytest.raises(OrganelleContractError) as captured:
        verify_capability("demo.capability", store=store, environment=environment)

    assert captured.value.code == "capability.equivalence_method_unsupported"
    assert store.records_for(entry.capability_id) == ()
    assert environment.executor.inspection_calls == 0
    assert environment.fixture_calls == 0


def test_asset_observation_failure_writes_no_record_or_fixture_evidence(tmp_path: Path) -> None:
    entry = _entry(tmp_path, bundle=_bundle_with_fixture())
    index = CapabilityIndex(entries=(entry,))
    environment = _VerificationEnvironment(index, fail_observation=True)
    store = VerificationStore(tmp_path / "records")

    with pytest.raises(OrganelleDependencyError) as captured:
        verify_capability("demo.capability", store=store, environment=environment)

    assert captured.value.code == "capability.asset_unresolvable"
    assert store.records_for(entry.capability_id) == ()
    assert environment.executor.inspection_calls == 0
    assert environment.fixture_calls == 0


def test_verification_rejects_provider_required_binding_before_fixture(
    tmp_path: Path,
) -> None:
    payload = _bundle_with_fixture().model_dump(mode="python", by_alias=True)
    contract = cast(dict[str, object], payload["contract"])
    binding = cast(dict[str, object], contract["binding"])
    binding["result_codec"] = "artifact"
    bundle = CapabilityBundle.model_validate(payload)
    entry = _entry(tmp_path, bundle=bundle)
    environment = _VerificationEnvironment(CapabilityIndex(entries=(entry,)))
    store = VerificationStore(tmp_path / "records")

    with pytest.raises(OrganelleContractError) as captured:
        verify_capability("demo.capability", store=store, environment=environment)

    assert captured.value.code == "capability.execution_provider_required"
    assert environment.executor.inspection_calls == 0
    assert environment.fixture_calls == 0
    assert environment.events == []
    assert store.records_for(entry.capability_id) == ()


@pytest.mark.parametrize("provider_shape", ["read_final_write", "writer"])
def test_verification_provider_boundaries_stop_before_worker_inspection(
    tmp_path: Path,
    provider_shape: str,
) -> None:
    payload = _bundle_with_fixture().model_dump(mode="python", by_alias=True)
    contract = cast(dict[str, object], payload["contract"])
    binding = cast(dict[str, object], contract["binding"])
    if provider_shape == "read_final_write":
        contract["stage"] = "read"
        contract["output_kind"] = "data"
        contract["output_modalities"] = ("worker_data",)
        binding["parameters"] = ({"name": "output", "codec": "path"},)
        capability_id = "demo.capability"
    else:
        capability_id = "demo.write"
        cast(dict[str, object], payload["capability"])["id"] = capability_id
        contract.update(
            {
                "operation_id": capability_id,
                "stage": "consume",
                "input_kind": "result",
                "output_kind": "result",
            }
        )
    bundle = CapabilityBundle.model_validate(payload)
    if capability_id == "demo.capability":
        entry = _entry(tmp_path, bundle=bundle)
    else:
        identity = _execution_identity(
            capability_id=capability_id,
            callable_locator=cast(str, bundle.contract.callable_locator),
        )
        entry = CapabilityEntry(
            capability_id=capability_id,
            content_hash=_BUNDLE_HASH,
            bundle_root=tmp_path,
            bundle=bundle,
            origins=(
                CapabilityOrigin(
                    channel="local",
                    source_path=str(tmp_path / "capability.toml"),
                    search_root=str(tmp_path),
                ),
            ),
            execution_identity=identity,
        )
    environment = _VerificationEnvironment(CapabilityIndex(entries=(entry,)))

    with pytest.raises(OrganelleContractError) as captured:
        verify_capability(
            capability_id,
            store=VerificationStore(tmp_path / "records"),
            environment=environment,
        )

    assert captured.value.code == "capability.execution_provider_required"
    assert environment.executor.inspection_calls == 0
    assert environment.executor.invocation_calls == 0
    assert environment.events == []


def test_external_verification_requires_provider_before_worker_inspection(
    tmp_path: Path,
) -> None:
    payload = _bundle().model_dump(mode="python", by_alias=True)
    cast(dict[str, object], payload["capability"])["implementation"] = "external"
    cast(dict[str, object], payload["contract"])["dependencies"] = (
        {"kind": "executable", "name": "demo_tool"},
    )
    payload["probe"] = ({"dependency": "demo_tool", "help_argv": ("--help",)},)
    entry = _entry(
        tmp_path,
        implementation="external",
        bundle=CapabilityBundle.model_validate(payload),
    )
    environment = _VerificationEnvironment(CapabilityIndex(entries=(entry,)))

    with pytest.raises(OrganelleContractError) as captured:
        verify_capability(
            entry.capability_id,
            store=VerificationStore(tmp_path / "records"),
            environment=environment,
        )

    assert captured.value.code == "capability.execution_provider_required"
    assert environment.executor.inspection_calls == 0
    assert environment.executor.invocation_calls == 0
    assert environment.events == []


def test_admit_capabilities_changes_registry_membership_only_after_record_exists(
    tmp_path: Path,
) -> None:
    payload = _bundle().model_dump(mode="python", by_alias=True)
    contract = cast(dict[str, object], payload["contract"])
    binding = cast(dict[str, object], contract["binding"])
    binding["result_codec"] = "canonical_json"
    entry = _entry(tmp_path, bundle=CapabilityBundle.model_validate(payload))
    index = CapabilityIndex(entries=(entry,))
    store = VerificationStore(tmp_path / "records")

    rejected = admit_capabilities(index, store=store)
    assert rejected.list() == ()
    rejection = rejected.rejection(entry.capability_id)
    assert rejection is not None
    assert rejection.code == "capability.verification_missing"

    environment = EnvironmentSnapshot(platform=current_platform_identity(), observed_environment={})
    fixture_hash = hash_fixture_dataset(tmp_path)
    record = _record(equivalence=()).model_copy(
        update={
            "fixture_dataset_hash": fixture_hash,
            "environment_key": compute_environment_key(environment),
            "platform": environment.platform,
        }
    )
    store.write(record)
    admitted = admit_capabilities(index, store=store)

    assert tuple(item.capability_id for item in admitted.list()) == ("demo.capability",)
    assert admitted.list()[0].worker_parameters == ()


def test_admission_rejects_frozen_worker_schema_drift(tmp_path: Path) -> None:
    payload = _bundle().model_dump(mode="python", by_alias=True)
    contract = cast(dict[str, object], payload["contract"])
    binding = cast(dict[str, object], contract["binding"])
    binding["result_codec"] = "canonical_json"
    entry = _entry(tmp_path, bundle=CapabilityBundle.model_validate(payload))
    environment = EnvironmentSnapshot(platform=current_platform_identity(), observed_environment={})
    record = _record(equivalence=()).model_copy(
        update={
            "fixture_dataset_hash": hash_fixture_dataset(tmp_path),
            "environment_key": compute_environment_key(environment),
            "platform": environment.platform,
            "parameter_schema": {
                "type": "object",
                "properties": {"injected": {"type": "string"}},
                "additionalProperties": False,
            },
        }
    )
    store = VerificationStore(tmp_path / "records")
    store.write(record)

    admitted = admit_capabilities(CapabilityIndex(entries=(entry,)), store=store)

    rejection = admitted.rejection(entry.capability_id)
    assert rejection is not None
    assert rejection.code == "capability.schema_drift"
    assert admitted.list() == ()


def test_resolve_rejects_provider_bound_frozen_schema_without_executor_calls(
    tmp_path: Path,
) -> None:
    entry = _admitted_entry(tmp_path).model_copy(
        update={
            "parameter_schema": {
                "type": "object",
                "properties": {"output": {"type": "string"}},
                "additionalProperties": False,
            }
        }
    )
    executor = _VerificationExecutor([])
    registry = OperationRegistry(
        capability_source=CapabilityIndex(entries=(entry,)).binding_source(executor=executor)
    )

    with pytest.raises(OrganelleContractError) as captured:
        registry.require(entry.capability_id)

    assert captured.value.code == "capability.execution_provider_required"
    assert executor.inspection_calls == 0
    assert executor.invocation_calls == 0


def _canonical_result(*, value: int = 1) -> OrganelleResult:
    return OrganelleResult(
        operation_id="demo.capability",
        scope="none",
        status="ok",
        summary_text="lazy capability executed",
        metrics=FrozenMap({"value": value}),
    )


def _admitted_entry(
    tmp_path: Path,
    *,
    origin: str = "local",
    callable_locator: str = "demo.impl:run",
) -> CapabilityEntry:
    bundle = _bundle(callable_locator=callable_locator)
    worker_parameters = (
        WorkerParameter(
            name="value",
            kind="keyword_only",
            required=False,
            default=1,
            annotation="int",
        ),
    )
    result_codec = ResultCodec.CANONICAL if origin == "core" else ResultCodec.CANONICAL_JSON
    bundle = bundle.model_copy(
        update={
            "contract": bundle.contract.model_copy(
                update={
                    "binding": PythonBindingSpec(
                        argument_mode=ArgumentMode.NAMED_PARAMETERS,
                        parameters=(ParameterBindingSpec(name="value", codec=ParameterCodec.JSON),),
                        result_codec=result_codec,
                    )
                }
            )
        }
    )
    frozen_schema = (
        parameter_schema(bind_python_capability(bundle, _canonical_result, frozen_schema=None))
        if origin == "core"
        else worker_parameter_schema(bundle, worker_parameters)
    )
    entry = _entry(tmp_path, bundle=bundle)
    return entry.model_copy(
        update={
            "origins": (
                CapabilityOrigin(
                    channel=origin,
                    source_path=str(tmp_path / "capability.toml"),
                    search_root=str(tmp_path),
                ),
            ),
            "execution_identity": None if origin == "core" else entry.execution_identity,
            "status": "admitted",
            "parameter_schema": frozen_schema,
            "verification_environment_key": compute_environment_key(_environment()),
            "worker_parameters": None if origin == "core" else worker_parameters,
        }
    )


def _write_lazy_module(
    tmp_path: Path, *, drift: bool = False, import_marker: Path | None = None
) -> None:
    signature = "value: str = '1'" if drift else "value: int = 1"
    import_marker_line = (
        f"Path({str(import_marker)!r}).write_text('imported', encoding='utf-8')\n"
        if import_marker is not None
        else ""
    )
    marker_line = (
        f"    Path({str(tmp_path / 'executed.txt')!r}).write_text('ran', encoding='utf-8')\n"
        if drift
        else ""
    )
    (tmp_path / "demo_impl.py").write_text(
        "from pathlib import Path\n"
        "from organelleverse.core.result import OrganelleResult\n"
        f"{import_marker_line}"
        f"def run(*, {signature}) -> OrganelleResult:\n"
        f"{marker_line}"
        "    return OrganelleResult(operation_id='demo.capability', scope='none', "
        "status='ok', summary_text='lazy capability executed', metrics={'value': value})\n",
        encoding="utf-8",
    )


def _canonical_data_result(data: OrganelleData) -> OrganelleResult:
    return OrganelleResult(
        operation_id="demo.capability",
        scope="none",
        status="ok",
        summary_text=f"received {data.modality}",
    )


def _admitted_data_entry(tmp_path: Path) -> CapabilityEntry:
    bundle = _bundle()
    bundle = bundle.model_copy(
        update={
            "contract": bundle.contract.model_copy(
                update={
                    "input_kind": CoreKind.DATA,
                    "input_modalities": ("matrix",),
                    "binding": PythonBindingSpec(
                        argument_mode=ArgumentMode.CANONICAL_CORE,
                        result_codec=ResultCodec.CANONICAL,
                    ),
                    "callable_locator": "demo_data_impl:run",
                }
            )
        }
    )
    bound = bind_python_capability(bundle, _canonical_data_result, frozen_schema=None)
    return _entry(tmp_path, bundle=bundle).model_copy(
        update={
            "origins": (
                CapabilityOrigin(
                    channel="core",
                    source_path=str(tmp_path / "capability.toml"),
                    search_root=str(tmp_path),
                ),
            ),
            "execution_identity": None,
            "status": "admitted",
            "parameter_schema": parameter_schema(bound),
            "verification_environment_key": compute_environment_key(_environment()),
        }
    )


def _matrix_contract() -> DataContract:
    return DataContract(
        modality="matrix",
        schema_version="matrix.v1",
        validator=lambda data: data,
        schema_provider=lambda: {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {"modality": {"type": "string", "const": "matrix"}},
            "required": ["modality"],
        },
    )


def test_registry_lists_describes_and_serves_frozen_schema_without_importing_implementation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delitem(sys.modules, "demo_impl", raising=False)
    entry = _admitted_entry(tmp_path, callable_locator="demo_impl:run")
    _write_lazy_module(tmp_path)
    monkeypatch.setattr(sys, "path", [str(tmp_path), *sys.path])
    registry = OperationRegistry()
    registry.attach_capability_source(
        CapabilityIndex(entries=(entry,)).binding_source(
            trust_store=TrustStore(tmp_path / "trust.json")
        )
    )

    assert tuple(spec.operation_id for spec in registry.list()) == ("demo.capability",)
    assert registry.describe("demo.capability").title == "Demo capability"
    assert registry.parameter_schema("demo.capability") == entry.parameter_schema
    invocation = json.dumps(registry.invocation_schema("demo.capability"), sort_keys=True)
    assert '"const": "demo.capability"' in invocation
    assert "demo_impl" not in sys.modules


@pytest.mark.parametrize("channels", (("local",), ("core", "local")))
def test_untrusted_non_pure_core_is_rejected_before_provider_and_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    channels: tuple[str, ...],
) -> None:
    monkeypatch.delitem(sys.modules, "demo_impl", raising=False)
    bundle_root = tmp_path / "bundle"
    bundle_root.mkdir()
    code_root = bundle_root / "code"
    code_root.mkdir()
    package_root = code_root / "demo_impl"
    package_root.mkdir()
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    entry = _admitted_entry(bundle_root, callable_locator="demo_impl.demo_impl:run")
    entry = entry.model_copy(
        update={
            "origins": tuple(
                CapabilityOrigin(
                    channel=channel,
                    source_path=str(bundle_root / channel / "capability.toml"),
                    search_root=str(bundle_root / channel),
                )
                for channel in channels
            )
        }
    )
    _write_lazy_module(package_root)
    content_hash = hash_bundle(bundle_root)
    identity = inspect_bundle_code(
        bundle_root,
        capability_id=entry.capability_id,
        bundle_content_hash=content_hash,
        callable_locator="demo_impl.demo_impl:run",
    )
    entry = entry.model_copy(update={"content_hash": content_hash, "execution_identity": identity})
    monkeypatch.setattr(sys, "path", [str(code_root), *sys.path])
    registry = OperationRegistry()
    registry.attach_capability_source(
        CapabilityIndex(entries=(entry,)).binding_source(
            trust_store=TrustStore(tmp_path / "trust.json")
        )
    )

    with pytest.raises(OrganellePermissionError) as captured:
        registry.invoke("demo.capability", input=None, parameters={})

    assert captured.value.code == "capability.untrusted"
    assert "demo_impl" not in sys.modules


@pytest.mark.parametrize("channels", (("local",), ("core", "local")))
def test_trusted_non_pure_core_requires_provider_before_ambient_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    channels: tuple[str, ...],
) -> None:
    monkeypatch.delitem(sys.modules, "demo_impl", raising=False)
    bundle_root = tmp_path / "bundle"
    bundle_root.mkdir()
    code_root = bundle_root / "code"
    code_root.mkdir()
    package_root = code_root / "demo_impl"
    package_root.mkdir()
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    marker = package_root / "imported.txt"
    entry = _admitted_entry(bundle_root, callable_locator="demo_impl.demo_impl:run")
    entry = entry.model_copy(
        update={
            "bundle": entry.bundle.model_copy(
                update={
                    "contract": entry.bundle.contract.model_copy(
                        update={
                            "binding": entry.bundle.contract.binding.model_copy(
                                update={"result_codec": ResultCodec.CANONICAL}
                            )
                        }
                    )
                }
            )
        }
    )
    entry = entry.model_copy(
        update={
            "origins": tuple(
                CapabilityOrigin(
                    channel=channel,
                    source_path=str(bundle_root / channel / "capability.toml"),
                    search_root=str(bundle_root / channel),
                )
                for channel in channels
            )
        }
    )
    _write_lazy_module(package_root, import_marker=marker)
    monkeypatch.setattr(sys, "path", [str(code_root), *sys.path])
    store = TrustStore(tmp_path / "trust.json")
    assert entry.execution_identity is not None
    trust(entry.capability_id, entry.execution_identity, store=store)
    registry = OperationRegistry()
    registry.attach_capability_source(
        CapabilityIndex(entries=(entry,)).binding_source(trust_store=store)
    )

    with pytest.raises(OrganelleContractError) as captured:
        registry.invoke("demo.capability", input=None, parameters={})

    assert captured.value.code == "capability.execution_provider_required"
    assert not marker.exists()
    assert "demo_impl" not in sys.modules


def test_schema_drift_is_detected_before_the_scientific_callable_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delitem(sys.modules, "demo_impl", raising=False)
    entry = _admitted_entry(tmp_path, origin="core", callable_locator="demo_impl:run")
    _write_lazy_module(tmp_path, drift=True)
    monkeypatch.setattr(sys, "path", [str(tmp_path), *sys.path])
    registry = OperationRegistry()
    registry.attach_capability_source(CapabilityIndex(entries=(entry,)).binding_source())

    with pytest.raises(OrganelleContractError) as captured:
        registry.invoke("demo.capability", input=None, parameters={})

    assert captured.value.code == "capability.schema_drift"
    assert not (tmp_path / "executed.txt").exists()


def test_composite_contract_remains_describable_while_invocation_reports_unsupported(
    tmp_path: Path,
) -> None:
    decision = evaluate_admission(
        _entry(tmp_path, implementation="composite"),
        record=None,
        current_environment=_environment(),
        current_fixture_hash=_FIXTURE_HASH,
        reference_identities={},
    )
    entry = _entry(tmp_path, implementation="composite").model_copy(
        update={"status": "unsupported", "diagnostic": decision.diagnostic}
    )
    registry = OperationRegistry()
    registry.attach_capability_source(CapabilityIndex(entries=(entry,)).binding_source())

    assert registry.describe("demo.capability").operation_id == "demo.capability"
    with pytest.raises(OrganelleContractError) as captured:
        registry.invoke("demo.capability", input=None, parameters={})
    assert captured.value.code == "capability.implementation_not_supported"
    assert captured.value.as_dict()["details"]["capability_id"] == "demo.capability"


def test_registry_preserves_rejection_code_and_structured_bundle_identity(tmp_path: Path) -> None:
    entry = _entry(tmp_path)
    diagnostic = CapabilityDiagnostic(
        code="capability.verification_missing",
        message="capability has no verification record",
        capability_id=entry.capability_id,
        origins=entry.origins,
    )
    rejected = entry.model_copy(update={"status": "rejected", "diagnostic": diagnostic})
    registry = OperationRegistry(
        capability_source=CapabilityIndex(entries=(rejected,)).binding_source()
    )

    with pytest.raises(OrganelleContractError) as captured:
        registry.require(entry.capability_id)

    assert captured.value.code == "capability.verification_missing"
    details = captured.value.as_dict()["details"]
    assert details["capability_id"] == entry.capability_id
    assert details["origins"][0]["channel"] == "local"

    for schema_reader in (registry.parameter_schema, registry.invocation_schema):
        with pytest.raises(OrganelleContractError) as schema_error:
            schema_reader(entry.capability_id)
        assert schema_error.value.code == "capability.verification_missing"


def test_registry_reports_capability_conflict_instead_of_unknown(tmp_path: Path) -> None:
    origin = _entry(tmp_path).origins[0]
    conflict = CapabilityConflict(
        capability_id="demo.capability",
        members=(
            CapabilityConflictMember(
                content_hash="sha256:" + "1" * 64,
                origin_channel=origin.channel,
                source_path=origin.source_path,
            ),
            CapabilityConflictMember(
                content_hash="sha256:" + "2" * 64,
                origin_channel="project",
                source_path=str(tmp_path / "other" / "capability.toml"),
            ),
        ),
    )
    registry = OperationRegistry(
        capability_source=CapabilityIndex(conflict_items=(conflict,)).binding_source()
    )

    with pytest.raises(OrganelleContractError) as captured:
        registry.describe("demo.capability")

    assert captured.value.code == "capability.conflict"
    assert len(captured.value.as_dict()["details"]["members"]) == 2


def test_pre_migration_authoritative_binding_is_not_disabled_by_bundle_conflict(
    tmp_path: Path,
) -> None:
    entry = _admitted_entry(tmp_path, origin="core")
    conflict = CapabilityConflict(
        capability_id=entry.capability_id,
        members=(
            CapabilityConflictMember(
                content_hash="sha256:" + "1" * 64,
                origin_channel="local",
                source_path=str(tmp_path / "one" / "capability.toml"),
            ),
            CapabilityConflictMember(
                content_hash="sha256:" + "2" * 64,
                origin_channel="project",
                source_path=str(tmp_path / "two" / "capability.toml"),
            ),
        ),
    )
    registry = OperationRegistry()
    binding = registry.register(entry.bundle.contract, _canonical_result)
    authoritative = cast(set[str], vars(registry)["_authoritative_operation_ids"])
    authoritative.add(entry.capability_id)
    registry.attach_capability_source(CapabilityIndex(conflict_items=(conflict,)).binding_source())

    assert registry.require(entry.capability_id) is binding
    assert registry.describe(entry.capability_id) == binding.spec
    assert entry.capability_id in {spec.operation_id for spec in registry.list()}
    assert registry.parameter_schema(entry.capability_id)
    assert registry.invocation_schema(entry.capability_id)


def test_plain_registry_refuses_a_second_implementation_for_a_source_id(tmp_path: Path) -> None:
    entry = _admitted_entry(tmp_path, origin="core")
    registry = OperationRegistry()
    registry.register(entry.bundle.contract, _canonical_result)

    with pytest.raises(OrganelleContractError) as captured:
        registry.attach_capability_source(CapabilityIndex(entries=(entry,)).binding_source())

    assert captured.value.code == "contract.duplicate_operation_id"


def test_lazy_binding_reuses_registry_data_contract_checks_before_scientific_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "demo_data_impl", raising=False)
    monkeypatch.setattr(sys, "path", [str(tmp_path), *sys.path])
    marker = tmp_path / "data-executed.txt"
    (tmp_path / "demo_data_impl.py").write_text(
        "from pathlib import Path\n"
        "from organelleverse.core.data import OrganelleData\n"
        "from organelleverse.core.result import OrganelleResult\n"
        "def run(data: OrganelleData) -> OrganelleResult:\n"
        f"    Path({str(marker)!r}).write_text('ran', encoding='utf-8')\n"
        "    return OrganelleResult(operation_id='demo.capability', scope='none', "
        "status='ok', summary_text='ran')\n",
        encoding="utf-8",
    )
    entry = _admitted_data_entry(tmp_path)
    registry = OperationRegistry(
        data_contracts=(_matrix_contract(),),
        capability_source=CapabilityIndex(entries=(entry,)).binding_source(),
    )

    with pytest.raises(OrganelleInputError) as captured:
        registry.invoke(
            "demo.capability",
            input=OrganelleData(modality="alignment"),
            parameters={},
        )

    assert captured.value.code == "input.unsupported_operation_modality"
    assert not marker.exists()


def test_suggestion_validation_uses_an_admitted_lazy_targets_frozen_schema_without_import(
    tmp_path: Path,
) -> None:
    target = _admitted_entry(tmp_path, origin="core")
    target_bundle = target.bundle.model_copy(
        update={
            "capability": target.bundle.capability.model_copy(update={"id": "demo.target"}),
            "contract": target.bundle.contract.model_copy(
                update={"operation_id": "demo.target", "callable_locator": "never_import:run"}
            ),
        }
    )
    target = target.model_copy(update={"capability_id": "demo.target", "bundle": target_bundle})
    source = CapabilityIndex(entries=(target,)).binding_source()
    registry = OperationRegistry()

    def suggest(*, value: int = 1) -> OrganelleResult:
        return OrganelleResult(
            operation_id="demo.suggester",
            scope="none",
            status="ok",
            suggested_operations=(
                OperationSuggestion(
                    operation_id="demo.target",
                    reason_code="test.followup",
                    parameter_changes=FrozenMap({"value": value}),
                ),
            ),
        )

    suggester_spec = target_bundle.contract.model_copy(
        update={"operation_id": "demo.suggester", "callable_locator": "tests:suggest"}
    )
    registry.register(suggester_spec, suggest)
    registry.attach_capability_source(source)

    result = registry.invoke("demo.suggester", input=None, parameters={})

    assert isinstance(result, OrganelleResult)
    assert result.suggested_operations[0].operation_id == "demo.target"
    assert "never_import" not in sys.modules
