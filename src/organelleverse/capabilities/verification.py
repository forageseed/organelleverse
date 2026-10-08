from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
import platform as platform_module
import re
import sys
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, cast
from uuid import uuid4

from pydantic import ConfigDict, Field, JsonValue

from organelleverse.core.errors import OrganelleContractError, OrganelleError
from organelleverse.operations.spec import (
    CoreKind,
    ParameterCodec,
    RewriteSpec,
    StrictSpecModel,
)

from .code_identity import ExecutionIdentity
from .hashing import hash_fixture_dataset
from .index import CapabilityEntry, CapabilityIndex
from .models import EquivalenceMethod, FixtureSpec, ImplementationKind
from .worker import OneShotBundleWorkerExecutor
from .worker_contracts import WorkerParameter

if TYPE_CHECKING:
    from .worker import BundleWorkerExecutor

_HASH_PATTERN = r"^sha256:[0-9a-f]{64}$"
_HASH_RE = re.compile(_HASH_PATTERN)
_ENVIRONMENT_KEY_RE = _HASH_RE
_RECORD_SCHEMA = "organelleverse.verification.v2"
_FIXTURE_EVALUATOR_VERSION = "organelleverse.fixture-evaluator.v2"


class PlatformIdentity(StrictSpecModel):
    os: str = Field(min_length=1)
    arch: str = Field(min_length=1)
    libc: str = Field(min_length=1)
    python: str = Field(min_length=1)


class FastFileKey(StrictSpecModel):
    size: int = Field(ge=0)
    mtime_ns: int = Field(ge=0)


class ObservedDependency(StrictSpecModel):
    kind: Literal["executable", "asset"]
    content_hash: str = Field(pattern=_HASH_PATTERN)
    realpath: str = ""
    fast_key: FastFileKey | None = None
    observed_version: str = ""
    satisfied_requires: tuple[str, ...] = ()


class AdapterRequirement(StrictSpecModel):
    requires: tuple[str, ...] = ()


class EnvironmentSnapshot(StrictSpecModel):
    platform: PlatformIdentity
    observed_environment: dict[str, ObservedDependency]


class CandidateEvidence(StrictSpecModel):
    bundle_content_hash: str = Field(pattern=_HASH_PATTERN)
    environment_key: str = Field(pattern=_HASH_PATTERN)
    produced_hash: str = Field(pattern=_HASH_PATTERN)


class ReferenceEvidence(StrictSpecModel):
    capability_id: str = Field(min_length=1)
    bundle_content_hash: str = Field(pattern=_HASH_PATTERN)
    environment_key: str = Field(pattern=_HASH_PATTERN)
    produced_hash: str = Field(pattern=_HASH_PATTERN)


class EquivalenceEvidence(StrictSpecModel):
    case: str = Field(min_length=1)
    method: Literal["exact", "numeric_tolerance"]
    tolerance: float | None = None
    evaluator_version: str = Field(min_length=1)
    verdict: Literal["pass", "fail"]
    fixture_dataset_hash: str = Field(pattern=_HASH_PATTERN)
    candidate: CandidateEvidence
    reference: ReferenceEvidence | None = None
    determinism_verdict: Literal["pass", "fail", "not_checked"] = "not_checked"
    determinism_produced_hash: str | None = Field(default=None, pattern=_HASH_PATTERN)
    rewrite_verdict: Literal["pass", "fail", "not_checked"] = "not_checked"
    rewrite_reference_hash: str | None = Field(default=None, pattern=_HASH_PATTERN)
    rewrite_original_callable: str | None = None


class VerificationRecord(StrictSpecModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        revalidate_instances="always",
        populate_by_name=True,
    )

    schema_version: Literal["organelleverse.verification.v2"] = Field(
        default=_RECORD_SCHEMA, alias="schema"
    )
    protocol_version: str
    capability_id: str
    bundle_content_hash: str = Field(pattern=_HASH_PATTERN)
    execution_identity: ExecutionIdentity | None = None
    fixture_dataset_hash: str = Field(pattern=_HASH_PATTERN)
    adapter_requirements: dict[str, AdapterRequirement]
    observed_environment: dict[str, ObservedDependency]
    environment_key: str = Field(pattern=_HASH_PATTERN)
    equivalence: tuple[EquivalenceEvidence, ...]
    platform: PlatformIdentity
    worker_parameters: tuple[WorkerParameter, ...]
    parameter_schema: dict[str, JsonValue]
    verified_at: str
    organelleverse_version: str


def current_platform_identity() -> PlatformIdentity:
    libc_name, libc_version = platform_module.libc_ver()
    libc = f"{libc_name}-{libc_version}" if libc_name or libc_version else "unknown"
    return PlatformIdentity(
        os=platform_module.system().lower() or sys.platform,
        arch=platform_module.machine().lower() or "unknown",
        libc=libc,
        python=platform_module.python_version(),
    )


def compute_environment_key(snapshot: EnvironmentSnapshot) -> str:
    dependencies = {
        name: {
            "kind": item.kind,
            "content_hash": item.content_hash,
            "satisfied_requires": list(item.satisfied_requires),
        }
        for name, item in sorted(snapshot.observed_environment.items())
    }
    payload = {
        "platform": snapshot.platform.model_dump(mode="json"),
        "observed_environment": dependencies,
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(canonical).hexdigest()}"


def _hash_parts(content_hash: str) -> tuple[str, str]:
    if _HASH_RE.fullmatch(content_hash) is None:
        raise ValueError(f"invalid sha256 content identity: {content_hash!r}")
    algorithm, digest = content_hash.split(":", 1)
    return algorithm, digest


class VerificationStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def path_for(
        self,
        capability_id: str,
        bundle_content_hash: str,
        environment_key: str,
    ) -> Path:
        algorithm, bundle_digest = _hash_parts(bundle_content_hash)
        _, environment_digest = _hash_parts(environment_key)
        if not re.fullmatch(r"[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*", capability_id):
            raise ValueError(f"invalid capability id for verification path: {capability_id!r}")
        return self.root / capability_id / algorithm / bundle_digest / f"{environment_digest}.json"

    def write(self, record: VerificationRecord) -> Path:
        path = self.path_for(
            record.capability_id,
            record.bundle_content_hash,
            record.environment_key,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        payload = (
            json.dumps(
                record.model_dump(mode="json", by_alias=True),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        )
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
        return path

    def read(
        self,
        capability_id: str,
        bundle_content_hash: str,
        environment_key: str,
    ) -> VerificationRecord | None:
        path = self.path_for(capability_id, bundle_content_hash, environment_key)
        if not path.is_file():
            return None
        try:
            payload = cast(object, json.loads(path.read_text(encoding="utf-8")))
            if not isinstance(payload, dict):
                return None
            typed_payload = cast(dict[str, object], payload)
            if typed_payload.get("schema") != _RECORD_SCHEMA:
                return None
            return VerificationRecord.model_validate(typed_payload)
        except (OSError, ValueError):
            return None

    def records_for(self, capability_id: str) -> tuple[VerificationRecord, ...]:
        capability_root = self.root / capability_id
        if not capability_root.is_dir():
            return ()
        records: list[VerificationRecord] = []
        for path in sorted(capability_root.glob("sha256/*/*.json")):
            try:
                payload = cast(object, json.loads(path.read_text(encoding="utf-8")))
                if isinstance(payload, dict):
                    typed_payload = cast(dict[str, object], payload)
                    if typed_payload.get("schema") == _RECORD_SCHEMA:
                        records.append(VerificationRecord.model_validate(typed_payload))
            except (OSError, ValueError):
                continue
        return tuple(sorted(records, key=lambda item: item.environment_key))


def default_verification_store() -> VerificationStore:
    home = Path(os.environ.get("ORGANELLEVERSE_HOME", Path.home() / ".organelleverse"))
    return VerificationStore(home / "verifications")


class VerificationEnvironment(Protocol):
    @property
    def index(self) -> CapabilityIndex: ...

    @property
    def executor(self) -> BundleWorkerExecutor: ...

    def observe_environment(self, entry: CapabilityEntry) -> EnvironmentSnapshot: ...

    def evaluate_fixture(
        self,
        entry: CapabilityEntry,
        fixture: FixtureSpec,
        *,
        executor: BundleWorkerExecutor,
        environment_key: str,
        fixture_dataset_hash: str,
    ) -> EquivalenceEvidence: ...


def _is_pure_core_native(entry: CapabilityEntry) -> bool:
    return entry.bundle.capability.implementation is ImplementationKind.NATIVE and all(
        origin.channel == "core" for origin in entry.origins
    )


def _resolve_core_callable(capability_id: str, locator: str) -> Callable[..., object]:
    module_name, attribute_name = locator.split(":", 1)
    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        raise OrganelleContractError(
            code="capability.core_module_unresolvable",
            message=f"core capability implementation module cannot be imported: {capability_id}",
            details={
                "capability_id": capability_id,
                "callable_locator": locator,
                "reason": str(error),
            },
        ) from error
    implementation = getattr(module, attribute_name, None)
    if not callable(implementation):
        raise OrganelleContractError(
            code="capability.binding_invalid",
            message=f"callable locator does not resolve to a callable: {locator}",
            details={"capability_id": capability_id, "callable_locator": locator},
        )
    return implementation


def verify_capability(
    capability_id: str,
    *,
    store: VerificationStore,
    environment: VerificationEnvironment,
) -> VerificationRecord:
    entry = environment.index.describe(capability_id)
    if entry.bundle.capability.implementation is ImplementationKind.COMPOSITE:
        raise OrganelleContractError(
            code="capability.implementation_not_supported",
            message=f"composite capability is not executable in v1: {capability_id}",
        )
    unsupported = [
        fixture.case
        for fixture in entry.bundle.fixtures
        if fixture.equivalence is EquivalenceMethod.SET_MEMBERSHIP
    ]
    if unsupported:
        raise OrganelleContractError(
            code="capability.equivalence_method_unsupported",
            message="set_membership equivalence is declared but not implemented in v1",
            details={"capability_id": capability_id, "cases": unsupported},
        )
    execution_identity = entry.execution_identity
    core_native = _is_pure_core_native(entry)
    if execution_identity is None and not core_native:
        raise OrganelleContractError(
            code="capability.execution_provider_required",
            message=f"capability has no content-addressed execution provider: {capability_id}",
            details={"capability_id": capability_id},
        )
    if core_native:
        implementation = _resolve_core_callable(
            capability_id, cast(str, entry.bundle.contract.callable_locator)
        )
        from organelleverse.operations.python_binding import bind_python_capability
        from organelleverse.operations.schemas import parameter_schema

        bound = bind_python_capability(entry.bundle, implementation, frozen_schema=None)
        worker_parameters: tuple[WorkerParameter, ...] = ()
        schema = cast(dict[str, JsonValue], parameter_schema(bound))
        fixture_dataset_hash = hash_fixture_dataset(entry.bundle_root)
        snapshot = environment.observe_environment(entry)
        environment_key = compute_environment_key(snapshot)
    else:
        assert execution_identity is not None, (
            "non-core-native capability reached verification without an execution identity"
        )
        from organelleverse.operations.python_binding import validate_worker_contract

        validate_worker_contract(entry.bundle)
        fixture_dataset_hash = hash_fixture_dataset(entry.bundle_root)
        snapshot = environment.observe_environment(entry)
        environment_key = compute_environment_key(snapshot)
        inspection = environment.executor.inspect(entry)
        if inspection.execution_identity != execution_identity:
            raise OrganelleContractError(
                code="capability.worker_identity_mismatch",
                message="worker inspection identity does not match the discovered bundle",
                details={
                    "capability_id": capability_id,
                    "expected_execution_digest": execution_identity.digest,
                    "actual_execution_digest": inspection.execution_identity.digest,
                },
            )
        from organelleverse.operations.python_binding import worker_parameter_schema

        worker_parameters = inspection.parameters
        schema = cast(
            dict[str, JsonValue],
            worker_parameter_schema(entry.bundle, worker_parameters),
        )
    verified_entry = entry.model_copy(
        update={
            "parameter_schema": schema,
            "worker_parameters": worker_parameters,
        }
    )
    equivalence = tuple(
        environment.evaluate_fixture(
            verified_entry,
            fixture,
            executor=environment.executor,
            environment_key=environment_key,
            fixture_dataset_hash=fixture_dataset_hash,
        )
        for fixture in entry.bundle.fixtures
    )
    requirements = {
        probe.dependency: AdapterRequirement(requires=probe.requires)
        for probe in entry.bundle.probes
    }
    record = VerificationRecord(
        protocol_version="1.0",
        capability_id=capability_id,
        bundle_content_hash=entry.content_hash,
        execution_identity=execution_identity,
        fixture_dataset_hash=fixture_dataset_hash,
        adapter_requirements=requirements,
        observed_environment=snapshot.observed_environment,
        environment_key=environment_key,
        equivalence=equivalence,
        platform=snapshot.platform,
        worker_parameters=worker_parameters,
        parameter_schema=schema,
        verified_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        organelleverse_version=_package_version(),
    )
    store.write(record)
    return record


def _package_version() -> str:
    try:
        return importlib.metadata.version("organelleverse")
    except importlib.metadata.PackageNotFoundError:
        return "0.0.1"


def _canonical_fixture_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _fixture_produced_hash(value: object) -> str:
    return f"sha256:{hashlib.sha256(_canonical_fixture_bytes(value)).hexdigest()}"


_VOLATILE_RESULT_FIELDS = frozenset({"object_id"})
_VOLATILE_PROVENANCE_FIELDS = frozenset(
    {"object_id", "run_id", "parameters_hash", "package_version", "git_commit"}
)


def strip_run_provenance(result_json: object) -> object:
    if not isinstance(result_json, dict):
        return result_json
    result = cast(dict[str, object], result_json)
    stripped = {key: value for key, value in result.items() if key not in _VOLATILE_RESULT_FIELDS}
    provenance = stripped.get("provenance")
    if isinstance(provenance, dict):
        provenance_dict = cast(dict[str, object], provenance)
        normalized_provenance = {
            key: value
            for key, value in provenance_dict.items()
            if key not in _VOLATILE_PROVENANCE_FIELDS
        }
        translations = normalized_provenance.get("path_translations")
        if isinstance(translations, list):
            translation_records = cast(list[object], translations)
            normalized_provenance["path_translations"] = [
                {
                    key: value
                    for key, value in cast(dict[str, object], record).items()
                    if key not in {"original_value", "translated_value"}
                }
                if isinstance(record, dict)
                else record
                for record in translation_records
            ]
        materializations = normalized_provenance.get("inline_materializations")
        if isinstance(materializations, list):
            materialization_records = cast(list[object], materializations)
            normalized_provenance["inline_materializations"] = [
                {
                    key: value
                    for key, value in cast(dict[str, object], record).items()
                    if key != "materialized_path"
                }
                if isinstance(record, dict)
                else record
                for record in materialization_records
            ]
        stripped["provenance"] = normalized_provenance
    return stripped


def _numeric_tolerance_equal(produced: object, expected: object, tolerance: float) -> bool:
    if isinstance(expected, bool) or isinstance(produced, bool):
        return produced is expected
    if isinstance(expected, (int, float)) and isinstance(produced, (int, float)):
        return abs(float(produced) - float(expected)) <= tolerance
    if isinstance(expected, dict) and isinstance(produced, dict):
        expected_dict = cast(dict[str, object], expected)
        produced_dict = cast(dict[str, object], produced)
        if set(expected_dict) != set(produced_dict):
            return False
        return all(
            _numeric_tolerance_equal(produced_dict[key], expected_dict[key], tolerance)
            for key in expected_dict
        )
    if isinstance(expected, list) and isinstance(produced, list):
        expected_list = cast(list[object], expected)
        produced_list = cast(list[object], produced)
        if len(expected_list) != len(produced_list):
            return False
        return all(
            _numeric_tolerance_equal(p, e, tolerance)
            for p, e in zip(produced_list, expected_list, strict=True)
        )
    return produced == expected


def _compare_json_values(
    produced: object,
    expected: object,
    *,
    method: Literal["exact", "numeric_tolerance"],
    tolerance: float | None,
) -> bool:
    produced_clean = strip_run_provenance(produced)
    expected_clean = strip_run_provenance(expected)
    if method == "exact":
        return _canonical_fixture_bytes(produced_clean) == _canonical_fixture_bytes(expected_clean)
    if tolerance is None:
        raise OrganelleContractError(
            code="capability.fixture_tolerance_missing",
            message="numeric_tolerance comparison requires a tolerance",
            details={},
        )
    return _numeric_tolerance_equal(produced_clean, expected_clean, tolerance)


def _resolve_fixture_file(
    entry: CapabilityEntry, relative: str, *, field: str, fixture_case: str, directory: bool = False
) -> Path:
    try:
        bundle_root = entry.bundle_root.resolve(strict=True)
    except OSError as error:
        raise OrganelleContractError(
            code="capability.fixture_file_missing",
            message=f"capability bundle root could not be read: {entry.bundle_root}",
            details={
                "capability_id": entry.capability_id,
                "case": fixture_case,
                "field": field,
                "reason": str(error),
            },
        ) from error
    candidate = bundle_root / relative
    if candidate.is_symlink():
        raise OrganelleContractError(
            code="capability.fixture_path_unsafe",
            message=f"fixture {field} path must not be a symbolic link: {relative}",
            details={
                "capability_id": entry.capability_id,
                "case": fixture_case,
                "field": field,
                "path": relative,
            },
        )
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise OrganelleContractError(
            code="capability.fixture_file_missing",
            message=f"fixture {field} file does not exist: {relative}",
            details={
                "capability_id": entry.capability_id,
                "case": fixture_case,
                "field": field,
                "path": relative,
            },
        ) from error
    matches_kind = resolved.is_dir() if directory else resolved.is_file()
    if not matches_kind or not resolved.is_relative_to(bundle_root):
        raise OrganelleContractError(
            code="capability.fixture_path_unsafe",
            message=(
                f"fixture {field} path escapes the bundle root or is not a regular file: {relative}"
            ),
            details={
                "capability_id": entry.capability_id,
                "case": fixture_case,
                "field": field,
                "path": relative,
            },
        )
    return resolved


def _resolve_fixture_parameters(entry: CapabilityEntry, fixture: FixtureSpec) -> dict[str, object]:
    bindings = {item.name: item for item in entry.bundle.contract.binding.parameters}
    resolved: dict[str, object] = {}
    for name, value in fixture.parameters.items():
        binding = bindings.get(name)
        if binding is None:
            raise OrganelleContractError(
                code="capability.fixture_parameter_undeclared",
                message=f"fixture parameter is not a declared binding parameter: {name}",
                details={
                    "capability_id": entry.capability_id,
                    "case": fixture.case,
                    "parameter": name,
                },
            )
        if binding.codec is ParameterCodec.DIRECTORY and binding.path_role == "input":
            if not isinstance(value, str):
                raise OrganelleContractError(
                    code="capability.fixture_parameter_invalid",
                    message=f"directory-codec fixture parameter must be a string: {name}",
                    details={
                        "capability_id": entry.capability_id,
                        "case": fixture.case,
                        "parameter": name,
                    },
                )
            resolved[name] = str(
                _resolve_fixture_file(
                    entry,
                    value,
                    field=f"parameters.{name}",
                    fixture_case=fixture.case,
                    directory=True,
                )
            )
        elif binding.codec is ParameterCodec.PATH:
            scalar_path = isinstance(value, str)
            if scalar_path:
                value = [value]
            if (
                not isinstance(value, list)
                or not value
                or not all(isinstance(item, str) for item in value)
            ):
                raise OrganelleContractError(
                    code="capability.fixture_parameter_invalid",
                    message=f"path-codec fixture parameter must be a string or non-empty string list: {name}",
                    details={
                        "capability_id": entry.capability_id,
                        "case": fixture.case,
                        "parameter": name,
                    },
                )
            resolved_paths = [
                str(
                    _resolve_fixture_file(
                        entry,
                        item,
                        field=f"parameters.{name}",
                        fixture_case=fixture.case,
                    )
                )
                for item in value
            ]
            resolved[name] = resolved_paths[0] if scalar_path else resolved_paths
        else:
            resolved[name] = value
    return resolved


class LocalVerificationEnvironment:
    def __init__(
        self,
        index: CapabilityIndex,
        *,
        executor: BundleWorkerExecutor | None = None,
    ) -> None:
        self._index = index
        self._executor: BundleWorkerExecutor = (
            executor if executor is not None else OneShotBundleWorkerExecutor()
        )

    @property
    def index(self) -> CapabilityIndex:
        return self._index

    @property
    def executor(self) -> BundleWorkerExecutor:
        return self._executor

    def observe_environment(self, entry: CapabilityEntry) -> EnvironmentSnapshot:
        from .admission import LocalAdmissionEnvironment

        return LocalAdmissionEnvironment().observe_environment(entry)

    def evaluate_fixture(
        self,
        entry: CapabilityEntry,
        fixture: FixtureSpec,
        *,
        executor: BundleWorkerExecutor,
        environment_key: str,
        fixture_dataset_hash: str,
    ) -> EquivalenceEvidence:
        if fixture.adversarial:
            return self._evaluate_adversarial_fixture(
                entry,
                fixture,
                executor=executor,
                environment_key=environment_key,
                fixture_dataset_hash=fixture_dataset_hash,
            )
        if fixture.reference is not None:
            raise OrganelleContractError(
                code="capability.fixture_reference_unsupported",
                message=(
                    "LocalVerificationEnvironment does not evaluate reference-equivalence "
                    f"fixtures yet; capability {entry.capability_id!r} fixture "
                    f"{fixture.case!r} declares reference {fixture.reference!r}. Comparing a "
                    "candidate's produced hash to a second, live capability's produced hash "
                    "is still a data-level operation once both sides are computed - "
                    "resolving and safely executing that second capability is the missing "
                    "piece, not the comparison."
                ),
                details={
                    "capability_id": entry.capability_id,
                    "case": fixture.case,
                    "reference": fixture.reference,
                },
            )
        if entry.bundle.contract.input_kind is not CoreKind.NONE:
            raise OrganelleContractError(
                code="capability.fixture_input_kind_unsupported",
                message=(
                    "LocalVerificationEnvironment only evaluates fixtures for a 'none' "
                    f"input_kind capability; capability {entry.capability_id!r} declares "
                    f"input_kind {entry.bundle.contract.input_kind.value!r}. Constructing a "
                    "CoreObject from a fixture's 'input' JSON is a separate adapter this "
                    "evaluator does not implement yet."
                ),
                details={
                    "capability_id": entry.capability_id,
                    "case": fixture.case,
                    "input_kind": entry.bundle.contract.input_kind.value,
                },
            )
        assert fixture.expect is not None, (
            "FixtureSpec.validate_fixture guarantees exactly one of expect/reference; "
            "reference is None here, so expect must be set"
        )

        method = self._resolve_equivalence_method(entry, fixture)
        expected_path = _resolve_fixture_file(
            entry, fixture.expect, field="expect", fixture_case=fixture.case
        )
        try:
            expected_json = cast(object, json.loads(expected_path.read_text(encoding="utf-8")))
        except ValueError as error:
            raise OrganelleContractError(
                code="capability.fixture_file_invalid",
                message=f"fixture expect file is not valid JSON: {fixture.expect}",
                details={
                    "capability_id": entry.capability_id,
                    "case": fixture.case,
                    "reason": str(error),
                },
            ) from error

        parameters = _resolve_fixture_parameters(entry, fixture)
        produced_json = self._invoke_fixture(entry, executor, parameters)
        produced_hash = _fixture_produced_hash(produced_json)
        passed = _compare_json_values(
            produced_json, expected_json, method=method, tolerance=fixture.tolerance
        )

        determinism_verdict: Literal["pass", "fail", "not_checked"] = "not_checked"
        determinism_hash: str | None = None
        if not entry.bundle.contract.deterministic:
            second_json = self._invoke_fixture(entry, executor, parameters)
            determinism_hash = _fixture_produced_hash(second_json)
            determinism_passed = _compare_json_values(
                produced_json, second_json, method=method, tolerance=fixture.tolerance
            )
            determinism_verdict = "pass" if determinism_passed else "fail"
            passed = passed and determinism_passed

        rewrite_verdict: Literal["pass", "fail", "not_checked"] = "not_checked"
        rewrite_hash: str | None = None
        rewrite_callable: str | None = None
        rewrite = entry.bundle.contract.rewrite
        if rewrite is not None:
            original_json = self._invoke_original(entry, rewrite, parameters)
            rewrite_hash = _fixture_produced_hash(original_json)
            rewrite_callable = rewrite.original
            tolerance = rewrite.tolerance if rewrite.method == "numeric_tolerance" else None
            rewrite_passed = _compare_json_values(
                produced_json, original_json, method=rewrite.method, tolerance=tolerance
            )
            rewrite_verdict = "pass" if rewrite_passed else "fail"
            passed = passed and rewrite_passed

        return EquivalenceEvidence(
            case=fixture.case,
            method=method,
            tolerance=fixture.tolerance,
            evaluator_version=_FIXTURE_EVALUATOR_VERSION,
            verdict="pass" if passed else "fail",
            fixture_dataset_hash=fixture_dataset_hash,
            candidate=CandidateEvidence(
                bundle_content_hash=entry.content_hash,
                environment_key=environment_key,
                produced_hash=produced_hash,
            ),
            reference=None,
            determinism_verdict=determinism_verdict,
            determinism_produced_hash=determinism_hash,
            rewrite_verdict=rewrite_verdict,
            rewrite_reference_hash=rewrite_hash,
            rewrite_original_callable=rewrite_callable,
        )

    def _resolve_equivalence_method(
        self,
        entry: CapabilityEntry,
        fixture: FixtureSpec,
    ) -> Literal["exact", "numeric_tolerance"]:
        if fixture.equivalence is EquivalenceMethod.EXACT:
            return "exact"
        if fixture.equivalence is EquivalenceMethod.NUMERIC_TOLERANCE:
            if fixture.tolerance is None:
                raise OrganelleContractError(
                    code="capability.fixture_tolerance_missing",
                    message=(
                        "numeric_tolerance equivalence requires a tolerance; none was "
                        f"recorded on fixture {fixture.case!r}"
                    ),
                    details={"capability_id": entry.capability_id, "case": fixture.case},
                )
            return "numeric_tolerance"
        raise OrganelleContractError(
            code="capability.equivalence_method_unsupported",
            message=f"set_membership equivalence is not implemented: {entry.capability_id}",
            details={"capability_id": entry.capability_id, "case": fixture.case},
        )

    def _evaluate_adversarial_fixture(
        self,
        entry: CapabilityEntry,
        fixture: FixtureSpec,
        *,
        executor: BundleWorkerExecutor,
        environment_key: str,
        fixture_dataset_hash: str,
    ) -> EquivalenceEvidence:
        parameters = _resolve_fixture_parameters(entry, fixture)
        actual_code = "capability.unexpected_success"
        produced_json: object
        try:
            produced_json = self._invoke_fixture(entry, executor, parameters)
        except OrganelleError as exc:
            actual_code = exc.code
            produced_json = {"error": exc.as_dict()}
        except Exception as exc:
            actual_code = "capability.unexpected_error"
            produced_json = {"error": str(exc)}
        else:
            actual_code = "capability.unexpected_success"
        produced_hash = _fixture_produced_hash(produced_json)
        passed = actual_code == fixture.expect_failure_code
        return EquivalenceEvidence(
            case=fixture.case,
            method="exact",
            evaluator_version=_FIXTURE_EVALUATOR_VERSION,
            verdict="pass" if passed else "fail",
            fixture_dataset_hash=fixture_dataset_hash,
            candidate=CandidateEvidence(
                bundle_content_hash=entry.content_hash,
                environment_key=environment_key,
                produced_hash=produced_hash,
            ),
            reference=None,
            determinism_verdict="not_checked",
            determinism_produced_hash=None,
            rewrite_verdict="not_checked",
            rewrite_reference_hash=None,
            rewrite_original_callable=None,
        )

    def _invoke_original(
        self,
        entry: CapabilityEntry,
        rewrite: RewriteSpec,
        parameters: dict[str, object],
    ) -> object:
        if not _is_pure_core_native(entry):
            raise OrganelleContractError(
                code="capability.rewrite_unsupported",
                message=(
                    "rewrite equivalence is only supported for pure-core native capabilities in v1"
                ),
                details={"capability_id": entry.capability_id},
            )
        from organelleverse.operations.python_binding import bind_python_capability

        implementation = _resolve_core_callable(entry.capability_id, rewrite.original)
        bound = bind_python_capability(entry.bundle, implementation, frozen_schema=None)
        result = bound.invoke(None, parameters)
        return result.model_dump(mode="json")

    def _invoke_fixture(
        self,
        entry: CapabilityEntry,
        executor: BundleWorkerExecutor,
        parameters: dict[str, object],
    ) -> object:
        if _is_pure_core_native(entry):
            from organelleverse.operations.python_binding import bind_python_capability

            implementation = _resolve_core_callable(
                entry.capability_id, cast(str, entry.bundle.contract.callable_locator)
            )
            bound = bind_python_capability(entry.bundle, implementation, frozen_schema=None)
            result = bound.invoke(None, parameters)
            return result.model_dump(mode="json")

        identity = entry.execution_identity
        if identity is None:
            raise OrganelleContractError(
                code="capability.execution_provider_required",
                message=f"capability has no content-addressed execution provider: {entry.capability_id}",
                details={"capability_id": entry.capability_id},
            )
        with tempfile.TemporaryDirectory(prefix="organelleverse-fixture-staging-") as staging:
            worker_result = executor.invoke(
                entry,
                input=None,
                parameters=parameters,
                run_id=uuid4().hex,
                staging_root=Path(staging),
            )
        return worker_result.value


__all__ = [
    "AdapterRequirement",
    "CandidateEvidence",
    "EnvironmentSnapshot",
    "EquivalenceEvidence",
    "FastFileKey",
    "LocalVerificationEnvironment",
    "ObservedDependency",
    "PlatformIdentity",
    "ReferenceEvidence",
    "VerificationEnvironment",
    "VerificationRecord",
    "VerificationStore",
    "compute_environment_key",
    "current_platform_identity",
    "default_verification_store",
    "strip_run_provenance",
    "verify_capability",
]
