# pyright: basic
from __future__ import annotations

import os
import warnings
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

import organelleverse.capabilities.codecs as codecs
import organelleverse.capabilities.worker as worker_runtime
from organelleverse.capabilities.code_identity import ExecutionIdentity
from organelleverse.capabilities.codecs import CapabilityExecutionContext
from organelleverse.capabilities.index import CapabilityEntry
from organelleverse.capabilities.trust import TrustStore, trust
from organelleverse.capabilities.worker import (
    OneShotBundleWorkerExecutor,
    WorkerInvocationStrategy,
)
from organelleverse.capabilities.worker_contracts import WorkerParameter, WorkerResult
from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.data import LineageRecord, OrganelleData
from organelleverse.core.errors import (
    OrganelleContractError,
    OrganelleExecutionError,
    OrganelleInputError,
)
from organelleverse.core.frozen import FrozenMap
from organelleverse.core.genome import OrganelleGenome
from organelleverse.core.result import OrganelleResult
from organelleverse.operations.python_binding import validate_worker_contract
from organelleverse.operations.registry import CoreObject
from organelleverse.operations.spec import (
    CoreKind,
    OperationStage,
    ParameterBindingSpec,
    ParameterCodec,
    ResultCodec,
)
from organelleverse.runtime import managed_runs_root
from tests.capabilities.test_worker import (
    _admitted_custom_entry,
    _bundle,
    _entry,
    _worker_result_bundle,
    _write_bundle,
)


def _artifact(kind: str = "artifact") -> ArtifactRef:
    return ArtifactRef(
        kind=kind,
        uri="artifact.bin",
        format="bin",
        sha256="1" * 64,
        size_bytes=1,
        validated=True,
    )


def _context() -> CapabilityExecutionContext:
    return CapabilityExecutionContext(
        operation_id="demo.worker",
        operation_version="2.0",
        callable_locator="private_worker_pkg.impl:run",
        run_id="parent-run",
        input_object_ids=(),
        input_artifact_hashes=(),
        parameters_hash="a" * 64,
    )


@pytest.mark.parametrize("kind", ["data", "genome"])
def test_worker_supplied_data_or_genome_lineage_is_rejected(kind: str) -> None:
    forged = LineageRecord(
        parent_object_ids=("sha256:" + "b" * 64,),
        operation_id="forged.worker",
        operation_version="9.9",
        parameters_hash="f" * 64,
    )
    value: CoreObject
    if kind == "data":
        value = OrganelleData(modality="worker_data", lineage=(forged,))
    else:
        value = OrganelleGenome(
            organelle="mitochondrion",
            sequence=_artifact("sequence"),
            lineage=(forged,),
        )

    with pytest.raises(OrganelleContractError) as captured:
        codecs.normalize_worker_core_object(value, _context(), input=None)

    assert captured.value.code == "capability.result_codec_invalid"


def test_worker_data_inherits_trusted_input_lineage_then_parent_record() -> None:
    trusted = LineageRecord(
        parent_object_ids=(),
        operation_id="trusted.input",
        operation_version="1.0",
        parameters_hash="c" * 64,
    )
    input_data = OrganelleData(modality="trusted_data", lineage=(trusted,))
    worker_data = OrganelleData(modality="worker_data")
    context = _context().model_copy(update={"input_object_ids": (input_data.object_id,)})
    worker_id = worker_data.object_id

    normalized = codecs.normalize_worker_core_object(
        worker_data,
        context,
        input=input_data,
    )

    assert isinstance(normalized, OrganelleData)
    assert normalized.lineage[:-1] == input_data.lineage
    parent = normalized.lineage[-1]
    assert parent.parent_object_ids == (input_data.object_id,)
    assert parent.operation_id == context.operation_id
    assert parent.operation_version == context.operation_version
    assert parent.parameters_hash == context.parameters_hash
    assert normalized.object_id != worker_id


@pytest.mark.parametrize("kind", ["data", "genome"])
def test_inputless_worker_data_or_genome_receives_one_parent_lineage_record(
    kind: str,
) -> None:
    value: CoreObject
    if kind == "data":
        value = OrganelleData(modality="worker_data")
    else:
        value = OrganelleGenome(organelle="mitochondrion", sequence=_artifact("sequence"))

    normalized = codecs.normalize_worker_core_object(value, _context(), input=None)

    lineage = cast(OrganelleData | OrganelleGenome, normalized).lineage
    assert len(lineage) == 1
    assert lineage[0].parent_object_ids == ()
    assert lineage[0].operation_id == "demo.worker"
    assert lineage[0].operation_version == "2.0"
    assert lineage[0].parameters_hash == "a" * 64


def test_worker_result_keeps_existing_parent_provenance_normalization() -> None:
    value = OrganelleResult(
        operation_id="demo.worker",
        operation_version="9.9",
        scope="none",
        status="ok",
    )

    normalized = codecs.normalize_worker_core_object(value, _context(), input=None)

    assert normalized == codecs.normalize_worker_result(value, _context())


def _artifact_entry(
    root: Path,
    output_kind: CoreKind,
    *,
    source: str | None = None,
) -> CapabilityEntry:
    if source is None:
        base, _, _ = _write_bundle(root)
    else:
        base, _, _ = _write_bundle(root, source=source)
    bundle = _worker_result_bundle(
        base,
        result_codec=ResultCodec.CANONICAL_JSON,
        output_kind=output_kind,
    )
    parameters = (
        WorkerParameter(
            name="value",
            kind="keyword_only",
            required=False,
            default=1,
            annotation="int",
        ),
    )
    return _admitted_custom_entry(root, bundle, parameters)


class _StagingExecutor:
    def __init__(
        self,
        entry: CapabilityEntry,
        *,
        output_kind: CoreKind,
        attack: str | None = None,
        response_identity: ExecutionIdentity | None = None,
        output_content: bytes = b"parent verified worker bytes",
    ) -> None:
        self.entry = entry
        self.output_kind = output_kind
        self.attack = attack
        self.response_identity = response_identity
        self.output_content = output_content
        self.invoke_calls = 0
        self.run_ids: list[str] = []
        self.staging_roots: list[Path] = []
        self.inputs: list[CoreObject | None] = []
        self.parameters: list[dict[str, object]] = []

    def inspect(self, entry: CapabilityEntry):
        raise AssertionError("admitted artifact tests must not inspect again")

    def invoke(
        self,
        entry: CapabilityEntry,
        *,
        input: CoreObject | None,
        parameters: Mapping[str, object],
        run_id: str,
        staging_root: Path,
    ) -> WorkerResult:
        assert entry == self.entry
        assert entry.execution_identity is not None
        self.invoke_calls += 1
        self.run_ids.append(run_id)
        self.staging_roots.append(staging_root)
        self.inputs.append(input)
        self.parameters.append(dict(parameters))
        if self.attack in {"crash", "signal", "timeout"}:
            raise OrganelleExecutionError(
                code=f"capability.worker_{self.attack}",
                message=f"injected {self.attack}",
            )
        content = self.output_content
        relative_paths = ("nested/first.bin", "nested/deeper/second.bin")
        artifacts: list[ArtifactRef] = []
        for index, relative in enumerate(relative_paths):
            if self.attack == "missing" and index == 1:
                path = staging_root / relative
            else:
                path = staging_root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content + bytes([index]))
            declared = ArtifactRef(
                kind="artifact" if self.output_kind is not CoreKind.GENOME else "sequence",
                uri=relative,
                format="bin",
                sha256=(
                    "f" * 64
                    if self.attack == "hash" and index == 0
                    else __import__("hashlib").sha256(content + bytes([index])).hexdigest()
                ),
                size_bytes=len(content) + 1,
                validated=True,
            )
            artifacts.append(declared)
        if self.attack == "extra":
            (staging_root / "undeclared.bin").write_bytes(b"undeclared")
        if self.output_kind is CoreKind.RESULT:
            value: object = OrganelleResult(
                operation_id="demo.worker",
                operation_version="9.9",
                scope="none",
                status="ok",
                artifacts=tuple(artifacts),
            ).model_dump(mode="json")
        elif self.output_kind is CoreKind.DATA:
            value = OrganelleData(
                modality="worker_data",
                artifacts=FrozenMap.from_items({"first": artifacts[0], "second": artifacts[1]}),
            ).model_dump(mode="json")
        else:
            value = OrganelleGenome(
                organelle="mitochondrion",
                sequence=artifacts[0],
                source_manifests=(artifacts[1].model_copy(update={"kind": "manifest"}),),
            ).model_dump(mode="json")
        response_identity = entry.execution_identity
        if self.attack == "response_identity":
            assert self.response_identity is not None
            response_identity = self.response_identity
        return WorkerResult(
            request_id="artifact-result",
            execution_identity=response_identity,
            value=cast(object, value),  # type: ignore[arg-type]
            artifact_paths=("nested/first.bin",) if self.attack == "artifact_paths" else (),
        )


def _strategy(
    tmp_path: Path,
    entry: CapabilityEntry,
    executor: _StagingExecutor,
) -> WorkerInvocationStrategy:
    store = TrustStore(tmp_path / "trust.json")
    assert entry.execution_identity is not None
    trust(entry.capability_id, entry.execution_identity, store=store)
    return WorkerInvocationStrategy(entry, executor, store)


def _published_artifacts(value: CoreObject) -> tuple[ArtifactRef, ...]:
    if isinstance(value, OrganelleResult):
        return value.artifacts
    if isinstance(value, OrganelleData):
        return tuple(value.artifacts.values())
    return tuple(
        artifact
        for artifact in (value.sequence, value.annotation, *value.source_manifests)
        if artifact is not None
    )


@pytest.mark.parametrize("output_kind", [CoreKind.RESULT, CoreKind.DATA, CoreKind.GENOME])
def test_worker_publishes_verified_core_artifacts_from_owned_hidden_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    output_kind: CoreKind,
) -> None:
    cache = tmp_path / "cache"
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(cache))
    entry = _artifact_entry(tmp_path / f"bundle-{output_kind.value}", output_kind)
    executor = _StagingExecutor(entry, output_kind=output_kind)
    strategy = _strategy(tmp_path, entry, executor)

    published = strategy.invoke(None, {"value": 1})

    assert executor.invoke_calls == 1
    assert len(executor.run_ids) == 1
    staging = executor.staging_roots[0]
    request_id = executor.run_ids[0]
    assert staging.name == f".staging-{request_id}"
    assert not staging.exists()
    operation_root = managed_runs_root() / "demo.worker"
    completed = tuple(operation_root.iterdir())
    assert len(completed) == 1
    assert completed[0].name != request_id
    assert not completed[0].name.startswith(".staging-")
    artifacts = _published_artifacts(published)
    assert len(artifacts) == 2
    assert all(artifact.validated for artifact in artifacts)
    assert all(Path(artifact.uri).is_relative_to(completed[0]) for artifact in artifacts)
    assert all(
        Path(artifact.uri).read_bytes().startswith(b"parent verified") for artifact in artifacts
    )
    if isinstance(published, OrganelleResult):
        assert published.provenance is not None
        assert published.operation_version == "1.0"
    else:
        assert len(published.lineage) == 1
        assert published.lineage[0].operation_id == "demo.worker"


@pytest.mark.parametrize(
    ("attack", "expected_code"),
    [
        ("missing", "runtime.artifact_tree_mismatch"),
        ("extra", "runtime.artifact_tree_mismatch"),
        ("hash", "runtime.artifact_digest_mismatch"),
        ("artifact_paths", "capability.worker_protocol_invalid"),
        ("response_identity", "capability.worker_identity_mismatch"),
        ("crash", "capability.worker_crash"),
        ("signal", "capability.worker_signal"),
        ("timeout", "capability.worker_timeout"),
    ],
)
def test_worker_failure_cleans_owned_staging_and_publishes_no_partial_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attack: str,
    expected_code: str,
) -> None:
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    entry = _artifact_entry(tmp_path / f"bundle-{attack}", CoreKind.RESULT)
    response_identity = None
    if attack == "response_identity":
        other = _artifact_entry(
            tmp_path / "bundle-other-identity",
            CoreKind.RESULT,
            source="def run(*, value: int = 1):\n    return {'value': value + 99}\n",
        )
        response_identity = other.execution_identity
        assert response_identity is not None
    executor = _StagingExecutor(
        entry,
        output_kind=CoreKind.RESULT,
        attack=attack,
        response_identity=response_identity,
    )
    strategy = _strategy(tmp_path, entry, executor)

    with pytest.raises(
        (OrganelleContractError, OrganelleExecutionError, OrganelleInputError)
    ) as captured:
        strategy.invoke(None, {"value": 1})

    assert captured.value.code == expected_code
    operation_root = managed_runs_root() / "demo.worker"
    assert not operation_root.exists() or tuple(operation_root.iterdir()) == ()


def test_destination_conflict_preserves_existing_run_and_cleans_owned_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    entry = _artifact_entry(tmp_path / "bundle-conflict", CoreKind.RESULT)
    executor = _StagingExecutor(entry, output_kind=CoreKind.RESULT)
    strategy = _strategy(tmp_path, entry, executor)
    request_id = "request-conflict"
    publication_id = "publication-conflict"
    identifiers = iter([SimpleNamespace(hex=request_id), SimpleNamespace(hex=publication_id)])
    monkeypatch.setattr(worker_runtime, "uuid4", lambda: next(identifiers))
    destination = managed_runs_root() / "demo.worker" / publication_id
    destination.mkdir(parents=True)
    sentinel = destination / "valuable.bin"
    sentinel.write_bytes(b"keep")

    with pytest.raises(OrganelleInputError) as captured:
        strategy.invoke(None, {"value": 1})

    assert captured.value.code == "runtime.run_conflict"
    assert executor.run_ids == [request_id]
    assert sentinel.read_bytes() == b"keep"
    assert tuple(destination.parent.iterdir()) == (destination,)


def test_publication_failure_cleans_staging_without_publishing_partial_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    entry = _artifact_entry(tmp_path / "bundle-publication", CoreKind.RESULT)
    executor = _StagingExecutor(entry, output_kind=CoreKind.RESULT)
    strategy = _strategy(tmp_path, entry, executor)

    def fail_publication(*_args: object) -> CoreObject:
        raise OrganelleExecutionError(
            code="runtime.run_publication_failed",
            message="injected atomic publication failure",
        )

    monkeypatch.setattr(worker_runtime, "publish_staged_result", fail_publication)

    with pytest.raises(OrganelleExecutionError) as captured:
        strategy.invoke(None, {"value": 1})

    assert captured.value.code == "runtime.run_publication_failed"
    operation_root = managed_runs_root() / "demo.worker"
    assert not operation_root.exists() or tuple(operation_root.iterdir()) == ()


def test_concurrent_worker_publications_use_disjoint_staging_and_completed_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    entry = _artifact_entry(tmp_path / "bundle-concurrent", CoreKind.RESULT)
    executor = _StagingExecutor(entry, output_kind=CoreKind.RESULT)
    strategy = _strategy(tmp_path, entry, executor)

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = tuple(pool.map(lambda value: strategy.invoke(None, {"value": value}), range(4)))

    operation_root = managed_runs_root() / "demo.worker"
    completed = tuple(operation_root.iterdir())
    assert len(results) == len(completed) == 4
    assert len(set(executor.run_ids)) == 4
    assert len(set(executor.staging_roots)) == 4
    assert all(not path.exists() for path in executor.staging_roots)
    assert all(not path.name.startswith(".staging-") for path in completed)
    assert all(
        Path(artifact.uri).is_file()
        for result in results
        for artifact in cast(OrganelleResult, result).artifacts
    )


def test_worker_failure_cleanup_does_not_remove_sibling_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    entry = _artifact_entry(tmp_path / "bundle-sibling", CoreKind.RESULT)
    executor = _StagingExecutor(entry, output_kind=CoreKind.RESULT, attack="crash")
    strategy = _strategy(tmp_path, entry, executor)
    sibling = managed_runs_root() / "demo.worker" / ".staging-neighbor"
    sibling.mkdir(parents=True)
    (sibling / "valuable.bin").write_bytes(b"keep")

    with pytest.raises(OrganelleExecutionError):
        strategy.invoke(None, {"value": 1})

    assert (sibling / "valuable.bin").read_bytes() == b"keep"
    assert tuple(sibling.parent.iterdir()) == (sibling,)


def test_cleanup_warning_cannot_mask_original_worker_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    entry = _artifact_entry(tmp_path / "bundle-cleanup-warning", CoreKind.RESULT)
    executor = _StagingExecutor(entry, output_kind=CoreKind.RESULT, attack="crash")
    strategy = _strategy(tmp_path, entry, executor)

    def fail_cleanup(_path: Path) -> None:
        raise OSError("injected cleanup failure")

    monkeypatch.setattr(worker_runtime, "remove_managed_entry", fail_cleanup)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(OrganelleExecutionError) as captured:
            strategy.invoke(None, {"value": 1})

    assert captured.value.code == "capability.worker_crash"


def test_manifest_drift_in_staging_creation_window_never_invokes_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    entry = _artifact_entry(tmp_path / "bundle-create-window", CoreKind.RESULT)
    executor = _StagingExecutor(entry, output_kind=CoreKind.RESULT)
    strategy = _strategy(tmp_path, entry, executor)
    real_create = worker_runtime.create_staged_run

    def drift_after_create(operation_id: str, request_run_id: str) -> Path:
        staging = real_create(operation_id, request_run_id)
        (entry.bundle_root / "capability.toml").write_text(
            "schema='drift-after-staging'\n", encoding="utf-8"
        )
        return staging

    monkeypatch.setattr(worker_runtime, "create_staged_run", drift_after_create)
    with pytest.raises(OrganelleContractError) as captured:
        strategy.invoke(None, {"value": 1})

    assert captured.value.code == "capability.execution_identity_drift"
    assert executor.invoke_calls == 0
    operation_root = managed_runs_root() / "demo.worker"
    assert not operation_root.exists() or tuple(operation_root.iterdir()) == ()


def test_worker_cannot_reemit_path_parameter_bytes_as_an_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    root = tmp_path / "bundle-path-reemit"
    base, _, _ = _write_bundle(root)
    result_bundle = _worker_result_bundle(
        base,
        result_codec=ResultCodec.CANONICAL_JSON,
        output_kind=CoreKind.RESULT,
    )
    binding = result_bundle.contract.binding.model_copy(
        update={"parameters": (ParameterBindingSpec(name="source", codec=ParameterCodec.PATH),)}
    )
    bundle = result_bundle.model_copy(
        update={"contract": result_bundle.contract.model_copy(update={"binding": binding})}
    )
    parameters = (
        WorkerParameter(
            name="source",
            kind="keyword_only",
            required=True,
            default=None,
            annotation="path",
        ),
    )
    entry = _admitted_custom_entry(root, bundle, parameters)
    executor = _StagingExecutor(entry, output_kind=CoreKind.RESULT)
    strategy = _strategy(tmp_path, entry, executor)
    source = tmp_path / "trusted-input.bin"
    source.write_bytes(b"parent verified worker bytes\x00")

    with pytest.raises(OrganelleInputError) as captured:
        strategy.invoke(None, {"source": str(source)})

    assert captured.value.code == "runtime.input_artifact_reemission"
    assert not executor.staging_roots[0].exists()


def test_path_parameter_is_snapshotted_and_reverified_across_worker_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    root = tmp_path / "bundle-path-input-race"
    base, _, _ = _write_bundle(root)
    result_bundle = _worker_result_bundle(
        base,
        result_codec=ResultCodec.CANONICAL_JSON,
        output_kind=CoreKind.RESULT,
    )
    binding = result_bundle.contract.binding.model_copy(
        update={"parameters": (ParameterBindingSpec(name="source", codec=ParameterCodec.PATH),)}
    )
    bundle = result_bundle.model_copy(
        update={"contract": result_bundle.contract.model_copy(update={"binding": binding})}
    )
    parameters = (
        WorkerParameter(
            name="source",
            kind="keyword_only",
            required=True,
            default=None,
            annotation="path",
        ),
    )
    entry = _admitted_custom_entry(root, bundle, parameters)
    replacement = b"path bytes changed during execution"

    class MutatingPathExecutor(_StagingExecutor):
        def invoke(
            self,
            entry: CapabilityEntry,
            *,
            input: CoreObject | None,
            parameters: Mapping[str, object],
            run_id: str,
            staging_root: Path,
        ) -> WorkerResult:
            snapshot = Path(str(parameters["source"]))
            snapshot.chmod(0o600)
            snapshot.write_bytes(replacement)
            return super().invoke(
                entry,
                input=input,
                parameters=parameters,
                run_id=run_id,
                staging_root=staging_root,
            )

    executor = MutatingPathExecutor(
        entry,
        output_kind=CoreKind.RESULT,
        output_content=replacement,
    )
    strategy = _strategy(tmp_path, entry, executor)
    source = tmp_path / "mutable-path-input.bin"
    source.write_bytes(b"original trusted path bytes")

    with pytest.raises(OrganelleContractError) as captured:
        strategy.invoke(None, {"source": str(source)})

    assert captured.value.code == "capability.input_snapshot_modified"
    assert executor.invoke_calls == 1
    assert source.read_bytes() == b"original trusted path bytes"
    assert executor.staging_roots and not executor.staging_roots[0].exists()


def test_input_snapshot_name_conflict_never_deletes_the_unowned_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    entry = _artifact_entry(tmp_path / "bundle-input-snapshot-conflict", CoreKind.RESULT)
    executor = _StagingExecutor(entry, output_kind=CoreKind.RESULT)
    strategy = _strategy(tmp_path, entry, executor)
    original_mkdir = Path.mkdir
    collided: list[Path] = []

    def collide_at_snapshot(path: Path, *args: object, **kwargs: object) -> None:
        if path.name.startswith(".inputs-"):
            original_mkdir(path, *args, **kwargs)  # type: ignore[arg-type]
            (path / "valuable.bin").write_bytes(b"unowned replacement")
            collided.append(path)
            raise FileExistsError("injected input snapshot collision")
        original_mkdir(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "mkdir", collide_at_snapshot)

    with pytest.raises(OrganelleInputError) as captured:
        strategy.invoke(None, {"value": 1})

    assert captured.value.code == "capability.input_snapshot_conflict"
    assert executor.invoke_calls == 0
    assert len(collided) == 1
    assert (collided[0] / "valuable.bin").read_bytes() == b"unowned replacement"


def test_worker_cannot_reemit_core_input_artifact_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    root = tmp_path / "bundle-core-reemit"
    base, _, _ = _write_bundle(root)
    result_bundle = _worker_result_bundle(
        base,
        result_codec=ResultCodec.CANONICAL_JSON,
        output_kind=CoreKind.RESULT,
    )
    binding = result_bundle.contract.binding.model_copy(
        update={"parameters": (), "core_input_parameter": "data"}
    )
    bundle = result_bundle.model_copy(
        update={
            "contract": result_bundle.contract.model_copy(
                update={"input_kind": CoreKind.DATA, "binding": binding}
            )
        }
    )
    parameters = (
        WorkerParameter(
            name="data",
            kind="keyword_only",
            required=True,
            default=None,
            annotation="json",
        ),
    )
    entry = _admitted_custom_entry(root, bundle, parameters)
    executor = _StagingExecutor(entry, output_kind=CoreKind.RESULT)
    strategy = _strategy(tmp_path, entry, executor)
    source = tmp_path / "trusted-core-input.bin"
    source.write_bytes(b"parent verified worker bytes\x00")
    input_data = OrganelleData(
        modality="trusted_data",
        artifacts=FrozenMap.from_items(
            {"payload": ArtifactRef.from_path(source, kind="artifact", format="bin")}
        ),
    )

    with pytest.raises(OrganelleInputError) as captured:
        strategy.invoke(input_data, {})

    assert captured.value.code == "runtime.input_artifact_reemission"
    assert not executor.staging_roots[0].exists()


@pytest.mark.parametrize("input_kind", [CoreKind.RESULT, CoreKind.DATA, CoreKind.GENOME])
def test_core_input_artifact_identity_is_recomputed_before_reemission_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    input_kind: CoreKind,
) -> None:
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    root = tmp_path / "bundle-core-stale-claim"
    base, _, _ = _write_bundle(root)
    result_bundle = _worker_result_bundle(
        base,
        result_codec=ResultCodec.CANONICAL_JSON,
        output_kind=CoreKind.RESULT,
    )
    binding = result_bundle.contract.binding.model_copy(
        update={"parameters": (), "core_input_parameter": "value"}
    )
    bundle = result_bundle.model_copy(
        update={
            "contract": result_bundle.contract.model_copy(
                update={
                    "stage": (
                        OperationStage.CONSUME
                        if input_kind is CoreKind.RESULT
                        else OperationStage.ANALYZE
                    ),
                    "input_kind": input_kind,
                    "binding": binding,
                }
            )
        }
    )
    parameters = (
        WorkerParameter(
            name="value",
            kind="keyword_only",
            required=True,
            default=None,
            annotation="json",
        ),
    )
    entry = _admitted_custom_entry(root, bundle, parameters)
    executor = _StagingExecutor(entry, output_kind=CoreKind.RESULT)
    strategy = _strategy(tmp_path, entry, executor)
    content = b"parent verified worker bytes\x00"
    source = tmp_path / "trusted-core-input.bin"
    source.write_bytes(content)
    stale_claim = ArtifactRef(
        kind="artifact",
        uri=str(source),
        format="bin",
        sha256="f" * 64,
        size_bytes=len(content),
        validated=True,
    )
    if input_kind is CoreKind.RESULT:
        input_value: CoreObject = OrganelleResult(
            operation_id="demo.parent",
            scope="none",
            status="ok",
            artifacts=(stale_claim,),
        )
    elif input_kind is CoreKind.DATA:
        input_value = OrganelleData(
            modality="trusted_data",
            artifacts=FrozenMap.from_items({"payload": stale_claim}),
        )
    else:
        input_value = OrganelleGenome(organelle="mitochondrion", sequence=stale_claim)

    with pytest.raises(OrganelleInputError) as captured:
        strategy.invoke(input_value, {})

    assert captured.value.code == "input.artifact_digest_mismatch"
    assert executor.invoke_calls == 0
    assert executor.staging_roots == []


@pytest.mark.parametrize("input_kind", [CoreKind.RESULT, CoreKind.DATA, CoreKind.GENOME])
def test_valid_core_input_shapes_reach_worker_only_as_owned_snapshots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    input_kind: CoreKind,
) -> None:
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    root = tmp_path / f"bundle-core-snapshot-{input_kind.value}"
    base, _, _ = _write_bundle(root)
    result_bundle = _worker_result_bundle(
        base,
        result_codec=ResultCodec.CANONICAL_JSON,
        output_kind=CoreKind.RESULT,
    )
    binding = result_bundle.contract.binding.model_copy(
        update={"parameters": (), "core_input_parameter": "value"}
    )
    bundle = result_bundle.model_copy(
        update={
            "contract": result_bundle.contract.model_copy(
                update={
                    "stage": (
                        OperationStage.CONSUME
                        if input_kind is CoreKind.RESULT
                        else OperationStage.ANALYZE
                    ),
                    "input_kind": input_kind,
                    "binding": binding,
                }
            )
        }
    )
    parameters = (
        WorkerParameter(
            name="value",
            kind="keyword_only",
            required=True,
            default=None,
            annotation="json",
        ),
    )
    entry = _admitted_custom_entry(root, bundle, parameters)
    executor = _StagingExecutor(entry, output_kind=CoreKind.RESULT)
    strategy = _strategy(tmp_path, entry, executor)
    source = tmp_path / f"valid-{input_kind.value}.bin"
    source.write_bytes(b"distinct trusted input bytes")
    artifact = ArtifactRef.from_path(source, kind="artifact", format="bin")
    if input_kind is CoreKind.RESULT:
        input_value: CoreObject = OrganelleResult(
            operation_id="demo.parent",
            scope="none",
            status="ok",
            artifacts=(artifact,),
        )
    elif input_kind is CoreKind.DATA:
        input_value = OrganelleData(
            modality="trusted_data",
            artifacts=FrozenMap.from_items({"payload": artifact}),
        )
    else:
        input_value = OrganelleGenome(
            organelle="mitochondrion",
            sequence=artifact,
            source_manifests=(artifact.model_copy(update={"kind": "manifest"}),),
        )

    strategy.invoke(input_value, {})

    assert executor.invoke_calls == 1
    worker_input = executor.inputs[0]
    assert worker_input is not None
    assert worker_input.object_id == input_value.object_id
    snapshot_artifacts = _published_artifacts(worker_input)
    assert snapshot_artifacts
    assert all(".inputs-" in Path(item.uri).parent.name for item in snapshot_artifacts)
    assert all(not Path(item.uri).exists() for item in snapshot_artifacts)
    assert source.read_bytes() == b"distinct trusted input bytes"


def test_core_input_is_snapshotted_and_reverified_across_worker_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    root = tmp_path / "bundle-core-input-race"
    base, _, _ = _write_bundle(root)
    result_bundle = _worker_result_bundle(
        base,
        result_codec=ResultCodec.CANONICAL_JSON,
        output_kind=CoreKind.RESULT,
    )
    binding = result_bundle.contract.binding.model_copy(
        update={"parameters": (), "core_input_parameter": "data"}
    )
    bundle = result_bundle.model_copy(
        update={
            "contract": result_bundle.contract.model_copy(
                update={"input_kind": CoreKind.DATA, "binding": binding}
            )
        }
    )
    parameters = (
        WorkerParameter(
            name="data",
            kind="keyword_only",
            required=True,
            default=None,
            annotation="json",
        ),
    )
    entry = _admitted_custom_entry(root, bundle, parameters)
    replacement = b"bytes changed after parent verification"

    class MutatingInputExecutor(_StagingExecutor):
        def invoke(
            self,
            entry: CapabilityEntry,
            *,
            input: CoreObject | None,
            parameters: Mapping[str, object],
            run_id: str,
            staging_root: Path,
        ) -> WorkerResult:
            assert isinstance(input, OrganelleData)
            snapshot = Path(input.artifacts["payload"].uri)
            snapshot.chmod(0o600)
            snapshot.write_bytes(replacement)
            return super().invoke(
                entry,
                input=input,
                parameters=parameters,
                run_id=run_id,
                staging_root=staging_root,
            )

    executor = MutatingInputExecutor(
        entry,
        output_kind=CoreKind.RESULT,
        output_content=replacement,
    )
    strategy = _strategy(tmp_path, entry, executor)
    source = tmp_path / "mutable-core-input.bin"
    source.write_bytes(b"parent verified worker bytes\x00")
    input_data = OrganelleData(
        modality="trusted_data",
        artifacts=FrozenMap.from_items(
            {"payload": ArtifactRef.from_path(source, kind="artifact", format="bin")}
        ),
    )

    with pytest.raises(OrganelleContractError) as captured:
        strategy.invoke(input_data, {})

    assert captured.value.code == "capability.input_snapshot_modified"
    assert executor.invoke_calls == 1
    assert source.read_bytes() == b"parent verified worker bytes\x00"
    assert executor.staging_roots and not executor.staging_roots[0].exists()


def test_real_worker_callable_cwd_is_owned_staging_without_final_or_cache_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = tmp_path / "cache"
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(cache))
    source = (
        "import hashlib\n"
        "import os\n"
        "from pathlib import Path\n"
        "def run(*, value: int = 1):\n"
        "    content = b'worker artifact'\n"
        "    path = Path('artifact.bin')\n"
        "    path.write_bytes(content)\n"
        "    return {\n"
        "        'schema_version': 'organelleverse.result.v1',\n"
        "        'kind': 'result',\n"
        "        'operation_id': 'demo.worker',\n"
        "        'scope': 'none',\n"
        "        'status': 'ok',\n"
        "        'metrics': {\n"
        "            'cwd': str(Path.cwd()),\n"
        "            'cache_env': os.environ.get('ORGANELLEVERSE_CACHE_ROOT'),\n"
        "        },\n"
        "        'artifacts': [{\n"
        "            'schema_version': 'organelleverse.artifact.v1',\n"
        "            'kind': 'artifact',\n"
        "            'uri': 'artifact.bin',\n"
        "            'format': 'bin',\n"
        "            'media_type': 'application/octet-stream',\n"
        "            'sha256': hashlib.sha256(content).hexdigest(),\n"
        "            'size_bytes': len(content),\n"
        "            'validated': True,\n"
        "        }],\n"
        "    }\n"
    )
    root = tmp_path / "bundle-real"
    base, _, _ = _write_bundle(root, source=source)
    bundle = _worker_result_bundle(
        base,
        result_codec=ResultCodec.CANONICAL_JSON,
        output_kind=CoreKind.RESULT,
    )
    executor = OneShotBundleWorkerExecutor()
    discovered = _entry(root, bundle)
    inspection = executor.inspect(discovered)
    entry = _admitted_custom_entry(root, bundle, inspection.parameters)
    strategy = _strategy(tmp_path, entry, cast(_StagingExecutor, executor))

    result = strategy.invoke(None, {"value": 1})

    assert isinstance(result, OrganelleResult)
    cwd = Path(cast(str, result.metrics["cwd"]))
    assert cwd.name.startswith(".staging-")
    assert not cwd.exists()
    assert result.metrics["cache_env"] is None
    artifact = result.artifacts[0]
    completed = Path(artifact.uri).parent
    assert completed.parent == cwd.parent
    assert completed.name != cwd.name.removeprefix(".staging-")


@pytest.mark.skipif(os.name == "nt", reason="v1 controlled worker requires a POSIX provider")
@pytest.mark.parametrize(
    ("source", "expected_code"),
    [
        (
            "import os\ndef run(*, value: int = 1):\n    os._exit(7)\n",
            "capability.worker_nonzero_exit",
        ),
        pytest.param(
            "import os\n"
            "import signal\n"
            "def run(*, value: int = 1):\n"
            "    os.kill(os.getpid(), signal.SIGTERM)\n",
            "capability.worker_signal",
            marks=pytest.mark.skipif(os.name == "nt", reason="POSIX signal semantics"),
        ),
        (
            "import time\n"
            "def run(*, value: int = 1):\n"
            "    time.sleep(5)\n"
            "    return {'unreachable': True}\n",
            "capability.worker_timeout",
        ),
    ],
)
def test_real_worker_process_failure_cleans_managed_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: str,
    expected_code: str,
) -> None:
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    root = tmp_path / f"bundle-{expected_code}"
    base, _, _ = _write_bundle(root, source=source)
    bundle = _worker_result_bundle(
        base,
        result_codec=ResultCodec.CANONICAL_JSON,
        output_kind=CoreKind.RESULT,
    )
    # Long enough to outlast interpreter startup so the exit/signal cases reach
    # run(), short enough that the sleeping case still trips the timeout.
    executor = OneShotBundleWorkerExecutor(timeout_seconds=2.0)
    discovered = _entry(root, bundle)
    inspection = executor.inspect(discovered)
    entry = _admitted_custom_entry(root, bundle, inspection.parameters)
    strategy = _strategy(tmp_path, entry, cast(_StagingExecutor, executor))

    with pytest.raises(OrganelleExecutionError) as captured:
        strategy.invoke(None, {"value": 1})

    assert captured.value.code == expected_code
    operation_root = managed_runs_root() / entry.capability_id
    assert not operation_root.exists() or tuple(operation_root.iterdir()) == ()


@pytest.mark.parametrize(
    "parameter",
    [
        "output",
        "output_dir",
        "out_dir",
        "destination",
        "workspace",
        "work_dir",
        "output_path",
        "output_prefix",
        "save_path",
        "export_path",
    ],
)
def test_worker_contract_rejects_every_final_write_parameter_before_inspection(
    parameter: str,
) -> None:
    base = _bundle()
    binding = base.contract.binding.model_copy(
        update={"parameters": (ParameterBindingSpec(name=parameter, codec=ParameterCodec.JSON),)}
    )
    bundle = base.model_copy(
        update={"contract": base.contract.model_copy(update={"binding": binding})}
    )

    with pytest.raises(OrganelleContractError) as captured:
        validate_worker_contract(bundle)

    assert captured.value.code == "capability.execution_provider_required"


def test_bundle_local_writer_is_provider_required_before_inspection() -> None:
    base = _bundle()
    bundle = base.model_copy(
        update={
            "capability": base.capability.model_copy(update={"id": "demo.write"}),
            "contract": base.contract.model_copy(
                update={
                    "operation_id": "demo.write",
                    "stage": OperationStage.CONSUME,
                    "input_kind": CoreKind.RESULT,
                    "output_kind": CoreKind.RESULT,
                }
            ),
        }
    )

    with pytest.raises(OrganelleContractError) as captured:
        validate_worker_contract(bundle)

    assert captured.value.code == "capability.execution_provider_required"
