"""Released assembly writer: compute-then-write promotion contract.

``assembly.write`` is the explicit typed materializer for a managed assembly
Result. It performs no scientific work: it moves one completed managed run tree
to a caller-selected destination through :func:`organelleverse.runtime.promote_result`
and returns a relocated Result whose artifact references point directly at the
destination. The former managed run path becomes a safe symbolic link so the
pre-write Result references keep resolving without a second artifact tree.

These tests pin the writer contract directly: promotion leaves one tree, old
paths resolve through the link, conflicts and tampering fail closed, and the
typed writer agrees with the generic ``ov.write`` dispatcher and the Agent JSON
transport. The same-filesystem and cross-filesystem promotion primitives
themselves are exhaustively covered in ``tests/runtime``.
"""

from __future__ import annotations

import errno
from pathlib import Path
from typing import cast

import pytest

from organelleverse.assembly.api import write as assembly_write
from organelleverse.assembly.operations import ASSEMBLY_WRITE_SPEC
from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.errors import OrganelleInputError
from organelleverse.core.result import ErrorDetail, OrganelleResult
from organelleverse.operations import (
    OperationRegistry,
    parameter_schema,
)
from organelleverse.operations import (
    list as list_operations,
)
from organelleverse.operations.adapters import invoke_json
from organelleverse.operations.spec import CoreKind, OperationStage, SideEffect
from organelleverse.runtime import managed_run_path
from organelleverse.writer import write as generic_write


def _managed_assembly_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    run_id: str = "sha256-test",
) -> tuple[OrganelleResult, Path]:
    """Build one unpublished managed assembly run and its artifact-backed Result."""

    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    run_dir = managed_run_path("assembly.assemble", run_id)
    run_dir.mkdir(parents=True)
    fasta = run_dir / "assembly.fasta"
    manifest = run_dir / "assembly_run_manifest.json"
    genome = run_dir / "primary_genome.json"
    fasta.write_text(">contig\nACGT\n")
    manifest.write_text('{"kind":"assembly_run_manifest"}\n')
    genome.write_text('{"kind":"primary_genome"}\n')
    result = OrganelleResult(
        operation_id="assembly.assemble",
        scope="mitochondrion",
        status="ok",
        summary_text="assembled",
        artifacts=(
            ArtifactRef.from_path(
                fasta,
                kind="assembly_output",
                format="fasta",
                media_type="text/x-fasta",
            ),
            ArtifactRef.from_path(
                manifest,
                kind="assembly_run_manifest",
                format="json",
                media_type="application/json",
            ),
            ArtifactRef.from_path(
                genome,
                kind="primary_genome_manifest",
                format="json",
                media_type="application/json",
            ),
        ),
    )
    (run_dir / "result.json").write_text(result.model_dump_json(exclude={"object_id"}) + "\n")
    return result, run_dir


def _identical_managed_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    count: int,
) -> list[tuple[OrganelleResult, Path]]:
    """Build ``count`` content-identical managed runs under distinct run ids."""

    built: list[tuple[OrganelleResult, Path]] = []
    for index in range(count):
        result, run_dir = _managed_assembly_result(
            tmp_path,
            monkeypatch,
            run_id=f"sha256-identity-{index}",
        )
        built.append((result, run_dir))
    return built


def test_assembly_write_promotes_managed_result_and_leaves_one_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, run_dir = _managed_assembly_result(tmp_path, monkeypatch)
    old_artifact_paths = tuple(Path(item.uri) for item in result.artifacts)
    destination = tmp_path / "published"

    relocated = assembly_write(result, output=destination)

    assert relocated.object_id == result.object_id
    assert relocated.operation_id == "assembly.assemble"
    assert destination.is_dir()
    # The former managed run is now only a safe symbolic link to the destination.
    assert run_dir.is_symlink()
    assert run_dir.resolve() == destination.resolve()
    # The returned Result points directly at the destination.
    assert {Path(item.uri) for item in relocated.artifacts} == {
        destination / "assembly.fasta",
        destination / "assembly_run_manifest.json",
        destination / "primary_genome.json",
    }
    # Pre-write artifact references still resolve through the symbolic link.
    assert all(path.is_file() for path in old_artifact_paths)
    # Exactly one complete artifact tree remains on disk.
    assert len(list(tmp_path.rglob("assembly.fasta"))) == 1


def test_assembly_write_retry_reuses_same_verified_published_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, run_dir = _managed_assembly_result(tmp_path, monkeypatch)
    destination = tmp_path / "published"

    first = assembly_write(result, output=destination)
    retried_from_managed_handle = assembly_write(result, output=destination)
    retried_from_published_handle = assembly_write(first, output=destination)

    assert retried_from_managed_handle == first
    assert retried_from_published_handle == first
    assert run_dir.is_symlink()
    assert len(list(tmp_path.rglob("assembly.fasta"))) == 1


def test_assembly_write_retry_rejects_tampered_published_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, _run_dir = _managed_assembly_result(tmp_path, monkeypatch)
    destination = tmp_path / "published"
    published = assembly_write(result, output=destination)
    (destination / "assembly.fasta").write_text(">contig\nTTTT\n")

    with pytest.raises(OrganelleInputError) as captured:
        assembly_write(published, output=destination)

    assert captured.value.code == "runtime.destination_conflict"


def test_module_writer_and_generic_writer_return_equivalent_relocated_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (module_result, module_run), (generic_result, generic_run) = _identical_managed_results(
        tmp_path, monkeypatch, 2
    )
    module_destination = tmp_path / "module"
    generic_destination = tmp_path / "generic"

    module_relocated = assembly_write(module_result, output=module_destination)
    generic_relocated = generic_write(generic_result, generic_destination)

    # Identity is content-derived, so identical trees share an object id even
    # though their managed run paths and destinations differ.
    assert module_relocated.object_id == generic_relocated.object_id
    assert {a.sha256 for a in module_relocated.artifacts} == {
        a.sha256 for a in generic_relocated.artifacts
    }
    # Both paths routed through the promotion primitive: each destination is a
    # complete tree and each former managed run is only a symbolic link.
    assert module_destination.is_dir() and generic_destination.is_dir()
    assert module_run.is_symlink() and generic_run.is_symlink()
    for name in ("assembly.fasta", "assembly_run_manifest.json", "primary_genome.json"):
        assert (module_destination / name).read_bytes() == (generic_destination / name).read_bytes()


def test_direct_registry_and_agent_json_writer_calls_agree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        (direct_result, direct_run),
        (registry_result, registry_run),
        (agent_result, agent_run),
    ) = _identical_managed_results(tmp_path, monkeypatch, 3)
    direct_destination = tmp_path / "direct"
    registry_destination = tmp_path / "registry"
    agent_destination = tmp_path / "agent"

    registry = OperationRegistry()
    registry.register(ASSEMBLY_WRITE_SPEC, assembly_write)

    direct = assembly_write(direct_result, output=direct_destination)
    registered = cast(
        OrganelleResult,
        registry.invoke(
            "assembly.write",
            input=registry_result,
            parameters={"output": registry_destination},
        ),
    )
    response = invoke_json(
        {
            "operation_id": "assembly.write",
            "input": agent_result.model_dump(mode="json"),
            "parameters": {"output": str(agent_destination)},
        },
        registry=registry,
        granted_side_effects=set(ASSEMBLY_WRITE_SPEC.side_effects),
    )

    assert response["ok"] is True
    agent = OrganelleResult.model_validate(response["result"])
    # All three invocation paths promote identical trees to a Result with the
    # same identity and artifact digests.
    assert {direct.object_id, registered.object_id, agent.object_id} == {direct.object_id}
    assert {a.sha256 for a in direct.artifacts} == {a.sha256 for a in registered.artifacts}
    assert {a.sha256 for a in direct.artifacts} == {a.sha256 for a in agent.artifacts}
    for run in (direct_run, registry_run, agent_run):
        assert run.is_symlink()
    for destination in (direct_destination, registry_destination, agent_destination):
        assert (destination / "assembly.fasta").read_text() == ">contig\nACGT\n"


def test_assembly_write_cross_filesystem_promotes_one_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import organelleverse.runtime as runtime

    result, run_dir = _managed_assembly_result(tmp_path, monkeypatch)
    destination = tmp_path / "published"
    real_rename = runtime._rename_no_replace  # pyright: ignore[reportPrivateUsage]

    def cross_device_once(source: Path, target: Path) -> None:
        if source == run_dir:
            raise OSError(errno.EXDEV, "cross-device link")
        real_rename(source, target)

    monkeypatch.setattr(runtime, "_rename_no_replace", cross_device_once)

    relocated = assembly_write(result, output=destination)

    assert destination.is_dir()
    assert run_dir.is_symlink()
    assert Path(relocated.artifacts[0].uri).read_text() == ">contig\nACGT\n"
    assert len(list(tmp_path.rglob("assembly.fasta"))) == 1


def test_assembly_write_rejects_conflict_without_touching_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, run_dir = _managed_assembly_result(tmp_path, monkeypatch)
    conflict = tmp_path / "published"
    conflict.mkdir()
    (conflict / "unrelated").write_text("do not replace")

    with pytest.raises(OrganelleInputError) as captured:
        assembly_write(result, output=conflict)

    assert captured.value.code == "runtime.destination_conflict"
    assert (conflict / "unrelated").read_text() == "do not replace"
    assert run_dir.is_dir() and not run_dir.is_symlink()


def test_assembly_write_rejects_tampered_artifact_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, run_dir = _managed_assembly_result(tmp_path, monkeypatch)
    Path(result.artifacts[0].uri).write_text(">contig\nTTTT\n")
    destination = tmp_path / "published"

    with pytest.raises(OrganelleInputError) as captured:
        assembly_write(result, output=destination)

    assert captured.value.code == "runtime.artifact_digest_mismatch"
    assert run_dir.is_dir() and not run_dir.is_symlink()
    assert not destination.exists()


def test_assembly_write_promotes_failed_run_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, run_dir = _managed_assembly_result(tmp_path, monkeypatch)
    failed = result.model_copy(
        update={
            "status": "failed",
            "errors": (ErrorDetail(code="assembly.fixture_failure", message="fixture"),),
        }
    )
    destination = tmp_path / "published-failure"

    relocated = assembly_write(failed, output=destination)

    assert relocated.status == "failed"
    assert destination.is_dir()
    assert run_dir.is_symlink()


def test_assembly_write_rejects_non_assembly_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, run_dir = _managed_assembly_result(tmp_path, monkeypatch)
    non_assembly = result.model_copy(update={"operation_id": "qc.assembly"})

    with pytest.raises(OrganelleInputError) as captured:
        assembly_write(non_assembly, output=tmp_path / "published")

    assert captured.value.code == "assembly.write_input_contract"
    assert run_dir.is_dir() and not run_dir.is_symlink()


def test_assembly_write_spec_is_a_closed_consume_contract() -> None:
    assert ASSEMBLY_WRITE_SPEC.operation_id == "assembly.write"
    assert ASSEMBLY_WRITE_SPEC.contract_version == "1.0"
    assert ASSEMBLY_WRITE_SPEC.stage is OperationStage.CONSUME
    assert ASSEMBLY_WRITE_SPEC.input_kind is CoreKind.RESULT
    assert ASSEMBLY_WRITE_SPEC.output_kind is CoreKind.RESULT
    assert ASSEMBLY_WRITE_SPEC.organelle_types == ("mitochondrion", "plastid")
    assert ASSEMBLY_WRITE_SPEC.callable_locator == "organelleverse.assembly.api:write"
    assert ASSEMBLY_WRITE_SPEC.side_effects == (SideEffect.READ_FILES, SideEffect.WRITE_FILES)
    assert ASSEMBLY_WRITE_SPEC.deterministic is True
    assert ASSEMBLY_WRITE_SPEC.idempotent is True
    assert ASSEMBLY_WRITE_SPEC.cacheable is False
    assert ASSEMBLY_WRITE_SPEC.fallback.allowed is False
    assert "assembly.write" in {spec.operation_id for spec in list_operations()}


def test_assembly_write_parameter_schema_is_closed() -> None:
    schema = parameter_schema("assembly.write")
    properties = cast(dict[str, object], schema["properties"])

    assert schema["additionalProperties"] is False
    assert set(properties) == {"output"}
    assert schema["required"] == ["output"]


def test_assembly_facade_exports_the_typed_writer() -> None:
    import organelleverse.assembly as assembly_facade

    assert assembly_facade.write is assembly_write
