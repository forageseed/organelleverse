from __future__ import annotations

import errno
import hashlib
import os
import stat
from pathlib import Path

import pytest

import organelleverse.runtime as runtime
from organelleverse.capabilities.codecs import CapabilityExecutionContext, _build_provenance
from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.data import OrganelleData
from organelleverse.core.errors import OrganelleExecutionError, OrganelleInputError
from organelleverse.core.frozen import FrozenMap
from organelleverse.core.genome import OrganelleGenome
from organelleverse.core.result import OrganelleResult
from organelleverse.runtime import (
    cache_root,
    managed_run_path,
    managed_runs_root,
    publish_run,
    remove_managed_entry,
)


def _artifact(uri: str, content: bytes = b"verified bytes") -> ArtifactRef:
    return ArtifactRef(
        kind="artifact",
        uri=uri,
        format="bin",
        sha256=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
        validated=True,
    )


def _artifact_result(*artifacts: ArtifactRef) -> OrganelleResult:
    return OrganelleResult(
        operation_id="demo.worker",
        scope="none",
        status="ok",
        artifacts=artifacts,
    )


def test_execution_run_id_is_bookkeeping_not_result_identity() -> None:
    context = CapabilityExecutionContext(
        operation_id="demo.worker",
        operation_version="1.0",
        callable_locator="demo.worker:run",
        run_id="run-a",
        parameters_hash="a" * 64,
    )
    first = _build_provenance(context)
    second = _build_provenance(context.model_copy(update={"run_id": "run-b"}))

    assert first.run_id == "run-a"
    assert second.run_id == "run-b"
    assert first.object_id == second.object_id


def test_managed_paths_follow_cache_root_without_creating_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "cache"
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(root))

    assert cache_root() == root.resolve()
    assert managed_runs_root() == root.resolve() / "runs"
    assert managed_run_path("assembly.assemble", "sha256-abc") == (
        root.resolve() / "runs" / "assembly.assemble" / "sha256-abc"
    )
    assert not root.exists()


@pytest.mark.parametrize(
    ("operation_id", "run_id"),
    [
        ("../escape", "sha256-abc"),
        ("assembly/assemble", "sha256-abc"),
        ("assembly.assemble", "../escape"),
        ("assembly.assemble", "run/id"),
        ("Assembly.assemble", "sha256-abc"),
    ],
)
def test_managed_run_path_rejects_unsafe_components(
    operation_id: str,
    run_id: str,
) -> None:
    with pytest.raises(OrganelleInputError) as captured:
        managed_run_path(operation_id, run_id)

    assert captured.value.code == "runtime.invalid_run_identity"


def test_publish_run_is_atomic_and_no_replace(tmp_path: Path) -> None:
    temporary = tmp_path / ".temporary"
    completed = tmp_path / "completed"
    temporary.mkdir()
    (temporary / "result.json").write_text("{}\n")

    publish_run(temporary, completed)

    assert not temporary.exists()
    assert (completed / "result.json").read_text() == "{}\n"

    second = tmp_path / ".second"
    second.mkdir()
    (second / "result.json").write_text('{"other":true}\n')
    with pytest.raises(OrganelleInputError) as captured:
        publish_run(second, completed)
    assert captured.value.code == "runtime.run_conflict"
    assert second.is_dir()
    assert (completed / "result.json").read_text() == "{}\n"


def test_remove_managed_entry_unlinks_symlink_without_following_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    published = tmp_path / "published"
    published.mkdir()
    valuable = published / "valuable.txt"
    valuable.write_text("keep me")
    managed_link = managed_run_path("assembly.assemble", "sha256-link")
    managed_link.parent.mkdir(parents=True)
    managed_link.symlink_to(published, target_is_directory=True)

    remove_managed_entry(managed_link)

    assert not managed_link.exists()
    assert valuable.read_text() == "keep me"


def test_remove_managed_entry_refuses_paths_outside_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    outside = tmp_path / "published"
    outside.mkdir()
    (outside / "valuable.txt").write_text("keep me")

    with pytest.raises(OrganelleInputError) as captured:
        remove_managed_entry(outside)

    assert captured.value.code == "runtime.cleanup_outside_cache"
    assert (outside / "valuable.txt").read_text() == "keep me"


def test_remove_managed_entry_rejects_intermediate_symlink_escape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    outside = tmp_path / "outside"
    escaped_run = outside / "sha256-run"
    escaped_run.mkdir(parents=True)
    valuable = escaped_run / "valuable.txt"
    valuable.write_text("keep me")
    operation_link = managed_runs_root() / "assembly.assemble"
    operation_link.parent.mkdir(parents=True)
    operation_link.symlink_to(outside, target_is_directory=True)

    with pytest.raises(OrganelleInputError) as captured:
        remove_managed_entry(operation_link / "sha256-run")

    assert captured.value.code == "runtime.unsafe_managed_path"
    assert valuable.read_text() == "keep me"


def test_staging_is_hidden_sibling_and_refuses_preexisting_leaf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    staging = runtime.staged_run_path("demo.worker", "request-1")
    completed = managed_run_path("demo.worker", "publication-1")

    assert staging == completed.parent / ".staging-request-1"
    assert staging.parent == completed.parent
    created = runtime.create_staged_run("demo.worker", "request-1")
    assert created == staging
    assert created.is_dir()

    with pytest.raises(OrganelleInputError) as captured:
        runtime.create_staged_run("demo.worker", "request-1")

    assert captured.value.code == "runtime.staging_conflict"


def test_staging_creation_refuses_symlinked_operation_ancestor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    outside = tmp_path / "outside"
    outside.mkdir()
    operation = managed_runs_root() / "demo.worker"
    operation.parent.mkdir(parents=True)
    operation.symlink_to(outside, target_is_directory=True)

    with pytest.raises(OrganelleInputError) as captured:
        runtime.create_staged_run("demo.worker", "request-1")

    assert captured.value.code == "runtime.unsafe_managed_path"
    assert not (outside / ".staging-request-1").exists()


def test_staged_verification_rejects_lexical_dotdot_root_outside_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    outside = managed_runs_root().parent / "outside"
    outside.mkdir(parents=True)
    lexical_escape = managed_runs_root() / ".." / "outside"

    with pytest.raises(OrganelleInputError) as captured:
        runtime.verify_staged_artifacts(_artifact_result(), lexical_escape)

    assert captured.value.code == "runtime.staging_outside_cache"


def test_staged_artifact_rejects_external_hardlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    outside = tmp_path / "external.bin"
    outside.write_bytes(b"shared inode")
    staging = runtime.create_staged_run("demo.worker", "request-hardlink")
    os.link(outside, staging / "artifact.bin")

    with pytest.raises(OrganelleInputError) as captured:
        runtime.verify_staged_artifacts(
            _artifact_result(_artifact("artifact.bin", b"shared inode")),
            staging,
        )

    assert captured.value.code == "runtime.unsafe_managed_tree"
    assert outside.read_bytes() == b"shared inode"


def test_staged_artifact_rejects_copied_input_hash_but_allows_new_duplicate_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    content = b"trusted input bytes"
    digest = hashlib.sha256(content).hexdigest()
    copied = runtime.create_staged_run("demo.worker", "request-copy")
    (copied / "artifact.bin").write_bytes(content)

    with pytest.raises(OrganelleInputError) as captured:
        runtime.verify_staged_artifacts(
            _artifact_result(_artifact("artifact.bin", content)),
            copied,
            trusted_input_hashes=frozenset({digest}),
        )

    assert captured.value.code == "runtime.input_artifact_reemission"

    ordinary = runtime.create_staged_run("demo.worker", "request-duplicate-output")
    (ordinary / "first.bin").write_bytes(content)
    (ordinary / "second.bin").write_bytes(content)
    value = _artifact_result(
        _artifact("first.bin", content),
        _artifact("second.bin", content),
    )
    assert runtime.verify_staged_artifacts(value, ordinary) == (
        Path("first.bin"),
        Path("second.bin"),
    )


def test_staged_artifact_read_failure_is_structured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    staging = runtime.create_staged_run("demo.worker", "request-unreadable")
    content = b"verified bytes"
    (staging / "artifact.bin").write_bytes(content)
    value = _artifact_result(_artifact("artifact.bin", content))

    def fail_read(*_args: object, **_kwargs: object) -> ArtifactRef:
        raise PermissionError("injected unreadable staged artifact")

    monkeypatch.setattr(runtime.ArtifactRef, "from_path", fail_read)

    with pytest.raises(OrganelleInputError) as captured:
        runtime.verify_staged_artifacts(value, staging)

    assert captured.value.code == "runtime.artifact_read_failed"
    assert captured.value.as_dict()["details"]["path"] == str(staging / "artifact.bin")


def test_staged_run_is_created_private_to_the_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))

    staging = runtime.create_staged_run("demo.worker", "request-mode")

    assert stat.S_IMODE(staging.stat().st_mode) == 0o700


@pytest.mark.parametrize("blocked_component", ["runs", "operation"])
def test_staging_creation_wraps_regular_file_ancestors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    blocked_component: str,
) -> None:
    cache = tmp_path / "cache"
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(cache))
    if blocked_component == "runs":
        cache.mkdir()
        (cache / "runs").write_text("blocked", encoding="utf-8")
    else:
        managed_runs_root().mkdir(parents=True)
        (managed_runs_root() / "demo.worker").write_text("blocked", encoding="utf-8")

    with pytest.raises(OrganelleExecutionError) as captured:
        runtime.create_staged_run("demo.worker", f"request-{blocked_component}")

    assert captured.value.code == "runtime.staging_create_failed"


@pytest.mark.parametrize(
    "uri",
    [
        "/absolute.bin",
        "../escape.bin",
        "nested/../escape.bin",
        ".",
        "nested/./artifact.bin",
        "nested\\artifact.bin",
        "nested\x00artifact.bin",
        "nested//artifact.bin",
        "C:/anchored.bin",
        "result.json",
    ],
)
def test_staged_artifact_uri_requires_strict_relative_posix_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    uri: str,
) -> None:
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    staging = runtime.create_staged_run("demo.worker", "request-uri")
    safe = staging / "nested" / "artifact.bin"
    safe.parent.mkdir()
    safe.write_bytes(b"verified bytes")

    with pytest.raises(OrganelleInputError) as captured:
        runtime.verify_staged_artifacts(_artifact_result(_artifact(uri)), staging)

    assert captured.value.code == "runtime.invalid_artifact_uri"


def test_staged_artifacts_reject_duplicate_declarations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    staging = runtime.create_staged_run("demo.worker", "request-duplicate")
    (staging / "artifact.bin").write_bytes(b"verified bytes")
    artifact = _artifact("artifact.bin")

    with pytest.raises(OrganelleInputError) as captured:
        runtime.verify_staged_artifacts(_artifact_result(artifact, artifact), staging)

    assert captured.value.code == "runtime.duplicate_artifact"


@pytest.mark.parametrize("attack", ["missing", "extra", "hash", "size", "validated"])
def test_staged_artifact_tree_and_parent_recomputation_are_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attack: str,
) -> None:
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    staging = runtime.create_staged_run("demo.worker", f"request-{attack}")
    content = b"verified bytes"
    declared = _artifact("artifact.bin", content)
    if attack != "missing":
        (staging / "artifact.bin").write_bytes(content)
    if attack == "extra":
        (staging / "undeclared.bin").write_bytes(b"undeclared")
    elif attack == "hash":
        declared = declared.model_copy(update={"sha256": "f" * 64})
    elif attack == "size":
        declared = declared.model_copy(update={"size_bytes": len(content) + 1})
    elif attack == "validated":
        declared = declared.model_copy(update={"validated": False})

    with pytest.raises(OrganelleInputError) as captured:
        runtime.verify_staged_artifacts(_artifact_result(declared), staging)

    expected = {
        "missing": "runtime.artifact_tree_mismatch",
        "extra": "runtime.artifact_tree_mismatch",
        "hash": "runtime.artifact_digest_mismatch",
        "size": "runtime.artifact_digest_mismatch",
        "validated": "runtime.artifact_validation_mismatch",
    }
    assert captured.value.code == expected[attack]


@pytest.mark.parametrize("symlink_kind", ["leaf", "directory"])
def test_staged_artifact_tree_rejects_symlinks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    symlink_kind: str,
) -> None:
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    staging = runtime.create_staged_run("demo.worker", f"request-symlink-{symlink_kind}")
    outside = tmp_path / "outside"
    outside.mkdir()
    external = outside / "artifact.bin"
    external.write_bytes(b"verified bytes")
    if symlink_kind == "leaf":
        (staging / "artifact.bin").symlink_to(external)
        uri = "artifact.bin"
    else:
        (staging / "nested").symlink_to(outside, target_is_directory=True)
        uri = "nested/artifact.bin"

    with pytest.raises(OrganelleInputError) as captured:
        runtime.verify_staged_artifacts(_artifact_result(_artifact(uri)), staging)

    assert captured.value.code == "runtime.unsafe_managed_tree"
    assert external.read_bytes() == b"verified bytes"


@pytest.mark.parametrize("kind", ["result", "data", "genome"])
def test_publish_staged_core_object_supports_all_artifact_bearing_l1_types(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    staging = runtime.create_staged_run("demo.worker", f"request-{kind}")
    completed = managed_run_path("demo.worker", f"publication-{kind}")
    content = b"verified bytes"
    artifact = _artifact("nested/artifact.bin", content)
    path = staging / artifact.uri
    path.parent.mkdir()
    path.write_bytes(content)
    if kind == "result":
        value = _artifact_result(artifact)
    elif kind == "data":
        value = OrganelleData(
            modality="worker_data",
            artifacts=FrozenMap.from_items({"payload": artifact}),
        )
    else:
        value = OrganelleGenome(organelle="mitochondrion", sequence=artifact)
    original_id = value.object_id
    artifact_id = artifact.object_id

    published = runtime.publish_staged_result(value, staging, completed)

    assert published.object_id == original_id
    assert not staging.exists()
    assert (completed / "nested" / "artifact.bin").read_bytes() == content
    if isinstance(published, OrganelleResult):
        relocated = published.artifacts[0]
    elif isinstance(published, OrganelleData):
        relocated = published.artifacts["payload"]
    else:
        assert published.sequence is not None
        relocated = published.sequence
    assert relocated.object_id == artifact_id
    assert relocated.uri == str(completed / "nested" / "artifact.bin")


def test_post_publication_verification_failure_rolls_back_only_new_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    staging = runtime.create_staged_run("demo.worker", "request-tamper")
    completed = managed_run_path("demo.worker", "publication-tamper")
    sibling = managed_run_path("demo.worker", "existing")
    sibling.mkdir(parents=True)
    (sibling / "valuable.bin").write_bytes(b"keep")
    content = b"verified bytes"
    (staging / "artifact.bin").write_bytes(content)
    value = _artifact_result(_artifact("artifact.bin", content))
    original_verify = runtime.verify_staged_artifacts
    original_fsync = runtime._fsync_directory  # pyright: ignore[reportPrivateUsage]
    calls = 0
    fsynced: list[Path] = []

    def fail_after_publish(core: object, root: Path) -> tuple[Path, ...]:
        nonlocal calls
        calls += 1
        if calls == 2:
            (root / "artifact.bin").write_bytes(b"tampered")
        return original_verify(core, root)  # type: ignore[arg-type]

    def record_fsync(path: Path) -> None:
        fsynced.append(path)
        original_fsync(path)

    monkeypatch.setattr(runtime, "verify_staged_artifacts", fail_after_publish)
    monkeypatch.setattr(runtime, "_fsync_directory", record_fsync)

    with pytest.raises(OrganelleExecutionError) as captured:
        runtime.publish_staged_result(value, staging, completed)

    assert captured.value.code == "runtime.rollback_failed"
    assert not completed.exists()
    assert (sibling / "valuable.bin").read_bytes() == b"keep"
    preserved = Path(str(captured.value.as_dict()["details"]["preserved_path"]))
    assert (preserved / "artifact.bin").read_bytes() == b"tampered"
    assert fsynced.count(completed.parent) >= 2


def test_post_publication_rollback_never_deletes_path_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    staging = runtime.create_staged_run("demo.worker", "request-replacement")
    completed = managed_run_path("demo.worker", "publication-replacement")
    moved = managed_run_path("demo.worker", "moved-owned-run")
    content = b"verified bytes"
    (staging / "artifact.bin").write_bytes(content)
    value = _artifact_result(_artifact("artifact.bin", content))
    original_verify = runtime.verify_staged_artifacts
    calls = 0

    def replace_after_publish(core: object, root: Path) -> tuple[Path, ...]:
        nonlocal calls
        calls += 1
        if calls == 2:
            root.rename(moved)
            root.mkdir()
            (root / "valuable.bin").write_bytes(b"replacement")
            raise OrganelleInputError(
                code="runtime.artifact_tree_mismatch",
                message="injected final verification failure",
            )
        return original_verify(core, root)  # type: ignore[arg-type]

    monkeypatch.setattr(runtime, "verify_staged_artifacts", replace_after_publish)

    with pytest.raises(OrganelleExecutionError) as captured:
        runtime.publish_staged_result(value, staging, completed)

    assert captured.value.code == "runtime.rollback_failed"
    assert captured.value.__cause__ is not None
    assert (completed / "valuable.bin").read_bytes() == b"replacement"
    assert (moved / "artifact.bin").read_bytes() == content


def test_post_publication_rollback_never_deletes_replacement_inserted_at_removal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    staging = runtime.create_staged_run("demo.worker", "request-removal-race")
    completed = managed_run_path("demo.worker", "publication-removal-race")
    moved = managed_run_path("demo.worker", "moved-owned-removal-race")
    content = b"verified bytes"
    (staging / "artifact.bin").write_bytes(content)
    value = _artifact_result(_artifact("artifact.bin", content))
    original_verify = runtime.verify_staged_artifacts
    original_rename = runtime._rename_no_replace  # pyright: ignore[reportPrivateUsage]
    verify_calls = 0

    def fail_final_verify(core: object, root: Path) -> tuple[Path, ...]:
        nonlocal verify_calls
        verify_calls += 1
        if verify_calls == 2:
            raise OrganelleInputError(
                code="runtime.artifact_tree_mismatch",
                message="injected final verification failure",
            )
        return original_verify(core, root)  # type: ignore[arg-type]

    def replace_when_rollback_transfer_starts(source: Path, destination: Path) -> None:
        if source == completed:
            source.rename(moved)
            source.mkdir()
            (source / "valuable.bin").write_bytes(b"replacement")
        original_rename(source, destination)

    monkeypatch.setattr(runtime, "verify_staged_artifacts", fail_final_verify)
    monkeypatch.setattr(
        runtime,
        "_rename_no_replace",
        replace_when_rollback_transfer_starts,
    )

    with pytest.raises(OrganelleExecutionError) as captured:
        runtime.publish_staged_result(value, staging, completed)

    assert captured.value.code == "runtime.rollback_failed"
    assert (completed / "valuable.bin").read_bytes() == b"replacement"
    assert (moved / "artifact.bin").read_bytes() == content


def test_post_publication_rollback_fsync_failure_reports_preserved_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    staging = runtime.create_staged_run("demo.worker", "request-rollback-error")
    completed = managed_run_path("demo.worker", "publication-rollback-error")
    content = b"verified bytes"
    (staging / "artifact.bin").write_bytes(content)
    value = _artifact_result(_artifact("artifact.bin", content))
    original_verify = runtime.verify_staged_artifacts
    calls = 0

    def fail_final_verify(core: object, root: Path) -> tuple[Path, ...]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OrganelleInputError(
                code="runtime.artifact_tree_mismatch",
                message="injected final verification failure",
            )
        return original_verify(core, root)  # type: ignore[arg-type]

    original_fsync = runtime._fsync_directory  # pyright: ignore[reportPrivateUsage]

    def fail_rollback_fsync(path: Path) -> None:
        if any(item.name.startswith(f".rollback-{completed.name}-") for item in path.iterdir()):
            raise OSError("injected rollback fsync failure")
        original_fsync(path)

    monkeypatch.setattr(runtime, "verify_staged_artifacts", fail_final_verify)
    monkeypatch.setattr(runtime, "_fsync_directory", fail_rollback_fsync)

    with pytest.raises(OrganelleExecutionError) as captured:
        runtime.publish_staged_result(value, staging, completed)

    assert captured.value.code == "runtime.rollback_failed"
    assert captured.value.__cause__ is not None
    assert not completed.exists()
    preserved = Path(str(captured.value.as_dict()["details"]["preserved_path"]))
    assert preserved.parent == completed.parent
    assert (preserved / "artifact.bin").read_bytes() == content


def test_replacement_restore_fsync_failure_reports_the_restored_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "published"
    owned = tmp_path / "owned"
    owned.mkdir()
    owned_identity = owned.stat(follow_symlinks=False)
    destination.mkdir()
    (destination / "valuable.bin").write_bytes(b"replacement")
    original_fsync = runtime._fsync_directory  # pyright: ignore[reportPrivateUsage]
    fsync_calls = 0

    def fail_after_restore(path: Path) -> None:
        nonlocal fsync_calls
        fsync_calls += 1
        if fsync_calls == 2:
            raise OSError("injected restored replacement fsync failure")
        original_fsync(path)

    monkeypatch.setattr(runtime, "_fsync_directory", fail_after_restore)

    with pytest.raises(OrganelleExecutionError) as captured:
        runtime._rollback_owned_publication(  # pyright: ignore[reportPrivateUsage]
            destination,
            (owned_identity.st_dev, owned_identity.st_ino),
            OrganelleInputError(code="runtime.injected", message="injected failure"),
        )

    assert captured.value.code == "runtime.rollback_failed"
    assert captured.value.as_dict()["details"]["preserved_path"] == str(destination)
    assert (destination / "valuable.bin").read_bytes() == b"replacement"


def test_post_publication_rollback_never_recursively_deletes_quarantine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    staging = runtime.create_staged_run("demo.worker", "request-no-rollback-delete")
    completed = managed_run_path("demo.worker", "publication-no-rollback-delete")
    content = b"verified bytes"
    (staging / "artifact.bin").write_bytes(content)
    value = _artifact_result(_artifact("artifact.bin", content))
    original_verify = runtime.verify_staged_artifacts
    original_remove = runtime._remove_path_no_follow  # pyright: ignore[reportPrivateUsage]
    verify_calls = 0

    def fail_final_verify(core: object, root: Path) -> tuple[Path, ...]:
        nonlocal verify_calls
        verify_calls += 1
        if verify_calls == 2:
            raise OrganelleInputError(
                code="runtime.artifact_tree_mismatch",
                message="injected final verification failure",
            )
        return original_verify(core, root)  # type: ignore[arg-type]

    def reject_rollback_removal(path: Path) -> None:
        if path.name.startswith(f".rollback-{completed.name}-"):
            raise AssertionError("published rollback must preserve rather than delete by path")
        original_remove(path)

    monkeypatch.setattr(runtime, "verify_staged_artifacts", fail_final_verify)
    monkeypatch.setattr(runtime, "_remove_path_no_follow", reject_rollback_removal)

    with pytest.raises(OrganelleExecutionError) as captured:
        runtime.publish_staged_result(value, staging, completed)

    assert captured.value.code == "runtime.rollback_failed"
    assert not completed.exists()
    assert not staging.exists()
    preserved = Path(str(captured.value.as_dict()["details"]["preserved_path"]))
    assert (preserved / "artifact.bin").read_bytes() == content


def test_publish_run_fsyncs_destination_parent_after_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    temporary = tmp_path / ".temporary"
    completed = tmp_path / "completed"
    temporary.mkdir()
    original_fsync = runtime._fsync_directory  # pyright: ignore[reportPrivateUsage]
    calls: list[Path] = []

    def record_fsync(path: Path) -> None:
        calls.append(path)
        original_fsync(path)

    monkeypatch.setattr(runtime, "_fsync_directory", record_fsync)

    runtime.publish_run(temporary, completed)

    assert completed.parent in calls


def test_publish_run_parent_creation_failure_is_structured_before_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    temporary = tmp_path / ".temporary"
    completed = tmp_path / "completed"
    temporary.mkdir()
    original_mkdir = Path.mkdir

    def fail_destination_parent(
        path: Path,
        mode: int = 0o777,
        parents: bool = False,
        exist_ok: bool = False,
    ) -> None:
        if path == completed.parent:
            raise OSError(errno.EIO, "injected destination parent failure")
        original_mkdir(path, mode=mode, parents=parents, exist_ok=exist_ok)

    monkeypatch.setattr(Path, "mkdir", fail_destination_parent)

    with pytest.raises(OrganelleExecutionError) as captured:
        runtime.publish_run(temporary, completed)

    assert captured.value.code == "runtime.run_publication_failed"
    assert temporary.is_dir()
    assert not completed.exists()


def test_publish_run_source_identity_failure_is_structured_before_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    temporary = tmp_path / ".temporary"
    completed = tmp_path / "completed"
    temporary.mkdir()
    original_stat = Path.stat

    def fail_source_identity(path: Path, *, follow_symlinks: bool = True) -> os.stat_result:
        if path == temporary and not follow_symlinks:
            raise OSError(errno.EIO, "injected source identity failure")
        return original_stat(path, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(Path, "stat", fail_source_identity)

    with pytest.raises(OrganelleExecutionError) as captured:
        runtime.publish_run(temporary, completed)

    assert captured.value.code == "runtime.run_publication_failed"
    assert temporary.is_dir()
    assert not completed.exists()


def test_staged_tree_sync_failure_is_structured_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    staging = runtime.create_staged_run("demo.worker", "request-sync-failure")
    completed = managed_run_path("demo.worker", "publication-sync-failure")
    content = b"verified bytes"
    (staging / "artifact.bin").write_bytes(content)
    value = _artifact_result(_artifact("artifact.bin", content))

    def fail_sync(_: Path) -> None:
        raise OSError(errno.EIO, "injected staged tree sync failure")

    monkeypatch.setattr(runtime, "_fsync_tree", fail_sync)

    with pytest.raises(OrganelleExecutionError) as captured:
        runtime.publish_staged_result(value, staging, completed)

    assert captured.value.code == "runtime.staging_sync_failed"
    assert staging.is_dir()
    assert not completed.exists()
