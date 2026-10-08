# pyright: reportPrivateUsage=false
from __future__ import annotations

import errno
import json
import warnings
from pathlib import Path

import pytest

from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.errors import OrganelleExecutionError, OrganelleInputError
from organelleverse.core.result import OrganelleResult
from organelleverse.runtime import managed_run_path, promote_result, remove_managed_entry


def _managed_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[OrganelleResult, Path]:
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    run_dir = managed_run_path("assembly.assemble", "sha256-test")
    normalized = run_dir / "normalized"
    normalized.mkdir(parents=True)
    fasta = normalized / "assembly.fasta"
    manifest = run_dir / "assembly_run_manifest.json"
    fasta.write_text(">contig\nACGT\n")
    manifest.write_text('{"kind":"assembly_run_manifest"}\n')
    result = OrganelleResult(
        operation_id="assembly.assemble",
        scope="mitochondrion",
        status="ok",
        summary_text="assembled",
        artifacts=(
            ArtifactRef.from_path(
                fasta,
                kind="primary_fasta",
                format="fasta",
                media_type="text/x-fasta",
            ),
            ArtifactRef.from_path(
                manifest,
                kind="assembly_run_manifest",
                format="json",
                media_type="application/json",
            ),
        ),
    )
    (run_dir / "result.json").write_text(result.model_dump_json(exclude={"object_id"}) + "\n")
    return result, run_dir


def test_same_filesystem_promotion_moves_tree_and_relocates_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, run_dir = _managed_result(tmp_path, monkeypatch)
    old_artifact_paths = tuple(Path(item.uri) for item in result.artifacts)
    destination = tmp_path / "published"

    relocated = promote_result(result, destination)

    assert relocated.object_id == result.object_id
    assert destination.is_dir()
    assert run_dir.is_symlink()
    assert run_dir.resolve() == destination.resolve()
    assert all(path.is_file() for path in old_artifact_paths)
    assert {Path(item.uri) for item in relocated.artifacts} == {
        destination / "normalized" / "assembly.fasta",
        destination / "assembly_run_manifest.json",
    }
    restored = OrganelleResult.model_validate_json((destination / "result.json").read_bytes())
    assert restored == relocated
    assert not any(
        path.is_dir() and not path.is_symlink()
        for path in (tmp_path / "cache").rglob("*")
        if path.name == "sha256-test"
    )


def test_cross_filesystem_promotion_copies_verifies_commits_and_removes_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import organelleverse.runtime as runtime

    result, run_dir = _managed_result(tmp_path, monkeypatch)
    destination = tmp_path / "published"
    real_rename = runtime._rename_no_replace
    calls = 0

    def cross_device_once(source: Path, target: Path) -> None:
        nonlocal calls
        calls += 1
        if source == run_dir:
            raise OSError(errno.EXDEV, "cross-device link")
        real_rename(source, target)

    monkeypatch.setattr(runtime, "_rename_no_replace", cross_device_once)

    relocated = promote_result(result, destination)

    assert calls >= 2
    assert destination.is_dir()
    assert run_dir.is_symlink()
    assert Path(relocated.artifacts[0].uri).read_text() == ">contig\nACGT\n"
    assert len(list(tmp_path.rglob("assembly.fasta"))) == 1
    assert {path.resolve() for path in tmp_path.rglob("assembly.fasta")} == {
        destination.resolve() / "normalized" / "assembly.fasta"
    }


def test_cross_filesystem_source_cleanup_failure_never_deletes_verified_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import organelleverse.runtime as runtime

    result, run_dir = _managed_result(tmp_path, monkeypatch)
    destination = tmp_path / "published"
    real_rename = runtime._rename_no_replace

    def cross_device_source(source: Path, target: Path) -> None:
        if source == run_dir:
            raise OSError(errno.EXDEV, "cross-device link")
        real_rename(source, target)

    def partial_cleanup(source: Path) -> None:
        (source / "normalized" / "assembly.fasta").unlink()
        raise OSError(errno.EIO, "injected partial cleanup failure")

    monkeypatch.setattr(runtime, "_rename_no_replace", cross_device_source)
    monkeypatch.setattr(runtime, "remove_managed_entry", partial_cleanup)

    with pytest.warns(RuntimeWarning, match="source cleanup"):
        relocated = promote_result(result, destination)

    assert destination.is_dir()
    assert Path(relocated.artifacts[0].uri).read_text() == ">contig\nACGT\n"
    assert not (run_dir / "normalized" / "assembly.fasta").exists()
    assert run_dir.with_name(f"{run_dir.name}.pointer.json").is_file()


def test_cross_filesystem_structured_source_cleanup_failure_keeps_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import organelleverse.runtime as runtime

    result, run_dir = _managed_result(tmp_path, monkeypatch)
    destination = tmp_path / "published"
    real_rename = runtime._rename_no_replace

    def cross_device_source(source: Path, target: Path) -> None:
        if source == run_dir:
            raise OSError(errno.EXDEV, "cross-device link")
        real_rename(source, target)

    def reject_cleanup(source: Path) -> None:
        raise OrganelleInputError(
            code="runtime.unsafe_managed_path",
            message="injected managed source identity change",
            details={"path": str(source)},
        )

    monkeypatch.setattr(runtime, "_rename_no_replace", cross_device_source)
    monkeypatch.setattr(runtime, "remove_managed_entry", reject_cleanup)

    with pytest.warns(RuntimeWarning, match="source cleanup"):
        relocated = promote_result(result, destination)

    assert destination.is_dir()
    assert Path(relocated.artifacts[0].uri).read_text() == ">contig\nACGT\n"
    assert run_dir.is_dir()
    assert run_dir.with_name(f"{run_dir.name}.pointer.json").is_file()


def test_cross_filesystem_cleanup_warning_as_error_never_rolls_back_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import organelleverse.runtime as runtime

    result, run_dir = _managed_result(tmp_path, monkeypatch)
    destination = tmp_path / "published"
    real_rename = runtime._rename_no_replace

    def cross_device_source(source: Path, target: Path) -> None:
        if source == run_dir:
            raise OSError(errno.EXDEV, "cross-device link")
        real_rename(source, target)

    def reject_cleanup(source: Path) -> None:
        raise OrganelleInputError(
            code="runtime.unsafe_managed_path",
            message="injected managed source identity change",
            details={"path": str(source)},
        )

    monkeypatch.setattr(runtime, "_rename_no_replace", cross_device_source)
    monkeypatch.setattr(runtime, "remove_managed_entry", reject_cleanup)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(RuntimeWarning, match="source cleanup"):
            promote_result(result, destination)

    assert destination.is_dir()
    assert (destination / "normalized" / "assembly.fasta").read_text() == ">contig\nACGT\n"
    assert run_dir.is_dir()
    assert not list(tmp_path.glob(".rollback-published-*"))
    assert run_dir.with_name(f"{run_dir.name}.pointer.json").is_file()
    stored = OrganelleResult.model_validate_json((destination / "result.json").read_bytes())
    assert all(Path(artifact.uri).is_relative_to(destination) for artifact in stored.artifacts)
    assert promote_result(result, destination) == stored


def test_promotion_rejects_tampered_artifact_without_touching_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, run_dir = _managed_result(tmp_path, monkeypatch)
    Path(result.artifacts[0].uri).write_text(">contig\nTTTT\n")
    destination = tmp_path / "published"

    with pytest.raises(OrganelleInputError) as captured:
        promote_result(result, destination)

    assert captured.value.code == "runtime.artifact_digest_mismatch"
    assert run_dir.is_dir()
    assert not destination.exists()


def test_promotion_rejects_artifacts_stored_under_wrong_operation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, run_dir = _managed_result(tmp_path, monkeypatch)
    wrong = result.model_copy(update={"operation_id": "qc.assembly"})

    with pytest.raises(OrganelleInputError) as captured:
        promote_result(wrong, tmp_path / "published")

    assert captured.value.code == "runtime.artifact_operation_mismatch"
    assert run_dir.is_dir()


def test_promotion_rejects_managed_run_reached_through_intermediate_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    outside = tmp_path / "outside"
    escaped_run = outside / "sha256-test"
    escaped_run.mkdir(parents=True)
    fasta = escaped_run / "assembly.fasta"
    fasta.write_text(">contig\nACGT\n")
    operation_link = tmp_path / "cache" / "runs" / "assembly.assemble"
    operation_link.parent.mkdir(parents=True)
    operation_link.symlink_to(outside, target_is_directory=True)
    result = OrganelleResult(
        operation_id="assembly.assemble",
        scope="mitochondrion",
        status="ok",
        artifacts=(
            ArtifactRef.from_path(
                operation_link / "sha256-test" / "assembly.fasta",
                kind="primary_fasta",
                format="fasta",
            ),
        ),
    )

    with pytest.raises(OrganelleInputError) as captured:
        promote_result(result, tmp_path / "published")

    assert captured.value.code == "runtime.unsafe_managed_path"
    assert fasta.read_text() == ">contig\nACGT\n"


def test_promotion_rejects_conflict_and_destination_nested_in_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, run_dir = _managed_result(tmp_path, monkeypatch)
    conflict = tmp_path / "published"
    conflict.mkdir()
    (conflict / "unrelated").write_text("do not replace")

    with pytest.raises(OrganelleInputError) as conflict_error:
        promote_result(result, conflict)
    assert conflict_error.value.code == "runtime.destination_conflict"
    assert (conflict / "unrelated").read_text() == "do not replace"

    nested = run_dir / "published"
    with pytest.raises(OrganelleInputError) as nested_error:
        promote_result(result, nested)
    assert nested_error.value.code == "runtime.unsafe_destination"
    assert run_dir.is_dir()


def test_pointer_fallback_contains_no_artifact_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import organelleverse.runtime as runtime

    result, run_dir = _managed_result(tmp_path, monkeypatch)
    destination = tmp_path / "published"

    def unavailable(*_: object, **__: object) -> None:
        raise OSError(errno.EPERM, "symlinks unavailable")

    monkeypatch.setattr(runtime.os, "symlink", unavailable)
    relocated = promote_result(result, destination)

    pointer = run_dir.with_name(f"{run_dir.name}.pointer.json")
    assert not run_dir.exists()
    assert pointer.is_file()
    payload = json.loads(pointer.read_text())
    assert payload["destination"] == str(destination.resolve())
    assert payload["result_id"] == relocated.object_id
    assert not any(path.name == "assembly.fasta" for path in pointer.parent.rglob("*"))

    remove_managed_entry(run_dir)
    assert not pointer.exists()
    assert Path(relocated.artifacts[0].uri).is_file()


def test_pointer_metadata_failure_does_not_fail_verified_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import organelleverse.runtime as runtime

    result, run_dir = _managed_result(tmp_path, monkeypatch)
    destination = tmp_path / "published"

    def fail_pointer(*_: object, **__: object) -> None:
        raise OSError(errno.EIO, "injected pointer failure")

    monkeypatch.setattr(runtime, "_install_managed_pointer", fail_pointer)

    with pytest.warns(RuntimeWarning, match="managed pointer"):
        relocated = promote_result(result, destination)

    assert destination.is_dir()
    assert not run_dir.exists()
    assert Path(relocated.artifacts[0].uri).read_text() == ">contig\nACGT\n"
    assert (
        OrganelleResult.model_validate_json((destination / "result.json").read_bytes()) == relocated
    )


@pytest.mark.parametrize("cross_filesystem", [False, True])
def test_failed_post_commit_verification_preserves_every_owned_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cross_filesystem: bool,
) -> None:
    import organelleverse.runtime as runtime

    result, run_dir = _managed_result(tmp_path, monkeypatch)
    destination = tmp_path / "published"
    original_result_json = (run_dir / "result.json").read_bytes()
    if cross_filesystem:
        real_rename = runtime._rename_no_replace

        def cross_device_source(source: Path, target: Path) -> None:
            if source == run_dir:
                raise OSError(errno.EXDEV, "cross-device link")
            real_rename(source, target)

        monkeypatch.setattr(runtime, "_rename_no_replace", cross_device_source)

    def fail_verification(*_: object, **__: object) -> None:
        raise OrganelleExecutionError(
            code="runtime.promotion_verification_failed",
            message="injected verification failure",
        )

    monkeypatch.setattr(runtime, "_verify_promoted_tree", fail_verification)

    with pytest.raises(OrganelleExecutionError) as captured:
        promote_result(result, destination)

    assert captured.value.code == "runtime.rollback_failed"
    recovery = Path(str(captured.value.as_dict()["details"]["preserved_path"]))
    assert (recovery / "normalized" / "assembly.fasta").is_file()
    assert not destination.exists()
    if cross_filesystem:
        assert run_dir.is_dir()
        assert not run_dir.is_symlink()
        assert (run_dir / "result.json").read_bytes() == original_result_json
        expected_copies = 2
    else:
        assert not run_dir.exists()
        assert (recovery / "result.json").read_bytes() == original_result_json
        expected_copies = 1
    assert len(list(tmp_path.rglob("assembly.fasta"))) == expected_copies


def test_cross_filesystem_rollback_never_deletes_destination_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import organelleverse.runtime as runtime

    result, run_dir = _managed_result(tmp_path, monkeypatch)
    destination = tmp_path / "published"
    moved = tmp_path / "moved-owned-publication"
    real_rename = runtime._rename_no_replace

    def cross_device_then_replace(source: Path, target: Path) -> None:
        if source == run_dir:
            raise OSError(errno.EXDEV, "cross-device link")
        if source == destination:
            source.rename(moved)
            source.mkdir()
            (source / "valuable.bin").write_bytes(b"replacement")
        real_rename(source, target)

    def fail_verification(*_: object, **__: object) -> None:
        raise OrganelleExecutionError(
            code="runtime.promotion_verification_failed",
            message="injected verification failure",
        )

    monkeypatch.setattr(runtime, "_rename_no_replace", cross_device_then_replace)
    monkeypatch.setattr(runtime, "_verify_promoted_tree", fail_verification)

    with pytest.raises(OrganelleExecutionError) as captured:
        promote_result(result, destination)

    assert captured.value.code == "runtime.rollback_failed"
    assert (destination / "valuable.bin").read_bytes() == b"replacement"
    assert (moved / "normalized" / "assembly.fasta").read_text() == ">contig\nACGT\n"
    assert run_dir.is_dir()


def test_same_filesystem_rollback_never_relocates_destination_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import organelleverse.runtime as runtime

    result, run_dir = _managed_result(tmp_path, monkeypatch)
    destination = tmp_path / "published"
    moved = tmp_path / "moved-owned-publication"

    def replace_during_verification(*_: object, **__: object) -> None:
        destination.rename(moved)
        destination.mkdir()
        (destination / "valuable.bin").write_bytes(b"replacement")
        raise OrganelleExecutionError(
            code="runtime.promotion_verification_failed",
            message="injected verification failure",
        )

    monkeypatch.setattr(runtime, "_verify_promoted_tree", replace_during_verification)

    with pytest.raises(OrganelleExecutionError) as captured:
        promote_result(result, destination)

    assert captured.value.code == "runtime.rollback_failed"
    assert (destination / "valuable.bin").read_bytes() == b"replacement"
    assert (moved / "normalized" / "assembly.fasta").read_text() == ">contig\nACGT\n"
    assert not run_dir.exists()


@pytest.mark.parametrize("failure_site", ["artifact_verification", "tree_manifest"])
def test_promotion_preflight_io_failure_is_structured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_site: str,
) -> None:
    import organelleverse.runtime as runtime

    result, run_dir = _managed_result(tmp_path, monkeypatch)
    destination = tmp_path / "published"

    def fail_io(*_: object, **__: object) -> None:
        raise OSError(errno.EIO, f"injected {failure_site} failure")

    if failure_site == "artifact_verification":
        monkeypatch.setattr(runtime, "_verify_result_artifacts", fail_io)
    else:
        monkeypatch.setattr(runtime, "_tree_manifest", fail_io)

    with pytest.raises(OrganelleExecutionError) as captured:
        promote_result(result, destination)

    assert captured.value.code == "runtime.promotion_failed"
    assert run_dir.is_dir()
    assert not destination.exists()


def test_reused_publication_artifact_read_failure_is_structured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import organelleverse.runtime as runtime

    result, _ = _managed_result(tmp_path, monkeypatch)
    destination = tmp_path / "published"
    promote_result(result, destination)

    def fail_read(*_: object, **__: object) -> ArtifactRef:
        raise OSError(errno.EIO, "injected published artifact read failure")

    monkeypatch.setattr(runtime.ArtifactRef, "from_path", fail_read)

    with pytest.raises(OrganelleInputError) as captured:
        promote_result(result, destination)

    assert captured.value.code == "runtime.destination_conflict"
    assert destination.is_dir()


def test_cross_filesystem_rollback_never_recursively_deletes_published_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import organelleverse.runtime as runtime

    result, run_dir = _managed_result(tmp_path, monkeypatch)
    destination = tmp_path / "published"
    real_rename = runtime._rename_no_replace
    real_remove = runtime._remove_path_no_follow

    def cross_device_source(source: Path, target: Path) -> None:
        if source == run_dir:
            raise OSError(errno.EXDEV, "cross-device link")
        real_rename(source, target)

    def fail_verification(*_: object, **__: object) -> None:
        raise OrganelleExecutionError(
            code="runtime.promotion_verification_failed",
            message="injected verification failure",
        )

    def reject_published_copy_removal(path: Path) -> None:
        if path.name.startswith(f".rollback-{destination.name}-"):
            raise AssertionError("published rollback must preserve rather than delete by path")
        real_remove(path)

    monkeypatch.setattr(runtime, "_rename_no_replace", cross_device_source)
    monkeypatch.setattr(runtime, "_verify_promoted_tree", fail_verification)
    monkeypatch.setattr(runtime, "_remove_path_no_follow", reject_published_copy_removal)

    with pytest.raises(OrganelleExecutionError) as captured:
        promote_result(result, destination)

    assert captured.value.code == "runtime.rollback_failed"
    details = captured.value.as_dict()["details"]
    recovery = Path(str(details["preserved_path"]))
    assert not destination.exists()
    assert (recovery / "normalized" / "assembly.fasta").read_text() == ">contig\nACGT\n"
    assert run_dir.is_dir()
