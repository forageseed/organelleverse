"""Tests for cross-execution-context path translation (T-A1.2)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from organelleverse.core.errors import OrganelleInputError
from organelleverse.environments.path_translation import (
    ExecutionContext,
    translate_path,
    validate_path_accessible,
)


def test_host_to_wsl_windows_drive() -> None:
    result = translate_path(
        "C:\\data\\reads.fq",
        source_context=ExecutionContext.HOST,
        target_context=ExecutionContext.WSL,
    )
    assert result.translated_value == "/mnt/c/data/reads.fq"
    assert result.original_value == "C:\\data\\reads.fq"


def test_wsl_to_host_drive() -> None:
    result = translate_path(
        "/mnt/d/project/sample.fasta",
        source_context=ExecutionContext.WSL,
        target_context=ExecutionContext.HOST,
    )
    assert result.translated_value == "D:\\project\\sample.fasta"


def test_host_to_container_with_mount_map() -> None:
    result = translate_path(
        "C:\\data\\batch1\\x.fasta",
        source_context=ExecutionContext.HOST,
        target_context=ExecutionContext.CONTAINER,
        mount_map={"C:\\data": "/data"},
    )
    assert result.translated_value == "/data/batch1/x.fasta"


def test_host_mount_root_to_container_mount_root() -> None:
    result = translate_path(
        "C:\\data",
        source_context=ExecutionContext.HOST,
        target_context=ExecutionContext.CONTAINER,
        mount_map={"C:\\data": "/data"},
    )
    assert result.translated_value == "/data"


def test_container_to_host_with_mount_map() -> None:
    result = translate_path(
        "/data/batch1/x.fasta",
        source_context=ExecutionContext.CONTAINER,
        target_context=ExecutionContext.HOST,
        mount_map={"C:\\data": "/data"},
    )
    assert result.translated_value == "C:\\data\\batch1\\x.fasta"


def test_container_mount_root_to_host_mount_root() -> None:
    result = translate_path(
        "/data",
        source_context=ExecutionContext.CONTAINER,
        target_context=ExecutionContext.HOST,
        mount_map={"C:\\data": "/data"},
    )
    assert result.translated_value == "C:\\data"


@pytest.mark.parametrize(
    ("mount_map", "host_path", "container_path"),
    [
        ({"/": "/work"}, "/srv/sample.fa", "/work/srv/sample.fa"),
        ({"/srv": "/"}, "/srv/sample.fa", "/sample.fa"),
        ({"C:\\": "/host"}, "C:\\sample.fa", "/host/sample.fa"),
    ],
)
def test_root_mounts_translate_children_in_both_directions(
    mount_map: dict[str, str], host_path: str, container_path: str
) -> None:
    forward = translate_path(
        host_path,
        source_context=ExecutionContext.HOST,
        target_context=ExecutionContext.CONTAINER,
        mount_map=mount_map,
    )
    reverse = translate_path(
        container_path,
        source_context=ExecutionContext.CONTAINER,
        target_context=ExecutionContext.HOST,
        mount_map=mount_map,
    )
    assert forward.translated_value == container_path
    assert reverse.translated_value == host_path


def test_container_to_host_preserves_unix_host_separator() -> None:
    result = translate_path(
        "/data/batch1/x.fasta",
        source_context=ExecutionContext.CONTAINER,
        target_context=ExecutionContext.HOST,
        mount_map={"/home/user/data": "/data"},
    )
    assert result.translated_value == "/home/user/data/batch1/x.fasta"


def test_same_context_is_identity() -> None:
    for context in (ExecutionContext.HOST, ExecutionContext.WSL, ExecutionContext.CONTAINER):
        result = translate_path("/some/path", source_context=context, target_context=context)
        assert result.translated_value == "/some/path"


def test_remote_translation_is_fail_closed() -> None:
    with pytest.raises(OrganelleInputError, match="remote"):
        translate_path(
            "/data/x.fasta",
            source_context=ExecutionContext.HOST,
            target_context=ExecutionContext.REMOTE,
        )
    with pytest.raises(OrganelleInputError, match="remote"):
        translate_path(
            "/data/x.fasta",
            source_context=ExecutionContext.REMOTE,
            target_context=ExecutionContext.HOST,
        )


def test_wsl_outside_mnt_is_fail_closed() -> None:
    with pytest.raises(OrganelleInputError, match="not under /mnt"):
        translate_path(
            "/home/user/x.fasta",
            source_context=ExecutionContext.WSL,
            target_context=ExecutionContext.HOST,
        )


def test_container_requires_mount_map() -> None:
    with pytest.raises(OrganelleInputError, match="mount_map"):
        translate_path(
            "C:\\data\\x.fasta",
            source_context=ExecutionContext.HOST,
            target_context=ExecutionContext.CONTAINER,
        )


@pytest.mark.parametrize(
    ("path", "source", "target"),
    [
        ("/data/../secret", ExecutionContext.CONTAINER, ExecutionContext.HOST),
        ("/mnt/c/../../secret", ExecutionContext.WSL, ExecutionContext.HOST),
        ("C:\\data\\..\\secret", ExecutionContext.HOST, ExecutionContext.CONTAINER),
    ],
)
def test_parent_traversal_is_fail_closed(
    path: str, source: ExecutionContext, target: ExecutionContext
) -> None:
    with pytest.raises(OrganelleInputError) as captured:
        translate_path(
            path,
            source_context=source,
            target_context=target,
            mount_map={"C:\\data": "/data"},
        )
    assert captured.value.code == "execution_context.path_traversal"


@pytest.mark.parametrize(
    "mount_map",
    [
        {"relative": "/data"},
        {"/host/data": "relative"},
        {"/host/../secret": "/data"},
        {"/host/a": "/data", "/host/b": "/data"},
    ],
)
def test_invalid_or_ambiguous_mount_maps_are_rejected(
    mount_map: dict[str, str],
) -> None:
    with pytest.raises(OrganelleInputError) as captured:
        translate_path(
            "/data/file",
            source_context=ExecutionContext.CONTAINER,
            target_context=ExecutionContext.HOST,
            mount_map=mount_map,
        )
    assert captured.value.code in {
        "execution_context.mount_map_invalid",
        "execution_context.mount_map_ambiguous",
        "execution_context.path_traversal",
    }


def test_validate_path_accessible_rejects_missing() -> None:
    with pytest.raises(OrganelleInputError, match="does not exist"):
        validate_path_accessible("/nonexistent/organelleverse-test-path")


@pytest.mark.skipif(sys.platform == "win32", reason="unix-only permission test")
def test_validate_path_accessible_accepts_readable_file(tmp_path: Path) -> None:
    path = tmp_path / "sample.fasta"
    path.write_text(">sample\nACGT")
    resolved = validate_path_accessible(str(path))
    assert resolved == path.resolve()
