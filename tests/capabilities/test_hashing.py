from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import BinaryIO, Never

import pytest

from organelleverse.capabilities.hashing import hash_bundle, hash_fixture_dataset, hash_tree
from organelleverse.core.errors import OrganelleContractError


def _expected(entries: list[tuple[str, bytes]]) -> str:
    leaves = [[name, hashlib.sha256(content).hexdigest()] for name, content in entries]
    canonical = json.dumps(
        sorted(leaves), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return f"sha256:{hashlib.sha256(canonical).hexdigest()}"


def test_hash_tree_is_the_canonical_sorted_relative_path_merkle(tmp_path: Path) -> None:
    (tmp_path / "z.txt").write_bytes(b"last")
    nested = tmp_path / "a"
    nested.mkdir()
    (nested / "first.bin").write_bytes(b"first")

    assert hash_tree(tmp_path) == _expected([("a/first.bin", b"first"), ("z.txt", b"last")])


class _BoundedReader:
    def __init__(self, handle: BinaryIO, read_sizes: list[int]) -> None:
        self._handle = handle
        self._read_sizes = read_sizes

    def __enter__(self) -> _BoundedReader:
        return self

    def __exit__(self, *_args: object) -> None:
        self._handle.close()

    def read(self, size: int = -1) -> bytes:
        assert 0 < size <= 1024 * 1024
        self._read_sizes.append(size)
        return self._handle.read(size)


def test_hash_tree_streams_bounded_chunks_without_changing_the_canonical_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = b"bounded-streaming" * 150_000
    (tmp_path / "large.bin").write_bytes(content)
    read_sizes: list[int] = []
    real_open = Path.open

    def forbid_read_bytes(_path: Path) -> Never:
        raise AssertionError("hash_tree must not load a whole file with Path.read_bytes")

    def tracking_open(path: Path, mode: str = "r") -> _BoundedReader:
        assert mode == "rb"
        return _BoundedReader(real_open(path, mode), read_sizes)

    monkeypatch.setattr(Path, "read_bytes", forbid_read_bytes)
    monkeypatch.setattr(Path, "open", tracking_open)

    assert hash_tree(tmp_path) == _expected([("large.bin", content)])
    assert len(read_sizes) >= 3
    assert max(read_sizes) <= 1024 * 1024


def test_hash_tree_translates_stream_read_oserrors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "unreadable.bin"
    target.write_bytes(b"content")

    def fail_open(_path: Path, mode: str = "r") -> Never:
        raise OSError(f"cannot read in {mode}")

    monkeypatch.setattr(Path, "open", fail_open)

    with pytest.raises(OrganelleContractError) as captured:
        hash_tree(tmp_path)

    assert captured.value.code == "capability.bundle_read_failed"
    assert captured.value.as_dict()["details"]["path"] == str(target)


def test_bundle_hash_covers_every_file_including_fixtures(tmp_path: Path) -> None:
    (tmp_path / "capability.toml").write_text("schema='x'", encoding="utf-8")
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    fixture = fixtures / "input.txt"
    fixture.write_text("one", encoding="utf-8")
    before = hash_bundle(tmp_path)
    fixture.write_text("two", encoding="utf-8")
    assert hash_bundle(tmp_path) != before


def test_fixture_hash_covers_only_the_fixture_subtree(tmp_path: Path) -> None:
    (tmp_path / "capability.toml").write_text("first", encoding="utf-8")
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    (fixtures / "expected.json").write_text("{}", encoding="utf-8")
    before = hash_fixture_dataset(tmp_path)
    (tmp_path / "capability.toml").write_text("second", encoding="utf-8")
    assert hash_fixture_dataset(tmp_path) == before
    (fixtures / "expected.json").write_text('{"changed":true}', encoding="utf-8")
    assert hash_fixture_dataset(tmp_path) != before


def test_empty_fixture_dataset_has_a_stable_hash(tmp_path: Path) -> None:
    assert hash_fixture_dataset(tmp_path) == _expected([])


def test_hashing_rejects_a_symlinked_file_instead_of_hashing_outside_the_bundle(
    tmp_path: Path,
) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside.txt"
    outside.write_text("secret", encoding="utf-8")
    (tmp_path / "escape").symlink_to(outside)

    with pytest.raises(OrganelleContractError) as captured:
        hash_tree(tmp_path)

    assert captured.value.code == "capability.bundle_path_unsafe"
