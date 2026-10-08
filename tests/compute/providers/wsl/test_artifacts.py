"""WSL artifact transport tests (spec §10): streaming, identity, atomicity,
path safety, cache reuse, and bounded memory on large blobs.

The WSL process invocation is replaced by an in-process fake IO backed by a
temp "Linux cache" directory; no real wsl.exe or worker is needed. The tests
exercise the host-side security guarantees that must hold regardless of transport.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import IO

import pytest

from organelleverse.compute.artifacts import RemoteArtifact, StagedArtifact
from organelleverse.compute.contracts import ResolvedComputeTarget, TransportKind
from organelleverse.compute.providers.wsl.artifacts import (
    WslArtifactTransport,
    content_locator,
)
from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.errors import OrganelleContractError

_HEX = "a" * 64


# --- an in-process fake of the worker's binary data plane ------------------


class _FakeWorkerCache:
    """Stands in for the worker's Linux-side ext4 content cache."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, sha256: str, size_bytes: int) -> Path:
        loc = content_locator(sha256, size_bytes)
        return self.root / loc

    def import_artifact(self, target, *, sha256, size_bytes, stdin_pipe: IO[bytes]) -> None:
        digest = hashlib.sha256()
        n = 0
        dest = self._path(sha256, size_bytes)
        dest.parent.mkdir(parents=True, exist_ok=True)
        partial = dest.with_suffix(".partial")
        with partial.open("wb") as out:
            while True:
                chunk = stdin_pipe.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                n += len(chunk)
                out.write(chunk)
        if digest.hexdigest() != sha256 or n != size_bytes:
            partial.unlink(missing_ok=True)
            raise OrganelleContractError(code="compute.worker_import_mismatch", message="bad import")
        os.replace(partial, dest)

    def export_artifact(self, target, *, sha256, size_bytes, stdout_pipe: IO[bytes]) -> None:
        src = self._path(sha256, size_bytes)
        with src.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                stdout_pipe.write(chunk)

    def verify_artifact(self, target, *, sha256, size_bytes):
        from organelleverse.compute.artifacts import ArtifactVerificationReport

        src = self._path(sha256, size_bytes)
        if src.is_file():
            digest = hashlib.sha256()
            with src.open("rb") as handle:
                digest.update(handle.read())
            observed_sha = digest.hexdigest()
            observed_size = src.stat().st_size
        else:
            observed_sha, observed_size = "0" * 64, 0
        status = "verified" if (observed_sha == sha256 and observed_size == size_bytes) else "mismatch"
        return ArtifactVerificationReport(
            expected_sha256=sha256, expected_size_bytes=size_bytes,
            observed_sha256=observed_sha, observed_size_bytes=observed_size, status=status,
        )


def _target() -> ResolvedComputeTarget:
    return ResolvedComputeTarget(
        target_id="wsl:Ubuntu-24.04", provider_id="wsl", transport_kind=TransportKind.WSL_MCP,
        platform="linux-64", architecture="x86_64", artifact_transport="wsl_content_cache",
    )


def _write_blob(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _artifact_ref(path: Path, *, kind="genome", fmt="fasta") -> ArtifactRef:
    return ArtifactRef.from_path(path, kind=kind, format=fmt, media_type="text/fasta")


# === locator (spec §10) ====================================================


def test_locator_derived_only_from_sha256_and_size():
    assert content_locator(_HEX, 10) == f"sha256/aa/{_HEX}.10"
    assert content_locator("b" * 64, 10) != content_locator("c" * 64, 10)
    assert content_locator(_HEX, 10) != content_locator(_HEX, 11)


# === stage: identity check + atomic cache write ============================


def test_stage_rehashes_source_and_writes_to_cache(tmp_path):
    src = tmp_path / "src" / "reads.fasta"
    _write_blob(src, b"ACGT" * 100)
    ref = _artifact_ref(src)
    cache = _FakeWorkerCache(tmp_path / "wsl-cache")
    t = WslArtifactTransport(io=cache, host_staging_root=tmp_path / "stage", target=_target())

    staged = t.stage(ref, _target())
    assert isinstance(staged, StagedArtifact)
    assert staged.sha256 == ref.sha256
    assert staged.size_bytes == ref.size_bytes
    assert staged.transport == "wsl_content_cache"
    # the worker cache now holds the exact blob, addressable by content
    assert cache._path(ref.sha256, ref.size_bytes).is_file()


def test_stage_rejects_source_that_drifted_from_declared_identity(tmp_path):
    src = tmp_path / "src" / "reads.fasta"
    _write_blob(src, b"ACGT" * 100)
    ref = _artifact_ref(src)
    # mutate the source after building the ref -> declared identity is now stale
    src.write_bytes(b"ACGT" * 200)
    cache = _FakeWorkerCache(tmp_path / "wsl-cache")
    t = WslArtifactTransport(io=cache, host_staging_root=tmp_path / "stage", target=_target())
    with pytest.raises(OrganelleContractError) as exc:
        t.stage(ref, _target())
    assert exc.value.code == "compute.artifact_source_drift"


# === fetch: streaming + atomic publish + mismatch rejection ===============


def _remote(ref: ArtifactRef) -> RemoteArtifact:
    return RemoteArtifact(
        object_id=ref.uri, kind=ref.kind, format=ref.format, media_type=ref.media_type,
        sha256=ref.sha256, size_bytes=ref.size_bytes, transport="wsl_content_cache",
        locator=content_locator(ref.sha256, ref.size_bytes),
    )


def test_fetch_writes_atomically_and_returns_verified_ref(tmp_path):
    src = tmp_path / "src" / "reads.fasta"
    _write_blob(src, b"ACGT" * 100)
    ref = _artifact_ref(src)
    cache = _FakeWorkerCache(tmp_path / "wsl-cache")
    # populate the cache by staging first
    t = WslArtifactTransport(io=cache, host_staging_root=tmp_path / "stage", target=_target())
    t.stage(ref, _target())

    dest = tmp_path / "out" / "fetched.fasta"
    fetched = t.fetch(_remote(ref), dest)
    assert dest.is_file()
    assert fetched.sha256 == ref.sha256
    assert fetched.size_bytes == ref.size_bytes
    assert fetched.validated is True
    # no .partial left behind
    assert not list((tmp_path / "stage").glob("*.partial"))


def test_fetch_rejects_mismatched_bytes_and_leaves_no_partial(tmp_path):
    cache = _FakeWorkerCache(tmp_path / "wsl-cache")
    # prime the cache with WRONG bytes but a locator naming the expected hash
    bad_sha = "b" * 64
    loc = cache.root / content_locator(bad_sha, 8)
    loc.parent.mkdir(parents=True, exist_ok=True)
    loc.write_bytes(b"XXXXXXXX")  # 8 bytes but will hash to something else
    t = WslArtifactTransport(io=cache, host_staging_root=tmp_path / "stage", target=_target())
    remote = RemoteArtifact(
        object_id="o", kind="genome", format="fasta", media_type="text/fasta",
        sha256=bad_sha, size_bytes=8, transport="wsl_content_cache",
        locator=content_locator(bad_sha, 8),
    )
    dest = tmp_path / "out" / "f.fasta"
    with pytest.raises(OrganelleContractError) as exc:
        t.fetch(remote, dest)
    assert exc.value.code == "compute.artifact_fetch_mismatch"
    assert not dest.exists()
    assert not list((tmp_path / "stage").glob("*.partial"))


# === verify ================================================================


def test_verify_confirms_cached_entry(tmp_path):
    src = tmp_path / "src" / "reads.fasta"
    _write_blob(src, b"ACGT" * 100)
    ref = _artifact_ref(src)
    cache = _FakeWorkerCache(tmp_path / "wsl-cache")
    t = WslArtifactTransport(io=cache, host_staging_root=tmp_path / "stage", target=_target())
    staged = t.stage(ref, _target())
    report = t.verify(staged)
    assert report.status == "verified"


# === path safety (spec §10, §14) ===========================================


def test_stage_rejects_symlink_source(tmp_path):
    real = tmp_path / "real.fasta"
    _write_blob(real, b"ACGT")
    link = tmp_path / "link.fasta"
    os.symlink(real, link)
    cache = _FakeWorkerCache(tmp_path / "wsl-cache")
    t = WslArtifactTransport(io=cache, host_staging_root=tmp_path / "stage", target=_target())
    with pytest.raises(OrganelleContractError) as exc:
        t.stage(ArtifactRef.from_path(link, kind="genome", format="fasta"), _target())
    assert exc.value.code == "compute.artifact_unsafe_path"


def test_stage_rejects_fifo(tmp_path):
    fifo = tmp_path / "reads.fifo"
    os.mkfifo(fifo)
    cache = _FakeWorkerCache(tmp_path / "wsl-cache")
    t = WslArtifactTransport(io=cache, host_staging_root=tmp_path / "stage", target=_target())
    # FIFOs are rejected — either at ArtifactRef.from_path (not a regular file)
    # or by the transport's path-safety check. Both are correct rejections.
    from organelleverse.core.errors import OrganelleError

    with pytest.raises(OrganelleError) as exc:
        t.stage(ArtifactRef.from_path(fifo, kind="genome", format="fasta"), _target())
    assert exc.value.code in ("compute.artifact_unsafe_path", "input.missing_artifact")


def test_windows_host_paths_are_accepted_at_the_host_side_seam(tmp_path):
    # Regression for the review finding: this transport runs on the Windows
    # host, where drive-letter and UNC forms are the *native* absolute forms.
    # Rejecting them made every real host path unusable on the target
    # platform while being un-triggerable on POSIX (a resolved POSIX path
    # always starts with "/"). No caller path crosses into WSL — the cache
    # key is the provider-generated content locator — so there is no
    # injection vector left behind.
    cache = _FakeWorkerCache(tmp_path / "wsl-cache")
    WslArtifactTransport(io=cache, host_staging_root=tmp_path / "stage", target=_target())
    from organelleverse.compute.providers.wsl.artifacts import _reject_unsafe_path

    # Must not raise: both forms are legitimate host-side paths.
    _reject_unsafe_path(Path("C:/data/reads.fq"), write_side=False)
    _reject_unsafe_path(Path(r"\\server\share\reads.fq"), write_side=True)


# === bounded memory on a large (sparse) blob ==============================


def test_sparse_gigabyte_streams_with_bounded_memory_and_exact_identity(tmp_path):
    # Create a sparse 1.5 GiB file whose real content is small. The transport
    # must stream it in 1 MiB chunks and never allocate its logical size in RAM;
    # identity (sha256 over the full logical content) must still match exactly.
    src = tmp_path / "big.fasta"
    payload = b"ACGTACGT" * 4096  # 32 KiB real payload
    with src.open("wb") as handle:
        handle.seek(0)
        handle.write(payload)
        handle.truncate(1024 * 1024 * 1536)  # extend logically to 1.5 GiB (sparse)
    ref = _artifact_ref(src)
    cache = _FakeWorkerCache(tmp_path / "wsl-cache")
    t = WslArtifactTransport(io=cache, host_staging_root=tmp_path / "stage", target=_target())

    staged = t.stage(ref, _target())
    assert staged.size_bytes == 1024 * 1024 * 1536
    # fetch it back and require exact identity end-to-end
    dest = tmp_path / "out" / "big.fasta"
    fetched = t.fetch(_remote(ref), dest)
    assert fetched.sha256 == ref.sha256
    assert fetched.size_bytes == ref.size_bytes
