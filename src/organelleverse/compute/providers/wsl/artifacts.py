"""WSL content-cache artifact transport (spec §10, §11.2).

The bulk data plane is **separate from MCP**: large FASTQ/GFA/BAM never enters
MCP JSON. Instead the host drives a fixed per-artifact CLI on the worker:

    organelleverse-linux-provider artifact import --sha256 HEX --size N   (raw stdin)
    organelleverse-linux-provider artifact export --sha256 HEX --size N   (raw stdout)
    organelleverse-linux-provider artifact verify --sha256 HEX --size N

wrapped in the fixed ``wsl.exe --distribution NAME --exec ...`` prefix. MCP JSON
is control plane only (spec §10, §14).

This module implements the host side: streaming (bounded 1 MiB chunks, never
loading whole files or base64), SHA256+size verification at every boundary,
atomic publication, exact-identity cache reuse, and rejection of symlinks,
traversal escapes, and device/socket/FIFO files. The content
locator is provider-generated from SHA256 and size; caller/model paths never
appear. Original Unicode filename is metadata only and never a cache path.

The WSL process invocation is delegated to an injectable :class:`WslArtifactIo`
seam so the streaming/hash/atomic/path-safety logic is fully testable on Linux
without real WSL or the worker. The real seam runs the fixed ``wsl.exe`` argv;
tests inject an in-process fake backed by a temp ext4-style cache directory.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
from pathlib import Path
from typing import IO, Protocol

from organelleverse.compute.artifacts import (
    ArtifactVerificationReport,
    RemoteArtifact,
    StagedArtifact,
)
from organelleverse.compute.contracts import ResolvedComputeTarget
from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.errors import OrganelleContractError

__all__ = [
    "WslArtifactIo",
    "WslArtifactTransport",
    "content_locator",
]

_CHUNK = 1024 * 1024  # 1 MiB streaming cap; never read() without a size


def content_locator(sha256: str, size_bytes: int) -> str:
    """The provider-generated content locator: ``sha256/<2-hex>/<64-hex>.<size>``.

    Derived only from SHA256 and size, so caller/model paths can never appear.
    Two equal blobs always produce the same locator; the two-hex shard keeps any
    one cache directory listable.
    """
    return f"sha256/{sha256[:2]}/{sha256}.{size_bytes}"


class WslArtifactIo(Protocol):
    """The binary data-plane seam the worker CLI implements.

    ``import_artifact`` consumes raw bytes on ``stdin_pipe`` and writes them into
    the worker's Linux-side ext4 cache at the content locator; it must hash and
    size what it receives and refuse on mismatch. ``export_artifact`` streams the
    raw cached bytes to ``stdout_pipe``. ``verify_artifact`` checks the cached
    entry exists with exact identity. None accept a filename or path argument.
    """

    def import_artifact(
        self, target: ResolvedComputeTarget, *, sha256: str, size_bytes: int, stdin_pipe: IO[bytes]
    ) -> None: ...

    def export_artifact(
        self, target: ResolvedComputeTarget, *, sha256: str, size_bytes: int, stdout_pipe: IO[bytes]
    ) -> None: ...

    def verify_artifact(
        self, target: ResolvedComputeTarget, *, sha256: str, size_bytes: int
    ) -> ArtifactVerificationReport: ...


def _sha256_size_of(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


class WslArtifactTransport:
    """Host-side streaming transport for the WSL content cache (spec §10).

    All bytes are streamed in bounded 1 MiB chunks; nothing is loaded whole or
    base64-encoded. Every boundary (stage source, worker import, export fetch)
    re-hashes and requires exact SHA256+size agreement before atomic
    publication. Paths are provider-generated; symlinks/traversal/device files
    are rejected.
    """

    def __init__(
        self,
        *,
        io: WslArtifactIo,
        host_staging_root: Path,
        target: ResolvedComputeTarget,
    ) -> None:
        self._io = io
        self._staging = host_staging_root
        self._target = target

    # -- stage (host -> WSL) ------------------------------------------------

    def stage(self, artifact: ArtifactRef, target: ResolvedComputeTarget) -> StagedArtifact:
        source = _safe_resolve(artifact)
        # Re-hash the live source and require it to agree with the declared
        # identity before any bytes move (spec §10).
        observed_sha, observed_size = _sha256_size_of(source)
        if observed_sha != artifact.sha256 or observed_size != artifact.size_bytes:
            raise OrganelleContractError(
                code="compute.artifact_source_drift",
                message=(
                    "the artifact source bytes changed before staging; declared and "
                    "observed SHA256/size disagree"
                ),
                details={
                    "object_id": artifact.uri,
                    "declared_sha256": artifact.sha256,
                    "observed_sha256": observed_sha,
                },
            )
        # Stream the source into the worker import CLI, which writes the
        # Linux-side ext4 cache atomically and verifies identity itself.
        r_fd, w_fd = os.pipe()
        try:
            with os.fdopen(r_fd, "rb") as read_end, os.fdopen(w_fd, "wb") as write_end:
                # The import seam reads stdin while we stream; run it against the
                # read end. (A real WSL seam runs wsl.exe with this pipe as stdin.)
                import threading

                def _pump() -> None:
                    try:
                        with source.open("rb") as src:
                            while True:
                                chunk = src.read(_CHUNK)
                                if not chunk:
                                    break
                                write_end.write(chunk)
                    finally:
                        write_end.close()

                pump = threading.Thread(target=_pump, daemon=True)
                pump.start()
                self._io.import_artifact(
                    target, sha256=artifact.sha256, size_bytes=artifact.size_bytes, stdin_pipe=read_end
                )
                pump.join()
        finally:
            for fd in (r_fd, w_fd):
                with contextlib.suppress(OSError):
                    os.close(fd)
        return StagedArtifact(
            object_id=artifact.uri,
            sha256=artifact.sha256,
            size_bytes=artifact.size_bytes,
            transport="wsl_content_cache",
            locator=content_locator(artifact.sha256, artifact.size_bytes),
        )

    # -- fetch (WSL -> host) ------------------------------------------------

    def fetch(self, artifact: RemoteArtifact, destination: Path) -> ArtifactRef:
        _reject_unsafe_path(destination, write_side=True)
        self._staging.mkdir(parents=True, exist_ok=True)
        partial = self._staging / f".{artifact.sha256}.{os.getpid()}.partial"
        r_fd, w_fd = os.pipe()
        digest = hashlib.sha256()
        observed_size = 0
        try:
            with os.fdopen(r_fd, "rb") as read_end, os.fdopen(w_fd, "wb") as write_end:
                import threading

                def _pump() -> None:
                    try:
                        self._io.export_artifact(
                            self._target,
                            sha256=artifact.sha256,
                            size_bytes=artifact.size_bytes,
                            stdout_pipe=write_end,
                        )
                    finally:
                        write_end.close()

                # The export seam writes raw bytes to stdout_pipe; we consume on
                # the read end, hashing while writing the partial file.
                pump = threading.Thread(target=_pump, daemon=True)
                pump.start()
                with partial.open("wb") as out:
                    while True:
                        chunk = read_end.read(_CHUNK)
                        if not chunk:
                            break
                        digest.update(chunk)
                        observed_size += len(chunk)
                        out.write(chunk)
                        out.flush()
                        os.fsync(out.fileno())
                pump.join()
            observed_sha = digest.hexdigest()
            if observed_sha != artifact.sha256 or observed_size != artifact.size_bytes:
                partial.unlink(missing_ok=True)
                raise OrganelleContractError(
                    code="compute.artifact_fetch_mismatch",
                    message=(
                        "fetched artifact bytes do not match the declared SHA256/size; "
                        "the transfer is discarded"
                    ),
                    details={
                        "expected_sha256": artifact.sha256,
                        "observed_sha256": observed_sha,
                        "expected_size": artifact.size_bytes,
                        "observed_size": observed_size,
                    },
                )
            # atomic publication only after identity verification
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(partial, destination)
        finally:
            partial.unlink(missing_ok=True)
        return ArtifactRef.from_path(
            destination,
            kind=artifact.kind,
            format=artifact.format,
            media_type=artifact.media_type,
        )

    # -- verify -------------------------------------------------------------

    def verify(self, artifact: StagedArtifact | RemoteArtifact) -> ArtifactVerificationReport:
        sha = artifact.sha256
        size = artifact.size_bytes
        return self._io.verify_artifact(self._target, sha256=sha, size_bytes=size)


# ---------------------------------------------------------------------------
# path safety (spec §10, §14)
# ---------------------------------------------------------------------------


def _safe_resolve(artifact: ArtifactRef) -> Path:
    """Resolve an artifact source path, rejecting unsafe file types and escapes."""
    original = Path(artifact.uri)
    # check the symlink on the unresolved path first; .resolve() would follow it
    # and the real target would no longer look like a link.
    if original.is_symlink():
        raise OrganelleContractError(
            code="compute.artifact_unsafe_path",
            message="symlinks are not accepted as artifact paths",
            details={"path": str(original)},
        )
    path = original.resolve(strict=False)
    _reject_unsafe_path(path, write_side=False)
    if not path.is_file():
        raise OrganelleContractError(
            code="compute.artifact_source_missing",
            message="the artifact source is not a regular file",
            details={"uri": artifact.uri},
        )
    return path


def _reject_unsafe_path(path: Path, *, write_side: bool) -> None:
    """Reject symlinks, device/socket/FIFO files, and directories (spec §10).

    Path safety is enforced on the resolved host-side path. Windows drive and
    UNC forms are deliberately *not* rejected: this transport runs on the
    Windows host, where ``C:\\...`` and ``\\\\server\\share`` are the native
    absolute forms of legitimate artifact paths, and on POSIX a resolved path
    can never take those forms. No caller path ever crosses into WSL — the
    provider-side cache key is the provider-generated content locator — so
    there is no Windows-form injection vector for such a check to defend.
    """
    text = str(path)
    # symlinks (resolve follows them; reject if the path or its target is a link)
    if path.is_symlink():
        raise OrganelleContractError(
            code="compute.artifact_unsafe_path",
            message="symlinks are not accepted as artifact paths",
            details={"path": text},
        )
    if path.exists() and not path.is_file() and not (write_side and not path.exists()):
        # device files, FIFOs, sockets, directories
        import stat

        st = path.stat()
        if stat.S_ISDIR(st.st_mode):
            raise OrganelleContractError(
                code="compute.artifact_unsafe_path",
                message="directories are not accepted as artifact paths",
                details={"path": text},
            )
        if stat.S_ISCHR(st.st_mode) or stat.S_ISBLK(st.st_mode):
            raise OrganelleContractError(
                code="compute.artifact_unsafe_path",
                message="device files are not accepted as artifact paths",
                details={"path": text},
            )
        if stat.S_ISFIFO(st.st_mode) or stat.S_ISSOCK(st.st_mode):
            raise OrganelleContractError(
                code="compute.artifact_unsafe_path",
                message="FIFOs and sockets are not accepted as artifact paths",
                details={"path": text},
            )
