"""scp between this machine and a compute host, safe for Windows local paths.

scp tells remote from local by a colon: ``host:path`` is remote. A Windows local
path carries one too (``C:\\Users\\...``). Windows' own OpenSSH scp special-cases
drive letters, but another scp earlier on PATH (Git for Windows' msys build,
for one) reads ``C:\\data\\a.tif`` as "path \\data\\a.tif on host C". So the local
side is never passed as an absolute path: each call runs in the local file's
directory and names it ``./<name>``, which has no colon on any platform (and a
leading ``./`` also keeps a name starting with ``-`` from reading as an option).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

SCP = ("scp", "-q", "-o", "BatchMode=yes")


def _local(name: str) -> str:
    return f"./{name}"


def upload(runner: Callable[..., Any], host: str, files: Iterable[Path], remote_dir: str, *, timeout: float) -> None:
    """Copy local files into ``remote_dir`` on ``host``, keeping their names."""
    by_parent: dict[Path, list[str]] = {}
    for path in files:
        by_parent.setdefault(Path(path).parent, []).append(Path(path).name)
    for parent, names in by_parent.items():
        runner(
            [*SCP, *map(_local, names), f"{host}:{remote_dir}/"],
            cwd=str(parent), check=True, capture_output=True, text=True, timeout=timeout,
        )


def upload_as(runner: Callable[..., Any], host: str, local: Path, remote_path: str, *, timeout: float) -> None:
    """Copy one local file to an explicit remote path (a rename on arrival)."""
    local = Path(local)
    runner(
        [*SCP, _local(local.name), f"{host}:{remote_path}"],
        cwd=str(local.parent), check=True, capture_output=True, text=True, timeout=timeout,
    )


def download(runner: Callable[..., Any], host: str, remote_paths: Iterable[str], local_dir: Path, *, timeout: float) -> None:
    """Copy remote files into the local directory ``local_dir``."""
    runner(
        [*SCP, *(f"{host}:{path}" for path in remote_paths), "."],
        cwd=str(local_dir), check=True, capture_output=True, text=True, timeout=timeout,
    )
