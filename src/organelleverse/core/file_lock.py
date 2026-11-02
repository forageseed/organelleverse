"""Cross-platform exclusive locking of an already-open file descriptor."""
import importlib
import os
from collections.abc import Callable
from typing import cast

def _lock_file_descriptor(descriptor: int) -> None:
    module = importlib.import_module("msvcrt" if os.name == "nt" else "fcntl")
    if os.name == "nt":  # pragma: no cover - exercised on Windows desktop builds
        if os.fstat(descriptor).st_size == 0:
            os.write(descriptor, b"\0")
            os.fsync(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        locking = cast(Callable[[int, int, int], None], module.locking)
        locking(descriptor, cast(int, module.LK_LOCK), 1)
        return
    flock = cast(Callable[[int, int], None], module.flock)
    flock(descriptor, cast(int, module.LOCK_EX))

def _unlock_file_descriptor(descriptor: int) -> None:
    module = importlib.import_module("msvcrt" if os.name == "nt" else "fcntl")
    if os.name == "nt":  # pragma: no cover - exercised on Windows desktop builds
        os.lseek(descriptor, 0, os.SEEK_SET)
        locking = cast(Callable[[int, int, int], None], module.locking)
        locking(descriptor, cast(int, module.LK_UNLCK), 1)
        return
    flock = cast(Callable[[int, int], None], module.flock)
    flock(descriptor, cast(int, module.LOCK_UN))
