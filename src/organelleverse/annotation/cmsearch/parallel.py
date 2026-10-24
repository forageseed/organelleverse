"""Thread fan-out for the covariance-model search.

The Rust CYK kernel releases the GIL (``py.allow_threads``), so independent windows and splice candidates can be scored
on several threads. ``pmap`` keeps the input order, so callers build exactly the lists a serial loop built and results
do not depend on the thread count. The count is set per call site with ``cyk_threads`` (default 1: serial); worker
threads do not inherit it, so a ``pmap`` inside a worker runs serially instead of nesting pools.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")

_THREADS: ContextVar[int] = ContextVar("cyk_threads", default=1)


@contextlib.contextmanager
def cyk_threads(n: int) -> Iterator[None]:
    """Use up to ``n`` threads for covariance-model scoring inside this block."""
    token = _THREADS.set(max(1, int(n)))
    try:
        yield
    finally:
        _THREADS.reset(token)


def cyk_thread_count() -> int:
    """Threads set by the enclosing ``cyk_threads`` block (1 outside one)."""
    return _THREADS.get()


def pmap(fn: Callable[[T], R], items: Iterable[T]) -> list[R]:
    """``[fn(x) for x in items]``, on up to ``cyk_threads`` threads, in input order."""
    items = list(items)
    n = min(_THREADS.get(), len(items))
    if n <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(max_workers=n) as pool:
        return list(pool.map(fn, items))
