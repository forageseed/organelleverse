"""Optional Rust acceleration.

A kernel is bound to its public name only when an equivalence fixture pins it
against the Python reference it replaces. Everything else resolves to ``None``,
so callers take their pure-Python path instead. That is not caution for its own
sake: a native kernel that computes something subtly different is worse than no
kernel at all, because the difference is silent.

``kmer_jaccard`` is the historical reason this gate exists: the crate's
original kernel shared ``encode_kmers`` with ``kmer_overlap`` (which indexes
each k-mer *and its reverse complement* for MTPT/NUMT detection), so the
strand-specific barcode Jaccard diverged from its Python reference
(``AAAA`` vs ``TTTT`` at k=3 gave 1.0 instead of 0.0) and was withheld. The
crate has since grown a strand-specific entry point
(``encode_kmers_stranded``: forward k-mers only, N-windows skipped, input
upper-cased); the kernel now matches the barcode reference exactly and is
pinned like the others. The gate itself stays: a native kernel that computes
something subtly different is worse than no kernel at all, because the
difference is silent.

Callers must still guard on the kernel itself (``if accel.kmer_overlap is not
None``), never on :data:`HAS_RUST`: a build can be present and a kernel still be
withheld.

To benchmark a withheld kernel, set ``ORGANELLEVERSE_ACCEL_UNVERIFIED`` to a
comma-separated list of names, or ``all``. That is an explicit, per-process
decision — never a default.

The crate is found without installing it: when ``organelleverse_rust`` is not
importable, the directory named by ``ORGANELLEVERSE_NATIVE_DIR`` and then the
checkout's ``rust/target/release`` are searched for the built library, so a
mounted build directory or a plain ``cargo build --release`` is picked up
without ``PYTHONPATH``. Only the kernel's Python/ABI tag and its name decide
what loads; the equivalence gate above still decides what is bound.
"""

from __future__ import annotations

from organelleverse._layout import checkout_root

import importlib.machinery
import importlib.util
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

__all__ = [
    "HAS_RUST",
    "UNVERIFIED_KERNELS",
    "VERIFIED_KERNELS",
    "available_kernels",
    "cm_cyk",
    "cm_cyk_trace",
    "erc_correlation",
    "kmer_jaccard",
    "kmer_overlap",
    "mafft_align",
    "withheld_kernels",
]

#: Kernels with an equivalence fixture pinning them to the Python reference.
#: mafft_align / kmer_overlap / erc_correlation / kmer_jaccard are pinned in
#: tests/accel/test_kernel_equivalence.py (mafft vs the mafft CLI, kmer scan
#: vs the transfer module's Python scan, ERC vs scipy.stats, jaccard vs the
#: barcode reference after the strand-specific fix).
VERIFIED_KERNELS = (
    "cm_cyk",
    "cm_cyk_trace",
    "erc_correlation",
    "kmer_jaccard",
    "kmer_overlap",
    "mafft_align",
)

#: Kernels present in the crate but not yet pinned against a reference
#: implementation (none currently — the mechanism stays for the next kernel
#: that needs it).
UNVERIFIED_KERNELS = ()

_KERNEL_NAMES = (*VERIFIED_KERNELS, *UNVERIFIED_KERNELS)
_OPT_IN_ENV = "ORGANELLEVERSE_ACCEL_UNVERIFIED"


def _opted_in() -> frozenset[str]:
    """Kernel names this process explicitly accepts without verification."""
    raw = os.environ.get(_OPT_IN_ENV, "").strip()
    if not raw:
        return frozenset()
    if raw.lower() == "all":
        return frozenset(UNVERIFIED_KERNELS)
    return frozenset(part.strip() for part in raw.split(",") if part.strip())


NATIVE_DIR_ENV = "ORGANELLEVERSE_NATIVE_DIR"


def _native_directories() -> Iterator[Path]:
    """Where a built ``organelleverse_rust`` may sit when it is not importable."""

    configured = os.environ.get(NATIVE_DIR_ENV, "").strip()
    if configured:
        yield Path(configured)
    # <checkout>/src/accel/__init__.py -> <checkout>/rust/target/release
    yield checkout_root() / "rust" / "target" / "release"


def _load_from_directory(directory: Path) -> ModuleType | None:
    """Load ``organelleverse_rust`` from a built library in ``directory``, or None."""

    if not directory.is_dir():
        return None
    suffixes = tuple(importlib.machinery.EXTENSION_SUFFIXES)
    for path in sorted(directory.glob("*organelleverse_rust*")):
        if not path.name.endswith(suffixes):
            continue
        try:
            loader = importlib.machinery.ExtensionFileLoader("organelleverse_rust", str(path))
            spec = importlib.util.spec_from_file_location(
                "organelleverse_rust", str(path), loader=loader
            )
            if spec is None:
                continue
            module = importlib.util.module_from_spec(spec)
            loader.exec_module(module)
        except (ImportError, OSError):
            continue
        sys.modules["organelleverse_rust"] = module
        return module
    return None


def _import_native() -> ModuleType:
    try:
        import organelleverse_rust

        return organelleverse_rust
    except ImportError:
        pass
    for directory in _native_directories():
        module = _load_from_directory(directory)
        if module is not None:
            return module
    raise ImportError("organelleverse_rust is not installed and was not found in "
                      f"{NATIVE_DIR_ENV} or rust/target/release")


_kernels: dict[str, Any]
try:
    _rust = _import_native()

    HAS_RUST: bool = True
    _opt_in = _opted_in()
    _kernels = {
        name: getattr(_rust, name, None) if name in VERIFIED_KERNELS or name in _opt_in else None
        for name in _KERNEL_NAMES
    }
except ImportError:
    HAS_RUST = False
    _kernels = dict.fromkeys(_KERNEL_NAMES)

cm_cyk: Any = _kernels["cm_cyk"]
cm_cyk_trace: Any = _kernels["cm_cyk_trace"]
erc_correlation: Any = _kernels["erc_correlation"]
kmer_jaccard: Any = _kernels["kmer_jaccard"]
kmer_overlap: Any = _kernels["kmer_overlap"]
mafft_align: Any = _kernels["mafft_align"]


def available_kernels() -> tuple[str, ...]:
    """Return the kernels callers will actually reach, in declaration order."""
    return tuple(name for name in _KERNEL_NAMES if _kernels[name] is not None)


def withheld_kernels() -> tuple[str, ...]:
    """Return kernels the build provides but this layer is withholding.

    Non-empty means the crate is ahead of its equivalence fixtures. Each name
    here is a native implementation nobody has proven equivalent to the Python
    path it would replace.
    """
    if not HAS_RUST:
        return ()
    return tuple(name for name in UNVERIFIED_KERNELS if _kernels[name] is None)
