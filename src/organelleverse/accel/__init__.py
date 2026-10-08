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
"""

from __future__ import annotations

import os
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


_kernels: dict[str, Any]
try:
    import organelleverse_rust as _rust

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
