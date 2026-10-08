"""Content-integrity checks: does a downloaded file look like what it claims.

Catches a WAF/error/app-shell page saved with a 200 status and a genome
file's extension — observed directly against TAIR while writing this plan
(``index-auto.jsp`` returns an identical 2192-byte HTML shell for every
path, real or fabricated), not a hypothetical failure mode.
"""

from __future__ import annotations

import gzip
from pathlib import Path

from ..core.errors import OrganelleExecutionError

__all__ = [
    "SHARED_INCLUDE_KINDS",
    "looks_like_fasta",
    "looks_like_gff",
    "validate_downloaded_content",
]

# The full ``include`` vocabulary shared across GIR/IMP/PGD/TAIR (NCBI's own
# ``_INCLUDE`` in nuclear.py stays separate and unaffected — it never uses
# ``tpm``). A source that does not offer a given kind treats a request for it
# as a per-record miss, never an ``input.unknown_include`` error — only a
# value outside this whole set is a real parameter error. See the design
# doc's "include — mapping table".
SHARED_INCLUDE_KINDS = frozenset({"protein", "cds", "gff3", "genome", "tpm", "rna", "seq-report"})


def looks_like_fasta(data: bytes) -> bool:
    return data.lstrip()[:1] == b">"


def looks_like_gff(data: bytes) -> bool:
    for line in data.splitlines():
        if not line or line.startswith(b"#"):
            continue
        if len(line.split(b"\t")) >= 9:
            return True
    return False


def validate_downloaded_content(path: Path, *, kind: str, gzipped: bool, source: str) -> None:
    """Raise ``network.<source>.content_invalid`` if ``path`` is not plausible content for ``kind``."""
    raw = path.read_bytes()
    body = gzip.decompress(raw) if gzipped else raw
    ok = looks_like_gff(body) if kind == "gff3" else looks_like_fasta(body)
    if not ok:
        raise OrganelleExecutionError(
            code=f"network.{source}.content_invalid",
            message=f"downloaded {kind} for {source} does not look like {kind} content",
            details={"path": str(path), "kind": kind},
            retryable=True,
        )
