"""Translation with the bacterial/plastid code (NCBI table 11), the same result as ``Bio.Seq.Seq.translate(table=11)``.

Splice refinement translates tens of thousands of candidate spliced sequences per gene (Ginkgo: 650,000 calls,
most of a plastome annotation), and Biopython translates codon by codon in Python. An upper-case A/C/G/T sequence
whose length is a multiple of three is translated here through a 64-entry lookup table built from Biopython's own
table 11; anything else (lower case, N or other ambiguity codes, a partial codon) is handed to Biopython unchanged,
so ambiguous codons and warnings behave exactly as before.
"""

from __future__ import annotations

import numpy as np
from Bio.Data.CodonTable import unambiguous_dna_by_id
from Bio.Seq import Seq

_BASES = "TCAG"
_ENCODE = bytes.maketrans(b"TCAG", b"\x00\x01\x02\x03")


def _lookup() -> np.ndarray:
    table = unambiguous_dna_by_id[11]
    out = np.zeros(64, dtype=np.uint8)
    for i, a in enumerate(_BASES):
        for j, b in enumerate(_BASES):
            for k, c in enumerate(_BASES):
                codon = a + b + c
                aa = "*" if codon in table.stop_codons else table.forward_table[codon]
                out[16 * i + 4 * j + k] = ord(aa)
    return out


_TABLE = _lookup()


def translate11(nt: str) -> str:
    """``str(Seq(nt).translate(table=11))``."""
    if len(nt) % 3 or not nt.isascii():
        return str(Seq(nt).translate(table=11))
    codes = np.frombuffer(nt.encode("ascii").translate(_ENCODE), dtype=np.uint8)
    if codes.size == 0:
        return ""
    if codes.max() > 3:  # anything but upper-case TCAG
        return str(Seq(nt).translate(table=11))
    idx = (codes[0::3].astype(np.intp) << 4) | (codes[1::3].astype(np.intp) << 2) | codes[2::3]
    return _TABLE[idx].tobytes().decode("ascii")
