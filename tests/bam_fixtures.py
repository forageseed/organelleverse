"""Construct a minimal BAM header for real Rust-backed test fixture writers.

rubam accepts its native Header object, but cannot construct one from a dict.
The BAM wire header below contains no records; all alignment records are still
written, sorted, indexed and read through rubam's actual compiled implementation.
"""

import struct
from pathlib import Path

import rubam
from Bio.bgzf import BgzfWriter


def bam_header(header: dict, directory: Path):
    lines = []
    for tag, values in header.items():
        for row in values if isinstance(values, list) else [values]:
            lines.append(
                "@" + tag + "\t" + "\t".join(f"{key}:{value}" for key, value in row.items())
            )
    text = ("\n".join(lines) + "\n").encode()
    data = b"BAM\x01" + struct.pack("<i", len(text)) + text
    references = header.get("SQ", [])
    data += struct.pack("<i", len(references))
    for reference in references:
        name = reference["SN"].encode() + b"\0"
        data += struct.pack("<i", len(name)) + name + struct.pack("<i", reference["LN"])
    path = directory / "fixture-header.bam"
    with BgzfWriter(str(path), "wb") as output:
        output.write(data)
    with rubam.AlignmentFile(str(path), "rb") as source:
        return source.header
