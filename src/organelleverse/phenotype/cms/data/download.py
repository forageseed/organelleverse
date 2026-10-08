"""Download only accession-checked CMS seeds into the managed cache.

The packaged reviewed reference and legacy BLAST files are never overwritten.
No BLAST database is required by the current direct-subject assessment.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from tempfile import NamedTemporaryFile

from ...._bio import read_fasta
from ....core.errors import OrganelleInputError
from ..resources import cms_protein_fasta, cms_reference_json


def download_cms_sequences(email: str = "user@example.com") -> Path:
    """Fetch pinned reviewed proteins and validate them before atomic cache publication.

    Every accession, expected length and complete amino-acid sequence must match
    the packaged reviewed seed. Failure leaves any earlier cached file untouched.
    ``ORGANELLEVERSE_CACHE_ROOT`` selects the cache through the existing runtime.
    """
    from Bio import Entrez, SeqIO

    from ....runtime import cache_root

    Entrez.email = email
    references = json.loads(cms_reference_json().read_text(encoding="utf-8"))
    expected = dict(read_fasta(cms_protein_fasta()))
    records = []
    for index, metadata in enumerate(references):
        if index:
            time.sleep(0.4)
        accession = metadata["accession"]
        with Entrez.efetch(db="protein", id=accession, rettype="fasta", retmode="text") as handle:
            record = SeqIO.read(handle, "fasta")
        if (
            record.id != accession
            or len(record) != metadata["length_aa"]
            or str(record.seq) != expected[metadata["name"]]
        ):
            raise OrganelleInputError(
                code="cms.reference_download_mismatch",
                message=f"Downloaded {accession} does not match the reviewed protein accession/length/sequence",
            )
        record.id = metadata["name"]
        record.description = f"protein_id={accession} nucleotide={metadata['nucleotide_accession']}"
        records.append(record)
    target = cache_root() / "references" / "cms" / "cms_proteins.fasta"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=target.parent, suffix=".fasta", delete=False
        ) as handle:
            temporary = Path(handle.name)
            SeqIO.write(records, handle, "fasta")
        temporary.replace(target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target


if __name__ == "__main__":
    download_cms_sequences()
