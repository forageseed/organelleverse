"""Tests for the v2 nuclear-download fix (live-verified 2026-08-15).

The legacy GET-by-accession download form returns HTTP 400 for every caller;
v2 requires POST + JSON body. Anonymous POST answers with a metadata-only
stub package (README/catalog/jsonl under ncbi_dataset/data/, no sequence
payload) which must be refused with the exact remedy rather than written
out as the requested assembly.
"""

from __future__ import annotations

import io
import zipfile

import pytest

from organelleverse.core.errors import OrganelleExecutionError
from organelleverse.fetch.nuclear import _require_real_dataset_package


def _zip(entries: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def test_stub_package_is_refused_with_the_remedy() -> None:
    stub = _zip(
        {
            "ncbi_dataset_data/README.md": b"boilerplate",
            "ncbi_dataset/data/assembly_data_report.jsonl": b"{}",
            "ncbi_dataset/data/dataset_catalog.json": b"{}",
            "md5sum.txt": b"",
        }
    )
    with pytest.raises(OrganelleExecutionError) as exc:
        _require_real_dataset_package(stub, accessions=["GCF_1"], include=("protein",))
    assert exc.value.code == "fetch.nuclear_download_stub"
    assert "API key" in exc.value.message


def test_real_package_with_sequence_payload_passes() -> None:
    real = _zip(
        {
            "ncbi_dataset_data/README.md": b"boilerplate",
            "ncbi_dataset/data/dataset_catalog.json": b"{}",
            "ncbi_dataset/data/GCF_1/protein.faa": b">tr|p1\nMKV",
        }
    )
    _require_real_dataset_package(real, accessions=["GCF_1"], include=("protein",))


def test_non_zip_bytes_pass_through() -> None:
    # not a zip: downstream consumers report it; the guard stays silent
    _require_real_dataset_package(b"not a zip", accessions=["GCF_1"], include=())
