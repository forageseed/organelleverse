"""Real PMAT2 release gate using the official Malus HiFi fixture."""

from __future__ import annotations

import hashlib
import os
import shutil
import time
import urllib.request
from pathlib import Path

import pytest

from organelleverse.assembly.api import assemble
from organelleverse.assembly.contracts import LongReadLibrary, PmatParameters
from organelleverse.assembly.manifests import AssemblyRunManifest
from organelleverse.io_reads import read_reads

_FIXTURE_URL = (
    "https://github.com/bichangwei/PMAT/releases/download/v1.1.0/Malus_domestica.540Mb.fasta.gz"
)
_FIXTURE_SHA256 = "0b44bfb960bce31a9438bab449216623098bc6cf9829d4ee973901578460d7ca"
_FIXTURE_SIZE = 160_885_268


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fixture() -> Path:
    cache_home = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    destination = cache_home / "organelleverse" / "fixtures" / "pmat" / Path(_FIXTURE_URL).name
    if (
        destination.is_file()
        and destination.stat().st_size == _FIXTURE_SIZE
        and _sha256(destination) == _FIXTURE_SHA256
    ):
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with urllib.request.urlopen(_FIXTURE_URL) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        if temporary.stat().st_size != _FIXTURE_SIZE or _sha256(temporary) != _FIXTURE_SHA256:
            raise AssertionError("downloaded PMAT fixture identity does not match the release gate")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _fasta_bases(path: Path) -> int:
    return sum(
        len(line.strip())
        for line in path.read_text(encoding="ascii").splitlines()
        if line and not line.startswith(">")
    )


@pytest.mark.release_assembly_pmat
def test_official_malus_mitochondrial_assembly_and_verified_reuse(tmp_path: Path) -> None:
    assert str(tmp_path).startswith("/dev/shm/"), "release gate must run on /dev/shm"
    fixture = _fixture()
    data = read_reads(
        long_libraries=(
            LongReadLibrary(
                technology="pacbio_hifi",
                quality_state="ccs",
                reads=fixture,
            ),
        )
    )
    parameters = PmatParameters(keep_sequences_in_memory=True)

    first = assemble(
        data,
        organelle="mitochondrion",
        method="pmat",
        backend_parameters=parameters,
        threads=32,
        timeout_seconds=3600,
    )

    assert first.status == "ok"
    assert first.provenance is not None
    assert first.provenance.actual_backend == "pmat"
    assert (
        first.provenance.software_versions["pmat"]
        == "PMAT2 2.1.5 + OrganelleVerse orientation patch"
    )
    record_path = next(
        artifact.resolve() for artifact in first.artifacts if artifact.kind == "assembly_run_record"
    )
    run_dir = record_path.parent
    manifest = AssemblyRunManifest.model_validate_json(record_path.read_bytes())
    assert manifest.selected_backend == "pmat"
    assert manifest.organelle == "mitochondrion"
    fasta = run_dir / "normalized" / "assembly.fasta"
    graph = run_dir / "normalized" / "assembly.gfa"
    assessment = run_dir / "normalized" / "assessment.txt"
    assert 350_000 <= _fasta_bases(fasta) <= 450_000
    assert graph.is_file() and "S\t" in graph.read_text(encoding="ascii")
    assert "24/24 (100.0%)" in assessment.read_text(encoding="ascii")

    started = time.monotonic()
    second = assemble(
        data,
        organelle="mitochondrion",
        method="pmat",
        backend_parameters=parameters,
        threads=32,
        timeout_seconds=3600,
    )
    assert second.object_id == first.object_id
    assert time.monotonic() - started < 60
