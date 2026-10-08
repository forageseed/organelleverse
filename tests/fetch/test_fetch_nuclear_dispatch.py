"""fetch_nuclear_genome's source dispatch: default stays NCBI-only, and
``auto`` tries sources in priority order with atomic, non-polluting publish.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest

from organelleverse.core.errors import OrganelleExecutionError
from organelleverse.fetch._http import HttpResponse
from organelleverse.fetch.nuclear import fetch_nuclear_genome

_EMPTY_DATASETS_REPORT = json.dumps({"reports": []}).encode()


def _datasets_report(accession: str, organism: str) -> bytes:
    return json.dumps(
        {
            "reports": [
                {
                    "accession": accession,
                    "assembly_info": {"assembly_level": "Chromosome"},
                    "organism": {"organism_name": organism, "tax_id": "1"},
                    "assembly_stats": {},
                }
            ]
        }
    ).encode()


class _NcbiMissTransport:
    """NCBI always misses; GIR/IMP/PGD are never reached in a dispatch-only test."""

    def get(self, url: str, *, timeout: float = 120.0, headers=None) -> bytes:
        return _EMPTY_DATASETS_REPORT

    def post(self, url: str, data: bytes, *, timeout: float = 120.0, headers=None) -> bytes:
        raise AssertionError("not used in this test")

    def head(self, url: str, *, timeout: float = 30.0, headers=None) -> int:
        return 404

    def get_response(self, url: str, **kwargs) -> HttpResponse:
        return HttpResponse(status=404, body=b"{}")

    def download_to_path(self, url: str, dest: Path, **kwargs) -> int:
        return 404


def test_the_default_source_is_ncbi_and_existing_behavior_is_unchanged(tmp_path: Path) -> None:
    class NcbiHit(_NcbiMissTransport):
        def get(self, url: str, *, timeout: float = 120.0, headers=None) -> bytes:
            return _datasets_report("GCF_000001.1", "Malus domestica")

    data = fetch_nuclear_genome(
        taxon="Malus domestica", dest=tmp_path, download=False, transport=NcbiHit()
    )
    assert data.payload["manifest"]["source"] == "nuclear_assembly"
    assert data.payload["records"][0]["accession"] == "GCF_000001.1"


def test_an_unknown_source_is_rejected() -> None:
    with pytest.raises(Exception) as raised:  # OrganelleParameterError
        fetch_nuclear_genome(taxon="Malus domestica", dest="unused", source="bogus")  # type: ignore[arg-type]
    assert getattr(raised.value, "code", None) == "input.unknown_source"


def test_a_direct_source_call_annotates_ncbi_only_params_as_not_applying(tmp_path: Path) -> None:
    """GIR needs an ``ftp_client`` (exercised in test_fetch_gir.py), so this
    checks the shared ``_annotate_ncbi_only_params`` wrapper via TAIR instead
    — any non-NCBI source goes through the exact same wrapper."""

    class TairHit(_NcbiMissTransport):
        def get_response(self, url: str, **kwargs) -> HttpResponse:
            if "dir=" in url:
                return HttpResponse(status=200, body=b"{}")
            return HttpResponse(status=404, body=b"{}")

        def download_to_path(self, url: str, dest: Path, **kwargs) -> int:
            dest.parent.mkdir(parents=True, exist_ok=True)
            if "TAIR10_chr_all.fas.gz" in url:
                import gzip

                dest.write_bytes(gzip.compress(b">Chr1\nACGT\n"))
                return 200
            return 404

    data = fetch_nuclear_genome(
        taxon="Arabidopsis thaliana",
        dest=tmp_path,
        source="tair",
        include=("genome",),
        assembly_level="scaffold",
        reference_only=False,
        transport=TairHit(),
    )
    scope = data.payload["manifest"]["scope"]
    assert scope["assembly_level"] == {"value": "scaffold", "applies_to": "ncbi_only"}
    assert scope["reference_only"] == {"value": False, "applies_to": "ncbi_only"}


def test_auto_tries_tair_then_falls_through_to_a_miss_for_a_non_arabidopsis_taxon(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Hermetic: force GIR's leg to fail on the missing-credential check rather
    # than risk a real FTP attempt if this var happens to be set in the
    # ambient environment.
    #
    # NOTE: every source misses/errors for this fabricated taxon (see
    # test_auto_never_leaves_a_candidates_directory_behind_on_total_miss,
    # which exercises the identical taxon/transport/env combination) so this
    # is a total-miss case and `fetch_nuclear_genome` raises
    # `network.auto_exhausted`, exactly like that test and like
    # test_a_sources_error_does_not_end_the_auto_chain below. The brief's
    # original text called this without `pytest.raises` and asserted on a
    # `data` return value that is never produced on this path — fixed here to
    # match the sibling tests' (and the design's) actual total-miss contract
    # rather than weakening it.
    monkeypatch.delenv("ORGANELLEVERSE_GIR_FTP_PASSWORD", raising=False)
    with pytest.raises(OrganelleExecutionError) as raised:
        fetch_nuclear_genome(
            taxon="Zzyzxus nonexistiens",
            dest=tmp_path,
            source="auto",
            transport=_NcbiMissTransport(),
        )
    assert raised.value.code == "network.auto_exhausted"
    sources = [entry["source"] for entry in raised.value.details["sources_tried"]]
    assert "ncbi" in sources
    assert "imp" in sources
    assert "gir" in sources
    assert "pgd" in sources
    assert "tair" not in sources  # never attempted: not Arabidopsis


def test_auto_prefers_ncbi_when_it_hits_and_stops_the_chain(tmp_path: Path) -> None:
    class NcbiHit(_NcbiMissTransport):
        def get(self, url: str, *, timeout: float = 120.0, headers=None) -> bytes:
            return _datasets_report("GCF_000002.1", "Malus domestica")

    data = fetch_nuclear_genome(
        taxon="Malus domestica", dest=tmp_path, source="auto", download=False, transport=NcbiHit()
    )
    manifest = data.payload["manifest"]
    assert manifest["source"] == "nuclear_assembly"
    sources = [entry["source"] for entry in manifest["sources_tried"]]
    assert sources == ["ncbi"]  # short-circuited: GIR/IMP/PGD never attempted


def test_auto_never_leaves_a_candidates_directory_behind_on_success(tmp_path: Path) -> None:
    class NcbiHit(_NcbiMissTransport):
        def get(self, url: str, *, timeout: float = 120.0, headers=None) -> bytes:
            return _datasets_report("GCF_000003.1", "Malus domestica")

    fetch_nuclear_genome(
        taxon="Malus domestica", dest=tmp_path, source="auto", download=False, transport=NcbiHit()
    )
    assert not (tmp_path / ".auto_candidates").exists()


def test_auto_never_leaves_a_candidates_directory_behind_on_total_miss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ORGANELLEVERSE_GIR_FTP_PASSWORD", raising=False)
    with pytest.raises(OrganelleExecutionError) as raised:
        fetch_nuclear_genome(
            taxon="Zzyzxus nonexistiens",
            dest=tmp_path,
            source="auto",
            transport=_NcbiMissTransport(),
        )
    assert raised.value.code == "network.auto_exhausted"
    assert not (tmp_path / ".auto_candidates").exists()


def test_a_sources_error_does_not_end_the_auto_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ORGANELLEVERSE_GIR_FTP_PASSWORD", raising=False)

    class NcbiErrorsThenNothingElseAnswers(_NcbiMissTransport):
        def get(self, url: str, *, timeout: float = 120.0, headers=None) -> bytes:
            return b"not json at all"  # -> network.malformed_response from NCBI's own parser

    with pytest.raises(OrganelleExecutionError) as raised:
        fetch_nuclear_genome(
            taxon="Zzyzxus nonexistiens",
            dest=tmp_path,
            source="auto",
            transport=NcbiErrorsThenNothingElseAnswers(),
        )
    assert raised.value.code == "network.auto_exhausted"
    detail = raised.value.details["sources_tried"]
    ncbi_entry = next(entry for entry in detail if entry["source"] == "ncbi")
    assert ncbi_entry["status"] == "error"
    gir_entry = next(entry for entry in detail if entry["source"] == "gir")
    assert gir_entry["status"] == "error"


# --- Finding 2: a losing `auto` candidate's real, on-disk partial download --
# must be deleted, not just the temp directories that never got written to.


def _synthetic_fasta(seed: str, n_lines: int = 60, line_len: int = 60) -> bytes:
    """Deterministic, non-repeating FASTA body — long/varied enough that its
    gzip-compressed, zipped form still clears PGD's real ``_MIN_ZIP_BYTES``
    floor (repetitive filler compresses away to nothing and would silently
    make this fixture unrealistic vs. genuine PGD responses)."""
    bases = "ACGT"
    lines: list[str] = []
    state = seed.encode()
    for _ in range(n_lines):
        state = hashlib.sha256(state).digest()
        chars: list[str] = []
        chunk = state
        while len(chars) < line_len:
            chars.extend(bases[b % 4] for b in chunk)
            chunk = hashlib.sha256(chunk).digest()
        lines.append("".join(chars[:line_len]))
    return f">{seed}\n" + "\n".join(lines) + "\n"


def _pgd_zip(inner_filename: str, gz_content: bytes) -> bytes:
    digest = hashlib.md5(gz_content).hexdigest()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(inner_filename, gz_content)
        archive.writestr("PGdownload.md5sum", f"{digest}  {inner_filename}\n")
    return buffer.getvalue()


_IMP_LOSER_BYTES = (
    b">imp_loser_real_partial_download\n" + b"MEEQVGFGFIMPSHOULDNOTSURVIVE" * 3 + b"\n"
)
_PGD_WINNER_GZ = gzip.compress(_synthetic_fasta("pgd_winner").encode())
_PGD_WINNER_ZIP = _pgd_zip("acacia_confusa.pep.fa.gz", _PGD_WINNER_GZ)


class _ImpWritesRealBytesThenPgdWinsTransport(_NcbiMissTransport):
    """NCBI misses immediately (empty report). IMP resolves ``Acacia
    confusa`` via the committed cleaned whitelist (``imp_id="Aco1"``,
    confidence ``"heuristic_consistent"``) and actually downloads a real
    protein FASTA to its ``.auto_candidates/imp`` directory — but per the
    (never-``"verified"``) branch documented in ``_fetch_auto``, IMP can
    never win outright, so this genuine file becomes the chain's provisional
    fallback. GIR then errors out (no FTP credential configured — never even
    creates its candidate directory). PGD is the only source that actually
    answers with a full hit for every requested kind, so it wins and its
    file is promoted to ``dest``; IMP's real, already-downloaded bytes must
    be deleted along with everything else under ``.auto_candidates``."""

    def download_to_path(self, url: str, dest: Path, **kwargs) -> int:
        dest.parent.mkdir(parents=True, exist_ok=True)
        method = kwargs.get("method", "GET")
        data = kwargs.get("data")
        if method == "POST" and data is not None:
            body = json.loads(data)
            if body.get("files") == "acacia_confusa.pep.fa.gz":
                dest.write_bytes(_PGD_WINNER_ZIP)
                return 200
            dest.write_bytes(b"")
            return 404
        if url.endswith("Aco1.prot.fasta"):
            dest.write_bytes(_IMP_LOSER_BYTES)
            return 200
        return 404


def test_a_losing_candidates_real_partial_download_is_deleted_not_just_its_empty_temp_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ORGANELLEVERSE_GIR_FTP_PASSWORD", raising=False)

    data = fetch_nuclear_genome(
        taxon="Acacia confusa",
        dest=tmp_path,
        source="auto",
        include=("protein",),
        download=False,
        transport=_ImpWritesRealBytesThenPgdWinsTransport(),
    )

    manifest = data.payload["manifest"]
    sources = [entry["source"] for entry in manifest["sources_tried"]]
    assert sources == ["ncbi", "imp", "gir", "pgd"]
    imp_entry = next(entry for entry in manifest["sources_tried"] if entry["source"] == "imp")
    assert imp_entry["status"] == "unconfirmed"  # never "hit": see the never-"verified" branch

    # The winner's file really is at the real `dest` path, with PGD's content.
    winner_path = tmp_path / "protein.gz"
    assert winner_path.is_file()
    assert gzip.decompress(winner_path.read_bytes()) == _synthetic_fasta("pgd_winner").encode()

    # No trace of the loser's temp tree...
    assert not (tmp_path / ".auto_candidates").exists()
    # ...and, more specifically, IMP's real downloaded bytes never survive
    # anywhere under dest, under any name — not just "at its own old path".
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert b"IMPSHOULDNOTSURVIVE" not in path.read_bytes()
