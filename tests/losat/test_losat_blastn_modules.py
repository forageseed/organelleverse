"""LOSAT-first blastn in hgt/transfer (NCBI fallback), pinned by parity."""

from __future__ import annotations

import shutil

import pytest

from organelleverse._losat import resolve_losat

_LOSAT = resolve_losat()
_NCBI = shutil.which("blastn") is not None and shutil.which("makeblastdb") is not None

_DONOR = ">donor\n" + ("ACGTAGCTTAAGGCATCGGACTTAACTTAGGCA" * 8) + "\n"
_RECIPIENT = (
    ">recipient\n"
    + ("TTTACGTAGCTTAAGGCATCGGACTTAACTTAGGCA" * 8)  # 3-base prefix shift
    + "AAAAAAAAAAAAAAAA"
    + ("ACGTAGCTTAAGGCATCGGACTTAACTTAGGCA" * 8)
    + "\n"
)


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return path


@pytest.mark.skipif(_LOSAT is None, reason="vendored LOSAT binary not built")
@pytest.mark.skipif(not _NCBI, reason="parity needs NCBI blastn and makeblastdb on PATH")
def test_hgt_confirmation_losat_and_ncbi_agree(tmp_path, monkeypatch):
    from organelleverse.hgt.align import run_blastn_confirmation

    donor = _write(tmp_path, "donor.fa", _DONOR)
    recipient = _write(tmp_path, "recipient.fa", _RECIPIENT)

    losat_hits = run_blastn_confirmation(donor, recipient, min_len=60, min_identity=90.0)
    monkeypatch.setenv("ORG_VERSE_LOSAT_BIN", "ncbi")
    ncbi_hits = run_blastn_confirmation(donor, recipient, min_len=60, min_identity=90.0)

    def key(hit):
        return (
            hit.recipient_id,
            hit.donor_id,
            hit.recipient_start,
            hit.recipient_end,
            hit.donor_start,
            hit.donor_end,
        )

    assert {key(h) for h in losat_hits} == {key(h) for h in ncbi_hits}
    assert losat_hits, "the synthetic pair must produce at least one hit"


@pytest.mark.skipif(_LOSAT is None, reason="vendored LOSAT binary not built")
@pytest.mark.skipif(not _NCBI, reason="parity needs NCBI blastn and makeblastdb on PATH")
def test_transfer_blast_losat_and_ncbi_agree(tmp_path, monkeypatch):
    from organelleverse.transfer.transfer import detect_transfers_blast

    nuclear = _write(tmp_path, "nuclear.fa", _RECIPIENT.replace(">recipient", ">chrN"))
    organelle = _write(tmp_path, "organelle.fa", _DONOR.replace(">donor", ">pt"))

    losat_result = detect_transfers_blast(
        nuclear, organelle, organelle="plastid", min_identity=85.0, min_length=60
    )
    monkeypatch.setenv("ORG_VERSE_LOSAT_BIN", "ncbi")
    ncbi_result = detect_transfers_blast(
        nuclear, organelle, organelle="plastid", min_identity=85.0, min_length=60
    )

    assert losat_result.status == "ok" and ncbi_result.status == "ok"

    def keys(result):
        metrics = dict(result.metrics)
        return sorted(
            (c["nuclear_start"], c["nuclear_end"], round(c["identity"], 4))
            for c in metrics["candidates"]
        )

    assert keys(losat_result) == keys(ncbi_result)
    assert keys(losat_result), "the synthetic pair must produce candidates"


def test_explicit_ncbi_optout_and_bad_path_semantics(monkeypatch):
    """ORG_VERSE_LOSAT_BIN=ncbi fully disables; a bad path falls back to the
    vendored build (the plastome resolver convention)."""
    monkeypatch.setenv("ORG_VERSE_LOSAT_BIN", "ncbi")
    assert resolve_losat() is None
    monkeypatch.setenv("ORG_VERSE_LOSAT_BIN", "/nonexistent/losat")
    if _LOSAT is not None:
        assert resolve_losat() == _LOSAT
