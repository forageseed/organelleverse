"""Default automatic homology masking against the package's bundled plastomes."""

from __future__ import annotations

import random
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from organelleverse import heteroplasmy
from organelleverse.heteroplasmy import quantify_variant_fractions

CONTIG = "MT"


@pytest.fixture(autouse=True)
def _fresh_cache():
    heteroplasmy._AUTO_CACHE.clear()
    yield
    heteroplasmy._AUTO_CACHE.clear()


def _bam(path: Path, length: int, groups: list[tuple[int, list[str]]]) -> Path:
    """Reads of 20 bp whose 10th base is the tested base: site position = start0 + 10."""
    pysam = pytest.importorskip("pysam")
    header = {"HD": {"VN": "1.6", "SO": "coordinate"}, "SQ": [{"SN": CONTIG, "LN": length}]}
    unsorted = path.with_suffix(".unsorted.bam")
    with pysam.AlignmentFile(str(unsorted), "wb", header=header) as out:
        n = 0
        for start0, bases in groups:
            for base in bases:
                read = pysam.AlignedSegment()
                read.query_name = f"r{n}"
                read.query_sequence = "A" * 9 + base + "A" * 10
                read.flag = 0
                read.reference_id = 0
                read.reference_start = start0
                read.mapping_quality = 60
                read.cigar = [(0, 20)]
                read.query_qualities = pysam.qualitystring_to_array("I" * 20)
                out.write(read)
                n += 1
    pysam.sort("-o", str(path), str(unsorted))
    pysam.index(str(path))
    return path


def _two_site_bam(tmp_path: Path) -> Path:
    return _bam(tmp_path / "reads.bam", 100, [(0, ["A"] * 7 + ["G"] * 3), (40, ["A"] * 5 + ["G"] * 5)])


def _two_sites(path: Path) -> Path:
    path.write_text(
        "site_id\tseqid\tposition\tref\talt\n"
        f"site1\t{CONTIG}\t10\tA\tG\n"
        f"site2\t{CONTIG}\t50\tA\tG\n",
        encoding="utf-8",
    )
    return path


def _reference(tmp_path: Path, length: int = 100) -> Path:
    path = tmp_path / "ref.fa"
    path.write_text(f">{CONTIG}\n{'A' * length}\n", encoding="utf-8")
    return path


def _library(tmp_path: Path, monkeypatch, sequences: dict[str, str] | None = None) -> Path:
    """A stand-in for the bundled plastome reference directory."""
    from Bio import SeqIO
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord

    library = tmp_path / "plastomes"
    library.mkdir()
    for name, sequence in (sequences or {"Test": "ACGT" * 50}).items():
        record = SeqRecord(
            Seq(sequence), id=f"{name.upper()}.1", name=name.upper()[:16], description="test"
        )
        record.annotations["molecule_type"] = "DNA"
        SeqIO.write(record, library / f"{name}_chloroplast.gb", "genbank")
    monkeypatch.setattr(
        "organelleverse.annotation.plastome.db.default_plastome_reference_dir", lambda: library
    )
    return library


def _fake_search(candidates, *, status="ok", calls=None):
    def fake(nuclear_fasta, organelle_fasta, **kwargs):
        if calls is not None:
            calls.append({"target": str(nuclear_fasta), "source": str(organelle_fasta), **kwargs})
        return SimpleNamespace(
            status=status,
            summary_text="boom" if status == "failed" else "",
            metrics={"candidates": candidates},
            provenance=SimpleNamespace(software_versions={"losat": "0.1.0"}),
        )

    return fake


def _patch_search(monkeypatch, candidates, **kwargs):
    monkeypatch.setattr(
        "organelleverse.transfer.transfer.detect_transfers_blast",
        _fake_search(candidates, **kwargs),
    )


def _rows(result) -> dict:
    return {row["site_id"]: row for row in result.metrics["sites"]}


def _run(tmp_path: Path, **kwargs):
    if "reference_fasta" not in kwargs:  # setdefault would rewrite the file on every call
        kwargs["reference_fasta"] = _reference(tmp_path)
    return quantify_variant_fractions(
        _two_site_bam(tmp_path), _two_sites(tmp_path / "sites.tsv"), min_depth=5, **kwargs
    )


def _search_tool_available() -> bool:
    from organelleverse._losat import resolve_losat

    return resolve_losat() is not None or (
        shutil.which("blastn") is not None and shutil.which("makeblastdb") is not None
    )


@pytest.mark.skipif(not _search_tool_available(), reason="needs LOSAT or NCBI BLAST+")
def test_mitochondrial_runs_with_a_reference_are_masked_by_default(
    tmp_path: Path, monkeypatch
) -> None:
    rng = random.Random(11)
    mito = "".join(rng.choice("ACGT") for _ in range(3000))
    insert = list(mito[1000:1600])
    for i in range(0, len(insert), 33):  # ~3% divergence
        insert[i] = "ACGT"[("ACGT".index(insert[i]) + 1) % 4]
    plastome = (
        "".join(rng.choice("ACGT") for _ in range(700))
        + "".join(insert)
        + "".join(rng.choice("ACGT") for _ in range(700))
    )
    _library(tmp_path, monkeypatch, {"Close": plastome})
    (tmp_path / "mito.fa").write_text(f">{CONTIG}\n{mito}\n", encoding="utf-8")
    bam = _bam(
        tmp_path / "reads.bam",
        3000,
        [(1290, ["A"] * 6 + ["G"] * 4), (2490, ["A"] * 8 + ["G"] * 2)],
    )
    sites = tmp_path / "sites.tsv"
    sites.write_text(
        "site_id\tseqid\tposition\tref\talt\n"
        f"in_insert\t{CONTIG}\t1300\tA\tG\n"
        f"unique\t{CONTIG}\t2500\tA\tG\n",
        encoding="utf-8",
    )
    result = quantify_variant_fractions(
        bam, sites, min_depth=5, reference_fasta=tmp_path / "mito.fa"
    )
    assert result.status == "ok", result.summary_text
    rows = _rows(result)
    assert rows["in_insert"]["status"] == "masked" and rows["in_insert"]["mask_source"] == "homology"
    assert rows["unique"]["status"] == "ok" and rows["unique"]["alt_fraction"] == 0.2
    mask = result.metrics["homology_mask"]
    assert mask["mode"] == "bundled_plastomes" and mask["bundled_references"] == 1
    assert "homologous_regions_masked" in result.flags
    assert "homology_mask_from_bundled_references" in result.flags
    assert "numt_mapping_not_resolved" not in result.flags
    assert "homology_mask_unavailable" not in result.flags


def test_the_automatic_search_uses_the_bundled_plastomes_and_the_bam_reference(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list[dict] = []
    _library(tmp_path, monkeypatch, {"One": "ACGT" * 50, "Two": "TTGA" * 50})
    _patch_search(
        monkeypatch,
        [{"nuclear_seqid": CONTIG, "nuclear_start": 5, "nuclear_end": 20}],
        calls=calls,
    )
    result = _run(tmp_path, homology_min_identity=85.0, homology_min_length=150)
    rows = _rows(result)
    assert rows["site1"]["status"] == "masked" and rows["site1"]["mask_source"] == "homology"
    assert rows["site2"]["status"] == "ok"
    (call,) = calls
    assert call["target"].endswith("ref.fa")  # the BAM reference is searched ...
    assert call["organelle"] == "plastid"  # ... against the other organelle
    assert (call["min_identity"], call["min_length"]) == (85.0, 150)
    assert result.metrics["homology_mask"]["bundled_references"] == 2
    assert result.provenance.software_versions["losat"] == "0.1.0"


def test_the_switch_turns_the_automatic_mask_off(tmp_path: Path, monkeypatch) -> None:
    def boom(*args, **kwargs):
        raise AssertionError("no search expected")

    monkeypatch.setattr("organelleverse.transfer.transfer.detect_transfers_blast", boom)
    result = _run(tmp_path, auto_homology_mask=False)
    assert result.metrics["masked_sites"] == 0
    assert result.metrics["homology_mask"] == {
        "mode": "off",
        "bed": False,
        "bundled_references": None,
        "nuclear": False,
        "nuclear_intervals": 0,
        "note": None,
    }
    assert "numt_mapping_not_resolved" in result.flags
    assert "homology_mask_unavailable" not in result.flags


def test_without_a_reference_fasta_the_gap_is_flagged_not_hidden(tmp_path: Path) -> None:
    result = quantify_variant_fractions(
        _two_site_bam(tmp_path), _two_sites(tmp_path / "sites.tsv"), min_depth=5
    )
    assert result.status == "ok"
    assert result.metrics["masked_sites"] == 0
    assert "reference_fasta" in result.metrics["homology_mask"]["note"]
    assert "homology_mask_unavailable" in result.flags
    assert "numt_mapping_not_resolved" in result.flags
    assert "unavailable" in result.summary_text


def test_plastid_runs_are_not_masked_automatically_and_say_why(
    tmp_path: Path, monkeypatch
) -> None:
    _patch_search(monkeypatch, [{"nuclear_seqid": CONTIG, "nuclear_start": 1, "nuclear_end": 100}])
    result = _run(tmp_path, scope="plastid")
    assert result.metrics["masked_sites"] == 0
    assert "mitochondrial" in result.metrics["homology_mask"]["note"]
    assert "homology_mask_unavailable" not in result.flags  # not applicable, not a failure
    assert "numt_mapping_not_resolved" in result.flags


def test_a_failed_automatic_search_leaves_the_result_unmasked_but_flagged(
    tmp_path: Path, monkeypatch
) -> None:
    _library(tmp_path, monkeypatch)
    _patch_search(monkeypatch, [], status="failed")
    result = _run(tmp_path)
    assert result.status == "ok"
    assert result.metrics["masked_sites"] == 0
    assert "boom" in result.metrics["homology_mask"]["note"]
    assert "homology_mask_unavailable" in result.flags
    assert "numt_mapping_not_resolved" in result.flags
    assert "Automatic homology mask unavailable" in result.summary_text


def test_an_empty_bundled_library_is_flagged(tmp_path: Path, monkeypatch) -> None:
    library = tmp_path / "empty"
    library.mkdir()
    monkeypatch.setattr(
        "organelleverse.annotation.plastome.db.default_plastome_reference_dir", lambda: library
    )
    result = _run(tmp_path)
    assert "No bundled plastome references" in result.metrics["homology_mask"]["note"]
    assert "homology_mask_unavailable" in result.flags


def test_an_explicit_homology_fasta_replaces_the_bundled_library(
    tmp_path: Path, monkeypatch
) -> None:
    def boom():
        raise AssertionError("the bundled library must not be read")

    monkeypatch.setattr("organelleverse.annotation.plastome.db.default_plastome_reference_dir", boom)
    _patch_search(monkeypatch, [{"nuclear_seqid": CONTIG, "nuclear_start": 5, "nuclear_end": 20}])
    own = tmp_path / "own_plastome.fa"
    own.write_text(">cp\nACGT\n", encoding="utf-8")
    result = _run(tmp_path, homology_reference_fasta=own)
    assert result.metrics["homology_mask"]["mode"] == "explicit_fasta"
    assert "homology_mask_from_bundled_references" not in result.flags
    assert _rows(result)["site1"]["status"] == "masked"


def test_a_failed_explicit_search_is_still_an_error_under_the_default(
    tmp_path: Path, monkeypatch
) -> None:
    _patch_search(monkeypatch, [], status="failed")
    own = tmp_path / "own_plastome.fa"
    own.write_text(">cp\nACGT\n", encoding="utf-8")
    result = _run(tmp_path, homology_reference_fasta=own)
    assert result.status == "failed"
    assert result.errors[0].code == "heteroplasmy.homology_mask_failed"


def test_a_bed_and_the_automatic_mask_combine(tmp_path: Path, monkeypatch) -> None:
    _library(tmp_path, monkeypatch)
    _patch_search(monkeypatch, [{"nuclear_seqid": CONTIG, "nuclear_start": 5, "nuclear_end": 20}])
    bed = tmp_path / "numt.bed"
    bed.write_text(f"{CONTIG}\t40\t60\tNUMT\n", encoding="utf-8")
    result = _run(tmp_path, mask_regions_bed=bed)
    rows = _rows(result)
    assert rows["site1"]["mask_source"] == "homology"
    assert rows["site2"]["mask_source"] == "bed"
    assert result.metrics["masked_sites"] == 2
    assert result.metrics["homology_mask"]["bed"] is True
    assert result.metrics["homology_mask"]["mode"] == "bundled_plastomes"


def test_the_automatic_search_is_cached_per_reference_and_refreshed_when_it_changes(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list[dict] = []
    _library(tmp_path, monkeypatch)
    _patch_search(
        monkeypatch,
        [{"nuclear_seqid": CONTIG, "nuclear_start": 5, "nuclear_end": 20}],
        calls=calls,
    )
    reference = _reference(tmp_path)
    for _ in range(3):
        result = _run(tmp_path, reference_fasta=reference)
        assert _rows(result)["site1"]["status"] == "masked"
    assert len(calls) == 1  # one search for three BAMs sharing a reference
    reference.write_text(f">{CONTIG}\n{'A' * 120}\n", encoding="utf-8")  # a different file now
    _run(tmp_path, reference_fasta=reference)
    assert len(calls) == 2


def test_the_cache_hands_out_copies(tmp_path: Path, monkeypatch) -> None:
    _library(tmp_path, monkeypatch)
    _patch_search(monkeypatch, [{"nuclear_seqid": CONTIG, "nuclear_start": 5, "nuclear_end": 20}])
    reference = _reference(tmp_path)
    first, _, _ = heteroplasmy._auto_plastome_intervals(reference, 80.0, 100)
    first[CONTIG].append((0, 99, "mutated"))
    second, _, _ = heteroplasmy._auto_plastome_intervals(reference, 80.0, 100)
    assert (0, 99, "mutated") not in second[CONTIG]
