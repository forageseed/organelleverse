"""Masking of homologous (MTPT/NUMT-like) regions in heteroplasmy.quantify_variant_fractions."""

from __future__ import annotations

import random
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from organelleverse.heteroplasmy import quantify_variant_fractions

CONTIG = "MT"


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
    # site1 at position 10 (30% G), site2 at position 50 (50% G)
    return _bam(
        tmp_path / "reads.bam",
        100,
        [(0, ["A"] * 7 + ["G"] * 3), (40, ["A"] * 5 + ["G"] * 5)],
    )


def _two_sites(path: Path) -> Path:
    path.write_text(
        "site_id\tseqid\tposition\tref\talt\n"
        f"site1\t{CONTIG}\t10\tA\tG\n"
        f"site2\t{CONTIG}\t50\tA\tG\n",
        encoding="utf-8",
    )
    return path


def _bed(path: Path, *lines: str) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _rows(result) -> dict:
    return {row["site_id"]: row for row in result.metrics["sites"]}


def test_without_a_mask_nothing_changes_and_the_numt_caveat_stays(tmp_path: Path) -> None:
    result = quantify_variant_fractions(
        _two_site_bam(tmp_path), _two_sites(tmp_path / "sites.tsv"), min_depth=5
    )
    rows = _rows(result)
    assert rows["site1"]["alt_fraction"] == 0.3 and rows["site2"]["alt_fraction"] == 0.5
    assert all(row["mask_source"] is None for row in rows.values())
    assert result.metrics["masked_sites"] == 0
    assert "numt_mapping_not_resolved" in result.flags
    assert "homologous_regions_masked" not in result.flags


def test_bed_mask_skips_sites_inside_the_region(tmp_path: Path) -> None:
    bed = _bed(tmp_path / "mask.bed", "# plastid-derived insert", f"{CONTIG}\t0\t30\tMTPT1")
    result = quantify_variant_fractions(
        _two_site_bam(tmp_path),
        _two_sites(tmp_path / "sites.tsv"),
        min_depth=5,
        mask_regions_bed=bed,
    )
    rows = _rows(result)
    masked, kept = rows["site1"], rows["site2"]
    assert masked["status"] == "masked" and masked["mask_source"] == "bed"
    assert masked["alt_fraction"] is None and masked["depth"] == 0
    assert masked["ci95_lower"] is None
    assert kept["status"] == "ok" and kept["alt_fraction"] == 0.5 and kept["mask_source"] is None
    assert result.metrics["masked_sites"] == 1 and result.metrics["evaluated_sites"] == 1
    assert result.metrics["masked_interval_count"] == 1
    assert "homologous_regions_masked" in result.flags
    assert "numt_mapping_not_resolved" not in result.flags
    assert "1 masked" in result.summary_text


@pytest.mark.parametrize(
    ("start", "end", "is_masked"),
    [
        (9, 10, True),  # 0-based half-open [9, 10) is position 10 exactly
        (10, 11, False),  # [10, 11) is position 11
        (0, 9, False),  # [0, 9) ends at position 9
        (5, 100, True),
    ],
)
def test_bed_coordinates_are_zero_based_half_open(
    tmp_path: Path, start: int, end: int, is_masked: bool
) -> None:
    bed = _bed(tmp_path / "mask.bed", f"{CONTIG}\t{start}\t{end}")
    result = quantify_variant_fractions(
        _two_site_bam(tmp_path),
        _two_sites(tmp_path / "sites.tsv"),
        min_depth=5,
        mask_regions_bed=bed,
    )
    assert (_rows(result)["site1"]["status"] == "masked") is is_masked


def test_bed_for_another_contig_masks_nothing(tmp_path: Path) -> None:
    bed = _bed(tmp_path / "mask.bed", "OTHER\t0\t1000")
    result = quantify_variant_fractions(
        _two_site_bam(tmp_path),
        _two_sites(tmp_path / "sites.tsv"),
        min_depth=5,
        mask_regions_bed=bed,
    )
    assert result.metrics["masked_sites"] == 0
    assert "homologous_regions_masked" in result.flags  # a mask was applied, it was just empty


@pytest.mark.parametrize("line", [f"{CONTIG}\t30\t30", f"{CONTIG}\t-1\t10", f"{CONTIG}\ta\tb", CONTIG])
def test_invalid_bed_fails_clearly(tmp_path: Path, line: str) -> None:
    bed = _bed(tmp_path / "mask.bed", line)
    result = quantify_variant_fractions(
        _two_site_bam(tmp_path),
        _two_sites(tmp_path / "sites.tsv"),
        min_depth=5,
        mask_regions_bed=bed,
    )
    assert result.status == "failed"
    assert result.errors[0].code == "heteroplasmy.invalid_mask_bed"


def test_missing_bed_file_fails_clearly(tmp_path: Path) -> None:
    result = quantify_variant_fractions(
        _two_site_bam(tmp_path),
        _two_sites(tmp_path / "sites.tsv"),
        mask_regions_bed=tmp_path / "absent.bed",
    )
    assert result.status == "failed"
    assert result.errors[0].code == "heteroplasmy.invalid_mask_bed"


def _fake_search(candidates, *, status="ok", seen=None):
    def fake(nuclear_fasta, organelle_fasta, **kwargs):
        if seen is not None:
            seen.update(
                target=str(nuclear_fasta), source=str(organelle_fasta), **kwargs
            )
        return SimpleNamespace(
            status=status,
            summary_text="boom" if status == "failed" else "",
            metrics={"candidates": candidates},
            provenance=SimpleNamespace(software_versions={"losat": "0.1.0"}),
        )

    return fake


def _reference(tmp_path: Path, length: int = 100) -> Path:
    path = tmp_path / "ref.fa"
    path.write_text(f">{CONTIG}\n{'A' * length}\n", encoding="utf-8")
    return path


def test_homology_search_regions_are_masked(tmp_path: Path, monkeypatch) -> None:
    seen: dict = {}
    # BLAST can report either orientation; 1-based inclusive 20..5 covers positions 5-20
    candidates = [{"nuclear_seqid": CONTIG, "nuclear_start": 20, "nuclear_end": 5}]
    monkeypatch.setattr(
        "organelleverse.transfer.transfer.detect_transfers_blast",
        _fake_search(candidates, seen=seen),
    )
    plastid = tmp_path / "plastid.fa"
    plastid.write_text(">cp\nACGT\n", encoding="utf-8")
    result = quantify_variant_fractions(
        _two_site_bam(tmp_path),
        _two_sites(tmp_path / "sites.tsv"),
        min_depth=5,
        reference_fasta=_reference(tmp_path),
        homology_reference_fasta=plastid,
        homology_min_identity=85.0,
        homology_min_length=150,
    )
    rows = _rows(result)
    assert rows["site1"]["status"] == "masked" and rows["site1"]["mask_source"] == "homology"
    assert rows["site2"]["status"] == "ok"
    assert seen["target"].endswith("ref.fa") and seen["source"].endswith("plastid.fa")
    assert (seen["min_identity"], seen["min_length"]) == (85.0, 150)
    assert seen["organelle"] == "plastid"  # the other organelle of a mitochondrial scope
    assert result.provenance.software_versions["losat"] == "0.1.0"


def test_a_failed_homology_search_never_falls_back_to_unmasked(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "organelleverse.transfer.transfer.detect_transfers_blast",
        _fake_search([], status="failed"),
    )
    plastid = tmp_path / "plastid.fa"
    plastid.write_text(">cp\nACGT\n", encoding="utf-8")
    result = quantify_variant_fractions(
        _two_site_bam(tmp_path),
        _two_sites(tmp_path / "sites.tsv"),
        reference_fasta=_reference(tmp_path),
        homology_reference_fasta=plastid,
    )
    assert result.status == "failed"
    assert result.errors[0].code == "heteroplasmy.homology_mask_failed"
    assert "boom" in result.summary_text


def test_homology_masking_needs_the_bam_reference(tmp_path: Path) -> None:
    result = quantify_variant_fractions(
        _two_site_bam(tmp_path),
        _two_sites(tmp_path / "sites.tsv"),
        homology_reference_fasta=tmp_path / "plastid.fa",
    )
    assert result.status == "failed"
    assert result.errors[0].code == "heteroplasmy.mask_needs_reference"


@pytest.mark.parametrize("identity", [0, -5, 100.5])
def test_homology_identity_must_be_a_percentage(tmp_path: Path, identity: float) -> None:
    result = quantify_variant_fractions(
        _two_site_bam(tmp_path),
        _two_sites(tmp_path / "sites.tsv"),
        homology_min_identity=identity,
    )
    assert result.status == "failed"
    assert result.errors[0].code == "heteroplasmy.invalid_threshold"


def _search_tool_available() -> bool:
    from organelleverse._losat import resolve_losat

    return resolve_losat() is not None or (
        shutil.which("blastn") is not None and shutil.which("makeblastdb") is not None
    )


@pytest.mark.skipif(not _search_tool_available(), reason="needs LOSAT or NCBI BLAST+")
def test_a_diverged_plastid_insert_is_found_and_masked_with_the_real_search(
    tmp_path: Path,
) -> None:
    rng = random.Random(7)
    mito = "".join(rng.choice("ACGT") for _ in range(3000))
    insert = list(mito[1000:1600])
    for i in range(0, len(insert), 33):  # ~3% divergence, still far above the 80% default
        insert[i] = "ACGT"[("ACGT".index(insert[i]) + 1) % 4]
    flank = "".join(rng.choice("ACGT") for _ in range(700))
    plastid = flank + "".join(insert) + "".join(rng.choice("ACGT") for _ in range(700))
    (tmp_path / "mito.fa").write_text(f">{CONTIG}\n{mito}\n", encoding="utf-8")
    decoy = "".join(rng.choice("ACGT") for _ in range(1500))  # a second, unrelated record
    (tmp_path / "plastid.fa").write_text(f">cp\n{plastid}\n>decoy\n{decoy}\n", encoding="utf-8")

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
        bam,
        sites,
        min_depth=5,
        reference_fasta=tmp_path / "mito.fa",
        homology_reference_fasta=tmp_path / "plastid.fa",
    )
    assert result.status == "ok", result.summary_text
    rows = _rows(result)
    assert rows["in_insert"]["status"] == "masked" and rows["in_insert"]["mask_source"] == "homology"
    assert rows["unique"]["status"] == "ok" and rows["unique"]["alt_fraction"] == 0.2
