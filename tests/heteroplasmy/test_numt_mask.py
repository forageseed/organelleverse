"""Nuclear-copy (NUMT/NUPT) masking in heteroplasmy.quantify_variant_fractions."""

from __future__ import annotations

import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from organelleverse import heteroplasmy
from organelleverse.heteroplasmy import quantify_variant_fractions

mappy = pytest.importorskip("mappy")

CONTIG = "MT"
LENGTH = 3000


@pytest.fixture(autouse=True)
def _fresh_cache():
    heteroplasmy._AUTO_CACHE.clear()
    yield
    heteroplasmy._AUTO_CACHE.clear()


def _random(rng: random.Random, n: int) -> str:
    return "".join(rng.choice("ACGT") for _ in range(n))


def _diverge(sequence: str, every: int) -> str:
    bases = list(sequence)
    for i in range(0, len(bases), every):
        bases[i] = "ACGT"[("ACGT".index(bases[i]) + 1) % 4]
    return "".join(bases)


def _bam(path: Path, groups: list[tuple[int, list[str]]]) -> Path:
    """Reads of 20 bp whose 10th base is the tested base: site position = start0 + 10."""
    pysam = pytest.importorskip("pysam")
    header = {"HD": {"VN": "1.6", "SO": "coordinate"}, "SQ": [{"SN": CONTIG, "LN": LENGTH}]}
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


class _World:
    """A random organelle genome and a small nuclear genome holding copies of parts of it."""

    def __init__(self, tmp_path: Path, *, extra_records: str = "") -> None:
        rng = random.Random(5)
        self.mito = _random(rng, LENGTH)
        diverged = _diverge(self.mito[1000:1600], 20)  # ~5% divergence, 600 bp
        exact = self.mito[2000:2300]  # 300 bp, identical
        chr1 = _random(rng, 5000) + diverged + _random(rng, 8000)
        chr2 = _random(rng, 4000) + exact + _random(rng, 9000)
        self.reference = tmp_path / "mito.fa"
        self.reference.write_text(f">{CONTIG}\n{self.mito}\n", encoding="utf-8")
        self.nuclear = tmp_path / "nuclear.fa"
        self.nuclear.write_text(
            f">chr1\n{chr1}\n>chr2\n{chr2}\n{extra_records}", encoding="utf-8"
        )
        self.bam = _bam(
            tmp_path / "reads.bam",
            [
                (1290, ["A"] * 6 + ["G"] * 4),  # position 1300: inside the diverged copy
                (2140, ["A"] * 6 + ["G"] * 4),  # position 2150: inside the exact copy
                (490, ["A"] * 8 + ["G"] * 2),  # position 500: no nuclear copy
                (2690, ["A"] * 9 + ["G"] * 1),  # position 2700: no nuclear copy
            ],
        )
        self.sites = tmp_path / "sites.tsv"
        self.sites.write_text(
            "site_id\tseqid\tposition\tref\talt\n"
            f"diverged\t{CONTIG}\t1300\tA\tG\n"
            f"exact\t{CONTIG}\t2150\tA\tG\n"
            f"plain1\t{CONTIG}\t500\tA\tG\n"
            f"plain2\t{CONTIG}\t2700\tA\tG\n",
            encoding="utf-8",
        )

    def run(self, **kwargs):
        kwargs.setdefault("reference_fasta", self.reference)
        kwargs.setdefault("auto_homology_mask", False)
        return quantify_variant_fractions(self.bam, self.sites, min_depth=5, **kwargs)


def _rows(result) -> dict:
    return {row["site_id"]: row for row in result.metrics["sites"]}


def test_sites_in_nuclear_copies_are_masked(tmp_path: Path) -> None:
    world = _World(tmp_path)
    result = world.run(nuclear_reference_fasta=world.nuclear)
    assert result.status == "ok", result.summary_text
    rows = _rows(result)
    for site in ("diverged", "exact"):
        assert rows[site]["status"] == "masked" and rows[site]["mask_source"] == "nuclear"
    assert rows["plain1"]["status"] == "ok" and rows["plain1"]["alt_fraction"] == 0.2
    assert rows["plain2"]["status"] == "ok" and rows["plain2"]["alt_fraction"] == 0.1
    mask = result.metrics["homology_mask"]
    assert mask["nuclear"] is True and mask["nuclear_intervals"] >= 2
    assert 0.25 <= result.metrics["masked_reference_fraction"] <= 0.35  # (600 + 300) / 3000
    assert "numt_regions_masked" in result.flags
    assert "homologous_regions_masked" in result.flags
    assert "numt_mapping_not_resolved" not in result.flags
    assert result.provenance.software_versions["mappy"] == mappy.__version__


def test_without_the_option_nothing_is_masked_and_the_fraction_is_zero(tmp_path: Path) -> None:
    result = _World(tmp_path).run()
    assert result.metrics["masked_sites"] == 0
    assert result.metrics["masked_reference_fraction"] == 0.0
    assert result.metrics["homology_mask"]["nuclear"] is False
    assert "numt_regions_masked" not in result.flags


def test_the_minimum_length_decides_which_copies_count(tmp_path: Path) -> None:
    world = _World(tmp_path)
    rows = _rows(world.run(nuclear_reference_fasta=world.nuclear, homology_min_length=450))
    assert rows["diverged"]["status"] == "masked"  # ~600 bp copy
    assert rows["exact"]["status"] == "ok"  # 300 bp copy is below the threshold


def test_a_nuclear_record_named_like_the_reference_is_skipped(tmp_path: Path) -> None:
    probe_dir = tmp_path / "probe"
    probe_dir.mkdir()
    own = f">{CONTIG}\n{_World(probe_dir).mito}\n"  # the organelle itself, left in the nuclear FASTA
    world = _World(tmp_path, extra_records=own)
    result = world.run(nuclear_reference_fasta=world.nuclear)
    rows = _rows(result)
    assert rows["plain1"]["status"] == "ok" and rows["plain2"]["status"] == "ok"
    assert result.metrics["masked_reference_fraction"] < 0.5
    assert "homology_mask_covers_most_of_reference" not in result.flags


def test_organelle_sequence_left_in_the_nuclear_fasta_is_flagged(tmp_path: Path) -> None:
    probe_dir = tmp_path / "probe"
    probe_dir.mkdir()
    own_copy = f">Mt_copy\n{_World(probe_dir).mito}\n"  # same sequence, a different name
    world = _World(tmp_path, extra_records=own_copy)
    result = world.run(nuclear_reference_fasta=world.nuclear)
    assert result.metrics["masked_reference_fraction"] >= 0.8
    assert "homology_mask_covers_most_of_reference" in result.flags
    assert "contain organelle sequence" in result.summary_text
    assert all(row["status"] == "masked" for row in _rows(result).values())


def test_a_missing_nuclear_file_fails_clearly(tmp_path: Path) -> None:
    world = _World(tmp_path)
    result = world.run(nuclear_reference_fasta=tmp_path / "absent.fa")
    assert result.status == "failed"
    assert result.errors[0].code == "heteroplasmy.homology_mask_failed"


def test_the_nuclear_mask_needs_the_bam_reference(tmp_path: Path) -> None:
    world = _World(tmp_path)
    result = quantify_variant_fractions(
        world.bam, world.sites, nuclear_reference_fasta=world.nuclear
    )
    assert result.status == "failed"
    assert result.errors[0].code == "heteroplasmy.mask_needs_reference"


def test_a_missing_mappy_is_an_error_with_an_install_hint(tmp_path: Path, monkeypatch) -> None:
    world = _World(tmp_path)
    monkeypatch.setitem(sys.modules, "mappy", None)  # makes `import mappy` raise ImportError
    result = world.run(nuclear_reference_fasta=world.nuclear)
    assert result.status == "failed"
    assert result.errors[0].code == "heteroplasmy.mappy_missing"
    assert "organelleverse[align]" in result.summary_text


def test_the_search_is_cached_and_refreshed_when_the_nuclear_file_changes(
    tmp_path: Path, monkeypatch
) -> None:
    world = _World(tmp_path)
    built: list[int] = []
    real = mappy.Aligner

    def counting(*args, **kwargs):
        built.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(mappy, "Aligner", counting)
    for _ in range(3):
        assert _rows(world.run(nuclear_reference_fasta=world.nuclear))["exact"]["status"] == "masked"
    assert len(built) == 1  # one index and scan for three BAMs sharing both references
    with world.nuclear.open("a", encoding="utf-8") as handle:
        handle.write(">extra\nACGTACGTAC\n")  # a different file now
    world.run(nuclear_reference_fasta=world.nuclear)
    assert len(built) == 2


def test_plastid_runs_can_mask_nuclear_plastid_copies_too(tmp_path: Path) -> None:
    world = _World(tmp_path)
    result = world.run(scope="plastid", nuclear_reference_fasta=world.nuclear)
    assert _rows(result)["exact"]["mask_source"] == "nuclear"
    assert result.scope == "plastid"


def test_nuclear_plastid_and_bed_masks_combine_with_their_own_sources(
    tmp_path: Path, monkeypatch
) -> None:
    world = _World(tmp_path)
    monkeypatch.setattr(
        "organelleverse.transfer.transfer.detect_transfers_blast",
        lambda *a, **k: SimpleNamespace(
            status="ok",
            summary_text="",
            metrics={"candidates": [{"nuclear_seqid": CONTIG, "nuclear_start": 495, "nuclear_end": 510}]},
            provenance=SimpleNamespace(software_versions={"losat": "0.1.0"}),
        ),
    )
    plastid = tmp_path / "own_plastome.fa"
    plastid.write_text(">cp\nACGT\n", encoding="utf-8")
    bed = tmp_path / "extra.bed"
    bed.write_text(f"{CONTIG}\t2695\t2705\n", encoding="utf-8")
    result = world.run(
        nuclear_reference_fasta=world.nuclear,
        homology_reference_fasta=plastid,
        mask_regions_bed=bed,
    )
    sources = {site: row["mask_source"] for site, row in _rows(result).items()}
    assert sources == {
        "diverged": "nuclear",
        "exact": "nuclear",
        "plain1": "homology",
        "plain2": "bed",
    }
    assert result.metrics["masked_sites"] == 4
