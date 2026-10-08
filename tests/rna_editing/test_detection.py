"""Scientific contracts for de novo discovery (no supplied editing-site list)."""

import builtins
from pathlib import Path

import pytest
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqFeature import CompoundLocation, SeqFeature, SimpleLocation
from Bio.SeqRecord import SeqRecord

from organelleverse.core.errors import (
    OrganelleDependencyError,
    OrganelleInputError,
    OrganelleParameterError,
)
from organelleverse.rna_editing import detect_editing_sites

pysam = pytest.importorskip("pysam")


def write_bam(path, sequence, observations):
    header = {"HD": {"VN": "1.6", "SO": "coordinate"}, "SQ": [{"SN": "cp", "LN": len(sequence)}]}
    with pysam.AlignmentFile(str(path), "wb", header=header) as bam:
        for i, values in enumerate(observations):
            read = pysam.AlignedSegment()
            read.query_name = f"read{i}"
            read.query_sequence = values.get("seq", sequence)
            read.flag = values.get("flag", 0)
            read.reference_id = 0
            read.reference_start = 0
            read.mapping_quality = values.get("mapq", 60)
            read.cigarstring = values.get("cigar", f"{len(read.query_sequence)}M")
            read.query_qualities = values.get("qual", [40] * len(read.query_sequence))
            bam.write(read)
    pysam.index(str(path))
    return path


def setup_reads(
    tmp_path,
    observations=None,
    sequence="AAC" + "A" * 17 + "C" + "A" * 29 + "G" + "A" * 29 + "T" + "A" * 19,
):
    fasta = tmp_path / "ref.fa"
    fasta.write_text(f">cp\n{sequence}\n")
    if observations is None:
        edited = mutate(sequence, {20: "T", 50: "A", 80: "C"})
        observations = [{"seq": edited}] * 4 + [{}] * 6
    bam = write_bam(tmp_path / "rna.bam", sequence, observations)
    return bam, fasta, sequence


def mutate(sequence, changes):
    bases = list(sequence)
    for pos, base in changes.items():
        bases[pos] = base
    return "".join(bases)


def test_de_novo_both_directions_and_reverse_editing_background(tmp_path):
    bam, fasta, _ = setup_reads(tmp_path)
    result = detect_editing_sites(bam, fasta, scope="plastid")
    assert result.status == "ok"
    rows = result.metrics["sites"]
    assert [(r["position"], r["strand"], r["ref"], r["edited"]) for r in rows] == [
        (21, "+", "C", "T"),
        (51, "-", "G", "A"),
    ]
    assert all(r["depth"] == 10 and r["editing_fraction"] == 0.4 for r in rows)
    assert all(r["strand_evidence"] == "substitution_assumed" for r in rows)
    tc = next(r for r in result.metrics["mismatch_statistics"] if r["genomic_change"] == "T>C")
    assert tc["mismatch_reads"] == 4 and tc["passing_sites"] == 1
    assert not tc["is_c_to_u_compatible"]


@pytest.mark.parametrize(
    "library,flags", [("fr-firststrand", (81, 129)), ("fr-secondstrand", (65, 145))]
)
def test_paired_mates_are_assigned_to_transcript_strand(tmp_path, library, flags):
    sequence = "A" * 20 + "C" + "A" * 29 + "G" + "A" * 49
    ct = mutate(sequence, {20: "T"})
    ga = mutate(sequence, {50: "A"})
    observations = [{"seq": ct, "flag": flag} for flag in flags for _ in range(5)]
    observations += [{"seq": ga, "flag": flag ^ 16} for flag in flags for _ in range(5)]
    bam, fasta, _ = setup_reads(tmp_path, observations, sequence)
    rows = detect_editing_sites(bam, fasta, library_type=library).metrics["sites"]
    assert [(r["position"], r["strand"], r["depth"], r["editing_fraction"]) for r in rows] == [
        (21, "+", 10, 1.0),
        (51, "-", 10, 1.0),
    ]
    # Using the opposite protocol cannot call these substitutions as C-to-U.
    opposite = "fr-secondstrand" if library == "fr-firststrand" else "fr-firststrand"
    assert detect_editing_sites(bam, fasta, library_type=opposite).metrics["site_count"] == 0


@pytest.mark.parametrize("library,flag", [("fr-firststrand", 16), ("fr-secondstrand", 0)])
def test_single_end_stranded(tmp_path, library, flag):
    seq = "A" * 20 + "C" + "A" * 79
    bam, fasta, _ = setup_reads(tmp_path, [{"seq": mutate(seq, {20: "T"}), "flag": flag}] * 10, seq)
    assert detect_editing_sites(bam, fasta, library_type=library).metrics["site_count"] == 1


def test_quality_flags_end_trimming_and_thresholds(tmp_path):
    sequence = "AAC" + "A" * 17 + "C" + "A" * 79
    edited = mutate(sequence, {2: "T", 20: "T"})
    observations = [{"seq": edited}] * 3 + [{}] * 7
    observations += [{"seq": edited, "flag": f} for f in (256, 2048, 512, 1024)]
    observations += [
        {"seq": edited, "mapq": 24},
        {"seq": edited, "qual": [24] * 100},
        {"seq": edited, "qual": None},
    ]
    bam, fasta, _ = setup_reads(tmp_path, observations, sequence)
    result = detect_editing_sites(bam, fasta)
    assert result.metrics["site_count"] == 1
    site = result.metrics["sites"][0]
    assert (site["position"], site["depth"], site["edited_count"]) == (21, 10, 3)
    assert detect_editing_sites(bam, fasta, trim_read_ends=0).metrics["site_count"] == 2
    assert detect_editing_sites(bam, fasta, min_editing_fraction=0.31).metrics["site_count"] == 0
    assert detect_editing_sites(bam, fasta, min_edited_reads=4).metrics["site_count"] == 0
    assert detect_editing_sites(bam, fasta, min_depth=11).metrics["site_count"] == 0
    assert (
        detect_editing_sites(bam, fasta, exclude_duplicates=False).metrics["sites"][0]["depth"]
        == 11
    )


def test_cigar_skips_deletions_insertions_and_soft_clips(tmp_path):
    sequence = "A" * 20 + "C" + "A" * 9 + "C" + "A" * 9 + "C" + "A" * 9
    # 5S,20M,1D,9M,1N,9M,2I,10M; only reference 41 is observed T.
    query = "NNNNN" + "A" * 38 + "GG" + "T" + "A" * 9
    observations = [{"seq": query, "cigar": "5S20M1D9M1N9M2I10M"}] * 10
    bam, fasta, _ = setup_reads(tmp_path, observations, sequence)
    rows = detect_editing_sites(bam, fasta).metrics["sites"]
    assert [(r["position"], r["depth"]) for r in rows] == [(41, 10)]


@pytest.mark.parametrize(
    "variant_count,dna_depth,expected,reason",
    [(0, 10, 2, None), (1, 10, 0, "dna_nonreference"), (0, 9, 0, "insufficient_dna_depth")],
)
def test_dna_requires_coverage_and_excludes_any_nonreference(
    tmp_path, variant_count, dna_depth, expected, reason
):
    bam, fasta, sequence = setup_reads(tmp_path)
    # C>A/G>T DNA SNPs also exclude RNA C>T/G>A candidates.
    variant = mutate(sequence, {20: "A", 50: "T"})
    dna = write_bam(
        tmp_path / "dna.bam",
        sequence,
        [{"seq": variant}] * variant_count + [{}] * (dna_depth - variant_count),
    )
    result = detect_editing_sites(bam, fasta, dna_bam_path=dna)
    assert result.metrics["site_count"] == expected
    if reason:
        assert result.metrics["dna_exclusions"][reason] == 2


def test_joined_reverse_cds_and_phase_annotations(tmp_path):
    sequence = "A" * 20 + "CCA" + "A" * 27 + "GG" + "A" * 48
    bam, fasta, _ = setup_reads(tmp_path, sequence=sequence)
    rec = SeqRecord(Seq(sequence), id="cp", annotations={"molecule_type": "DNA"})
    rec.features = [
        SeqFeature(
            SimpleLocation(19, 26, strand=1),
            type="CDS",
            qualifiers={"gene": ["plus"], "codon_start": ["2"], "transl_table": ["11"]},
        ),
        SeqFeature(
            CompoundLocation(
                [SimpleLocation(50, 52, strand=-1), SimpleLocation(45, 49, strand=-1)]
            ),
            type="CDS",
            qualifiers={"gene": ["minus"]},
        ),
    ]
    gb = tmp_path / "ref.gb"
    SeqIO.write(rec, gb, "genbank")
    rows = detect_editing_sites(bam, fasta, annotation_genbank=gb).metrics["sites"]
    assert rows[0]["effects"][0]["ref_codon"] == "CCA"
    assert rows[0]["effects"][0]["amino_acid_change"] == "P>S"
    effect = rows[1]["effects"][0]
    assert (
        effect["ref_codon"],
        effect["edited_codon"],
        effect["codon_position"],
        effect["amino_acid_change"],
    ) == ("CCT", "CTT", 2, "P>L")


@pytest.mark.parametrize(
    "parameters",
    [
        {"min_depth": 0},
        {"min_editing_fraction": float("nan")},
        {"trim_read_ends": -1},
        {"min_depth": 1.2},
        {"scope": "wrong"},
        {"library_type": "wrong"},
    ],
)
def test_invalid_parameters_fail_clearly(tmp_path, parameters):
    with pytest.raises(OrganelleParameterError):
        detect_editing_sites(tmp_path / "missing.bam", tmp_path / "ref.fa", **parameters)


def test_missing_inputs_and_reference_mismatch(tmp_path):
    bam, fasta, _ = setup_reads(tmp_path)
    with pytest.raises(OrganelleInputError):
        detect_editing_sites(tmp_path / "missing.bam", fasta)
    fasta.write_text(">different\nAAA\n")
    with pytest.raises(OrganelleInputError, match="does not match"):
        detect_editing_sites(bam, fasta)


def test_missing_optional_dependency_has_installation_hint(monkeypatch):
    original = builtins.__import__

    def without_pysam(name, *args, **kwargs):
        if name == "pysam":
            raise ImportError("test missing dependency")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_pysam)
    with pytest.raises(OrganelleDependencyError, match=r"organelleverse\[qc\]"):
        detect_editing_sites("missing.bam", "missing.fa")


def test_published_reference_table_alleles_and_documented_coordinate_normalization():
    import csv

    data = Path(__file__).resolve().parents[1] / "data" / "rna_editing"
    reference = SeqIO.read(data / "NC_000932.1.fasta", "fasta")
    with (data / "arabidopsis_cp_known_34.tsv").open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == 34
    assert len({r["position"] for r in rows}) == 34
    for row in rows:
        assert reference.seq[int(row["position"]) - 1] == row["ref"]
        assert (row["ref"], row["edited"], row["strand"]) in (("C", "T", "+"), ("G", "A", "-"))
    corrected = [r for r in rows if r["position"] != r["source_position"]]
    assert [(r["gene"], r["source_position"], r["position"]) for r in corrected] == [
        ("rpl23", "86056", "86055")
    ]
    assert str(reference.seq[86053:86056].reverse_complement()) == "TCA"


def test_other_alleles_remain_in_frequency_denominator(tmp_path):
    sequence = "A" * 20 + "C" + "A" * 79
    observations = [{"seq": mutate(sequence, {20: "T"})}] * 3
    observations += [{"seq": mutate(sequence, {20: "G"})}] * 2 + [{}] * 5
    bam, fasta, _ = setup_reads(tmp_path, observations, sequence)
    site = detect_editing_sites(bam, fasta).metrics["sites"][0]
    assert (site["depth"], site["other_count"], site["editing_fraction"]) == (10, 2, 0.3)


def test_unindexed_bam_and_mismatched_annotation_fail(tmp_path):
    bam, fasta, sequence = setup_reads(tmp_path)
    gb = tmp_path / "wrong.gb"
    record = SeqRecord(Seq("A" * len(sequence)), id="cp", annotations={"molecule_type": "DNA"})
    SeqIO.write(record, gb, "genbank")
    with pytest.raises(OrganelleInputError, match="GenBank ID/sequence"):
        detect_editing_sites(bam, fasta, annotation_genbank=gb)
    Path(str(bam) + ".bai").unlink()
    with pytest.raises(OrganelleInputError, match="indexed"):
        detect_editing_sites(bam, fasta)


def test_depth_is_not_silently_capped_by_pileup_defaults(tmp_path):
    sequence = "A" * 8 + "C" + "A" * 11
    observations = [{"seq": mutate(sequence, {8: "T"})}] * 3000 + [{}] * 6000
    bam, fasta, _ = setup_reads(tmp_path, observations, sequence)
    site = detect_editing_sites(bam, fasta).metrics["sites"][0]
    assert (site["depth"], site["edited_count"], site["editing_fraction"]) == (9000, 3000, 1 / 3)
