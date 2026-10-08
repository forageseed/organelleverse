"""Splice/edit coordinate roundtrips and independent biological expectations."""

import json
from pathlib import Path

import pytest
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqFeature import CompoundLocation, SeqFeature, SimpleLocation
from Bio.SeqRecord import SeqRecord

from organelleverse.annotation import translate_edited_cds, write_edited_cds
from organelleverse.core.errors import OrganelleInputError, OrganelleParameterError
from organelleverse.core.frozen import thaw_json
from organelleverse.core.result import OrganelleResult


def evidence(sites=(), *, operation="rna_editing.detect_editing_sites"):
    return OrganelleResult(
        operation_id=operation,
        operation_version="1.0",
        scope="mitochondrion",
        status="ok",
        summary_text="Selected editing evidence",
        metrics={"sites": list(sites)},
        flags=("candidate_sites_not_validated_editing",),
    )


def genome(tmp_path, sequence, parts=None, *, strand=1, qualifiers=None, operator="join"):
    parts = parts or [(0, len(sequence))]
    locations = [SimpleLocation(a, b, strand=strand) for a, b in parts]
    location = locations[0] if len(locations) == 1 else CompoundLocation(locations, operator)
    r = SeqRecord(Seq(sequence), id="seq", name="seq", annotations={"molecule_type": "DNA"})
    r.features = [SeqFeature(location, type="CDS", qualifiers=qualifiers or {"gene": ["test"]})]
    path = tmp_path / "annotation.gb"
    SeqIO.write(r, path, "genbank")
    return path


def site(position, *, strand="+", ref="C", edited="T", **other):
    return {
        "seqid": "seq",
        "position": position,
        "strand": strand,
        "ref": ref,
        "edited": edited,
        **other,
    }


def result_row(path, *sites, **kwargs):
    result = translate_edited_cds(evidence(sites), annotation_genbank=path, **kwargs)
    return thaw_json(result.metrics)["cds"][0]


@pytest.mark.parametrize("direction", [1, -1])
def test_editing_start_gain_on_both_strands(tmp_path, direction):
    coding = "ACGAAATAA"
    raw = coding if direction == 1 else str(Seq(coding).reverse_complement())
    path = genome(tmp_path, raw, strand=direction)
    edit = site(2) if direction == 1 else site(8, strand="-", ref="G", edited="A")
    row = result_row(path, edit)
    assert row["spliced_cds_before"] == coding
    assert row["spliced_cds_after"] == "ATGAAATAA"
    assert row["protein_before"] == "TK" and row["protein_after"] == "MK"
    assert row["start_gain"] is True
    assert row["applied_sites"][0]["cds_position"] == 2


@pytest.mark.parametrize("direction", [1, -1])
def test_junction_split_codon_and_transcript_order(tmp_path, direction):
    raw = "ATGCC" + "GGGGG" + "ATAA"
    parts = [(0, 5), (10, 14)]
    pos = 5
    edit = site(pos)
    if direction == -1:
        raw = str(Seq(raw).reverse_complement())
        parts = [(14 - b, 14 - a) for a, b in parts]
        edit = site(14 - pos + 1, strand="-", ref="G", edited="A")
    row = result_row(genome(tmp_path, raw, parts, strand=direction), edit)
    assert row["spliced_cds_before"] == "ATGCCATAA"
    assert row["spliced_cds_after"] == "ATGCTATAA"
    assert row["protein_before"] == "MP" and row["protein_after"] == "ML"
    assert row["applied_sites"][0]["cds_position"] == 5


def test_join_across_origin_uses_stored_biological_order(tmp_path):
    row = result_row(genome(tmp_path, "CCATAAGGGATG", [(9, 12), (0, 6)]), site(2))
    assert row["spliced_cds_after"] == "ATGCTATAA"
    assert row["protein_after"] == "ML"


def test_stop_gain_retains_and_reports_internal_stop(tmp_path):
    row = result_row(genome(tmp_path, "ATGCAA AAATAA".replace(" ", "")), site(4))
    assert row["protein_after"] == "M*K"
    assert row["stop_gain_positions"] == [2]
    assert row["internal_stop_positions_after"] == [2]
    assert row["has_terminal_stop_after"] is True


def test_stop_gain_terminal_length_and_annotation_match(tmp_path):
    row = result_row(genome(tmp_path, "ATGCAA", qualifiers={"translation": ["M*"]}), site(4))
    assert row["protein_before"] == "MQ" and row["protein_after"] == "M"
    assert row["amino_acid_changes"] == [{"position": 2, "before": "Q", "after": "*"}]
    assert row["annotated_protein_matches_after"] is True


def test_stop_loss_and_start_loss_are_reported(tmp_path):
    row = result_row(
        genome(tmp_path, "ATGTAA"), site(2, ref="T", edited="C"), site(4, ref="T", edited="C")
    )
    assert row["start_loss"] is True
    assert row["stop_loss_positions"] == [2]
    assert row["protein_after"] == "TQ"


def test_alternate_start_normalization_does_not_invent_aa_difference(tmp_path):
    path = genome(tmp_path, "GTGAAATAA", qualifiers={"transl_table": ["11"]})
    row = result_row(path, site(1, ref="G", edited="A"))
    assert row["protein_before"] == row["protein_after"] == "MK"
    assert row["amino_acid_changes"] == []


def test_genetic_code_override_is_explicit(tmp_path):
    path = genome(tmp_path, "ATGTGA", qualifiers={"transl_table": ["4"]})
    assert result_row(path)["protein_after"] == "MW"
    assert result_row(path, genetic_code=1)["protein_after"] == "M"


def test_partial_codon_start_and_remainder_are_retained(tmp_path):
    row = result_row(
        genome(tmp_path, "CGTGAAATAAC", qualifiers={"codon_start": ["2"], "transl_table": ["11"]})
    )
    assert row["protein_after"] == "VK"
    assert row["trailing_bases"] == 1 and row["partial_5prime"] is True


def test_unmapped_intron_and_wrong_strand_not_applied(tmp_path):
    path = genome(tmp_path, "ATGCCGGGGCATAA", [(0, 5), (10, 14)])
    edits = [site(6, ref="G", edited="A"), site(4, strand="-")]
    r = translate_edited_cds(evidence(edits), annotation_genbank=path)
    assert thaw_json(r.metrics)["unmapped_sites"] == edits
    assert r.metrics["edited_cds_count"] == 0


def test_zero_evidence_excluded_while_low_frequency_retained(tmp_path):
    path = genome(tmp_path, "ACGCCATAA")
    r = translate_edited_cds(
        evidence([site(2, editing_fraction=0), site(5, editing_fraction=0.02, depth=100)]),
        annotation_genbank=path,
    )
    m = thaw_json(r.metrics)
    assert m["excluded_sites"][0]["reason"] == "no_edited_evidence"
    assert m["cds"][0]["spliced_cds_after"] == "ACGCTATAA"
    assert m["cds"][0]["applied_sites"][0]["editing_fraction"] == 0.02
    assert "selected_edit_scenario_not_phased_haplotype" in r.flags
    assert "candidate_sites_not_validated_editing" in r.flags


@pytest.mark.parametrize(
    "edit",
    [
        site(0),
        site(99),
        site(2, ref="A"),
        site(True),
        site(2, editing_fraction=1.1),
        site(2, strand="."),
        site(2, edited="U"),
    ],
)
def test_invalid_sites_fail_explicitly(tmp_path, edit):
    with pytest.raises(OrganelleInputError):
        result_row(genome(tmp_path, "ACGAAATAA"), edit)


def test_duplicate_coordinate_rejected(tmp_path):
    with pytest.raises(OrganelleInputError, match="Duplicate"):
        result_row(genome(tmp_path, "ACGAAATAA"), site(2), site(2))


@pytest.mark.parametrize("code", [28, 999, True, 1.0])
def test_invalid_genetic_codes_rejected(tmp_path, code):
    with pytest.raises(OrganelleParameterError):
        result_row(genome(tmp_path, "ATGAAATAA"), genetic_code=code)


@pytest.mark.parametrize(
    "qualifiers", [{"transl_except": ["(pos:4..6,aa:Sec)"]}, {"exception": ["ribosomal slippage"]}]
)
def test_special_translation_not_silently_ignored(tmp_path, qualifiers):
    with pytest.raises(OrganelleInputError, match="specialized"):
        result_row(genome(tmp_path, "ATGAAATAA", qualifiers=qualifiers))


def test_order_not_joined_and_overlapping_parts_rejected(tmp_path):
    with pytest.raises(OrganelleInputError, match="order"):
        result_row(genome(tmp_path, "ATGAAATAA", [(0, 3), (3, 9)], operator="order"))
    with pytest.raises(OrganelleInputError, match="overlapping"):
        result_row(genome(tmp_path, "ATGAAATAA", [(0, 6), (3, 9)]))


def test_real_pmga_cox2_three_exons_and_two_published_edits():
    data = Path(__file__).parent / "data"
    source = json.loads((data / "pmga_cox2.source.json").read_text())
    result = translate_edited_cds(
        evidence(source["sites"]), annotation_genbank=data / "pmga_cox2.gb"
    )
    row = thaw_json(result.metrics)["cds"][0]
    protein = str(SeqIO.read(data / "pmga_cox2.edited.protein.fasta", "fasta").seq)
    assert row["protein_after"] == protein
    assert len(row["parts"]) == 3 and len(row["spliced_cds_after"]) == 765
    assert row["protein_after"][231:233] == "FM"
    assert row["amino_acid_changes"] == source["expected_changes"]
    assert [s["cds_position"] for s in row["applied_sites"]] == [695, 698]
    assert row["annotated_protein_matches_before"] is True
    assert row["annotated_protein_matches_after"] is False


def test_real_negative_psba_unchanged_matches_curated_translation():
    reference = (
        Path(__file__).parents[2]
        / "src/organelleverse/annotation/data/plastome/references/Arabidopsis_thaliana_chloroplast.gb"
    )
    result = translate_edited_cds(evidence(), annotation_genbank=reference)
    row = next(r for r in thaw_json(result.metrics)["cds"] if r["gene"] == "psbA")
    assert row["protein_before"] == row["protein_after"]
    assert len(row["protein_after"]) == 353
    assert row["annotated_protein_matches_after"] is True


def test_both_editing_sources_capability_binding_writer_and_admission(tmp_path, monkeypatch):
    from organelleverse.capabilities.admission import admit_capabilities
    from organelleverse.capabilities.discovery import discover_capability_candidates
    from organelleverse.capabilities.parser import parse_capability_bundle
    from organelleverse.capabilities.verification import (
        LocalVerificationEnvironment,
        VerificationStore,
        verify_capability,
    )
    from organelleverse.operations.python_binding import bind_python_capability

    monkeypatch.setenv("ORGANELLEVERSE_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    root = Path(__file__).parents[2] / "src/organelleverse/capabilities"
    compute_bundle = parse_capability_bundle(
        root / "annotation-translate-edited-cds/capability.toml"
    )
    bound = bind_python_capability(compute_bundle, translate_edited_cds, None)
    assert set(bound.signature.parameter_model.model_fields) == {
        "annotation_genbank",
        "genetic_code",
    }
    path = genome(tmp_path, "ACGAAATAA")
    for operation in ["rna_editing.detect_editing_sites", "rna_editing.quantify_efficiency"]:
        result = bound.invoke(
            evidence([site(2)], operation=operation), {"annotation_genbank": str(path)}
        )
        assert thaw_json(result.metrics)["cds"][0]["protein_after"] == "MK"
    writer_bundle = parse_capability_bundle(root / "annotation-write-edited-cds/capability.toml")
    writer = bind_python_capability(writer_bundle, write_edited_cds, None)
    written = writer.invoke(result, {"output": str(tmp_path / "export")})
    assert len(written.artifacts) == 5
    assert str(SeqIO.read(tmp_path / "export/protein_after.fasta", "fasta").seq) == "MK"
    assert "cds_position" in (tmp_path / "export/editing_cds_mapping.tsv").read_text()
    index = discover_capability_candidates(entry_points=())
    store = VerificationStore(tmp_path / "verification")
    environment = LocalVerificationEnvironment(index)
    for name in ["annotation.translate_edited_cds", "annotation.write_edited_cds"]:
        verify_capability(name, store=store, environment=environment)
    admitted = admit_capabilities(index, store=store)
    assert admitted.binding_source().resolve("annotation.translate_edited_cds") is not None


@pytest.mark.parametrize("direction", [1, -1])
def test_real_bam_detection_and_quantification_pipe_to_translation(tmp_path, direction):
    import pysam

    from organelleverse.rna_editing import detect_editing_sites
    from organelleverse.rna_editing.efficiency import quantify_known_site_efficiency

    coding = "ACGAAATAA"
    sequence = coding if direction == 1 else str(Seq(coding).reverse_complement())
    pos = 2 if direction == 1 else 8
    alt = "T" if direction == 1 else "A"
    changed = sequence[: pos - 1] + alt + sequence[pos:]
    genbank = genome(tmp_path, sequence, strand=direction)
    fasta = tmp_path / "reference.fasta"
    fasta.write_text(">seq\n" + sequence + "\n")
    bam_path = tmp_path / "reads.bam"
    header = {"HD": {"VN": "1.6", "SO": "coordinate"}, "SQ": [{"SN": "seq", "LN": 9}]}
    with pysam.AlignmentFile(str(bam_path), "wb", header=header) as bam:
        for i in range(10):
            read = pysam.AlignedSegment()
            read.query_name = f"read{i}"
            read.query_sequence = changed if i < 4 else sequence
            read.flag = 0
            read.reference_id = 0
            read.reference_start = 0
            read.mapping_quality = 60
            read.cigarstring = "9M"
            read.query_qualities = [40] * 9
            bam.write(read)
    pysam.index(str(bam_path))
    detected = detect_editing_sites(bam_path, fasta, trim_read_ends=0)
    assert detected.metrics["site_count"] == 1
    sites_tsv = tmp_path / "known.tsv"
    ref = sequence[pos - 1]
    strand = "+" if direction == 1 else "-"
    sites_tsv.write_text(
        f"site_id\tseqid\tposition\tref\tedited\tstrand\ns1\tseq\t{pos}\t{ref}\t{alt}\t{strand}\n"
    )
    quantified = quantify_known_site_efficiency(bam_path, sites_tsv)
    for source in (detected, quantified):
        translated = translate_edited_cds(source, annotation_genbank=genbank)
        row = thaw_json(translated.metrics)["cds"][0]
        assert row["protein_after"] == "MK"
        assert row["start_gain"] is True
        assert row["applied_sites"][0]["depth"] == 10
        assert row["applied_sites"][0]["editing_fraction"] == 0.4
        assert row["applied_sites"][0]["cds_edited_base"] == "T"
        if source.operation_id == "rna_editing.detect_editing_sites":
            assert row["applied_sites"][0]["transcript_edited"] == "U"


def test_ambiguous_definite_stop_gain(tmp_path):
    row = result_row(genome(tmp_path, "ATGCAR"), site(4))
    assert row["stop_gain_positions"] == [2]
    assert row["protein_before"] == "MQ"
    assert row["protein_after"] == "M"


def test_partial_fuzzy_start_does_not_force_methionine(tmp_path):
    from Bio.SeqFeature import BeforePosition

    path = genome(tmp_path, "GTGAAATAA", qualifiers={"transl_table": ["11"]})
    record = SeqIO.read(path, "genbank")
    record.features[0].location = SimpleLocation(BeforePosition(0), 9, strand=1)
    SeqIO.write(record, path, "genbank")
    assert result_row(path)["protein_after"] == "VK"


def test_internal_fuzzy_junction_not_silently_joined(tmp_path):
    from Bio.SeqFeature import BeforePosition

    path = genome(tmp_path, "ATGCCGGGGCATAA", [(0, 5), (10, 14)])
    record = SeqIO.read(path, "genbank")
    record.features[0].location = CompoundLocation(
        [SimpleLocation(0, 5, strand=1), SimpleLocation(BeforePosition(10), 14, strand=1)]
    )
    SeqIO.write(record, path, "genbank")
    with pytest.raises(OrganelleInputError, match="uncertain internal"):
        result_row(path)
