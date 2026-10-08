"""ORF reference-engine equivalence, coordinates, and publication outputs."""

from pathlib import Path

import orfipy_core
import pytest
from Bio.Seq import Seq

from organelleverse import annotation
from organelleverse._bio import read_fasta
from organelleverse.annotation.orfs import find_orfs, write_orfs
from organelleverse.core.errors import OrganelleInputError, OrganelleParameterError
from organelleverse.core.frozen import thaw_json


def fasta(tmp_path, *sequences):
    path = tmp_path / "genome.fa"
    path.write_text("".join(f">seq_{i}\n{seq}\n" for i, seq in enumerate(sequences)))
    return path


def rows(result):
    return thaw_json(result.metrics)["orfs"]


def extract(sequence, row):
    fragments = [sequence[part["start"] - 1 : part["end"]] for part in row["parts"]]
    if row["strand"] == "-":
        fragments = [str(Seq(part).reverse_complement()) for part in fragments]
    return "".join(fragments)


@pytest.mark.parametrize("direction", ["+", "-"])
@pytest.mark.parametrize("offset", [0, 1, 2])
def test_original_coordinates_and_protein_in_all_frames(tmp_path, direction, offset):
    cds = "ATG" + "AAA" * 30 + "TAA"
    oriented = cds if direction == "+" else str(Seq(cds).reverse_complement())
    sequence = "C" * (15 + offset) + oriented + "C" * 6
    found = rows(find_orfs(fasta(tmp_path, sequence), strand=direction))
    assert len(found) == 1
    row = found[0]
    assert (row["start"], row["end"]) == (16 + offset, 15 + offset + len(cds))
    assert extract(sequence, row) == row["cds_sequence"] == cds
    assert row["protein_sequence"] == "M" + "K" * 30


@pytest.mark.parametrize("direction", ["+", "-"])
@pytest.mark.parametrize("cut", [1, 2, 3, 20, 65])
def test_circular_origin_parts_roundtrip(tmp_path, direction, cut):
    cds = "ATG" + "AAA" * 30 + "TAA"
    original = "C" * 15 + cds + "C" * 6
    rotated = original[15 + cut :] + original[: 15 + cut]
    sequence = rotated if direction == "+" else str(Seq(rotated).reverse_complement())
    found = rows(find_orfs(fasta(tmp_path, sequence), strand=direction, circular=True))
    assert len(found) == 1
    row = found[0]
    assert row["wraps_origin"] is True
    assert len(row["parts"]) == 2
    assert extract(sequence, row) == row["cds_sequence"] == cds
    assert row["protein_sequence"] == "M" + "K" * 30
    assert rows(find_orfs(fasta(tmp_path, sequence), strand=direction)) == []


def test_table_starts_and_initial_methionine(tmp_path):
    path = fasta(tmp_path, "CCC" + "GTG" + "AAA" * 30 + "TAA" + "CCC")
    assert not rows(find_orfs(path, strand="+"))
    found = rows(find_orfs(path, strand="+", genetic_code=11, start_mode="table"))
    assert len(found) == 1
    assert found[0]["start_codon"] == "GTG"
    assert found[0]["protein_sequence"] == "M" + "K" * 30


def test_genetic_code_changes_internal_tga_from_stop_to_tryptophan(tmp_path):
    path = fasta(tmp_path, "ATG" + "AAA" * 10 + "TGA" + "AAA" * 20 + "TAA")
    assert not rows(find_orfs(path, strand="+"))
    found = rows(find_orfs(path, strand="+", genetic_code=4))
    assert found[0]["protein_sequence"] == "M" + "K" * 10 + "W" + "K" * 20


def test_nested_same_frame_starts(tmp_path):
    path = fasta(tmp_path, "ATG" + "AAA" * 10 + "ATG" + "AAA" * 30 + "TAA")
    longest = rows(find_orfs(path, strand="+"))
    nested = rows(find_orfs(path, strand="+", include_nested=True))
    assert len(longest) == 1
    assert {row["start"] for row in nested} == {1, 34}
    assert len({row["end"] for row in nested}) == 1


def test_circular_longest_valid_start_survives_oversized_interval(tmp_path):
    # The first start in the doubled sequence gives a CDS > one revolution.
    # The downstream same-frame start is valid and must survive the span filter.
    sequence = "CTAATGATG" + "C" * 20
    expected = sequence[6:] + sequence[:4]
    hits = rows(find_orfs(fasta(tmp_path, sequence), circular=True, strand="+", min_aa=4))
    assert any(row["cds_sequence"] == expected for row in hits)
    assert all(len(row["cds_sequence"]) <= len(sequence) for row in hits)


@pytest.mark.parametrize(
    "sequence,flag",
    [("AAA" * 30 + "TAA", "partial_5prime"), ("ATG" + "AAA" * 30, "partial_3prime")],
)
def test_linear_partial_orfs_are_explicit(tmp_path, sequence, flag):
    path = fasta(tmp_path, sequence)
    assert not rows(find_orfs(path, strand="+"))
    hits = rows(find_orfs(path, strand="+", include_partial=True))
    assert len(hits) == 1
    assert hits[0][flag] is True
    assert extract(sequence, hits[0]) == hits[0]["cds_sequence"]


def test_any_start_does_not_invent_initial_methionine(tmp_path):
    path = fasta(tmp_path, "TAA" + "GTG" + "AAA" * 30 + "TAA")
    hit = rows(find_orfs(path, strand="+", start_mode="any"))[0]
    assert hit["protein_sequence"] == "V" + "K" * 30
    assert hit["start_codon"] is None


def test_any_start_circular_upstream_stop_is_checked_across_origin(tmp_path):
    cds = "GTG" + "AAA" * 30 + "TAA"
    hit = rows(find_orfs(fasta(tmp_path, cds), strand="+", start_mode="any", circular=True))[0]
    assert hit["cds_sequence"] == cds
    assert hit["partial_5prime"] is False


def test_contigs_are_independent_and_ids_are_unique(tmp_path):
    path = fasta(tmp_path, "ATG" + "AAA" * 30, "AAA" * 30 + "TAA")
    assert rows(find_orfs(path, strand="+")) == []
    cds = "ATG" + "AAA" * 30 + "TAA"
    hits = rows(find_orfs(fasta(tmp_path, cds, cds), strand="+"))
    assert len(hits) == 2
    assert len({row["id"] for row in hits}) == 2
    assert {row["sequence_id"] for row in hits} == {"seq_0", "seq_1"}


@pytest.mark.parametrize(
    "sequence",
    [
        "CCC" + "ATG" + "AAA" * 40 + "TAA" + "CCC",
        "TTA" + "TTT" * 40 + "CAT",
        "ATGAAATAGATG" + "CTG" * 30 + "TAA",
    ],
)
def test_complete_linear_coordinates_match_reference_engine(tmp_path, sequence):
    expected = {
        (start + 1, end, direction)
        for start, end, direction, _ in orfipy_core.orfs(
            sequence, starts=["ATG"], minlen=90, include_stop=True
        )
    }
    actual = {
        (row["start"], row["end"], row["strand"])
        for row in rows(find_orfs(fasta(tmp_path, sequence)))
    }
    assert actual == expected


def test_real_nc_000932_psba_dna_and_curated_translation(tmp_path):
    data = Path(__file__).parent / "data"
    coding = read_fasta(data / "nc_000932_psba.fasta")[0][1]
    protein = read_fasta(data / "nc_000932_psba.protein.fasta")[0][1]
    sequence = "C" * 15 + str(Seq(coding).reverse_complement()) + "C" * 6
    hit = next(
        row
        for row in rows(find_orfs(fasta(tmp_path, sequence), min_aa=300))
        if row["strand"] == "-"
    )
    assert (hit["start"], hit["end"]) == (16, 15 + len(coding))
    assert hit["cds_sequence"] == coding
    assert hit["protein_sequence"] == protein


@pytest.mark.parametrize("direction", ["+", "-"])
def test_writer_exports_circular_sequences_and_gff_phases(tmp_path, direction):
    cds = "ATG" + "AAA" * 30 + "TAA"
    sequence = "C" * 15 + cds + "C" * 6
    sequence = sequence[17:] + sequence[:17]
    if direction == "-":
        sequence = str(Seq(sequence).reverse_complement())
    result = find_orfs(fasta(tmp_path, sequence), circular=True, strand=direction)
    written = write_orfs(result, output=tmp_path / "out")
    assert written.operation_id == "annotation.write_orfs"
    assert len(written.artifacts) == 3
    assert read_fasta(tmp_path / "out/orfs.cds.fasta") == [("orf_1", cds)]
    assert read_fasta(tmp_path / "out/orfs.proteins.fasta") == [("orf_1", "M" + "K" * 30)]
    lines = [
        line
        for line in (tmp_path / "out/orfs.gff3").read_text().splitlines()
        if not line.startswith("#")
    ]
    assert len(lines) == 2
    assert lines[0].split("\t")[7] == "0"
    first = rows(result)[0]["parts"][0]
    assert int(lines[1].split("\t")[7]) == (-(first["end"] - first["start"] + 1)) % 3
    assert all("ID=orf_1;" in line for line in lines)


def test_circular_candidate_is_invariant_to_every_origin_position(tmp_path):
    sequence = "CTAATGATG" + "C" * 20
    expected = sequence[6:] + sequence[:4]
    for cut in range(len(sequence)):
        rotated = sequence[cut:] + sequence[:cut]
        for direction in ("+", "-"):
            oriented = rotated if direction == "+" else str(Seq(rotated).reverse_complement())
            hits = rows(
                find_orfs(fasta(tmp_path, oriented), circular=True, strand=direction, min_aa=4)
            )
            assert [hit["cds_sequence"] for hit in hits] == [expected], (cut, direction)
            assert extract(oriented, hits[0]) == expected


def test_missing_engine_raises_dependency_error_without_fallback(tmp_path, monkeypatch):
    import sys

    from organelleverse.core.errors import OrganelleDependencyError

    monkeypatch.setitem(sys.modules, "orfipy_core", None)
    with pytest.raises(OrganelleDependencyError, match="organelleverse\\[orf\\]"):
        find_orfs(fasta(tmp_path, "ATGAAATAA"))


@pytest.mark.parametrize(
    "options",
    [
        {"genetic_code": 999},
        {"genetic_code": 28},
        {"min_aa": 0},
        {"circular": True, "include_partial": True},
        {"start_mode": "bogus"},
    ],
)
def test_unsupported_scientific_options_fail_clearly(tmp_path, options):
    with pytest.raises(OrganelleParameterError):
        find_orfs(fasta(tmp_path, "ATGAAATAA"), **options)


def test_invalid_alignment_input_is_rejected(tmp_path):
    with pytest.raises(OrganelleInputError):
        find_orfs(fasta(tmp_path, "ATG---TAA"))


def test_definite_ambiguous_stop_is_not_translated_inside_orf(tmp_path):
    hit = rows(find_orfs(fasta(tmp_path, "ATGAAATARAAATAA"), min_aa=2, strand="+"))[0]
    assert hit["cds_sequence"] == "ATGAAATAR"
    assert hit["stop_codon"] == "TAR"
    assert hit["protein_sequence"] == "MK"
    assert hit["ambiguous_bases"] == 1


def test_public_exports_and_capability_contracts():
    from organelleverse.capabilities.parser import parse_capability_bundle
    from organelleverse.operations.dependencies import check_dependencies
    from organelleverse.operations.registry import OperationRegistry

    assert annotation.find_orfs is find_orfs
    assert annotation.write_orfs is write_orfs
    root = Path(__file__).parents[2] / "src/organelleverse/capabilities"
    discover = parse_capability_bundle(root / "annotation-find-orfs/capability.toml")
    writer = parse_capability_bundle(root / "annotation-write-orfs/capability.toml")
    assert discover.capability.id == "annotation.find_orfs"
    assert check_dependencies(discover.contract).ready
    # The writer consumes the canonical result through the shared typed registry.
    binding = OperationRegistry().register(writer.contract, write_orfs)
    assert binding.spec.operation_id == "annotation.write_orfs"


def test_capability_binding_exposes_options_and_executes_both_operations(tmp_path, monkeypatch):
    from organelleverse.capabilities.parser import parse_capability_bundle
    from organelleverse.operations.python_binding import bind_python_capability

    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    root = Path(__file__).parents[2] / "src/organelleverse/capabilities"
    compute_bundle = parse_capability_bundle(root / "annotation-find-orfs/capability.toml")
    writer_bundle = parse_capability_bundle(root / "annotation-write-orfs/capability.toml")
    # None is permitted for verification-stage binding; admission still requires
    # a verification record and frozen schema in the production registry.
    compute = bind_python_capability(compute_bundle, find_orfs, None)
    assert set(compute.signature.parameter_model.model_fields) == {
        "genome_fasta",
        "organelle",
        "genetic_code",
        "min_aa",
        "start_mode",
        "strand",
        "include_nested",
        "include_partial",
        "circular",
    }
    path = fasta(tmp_path, "GTG" + "AAA" * 30 + "TAA")
    result = compute.invoke(
        None, {"genome_fasta": str(path), "genetic_code": 11, "start_mode": "table", "strand": "+"}
    )
    assert rows(result)[0]["protein_sequence"] == "M" + "K" * 30
    writer = bind_python_capability(writer_bundle, write_orfs, None)
    output = writer.invoke(result, {"output": str(tmp_path / "out")})
    assert len(output.artifacts) == 3


def test_real_reference_fixture_verification_and_admission(tmp_path, monkeypatch):
    from organelleverse.capabilities.admission import admit_capabilities
    from organelleverse.capabilities.discovery import discover_capability_candidates
    from organelleverse.capabilities.verification import (
        LocalVerificationEnvironment,
        VerificationStore,
        verify_capability,
    )

    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    monkeypatch.setenv("ORGANELLEVERSE_HOME", str(tmp_path / "home"))
    index = discover_capability_candidates(entry_points=())
    store = VerificationStore(tmp_path / "verification")
    environment = LocalVerificationEnvironment(index)
    compute = verify_capability("annotation.find_orfs", store=store, environment=environment)
    assert {case.case for case in compute.equivalence} == {
        "nc_000932_psba_negative",
        "context_dependent_code_rejected",
    }
    assert all(case.verdict == "pass" for case in compute.equivalence)
    verify_capability("annotation.write_orfs", store=store, environment=environment)
    admitted = admit_capabilities(index, store=store)
    registry = admitted.binding_source()
    bound = registry.resolve("annotation.find_orfs")
    assert bound is not None
    source = (
        Path(__file__).parents[2]
        / "src/organelleverse/capabilities/annotation-find-orfs/fixtures/psba-negative.fasta"
    )
    result = bound.invoke(
        None, {"genome_fasta": str(source), "genetic_code": 11, "min_aa": 300, "strand": "-"}
    )
    assert len(rows(result)) == 1
