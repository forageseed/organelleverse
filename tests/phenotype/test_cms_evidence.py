"""Auditable CMS evidence, corrected reference identities and failure contracts."""

import io
import json
import os
import subprocess
from pathlib import Path

import pytest
from Bio import SeqIO

from organelleverse._bio import read_fasta
from organelleverse.core.errors import (
    OrganelleDependencyError,
    OrganelleExecutionError,
    OrganelleInputError,
    OrganelleParameterError,
)
from organelleverse.core.frozen import thaw_json
from organelleverse.phenotype.cms import (
    assess_cms_candidates,
    cms_protein_fasta,
    cms_reference_json,
    write_cms_evidence,
)
from organelleverse.phenotype.cms.evidence import _expression, _parse_hits, _unmatched

DATA = Path(__file__).parent / "data"


def test_reviewed_seed_accession_lengths_and_genomic_translation():
    metadata = json.loads(cms_reference_json().read_text())
    proteins = dict(read_fasta(cms_protein_fasta()))
    assert set(proteins) == {"orf79", "orf138"}
    assert [(m["name"], m["length_aa"]) for m in metadata] == [("orf79", 79), ("orf138", 138)]
    for row in metadata:
        record = SeqIO.read(DATA / f"{row['nucleotide_accession']}.gb", "genbank")
        feature = next(
            f
            for f in record.features
            if f.type == "CDS" and f.qualifiers.get("protein_id") == [row["accession"]]
        )
        assert feature.qualifiers["translation"][0] == proteins[row["name"]]
        assert str(feature.extract(record.seq).translate(to_stop=True)) == proteins[row["name"]]


def test_curated_nested_asset_does_not_change_parent_legacy_asset():
    from organelleverse.assets import verify

    assert verify("ov-asset:phenotype/cms_curated_reference@1.0")
    assert verify("ov-asset:phenotype/cms_reference@1.0")


def fake_tool(monkeypatch, *, output=""):
    import importlib

    implementation = importlib.import_module("organelleverse.phenotype.cms.evidence")

    calls = []
    monkeypatch.setattr(implementation.shutil, "which", lambda _: "/checked/LOSAT")

    def run(argv, **kwargs):
        calls.append(argv)
        if "--version" in argv:
            return subprocess.CompletedProcess(argv, 0, stdout="losat test\n", stderr="")
        return subprocess.CompletedProcess(argv, 0, stdout=output, stderr="")

    monkeypatch.setattr(implementation, "run_external", run)
    return calls


def test_no_hits_is_not_assigned_cms_causality_or_chimerism(tmp_path, monkeypatch):
    calls = fake_tool(monkeypatch)
    result = assess_cms_candidates(
        DATA / "cms_queries.fasta", conserved_proteins_fasta=DATA / "conserved_controls.fasta"
    )
    rows = thaw_json(result.metrics)["candidate_table"]
    assert len(rows) == 5
    assert all(r["evidence_labels"] == ["uncharacterized_orf"] for r in rows)
    assert all(r["unmatched_regions"] == [{"start": 1, "end": r["length_aa"]}] for r in rows)
    assert "cms_causality_not_established" in result.flags
    assert "fragment_homology_not_confirmed_chimerism" in result.flags
    assert "--seg" in calls[1] and calls[1][calls[1].index("--seg") + 1] == "no"
    assert "score" not in rows[0]
    assert result.metrics["alignment_backend"] == "losat_blastp"
    for call in calls[1:]:
        assert call[:2] == ["/checked/LOSAT", "blastp"]
        assert call[call.index("--comp-based-stats") + 1] == "2"
        assert call[call.index("--outfmt") + 1].endswith("qlen slen nident")


@pytest.mark.parametrize(
    "intervals,expected",
    [
        ([], [{"start": 1, "end": 10}]),
        ([(1, 10)], []),
        ([(2, 4), (4, 7)], [{"start": 1, "end": 1}, {"start": 8, "end": 10}]),
        (
            [(3, 4), (7, 8)],
            [{"start": 1, "end": 2}, {"start": 5, "end": 6}, {"start": 9, "end": 10}],
        ),
    ],
)
def test_unmatched_intervals_union_not_hit_count(intervals, expected):
    assert _unmatched(10, intervals) == expected


def test_parser_reports_both_coverages_and_thresholds():
    text = "q\ts\t100\t30\t1\t30\t1\t30\t1e-12\t90\t100\t30\t30\n"
    row = _parse_hits(text, {"q": "A" * 100}, {"s": "A" * 30}, 80, 20)["q"][0]
    assert row["query_coverage"] == 0.3 and row["subject_coverage"] == 1
    assert _parse_hits(text, {"q": "A" * 100}, {"s": "A" * 30}, 80, 31)["q"] == []


@pytest.mark.parametrize(
    "text",
    [
        "invalid",
        "q\ts\tx\t30\t1\t30\t1\t30\t1e-12\t90\t100\t30\t30",
        "q\ts\t100\t30\t0\t30\t1\t30\t1e-12\t90\t100\t30\t30",
        "q\ts\t100\t30\t1\t30\t1\t30\t1e-12\t90\t101\t30\t30",
        "other\ts\t100\t30\t1\t30\t1\t30\t1e-12\t90\t100\t30\t30",
    ],
)
def test_bad_blast_output_fails_instead_of_silent_empty_hit(text):
    with pytest.raises(OrganelleInputError):
        _parse_hits(text, {"q": "A" * 100}, {"s": "A" * 30}, 80, 20)


def test_expression_descriptive_ratios_zero_denominator_and_ids(tmp_path):
    path = tmp_path / "expression.tsv"
    path.write_text(
        "candidate_id\tcondition\treplicate\ttpm\nq\tsterile\t1\t8\nq\tsterile\t2\t12\nq\tmaintainer\t1\t2\nq\trestored\t1\t0\n"
    )
    data = _expression(path, {"q": "AAA"})["q"]
    assert data["mean_tpm"]["sterile"] == 10
    assert data["ratios"] == {"sterile_over_maintainer": 5, "sterile_over_restored": None}
    assert len(data["observations"]["sterile"]) == 2
    with pytest.raises(OrganelleInputError):
        _expression(path, {"other": "AAA"})


@pytest.mark.parametrize(
    "body",
    [
        "q\tsterile\t1\t-1",
        "q\tsterile\t1\tnan",
        "q\tunknown\t1\t5",
        "q\tsterile\t\t5",
        "q\tsterile\t1\t5\nq\tsterile\t1\t5",
    ],
)
def test_expression_invalid_or_duplicate_rows_rejected(tmp_path, body):
    path = tmp_path / "expression.tsv"
    path.write_text("candidate_id\tcondition\treplicate\ttpm\n" + body + "\n")
    with pytest.raises(OrganelleInputError):
        _expression(path, {"q": "AAA"})


@pytest.mark.parametrize(
    "options",
    [
        {"min_homolog_coverage": 0},
        {"min_identity": 101},
        {"evalue": 0},
        {"min_alignment_aa": 0},
        {"tm_window": 0},
        {"threads": True},
    ],
)
def test_invalid_thresholds_fail_before_external_execution(options):
    with pytest.raises(OrganelleParameterError):
        assess_cms_candidates(
            DATA / "cms_queries.fasta",
            conserved_proteins_fasta=DATA / "conserved_controls.fasta",
            **options,
        )


def test_missing_losat_explicit_dependency_error():
    with pytest.raises(OrganelleDependencyError):
        assess_cms_candidates(
            DATA / "cms_queries.fasta",
            conserved_proteins_fasta=DATA / "conserved_controls.fasta",
            losat_path="missing-cms-losat-executable",
        )


@pytest.mark.skipif(os.name == "nt", reason="POSIX executable fixture validates server stderr")
def test_external_failure_keeps_stderr(tmp_path):
    tool = tmp_path / "LOSAT"
    tool.write_text('#!/bin/sh\necho "specific BLAST failure cause" >&2\nexit 7\n')
    tool.chmod(0o755)
    with pytest.raises(OrganelleExecutionError, match="specific BLAST failure cause") as error:
        assess_cms_candidates(
            DATA / "cms_queries.fasta",
            conserved_proteins_fasta=DATA / "conserved_controls.fasta",
            losat_path=str(tool),
        )
    assert error.value.details["returncode"] == 7


def test_binding_all_options_fixture_verification_and_writer(tmp_path, monkeypatch):
    from organelleverse.capabilities.admission import admit_capabilities
    from organelleverse.capabilities.discovery import discover_capability_candidates
    from organelleverse.capabilities.parser import parse_capability_bundle
    from organelleverse.capabilities.verification import (
        LocalVerificationEnvironment,
        VerificationStore,
        verify_capability,
    )
    from organelleverse.operations.python_binding import bind_python_capability

    fake_tool(monkeypatch)
    monkeypatch.setenv("ORGANELLEVERSE_HOME", str(tmp_path / "home"))
    root = Path(__file__).parents[2] / "src/organelleverse/capabilities"
    bundle = parse_capability_bundle(root / "phenotype-assess-cms-candidates/capability.toml")
    bound = bind_python_capability(bundle, assess_cms_candidates, None)
    assert set(bound.signature.parameter_model.model_fields) == {
        "protein_fasta",
        "conserved_proteins_fasta",
        "cms_proteins_fasta",
        "expression_tsv",
        "topology_predictions",
        "min_identity",
        "min_homolog_coverage",
        "min_alignment_aa",
        "evalue",
        "tm_window",
        "tm_threshold",
        "threads",
        "timeout",
        "losat_path",
    }
    result = bound.invoke(
        None,
        {
            "protein_fasta": str(DATA / "cms_queries.fasta"),
            "conserved_proteins_fasta": str(DATA / "conserved_controls.fasta"),
        },
    )
    writer_bundle = parse_capability_bundle(root / "phenotype-write-cms-evidence/capability.toml")
    writer = bind_python_capability(writer_bundle, write_cms_evidence, None)
    exported = writer.invoke(result, {"output": str(tmp_path / "output")})
    assert len(exported.artifacts) == 2
    assert (
        json.loads((tmp_path / "output/cms_evidence.json").read_text())["metrics"][
            "candidate_count"
        ]
        == 5
    )
    index = discover_capability_candidates(entry_points=())
    store = VerificationStore(tmp_path / "verification")
    env = LocalVerificationEnvironment(index)
    record = verify_capability("phenotype.assess_cms_candidates", store=store, environment=env)
    assert record.equivalence[0].verdict == "pass"
    verify_capability("phenotype.write_cms_evidence", store=store, environment=env)
    assert (
        admit_capabilities(index, store=store)
        .binding_source()
        .resolve("phenotype.assess_cms_candidates")
        is not None
    )


def test_download_pinned_seed_to_cache_preserves_package(monkeypatch, tmp_path):
    from Bio import Entrez

    from organelleverse.phenotype.cms.data import download

    metadata = json.loads(cms_reference_json().read_text())
    proteins = dict(read_fasta(cms_protein_fasta()))
    original = cms_protein_fasta().read_bytes()

    def fetch(**kwargs):
        row = next(r for r in metadata if r["accession"] == kwargs["id"])
        return io.StringIO(f">{row['accession']}\n{proteins[row['name']]}\n")

    monkeypatch.setattr(Entrez, "efetch", fetch)
    monkeypatch.setattr(download.time, "sleep", lambda _: None)
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    path = download.download_cms_sequences(email="test@example.com")
    assert dict(read_fasta(path)) == proteins
    assert path.is_relative_to(tmp_path / "cache")
    assert cms_protein_fasta().read_bytes() == original
    saved = path.read_bytes()
    monkeypatch.setattr(Entrez, "efetch", lambda **kw: io.StringIO(f">{kw['id']}\nM{'A' * 78}\n"))
    with pytest.raises(OrganelleInputError, match="reviewed"):
        download.download_cms_sequences()
    assert path.read_bytes() == saved


@pytest.mark.integration
def test_live_losat_known_references_and_conserved_hydrophobic_controls():
    result = assess_cms_candidates(
        DATA / "cms_queries.fasta", conserved_proteins_fasta=DATA / "conserved_controls.fasta"
    )
    rows = {r["candidate_id"]: r for r in thaw_json(result.metrics)["candidate_table"]}
    for name in ["D14339.1_orf79", "Z18896.1_orf138"]:
        assert "known_cms_reference_homolog" in rows[name]["evidence_labels"]
        assert rows[name]["cms_reference_hits"][0]["identity_percent"] == 100
        assert rows[name]["cms_reference_hits"][0]["query_coverage"] == 1
    for gene in ["cox1", "atp6", "nad9"]:
        row = rows["control_" + gene]
        assert "conserved_gene_homolog" in row["evidence_labels"]
        assert "known_cms_reference_homolog" not in row["evidence_labels"]
        assert row["unmatched_regions"] == []


@pytest.mark.integration
def test_real_orf_discovery_to_cms_evidence_chain(tmp_path):
    from organelleverse.annotation import find_orfs, write_orfs

    for accession, product in [("D14339.1", "ORF79"), ("Z18896.1", "orf138")]:
        record = SeqIO.read(DATA / f"{accession}.gb", "genbank")
        fasta = tmp_path / f"{accession}.fasta"
        SeqIO.write(record, fasta, "fasta")
        orfs = find_orfs(fasta, organelle="mitochondrion", min_aa=30)
        exported = write_orfs(orfs, output=tmp_path / accession)
        protein_path = next(
            a.resolve() for a in exported.artifacts if a.uri.endswith("orfs.proteins.fasta")
        )
        result = assess_cms_candidates(
            protein_path, conserved_proteins_fasta=DATA / "conserved_controls.fasta"
        )
        feature = next(
            f
            for f in record.features
            if f.type == "CDS" and f.qualifiers.get("product") == [product]
        )
        target = feature.qualifiers["translation"][0]
        row = next(
            r
            for r in thaw_json(result.metrics)["candidate_table"]
            if r["protein_sequence"] == target
        )
        assert "known_cms_reference_homolog" in row["evidence_labels"]


def test_public_root_exports_and_partial_match_retains_unknown_region(monkeypatch):
    import importlib

    from organelleverse import phenotype

    assert phenotype.assess_cms_candidates is assess_cms_candidates
    assert phenotype.write_cms_evidence is write_cms_evidence
    implementation = importlib.import_module("organelleverse.phenotype.cms.evidence")
    fake_tool(monkeypatch)
    # A real core-control protein's first 60 residues have a passing match;
    # the rest is unmatched. This is fragment evidence, not a confirmed chimera.
    original = implementation._parse_hits

    def parse(text, queries, subjects, identity, min_aa):
        if "NC_021152_cox1" in subjects:
            qlen = len(queries["control_cox1"])
            slen = len(subjects["NC_021152_cox1"])
            text = f"control_cox1\tNC_021152_cox1\t100\t60\t1\t60\t1\t60\t1e-12\t90\t{qlen}\t{slen}\t60\n"
        return original(text, queries, subjects, identity, min_aa)

    monkeypatch.setattr(implementation, "_parse_hits", parse)
    result = assess_cms_candidates(
        DATA / "cms_queries.fasta", conserved_proteins_fasta=DATA / "conserved_controls.fasta"
    )
    row = next(
        r
        for r in thaw_json(result.metrics)["candidate_table"]
        if r["candidate_id"] == "control_cox1"
    )
    assert row["evidence_labels"] == ["partial_conserved_homology_with_unmatched_sequence"]
    assert row["unmatched_regions"] == [{"start": 61, "end": row["length_aa"]}]


def test_wrong_executable_is_rejected_without_fallback(monkeypatch):
    import importlib

    implementation = importlib.import_module("organelleverse.phenotype.cms.evidence")
    monkeypatch.setattr(implementation.shutil, "which", lambda _: "/wrong/blastp")
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout="blastp: 2.15.0+\n", stderr="")

    monkeypatch.setattr(implementation, "run_external", run)
    with pytest.raises(OrganelleDependencyError, match="Expected Rust LOSAT"):
        assess_cms_candidates(
            DATA / "cms_queries.fasta", conserved_proteins_fasta=DATA / "conserved_controls.fasta"
        )
    assert calls == [["/wrong/blastp", "--version"]]


@pytest.mark.integration
def test_live_losat_held_out_orf79_variant():
    result = assess_cms_candidates(
        DATA / "orf79_variant.fasta", conserved_proteins_fasta=DATA / "conserved_controls.fasta"
    )
    row = thaw_json(result.metrics)["candidate_table"][0]
    assert row["candidate_id"] == "AGC92804.1"
    assert row["evidence_labels"] == ["known_cms_reference_homolog"]
    hit = row["cms_reference_hits"][0]
    assert hit["subject_id"] == "orf79"
    assert hit["identical_residues"] == 78
    assert hit["alignment_length"] == 79
    assert hit["query_coverage"] == hit["subject_coverage"] == 1


@pytest.mark.integration
@pytest.mark.parametrize("query_name", ["cms_queries", "orf79_variant", "D14339.1", "Z18896.1"])
@pytest.mark.parametrize("subject_kind", ["cms", "core"])
@pytest.mark.parametrize("threads", [1, 2])
def test_losat_raw_output_matches_ncbi_reference_only(tmp_path, query_name, subject_kind, threads):
    # NCBI is an oracle in this test only, never a runtime backend/fallback.
    from organelleverse.annotation import find_orfs, write_orfs

    if query_name in {"D14339.1", "Z18896.1"}:
        record = SeqIO.read(DATA / f"{query_name}.gb", "genbank")
        genome = tmp_path / "genome.fasta"
        SeqIO.write(record, genome, "fasta")
        result = write_orfs(
            find_orfs(genome, organelle="mitochondrion", min_aa=30), output=tmp_path / "orfs"
        )
        query = next(a.resolve() for a in result.artifacts if a.uri.endswith("orfs.proteins.fasta"))
    else:
        query = DATA / f"{query_name}.fasta"
    subject = cms_protein_fasta() if subject_kind == "cms" else DATA / "conserved_controls.fasta"
    count = len(read_fasta(subject))
    fields = (
        "6 qseqid sseqid pident length qstart qend sstart send evalue bitscore qlen slen nident"
    )
    rust = [
        "LOSAT",
        "blastp",
        "--task",
        "blastp",
        "--query",
        str(query),
        "--subject",
        str(subject),
        "--outfmt",
        fields,
        "--seg",
        "no",
        "--comp-based-stats",
        "2",
        "--evalue",
        "1e-5",
        "--num-threads",
        str(threads),
        "--max-target-seqs",
        str(count),
    ]
    oracle = [
        "blastp",
        "-task",
        "blastp",
        "-query",
        str(query),
        "-subject",
        str(subject),
        "-outfmt",
        fields,
        "-seg",
        "no",
        "-comp_based_stats",
        "2",
        "-evalue",
        "1e-5",
        "-num_threads",
        str(threads),
        "-max_target_seqs",
        str(count),
    ]
    actual = subprocess.run(rust, capture_output=True, check=True, timeout=120).stdout
    expected = subprocess.run(oracle, capture_output=True, check=True, timeout=120).stdout
    assert actual == expected


@pytest.mark.skipif(
    os.name == "nt", reason="POSIX executable fixture validates server unsupported options"
)
def test_unsupported_losat_search_fails_with_stderr_without_fallback(tmp_path):
    tool = tmp_path / "LOSAT"
    tool.write_text(
        '#!/bin/sh\nif [ "$1" = "--version" ]; then echo "losat 0.1.0"; exit 0; fi\necho "unsupported comp-based-stats" >&2\nexit 2\n'
    )
    tool.chmod(0o755)
    with pytest.raises(OrganelleExecutionError, match="unsupported comp-based-stats"):
        assess_cms_candidates(
            DATA / "cms_queries.fasta",
            conserved_proteins_fasta=DATA / "conserved_controls.fasta",
            losat_path=str(tool),
        )
