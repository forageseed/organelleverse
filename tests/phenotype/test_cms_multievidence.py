"""CMS evidence axes and actual annotation/export identity joins."""

import importlib
import json
import subprocess
from pathlib import Path

import pytest

from organelleverse.annotation.models import (
    AnnotationDocument,
    AnnotationFeature,
    AnnotationRecord,
    FeatureQualifier,
    LocationPart,
)
from organelleverse.annotation.writer import materialize_annotation
from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.errors import OrganelleInputError
from organelleverse.core.frozen import FrozenMap, thaw_json
from organelleverse.core.result import OrganelleResult
from organelleverse.phenotype import assess_cms_annotation, predict_cms_topology, write_cms_evidence
from organelleverse.phenotype.cms.annotation_evidence import _context
from organelleverse.phenotype.cms.topology import load_tmbed_predictions

DATA = Path(__file__).parent / "data"


def fake_losat(monkeypatch):
    module = importlib.import_module("organelleverse.phenotype.cms.evidence")
    monkeypatch.setattr(module.shutil, "which", lambda _: "/tools/LOSAT")
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(
            argv, 0, stdout="losat test" if "--version" in argv else "", stderr=""
        )

    monkeypatch.setattr(module, "run_external", run)
    return calls


def feature(name, start, end, *, gene=None, strand=1, translation="MK"):
    qualifiers = [FeatureQualifier(name="translation", values=(translation,))]
    if gene:
        qualifiers.append(FeatureQualifier(name="gene", values=(gene,)))
    return AnnotationFeature(
        feature_id=name,
        seqid="r1",
        type="CDS",
        operator="single",
        parts=(LocationPart(start=start, end=end, strand=strand),),
        parents=(),
        qualifiers=tuple(qualifiers),
    )


def source(tmp_path, *, warning=False, duplicate=False):
    doc = AnnotationDocument(
        backend="mitochondrion",
        requested_stages=("pcg",),
        completed_stages=("pcg",),
        records=(
            AnnotationRecord(
                seqid="r1",
                name="r1",
                description="test",
                sequence="ATGAAATAACCCATGAAATAA",
                features=(
                    feature("f1", 0, 9, gene="nad1"),
                    feature("f2", 12, 21, gene="nad1" if duplicate else "orf2"),
                ),
            ),
        ),
        source_metadata={"topology": "circular"},
    )
    paths = materialize_annotation(doc, tmp_path / "annotation")
    return OrganelleResult(
        operation_id="annotation.annotate",
        scope="mitochondrion",
        status="warning" if warning else "ok",
        flags=("missing_expected_genes",) if warning else (),
        artifacts=(ArtifactRef.from_path(paths["json"], kind="annotation", format="json"),),
    ), doc


def test_topology_segments_signal_is_not_transmembrane(tmp_path):
    path = tmp_path / "pred"
    path.write_text(">p\nMACDEFGHIKLM\nSS.HHHhhBBb.\n")
    row = load_tmbed_predictions(path, {"p": "MACDEFGHIKLM"})["p"]
    assert row["transmembrane_segment_count"] == 4
    assert row["signal_peptide_predicted"]
    assert row["segments"][1] == {
        "start": 4,
        "end": 6,
        "type": "alpha_helix",
        "orientation": "inside_to_outside",
    }
    path.write_text(">p\nMAC\nSSS\n")
    assert load_tmbed_predictions(path, {"p": "MAC"})["p"]["transmembrane_segment_count"] == 0


@pytest.mark.parametrize(
    "text",
    [
        ">p\nMAD\n...\n",
        ">p\nMAC\nHH\n",
        ">p\nMAC\nioo\n",
        ">unknown\nMAC\n...\n",
        ">p\nMAC\n...\n>p\nMAC\n...\n",
        ">p\nMAC\n",
        "",
    ],
)
def test_topology_rejects_mismatched_partial_duplicate_or_unknown_input(tmp_path, text):
    path = tmp_path / "pred"
    path.write_text(text)
    with pytest.raises(OrganelleInputError):
        load_tmbed_predictions(path, {"p": "MAC"})


def test_topology_execution_normalizes_terminal_stop_and_requires_offline_model(
    tmp_path, monkeypatch
):
    module = importlib.import_module("organelleverse.phenotype.cms.topology")
    query = tmp_path / "query.fa"
    query.write_text(">p\nMAC*\n")
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    monkeypatch.setattr(module.shutil, "which", lambda _: "/tools/tmbed")
    monkeypatch.setenv("HF_HOME", "/wrong/model")

    def run(argv, **kwargs):
        assert "--no-cpu-fallback" in argv and "--no-use-gpu" in argv
        assert "HF_HOME" not in kwargs["env"] and kwargs["env"]["HF_HUB_OFFLINE"] == "1"
        assert "MAC*" not in Path(argv[argv.index("-f") + 1]).read_text()
        Path(argv[argv.index("-p") + 1]).write_text(">p\nMAC\nHHH\n")

    monkeypatch.setattr(module, "run_external", run)
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    result = predict_cms_topology(query, model_dir=model)
    assert result.metrics["prediction_count"] == 1
    assert result.artifacts[0].resolve().read_text() == ">p\nMAC\nHHH\n"

    def partial(argv, **kwargs):
        Path(argv[argv.index("-p") + 1]).write_text(">p\nMAD\nHHH\n")

    monkeypatch.setattr(module, "run_external", partial)
    with pytest.raises(OrganelleInputError):
        predict_cms_topology(query, model_dir=model)
    assert result.artifacts[0].resolve().read_text() == ">p\nMAC\nHHH\n"


def test_annotation_warning_all_axes_and_writer_export_ids(tmp_path, monkeypatch):
    fake_losat(monkeypatch)
    annotation, _doc = source(tmp_path, warning=True)
    topology = tmp_path / "pred"
    topology.write_text(">nad1\nMK\nHH\n>orf2\nMK\n..\n")
    expression = tmp_path / "tpm.tsv"
    expression.write_text(
        "candidate_id\tcondition\treplicate\ttpm\nnad1\tsterile\t1\t10\nnad1\tmaintainer\t1\t2\n"
    )
    result = assess_cms_annotation(
        annotation,
        conserved_proteins_fasta=DATA / "conserved_controls.fasta",
        topology_predictions=topology,
        expression_tsv=expression,
        min_alignment_aa=1,
    )
    assert result.status == "warning" and "source_annotation_needs_review" in result.flags
    rows = thaw_json(result.metrics)["candidate_table"]
    assert [row["candidate_id"] for row in rows] == ["nad1", "orf2"]
    assert rows[0]["expression"]["ratios"]["sterile_over_maintainer"] == 5
    assert rows[0]["transmembrane_topology"]["transmembrane_segment_count"] == 1
    assert (
        rows[1]["expression"] is None
        and not rows[1]["evidence_availability"]["condition_expression"]
    )
    assert rows[0]["genomic_context"]["neighbors"][0]["minimum_gap_bases"] == 0  # circular boundary
    assert all(row["evidence_availability"]["genomic_context"] for row in rows)
    written = write_cms_evidence(result, output=tmp_path / "report")
    assert written.status == "warning"
    assert "genomic_context" in (tmp_path / "report/cms_evidence.tsv").read_text()
    assert json.loads((tmp_path / "report/cms_evidence.json").read_text())["flags"] == list(
        result.flags
    )


def test_duplicate_export_ids_require_explicit_feature_id_mode(tmp_path, monkeypatch):
    fake_losat(monkeypatch)
    annotation, _ = source(tmp_path, duplicate=True)
    with pytest.raises(OrganelleInputError, match="ambiguous"):
        assess_cms_annotation(
            annotation, conserved_proteins_fasta=DATA / "conserved_controls.fasta"
        )
    result = assess_cms_annotation(
        annotation,
        conserved_proteins_fasta=DATA / "conserved_controls.fasta",
        protein_id_mode="feature_id",
    )
    assert [row["candidate_id"] for row in result.metrics["candidate_table"]] == ["f1", "f2"]


def test_negative_joined_overlap_uses_union(tmp_path):
    f = feature("f", 0, 9, strand=-1)
    other = f.evolve(
        feature_id="g",
        operator="join",
        parts=(LocationPart(start=4, end=9, strand=-1), LocationPart(start=0, end=6, strand=-1)),
    )
    record = AnnotationRecord(
        seqid="r1", name="r1", description="", sequence="A" * 21, features=(f, other)
    )
    row = _context(f, record, False, 5)
    assert row["neighbors"][0]["overlapping_bases"] == 9
    assert row["parts"][0]["strand"] == -1


def test_edited_mapping_preserves_before_and_reassesses_selected_after(tmp_path, monkeypatch):
    calls = fake_losat(monkeypatch)
    annotation, doc = source(tmp_path)
    # GenBank reparse IDs may differ: use exact sequence ID + ordered locations.
    row = {
        "feature_id": "reparsed-id",
        "seqid": "r1",
        "parts": [p.model_dump(mode="json") for p in doc.records[0].features[0].parts],
        "spliced_cds_before": "ATGAAATAA",
        "genetic_code": 1,
        "codon_start": 1,
        "protein_before": "MK",
        "protein_after": "ME",
        "applied_sites": [{"position": 4, "ref": "A", "edited": "G"}],
    }
    edited = OrganelleResult(
        operation_id="annotation.translate_edited_cds",
        scope="mitochondrion",
        status="ok",
        metrics={"cds": [row]},
        flags=("selected_edit_scenario_not_phased_haplotype",),
    )
    path = tmp_path / "edited.json"
    path.write_text(edited.model_dump_json())
    result = assess_cms_annotation(
        annotation, conserved_proteins_fasta=DATA / "conserved_controls.fasta", edited_cds_json=path
    )
    rows = thaw_json(result.metrics)["candidate_table"]
    assert len(calls) == 6  # before and edited scenarios, each version + 2 LOSAT calls
    assert rows[0]["protein_sequence"] == "MK"
    assert rows[0]["rna_editing"]["protein_after"] == "ME"
    assert "selected_edit_scenario_homology" in rows[0]["rna_editing"]
    assert rows[1]["rna_editing"] is None
    row["spliced_cds_before"] = "ATGCCCTAA"
    path.write_text(
        edited.model_copy(update={"metrics": FrozenMap.from_json({"cds": [row]})}).model_dump_json()
    )
    with pytest.raises(OrganelleInputError, match="differs"):
        assess_cms_annotation(
            annotation,
            conserved_proteins_fasta=DATA / "conserved_controls.fasta",
            edited_cds_json=path,
        )


@pytest.mark.parametrize("contents", ["garbage", "{}"])
def test_malformed_edited_result_fails_at_input_boundary(tmp_path, monkeypatch, contents):
    fake_losat(monkeypatch)
    annotation, _ = source(tmp_path)
    path = tmp_path / "edit.json"
    path.write_text(contents)
    with pytest.raises(OrganelleInputError):
        assess_cms_annotation(
            annotation,
            conserved_proteins_fasta=DATA / "conserved_controls.fasta",
            edited_cds_json=path,
        )


def test_capability_bindings_and_admission(tmp_path, monkeypatch):
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
    root = Path(__file__).parents[2] / "src/organelleverse/capabilities"
    for name, function in [
        ("phenotype.assess_cms_annotation", assess_cms_annotation),
        ("phenotype.predict_cms_topology", predict_cms_topology),
    ]:
        bundle = parse_capability_bundle(
            root / name.replace(".", "-").replace("_", "-") / "capability.toml"
        )
        bound = bind_python_capability(bundle, function, None)
        import inspect

        assert set(bound.signature.parameter_model.model_fields) == set(
            inspect.signature(function).parameters
        ) - ({"annotation_result"} if name.endswith("annotation") else set())
        index = discover_capability_candidates(entry_points=())
        store = VerificationStore(tmp_path / name)
        record = verify_capability(
            name, store=store, environment=LocalVerificationEnvironment(index)
        )
        assert all(e.verdict == "pass" for e in record.equivalence)
        if name.endswith("topology"):
            assert record.equivalence
        assert admit_capabilities(index, store=store).binding_source().resolve(name) is not None


def test_model_fixture_directory_resolution_has_no_output_destination():
    from organelleverse.capabilities.discovery import discover_capability_candidates
    from organelleverse.capabilities.verification import _resolve_fixture_parameters

    entry = next(
        e
        for e in discover_capability_candidates(entry_points=()).entries
        if e.capability_id == "phenotype.predict_cms_topology"
    )
    resolved = _resolve_fixture_parameters(entry, entry.bundle.fixtures[0])
    assert Path(resolved["model_dir"]).is_absolute()
    assert (Path(resolved["model_dir"]) / "config.json").is_file()
    assert "output" not in resolved


@pytest.mark.parametrize("relative", ["../outside", "/tmp", "fixtures/missing"])
def test_model_fixture_directory_rejects_unsafe_or_missing_paths(relative):
    from organelleverse.capabilities.discovery import discover_capability_candidates
    from organelleverse.capabilities.verification import _resolve_fixture_parameters
    from organelleverse.core.errors import OrganelleContractError

    entry = next(
        e
        for e in discover_capability_candidates(entry_points=()).entries
        if e.capability_id == "phenotype.predict_cms_topology"
    )
    fixture = entry.bundle.fixtures[0].model_copy(update={"parameters": {"model_dir": relative}})
    with pytest.raises(OrganelleContractError):
        _resolve_fixture_parameters(entry, fixture)


def test_fixed_upstream_topology_controls_are_independent_of_cms_classification():
    from organelleverse.phenotype.cms.evidence import _proteins

    expected = json.loads((DATA / "tmbed_cms_controls_source.json").read_text())
    rows = load_tmbed_predictions(
        DATA / "tmbed_cms_controls.pred", _proteins(DATA / "cms_queries.fasta")
    )
    assert {name: row["transmembrane_segment_count"] for name, row in rows.items()} == expected[
        "expected_transmembrane_counts"
    ]


def test_predicted_orfs_do_not_hide_annotated_gene_neighbors():
    target = feature("target", 0, 9)
    predicted = feature("predicted", 9, 18).evolve(
        qualifiers=(
            FeatureQualifier(name="inference", values=("ab initio prediction:orfipy:0.0.4",)),
        )
    )
    gene = feature("known", 27, 36, gene="atp6")
    record = AnnotationRecord(
        seqid="r1", name="r1", description="", sequence="A" * 36, features=(target, predicted, gene)
    )
    row = _context(target, record, None, 1)
    assert row["neighbors"][0]["feature_id"] == "predicted"
    assert row["annotated_neighbors"][0]["gene"] == "atp6"
    assert row["candidate_orf_neighbors"][0]["feature_id"] == "predicted"
    assert row["topology"] == "unknown" and not row["wraparound_neighbors_considered"]


def test_annotation_binding_invokes_full_consumer(tmp_path, monkeypatch):
    from organelleverse.capabilities.parser import parse_capability_bundle
    from organelleverse.operations.python_binding import bind_python_capability

    fake_losat(monkeypatch)
    original, _ = source(tmp_path)
    bundle = parse_capability_bundle(
        Path(__file__).parents[2]
        / "src/organelleverse/capabilities/phenotype-assess-cms-annotation/capability.toml"
    )
    bound = bind_python_capability(bundle, assess_cms_annotation, None)
    result = bound.invoke(
        original,
        {"conserved_proteins_fasta": str(DATA / "conserved_controls.fasta"), "neighbor_limit": 1},
    )
    assert result.operation_id == "phenotype.assess_cms_annotation"
    assert result.metrics["candidate_count"] == 2


@pytest.mark.integration
@pytest.mark.slow
def test_live_tmbed_wrapper_matches_captured_upstream_output(tmp_path, monkeypatch):
    import os

    model = os.environ.get("ORG_VERSE_TMBED_MODEL_DIR")
    if not model:
        pytest.skip(
            "Set ORG_VERSE_TMBED_MODEL_DIR to the installed upstream ProtT5 encoder for live TMbed integration"
        )
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    result = predict_cms_topology(DATA / "cms_queries.fasta", model_dir=Path(model), threads=4)
    assert (
        result.artifacts[0].resolve().read_bytes()
        == (DATA / "tmbed_cms_controls.pred").read_bytes()
    )
