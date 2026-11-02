"""QC results in the HTML report: checks, coverage, gene recovery, reference comparison."""

from __future__ import annotations

import json
from pathlib import Path

import organelleverse as ov
from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.genome import OrganelleGenome, OrganelleMetadata
from organelleverse.core.result import ErrorDetail, OrganelleResult

from .annotation_helpers import annotation_result_fixture
from .test_writer import _qc_result as assembly_qc_result


def _report_html(tmp_path: Path, results: list[OrganelleResult]) -> str:
    fasta = tmp_path / "genome.fa"
    fasta.write_text(">m\nACGTACGT\n")
    genome = OrganelleGenome(
        organelle="mitochondrion",
        sequence=ArtifactRef.from_path(fasta, kind="sequence", format="fasta"),
        metadata=OrganelleMetadata(species="Salvia miltiorrhiza"),
    )
    out = tmp_path / "report.html"
    ov.report.write(ov.report.build(genome, results), output=out)
    return out.read_text(encoding="utf-8")


def _canonical_report(result: OrganelleResult) -> dict:
    artifact = next(a for a in result.artifacts if a.kind.endswith("_qc_report"))
    return json.loads(artifact.resolve().read_text(encoding="utf-8"))


def test_assembly_qc_section_lists_every_check_and_the_decision(monkeypatch, tmp_path):
    qc = assembly_qc_result(monkeypatch, tmp_path)
    report = _canonical_report(qc)

    html = _report_html(tmp_path, [qc])

    assert "class='ov-checks'" in html
    for check in report["checks"]:
        assert f">{check['check_id']}</td>" in html
    decision = report["decision"].replace("_", " ")
    assert "<span class='status" in html and f">{decision}</span>" in html
    # needs-attention checks are listed first, so the table opens on them
    attention = sum(1 for c in report["checks"] if c["status"] in {"fail", "warn"})
    assert f"{attention} need attention" in html
    assert ("<details class='ov-group' open>" in html) == bool(attention)


def test_assembly_qc_section_draws_both_coverage_tracks(monkeypatch, tmp_path):
    qc = assembly_qc_result(monkeypatch, tmp_path)
    windows = _canonical_report(qc)["coverage_windows"]
    assert windows, "the deterministic fixture is expected to report coverage windows"

    html = _report_html(tmp_path, [qc])

    assert "class='ov-cov'" in html
    assert "class='cov-conf'" in html or "class='cov-any'" in html
    assert f"{len({w['sequence_id'] for w in windows})} sequence(s)" in html


def test_annotation_qc_section_shows_gene_recovery_and_missing_genes(monkeypatch, tmp_path):
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    qc = ov.qc.annotation(annotation_result_fixture(tmp_path / "source"))
    assessment = _canonical_report(qc)["gene_profile_assessment"]
    assert assessment["missing_expected_names"], "fixture annotates only three core genes"

    html = _report_html(tmp_path, [qc])

    expected = assessment["expected_profile_gene_count"]
    recovered = assessment["recovered_profile_gene_count"]
    assert f"<b>{recovered}/{expected}</b>" in html
    assert "class='flags miss'" in html
    assert f"<span>{assessment['missing_expected_names'][0]}</span>" in html
    assert "class='ov-checks'" in html


def test_reference_comparison_section_tabulates_raw_and_normalized(tmp_path):
    def comparison(coverage: float, identity: float, inverted: int) -> dict:
        return {
            "reference_coverage": coverage,
            "alignment_identity": identity,
            "inverted_blocks": [{"strand": -1}] * inverted,
        }

    result = OrganelleResult(
        operation_id="qc.compare_assembly_to_reference",
        scope="plastid",
        status="ok",
        summary_text="Compared 1 candidates to ref (plastid).",
        metrics={
            "reference": "ref",
            "comparisons": [
                {
                    "sample": "cand1",
                    "query_length": 154_000,
                    "raw": comparison(0.9950, 0.9987, 2),
                    "normalized": comparison(1.0, 0.9999, 0),
                }
            ],
        },
    )

    html = _report_html(tmp_path, [result])

    assert "ov-compare" in html
    assert ">cand1</td>" in html and ">154,000</td>" in html
    for text in ("99.50%", "99.87%", "100.00%", "99.99%"):
        assert text in html


def test_failed_qc_result_has_no_detail_to_read(tmp_path):
    failed = OrganelleResult(
        operation_id="qc.assembly",
        scope="none",
        status="failed",
        summary_text="QC needs an assembly result.",
        errors=(ErrorDetail(code="qc.input", message="bad input"),),
    )

    html = _report_html(tmp_path, [failed])

    assert "ov-checks" not in html.split("</style>", 1)[1]
    assert "qc.input" in html


def test_dashboard_reads_the_metric_names_of_an_assembly_qc_result(monkeypatch):
    from organelleverse.visualization import suite_plots

    monkeypatch.setattr(suite_plots, "_save", lambda fig, out: fig)
    metrics = {
        "qc_decision": "needs_review",
        "qc_pass_count": 12,
        "qc_warning_count": 5,
        "qc_failure_count": 1,
        "qc_not_assessed_count": 2,
        "any_alignment_coverage_breadth": 0.9987,
        "any_alignment_zero_coverage_bases": 212,
        "confident_coverage_breadth": 0.94,
        "confident_zero_coverage_bases": 9_100,
        "unsupported_junction_count": 1,
        "contradicted_junction_count": 0,
    }

    fig = suite_plots._render_qc_dashboard(metrics, "unused.png")
    coverage, junctions = fig.axes[2], fig.axes[3]

    assert "not assessed" not in [t.get_text() for t in coverage.texts]
    assert [t.get_text() for t in coverage.get_yticklabels()] == [
        "any alignment\nzero: 212 bp",
        "confident primary\nzero: 9 kb",
    ]
    flags = [t.get_text() for t in junctions.texts]
    assert "not provided" in flags  # flags live on the result, not in its metrics

    with_flags = suite_plots._render_qc_dashboard(
        {**metrics, "flags": ["qc.junction_unsupported"]}, "unused.png"
    )
    assert "qc.junction_unsupported" in [t.get_text() for t in with_flags.axes[3].texts]
