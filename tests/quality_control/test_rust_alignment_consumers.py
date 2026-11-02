"""Scientific alignment consumers execute with Rust and no pysam runtime.

On the server, the same small files are also read through an independent
pysam reader to compare the reported scientific metrics.
"""

import sys
from pathlib import Path

import pytest
import rubam

CASES = ("splicing", "exons", "editing_known", "editing_denovo", "variants", "depth")


def consumer(case, directory):
    if case in {"splicing", "exons"}:
        from organelleverse.rna_seq.quantification import (
            quantify_exon_expression,
            quantify_splicing_efficiency,
        )

        from tests.rna_seq.test_quantification import _bam, _write_table

        bam = _bam(directory)
        if case == "splicing":
            table = _write_table(
                directory / "introns.tsv",
                ["intron_id", "gene_id", "seqid", "start", "end", "strand"],
                [
                    {
                        "intron_id": "i",
                        "gene_id": "g",
                        "seqid": "chrM",
                        "start": 42,
                        "end": 51,
                        "strand": "-",
                    }
                ],
            )
            return lambda: quantify_splicing_efficiency(bam, table)
        table = _write_table(
            directory / "exons.tsv",
            ["feature_id", "gene_id", "seqid", "start", "end", "strand"],
            [
                {
                    "feature_id": "e",
                    "gene_id": "g",
                    "seqid": "chrM",
                    "start": 30,
                    "end": 70,
                    "strand": "-",
                }
            ],
        )
        return lambda: quantify_exon_expression(bam, table)
    if case == "editing_known":
        from organelleverse.rna_editing.efficiency import quantify_known_site_efficiency

        from tests.rna_editing.test_efficiency import _write_bam

        bam = directory / "reads.bam"
        _write_bam(bam)
        table = directory / "editing.tsv"
        table.write_text("site_id\tseqid\tposition\tref\tedited\tstrand\ns\tchrM\t11\tC\tT\t+\n")
        return lambda: quantify_known_site_efficiency(bam, table)
    if case == "editing_denovo":
        from organelleverse.rna_editing.detection import detect_editing_sites

        from tests.rna_editing.test_detection import setup_reads

        bam, fasta, _ = setup_reads(directory)
        return lambda: detect_editing_sites(bam, fasta, scope="plastid")
    from organelleverse.heteroplasmy import quantify_variant_fractions
    from organelleverse.heteroplasmy_panel import summarize_depth_windows

    from tests.heteroplasmy.test_quantify import _bam, _sites

    bam = _bam(directory / "reads.bam", ["A"] * 7 + ["G"] * 3)
    table = _sites(directory / "sites.tsv")
    if case == "variants":
        return lambda: quantify_variant_fractions(bam, table, min_depth=5, auto_homology_mask=False)
    return lambda: summarize_depth_windows({"sample": bam}, window_size=10)


@pytest.mark.parametrize("case", CASES)
def test_consumers_execute_without_pysam(case, tmp_path, monkeypatch):
    invoke = consumer(case, tmp_path)
    monkeypatch.setitem(sys.modules, "pysam", None)
    result = invoke()
    assert result.status == "ok"
    assert result.provenance is not None
    assert result.provenance.actual_backend == "rubam"
    assert "rubam" in result.provenance.software_versions
    assert "pysam" not in result.provenance.software_versions


@pytest.mark.parametrize("case", CASES)
def test_metrics_match_independent_reference_reader(case, tmp_path, monkeypatch):
    reference = pytest.importorskip("pysam", reason="independent server reference reader")
    invoke = consumer(case, tmp_path)
    actual = invoke()
    # This is a test-only comparison seam; production imports only rubam.
    monkeypatch.setattr(rubam, "AlignmentFile", reference.AlignmentFile)
    expected = invoke()
    assert actual.status == expected.status == "ok"
    assert actual.metrics == expected.metrics


def test_cram_reference_and_rust_reader_work_without_pysam(tmp_path, monkeypatch):
    from organelleverse.heteroplasmy import quantify_variant_fractions
    from organelleverse.rna_editing.efficiency import quantify_known_site_efficiency
    from organelleverse.rna_seq.quantification import quantify_exon_expression

    data = Path(__file__).parents[1] / "data" / "alignment"
    sites = tmp_path / "sites.tsv"
    sites.write_text("site_id\tseqid\tposition\tref\talt\tedited\tstrand\ns\tc\t10\tC\tT\tT\t+\n")
    exons = tmp_path / "exons.tsv"
    exons.write_text("feature_id\tgene_id\tseqid\tstart\tend\tstrand\ne\tg\tc\t1\t20\t+\n")
    monkeypatch.setitem(sys.modules, "pysam", None)
    kwargs = {"reference_fasta": data / "reference.fa"}
    variant = quantify_variant_fractions(
        data / "reads.cram", sites, min_depth=5, auto_homology_mask=False, **kwargs
    )
    editing = quantify_known_site_efficiency(data / "reads.cram", sites, **kwargs)
    expression = quantify_exon_expression(data / "reads.cram", exons, **kwargs)
    assert variant.status == editing.status == expression.status == "ok"
    assert variant.metrics["sites"][0]["alt_fraction"] == 0.4
    assert editing.metrics["sites"][0]["editing_fraction"] == 0.4
    assert expression.metrics["genes"][0]["read_count"] == 10


def test_published_editing_fixture_verifies_without_pysam(tmp_path, monkeypatch):
    from organelleverse.capabilities.discovery import discover_capabilities
    from organelleverse.capabilities.verification import (
        LocalVerificationEnvironment,
        VerificationStore,
        verify_capability,
    )

    monkeypatch.setitem(sys.modules, "pysam", None)
    monkeypatch.setenv("ORGANELLEVERSE_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("ORGANELLEVERSE_CAPABILITY_PATH", raising=False)
    monkeypatch.chdir(tmp_path)
    record = verify_capability(
        "rna_editing.detect_editing_sites",
        store=VerificationStore(tmp_path / "verifications"),
        environment=LocalVerificationEnvironment(discover_capabilities()),
    )
    assert record.equivalence
    assert all(evidence.verdict == "pass" for evidence in record.equivalence)
