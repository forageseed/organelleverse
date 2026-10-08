from pathlib import Path

import pytest

from organelleverse.phylogeny import partition, service


def test_default_uses_fast_relaxed_clustering(tmp_path):
    result = partition.select_partition_scheme("a.fa", "p.nex", output_dir=tmp_path, dry_run=True)
    argv = list(result.metrics["argv"])
    assert "-rclusterf" in argv and "-rcluster" not in argv
    assert argv[argv.index("-rclusterf") + 1] == "100"


def test_public_service_exposes_search_and_comparison_controls(tmp_path, monkeypatch):
    monkeypatch.setattr(service, "managed_run_path", lambda *args: tmp_path)
    result = service.select_partition_scheme(
        "a.fa",
        "p.nex",
        rcluster=100,
        rcluster_fast=False,
        rcluster_max=10000,
        compare_codon_positions=True,
        dry_run=True,
    )
    argv = list(result.metrics["argv"])
    assert argv[argv.index("-rcluster") + 1] == "100"
    assert argv[argv.index("-rcluster-max") + 1] == "10000"
    assert result.metrics["codon_position_comparison_planned"] is True
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    ("selected_bic", "preferred"), [(90, "selected"), (120, "codon_positions"), (None, None)]
)
def test_reports_independently_fitted_three_position_bic(
    tmp_path, monkeypatch, selected_bic, preferred
):
    aln = tmp_path / "a.fa"
    aln.write_text(">A\nATGATG\n>B\nATGATA\n>C\nATGATT\n>D\nATGATC\n")
    parts = tmp_path / "p.nex"
    parts.write_text("#nexus\nbegin sets;\ncharset gene = 1-6;\nend;\n")
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        prefix = Path(argv[argv.index("--prefix") + 1])
        baseline = len(calls) == 2
        if baseline:
            assert "-rclusterf" not in argv and "-rcluster" not in argv
            assert argv[argv.index("-m") + 1] == "MFP"
            assert "-B" not in argv
            assert argv[argv.index("-mrate") + 1] == "E,I,G,I+G"
            sets = partition.parse_nexus_sets(argv[argv.index("-p") + 1])["charsets"]
            assert [partition._charset_sites(spec) for _, spec in sets] == [[1, 4], [2, 5], [3, 6]]
            text = Path(argv[argv.index("-p") + 1]).read_text()
        else:
            text = parts.read_text()
        Path(str(prefix) + ".best_scheme.nex").write_text(text)
        Path(str(prefix) + ".iqtree").write_text(
            f"Bayesian information criterion (BIC) score: {100 if baseline else selected_bic}\n"
        )

    monkeypatch.setattr(partition, "resolve_iqtree", lambda *args: "/test/iqtree")
    monkeypatch.setattr(partition, "run_external", run)
    result = partition.select_partition_scheme(
        aln,
        parts,
        output_dir=tmp_path / "out",
        compare_codon_positions=True,
        bootstrap=1000,
    )
    if selected_bic is None:
        assert result.status == "failed"
        assert result.errors[0].code == "phylogeny.partition.comparison_failed"
        return
    assert result.status == "ok"
    assert len(calls) == 2
    comparison = result.metrics["codon_position_comparison"]
    assert comparison["n_partitions"] == 3
    assert comparison["bic"] == 100
    assert comparison["delta_bic"] == selected_bic - 100
    assert comparison["preferred"] == preferred
    assert result.metrics["bic"] == selected_bic
    assert "BIC" in result.summary_text


@pytest.mark.parametrize(("sequence", "spec"), [("ATGA", "1-4"), ("ATGATG", "1-3")])
def test_codon_comparison_rejects_incomparable_sites(tmp_path, monkeypatch, sequence, spec):
    from organelleverse.core.errors import OrganelleInputError

    aln = tmp_path / "aln.fa"
    aln.write_text(f">A\n{sequence}\n>B\n{sequence}\n")
    parts = tmp_path / "parts.nex"
    parts.write_text(f"#nexus\nbegin sets;\ncharset part = {spec};\nend;\n")
    monkeypatch.setattr(partition, "resolve_iqtree", lambda *args: "/test/iqtree")
    with pytest.raises(OrganelleInputError):
        partition.select_partition_scheme(
            aln,
            parts,
            output_dir=tmp_path / "out",
            compare_codon_positions=True,
        )
    assert not (tmp_path / "out").exists()
