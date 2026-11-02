"""Discriminative regressions for the 2026-10-25 evolutionary-method audit."""
from pathlib import Path
from subprocess import CompletedProcess

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqFeature import BeforePosition, SeqFeature, SimpleLocation
from Bio.SeqRecord import SeqRecord
import pytest

from organelleverse.core.errors import OrganelleExecutionError
from organelleverse.phylogeny import phylo
from organelleverse.phylogeny._results import failed_result
from organelleverse.phylogeny.partition import _codon_align, _codon_clean, _read_cds


def _genbank(path, sequence, *, translation=None, start=1, table=11, partial=False):
    record = SeqRecord(Seq(sequence), id=path.stem, name=path.stem, description="fixture")
    record.annotations["molecule_type"] = "DNA"
    qualifiers = {"gene": ["g"], "codon_start": [str(start)], "transl_table": [str(table)]}
    if translation is not None:
        qualifiers["translation"] = [translation]
    record.features = [SeqFeature(SimpleLocation(BeforePosition(0) if partial else 0, len(sequence)), type="CDS", qualifiers=qualifiers)]
    SeqIO.write(record, path, "genbank")
    return path


def test_shared_cds_missing_translation_is_still_protein(tmp_path):
    a = _genbank(tmp_path / "a.gb", "ATGGCTTAA", translation="MA")
    b = _genbank(tmp_path / "b.gb", "AATGGCTTAA", start=2)
    result = phylo.extract_shared_genes([a, b])
    assert result.status == "ok"
    assert [sequence for _, sequence in result.metrics["shared_records"]["g"]] == ["MA", "MA"]


def test_shared_cds_honors_translation_table_and_alternative_start(tmp_path):
    a = _genbank(tmp_path / "a.gb", "ATGTGA", table=4)
    b = _genbank(tmp_path / "b.gb", "GTGTGGTAA", table=11)
    result = phylo.extract_shared_genes([a, b])
    assert [sequence for _, sequence in result.metrics["shared_records"]["g"]] == ["MW", "MW"]


def test_partial_cds_is_not_given_an_invented_start_methionine(tmp_path):
    a = _genbank(tmp_path / "a.gb", "GTGTGGTAA", table=11, partial=True)
    b = _genbank(tmp_path / "b.gb", "GTGTGGTAA", table=11, partial=True)
    result = phylo.extract_shared_genes([a, b])
    assert [sequence for _, sequence in result.metrics["shared_records"]["g"]] == ["VW", "VW"]


def test_partition_cds_applies_frame_before_truncation(tmp_path):
    path = _genbank(tmp_path / "partial.gb", "AATGGCTTAA", start=2)
    _, genes = _read_cds(path, 11)
    assert genes["g"] == "ATGGCT"


@pytest.mark.parametrize(("sequence", "table", "expected"), [("ATGTGA", 4, "ATGTGA"), ("ATGTGA", 11, "ATG"), ("ATGAGA", 2, "ATG")])
def test_stop_removal_uses_requested_genetic_code(sequence, table, expected):
    assert _codon_clean(sequence, table) == expected


def test_partition_reports_alignment_failure_without_keyerror(monkeypatch):
    monkeypatch.setattr(phylo, "align", lambda *args, **kwargs: failed_result("align", summary_text="MAFFT unavailable", code="phylogeny.align.backend_missing"))
    with pytest.raises(OrganelleExecutionError, match="MAFFT unavailable"):
        _codon_align([("a", "ATGGCT"), ("b", "ATGGCT")], 11, "auto")


def _alignment(tmp_path):
    path = tmp_path / "alignment.fa"
    path.write_text(">a\nACGT\n>b\nACGA\n>c\nACGG\n>d\nACGC\n")
    return path


def _run(tmp_path, monkeypatch, executor, *, method="iqtree"):
    monkeypatch.setattr(phylo, "_which", lambda _: "audit-injected-backend")
    return phylo.run_build_tree(_alignment(tmp_path), method=method, output_dir=tmp_path / "result", executor=executor)


def test_raxml_plan_uses_bootstrap_count_switch():
    argv = phylo._build_raxml_argv("raxml-ng", "alignment.fa", "tree", bootstrap=250)
    assert "--all" in argv
    assert "--bootstrap" not in argv
    assert argv[argv.index("--bs-trees") + 1] == "250"


@pytest.mark.parametrize("returncode", [23, -9])
def test_nonzero_tree_builder_exit_is_failure(tmp_path, monkeypatch, returncode):
    result = _run(tmp_path, monkeypatch, lambda argv: CompletedProcess(argv, returncode))
    assert result.status == "failed"
    assert "tree_built" not in result.flags


def test_zero_exit_without_tree_is_failure(tmp_path, monkeypatch):
    result = _run(tmp_path, monkeypatch, lambda argv: CompletedProcess(argv, 0))
    assert result.status == "failed"
    assert not result.artifacts


@pytest.mark.parametrize("newick", ["not a tree", "(a,b,c,c);", "(a,b,c,x);", "(a,b,c);", "(a:-1,b:1,c:1,d:1);"])
def test_invalid_tree_or_leaf_set_is_failure(tmp_path, monkeypatch, newick):
    def execute(argv):
        Path(argv[argv.index("-pre") + 1] + ".treefile").write_text(newick)
        return CompletedProcess(argv, 0)

    result = _run(tmp_path, monkeypatch, execute)
    assert result.status == "failed"
    assert not result.artifacts


@pytest.mark.parametrize(("method", "prefix_flag", "suffix"), [("iqtree", "-pre", ".treefile"), ("iqtree3", "--prefix", ".treefile"), ("raxml", "--prefix", ".raxml.support")])
def test_valid_tree_is_published_with_backend_output_path(tmp_path, monkeypatch, method, prefix_flag, suffix):
    def execute(argv):
        Path(argv[argv.index(prefix_flag) + 1] + suffix).write_text("((a:1,b:1)90:1,c:1,d:1);")
        return CompletedProcess(argv, 0)

    result = _run(tmp_path, monkeypatch, execute, method=method)
    assert result.status == "ok"
    assert "tree_built" in result.flags
    assert len(result.artifacts) == 1
    assert result.metrics["output_tree"].endswith(suffix)


def test_fasttree_captures_stdout_without_shell_tokens(tmp_path, monkeypatch):
    def execute(argv):
        assert "<" not in argv and ">" not in argv
        assert argv[-1].endswith("alignment.fa")
        return CompletedProcess(argv, 0, stdout="((a:1,b:1)0.9:1,c:1,d:1);")

    result = _run(tmp_path, monkeypatch, execute, method="fasttree")
    assert result.status == "ok"
    assert len(result.artifacts) == 1


def test_existing_tree_is_not_mistaken_for_new_execution(tmp_path, monkeypatch):
    output = tmp_path / "result"
    output.mkdir()
    tree = output / "tree.treefile"
    original = "((a,b),c,d);"
    tree.write_text(original)
    calls = []
    result = _run(tmp_path, monkeypatch, lambda argv: calls.append(argv))
    assert result.status == "failed"
    assert calls == []
    assert tree.read_text() == original
