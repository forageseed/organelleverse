"""Scientific regressions: failures, model identities and estimability are preserved."""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from scipy.stats import chi2

from organelleverse.core.errors import OrganelleInputError
from organelleverse.selection import compute_kaks
from organelleverse.selection import codeml, models

kaks_module = importlib.import_module("organelleverse.selection.kaks")


def fasta(tmp_path, records, name="input.fa"):
    path = tmp_path / name
    path.write_text("".join(f">{name}\n{seq}\n" for name, seq in records))
    return path


def paml(tmp_path):
    path = tmp_path / "aligned.paml"
    path.write_text("3 9\nA\nATGGCTGCT\nB\nATGGCTGCC\nC\nATGGCCGCC\n")
    return path


def test_preparation_reuses_full_cds_validation_decisions(tmp_path):
    path = fasta(tmp_path, [("good", "ATGGCTGCTTAA"), ("good2", "ATGGCCGCTTAA"),
                            ("stop", "ATGTAAGCTTAA"), ("invalid", "ATGZZZGCTTAA")])
    validation = codeml.validate_cds(path)
    assert validation.metrics["n_failed"] == 2
    prepared = codeml.write_codeml_inputs(path, tmp_path / "prepared", use_mafft=False)
    assert prepared.status == "ok"
    assert prepared.metrics["n_valid_cds"] == 2
    assert prepared.metrics["n_rejected_cds"] == 2
    filtered = (tmp_path / "prepared/validated_cds.fasta").read_text()
    assert ">stop" not in filtered and ">invalid" not in filtered
    rejected = {row["name"]: row["issues"] for row in prepared.metrics["cds_validation"] if not row["passed"]}
    assert "internal_stop" in rejected["stop"]
    assert "invalid_characters" in rejected["invalid"]


def test_missing_mafft_does_not_generate_a_fake_alignment(tmp_path, monkeypatch):
    from organelleverse import accel
    monkeypatch.setattr(accel, "HAS_RUST", False)
    monkeypatch.setattr(codeml.shutil, "which", lambda name: None)
    path = fasta(tmp_path, [("a", "MACDE"), ("b", "MADE")])
    result = codeml.align_protein(path)
    assert result.status == "failed"
    assert "records" not in result.metrics


def test_mafft_execution_error_does_not_pad_or_change_backend(tmp_path, monkeypatch):
    monkeypatch.setattr(codeml.shutil, "which", lambda name: "mafft")
    def failed(path):
        raise RuntimeError("actual aligner failed")
    monkeypatch.setattr(codeml, "_run_mafft", failed)
    path = fasta(tmp_path, [("a", "MACDE"), ("b", "MADE")])
    with pytest.raises(RuntimeError, match="actual aligner failed"):
        codeml.align_protein(path)


def test_explicit_unaligned_mode_requires_existing_equal_columns(tmp_path):
    path = fasta(tmp_path, [("a", "MACDE"), ("b", "MADE")])
    assert codeml.align_protein(path, method="ungapped").status == "failed"


def test_failed_alignment_never_claims_codeml_ready(tmp_path, monkeypatch):
    monkeypatch.setattr(codeml.shutil, "which", lambda name: None)
    path = fasta(tmp_path, [("a", "ATGGCTGCT"), ("b", "ATGGCTGCC")])
    result = codeml.write_codeml_inputs(path, tmp_path / "prep")
    assert result.status == "failed"
    assert "codeml_ready" not in result.flags
    assert not (tmp_path / "prep/codon_aln.paml").exists()


@pytest.mark.parametrize("second, expected_df", [(None, 1), (["B"], 2)])
def test_clade_comparison_uses_m2a_rel_and_effective_branch_type_df(tmp_path, second, expected_df):
    result = models.write_clade_model(alignment=paml(tmp_path), tree="(A,B,C);",
                                     clade1_labels=["A"], clade2_labels=second,
                                     output_dir=tmp_path / "clade", run=False)
    control = (tmp_path / "clade/clade_null.ctl").read_text()
    assert "NSsites = 22" in control
    assert result.metrics["df"] == expected_df
    assert result.metrics["null_model"] == "M2a_rel"


@pytest.mark.parametrize("tree", ["((A,B)$1,C);", "((A,B)#1,C);"])
def test_clade_df_counts_actual_clade_or_branch_annotations(tmp_path, tree):
    result = models.write_clade_model(alignment=paml(tmp_path), tree=tree,
                                     output_dir=tmp_path / "clade", run=False)
    assert result.metrics["df"] == 1


def codeml_block(model, likelihood, beb=()):
    table = "".join(f"   {site} A {prob}\n" for site, prob in beb)
    return (f"NSsites Model {model}: example\n"
            f"lnL(ntime: 5  np: 10): {likelihood} +0.000000\n"
            "kappa (ts/tv) = 2.5\n"
            "p: 0.7 0.2 0.1\nw: 0.1 1.0 2.5\n"
            + ("Bayes Empirical Bayes (BEB) analysis\nPositive sites:\n" + table + "\n" if beb else ""))


def test_parser_keeps_likelihoods_and_beb_separate_for_every_model(tmp_path):
    path = tmp_path / "mlc"
    path.write_text(codeml_block(1, -100) + codeml_block(2, -90, [(1, .6), (2, .97)])
                    + codeml_block(8, -89, [(3, .99)]))
    result = models.parse_codeml_output(path)
    assert "lnL" not in result
    per_model = result["model_results"]
    assert [per_model[str(i)]["lnL"] for i in (1, 2, 8)] == [-100, -90, -89]
    assert per_model["2"]["kappa"] == 2.5
    assert per_model["2"]["site_classes"]["2"]["omega"] == 2.5
    assert [site["site"] for site in per_model["2"]["beb_sites"]] == [1, 2]
    assert [site["site"] for site in per_model["8"]["beb_sites"]] == [3]


def test_site_claims_require_correct_model_lrt_and_high_beb(tmp_path, monkeypatch):
    path = tmp_path / "mlc"
    path.write_text(codeml_block(1, -100) + codeml_block(2, -90, [(1, .6), (2, .97)])
                    + codeml_block(7, -100) + codeml_block(8, -99, [(3, .99)]))
    parsed = models.parse_codeml_output(path)
    monkeypatch.setattr(models, "run_codeml", lambda *a, **k: parsed)
    result = models.write_site_model(alignment=paml(tmp_path), tree="(A,B,C);",
                                    output_dir=tmp_path / "site", models="1 2 7 8")
    assert result.metrics["n_positive_sites"] == 1
    assert [site["site"] for site in result.metrics["positive_sites"]] == [2]
    tests = result.metrics["likelihood_ratio_tests"]
    assert tests["M1_vs_M2"]["significant"]
    assert not tests["M7_vs_M8"]["significant"]


def test_single_alternative_beb_without_null_is_not_evidence_of_selection(tmp_path, monkeypatch):
    monkeypatch.setattr(models, "run_codeml", lambda *a, **k: {"lnL": -90,
                          "beb_sites": [{"site": 1, "prob": .99, "significant": True}]})
    result = models.write_site_model(alignment=paml(tmp_path), tree="(A,B,C);",
                                    output_dir=tmp_path / "site", models="8")
    assert result.metrics["n_positive_sites"] == 0
    assert "positive_selection_sites" not in result.flags


def test_real_branch_omega_and_kappa_fields_are_used_without_none_comparison(tmp_path, monkeypatch):
    path = tmp_path / "mlc"
    path.write_text("lnL(ntime: 5  np: 10): -90 +0.0\nkappa (ts/tv) = 2.5\n"
                    "w (dN/dS) for branches: 0.2 2.5\n")
    parsed = models.parse_codeml_output(path)
    assert parsed["omega_foreground"] == 2.5
    assert parsed["omega_background"] == .2
    assert parsed["n_params"] == 10
    replies = iter([{"lnL": -100}, parsed])
    monkeypatch.setattr(models, "run_codeml", lambda *a, **k: next(replies))
    result = models.write_branch_model(alignment=paml(tmp_path), tree="(A,B,C);",
                                      foreground_labels=["A"], output_dir=tmp_path / "branch")
    assert result.metrics["omega_foreground"] == 2.5


def test_branchsite_lrt_uses_the_boundary_mixture_distribution(tmp_path, monkeypatch):
    replies = iter([{"lnL": -100}, {"lnL": -98.5, "beb_sites": [{"site": 1, "prob": .99}]}])
    monkeypatch.setattr(models, "run_codeml", lambda *a, **k: next(replies))
    result = models.write_branch_site_model(alignment=paml(tmp_path), tree="(A,B,C);",
                                           foreground_labels=["A"], output_dir=tmp_path / "bs")
    assert result.metrics["p_value"] == pytest.approx(.5 * chi2.sf(3, 1), abs=1e-6)
    assert result.metrics["significant"]
    assert result.metrics["n_positive_sites"] == 1


@pytest.mark.parametrize("writer, labels", [(models.write_branch_model, {"foreground_labels": ["A"]}),
    (models.write_branch_site_model, {"foreground_labels": ["A"]}),
    (models.write_clade_model, {"clade1_labels": ["A"]})])
def test_absent_codeml_results_are_failures_not_successful_plans(tmp_path, monkeypatch, writer, labels):
    monkeypatch.setattr(models, "run_codeml", lambda *a, **k: None)
    result = writer(alignment=paml(tmp_path), tree="(A,B,C);", output_dir=tmp_path / "run", **labels)
    assert result.status == "failed"


def test_saturated_jc_pairs_are_not_zero_distances_or_ordinary_mean_members(tmp_path):
    path = fasta(tmp_path, [("a", "GCTGCTGCT"), ("b", "GCCGCCGCC"), ("c", "GCTGCTGCC")])
    result = kaks_module.kaks(path, jukes_cantor=True)
    assert result.metrics["pairs"] == 3
    assert result.metrics["saturated_pairs"] == 1
    estimable = [row for row in result.metrics["pair_results"] if row["Ks"] is not None]
    saturated = [row for row in result.metrics["pair_results"] if row["Ks"] is None]
    assert saturated[0]["Ks_state"] == "saturated"
    assert saturated[0]["Ka_Ks_state"] == "saturated"
    assert result.metrics["Ks"] == pytest.approx(sum(row["Ks"] for row in estimable)/2, abs=1e-6)


def test_all_jc_saturated_has_no_numeric_mean_or_selection_flag(tmp_path):
    path = fasta(tmp_path, [("a", "GCTGCTGCT"), ("b", "GCCGCCGCC")])
    result = kaks_module.kaks(path, jukes_cantor=True)
    assert result.metrics["Ks"] is None and result.metrics["Ka"] is None
    assert result.metrics["Ka_Ks_state"] == "saturated"
    assert "positive_selection" not in result.flags


@pytest.mark.parametrize("other", ["ATGGCTGCT", "ATGGATGAT"])
def test_compute_kaks_uses_same_zero_denominator_states_as_result_api(tmp_path, other):
    path = fasta(tmp_path, [("a", "ATGGCTGCT"), ("b", other)])
    native, result = compute_kaks(path), kaks_module.kaks(path)
    for key in ("Ka", "Ks", "Ka_Ks", "Ka_Ks_state", "pairs"):
        assert native[key] == result.metrics[key]
    assert native["Ka_Ks"] is None


def test_codon_length_mismatch_is_not_silently_truncated(tmp_path):
    path = fasta(tmp_path, [("a", "ATGGCTGCT"), ("b", "ATGGCTGCTGCT")])
    with pytest.raises(OrganelleInputError, match="equal-length"):
        kaks_module.kaks(path)

@pytest.mark.parametrize("footer, returncode, accepted", [(False, 0, False), (True, 1, True), (True, -11, False)])
def test_runner_requires_completed_report_and_records_exit_status(tmp_path, monkeypatch, footer, returncode, accepted):
    from types import SimpleNamespace
    control = tmp_path / "run.ctl"
    control.write_text("outfile = mlc\n")
    monkeypatch.setattr(models.shutil, "which", lambda name: "/codeml")
    calls = []
    def execute(*args, **kwargs):
        calls.append(kwargs)
        (tmp_path / "mlc").write_text("lnL(ntime: 5 np: 7): -100.0\n"
                                      + ("Time used: 00:01\n" if footer else ""))
        return SimpleNamespace(returncode=returncode)
    monkeypatch.setattr(models.subprocess, "run", execute)
    result = models.run_codeml(control, tmp_path, timeout=7)
    assert len(calls) == 1 and calls[0]["timeout"] == 7
    if accepted:
        assert result["execution"]["returncode"] == 1
        assert result["execution"]["complete"]
    else:
        assert result is None


def test_negative_nested_model_lrt_is_an_optimization_error():
    with pytest.raises(OrganelleInputError, match="optimization"):
        models.likelihood_ratio_test(-100, -101, 1)


def test_prepared_tree_declares_one_tree_for_real_paml(tmp_path):
    output = tmp_path / "tree.nwk"
    returned = models._prepare_tree("(A,B,C);", ["A"], output)
    assert output.read_text().splitlines()[0] == "3 1"
    assert returned == "(A #1,B,C);"


def test_protein_aligner_forces_protein_alphabet_for_nucleotide_like_peptides(tmp_path, monkeypatch):
    from types import SimpleNamespace
    calls = []
    def execute(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(stdout=">a\nMAAAAA\n>b\nMAAAAA\n")
    monkeypatch.setattr(codeml, "run_external", execute)
    path = fasta(tmp_path, [("a", "MAAAAA"), ("b", "MAAAAA")])
    assert codeml._run_mafft(path) == [("a", "MAAAAA"), ("b", "MAAAAA")]
    assert "--amino" in calls[0]


def test_significant_branch_heterogeneity_is_not_a_positive_selection_test(tmp_path, monkeypatch):
    replies = iter([{"lnL": -100}, {"lnL": -95, "omega_foreground": 2.0, "omega_background": .2}])
    monkeypatch.setattr(models, "run_codeml", lambda *a, **k: next(replies))
    result = models.write_branch_model(alignment=paml(tmp_path), tree="(A,B,C);",
                                      foreground_labels=["A"], output_dir=tmp_path / "branch")
    assert result.metrics["significant"]
    assert result.metrics["foreground_omega_above_one"]
    assert result.metrics["test"] == "M0_vs_two_ratio_heterogeneity"
    assert "branch_rate_heterogeneity" in result.flags
    assert "positive_selection_branch" not in result.flags
    assert "branch-site" in result.metrics["inference_note"]


def test_real_clade_branch_type_table_is_preserved_by_site_class(tmp_path):
    output = tmp_path / "mlc"
    output.write_text("lnL(ntime: 11 np: 17): -897.036698\n"
        "site class             0         1         2\n"
        "proportion        0.91005   0.00000   0.08995\n"
        "branch type 0:    0.48622   1.00000   8.10512\n"
        "branch type 1:    0.48622   1.00000   0.00000\n")
    parsed = models.parse_codeml_output(output)
    assert parsed["site_classes"]["2"] == {"proportion": .08995,
            "omega_by_branch_type": {"0": 8.10512, "1": 0.0}}


def test_real_branchsite_class_table_keeps_foreground_and_background(tmp_path):
    output = tmp_path / "mlc"
    output.write_text("lnL(ntime: 11 np: 17): -897.036698\n"
        "site class             0        1       2a       2b\n"
        "proportion        0.70000  0.10000  0.17500  0.02500\n"
        "background w      0.20000  1.00000  0.20000  1.00000\n"
        "foreground w      0.20000  1.00000  2.50000  2.50000\n")
    parsed = models.parse_codeml_output(output)
    assert parsed["site_classes"]["2a"] == {"proportion": .175,
            "omega_foreground": 2.5, "omega_background": .2}
