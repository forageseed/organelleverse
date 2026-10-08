"""HyPhy selection analyses (BUSTED/aBSREL/RELAX/MEME/FEL).

Parser fixtures are trimmed real HyPhy 2.5.64 outputs: rbcL codon alignment
of 18 Poales plastomes (package annotation -> MAFFT -> pal2nal, IQ-TREE tree)
with the 8 C4 grass tips as foreground.
"""

from __future__ import annotations

import importlib.metadata
import os
from pathlib import Path

import pytest

from organelleverse.core.errors import OrganelleDependencyError, OrganelleInputError
from organelleverse.core.result import OrganelleResult
from organelleverse.selection import hyphy
from organelleverse.selection.hyphy import (
    check_hyphy,
    hyphy_busted,
    hyphy_fel,
    hyphy_relax,
    label_tree_for_hyphy,
    parse_hyphy_json,
)

FIXTURES = Path(__file__).parent / "fixtures" / "hyphy"
_HYPHY_IDS = (
    "selection.hyphy_busted",
    "selection.hyphy_absrel",
    "selection.hyphy_relax",
    "selection.hyphy_meme",
    "selection.hyphy_fel",
    "selection.check_hyphy",
    "selection.parse_hyphy_json",
)


# ---------------------------------------------------------------- tree labels


def test_codeml_marks_and_hyphy_tags_are_converted() -> None:
    tree, summary = label_tree_for_hyphy("(a:1,(b #1:1,c:1)$1:2,(d,e{Foreground}):0.5);")
    assert tree == (
        "(a:1,(b{Foreground}:1,c{Foreground}:1){Foreground}:2,(d,e{Foreground}):0.5);"
    )
    assert summary["foreground_tips"] == ["b", "c", "e"]
    assert summary["n_foreground_branches"] == 4
    assert summary["n_foreground_internal_branches"] == 1


def test_foreground_labels_match_whole_tip_names_only() -> None:
    # codeml's own _prepare_tree does substring replacement; here "A" must not tag "A2".
    tree, summary = label_tree_for_hyphy("((A:1,A2:1):1,B:1,C:1);", ["A"])
    assert tree == "((A{Foreground}:1,A2:1):1,B:1,C:1);"
    assert summary["foreground_tips"] == ["A"]


def test_mark_clade_tags_stem_and_every_branch_inside() -> None:
    tree, summary = label_tree_for_hyphy("(((a:1,b:1)90:1,c:1):1,d:1,e:1);", ["a", "b"], mark_clade=True)
    # support value 90 is dropped; stem of (a,b) is tagged
    assert tree == "(((a{Foreground}:1,b{Foreground}:1){Foreground}:1,c:1):1,d:1,e:1);"
    assert summary["n_foreground_internal_branches"] == 1


def test_mark_clade_spanning_the_root_is_refused() -> None:
    with pytest.raises(OrganelleInputError) as info:
        label_tree_for_hyphy("(a:1,(b:1,c:1):1,d:1);", ["a", "b"], mark_clade=True)
    assert info.value.code == "selection.hyphy_foreground_clade_is_root"


def test_unknown_foreground_label_is_an_input_error() -> None:
    with pytest.raises(OrganelleInputError) as info:
        label_tree_for_hyphy("(a,b,c);", ["z"])
    assert info.value.code == "selection.hyphy_foreground_not_in_tree"


# ---------------------------------------------------------------- alignments


def test_paml_sequential_alignment_is_read(tmp_path: Path) -> None:
    path = tmp_path / "aln.paml"
    path.write_text("3  6\nsa\nATGAAA\nsb\nATGAAG\nsc\nATG\nAAA\n")
    records = hyphy._read_alignment(path)
    assert records == [("sa", "ATGAAA"), ("sb", "ATGAAG"), ("sc", "ATGAAA")]


def test_terminal_stop_column_is_removed_but_internal_stop_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "aln.fasta"
    records = [("a", "ATGAAATAA"), ("b", "ATGAAGTAG"), ("c", "ATGAAA---")]
    cleaned, notes = hyphy._validate_codon_alignment(records, path)
    assert [seq for _, seq in cleaned] == ["ATGAAA", "ATGAAG", "ATGAAA"]
    assert notes == ["terminal_stop_codon_removed"]
    with pytest.raises(OrganelleInputError) as info:
        hyphy._validate_codon_alignment(
            [("a", "ATGTGAAAA"), ("b", "ATGAAAAAA"), ("c", "ATGAAAAAA")], path
        )
    assert info.value.code == "selection.hyphy_internal_stop_codon"


def test_out_of_frame_alignment_is_refused(tmp_path: Path) -> None:
    with pytest.raises(OrganelleInputError) as info:
        hyphy._validate_codon_alignment([("a", "ATGA"), ("b", "ATGA"), ("c", "ATGA")], tmp_path)
    assert info.value.code == "selection.hyphy_not_in_frame"


# ---------------------------------------------------------------- JSON parsing


def test_busted_json_key_results() -> None:
    parsed = parse_hyphy_json(FIXTURES / "busted.json")
    assert parsed["method"] == "busted"
    assert parsed["p_value"] == pytest.approx(0.2083141473151857)
    assert parsed["lrt"] == pytest.approx(1.751121668818087)
    assert parsed["significant"] is False
    assert parsed["n_tested_branches"] == 8
    assert parsed["omega_max"] == pytest.approx(26.52481527671854)
    assert len(parsed["omega_distribution_test"]) == 3


def test_absrel_json_lists_tested_branches_with_corrected_p() -> None:
    parsed = parse_hyphy_json(FIXTURES / "absrel.json")
    assert parsed["method"] == "absrel"
    assert parsed["n_tested"] == 8
    assert len(parsed["branches"]) == 8
    assert parsed["n_selected_branches"] == 0
    assert all(0 <= row["p_corrected"] <= 1 for row in parsed["branches"])


def test_relax_json_k_and_direction() -> None:
    parsed = parse_hyphy_json(FIXTURES / "relax.json")
    assert parsed["method"] == "relax"
    assert parsed["k"] == pytest.approx(0.6412883658487986)
    assert parsed["p_value"] == pytest.approx(0.01869915542853029)
    assert parsed["direction"] == "relaxed"
    assert parsed["significant"] is True
    assert "convergence-unstable-alernative" in parsed["convergence_warnings"]


def test_meme_and_fel_site_lists() -> None:
    meme = parse_hyphy_json(FIXTURES / "meme.json", alpha=0.05)
    assert [site["site"] for site in meme["sites"]] == [221, 279]
    assert meme["n_codons_tested"] == 480
    fel = parse_hyphy_json(FIXTURES / "fel.json", alpha=0.05)
    assert [site["site"] for site in fel["positive_sites"]] == [101, 142, 279]
    assert fel["n_negative_sites"] == 10


def test_unrecognised_json_is_an_input_error(tmp_path: Path) -> None:
    path = tmp_path / "x.json"
    path.write_text('{"foo": 1}')
    with pytest.raises(OrganelleInputError):
        parse_hyphy_json(path)


# ---------------------------------------------------------------- dependency


def test_missing_hyphy_raises_dependency_error_with_install_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ORGANELLEVERSE_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("ORGANELLEVERSE_HYPHY", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    import organelleverse.assembly.install as install

    monkeypatch.setattr(install, "_list_conda_envs", lambda: [])
    assert check_hyphy()["installed"] is False
    aln = tmp_path / "aln.fasta"
    aln.write_text(">a\nATGAAA\n>b\nATGAAG\n>c\nATGAAA\n")
    with pytest.raises(OrganelleDependencyError) as info:
        hyphy_busted(alignment=aln, tree="(a,b,c);")
    assert info.value.code == "selection.hyphy_not_found"
    assert "bioconda" in info.value.message


# ---------------------------------------------------------------- real HyPhy


def _real_hyphy() -> str | None:
    info = check_hyphy()
    return info["path"] if info.get("installed") and info.get("version_ok") else None


_TINY = {
    # first 90 codons of the real rbcL codon alignment used for validation
    "Zea": (
        "ATGTCACCACAAACAGAAACTAAAGCAAGTGTTGGATTTAAAGCTGGTGTTAAGGATTATAAATTGACTTACTACACC"
        "CCGGAGTACGAAACCAAGGATACTGATATCTTGGCAGCATTCCGAGTAACTCCTCAGCTCGGGGTTCCGCCTGAAGAA"
        "GCAGGAGCTGCAGTAGCTGCGGAATCTTCTACTGGTACATGGACAACTGTTTGGACTGATGGACTTACCAGTCTTGAT"
        "CGTTACAAAGGACGATGCTATCACATCGAGCCCGTT"
    ),
    "Sorghum": (
        "ATGTCACCACAAACAGAAACTAAAGCAAGTGTTGGATTTAAAGCTGGTGTTAAGGATTATAAATTGACTTACTACACC"
        "CCGGAGTACGAAACCAAGGATACTGATATCTTGGCAGCATTCCGAGTAACTCCTCAGCTCGGGGTTCCGCCTGAAGAA"
        "GCAGGAGCTGCAGTAGCTGCGGAATCTTCTACTGGTACATGGACAACTGTTTGGACTGATGGACTTACCAGTCTTGAT"
        "CGTTACAAAGGACGATGCTATCACATCGAGCCCGTT"
    ),
    "Oryza": (
        "ATGTCACCACAAACAGAAACTAAAGCAAGTGTTGGATTTAAAGCTGGTGTTAAGGATTATAAATTGACTTACTACACC"
        "CCGGAGTACGAAACCAAGGACACTGATATCTTGGCAGCATTCCGAGTAACTCCTCAGCCGGGGGTTCCGCCCGAAGAA"
        "GCAGGGGCTGCAGTAGCTGCCGAATCTTCTACTGGTACATGGACAACTGTTTGGACTGATGGACTTACCAGTCTTGAT"
        "CGTTACAAAGGCCGATGCTATCACATCGAGCCCGTT"
    ),
    "Avena": (
        "ATGTCACCACAAACAGAAACTAAAGCAAGTGTTGGATTTCAAGCTGGTGTTAAAGATTATAAATTGACTTACTACACC"
        "CCGGAGTATGAAACCAAGGATACTGATATCTTGGCAGCATTCCGAGTAACTCCTCAACCTGGGGTTCCGCCGGAAGAA"
        "GCAGGGGCTGCAGTAGCTGCCGAATCTTCTACTGGTACATGGACAACTGTTTGGACTGATGGACTTACCAGTCTTGAT"
        "CGTTACAAAGGACGATGCTATCACATCGAGCCTGTT"
    ),
    "Hordeum": (
        "ATGTCACCACAAACAGAAACTAAAGCAGGTGTTGGATTTCAAGCTGGTGTTAAAGATTATAAATTGACTTACTACACC"
        "CCAGAGTATGAAACTAAGGATACTGATATCTTGGCAGCATTCCGAGTAAGTCCTCAGCCTGGGGTTCCGCCCGAAGAA"
        "GCAGGGGCTGCAGTAGCTGCCGAATCTTCTACTGGTACATGGACAACTGTTTGGACTGATGGACTTACCAGTCTTGAT"
        "CGTTACAAAGGACGATGCTATCACATCGAGCCTGTT"
    ),
}


@pytest.mark.slow  # ~70 s: three real HyPhy fits
@pytest.mark.skipif(_real_hyphy() is None, reason="HyPhy >= 2.5 not installed")
def test_real_hyphy_fel_busted_and_relax_on_a_tiny_alignment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ORGANELLEVERSE_HOME", str(tmp_path / "home"))
    aln = tmp_path / "tiny.fasta"
    aln.write_text("".join(f">{k}\n{v}\n" for k, v in _TINY.items()))
    tree = tmp_path / "tiny.nwk"
    tree.write_text("((Zea:0.01,Sorghum:0.01):0.02,Oryza:0.03,(Avena:0.02,Hordeum:0.02):0.01);\n")

    fel = hyphy_fel(alignment=aln, tree=tree)
    assert isinstance(fel, OrganelleResult) and fel.status == "ok"
    assert fel.operation_id == "selection.hyphy_fel"
    assert fel.metrics["n_codons_tested"] == 90
    assert fel.metrics["branch_set"] == "All"
    raw = Path(fel.metrics["raw_json"])
    assert raw.is_file() and any(a.uri.endswith("fel.json") for a in fel.artifacts)
    assert fel.provenance.software_versions["hyphy"] == fel.metrics["hyphy_version"]

    busted = hyphy_busted(
        alignment=aln, tree=tree, foreground_labels=["Zea", "Sorghum"], mark_clade=True
    )
    assert busted.status == "ok"
    assert busted.metrics["n_tested_branches"] == 3  # Zea, Sorghum and their stem
    assert 0.0 <= busted.metrics["p_value"] <= 1.0

    relax = hyphy_relax(
        alignment=aln, tree=tree, foreground_labels=["Zea", "Sorghum"]
    )
    assert relax.status == "ok"
    assert relax.metrics["k"] is not None
    assert relax.metrics["direction"] in {"relaxed", "intensified", "unchanged"}


def test_relax_without_foreground_is_refused_before_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ORGANELLEVERSE_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(
        hyphy, "_require_hyphy", lambda method, path: {"path": "hyphy", "version": "2.5.64"}
    )
    aln = tmp_path / "aln.fasta"
    aln.write_text(">a\nATGAAA\n>b\nATGAAG\n>c\nATGAAA\n")
    with pytest.raises(OrganelleInputError) as info:
        hyphy_relax(alignment=aln, tree="(a,b,c);")
    assert info.value.code == "selection.hyphy_foreground_required"


# ---------------------------------------------------------------- capabilities


def test_hyphy_capabilities_are_discovered_admitted_and_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from organelleverse.capabilities.discovery import discover_capabilities
    from organelleverse.capabilities.index import CapabilityStatus
    from organelleverse.capabilities.verification import (
        LocalVerificationEnvironment,
        VerificationStore,
        verify_capability,
    )

    home = tmp_path / "home"
    project = tmp_path / "project"
    (project / ".organelleverse" / "capabilities").mkdir(parents=True)
    home.mkdir()
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda **kwargs: ())
    monkeypatch.setenv("ORGANELLEVERSE_HOME", str(home))
    monkeypatch.delenv("ORGANELLEVERSE_CAPABILITY_PATH", raising=False)
    monkeypatch.chdir(project)

    discovered = discover_capabilities()
    store = VerificationStore(home / "verifications")
    for capability_id in _HYPHY_IDS:
        assert discovered.describe(capability_id).origins[0].channel == "core"
        verify_capability(
            capability_id, store=store, environment=LocalVerificationEnvironment(discovered)
        )
    admitted = discover_capabilities()
    for capability_id in _HYPHY_IDS:
        assert admitted.describe(capability_id).status is CapabilityStatus.ADMITTED

    binding = admitted.binding_source().resolve("selection.parse_hyphy_json")
    assert binding is not None
    result = binding.invoke(None, {"json_path": str(FIXTURES / "relax.json")})
    assert result.status == "ok"
    assert result.metrics["hyphy_result"]["direction"] == "relaxed"
    assert os.environ["ORGANELLEVERSE_HOME"] == str(home)
