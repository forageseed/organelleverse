"""Localization heuristics: signal-peptide numbering, NLS consensus, and no forced Cytoplasm call."""

from __future__ import annotations

import random
from pathlib import Path

from organelleverse.localization.localize import _heuristic_predictions, _parse_deeploc_output
from organelleverse.localization.localize_core import score_nls, score_signal_peptide

# human preproinsulin: 24-residue signal peptide, MALWMRLLPLLALLALWGPDPAAA | FVNQ...
_PREPROINSULIN = (
    "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN"
)


def test_signal_peptide_cleavage_of_preproinsulin() -> None:
    sp = score_signal_peptide(_PREPROINSULIN)
    assert sp["has_sp"] and sp["cleavage_pos"] == 24
    assert sp["mature_seq"].startswith("FVNQ")


def test_classical_nls_is_found_and_random_sequences_mostly_are_not() -> None:
    assert score_nls("MAPKKKRKVEDPKDFPSELLSA")["has_nls"]  # SV40 large T
    rng = random.Random(3)
    seqs = ["M" + "".join(rng.choice("ACDEFGHIKLMNPQRSTVWY") for _ in range(399)) for _ in range(1000)]
    rate = sum(score_nls(s)["has_nls"] for s in seqs) / len(seqs)
    assert rate < 0.3  # the old pattern matched ~39% of these


def test_no_confident_call_is_unassigned_not_cytoplasm() -> None:
    summaries, _ = _heuristic_predictions([("plain", "M" + "AG" * 60)])
    assert summaries[0]["predicted"] == "Unassigned"


def test_deeploc_soluble_does_not_become_cytoplasm(tmp_path: Path) -> None:
    header = (
        "Protein_ID,Localizations,Signals,Membrane types,Cytoplasm,Nucleus,Extracellular,Cell membrane,"
        "Mitochondrion,Plastid,Endoplasmic reticulum,Lysosome/Vacuole,Golgi apparatus,Peroxisome,"
        "Peripheral,Transmembrane,Lipid anchor,Soluble"
    )
    row = "p1,Mitochondrion,Mitochondrial transit peptide,Soluble,0.05,0.02,0.01,0.01,0.92,0.03,0.01,0.01,0.01,0.01,0.02,0.01,0.01,0.97"
    csv = tmp_path / "results.csv"
    csv.write_text(header + "\n" + row + "\n")
    parsed = _parse_deeploc_output(csv)["p1"]
    assert parsed["predicted"] == "Mitochondrion"
    assert parsed["probabilities"]["Cytoplasm"] == 0.05
