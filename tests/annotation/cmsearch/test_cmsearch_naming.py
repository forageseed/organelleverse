from __future__ import annotations

from pathlib import Path

import pytest

from organelleverse.annotation.cmsearch.model import parse_cm
from organelleverse.annotation.cmsearch.naming import gene_name, name_trna
from tests._paths import PROJECT_ROOT

_PLANT_MITO_CM = PROJECT_ROOT / "src/organelleverse/annotation/data/models/trna/plant_mito_trna.cm"
pytestmark = pytest.mark.skipif(not _PLANT_MITO_CM.exists(), reason="plant-mito CM not built")

# Real Ranunculus mitochondrial tRNAs (cmsearch/PMGA-confirmed).
_TRNK_UUU = "GGGTGTATAGCTCAGTTGGTAGAGCATTGGGCTTTTAACCTAATGGTCGCAGGTTCAAGTCCTGCTATACCCA"
_TRNF_GAA = "GCGGATTTAGCTCAGTTGGTAGAGCAGAGGACTGAAAATCCTCGTGTCACCAGTTCAAATCTGGTAATCCGCT"


def test_gene_name_format():
    assert gene_name("K", "TTT") == "trnK(ttt)"
    assert gene_name("fM", "CAT") == "trnfM(cat)"


def test_name_trnk_uuu_via_traceback():
    model = parse_cm(_PLANT_MITO_CM)
    named = name_trna(model, _TRNK_UUU)
    assert named is not None
    anticodon, aa, gene = named
    assert anticodon == "TTT"
    assert aa == "K"
    assert gene == "trnK(ttt)"


def test_name_returns_none_on_non_trna():
    model = parse_cm(_PLANT_MITO_CM)
    assert name_trna(model, "ACGT" * 20) is None


def test_anticodon_columns_auto_calibrate():
    from organelleverse.annotation.cmsearch.naming import _calibrate_anticodon_columns

    model = parse_cm(_PLANT_MITO_CM)
    cols = _calibrate_anticodon_columns(model)
    # Three consecutive consensus columns; for this packaged build, 81/82/83.
    assert len(cols) == 3
    assert cols[1] == cols[0] + 1 and cols[2] == cols[1] + 1
    # Naming with auto-calibration reads the right anticodon.
    named = name_trna(model, _TRNK_UUU)
    assert named is not None and named[0] == "TTT"


_RF00005_CM = PROJECT_ROOT / "src/organelleverse/annotation/data/models/trna/trna.cm"


@pytest.mark.skipif(not _RF00005_CM.exists(), reason="RF00005 CM not packaged")
def test_naming_works_on_rf00005_via_calibration():
    # Auto-calibration must adapt to a different model (RF00005 has different
    # consensus column numbering than the plant-mito CM).
    model = parse_cm(_RF00005_CM)
    named = name_trna(model, _TRNK_UUU)
    assert named is not None
    assert named[0] == "TTT"
    assert named[1] == "K"


import os  # noqa: E402

_REAL = os.environ.get("ORGANELLEVERSE_REAL_DATA_DIR")
_FASTA = Path(_REAL) / "PMGA/output_Ranunculus/results/Ranunculus.fasta" if _REAL else None
_GFF = Path(_REAL) / "PMGA/output_Ranunculus/results/Ranunculus.gff" if _REAL else None


@pytest.mark.skipif(
    _FASTA is None or not _FASTA.exists() or _GFF is None or not _GFF.exists(),
    reason="needs ORGANELLEVERSE_REAL_DATA_DIR",
)
def test_annotate_trna_cm_names_match_pmga():
    import tempfile

    from organelleverse.annotation.mitochondrion.trna_native import annotate_trna_cm

    hits = annotate_trna_cm(_FASTA, Path(tempfile.mkdtemp()))
    by_start = {h.start: h for h in hits}

    correct = total = 0
    for line in _GFF.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        p = line.split("\t")
        if len(p) < 9 or p[2].lower() != "trna" or p[6] != "+":
            continue
        if not (60 <= int(p[4]) - int(p[3]) + 1 <= 90):
            continue
        name = next(
            (kv.split("=", 1)[1] for kv in p[8].split(";") if kv.lower().startswith("name=")), ""
        )
        if "-" not in name:
            continue
        pmga_ac = name.split("-")[-1].replace("U", "T")
        h = by_start.get(int(p[3]))
        if h is None or not h.anticodon:
            continue
        total += 1
        if h.anticodon.upper() == pmga_ac:
            correct += 1
    assert total > 0
    assert correct / total >= 0.90
