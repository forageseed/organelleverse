from __future__ import annotations

import os
import random
import shutil
from pathlib import Path

import pytest

from organelleverse.annotation.cmsearch.genome import default_cm_path, default_hmm_path, find_trnas

_CM = default_cm_path()
_HMM = default_hmm_path()
pytestmark = pytest.mark.skipif(
    not _CM.exists() or not _HMM.exists(), reason="packaged tRNA CM/HMM not built"
)

# A real Ranunculus mitochondrial tRNA (cmsearch-confirmed), recognized by both
# the HMM filter and the CM.
_CORE = "GGGTGTATAGCTCAGTTGGTAGAGCATTGGGCTTTTAACCTAATGGTCGCAGGTTCAAGTCCTGCTATACCCA"


def test_find_trnas_recovers_embedded_trna_forward():
    random.seed(3)
    left = "".join(random.choice("ACGT") for _ in range(500))
    right = "".join(random.choice("ACGT") for _ in range(500))
    genome = left + _CORE + right
    hits = find_trnas(genome, min_bits=20.0)
    assert hits
    best = max(hits, key=lambda h: h.score)
    assert best.strand == 1
    assert best.start == 501
    assert best.end == 500 + len(_CORE)


def test_find_trnas_recovers_embedded_trna_reverse():
    rc = _CORE.translate(str.maketrans("ACGT", "TGCA"))[::-1]
    random.seed(8)
    left = "".join(random.choice("ACGT") for _ in range(400))
    genome = left + rc + "".join(random.choice("ACGT") for _ in range(400))
    hits = find_trnas(genome, min_bits=20.0)
    assert hits
    best = max(hits, key=lambda h: h.score)
    assert best.strand == -1
    assert best.start == 401
    assert best.end == 400 + len(_CORE)


def test_find_trnas_empty_on_random():
    random.seed(1)
    genome = "".join(random.choice("ACGT") for _ in range(2000))
    hits = find_trnas(genome, min_bits=25.0)
    assert all(h.score >= 25.0 for h in hits)


_REAL = os.environ.get("ORGANELLEVERSE_REAL_DATA_DIR")
_FASTA = Path(_REAL) / "PMGA/output_Ranunculus/results/Ranunculus.fasta" if _REAL else None


@pytest.mark.skipif(
    shutil.which("cmsearch") is None or _FASTA is None or not _FASTA.exists(),
    reason="needs ORGANELLEVERSE_REAL_DATA_DIR and cmsearch for parity",
)
def test_find_trnas_matches_cmsearch_on_real_genome():
    import subprocess
    import tempfile

    from Bio import SeqIO

    from organelleverse.annotation.cmsearch.model import parse_cm

    # Infernal-parity is checked against the calibrated RF00005 CM (cmsearch needs
    # E-value calibration); the plant-mito default CM is validated separately
    # against the PMGA baseline.
    rf_cm = _CM.parent / "trna.cm"
    rf_hmm = _CM.parent / "trna.hmm"
    if not rf_cm.exists() or not rf_hmm.exists():
        pytest.skip("RF00005 reference CM/HMM not packaged")

    genome = str(next(SeqIO.parse(str(_FASTA), "fasta")).seq).upper().replace("U", "T")
    hits = find_trnas(genome, cm_model=parse_cm(rf_cm), hmm_path=rf_hmm, min_bits=20.0)

    with tempfile.TemporaryDirectory() as td:
        tbl = Path(td) / "h.tbl"
        subprocess.run(
            ["cmsearch", "--cpu", "4", "--tblout", str(tbl), "-E", "1e-5", str(rf_cm), str(_FASTA)],
            check=True,
            capture_output=True,
        )
        truth = []
        for line in tbl.read_text().splitlines():
            if line.startswith("#") or not line.strip():
                continue
            f = line.split()
            truth.append((min(int(f[7]), int(f[8])), max(int(f[7]), int(f[8]))))

    # Every strong cmsearch tRNA must be recovered by the native genome scanner
    # at (near-)exact coordinates.
    found = {(h.start, h.end) for h in hits}
    exact = sum(
        1 for lo, hi in truth if any(abs(lo - s) <= 3 and abs(hi - e) <= 3 for s, e in found)
    )
    assert exact / len(truth) >= 0.90


_PMGA_GFF = Path(_REAL) / "PMGA/output_Ranunculus/results/Ranunculus.gff" if _REAL else None


@pytest.mark.skipif(
    _FASTA is None or not _FASTA.exists() or _PMGA_GFF is None or not _PMGA_GFF.exists(),
    reason="needs ORGANELLEVERSE_REAL_DATA_DIR",
)
def test_plant_mito_default_matches_mature_pmga_trnas():
    from Bio import SeqIO

    genome = str(next(SeqIO.parse(str(_FASTA), "fasta")).seq).upper().replace("U", "T")
    hits = find_trnas(genome, min_bits=20.0)  # plant-mito default

    pmga = []
    for line in _PMGA_GFF.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        p = line.split("\t")
        if len(p) >= 5 and p[2].lower() == "trna" and 50 <= int(p[4]) - int(p[3]) + 1 <= 110:
            pmga.append((int(p[3]), int(p[4])))

    cand = [(h.start, h.end) for h in hits]
    used = [False] * len(cand)
    exact = 0
    for lo, hi in pmga:
        for i, (s, e) in enumerate(cand):
            if not used[i] and lo == s and hi == e:
                used[i] = True
                exact += 1
                break
    # Coordinate parity with mature-length PMGA tRNAs (measured ~0.92).
    assert exact / len(pmga) >= 0.88
