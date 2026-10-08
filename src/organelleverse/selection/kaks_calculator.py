"""KaKs_Calculator 3.0 binary hook — all 8 standard methods via subprocess.

KaKs_Calculator 3.0 (Zhang et al. 2021, Genomics Proteomics Bioinformatics)
implements all 8 standard Ka/Ks methods in optimized C++:

  NG86    — Nei-Gojobori 1986 (counting + JC)
  LWL85   — Li-Wu-Luo 1985 (degeneracy-class counting + K2P)
  LPB93   — Li 1993 / Pamilo-Bianchi 1993 (modified LWL85 weighting)
  MLWL85  — Modified LWL85 (Tzeng et al. 2004)
  MLPB93  — Modified LPB93 (Tzeng et al. 2004)
  YN00    — Yang-Nielsen 2000 (approximate-likelihood, F84)
  MYN     — Modified YN (Zhang et al. 2006, TN93)
  GY94    — Goldman-Yang 1994 (full maximum likelihood)
  MS      — Model Selection (14 models, AICc)
  MA      — Model Averaging (Akaike-weighted)

This module provides:
  - ``run_kaks_calculator()`` — run KaKs_Calculator on an AXT file
  - ``fasta_to_axt()`` — convert codon-aligned FASTA pairs to AXT format
  - ``parse_kaks_output()`` — parse the TSV output
  - ``kaks_calculator()`` — entry point (FASTA → AXT → run → parse)
  - Auto-locate (PATH + conda envs) + install hint (GitHub/bioconda)

Pure Python methods (NG86, LWL85, LPB93) are in ``kaks.py`` for cases where
KaKs_Calculator is not installed. This module covers the remaining methods
(GY94, YN00, MYN, MS, MA) that require numerical optimization.

Reference: https://doi.org/10.1016/j.gpb.2021.12.002
Source:    https://github.com/kullrich/kakscalculator2 (v2) / BIG-CAS (v3)
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from itertools import combinations
from pathlib import Path
from typing import Any

from . import _contract
from ..core.result import OrganelleResult
from .._bio import read_fasta

__all__ = [
    "SUPPORTED_METHODS",
    "check_kaks_calculator",
    "install_hint",
    "run_kaks_calculator",
    "fasta_to_axt",
    "parse_kaks_output",
    "kaks_calculator",
    "run_kaks_calculator_workflow",
]


SUPPORTED_METHODS = (
    "NG",
    "LWL",
    "LPB",
    "MLWL",
    "MLPB",
    "YN",
    "MYN",
    "GY",
    "MS",
    "MA",
)


# =========================================================================
# Auto-locate + install hint
# =========================================================================

_KAKS_INFO = {
    "cli": "KaKs",
    "url": "https://github.com/BGI-shenzhen/KaKs_Calculator",
    "conda": "bioconda::kakscalculator2",
    "note": "C++; all 8 Ka/Ks methods (NG/LWL/LPB/YN/MYN/GY/MS/MA). "
    "v3.0: doi.org/10.1016/j.gpb.2021.12.002",
}


def check_kaks_calculator() -> dict[str, Any]:
    """Check if KaKs_Calculator is installed (PATH + all conda envs)."""
    from ..assembly.install import _list_conda_envs

    cli = _KAKS_INFO["cli"]

    # 1. PATH
    path = shutil.which(cli)
    if path:
        return {"installed": True, "path": path, "env": None, "note": _KAKS_INFO["note"]}

    # 2. conda envs
    for env_bin in _list_conda_envs():
        candidate = env_bin / cli
        if candidate.exists():
            parts = env_bin.parts
            env_name = parts[parts.index("envs") + 1] if "envs" in parts else None
            return {
                "installed": True,
                "path": str(candidate),
                "env": env_name,
                "note": f"{_KAKS_INFO['note']} [env: {env_name}]",
            }

    return {"installed": False, "path": None, "env": None, "note": _KAKS_INFO["note"]}


def install_hint() -> str:
    """Install guide for KaKs_Calculator."""
    url = _KAKS_INFO["url"]
    conda = _KAKS_INFO["conda"].split("::")[-1]
    return (
        f"KaKs_Calculator not found. Install options:\n"
        f"  conda:  conda install -c bioconda -c conda-forge {conda}\n"
        f"  source: git clone {url}\n"
        f"  docs:   {url}\n"
        f"  note:   {_KAKS_INFO['note']}"
    )


# =========================================================================
# AXT format conversion
# =========================================================================


def fasta_to_axt(
    fasta_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """Convert codon-aligned FASTA pairs to AXT format for KaKs_Calculator.

    AXT format:
        <pair_name>
        <sequence_1>
        <sequence_2>
        (blank line)

    All-vs-all pairs are generated.
    """
    seqs = read_fasta(Path(fasta_path))
    if output_path is None:
        output_path = Path(fasta_path).with_suffix(".axt")
    output_path = Path(output_path)

    lines: list[str] = []
    for i, j in combinations(range(len(seqs)), 2):
        name_i, seq_i = seqs[i]
        name_j, seq_j = seqs[j]
        pair_name = f"{name_i}-{name_j}"
        lines.append(pair_name)
        lines.append(seq_i)
        lines.append(seq_j)
        lines.append("")  # blank line separator

    output_path.write_text("\n".join(lines) + "\n")
    return output_path


# =========================================================================
# Run KaKs_Calculator
# =========================================================================


def run_kaks_calculator(
    axt_file: str | Path,
    output_file: str | Path,
    *,
    method: str = "YN",
    genetic_code: int = 1,
    kaks_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run KaKs_Calculator on an AXT file.

    Parameters
    ----------
    axt_file : path to AXT format alignment
    output_file : path for output TSV
    method : one of NG/LWL/LPB/MLWL/MLPB/YN/MYN/GY/MS/MA
    genetic_code : NCBI translation table (1=standard, 11=bacterial/plastid)
    kaks_path : explicit path to KaKs binary (auto-located if None)

    Returns {success, method, output_path, stdout, stderr}.
    """
    # locate binary
    if kaks_path:
        binary = str(kaks_path)
    else:
        loc = check_kaks_calculator()
        binary = loc.get("path")
        if not binary:
            return {
                "success": False,
                "error": install_hint(),
                "method": method,
                "output_path": None,
            }

    cmd = [
        binary,
        "-i",
        str(axt_file),
        "-o",
        str(output_file),
        "-m",
        method,
        "-c",
        str(genetic_code),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    success = result.returncode == 0 and Path(output_file).exists()

    return {
        "success": success,
        "method": method,
        "output_path": str(output_file) if success else None,
        "command": " ".join(cmd),
        "stdout": result.stdout[:500],
        "stderr": result.stderr[:500],
        "returncode": result.returncode,
    }


# =========================================================================
# Parse output
# =========================================================================


def parse_kaks_output(output_file: str | Path) -> list[dict[str, Any]]:
    """Parse KaKs_Calculator output TSV.

    Columns: Pair, KaKs_Ka, KaKs_Ks, KaKs_Ka/Ks, KaKs_P-Value(Cha),
             KaKs_P-Value(Fisher), KaKs_Length, ...

    Returns list of dicts, one per pair.
    """
    text = Path(output_file).read_text(errors="replace")
    lines = text.strip().splitlines()
    if not lines:
        return []

    # header
    header = lines[0].split("\t")
    results: list[dict[str, Any]] = []
    for line in lines[1:]:
        fields = line.split("\t")
        if len(fields) < 4:
            continue
        row: dict[str, Any] = {}
        for i, col in enumerate(header):
            if i < len(fields):
                val = fields[i].strip()
                # try numeric conversion
                try:
                    row[col] = float(val)
                except ValueError:
                    row[col] = val
        results.append(row)
    return results


# =========================================================================
# Entry point: FASTA → AXT → KaKs → parse
# =========================================================================


def kaks_calculator(
    cds_fasta: str | Path,
    *,
    method: str = "YN",
    genetic_code: int = 1,
    kaks_path: str | Path | None = None,
) -> OrganelleResult:
    """Compute Ka/Ks metrics without binding a user-facing output directory."""
    with tempfile.TemporaryDirectory() as tmp:
        result = run_kaks_calculator_workflow(
            cds_fasta,
            method=method,
            output_dir=tmp,
            genetic_code=genetic_code,
            kaks_path=kaks_path,
        )
    if result.status != "ok":
        return result
    return result.model_copy(
        update={
            "artifacts": (),
            "flags": result.flags + ("output_dir_unbound",),
            "summary_text": (
                result.summary_text
                + " This is a compute-only result; call run_kaks_calculator_workflow() to keep files."
            ),
        }
    )


def run_kaks_calculator_workflow(
    cds_fasta: str | Path,
    *,
    method: str = "YN",
    output_dir: str | Path,
    genetic_code: int = 1,
    kaks_path: str | Path | None = None,
) -> OrganelleResult:
    """Compute Ka/Ks using KaKs_Calculator 3.0.

    Full pipeline: FASTA → AXT → KaKs_Calculator → parse results.

    Parameters
    ----------
    cds_fasta : path to codon-aligned CDS FASTA (≥2 sequences)
    method : NG/LWL/LPB/MLWL/MLPB/YN/MYN/GY/MS/MA
    output_dir : directory for intermediate files (default: temp)
    genetic_code : NCBI table (1=standard, 11=plastid)
    kaks_path : explicit path to KaKs binary
    """
    method = method.upper()
    parameters = {"method": method, "genetic_code": genetic_code}
    if method not in SUPPORTED_METHODS:
        return _contract.failed(
            "kaks_calculator",
            summary_text=f"Method '{method}' not supported. Use one of: {SUPPORTED_METHODS}",
            anomalies=[f"bad_method:{method}"],
            parameters=parameters,
        )

    # check binary — use explicit path if given, else auto-locate
    if kaks_path:
        if not Path(kaks_path).exists():
            return _contract.failed(
                "kaks_calculator",
                summary_text=install_hint(),
                anomalies=["kaks_calculator_not_found"],
                parameters=parameters,
            )
        binary = str(kaks_path)
        loc = {"installed": True, "path": binary, "env": None, "note": ""}
    else:
        loc = check_kaks_calculator()
        if not loc["installed"]:
            return _contract.failed(
                "kaks_calculator",
                summary_text=install_hint(),
                anomalies=["kaks_calculator_not_found"],
                parameters=parameters,
            )
        binary = loc["path"]

    # prepare output dir
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
    else:
        import tempfile

        out = Path(tempfile.mkdtemp())

    # FASTA → AXT
    axt_path = out / "input.axt"
    fasta_to_axt(cds_fasta, axt_path)

    # run KaKs_Calculator
    kaks_output = out / "kaks_result.tsv"
    run_result = run_kaks_calculator(
        axt_path, kaks_output, method=method, genetic_code=genetic_code, kaks_path=binary
    )

    if not run_result["success"]:
        return _contract.failed(
            "kaks_calculator",
            summary_text=f"KaKs_Calculator failed: {run_result.get('stderr', '')}",
            anomalies=["kaks_run_failed"],
            method=method,
            parameters=parameters,
        )

    # parse
    pairs = parse_kaks_output(kaks_output)
    if not pairs:
        return _contract.failed(
            "kaks_calculator",
            summary_text="KaKs_Calculator produced no results.",
            anomalies=["no_results"],
            parameters=parameters,
        )

    # aggregate
    ka_values = [p.get("KaKs_Ka", 0) for p in pairs if isinstance(p.get("KaKs_Ka"), (int, float))]
    ks_values = [p.get("KaKs_Ks", 0) for p in pairs if isinstance(p.get("KaKs_Ks"), (int, float))]
    mean_ka = sum(ka_values) / len(ka_values) if ka_values else 0
    mean_ks = sum(ks_values) / len(ks_values) if ks_values else 0
    ratio = mean_ka / mean_ks if mean_ks else 0

    return _contract.ok(
        "kaks_calculator",
        artifacts=_contract.artifacts((kaks_output,)),
        metrics={
            "method": method,
            "n_pairs": len(pairs),
            "Ka": round(mean_ka, 6),
            "Ks": round(mean_ks, 6),
            "Ka_Ks": round(ratio, 6),
            "binary": loc["path"],
            # The per-pair rows the pre-v1 contract carried in ``key_findings``
            # live here now: a canonical Finding holds one scalar, so the
            # structured pair table belongs in ``metrics``.
            "pair_results": [
                {
                    "name": p.get("Pair", ""),
                    "Ka": p.get("KaKs_Ka"),
                    "Ks": p.get("KaKs_Ks"),
                }
                for p in pairs[:10]
            ],
        },
        findings=_contract.findings(
            "kaks_calculator",
            (
                ("method", method),
                ("Ka", round(mean_ka, 6)),
                ("Ks", round(mean_ks, 6)),
                ("Ka_Ks", round(ratio, 6)),
                *[(f"pair:{p.get('Pair', '')}", p.get("KaKs_Ka")) for p in pairs[:10]],
            ),
        ),
        flags=("purifying" if ratio < 1 else "positive_selection" if ratio > 1 else "neutral",),
        summary_text=(
            f"KaKs_Calculator ({method}): {len(pairs)} pairs, "
            f"mean Ka={mean_ka:.6f}, Ks={mean_ks:.6f}, Ka/Ks={ratio:.6f}."
        ),
        method=f"KaKs_Calculator_{method}",
        parameters=parameters,
    )
