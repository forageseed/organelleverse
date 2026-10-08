"""Backend installer + auto-locate for population genetics tools.

Same pattern as assembly: scan PATH + all conda envs, give GitHub/bioconda
links when not found.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any
from ..assembly.install import _list_conda_envs

__all__ = [
    "POP_BACKEND_INFO",
    "check_backend",
    "check_all_backends",
    "install_hint",
]


POP_BACKEND_INFO: dict[str, dict[str, str]] = {
    "bwa": {
        "url": "https://github.com/lh3/bwa",
        "conda": "bioconda::bwa",
        "cli": "bwa",
        "note": "C; BWT-based short-read aligner",
        "tier": "conda",
    },
    "samtools": {
        "url": "https://github.com/samtools/samtools",
        "conda": "bioconda::samtools",
        "cli": "samtools",
        "note": "C; BAM/CRAM manipulation + variant calling utilities",
        "tier": "conda",
    },
    "deepvariant": {
        "url": "https://github.com/google/deepvariant",
        "conda": "bioconda::deepvariant",
        "cli": "run_deepvariant",
        "note": "C++/Python; deep-learning variant caller (Google)",
        "tier": "conda",
    },
    "gatk": {
        "url": "https://github.com/broadinstitute/gatk",
        "conda": "bioconda::gatk4",
        "cli": "gatk",
        "note": "Java; GATK4 best-practices variant calling (Broad)",
        "tier": "conda",
    },
    "bcftools": {
        "url": "https://github.com/samtools/bcftools",
        "conda": "bioconda::bcftools",
        "cli": "bcftools",
        "note": "C; VCF/BCF manipulation + variant calling",
        "tier": "conda",
    },
    "vcftools": {
        "url": "https://github.com/vcftools/vcftools",
        "conda": "bioconda::vcftools",
        "cli": "vcftools",
        "note": "C++; VCF statistics + filtering (pi/Fst/quality)",
        "tier": "conda",
    },
    "gemma": {
        "url": "https://github.com/xiangzhou/GEMMA",
        "conda": "bioconda::gemma",
        "cli": "gemma",
        "note": "C++; genome-wide association study (GWAS) + kinship",
        "tier": "conda",
    },
    "sift4g": {
        "url": "https://github.com/rvaser/sift",
        "conda": "bioconda::sift_blink",
        "cli": "sift4g",
        "note": "C++; predicting deleterious coding variants (SIFT4G)",
        "tier": "conda",
    },
    "plink": {
        "url": "https://www.cog-genomics.org/plink/",
        "conda": "bioconda::plink",
        "cli": "plink",
        "note": "C/C++; whole-genome association analysis toolset",
        "tier": "conda",
    },
}


def check_backend(name: str, *, scan_envs: bool = True) -> dict[str, Any]:
    """Check if a population backend is installed.

    Scans PATH + all conda/micromamba envs.
    Returns {name, installed, path, env, note}.
    """
    info = POP_BACKEND_INFO.get(name)
    if not info:
        return {
            "name": name,
            "installed": False,
            "path": None,
            "env": None,
            "note": f"Unknown backend: {name}",
        }

    cli = info.get("cli", name)
    path = shutil.which(cli)
    if path:
        return {
            "name": name,
            "installed": True,
            "path": path,
            "env": None,
            "note": info.get("note", ""),
        }
    if scan_envs:
        for env_bin in _list_conda_envs():
            candidate = env_bin / cli
            if candidate.exists() and candidate.is_file():
                parts = env_bin.parts
                env_name = (
                    parts[parts.index("envs") + 1]
                    if "envs" in parts and parts.index("envs") + 1 < len(parts)
                    else None
                )
                return {
                    "name": name,
                    "installed": True,
                    "path": str(candidate),
                    "env": env_name,
                    "note": f"{info.get('note', '')} [env: {env_name}]".strip(),
                }
    return {
        "name": name,
        "installed": False,
        "path": None,
        "env": None,
        "note": info.get("note", ""),
    }


def check_all_backends(*, scan_envs: bool = True) -> dict[str, dict[str, Any]]:
    """Check all population backends. Returns {name: status_dict}."""
    return {name: check_backend(name, scan_envs=scan_envs) for name in POP_BACKEND_INFO}


def install_hint(name: str) -> str:
    """Return a human-readable install guide with URLs."""
    info = POP_BACKEND_INFO.get(name)
    if not info:
        return f"Unknown backend: {name}"
    url = info.get("url", "?")
    conda = info.get("conda", "").split("::")[-1] if info.get("conda") else None
    lines = [f"Backend '{name}' not found. Install options:"]
    if conda:
        lines.append(f"  conda:  conda install -c bioconda -c conda-forge {conda}")
    lines.append(f"  docs:   {url}")
    if info.get("note"):
        lines.append(f"  note:   {info['note']}")
    return "\n".join(lines)
