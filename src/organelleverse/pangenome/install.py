"""Backend installer + auto-locate for pangenome tools.

Same pattern as assembly: scan PATH + all conda envs, give GitHub/bioconda
links when not found.
"""

from __future__ import annotations

import shutil
from typing import Any

from ..assembly.install import _list_conda_envs

__all__ = [
    "PANTOOLS_GRAPH_DIAGNOSTIC",
    "PAN_BACKEND_INFO",
    "check_all_backends",
    "check_backend",
    "install_hint",
]


PANTOOLS_GRAPH_DIAGNOSTIC = (
    "PanTools 4.3.5 property-export adapter preserves sample/molecule paths and "
    "verifies their exact sequences. Requires a compatible Java runtime (Java 11 "
    "verified) and molecules at least k bases. Output GFA retains k-1 overlaps; "
    "ODGI operations requiring zero overlaps report that representation limit."
)


PAN_BACKEND_INFO: dict[str, dict[str, str]] = {
    "minigraph": {
        "url": "https://github.com/lh3/minigraph",
        "conda": "bioconda::minigraph",
        "cli": "minigraph",
        "note": "C; sequence-to-graph alignment + pangenome graph construction",
        "tier": "conda",
    },
    "pggb": {
        "url": "https://github.com/pangenome/pggb",
        "conda": "bioconda::pggb",
        "cli": "pggb",
        "note": "wfmash + seqwish + smoothxg; all-vs-all pangenome graph",
        "tier": "conda",
    },
    "pantools": {
        "url": "https://git.wur.nl/bioinformatics/pantools",
        "conda": "bioconda::pantools",
        "cli": "pantools",
        "note": "Java; k-mer based pangenome database. " + PANTOOLS_GRAPH_DIAGNOSTIC,
        "tier": "conda",
    },
    "mummer": {
        "url": "https://github.com/mummer4/mummer",
        "conda": "bioconda::mummer4",
        "cli": "nucmer",
        "note": "C++; pairwise whole-genome alignment (nucmer/delta-filter/show-coords)",
        "tier": "conda",
    },
    "reputer": {
        "url": "https://bibiserv.cebitec.uni-bielefeld.de/reputer",
        "cli": "reputer",
        "note": "Commercial (Bielefeld); direct/inverted/palindromic repeat detection",
        "tier": "source",
    },
    "genespace": {
        "url": "https://github.com/ltgooo/genespace",
        "cli": "genespace",
        "note": "R package; orthology-based synteny visualization",
        "tier": "source",
    },
    "minimap2": {
        "url": "https://github.com/lh3/minimap2",
        "conda": "bioconda::minimap2",
        "cli": "minimap2",
        "note": "C; versatile pairwise alignment for long reads / assemblies",
        "tier": "conda",
    },
}


def check_backend(name: str, *, scan_envs: bool = True) -> dict[str, Any]:
    """Check if a pangenome backend is installed.

    Scans PATH + all conda/micromamba envs.
    Returns {name, installed, path, env, note}.
    """
    info = PAN_BACKEND_INFO.get(name)
    if not info:
        return {
            "name": name,
            "installed": False,
            "path": None,
            "env": None,
            "note": f"Unknown backend: {name}",
        }

    cli = info.get("cli", name)
    # 1. PATH
    path = shutil.which(cli)
    if path:
        return {
            "name": name,
            "installed": True,
            "path": path,
            "env": None,
            "note": info.get("note", ""),
        }
    # 2. conda envs
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
    """Check all pangenome backends. Returns {name: status_dict}."""
    return {name: check_backend(name, scan_envs=scan_envs) for name in PAN_BACKEND_INFO}


def install_hint(name: str) -> str:
    """Return a human-readable install guide with URLs."""
    info = PAN_BACKEND_INFO.get(name)
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
