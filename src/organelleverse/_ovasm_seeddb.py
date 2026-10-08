"""Versioned offline DNA seed databases shipped with the Python distribution."""

from __future__ import annotations

import json
from pathlib import Path

VERSION = "v20261007"
ROOT = Path(__file__).parent / "data" / "ovasm_seeddb" / VERSION


def bundled_seed(organelle: str) -> tuple[Path, dict]:
    if organelle not in {"mitochondrion", "plastid"}:
        raise ValueError("SeedDB organelle must be mitochondrion or plastid")
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    database = manifest["databases"][organelle]
    path = ROOT / database["file"]
    if not path.is_file():
        raise FileNotFoundError(f"Bundled OVASM SeedDB FASTA not found: {path}")
    return path, {
        "source": "builtin",
        "name": manifest["name"],
        "version": manifest["version"],
        "organelle": organelle,
        "reference_count": database["reference_count"],
        "sequence_count": database["sequence_count"],
        "manifest": str(ROOT / "manifest.json"),
    }


GENEDB_VERSION = "v20261003"
GENEDB_ROOT = Path(__file__).parent / "data" / "ovasm_genedb" / GENEDB_VERSION


def bundled_gene_db() -> tuple[Path, dict]:
    """Organelle protein database of ``ovasm label`` (headers organelle|gene|accession)."""
    manifest = json.loads((GENEDB_ROOT / "manifest.json").read_text(encoding="utf-8"))
    path = GENEDB_ROOT / manifest["file"]
    if not path.is_file():
        raise FileNotFoundError(f"Bundled OVASM GeneDB FASTA not found: {path}")
    return path, {
        "name": manifest["name"],
        "version": manifest["version"],
        "proteins": manifest["proteins"],
        "genes": manifest["genes"],
    }
