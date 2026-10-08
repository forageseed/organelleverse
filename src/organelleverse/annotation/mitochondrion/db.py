"""Database manager — verify, locate, and load reference data."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from ..data import mitochondrion_data_dir

_DATA_DIR = mitochondrion_data_dir()


def default_mitochondrion_reference_dir() -> Path:
    """Return the packaged mitochondrial annotation reference directory."""
    return _DATA_DIR


class DBManager:
    """Manage mitochondrial annotation reference data."""

    def __init__(self, db_path: Path | None = None):
        self.data_dir = Path(db_path) if db_path else _DATA_DIR

    @property
    def hmm_dir(self) -> Path:
        return self.data_dir / "hmm_profiles" / "pcg"

    @property
    def combined_hmm(self) -> Path:
        return self.hmm_dir / "organelleverse_mitochondrion_pcg.hmm"

    @property
    def blast_ref_dir(self) -> Path:
        return self.data_dir / "blast_refs" / "pcg"

    @property
    def exon_ref_dir(self) -> Path:
        return self.data_dir / "blast_refs" / "exons"

    @property
    def rrna_ref_dir(self) -> Path:
        """rRNA reference directory - prefers mitochondrial, falls back to general."""
        mito_dir = self.data_dir / "blast_refs" / "rrna_mito"
        if mito_dir.exists() and any(mito_dir.glob("*.fasta")):
            return mito_dir
        return self.data_dir / "blast_refs" / "rrna"

    @property
    def trna_ref_dir(self) -> Path:
        return self.data_dir / "blast_refs" / "trna"

    @property
    def gene_info_dir(self) -> Path:
        return self.data_dir / "gene_info"

    @lru_cache  # noqa: B019 - intentional per-manager metadata cache
    def load_gene_metadata(self) -> dict:
        """Load gene_categories.json."""
        path = self.gene_info_dir / "gene_categories.json"
        if path.exists():
            return json.loads(path.read_text())
        return {}

    @lru_cache  # noqa: B019 - intentional per-manager metadata cache
    def load_product_map(self) -> dict[str, str]:
        """Load gene_name -> product description mapping."""
        meta = self.load_gene_metadata()
        return meta.get("product_map", {})

    @lru_cache  # noqa: B019 - intentional per-manager metadata cache
    def load_alias_map(self) -> dict[str, str]:
        """Load gene alias -> standard name mapping."""
        meta = self.load_gene_metadata()
        return meta.get("alias_map", {})

    def resolve_gene_name(self, name: str) -> str:
        """Resolve a gene alias to standard name."""
        aliases = self.load_alias_map()
        return aliases.get(name, name)

    def get_product(self, gene_name: str) -> str:
        """Get product description for a gene."""
        products = self.load_product_map()
        return products.get(gene_name, f"{gene_name} protein")

    def is_stop_gain_gene(self, gene_name: str) -> bool:
        """Check if gene requires RNA editing for stop codon creation."""
        meta = self.load_gene_metadata()
        genes = meta.get("special_handling", {}).get("stop_gain_rna_editing", {}).get("genes", [])
        return gene_name in genes

    def is_start_gain_gene(self, gene_name: str) -> bool:
        """Check if gene requires RNA editing for start codon creation."""
        meta = self.load_gene_metadata()
        genes = meta.get("special_handling", {}).get("start_gain_rna_editing", {}).get("genes", [])
        return gene_name in genes

    def verify(self) -> list[str]:
        """Verify database completeness. Returns list of issues."""
        issues = []
        if not self.data_dir.exists():
            issues.append(f"Data directory not found: {self.data_dir}")
            return issues

        if not self.combined_hmm.exists():
            # Check for individual HMMs
            hmms = list(self.hmm_dir.glob("*.hmm")) if self.hmm_dir.exists() else []
            if not hmms:
                issues.append(
                    "No HMM profiles found. Build the mitochondrial annotation database first."
                )
            else:
                issues.append(f"Combined HMM not found, but {len(hmms)} individual profiles exist.")

        meta = self.gene_info_dir / "gene_categories.json"
        if not meta.exists():
            issues.append(
                "gene_categories.json not found. Build the mitochondrial annotation database first."
            )

        return issues
