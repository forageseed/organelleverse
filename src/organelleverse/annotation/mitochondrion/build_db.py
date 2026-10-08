"""Build mitochondrial HMM profiles and reference databases.

Usage:
    python -m organelleverse.annotation.mitochondrion.build_db --source ref_db01
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path

from Bio import SeqIO

from organelleverse.core.external import MESSAGE_TAIL_LINES, tail_lines

logger = logging.getLogger(__name__)


def build_hmm_db(source_dir: Path, output_dir: Path, mafft: str = "mafft") -> None:
    """Build HMM profiles from PMGA v1 protein FASTA files.

    Args:
        source_dir: Path to ref_db01/ (contains CommonGenes/, OverlapGenes/, smallExonGenes/)
        output_dir: Path to the mitochondrial annotation data directory.
        mafft: Path to mafft executable
    """
    hmm_dir = output_dir / "hmm_profiles" / "pcg"
    blast_dir = output_dir / "blast_refs" / "pcg"
    rrna_dir = output_dir / "blast_refs" / "rrna"
    info_dir = output_dir / "gene_info"

    for d in [hmm_dir, blast_dir, rrna_dir, info_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Collect all protein FASTA files
    protein_files = []
    for category in ["CommonGenes", "OverlapGenes", "smallExonGenes"]:
        cat_dir = source_dir / category
        if cat_dir.exists():
            for f in sorted(cat_dir.glob("*.Protein.fasta")):
                gene_name = f.stem.replace(".Protein", "")
                protein_files.append((gene_name, category, f))

    logger.info(f"Found {len(protein_files)} gene protein FASTA files")

    # Build HMM profile for each gene
    all_hmm_paths = []
    for gene_name, category, fasta_path in protein_files:
        # Copy protein FASTA for BLAST fallback
        dest = blast_dir / f"{gene_name}.Protein.fasta"
        shutil.copy2(fasta_path, dest)

        # Build HMM: MAFFT alignment -> hmmbuild
        records = list(SeqIO.parse(str(fasta_path), "fasta"))
        if len(records) < 2:
            logger.warning(
                f"Skipping {gene_name}: only {len(records)} sequences (need >= 2 for HMM)"
            )
            continue

        # MAFFT alignment
        aligned_path = hmm_dir / f"{gene_name}.aligned.fasta"
        try:
            result = subprocess.run(
                [mafft, "--auto", "--thread", "-1", str(fasta_path)],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                logger.warning(
                    f"MAFFT failed for {gene_name}: "
                    f"{tail_lines(result.stderr, MESSAGE_TAIL_LINES)}"
                )
                continue
            aligned_path.write_text(result.stdout)
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.warning(f"MAFFT error for {gene_name}: {e}")
            continue

        # Build HMM with pyhmmer
        hmm_path = hmm_dir / f"{gene_name}.hmm"
        try:
            _build_hmm_with_pyhmmer(aligned_path, hmm_path, gene_name)
            all_hmm_paths.append(hmm_path)
        except Exception as e:
            logger.warning(f"HMM build failed for {gene_name}: {e}")
            continue

        logger.info(f"  Built {gene_name}.hmm ({len(records)} seqs, {category})")

    # Concatenate all HMMs into one database and press
    if all_hmm_paths:
        combined = hmm_dir / "organelleverse_mitochondrion_pcg.hmm"
        with open(combined, "w") as out:
            for p in all_hmm_paths:
                out.write(p.read_text())
        logger.info(f"Combined {len(all_hmm_paths)} HMM profiles into {combined}")

    # Copy rRNA references
    rrna_src = source_dir / "rRNA"
    if rrna_src.exists():
        for f in rrna_src.glob("*.rRNA.fasta"):
            shutil.copy2(f, rrna_dir / f.name)

    # Copy exon-intron boundary reference if exists
    boundary_src = (
        source_dir.parent.parent.parent / "scripts" / "mtgenes_exon_intron_boundary.fasta"
    )
    if boundary_src.exists():
        bd_dir = output_dir / "exon_intron_boundary"
        bd_dir.mkdir(exist_ok=True)
        shutil.copy2(boundary_src, bd_dir / "mtgenes_exon_intron_boundary.fasta")

    # Copy existing HMM models (start/stop codon detection)
    hmm_src = source_dir.parent.parent.parent / "scripts" / "HMM-models"
    if hmm_src.exists():
        special_dir = hmm_dir / "special"
        special_dir.mkdir(exist_ok=True)
        for f in hmm_src.glob("*.hmm"):
            shutil.copy2(f, special_dir / f.name)

    logger.info(f"Database built: {len(all_hmm_paths)} HMM profiles in {hmm_dir}")


def _build_hmm_with_pyhmmer(aligned_path: Path, output_path: Path, name: str) -> None:
    """Build a single HMM profile using pyhmmer."""
    import pyhmmer
    from pyhmmer.easel import Alphabet, MSAFile

    alphabet = Alphabet.amino()
    with MSAFile(str(aligned_path), digital=True, alphabet=alphabet) as msa_file:
        msa = msa_file.read()

    msa.name = name.encode()
    builder = pyhmmer.plan7.Builder(alphabet)
    background = pyhmmer.plan7.Background(alphabet)

    hmm, _, _ = builder.build_msa(msa, background)
    with open(output_path, "wb") as f:
        hmm.write(f)


def build_gene_metadata(source_dir: Path, output_dir: Path) -> None:
    """Build gene_categories.json from PMGA v1 metadata files."""
    info_dir = output_dir / "gene_info"
    info_dir.mkdir(parents=True, exist_ok=True)

    # Parse genetoproduct.txt
    product_map = {}
    gtp = source_dir / "genetoproduct.txt"
    if gtp.exists():
        for line in gtp.read_text().strip().split("\n"):
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                product_map[parts[0]] = parts[1]

    # Parse mitochondrial gene aliases
    alias_map = {}
    mt_alias = source_dir / "list_gene_alias.mt.txt"
    if mt_alias.exists():
        for line in mt_alias.read_text().strip().split("\n"):
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                alias_map[parts[0]] = parts[1]

    # Parse chloroplast gene aliases
    cp_alias = source_dir / "list_gene_alias.cp.txt"
    if cp_alias.exists():
        for line in cp_alias.read_text().strip().split("\n"):
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                alias_map[parts[0]] = parts[1]

    # Build structured metadata
    metadata = {
        "version": "2.0",
        "source": "PMGA v1 ref_db01",
        "product_map": product_map,
        "alias_map": alias_map,
        "gene_categories": {
            "core_complex_i": [
                "nad1",
                "nad2",
                "nad3",
                "nad4",
                "nad4L",
                "nad5",
                "nad6",
                "nad7",
                "nad9",
            ],
            "core_complex_iii": ["cob"],
            "core_complex_iv": ["cox1", "cox2", "cox3"],
            "core_complex_v": ["atp1", "atp4", "atp6", "atp8", "atp9"],
            "ccm": ["ccmB", "ccmC", "ccmFC", "ccmFN"],
            "ribosomal": [
                "rpl2",
                "rpl5",
                "rpl10",
                "rpl16",
                "rps1",
                "rps2",
                "rps3",
                "rps4",
                "rps7",
                "rps10",
                "rps11",
                "rps12",
                "rps13",
                "rps14",
                "rps19",
            ],
            "other": ["matR", "mttB", "sdh3", "sdh4"],
            "overlap_genes": ["cox3"],
            "small_exon_genes": [
                "nad1",
                "nad2",
                "nad5",
                "cox1",
                "cox2",
                "nad4",
                "nad4L",
                "rps3",
                "sdh3",
                "sdh4",
            ],
        },
        "special_handling": {
            "stop_gain_rna_editing": {
                "genes": ["ccmFC", "rps10", "atp9", "atp6", "rps11"],
                "description": "C-to-U editing creates stop codons (CAA->UAA, CGA->UGA, CAG->UAG)",
            },
            "start_gain_rna_editing": {
                "genes": ["cox1", "nad1", "nad4L", "rps10"],
                "description": "ACG->AUG editing creates start codons",
            },
            "special_start": {
                "mttB": {"allowed_start": ["ATA", "ATG"]},
                "rpl16": {"note": "may have truncated first 108bp"},
            },
        },
        "genetic_code": {
            "table": 1,
            "name": "Standard",
            "note": "Plant mitochondria use standard genetic code (NCBI Table 1), NOT Table 2",
        },
    }

    out_file = info_dir / "gene_categories.json"
    out_file.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    logger.info(
        f"Wrote gene metadata to {out_file} ({len(product_map)} products, {len(alias_map)} aliases)"
    )
