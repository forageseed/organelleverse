"""Partitioned supermatrices and IQ-TREE partition-scheme/model selection.

Three steps that the literature usually performs with PhyloSuite +
PartitionFinder/ModelFinder before a Bayesian (MrBayes) analysis:

1. :func:`build_partitioned_supermatrix` - pull nucleotide CDS for genes shared
   by several annotated genomes, align each gene *codon-aware* (translate,
   align the proteins with the package's own :func:`~.phylo.align`, then
   back-translate so codons never split) and concatenate them. Emits a FASTA
   supermatrix and a NEXUS ``sets`` block with one charset per gene or per
   gene codon position (``1-1521\\3`` style).
2. :func:`select_partition_scheme` - run IQ-TREE with ``-p <partitions>`` and
   ``-m MFP+MERGE`` (ModelFinder + the PartitionFinder-style greedy merging of
   Lanfear et al. 2012/2017; ``-rclusterf`` uses fast relaxed clustering) and
   parse the best scheme (``<prefix>.best_scheme.nex``) into a model table.
   ``model_set="mrbayes"`` passes ``-mset mrbayes`` so ModelFinder only tests
   substitution models MrBayes can express, which makes the MrBayes model
   mapping exact.
3. :func:`iqtree_to_mrbayes_model` - the documented IQ-TREE -> MrBayes mapping
   (see its docstring) used by the MrBayes NEXUS writer.

IQ-TREE is a complete inference program; it is called as an external tool via
:func:`organelleverse.core.external.run_external`.
"""

from __future__ import annotations

import os
import re
import shutil
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..core.errors import OrganelleDependencyError, OrganelleInputError
from ..core.external import run_external
from ..core.result import OrganelleResult
from ._results import artifact_for, failed_result, findings, ok_result, provenance

__all__ = [
    "IQTREE_INSTALL_HINT",
    "PartitionCharset",
    "build_partitioned_supermatrix",
    "iqtree_to_mrbayes_model",
    "parse_nexus_sets",
    "resolve_iqtree",
    "select_partition_scheme",
]

IQTREE_INSTALL_HINT = (
    "IQ-TREE not found (tried iqtree3, iqtree2, iqtree on PATH). Install: "
    "conda install -c bioconda -c conda-forge iqtree  (or pass iqtree_bin=...)."
)

_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.\-]")


# ---------------------------------------------------------------------------
# NEXUS sets parsing / writing
# ---------------------------------------------------------------------------

PartitionCharset = tuple[str, str]  # (name, "1-100\\3 201-300")


def _strip_nexus_comments(text: str) -> str:
    return re.sub(r"\[[^\]]*\]", "", text)


def parse_nexus_sets(path: str | Path) -> dict[str, Any]:
    """Parse ``charset`` and ``charpartition`` statements of a NEXUS sets block.

    Returns ``{"charsets": [(name, spec), ...], "models": {name: model}}``.
    ``models`` is filled from an IQ-TREE style ``charpartition x = MODEL:
    name, MODEL2: name2;`` statement (as written in ``*.best_scheme.nex``);
    it is empty for a plain partition file.
    """
    text = _strip_nexus_comments(Path(path).read_text())
    charsets: list[PartitionCharset] = []
    models: dict[str, str] = {}
    for statement in text.split(";"):
        stmt = " ".join(statement.split())
        match = re.match(r"(?i)^charset\s+(\S+)\s*=\s*(.+)$", stmt)
        if match:
            name = match.group(1).strip("'\"")
            spec = match.group(2).strip()
            # IQ-TREE writes "charset x = file.fa: 1-10;" when files differ.
            if ":" in spec:
                spec = spec.split(":", 1)[1].strip()
            charsets.append((name, spec))
            continue
        match = re.match(r"(?i)^charpartition\s+\S+\s*=\s*(.+)$", stmt)
        if match:
            for item in match.group(1).split(","):
                item = item.strip()
                if ":" in item:
                    model, name = item.rsplit(":", 1)
                    models[name.strip().strip("'\"")] = model.strip()
    if not charsets:
        raise OrganelleInputError(
            code="phylogeny.partition.no_charsets",
            message=f"no charset statements found in {path}",
        )
    return {"charsets": charsets, "models": models}


def _charset_sites(spec: str, nchar: int | None = None) -> list[int]:
    """Expand a NEXUS charset spec (``1-100\\3 5 7-9``) to 1-based sites."""
    sites: list[int] = []
    for token in spec.replace(",", " ").split():
        match = re.fullmatch(r"(\d+)(?:-(\d+|\.))?(?:\\(\d+))?", token)
        if not match:
            raise OrganelleInputError(
                code="phylogeny.partition.bad_charset",
                message=f"unsupported charset token {token!r}",
            )
        start = int(match.group(1))
        end_token = match.group(2)
        if end_token == ".":
            if nchar is None:
                raise OrganelleInputError(
                    code="phylogeny.partition.bad_charset",
                    message="'.' range end requires the alignment length",
                )
            end = nchar
        else:
            end = int(end_token) if end_token else start
        step = int(match.group(3)) if match.group(3) else 1
        sites.extend(range(start, end + 1, step))
    return sites


# ---------------------------------------------------------------------------
# Supermatrix construction
# ---------------------------------------------------------------------------


def _taxon_name(record: Any, path: Path) -> str:
    accessions = record.annotations.get("accessions") or []
    candidate = accessions[0] if accessions and accessions[0] else record.name or record.id
    if not candidate or candidate in {"<unknown", "unknown"}:
        candidate = path.stem
    return _SAFE_NAME.sub("_", str(candidate))


def _read_cds(path: Path, genetic_code: int) -> tuple[Any, dict[str, str]]:
    from Bio import SeqIO  # type: ignore

    record = next(SeqIO.parse(str(path), "genbank"))
    genes: dict[str, str] = {}
    for feature in record.features:
        if feature.type != "CDS":
            continue
        gene = (feature.qualifiers.get("gene") or [""])[0].strip()
        if not gene:
            continue
        seq = str(feature.extract(record.seq)).upper()
        # Keep the longest copy (IR duplicates are identical; fragments shorter).
        if len(seq) > len(genes.get(gene, "")):
            genes[gene] = seq
    return record, genes


def _codon_clean(seq: str) -> str:
    seq = re.sub(r"[^ACGTRYSWKMBDHVN]", "N", seq.upper())
    seq = seq[: len(seq) - len(seq) % 3]
    if seq[-3:] in {"TAA", "TAG", "TGA"}:
        seq = seq[:-3]
    return seq


def _translate(seq: str, genetic_code: int) -> str:
    from Bio.Seq import Seq  # type: ignore

    protein = str(Seq(seq).translate(table=genetic_code))
    return protein.replace("*", "X")


def _codon_align(
    records: list[tuple[str, str]], genetic_code: int, method: str
) -> tuple[list[tuple[str, str]], str]:
    """Align CDS by aligning proteins with :func:`phylo.align` and back-translating."""
    import tempfile

    from .phylo import align

    proteins = [(name, _translate(seq, genetic_code)) for name, seq in records]
    with tempfile.TemporaryDirectory() as tmp:
        fasta = Path(tmp) / "proteins.fa"
        fasta.write_text("".join(f">{n}\n{p}\n" for n, p in proteins))
        result = align(fasta, method=method)
    aligned = dict(result.metrics["alignment"])
    aligner = str(result.metrics["method"])
    if aligner == "concat_placeholder":
        lengths = {len(p) for _, p in proteins}
        if len(lengths) != 1:
            raise OrganelleDependencyError(
                code="phylogeny.partition.no_aligner",
                message=(
                    "no multiple-sequence aligner available (Rust MAFFT port not built and "
                    "no 'mafft' on PATH); install: conda install -c bioconda mafft"
                ),
            )
    out: list[tuple[str, str]] = []
    for name, nt in records:
        prot_aln = aligned[name]
        codons: list[str] = []
        pos = 0
        for residue in prot_aln:
            if residue == "-":
                codons.append("---")
            else:
                codons.append(nt[pos : pos + 3])
                pos += 3
        back = "".join(codons)
        if back.replace("-", "") != nt:
            raise OrganelleInputError(
                code="phylogeny.partition.backtranslation_mismatch",
                message=f"codon back-translation failed for {name}",
            )
        out.append((name, back))
    return out, aligner


def build_partitioned_supermatrix(
    genbank_paths: Sequence[str | Path],
    *,
    output_dir: str | Path,
    genes: Sequence[str] | None = None,
    taxon_names: Sequence[str] | None = None,
    codon_positions: bool = True,
    min_taxa_fraction: float = 1.0,
    min_codons: int = 30,
    max_length_deviation: float | None = 0.2,
    genetic_code: int = 11,
    align_method: str = "auto",
) -> OrganelleResult:
    """Build a codon-aligned, gene-partitioned CDS supermatrix.

    ``genes`` restricts/orders the gene set (default: every gene whose CDS is
    present in at least ``min_taxa_fraction`` of the genomes; the default 1.0
    is the strict shared-gene set of :func:`phylo.extract_shared_genes`).
    Taxa missing a gene get ``?`` for its columns. Each gene is aligned
    codon-aware (protein alignment via :func:`phylo.align`, back-translated),
    so codon position charsets stay in frame. Terminal stop codons and
    trailing partial codons are removed.

    ``max_length_deviation`` guards against partial or mis-annotated CDS
    (e.g. a missing exon of an intron-containing gene): a gene is dropped
    (reported under ``skipped_genes`` as ``length_outlier:...``) when any
    taxon's CDS length differs from the gene's median length by more than
    this fraction. ``None`` disables the check.

    Writes ``supermatrix.fasta``, ``partitions.nex`` (NEXUS sets block for
    IQ-TREE ``-p`` / MrBayes) and ``genes.tsv`` under ``output_dir``.
    """
    paths = [Path(p) for p in genbank_paths]
    params = {
        "genbank_paths": [str(p) for p in paths],
        "genes": list(genes) if genes else None,
        "taxon_names": list(taxon_names) if taxon_names else None,
        "codon_positions": codon_positions,
        "min_taxa_fraction": min_taxa_fraction,
        "min_codons": min_codons,
        "max_length_deviation": max_length_deviation,
        "genetic_code": genetic_code,
        "align_method": align_method,
    }
    if len(paths) < 4:
        return failed_result(
            "build_partitioned_supermatrix",
            summary_text="At least 4 annotated genomes are required for a tree.",
            code="phylogeny.partition.too_few_taxa",
            anomalies=["too_few_taxa"],
        )
    if taxon_names is not None and len(taxon_names) != len(paths):
        raise OrganelleInputError(
            code="phylogeny.partition.taxon_names_mismatch",
            message="taxon_names must have one entry per GenBank path",
        )
    taxa: list[str] = []
    per_taxon: list[dict[str, str]] = []
    for index, path in enumerate(paths):
        record, cds = _read_cds(path, genetic_code)
        name = _SAFE_NAME.sub("_", taxon_names[index]) if taxon_names else _taxon_name(record, path)
        taxa.append(name)
        per_taxon.append({g: _codon_clean(s) for g, s in cds.items()})
    if len(set(taxa)) != len(taxa):
        dupes = sorted({t for t in taxa if taxa.count(t) > 1})
        raise OrganelleInputError(
            code="phylogeny.partition.duplicate_taxa",
            message=f"duplicate taxon names {dupes}; pass taxon_names explicitly",
        )

    n = len(taxa)
    if genes:
        candidates = list(dict.fromkeys(genes))
    else:
        counts: dict[str, int] = {}
        for gmap in per_taxon:
            for gene in gmap:
                counts[gene] = counts.get(gene, 0) + 1
        candidates = sorted(g for g, c in counts.items() if c >= min_taxa_fraction * n)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    blocks: list[tuple[str, list[tuple[str, str]]]] = []
    skipped: dict[str, str] = {}
    aligner = ""
    for gene in candidates:
        present = [
            (taxa[i], per_taxon[i][gene])
            for i in range(n)
            if len(per_taxon[i].get(gene, "")) >= 3 * min_codons
        ]
        if len(present) < max(4, min_taxa_fraction * n):
            skipped[gene] = f"present_in_{len(present)}_of_{n}"
            continue
        if max_length_deviation is not None:
            lengths = sorted(len(seq) for _, seq in present)
            median = lengths[len(lengths) // 2]
            outliers = [
                taxon
                for taxon, seq in present
                if abs(len(seq) - median) > max_length_deviation * median
            ]
            if outliers:
                skipped[gene] = "length_outlier:" + ",".join(outliers)
                continue
        aligned, aligner = _codon_align(present, genetic_code, align_method)
        blocks.append((gene, aligned))
    if not blocks:
        return failed_result(
            "build_partitioned_supermatrix",
            summary_text="No gene passed the taxon-coverage and length filters.",
            code="phylogeny.partition.no_genes",
            anomalies=["no_genes"],
            details={"skipped": skipped},
        )

    concatenated = {t: [] for t in taxa}
    charsets: list[PartitionCharset] = []
    gene_rows: list[dict[str, Any]] = []
    start = 1
    for gene, aligned in blocks:
        length = len(aligned[0][1])
        seqs = dict(aligned)
        for taxon in taxa:
            concatenated[taxon].append(seqs.get(taxon, "?" * length))
        end = start + length - 1
        safe = _SAFE_NAME.sub("_", gene)
        if codon_positions:
            for pos in (1, 2, 3):
                charsets.append((f"{safe}_pos{pos}", f"{start + pos - 1}-{end}\\3"))
        else:
            charsets.append((safe, f"{start}-{end}"))
        gene_rows.append(
            {"gene": gene, "start": start, "end": end, "length": length, "n_taxa": len(seqs)}
        )
        start = end + 1
    total = start - 1

    fasta = output / "supermatrix.fasta"
    fasta.write_text("".join(f">{t}\n{''.join(concatenated[t])}\n" for t in taxa))
    nexus = output / "partitions.nex"
    nexus.write_text(
        "#nexus\nbegin sets;\n"
        + "".join(f"  charset {name} = {spec};\n" for name, spec in charsets)
        + "end;\n"
    )
    table = output / "genes.tsv"
    table.write_text(
        "gene\tstart\tend\tlength\tn_taxa\n"
        + "".join(
            f"{r['gene']}\t{r['start']}\t{r['end']}\t{r['length']}\t{r['n_taxa']}\n"
            for r in gene_rows
        )
    )
    artifacts = tuple(
        a
        for a in (
            artifact_for(fasta, kind="alignment", format="fasta"),
            artifact_for(nexus, kind="partition_scheme", format="nexus"),
            artifact_for(table, kind="table", format="tsv"),
        )
        if a is not None
    )
    return ok_result(
        "build_partitioned_supermatrix",
        metrics={
            "n_taxa": n,
            "taxa": taxa,
            "n_genes": len(blocks),
            "genes": [g for g, _ in blocks],
            "skipped_genes": skipped,
            "alignment_length": total,
            "n_partitions": len(charsets),
            "codon_positions": codon_positions,
            "aligner": aligner,
            "gene_table": gene_rows,
            "supermatrix_fasta": str(fasta),
            "partitions_nexus": str(nexus),
        },
        result_findings=findings(
            ("n_taxa", n), ("n_genes", len(blocks)), ("alignment_length", total)
        ),
        flags=("supermatrix_built",),
        artifacts=artifacts,
        summary_text=(
            f"{len(blocks)} genes x {n} taxa codon-aligned ({aligner}); "
            f"{total} bp in {len(charsets)} partitions."
        ),
        result_provenance=provenance(
            "build_partitioned_supermatrix", method=f"codon_align:{aligner}", parameters=params
        ),
    )


# ---------------------------------------------------------------------------
# IQ-TREE -> MrBayes model mapping
# ---------------------------------------------------------------------------

# nst per IQ-TREE base DNA model, and whether its base frequencies are equal.
_BASE_MODELS: dict[str, tuple[int, bool, bool]] = {
    # name: (MrBayes nst, equal_freqs, exact)
    "JC": (1, True, True),
    "JC69": (1, True, True),
    "F81": (1, False, True),
    "K2P": (2, True, True),
    "K80": (2, True, True),
    "HKY": (2, False, True),
    "HKY85": (2, False, True),
    "SYM": (6, True, True),
    "GTR": (6, False, True),
    # Not expressible in MrBayes: nearest nesting model (nst=6) is used.
    "TNE": (6, True, False),
    "TN": (6, False, False),
    "TRN": (6, False, False),
    "TN93": (6, False, False),
    "K3P": (6, True, False),
    "K81": (6, True, False),
    "K3PU": (6, False, False),
    "K81U": (6, False, False),
    "TPM2": (6, True, False),
    "TPM2U": (6, False, False),
    "TPM3": (6, True, False),
    "TPM3U": (6, False, False),
    "TIME": (6, True, False),
    "TIM": (6, False, False),
    "TIM2E": (6, True, False),
    "TIM2": (6, False, False),
    "TIM3E": (6, True, False),
    "TIM3": (6, False, False),
    "TVME": (6, True, False),
    "TVM": (6, False, False),
}


def iqtree_to_mrbayes_model(model: str) -> dict[str, Any]:
    """Map one IQ-TREE DNA model string to MrBayes ``lset``/``prset`` settings.

    Mapping rules (MrBayes 3.2 supports only nst=1/2/6, ``rates=equal|gamma|
    propinv|invgamma`` and a Dirichlet or fixed base-frequency prior):

    * Substitution matrix: JC/F81 -> ``nst=1``; K2P(K80)/HKY -> ``nst=2``;
      SYM/GTR -> ``nst=6``. These are exact. Every other IQ-TREE matrix
      (TN/TNe, K3P/K3Pu, TPM*, TIM*, TVM*) is nested in GTR, so it maps to
      ``nst=6`` and is flagged ``exact=False`` (MrBayes estimates the extra
      exchangeabilities instead of constraining them).
    * Base frequencies: models with equal frequencies (JC, K2P, SYM and the
      ``e``-suffixed TNe/TIMe/TIM2e/TIM3e/TVMe, K3P, TPM2, TPM3) or ``+FQ`` ->
      ``prset statefreqpr=fixed(equal)``; otherwise (``+F``, ``+FO`` or the
      unequal-frequency base models) -> ``statefreqpr=dirichlet(1,1,1,1)``.
      Note that IQ-TREE's ``F81``/``HKY``/``GTR`` carry empirical unequal
      frequencies even when ``+F`` is not printed (verified against the
      explicit ``+F{...}``/``+FQ`` terms of ``*.best_model.nex``).
    * Rate heterogeneity: ``+G``/``+Gk`` -> ``rates=gamma ngammacat=k``
      (default 4); ``+I`` -> ``rates=propinv``; ``+I+G`` -> ``invgamma``. The
      FreeRate ``+Rk`` has no MrBayes equivalent and is approximated by
      ``gamma`` (``+I+Rk`` -> ``invgamma``) with ``exact=False``.
    * ``+ASC`` (ascertainment bias) and non-DNA models are rejected.

    Running ModelFinder with ``-mset mrbayes -mrate E,I,G,I+G`` (the default
    of :func:`select_partition_scheme`) restricts the candidates to
    JC/F81/K2P/HKY/SYM/GTR x {none,+I,+G,+I+G}, so every mapping is exact
    (``-mset`` alone still admits FreeRate ``+R``).
    """
    parts = [p for p in model.strip().split("+") if p]
    if not parts:
        raise OrganelleInputError(
            code="phylogeny.mrbayes.bad_model", message=f"empty model string {model!r}"
        )
    base = parts[0].upper()
    base = base.split("{", 1)[0]
    if base not in _BASE_MODELS:
        raise OrganelleInputError(
            code="phylogeny.mrbayes.unsupported_model",
            message=f"IQ-TREE model {model!r} has no MrBayes DNA equivalent",
        )
    nst, equal_freqs, exact = _BASE_MODELS[base]
    notes: list[str] = []
    if not exact:
        notes.append(f"{parts[0]} not in MrBayes; nested in GTR -> nst=6")
    has_i = False
    gamma_cats: int | None = None
    freerate = False
    for mod in parts[1:]:
        key = mod.upper().split("{", 1)[0]
        if key == "F" or key == "FO":
            equal_freqs = False
        elif key == "FQ":
            equal_freqs = True
        elif key == "I":
            has_i = True
        elif key.startswith("G"):
            gamma_cats = int(key[1:]) if key[1:].isdigit() else 4
        elif key.startswith("R"):
            freerate = True
            gamma_cats = int(key[1:]) if key[1:].isdigit() else 4
            exact = False
            notes.append(f"FreeRate +{mod} approximated by discrete gamma")
        elif key.startswith("ASC"):
            raise OrganelleInputError(
                code="phylogeny.mrbayes.unsupported_model",
                message=f"{model!r}: +ASC is not supported by MrBayes lset",
            )
        else:
            raise OrganelleInputError(
                code="phylogeny.mrbayes.unsupported_model",
                message=f"{model!r}: unsupported modifier +{mod}",
            )
    if gamma_cats is not None and has_i:
        rates = "invgamma"
    elif gamma_cats is not None:
        rates = "gamma"
    elif has_i:
        rates = "propinv"
    else:
        rates = "equal"
    return {
        "iqtree_model": model,
        "nst": nst,
        "rates": rates,
        "ngammacat": gamma_cats or 4,
        "statefreqpr": "fixed(equal)" if equal_freqs else "dirichlet(1,1,1,1)",
        "exact": exact,
        "freerate_approximated": freerate,
        "note": "; ".join(notes),
    }


# ---------------------------------------------------------------------------
# IQ-TREE partition scheme / model selection
# ---------------------------------------------------------------------------


def resolve_iqtree(iqtree_bin: str | None = None) -> str | None:
    """Locate IQ-TREE: explicit path, ``ORGANELLEVERSE_IQTREE_BIN``, PATH."""
    for candidate in (iqtree_bin, os.environ.get("ORGANELLEVERSE_IQTREE_BIN")):
        if candidate:
            found = shutil.which(candidate) or (
                candidate if Path(candidate).is_file() and os.access(candidate, os.X_OK) else None
            )
            if found:
                return found
    for name in ("iqtree3", "iqtree2", "iqtree"):
        found = shutil.which(name)
        if found:
            return found
    return None


def _parse_iqtree_report(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {}
    if not path.is_file():
        return info
    text = path.read_text(errors="replace")
    for key, pattern in (
        ("log_likelihood", r"Log-likelihood of the tree:\s*(-?[\d.]+)"),
        ("bic", r"Bayesian information criterion \(BIC\) score:\s*(-?[\d.]+)"),
        ("aicc", r"Corrected Akaike information criterion \(AICc\) score:\s*(-?[\d.]+)"),
    ):
        match = re.search(pattern, text)
        if match:
            info[key] = float(match.group(1))
    match = re.search(r"IQ-TREE (?:multicore )?version (\S+)", text)
    if match:
        info["iqtree_version"] = match.group(1)
    return info


def select_partition_scheme(
    alignment_fasta: str | Path,
    partition_nexus: str | Path,
    *,
    output_dir: str | Path,
    merge: bool = True,
    model_set: str | None = "mrbayes",
    rcluster: int = 100,
    rcluster_fast: bool = True,
    rcluster_max: int | None = None,
    compare_codon_positions: bool = False,
    bootstrap: int = 0,
    seed: int = 42,
    threads: int = 1,
    iqtree_bin: str | None = None,
    timeout: float | None = None,
    dry_run: bool = False,
) -> OrganelleResult:
    """Select the best partition scheme and per-partition models with IQ-TREE.

    Runs ``iqtree -s <aln> -p <partitions> -m MFP+MERGE [-mset mrbayes
    -mrate E,I,G,I+G] -rclusterf <pct> --prefix <out>/scheme --seed <seed> -T <threads>
    [-B <bootstrap>]``. ``-p`` is IQ-TREE's edge-proportional partition model
    (shared topology and branch lengths, one rate multiplier per partition),
    which matches MrBayes ``prset ratepr=variable`` with linked branch lengths.
    ``merge=False`` uses ``-m MFP`` (models per input partition, no merging).
    ``MFP`` continues to a full ML tree search on the selected scheme, so the
    result also carries the partitioned ML tree (with UFBoot support when
    ``bootstrap >= 1000``).

    Fast relaxed clustering (PartitionFinder2; Lanfear et al. 2017) is the
    default, considering 100% of pairs up to IQ-TREE's candidate cap.
    ``rcluster_fast=False`` restores ordinary ``-rcluster``;
    ``rcluster_max`` sets the maximum number of candidate pairs (``None``
    leaves IQ-TREE's default of max(1000, 10 * number of subsets)). These
    searches are approximate, and increasing the percentage need not improve BIC.

    For an in-frame CDS alignment starting at codon position 1, explicitly
    set ``compare_codon_positions=True`` to fit three fixed position subsets
    with the same model set, seed and edge-proportional model, without merging
    or bootstrapping. All alignment sites must be covered exactly once by the
    input scheme. ``metrics["codon_position_comparison"]["delta_bic"]`` is
    selected-scheme BIC minus the three-position BIC (negative favors the
    selected scheme). The comparison is reported, never substituted for the
    selected scheme. It costs an additional model fit and ML tree search.

    Outputs (under ``output_dir``): ``scheme.best_scheme.nex`` (NEXUS sets
    block with the merged charsets and a ``charpartition`` of models - the
    input of :func:`run_mrbayes`), ``models.tsv`` (partition, IQ-TREE model,
    sites, merged subsets, MrBayes lset/prset) and ``scheme.treefile``.
    """
    alignment = Path(alignment_fasta)
    parts_path = Path(partition_nexus)
    out = Path(output_dir)
    params = {
        "alignment_fasta": str(alignment),
        "partition_nexus": str(parts_path),
        "merge": merge,
        "model_set": model_set,
        "rcluster": rcluster,
        "rcluster_fast": rcluster_fast,
        "rcluster_max": rcluster_max,
        "compare_codon_positions": compare_codon_positions,
        "bootstrap": bootstrap,
        "seed": seed,
        "threads": threads,
    }
    binary = resolve_iqtree(iqtree_bin)
    prefix = out / "scheme"
    argv = [
        binary or "iqtree3",
        "-s",
        str(alignment),
        "-p",
        str(parts_path),
        "-m",
        "MFP+MERGE" if merge else "MFP",
        "--prefix",
        str(prefix),
        "--seed",
        str(seed),
        "-T",
        str(threads),
    ]
    if model_set:
        argv += ["-mset", model_set]
    if model_set == "mrbayes":
        # -mset only restricts the substitution matrices; FreeRate (+R) would
        # still be tested, and MrBayes cannot express it.
        argv += ["-mrate", "E,I,G,I+G"]
    if merge:
        argv += ["-rclusterf" if rcluster_fast else "-rcluster", str(rcluster)]
        if rcluster_max is not None:
            argv += ["-rcluster-max", str(rcluster_max)]
    if bootstrap:
        argv += ["-B", str(bootstrap)]
    if dry_run:
        return ok_result(
            "select_partition_scheme",
            metrics={
                "argv": argv,
                "backend_found": binary is not None,
                "planned": True,
                "codon_position_comparison_planned": compare_codon_positions,
            },
            flags=("scheme_planned",) + (() if binary else ("backend_missing",)),
            summary_text="IQ-TREE partition scheme selection planned."
            + ("" if binary else " " + IQTREE_INSTALL_HINT),
            result_provenance=provenance(
                "select_partition_scheme",
                method="iqtree",
                argv=argv,
                parameters=params,
                random_seed=seed,
            ),
        )
    if binary is None:
        raise OrganelleDependencyError(
            code="phylogeny.partition.iqtree_missing", message=IQTREE_INSTALL_HINT
        )
    input_sets = parse_nexus_sets(parts_path)
    input_sites = {name: frozenset(_charset_sites(spec)) for name, spec in input_sets["charsets"]}
    if compare_codon_positions:
        from .._bio import read_fasta

        lengths = {len(seq) for _, seq in read_fasta(alignment)}
        nchar = next(iter(lengths), 0)
        if len(lengths) != 1 or not nchar or nchar % 3:
            raise OrganelleInputError(
                code="phylogeny.partition.codon_alignment",
                message="codon-position comparison requires an in-frame alignment of equal lengths divisible by 3",
            )
        if (
            set().union(*input_sites.values()) != set(range(1, nchar + 1))
            or sum(map(len, input_sites.values())) != nchar
        ):
            raise OrganelleInputError(
                code="phylogeny.partition.comparison_sites",
                message="BIC comparison requires the input partitions to cover every alignment site exactly once",
            )
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    run_external(
        argv,
        cwd=out,
        timeout=timeout,
        tool="iqtree",
        code="phylogeny.partition.iqtree_failed",
    )
    runtime = time.monotonic() - started

    best_nex = Path(f"{prefix}.best_scheme.nex")
    if not best_nex.is_file():
        return failed_result(
            "select_partition_scheme",
            summary_text=f"IQ-TREE finished but {best_nex.name} is missing.",
            code="phylogeny.partition.no_best_scheme",
            anomalies=["missing_best_scheme"],
            result_provenance=provenance(
                "select_partition_scheme",
                method="iqtree",
                argv=argv,
                parameters=params,
                random_seed=seed,
            ),
        )
    scheme = parse_nexus_sets(best_nex)
    rows: list[dict[str, Any]] = []
    for name, spec in scheme["charsets"]:
        sites = frozenset(_charset_sites(spec))
        members = sorted(n for n, s in input_sites.items() if s and s <= sites)
        model = scheme["models"].get(name, "")
        row: dict[str, Any] = {
            "partition": name,
            "iqtree_model": model,
            "n_sites": len(sites),
            "n_subsets": len(members),
            "subsets": members,
        }
        if model:
            try:
                row["mrbayes"] = iqtree_to_mrbayes_model(model)
            except OrganelleInputError as error:
                row["mrbayes"] = {"error": error.message}
        rows.append(row)

    models_tsv = out / "models.tsv"
    models_tsv.write_text(
        "partition\tiqtree_model\tn_sites\tn_subsets\tmrbayes_nst\tmrbayes_rates\t"
        "mrbayes_statefreqpr\tmapping_exact\tsubsets\n"
        + "".join(
            f"{r['partition']}\t{r['iqtree_model']}\t{r['n_sites']}\t{r['n_subsets']}\t"
            f"{r.get('mrbayes', {}).get('nst', '')}\t{r.get('mrbayes', {}).get('rates', '')}\t"
            f"{r.get('mrbayes', {}).get('statefreqpr', '')}\t"
            f"{r.get('mrbayes', {}).get('exact', '')}\t{','.join(r['subsets'])}\n"
            for r in rows
        )
    )
    report = _parse_iqtree_report(Path(f"{prefix}.iqtree"))
    comparison_metrics: dict[str, Any] = {}
    comparison_artifacts = ()
    comparison_summary = ""
    if compare_codon_positions:
        baseline_input = out / "codon_positions.nex"
        baseline_input.write_text(
            "#nexus\nbegin sets;\n"
            + "".join(f"  charset pos{pos} = {pos}-{nchar}\\3;\n" for pos in (1, 2, 3))
            + "end;\n"
        )
        baseline = select_partition_scheme(
            alignment,
            baseline_input,
            output_dir=out / "codon_position_baseline",
            merge=False,
            model_set=model_set,
            seed=seed,
            threads=threads,
            iqtree_bin=binary,
            timeout=timeout,
        )
        if baseline.status != "ok" or "bic" not in baseline.metrics or "bic" not in report:
            return failed_result(
                "select_partition_scheme",
                code="phylogeny.partition.comparison_failed",
                summary_text="IQ-TREE did not produce both BIC scores for the requested comparison.",
                details={
                    "baseline_status": baseline.status,
                    "baseline_summary": baseline.summary_text,
                },
            )
        delta = round(report["bic"] - baseline.metrics["bic"], 4)
        comparison_metrics["codon_position_comparison"] = {
            "n_partitions": baseline.metrics["n_partitions"],
            "bic": baseline.metrics["bic"],
            "delta_bic": delta,
            "preferred": "selected" if delta < 0 else "codon_positions" if delta > 0 else "tie",
            "runtime_seconds": baseline.metrics["runtime_seconds"],
            "argv": baseline.metrics["argv"],
            "best_scheme_nexus": baseline.metrics["best_scheme_nexus"],
        }
        comparison_artifacts = baseline.artifacts
        comparison_summary = (
            f" BIC: selected {report['bic']:.4f}, three codon positions "
            f"{baseline.metrics['bic']:.4f}; delta={delta:+.4f} (lower is better)."
        )
    treefile = Path(f"{prefix}.treefile")
    artifacts = tuple(
        a
        for a in (
            artifact_for(best_nex, kind="partition_scheme", format="nexus"),
            artifact_for(Path(f"{prefix}.best_model.nex"), kind="partition_scheme", format="nexus"),
            artifact_for(models_tsv, kind="table", format="tsv"),
            artifact_for(treefile, kind="phylogenetic_tree", format="newick"),
            artifact_for(Path(f"{prefix}.iqtree"), kind="report", format="text"),
        )
        if a is not None
    ) + comparison_artifacts
    return ok_result(
        "select_partition_scheme",
        metrics={
            "n_input_partitions": len(input_sets["charsets"]),
            "n_partitions": len(rows),
            "partitions": rows,
            "merge": merge,
            "model_set": model_set,
            "all_mappings_exact": all(r.get("mrbayes", {}).get("exact") is True for r in rows),
            "best_scheme_nexus": str(best_nex),
            "models_tsv": str(models_tsv),
            "treefile": str(treefile) if treefile.is_file() else None,
            "runtime_seconds": round(runtime, 2),
            "total_runtime_seconds": round(time.monotonic() - started, 2),
            "argv": argv,
            **report,
            **comparison_metrics,
        },
        result_findings=findings(
            ("n_input_partitions", len(input_sets["charsets"])), ("n_partitions", len(rows))
        ),
        flags=("partition_scheme_selected",),
        artifacts=artifacts,
        summary_text=(
            f"IQ-TREE {'MFP+MERGE' if merge else 'MFP'}: {len(input_sets['charsets'])} input "
            f"partitions -> {len(rows)} partitions ({runtime:.0f} s)." + comparison_summary
        ),
        result_provenance=provenance(
            "select_partition_scheme",
            method="iqtree",
            argv=argv,
            parameters=params,
            random_seed=seed,
        ),
    )
