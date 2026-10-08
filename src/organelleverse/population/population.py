"""Population genetics + cytonuclear interaction.

- call_variants(): plan a deepvariant/gatk/bcftools command (executor hook).
  Each backend's real CLI is built (REF + per-BAM HaplotypeCaller / mpileup|call
  / run_deepvariant). Organelles are haploid, so ``ploidy`` defaults to 1 and is
  spelled the way each backend spells it (``bcftools call --ploidy``,
  ``gatk HaplotypeCaller --sample-ploidy``). DeepVariant calls diploid genotypes
  only and has no ploidy switch, so its plan carries no ploidy flag and is
  flagged ``deepvariant_diploid_only`` — for haploid organelles use bcftools or
  gatk.
- fst_scan(): compute F_ST between populations from a VCF. Parsing uses vcfpy
  (the standard Python VCF parser); per-site F_ST uses the simplified
  heterozygosity partition ``(H_T - H_S) / H_T``.
- detect_numt(): NUMT detection by local alignment of the mitochondrion against
  every nuclear record (LOSAT, then NCBI blastn; both strands). Fragments carry
  ``seqid``, 1-based coordinates inside that record, ``strand`` and ``identity``.
  Only without any aligner does it fall back to the exact k-mer scan (flag
  ``kmer_fallback``), which matches the *concatenation* of the nuclear records:
  those fragments are mapped back onto the record they sit in, and one crossing
  the artificial join between two records is clipped at that boundary (and
  flagged) instead of being reported as one impossible interval.
- prepare_gemma_input(): convert VCF → GEMMA dosage file (bcftools norm → plink
  make-bed → plink recode bimbam → dosage rewrite), the only genotype format
  GEMMA accepts.
- cytonuclear_gwas(): full GEMMA pipeline (VCF→BIMBAM preprocessing + kinship
  via ``-gk`` then LMM via ``-k -lmm``), or just the GEMMA steps when the caller
  supplies a ready dosage file.

Every entry point returns the canonical frozen :class:`organelleverse.core.OrganelleResult`.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from ..core.artifacts import ArtifactRef
from ..core.provenance import ResultProvenance
from ..core.frozen import FrozenMap
from ..core.result import ErrorDetail, Finding, OrganelleResult

_OPERATION_VERSION = "1.0"
_SCOPE = "mitochondrion"


# -- canonical contract helpers -------------------------------------------


def _package_version() -> str:
    try:
        return version("organelleverse")
    except PackageNotFoundError:
        return "0.0.1"


def _parameters_hash(parameters: dict[str, Any]) -> str:
    payload = json.dumps(
        parameters,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _provenance(
    operation: str,
    *,
    parameters: dict[str, Any],
    method: str = "",
    argv: tuple[str, ...] = (),
) -> ResultProvenance:
    """Build canonical provenance for one ``population.*`` operation."""
    return ResultProvenance(
        operation_id=f"population.{operation}",
        operation_version=_OPERATION_VERSION,
        package_version=_package_version(),
        git_commit=os.environ.get("ORG_VERSE_GIT_COMMIT", ""),
        parameters_hash=_parameters_hash(parameters),
        requested_backend=method,
        actual_backend=method,
        attempted_backends=(method,) if method else (),
        argv=argv,
    )


def _failed(
    operation: str,
    *,
    code: str,
    message: str,
    parameters: dict[str, Any],
    method: str = "",
) -> OrganelleResult:
    """Canonical failed result (a failed result must carry at least one error)."""
    return OrganelleResult(
        operation_id=f"population.{operation}",
        operation_version=_OPERATION_VERSION,
        scope=_SCOPE,
        status="failed",
        summary_text=message,
        provenance=_provenance(operation, parameters=parameters, method=method),
        errors=(ErrorDetail(code=code, message=message),),
    )


def call_variants(
    bam_dir: str | Path,
    *,
    ref_path: str | Path | None = None,
    method: str = "deepvariant",
    output_dir: str | Path | None = None,
    executor: Callable[[list[str]], Any] | None = None,
    ploidy: int = 1,
) -> OrganelleResult:
    """Plan/run population variant calling (deepvariant/gatk/bcftools).

    Builds the real per-tool CLI:

    - ``deepvariant``: ``run_deepvariant --model_type=WGS --ref=<ref>
      --reads=<bam> --output_vcf=<out.vcf>`` (one invocation per BAM; needs
      REF). DeepVariant only calls diploid genotypes and has no ploidy switch
      (only ``--haploid_contigs`` for X/Y), so no ploidy flag is emitted and a
      ``ploidy`` other than 2 sets the ``deepvariant_diploid_only`` flag. For
      haploid organelles use bcftools or gatk.
    - ``gatk``: ``gatk HaplotypeCaller -R <ref> -I <bam> -O <out.g.vcf>
      -OVT GP --sample-ploidy <ploidy>`` (per-sample gVCF; GATK best-practices).
    - ``bcftools``: ``bcftools mpileup -f <ref> <bam> | bcftools call -mv
      --ploidy <ploidy> -Oz -o <out.vcf>`` (pileup + call; needs REF).

    ``ploidy`` defaults to 1 because organelles are haploid: with the callers'
    diploid default every heterozygous organelle site comes back as an
    uncallable ``0/1`` genotype.

    ``ref_path`` is required for all three backends (a reference FASTA). When
    omitted the function still returns a plan but flags it as incomplete.
    Auto-locates the backend (PATH + all conda envs); if not found, returns
    plan-only with an install hint (GitHub/bioconda link). An ``executor`` that
    raises turns the result into ``status="failed"`` — a silent ``planned``
    would hide a broken run behind an ok flag.
    """
    tool = {"deepvariant": "deepvariant", "gatk": "gatk", "bcftools": "bcftools"}.get(
        method, "deepvariant"
    )

    # ── Auto-locate backend ───────────────────────────────────────────
    from .install import check_backend, install_hint

    loc = check_backend(method, scan_envs=True)
    backend_path = loc.get("path") if loc.get("installed") else None
    backend_env = loc.get("env")
    warning = ""
    if not backend_path:
        warning = "\n  ⚠ " + install_hint(method)
    bin_name = backend_path or tool

    out = str(output_dir or ".") + "/"
    ref = str(ref_path) if ref_path else "<REF.fasta>"
    ref_missing = ref_path is None

    # Build the real per-tool argv. ``bam_dir`` is expanded to the first BAM
    # for the plan; the executor layer is expected to iterate over all BAMs.
    bam = str(Path(bam_dir) / "<sample>.bam")
    if method == "deepvariant":
        argv = [
            bin_name,
            "--model_type=WGS",
            f"--ref={ref}",
            f"--reads={bam}",
            f"--output_vcf={out}<sample>.vcf.gz",
        ]
    elif method == "gatk":
        argv = [
            bin_name,
            "HaplotypeCaller",
            "-R",
            ref,
            "-I",
            bam,
            "-O",
            f"{out}<sample>.g.vcf.gz",
            "-OVT",
            "GP",
            "--sample-ploidy",
            str(ploidy),
        ]
    else:  # bcftools: mpileup | call (two-stage pipeline)
        argv = [
            bin_name,
            "mpileup",
            "-f",
            ref,
            bam,
            "|",
            bin_name,
            "call",
            "-mv",
            "--ploidy",
            str(ploidy),
            "-Oz",
            "-o",
            f"{out}<sample>.vcf.gz",
        ]

    ran = False
    error: ErrorDetail | None = None
    if executor and backend_path and not ref_missing:
        try:
            executor(list(argv))
            ran = True
        except Exception as exc:
            error = ErrorDetail(
                code="population.call_variants.executor_failed",
                message=f"{method} variant calling failed: {exc}",
            )
    flags = []
    if error is not None:
        flags.append("execution_failed")
    elif ran:
        flags.append("variants_called")
    else:
        flags.append("call_planned")
    if ref_missing:
        flags.append("missing_reference")
    ploidy_applied = method != "deepvariant"
    if not ploidy_applied and ploidy != 2:
        flags.append("deepvariant_diploid_only")
    state = "failed" if error is not None else ("ran" if ran else "planned")
    return OrganelleResult(
        operation_id="population.call_variants",
        operation_version=_OPERATION_VERSION,
        scope=_SCOPE,
        status="failed" if error is not None else "ok",
        summary_text=(
            f"Variant calling ({method}, ploidy {ploidy}); {state}."
            + (f"\n  ✗ {error.message}" if error is not None else "")
            + (" ⚠ no ref_path given." if ref_missing else "")
            + warning
        ),
        metrics={
            "method": method,
            "ploidy": ploidy,
            "ploidy_applied": ploidy_applied,
            "backend_found": backend_path is not None,
            "backend_env": backend_env,
            "reference_provided": not ref_missing,
            "argv": list(argv),
        },
        findings=(Finding(code="population.call_variants.method", metric="method", value=method),),
        flags=tuple(flags),
        artifacts=(),
        errors=(error,) if error is not None else (),
        provenance=_provenance(
            "call_variants",
            parameters={
                "bam_dir": str(bam_dir),
                "ref_path": str(ref_path) if ref_path is not None else None,
                "method": method,
                "ploidy": ploidy,
            },
            method=method,
            argv=tuple(argv),
        ),
    )


def fst_scan(
    vcf_path: str | Path,
    *,
    pop_assignments: dict[str, str],
    window_size: int = 1000,
    step: int = 500,
    top_fraction: float = 0.05,
    estimator: str = "heterozygosity",
) -> OrganelleResult:
    """Compute windowed F_ST between two populations from a VCF.

    ``estimator`` selects the per-site statistic (``heterozygosity``, the default
    ``(H_T - H_S) / H_T`` clamped at 0; ``weir_cockerham``, the Weir & Cockerham
    (1984) theta as vcftools/scikit-allel compute it; or ``hudson``, the Hudson
    (1992) estimator on allele counts). A window is the mean of its per-site
    values (>= 3 SNPs). The estimators do not give interchangeable numbers; see
    ``docs/operations/population-fst-estimators.md``.

    ``pop_assignments`` maps sample_id -> 'pop1' or 'pop2'. Only concrete
    biallelic SNPs enter the scan (a symbolic ALT such as ``*``, which spans
    the upstream deletion, is not an allele and is skipped). Samples named in
    ``pop_assignments`` but absent from the VCF are reported in
    ``samples_missing_from_vcf`` — silently dropping them shrinks a population
    (a one-letter typo then moves F_ST for the whole window).
    """
    parameters = {
        "vcf_path": str(vcf_path),
        "pop_assignments": dict(pop_assignments),
        "window_size": window_size,
        "step": step,
        "top_fraction": top_fraction,
        "estimator": estimator,
    }
    if estimator not in FST_ESTIMATORS:
        return _failed(
            "fst_scan",
            code="population.fst_scan.unsupported_estimator",
            message=f"Unsupported F_ST estimator {estimator!r}; choose one of "
            + ", ".join(sorted(FST_ESTIMATORS))
            + ".",
            parameters=parameters,
            method="organelleverse",
        )
    site_fst = FST_ESTIMATORS[estimator]
    snps = _parse_vcf_biallelic(vcf_path)
    if not snps:
        return _failed(
            "fst_scan",
            code="population.fst_scan.no_snps",
            message="No biallelic SNPs found.",
            parameters=parameters,
            method="organelleverse",
        )
    pop1 = [s for s, p in pop_assignments.items() if p == "pop1"]
    pop2 = [s for s, p in pop_assignments.items() if p == "pop2"]
    if not pop1 or not pop2:
        return _failed(
            "fst_scan",
            code="population.fst_scan.population_setup",
            message="fst_scan needs samples in both populations.",
            parameters=parameters,
            method="organelleverse",
        )
    vcf_samples: set[str] = set()
    for calls in snps.values():
        vcf_samples.update(calls)
    missing_samples = sorted(s for s in pop_assignments if s not in vcf_samples)
    positions = sorted(snps)
    windows = []
    for start in range(positions[0], positions[-1], step):
        end = start + window_size
        win_snps = [p for p in positions if start <= p < end]
        if len(win_snps) < 3:
            continue
        fst_vals = [site_fst(snps[p], pop1, pop2) for p in win_snps]
        mean_fst = sum(fst_vals) / len(fst_vals)
        windows.append(
            {"start": start, "end": end, "fst": round(mean_fst, 4), "n_snps": len(win_snps)}
        )
    windows.sort(key=lambda w: w["fst"], reverse=True)
    top_n = max(1, int(len(windows) * top_fraction)) if windows else 0
    top_windows = windows[:top_n]
    summary = (
        f"{len(windows)} windows; top {len(top_windows)} F_ST candidates "
        f"(max={windows[0]['fst']:.3f})."
        if windows
        else f"No windows with ≥3 biallelic SNPs (step {step}, window {window_size})."
    )
    if missing_samples:
        summary += (
            " ⚠ pop_assignments samples absent from the VCF: "
            + ", ".join(missing_samples)
            + "."
        )
    flags = []
    if top_windows:
        flags.append("selection_candidates")
    if missing_samples:
        flags.append("population_samples_missing_from_vcf")
    return OrganelleResult(
        operation_id="population.fst_scan",
        operation_version=_OPERATION_VERSION,
        scope=_SCOPE,
        status="ok",
        summary_text=summary,
        metrics={
            "windows": len(windows),
            "top_candidates": len(top_windows),
            "max_fst": windows[0]["fst"] if windows else 0,
            "top_windows": list(top_windows[:5]),
            "n_snps": len(snps),
            "estimator": estimator,
            "population_sizes": {"pop1": len(pop1), "pop2": len(pop2)},
            "samples_missing_from_vcf": missing_samples,
        },
        findings=tuple(
            Finding(
                code="population.fst_scan.fst_window",
                metric=f"fst_window@{w['start']}",
                value=w["fst"],
            )
            for w in top_windows[:5]
        ),
        flags=tuple(flags),
        artifacts=(),
        provenance=_provenance("fst_scan", parameters=parameters, method="organelleverse"),
    )


def detect_numt(
    nuclear_fasta: str | Path,
    mito_fasta: str | Path,
    *,
    k: int = 31,
    min_len: int = 100,
    min_identity: float = 80.0,
) -> OrganelleResult:
    """Detect NUMTs (nuclear mitochondrial DNA segments) by local alignment.

    The mitochondrion is aligned against every nuclear record with LOSAT (NCBI
    ``blastn`` when LOSAT is unavailable); an HSP of >= ``min_len`` alignment
    columns and >= ``min_identity`` percent identity on either strand is a
    fragment, merged on nuclear coordinates. Each fragment carries ``seqid``,
    1-based ``start``/``end`` inside that nuclear record, ``strand`` and
    ``identity``; ``reference_records_scanned`` counts the nuclear records.

    The exact k-mer scan this replaced recovered 455,041 bp on the rice
    mitochondrion against the nuclear genome where alignment finds 729,140 bp
    (``k`` did not matter), so ``k`` is used only when neither aligner is
    installed: that last-resort scan
    sets the ``kmer_fallback`` flag, and its fragments (offsets into the
    concatenated records) are mapped back onto the record they sit in, with
    fragments crossing a record join clipped and flagged
    (``fragment_clipped_at_reference_record_boundary``; pieces shorter than
    ``min_len`` are dropped and counted in ``boundary_pieces_dropped``).
    """
    from ..transfer.transfer import _detect_transfer_alignment

    result = _detect_transfer_alignment(
        nuclear_fasta,
        mito_fasta,
        k=k,
        min_len=min_len,
        min_identity=min_identity,
        op="detect_numt",
        organelle="mito",
    )
    if result.status != "ok":
        return result
    if "kmer_fallback" in result.flags:
        return _map_concatenated_fragments(result, nuclear_fasta, min_len)
    metrics = dict(result.metrics)
    metrics["reference_records_scanned"] = metrics["target_records"]
    return result.model_copy(update={"metrics": FrozenMap(metrics)})


def _map_concatenated_fragments(
    result: OrganelleResult, nuclear_fasta: str | Path, min_len: int
) -> OrganelleResult:
    """Map k-mer fragments (offsets into the concatenated records) onto records."""
    from .._bio import read_fasta

    records = [(name, len(seq)) for name, seq in read_fasta(Path(nuclear_fasta))]
    bounds: list[tuple[str, int, int]] = []  # (seqid, first_pos, last_pos), 1-based
    offset = 0
    for name, length in records:
        bounds.append((name, offset + 1, offset + length))
        offset += length

    fragments: list[dict[str, Any]] = []
    clipped = False
    dropped = 0
    for fragment in result.metrics.get("fragments", ()):
        start, end = int(fragment["start"]), int(fragment["end"])
        strand = fragment.get("strand", "+")
        for seqid, first, last in bounds:
            lo, hi = max(start, first), min(end, last)
            if hi < lo:
                continue  # fragment does not touch this record
            if lo > start or hi < end:
                clipped = True  # truncated at an interior record boundary
            if hi - lo + 1 < min_len:
                dropped += 1
                continue
            fragments.append(
                {
                    "seqid": seqid,
                    "start": lo - first + 1,
                    "end": hi - first + 1,
                    "length": hi - lo + 1,
                    "strand": strand,
                }
            )

    metrics = dict(result.metrics)
    metrics["fragments"] = fragments
    metrics["reference_records_scanned"] = len(records)
    metrics["concatenation_length"] = offset
    metrics["boundary_pieces_dropped"] = dropped
    flags = tuple(result.flags)
    if clipped:
        flags += ("fragment_clipped_at_reference_record_boundary",)
    summary = (
        f"{len(fragments)} NUMT fragment(s) in {len(records)} nuclear record(s); "
        f"{metrics.get('total_transferred_bp', 0)} bp (exact k-mer fallback)."
        + (" ⚠ fragments clipped at record boundaries." if clipped else "")
    )
    return result.model_copy(
        update={"summary_text": summary, "metrics": FrozenMap(metrics), "flags": flags}
    )


def prepare_gemma_input(
    vcf_path: str | Path,
    *,
    phenotype: str | Path,
    out_dir: str | Path,
    executor: Callable[[list[str]], Any] | None = None,
) -> OrganelleResult:
    """Convert VCF → GEMMA dosage genotype file (GEMMA's required input).

    Standard pipeline (GEMMA issue #103 + PLINK 1.9 docs):

    0. ``bcftools norm -m -any <vcf> -Oz -o <out>/norm.vcf.gz``
       split multi-allelic records (GEMMA is biallelic-only).
    1. ``plink --vcf <norm.vcf.gz> --double-id --allow-extra-chr --make-bed
       --out <out>/data`` — VCF → PLINK binary (.bed/.bim/.fam).
    2. ``plink --bfile <out>/data --double-id --allow-extra-chr 0 --recode
       bimbam --out <out>/data_bimbam`` — PLINK → BIMBAM letter genotypes.
    3. :func:`_bimbam_to_gemma` — rewrite those letters as the numeric dosages
       GEMMA parses, at ``<out>/data.gemma.geno.txt``.

    ``--double-id`` matters because organelle accessions routinely contain
    ``_`` (plink exits 3 on a second ``_``) and ``--allow-extra-chr`` because
    organelle contig names are not 1..22/X (the recode step needs the ``0``
    modifier as well).

    Auto-locates bcftools + plink (PATH + all conda envs). Returns the argv
    steps and the GEMMA-ready genotype path; runs them sequentially when an
    ``executor`` is provided and both tools are installed. A failing step (or a
    failed dosage rewrite) yields ``status="failed"`` — reporting ``planned``
    after three steps already ran would hide a broken pipeline.
    """
    from .install import check_backend, install_hint

    out_dir = Path(out_dir)
    if executor is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    bcftools = check_backend("bcftools", scan_envs=True)
    plink = check_backend("plink", scan_envs=True)
    bcftools_bin = bcftools["path"] or "bcftools"
    plink_bin = plink["path"] or "plink"
    missing = []
    if not bcftools["installed"]:
        missing.append("bcftools")
    if not plink["installed"]:
        missing.append("plink")

    argv_steps, plink_geno, gemma_geno = _gemma_prep_steps(
        bcftools_bin, plink_bin, str(vcf_path), out_dir
    )

    ran = False
    error: ErrorDetail | None = None
    if executor and not missing:
        try:
            for argv in argv_steps:
                executor(list(argv))
            ran = True
        except Exception as exc:
            error = ErrorDetail(
                code="population.prepare_gemma_input.step_failed",
                message=f"VCF→BIMBAM step failed: {exc}",
            )
    geno_ready = False
    if ran and error is None:
        try:
            _bimbam_to_gemma(plink_geno, gemma_geno)
            geno_ready = Path(gemma_geno).is_file()
        except Exception as exc:
            error = ErrorDetail(
                code="population.prepare_gemma_input.bimbam_to_gemma_failed",
                message=f"rewriting {plink_geno} as GEMMA dosages failed: {exc}",
            )

    warning = ""
    if missing:
        warning = "\n  ⚠ missing: " + ", ".join(missing)
        for m in missing:
            warning += "\n" + install_hint(m)
    flags = []
    if error is not None:
        flags.append("preparation_failed")
    elif ran:
        flags.append("bimbam_prepared")
    else:
        flags.append("preparation_planned")
    if missing:
        flags.append("missing_backends")
    artifacts: tuple[ArtifactRef, ...] = ()
    if geno_ready:
        artifacts = (
            ArtifactRef.from_path(
                gemma_geno,
                kind="bimbam_genotype",
                format="bimbam",
                media_type="text/csv",
            ),
        )
    state = "failed" if error is not None else ("ran" if ran else "planned")
    return OrganelleResult(
        operation_id="population.prepare_gemma_input",
        operation_version=_OPERATION_VERSION,
        scope=_SCOPE,
        status="failed" if error is not None else "ok",
        summary_text=(
            f"VCF→GEMMA dosage conversion; {state} "
            f"({len(argv_steps)} argv steps + dosage rewrite)."
            + (f"\n  ✗ {error.message}" if error is not None else "")
            + warning
        ),
        metrics={
            "method": "bcftools+plink",
            "backends_found": {"bcftools": bcftools["installed"], "plink": plink["installed"]},
            "steps": [
                "bcftools_norm",
                "plink_make_bed",
                "plink_recode_bimbam",
                "bimbam_to_gemma",
            ],
            "plink_bimbam_path": plink_geno,
            "bimbam_geno_path": gemma_geno,
            "phenotype_path": str(phenotype),
            "argv_steps": argv_steps,
        },
        findings=(
            Finding(
                code="population.prepare_gemma_input.bimbam_geno_path",
                metric="bimbam_geno_path",
                value=gemma_geno,
            ),
        ),
        flags=tuple(flags),
        artifacts=artifacts,
        errors=(error,) if error is not None else (),
        provenance=_provenance(
            "prepare_gemma_input",
            parameters={
                "vcf_path": str(vcf_path),
                "phenotype": str(phenotype),
            },
            method="bcftools+plink",
            argv=tuple(" ".join(s) for s in argv_steps),
        ),
    )


def cytonuclear_gwas(
    vcf_path: str | Path,
    *,
    phenotype: str | Path,
    method: str = "gemma",
    output_prefix: str = "cyto_gwas",
    out_dir: str | Path = "gemma_out",
    executor: Callable[[list[str]], Any] | None = None,
    prepare_input: bool = True,
) -> OrganelleResult:
    """Plan/run a cytonuclear GWAS — the GEMMA pipeline as GEMMA runs it.

    Encapsulates the complete workflow (per the GEMMA manual and
    https://github.com/genetics-statistics/GEMMA/issues/103), including the
    VCF→dosage preprocessing that GEMMA requires (see
    :func:`prepare_gemma_input` for why each plink flag is there):

    0-2. **bcftools norm / plink make-bed / plink recode bimbam** — plus the
       dosage rewrite producing ``<out_dir>/data.gemma.geno.txt``.
    3. **gemma -gk 1 -outdir <out_dir> -o <prefix>_kinship** — centered
       relatedness matrix; GEMMA writes it to ``<out_dir>/<prefix>_kinship.cXX.txt``.
    4. **gemma -k <that file> -lmm 1 -outdir <out_dir> -o <prefix>_lmm** —
       univariate linear mixed model association.

    The ``-k`` path must be the ``.cXX.txt`` file GEMMA actually writes (the
    ``.sXX.txt`` name would point at a file that never exists), and ``-outdir``
    keeps every artifact inside the run directory instead of GEMMA's default
    ``./output``.

    When ``prepare_input=False`` the caller has already produced a BIMBAM
    dosage file; pass it as ``vcf_path`` and only steps 3-4 are planned.

    Auto-locates every backend (PATH + all conda envs); returns the full plan
    plus an install hint for whichever tools are missing. With ``executor``
    set and all tools present, runs the steps sequentially; a failing step
    yields ``status="failed"``.
    """
    from .install import check_backend, install_hint

    out_dir = Path(out_dir)
    if executor is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    # ── Resolve tool paths ────────────────────────────────────────────
    bcftools = check_backend("bcftools", scan_envs=True)
    plink = check_backend("plink", scan_envs=True)
    gemma = check_backend(method, scan_envs=True)
    bcftools_bin = bcftools["path"] or "bcftools"
    plink_bin = plink["path"] or "plink"
    gemma_bin = gemma["path"] or method
    missing = [
        n
        for n, s in (("bcftools", bcftools), ("plink", plink), (method, gemma))
        if not s["installed"] and (prepare_input or (n != "bcftools" and n != "plink"))
    ]

    # ── Build the argv steps ──────────────────────────────────────────
    argv_steps: list[list[str]] = []
    if prepare_input:
        prep_steps, plink_geno, gemma_geno = _gemma_prep_steps(
            bcftools_bin, plink_bin, str(vcf_path), out_dir
        )
        argv_steps += prep_steps
        geno_arg = gemma_geno
    else:
        plink_geno, gemma_geno = None, None
        geno_arg = str(vcf_path)  # caller-provided BIMBAM dosages
    kinship_prefix = f"{output_prefix}_kinship"
    lmm_prefix = f"{output_prefix}_lmm"
    kinship_path = str(out_dir / f"{kinship_prefix}.cXX.txt")
    argv_steps += [
        [
            gemma_bin,
            "-g",
            geno_arg,
            "-p",
            str(phenotype),
            "-gk",
            "1",
            "-outdir",
            str(out_dir),
            "-o",
            kinship_prefix,
        ],
        [
            gemma_bin,
            "-g",
            geno_arg,
            "-p",
            str(phenotype),
            "-k",
            kinship_path,
            "-lmm",
            "1",
            "-outdir",
            str(out_dir),
            "-o",
            lmm_prefix,
        ],
    ]

    # ── Execute if possible ───────────────────────────────────────────
    ran = False
    error: ErrorDetail | None = None
    if executor and not missing:
        try:
            n_prep = len(argv_steps) - 2  # the two GEMMA steps come last
            for i, argv in enumerate(argv_steps):
                executor(list(argv))
                if prepare_input and i == n_prep - 1:
                    # the GEMMA steps read the dosage file, so rewrite plink's
                    # letter output before reaching them
                    _bimbam_to_gemma(plink_geno, gemma_geno)
            ran = True
        except Exception as exc:
            error = ErrorDetail(
                code="population.cytonuclear_gwas.step_failed",
                message=f"cytonuclear GWAS step failed: {exc}",
            )

    warning = ""
    if missing:
        warning = "\n  ⚠ missing: " + ", ".join(missing)
        for m in missing:
            warning += "\n" + install_hint(m)

    step_names = [
        *(
            [
                "bcftools_norm",
                "plink_make_bed",
                "plink_recode_bimbam",
                "bimbam_to_gemma",
            ]
            if prepare_input
            else []
        ),
        "gemma_kinship_-gk",
        "gemma_lmm_-lmm",
    ]
    flags = []
    if error is not None:
        flags.append("gwas_failed")
    elif ran:
        flags.append("gwas_run")
    else:
        flags.append("gwas_planned")
    if missing:
        flags.append("missing_backends")
    state = "failed" if error is not None else ("ran" if ran else "planned")
    return OrganelleResult(
        operation_id="population.cytonuclear_gwas",
        operation_version=_OPERATION_VERSION,
        scope=_SCOPE,
        status="failed" if error is not None else "ok",
        summary_text=(
            f"Cytonuclear GWAS ({method}); {state} "
            f"({len(argv_steps)} argv steps: " + " → ".join(step_names) + ")."
        )
        + (f"\n  ✗ {error.message}" if error is not None else "")
        + warning,
        metrics={
            "method": method,
            "backends_found": {
                "bcftools": bcftools["installed"],
                "plink": plink["installed"],
                method: gemma["installed"],
            },
            "steps": step_names,
            "n_steps": len(argv_steps),
            "prepare_input": prepare_input,
            "geno_path": geno_arg,
            "kinship_path": kinship_path,
            "argv_steps": argv_steps,
        },
        findings=(
            Finding(code="population.cytonuclear_gwas.method", metric="method", value=method),
        ),
        flags=tuple(flags),
        artifacts=(),
        errors=(error,) if error is not None else (),
        provenance=_provenance(
            "cytonuclear_gwas",
            parameters={
                "vcf_path": str(vcf_path),
                "phenotype": str(phenotype),
                "method": method,
                "prepare_input": prepare_input,
            },
            method=method,
            argv=tuple(" ".join(s) for s in argv_steps),
        ),
    )


# -- helpers --------------------------------------------------------------


def _gemma_prep_steps(
    bcftools_bin: str,
    plink_bin: str,
    vcf_path: str,
    out_dir: Path,
) -> tuple[list[list[str]], str, str]:
    """Build the VCF → GEMMA-dosage preprocessing argv steps.

    Returns ``(steps, plink_bimbam_path, gemma_geno_path)``. ``plink --recode
    bimbam`` writes ``<prefix>.recode.geno.txt`` (not ``.geno.txt``), and that
    file still needs the dosage rewrite, so the path GEMMA must read is
    ``<out_dir>/data.gemma.geno.txt``.
    """
    norm_vcf = str(out_dir / "norm.vcf.gz")
    bed_prefix = str(out_dir / "data")
    bimbam_prefix = str(out_dir / "data_bimbam")
    plink_geno = bimbam_prefix + ".recode.geno.txt"
    gemma_geno = str(out_dir / "data.gemma.geno.txt")
    steps = [
        [bcftools_bin, "norm", "-m", "-any", vcf_path, "-Oz", "-o", norm_vcf],
        [
            plink_bin,
            "--vcf",
            norm_vcf,
            "--double-id",
            "--allow-extra-chr",
            "--make-bed",
            "--out",
            bed_prefix,
        ],
        [
            plink_bin,
            "--bfile",
            bed_prefix,
            "--double-id",
            "--allow-extra-chr",
            "0",
            "--recode",
            "bimbam",
            "--out",
            bimbam_prefix,
        ],
    ]
    return steps, plink_geno, gemma_geno


def _bimbam_to_gemma(src: str | Path, dst: str | Path) -> Path:
    """Rewrite ``plink --recode bimbam`` output as the dosages GEMMA parses.

    plink writes a 3-line header (two counts + ``IND,<sample names>``) followed
    by ``<snp_id>,<genotype>,...`` rows of *letters*; GEMMA's BIMBAM reader
    aborts on that file (``gemma_io.cpp:699``). The rewritten rows are
    ``<snp_id>,<allele1>,<allele2>,<dosage>,...`` with numeric dosages: the
    allele pair is the sorted set of ACGT bases observed in the row and the
    dosage is the count of ``allele2`` (0/1/2), so the coding does not depend
    on plink's per-variant allele choice. Missing calls (``NN``) become NA.
    Uniformly rescaling a dosage column leaves GEMMA's kinship and PVE
    unchanged (verified: 0/1 vs 0/2 coding scales K by exactly 4.0 and PVE is
    identical to 6 decimals), so any monotone 0/1/2 coding is equivalent here.
    """
    src_path, dst_path = Path(src), Path(dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    try:
        # streamed row by row: nuclear-scale genotype files do not fit in memory
        with (
            src_path.open(encoding="utf-8") as reader,
            dst_path.open("w", encoding="utf-8", newline="\n") as writer,
        ):
            for line in reader:
                fields = line.rstrip("\r\n").split(",")
                if len(fields) < 2 or fields[0] == "IND":
                    continue  # plink's 3-line header (counts + IND,<samples>)
                snp_id, genotypes = fields[0], fields[1:]
                bases = sorted({ch for g in genotypes for ch in g.upper() if ch in "ACGT"})
                if not bases:
                    raise ValueError(f"no ACGT allele in row {snp_id!r} of {src_path}")
                allele1, allele2 = bases[0], bases[-1]
                dosages = []
                for genotype in genotypes:
                    gt = genotype.upper()
                    if len(gt) == 2 and set(gt) <= set("ACGT"):
                        dosages.append(str(gt.count(allele2)))
                    else:
                        dosages.append("NA")
                writer.write(",".join([snp_id, allele1, allele2, *dosages]) + "\n")
                n_rows += 1
        if not n_rows:
            raise ValueError(f"no genotype rows found in {src_path}")
    except BaseException:
        dst_path.unlink(missing_ok=True)  # never leave a partial dosage file behind
        raise
    return dst_path


def _parse_vcf_biallelic(vcf_path: str | Path) -> dict[int, dict[str, str]]:
    """Parse a VCF and return {pos: {sample: genotype}} for biallelic SNPs.

    Uses vcfpy (the standard Python VCF parser; already a project dependency),
    so every VCF spec quirk (multi-allelic records, indels, phased/unphased GT,
    missing `./.`) is handled correctly. Only records whose single ALT is a
    concrete single-nucleotide ACGT allele are kept: a length-1 filter alone
    admits symbolic alleles such as ``*`` (the spanning deletion bcftools writes
    for a record upstream of an indel) or ``<NON_REF>``, which are not alleles
    and silently corrupt the population statistics computed from the result.
    """
    import vcfpy

    reader = vcfpy.Reader.from_path(str(vcf_path))
    try:
        snps: dict[int, dict[str, str]] = {}
        for record in reader:
            ref = str(record.REF).upper()
            if len(ref) != 1 or ref not in "ACGT":
                continue
            alt = record.ALT
            if len(alt) != 1:
                continue
            alt_value = str(alt[0].value).upper()
            if len(alt_value) != 1 or alt_value not in "ACGT":
                continue
            snps[int(record.POS)] = {
                str(call.sample): str(call.data.get("GT", "./.")) for call in record.calls
            }
    finally:
        reader.close()
    return snps


def _fst_at_site(geno: dict[str, str], pop1: list[str], pop2: list[str]) -> float:
    """Compute simplified heterozygosity-based F_ST at one biallelic site."""

    def alt_freq(samples):
        alleles = []
        for s in samples:
            gt = geno.get(s, "./.")
            alleles.extend(gt.replace("|", "/").split("/"))
        alts = [a for a in alleles if a in "01"]
        return sum(int(a) for a in alts) / len(alts) if alts else 0.0, len(alts)

    p1, n1 = alt_freq(pop1)
    p2, n2 = alt_freq(pop2)
    if n1 == 0 or n2 == 0:
        return 0.0
    n = n1 + n2
    pbar = (p1 * n1 + p2 * n2) / n
    h_total = 2 * pbar * (1 - pbar)
    if h_total == 0:
        return 0.0
    h_within = (n1 * 2 * p1 * (1 - p1) + n2 * 2 * p2 * (1 - p2)) / n
    if h_within >= h_total:
        return 0.0
    return (h_total - h_within) / h_total


def _called_alleles(gt: str) -> list[str]:
    """Called alleles ('0'/'1') of one GT string; [] unless 1 (haploid) or 2 calls."""
    alleles = gt.replace("|", "/").split("/")
    if len(alleles) not in (1, 2) or any(a not in ("0", "1") for a in alleles):
        return []
    return alleles


def _individual_stats(geno: dict[str, str], samples: list[str]) -> tuple[int, float, float]:
    """(individuals, alt allele frequency, heterozygote frequency) of one population.

    A haploid call counts as a homozygous diploid (what vcftools does with a
    haploid organelle VCF written as 0/0, 1/1); a missing or non-biallelic call
    drops the individual.
    """
    n = alt = het = 0
    for sample in samples:
        alleles = _called_alleles(geno.get(sample, "./."))
        if not alleles:
            continue
        n += 1
        if len(alleles) == 1:
            alt += 2 * int(alleles[0])
        else:
            copies = int(alleles[0]) + int(alleles[1])
            alt += copies
            het += copies == 1
    return (n, alt / (2 * n), het / n) if n else (0, 0.0, 0.0)


def _fst_weir_cockerham(geno: dict[str, str], pop1: list[str], pop2: list[str]) -> float:
    """Weir & Cockerham (1984) per-site theta = a / (a + b + c), two populations.

    Unclamped (negative values are legitimate); 0.0 when undefined (a population
    without calls, fewer than two individuals on average, or a zero denominator).
    """
    (n1, p1, h1), (n2, p2, h2) = _individual_stats(geno, pop1), _individual_stats(geno, pop2)
    if n1 == 0 or n2 == 0:
        return 0.0
    r = 2
    n_bar = (n1 + n2) / r
    if n_bar <= 1:
        return 0.0
    n_c = (r * n_bar - (n1**2 + n2**2) / (r * n_bar)) / (r - 1)
    p_bar = (n1 * p1 + n2 * p2) / (r * n_bar)
    s2 = (n1 * (p1 - p_bar) ** 2 + n2 * (p2 - p_bar) ** 2) / ((r - 1) * n_bar)
    h_bar = (n1 * h1 + n2 * h2) / (r * n_bar)
    a = n_bar / n_c * (s2 - (p_bar * (1 - p_bar) - (r - 1) / r * s2 - h_bar / 4) / (n_bar - 1))
    b = n_bar / (n_bar - 1) * (
        p_bar * (1 - p_bar) - (r - 1) / r * s2 - (2 * n_bar - 1) / (4 * n_bar) * h_bar
    )
    c = h_bar / 2
    denominator = a + b + c
    return a / denominator if denominator else 0.0


def _fst_hudson(geno: dict[str, str], pop1: list[str], pop2: list[str]) -> float:
    """Hudson (1992) per-site F_ST on allele counts (haploid calls count once).

    ``num / den`` with ``num = (p1 - p2)^2 - p1(1-p1)/(n1-1) - p2(1-p2)/(n2-1)`` and
    ``den = p1(1-p2) + p2(1-p1)``, as scikit-allel's ``hudson_fst``; unclamped, 0.0
    when undefined (fewer than two alleles in a population, or a monomorphic site).
    """
    freqs = []
    for samples in (pop1, pop2):
        alleles = [a for s in samples for a in _called_alleles(geno.get(s, "./."))]
        freqs.append((len(alleles), sum(map(int, alleles)) / len(alleles)) if alleles else (0, 0.0))
    (n1, p1), (n2, p2) = freqs
    if n1 < 2 or n2 < 2:
        return 0.0
    numerator = (p1 - p2) ** 2 - p1 * (1 - p1) / (n1 - 1) - p2 * (1 - p2) / (n2 - 1)
    denominator = p1 * (1 - p2) + p2 * (1 - p1)
    return numerator / denominator if denominator > 0 else 0.0


#: per-site F_ST estimators selectable with ``estimator=``
FST_ESTIMATORS = {
    "heterozygosity": _fst_at_site,
    "weir_cockerham": _fst_weir_cockerham,
    "hudson": _fst_hudson,
}
