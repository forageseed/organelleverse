"""Organelle pangenome construction — self-contained core.

build_graph(): run a verified minigraph/pggb pangenome-graph build using each
  tool's real CLI (minigraph -xggs to stdout, spawned through the controlled
  _runner; pggb -i/-o/-n/-s/-p) + a self-contained k-mer presence matrix.
classify_sequences(): classify each sequence as core / variable / private
  based on presence across genomes (k-mer overlap >= overlap_threshold).
gene_pav(): gene presence/absence variation matrix (pure Python).
pan_repeats(): shared repeat detection across genomes (k-mer based).
"""

from __future__ import annotations

import shutil
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from .._bio import read_fasta
from ..annotation.genbank import parse_genbank
from ..core.artifacts import ArtifactRef
from ..core.errors import OrganelleInputError
from ..core.external import failure_message
from ..core.genome import OrganelleGenome
from ..core.result import ErrorDetail, OrganelleResult
from ..quality_control.contracts import GfaSummary
from ..quality_control.static import validate_gfa
from ._contract import (
    OPERATION_VERSION,
    annotation_path,
    artifact_from_path,
    finding,
    input_artifact_hashes,
    input_object_ids,
    make_provenance,
    result_scope,
    sequence_path,
    utc_now,
)
from ._runner import CommandRecord, run_command
from .backend_evidence import record_backend_version
from .graph import load_gfa
from .pggb_outputs import related_artifacts
from .project import PangenomeProject
from .tuning import SEGMENT_POLICY_SOURCE, protected_segment_length


def build_graph(
    genomes: list[OrganelleGenome],
    *,
    output_dir: str | Path,
    method: str = "minigraph",
    k: int = 31,
    pantools_memory_mb: int = 4096,
    threads: int = 8,
    n_haplotypes: int | None = None,
    segment_length: int = 5000,
    reference_index: int = 0,
    identity: float = 90.0,
    executor: Callable[[list[str]], Any] | None = None,
) -> OrganelleResult:
    """Build a pangenome graph.

    Per-method CLI (each backend's real interface):

    - **minigraph** (default): ``minigraph -xggs -t <T> <ref.fa> <sample.fa...>``
      spawned through the controlled runner (``_runner.run_command``), which
      streams the backend's **stdout** into ``out/pangenome.gfa`` — no shell
      redirection token ever appears in ``argv``. The first genome is the
      reference (minigraph is reference-based). Success requires exit code 0
      plus a structurally valid, non-empty GFA with at least one segment;
      only then is ``graph_built`` reported.
    - **pggb**: ``pggb -i <pansn.fa> -o <out/pggb> -n <n_haplotypes>
      -s <segment_length> -p <identity> -t <T>``. Reference-free all-vs-all
      (wfmash+seqwish+smoothxg): every record of every genome is rewritten
      into one PAN-SN FASTA (``sample#1#record`` headers, sample normalized
      from accession-or-species with a unique fallback), the percent identity
      is passed as a numeric ``-p`` (default 90), and the
      ``*.smooth.final.gfa`` deposited under the pggb output tree is
      discovered and validated afterwards.
    The process boundary is private: builds spawn through
    ``_runner.run_command`` (bound into this module's namespace), so tests
    replace ``organelleverse.pangenome.pangenome.run_command`` instead of
    threading an injection parameter through the public signature.
    ``executor`` is retained only for source compatibility and is never
    invoked. Backends without a verified builder fail closed.
    """
    started_at = utc_now()
    parameters: dict[str, Any] = {
        "output_dir": str(output_dir),
        "method": method,
        "k": k,
        **({"pantools_memory_mb": pantools_memory_mb} if method == "pantools" else {}),
        "threads": threads,
        "n_haplotypes": n_haplotypes,
        "segment_length": segment_length,
        "reference_index": reference_index,
        "identity": identity,
        "executor_supplied": executor is not None,
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fastas = [str(path) for path in (sequence_path(g) for g in genomes) if path is not None]
    if not fastas:
        return OrganelleResult(
            operation_id="pangenome.build_graph",
            operation_version=OPERATION_VERSION,
            scope=result_scope(genomes),
            status="failed",
            summary_text="build_graph requires >=1 genome with a sequence artifact.",
            errors=(
                ErrorDetail(
                    code="pangenome.no_sequences",
                    message="build_graph requires at least one genome carrying a sequence.",
                    details={"genome_count": len(genomes)},
                ),
            ),
            provenance=make_provenance(
                operation_id="pangenome.build_graph",
                parameters=parameters,
                input_object_ids=input_object_ids(genomes),
                input_artifact_hashes=input_artifact_hashes(genomes),
                started_at=started_at,
                finished_at=utc_now(),
            ),
        )

    if method not in {"minigraph", "pggb", "pantools"}:
        return _graph_failure(
            genomes,
            code="pangenome.unsupported_backend",
            message=f"Unsupported or unverified pangenome backend: {method}.",
            details={"method": method, "supported": ["minigraph", "pggb", "pantools"]},
            parameters=parameters,
            argv=[],
            started_at=started_at,
            backend=method,
        )

    invalid_parameter: tuple[str, Any, str] | None = None
    if k < 1:
        invalid_parameter = ("k", k, "must be >= 1")
    elif threads < 1:
        invalid_parameter = ("threads", threads, "must be >= 1")
    elif n_haplotypes is not None and n_haplotypes < 1:
        invalid_parameter = ("n_haplotypes", n_haplotypes, "must be >= 1 when supplied")
    elif segment_length < 1:
        invalid_parameter = ("segment_length", segment_length, "must be >= 1")
    elif not 0 < identity <= 100:
        invalid_parameter = ("identity", identity, "must be > 0 and <= 100")
    elif method == "minigraph" and not 0 <= reference_index < len(fastas):
        invalid_parameter = (
            "reference_index",
            reference_index,
            f"must be between 0 and {len(fastas) - 1}",
        )
    if invalid_parameter is not None:
        name, value, constraint = invalid_parameter
        return _graph_failure(
            genomes,
            code="pangenome.invalid_parameter",
            message=f"Invalid {name}={value!r}: {constraint}.",
            details={"parameter": name, "value": value, "constraint": constraint},
            parameters=parameters,
            argv=[],
            started_at=started_at,
            backend=method,
        )

    if method == "pggb":
        parameters["requested_segment_length"] = segment_length
        segment_length = protected_segment_length(genomes, segment_length)
        parameters["segment_length"] = segment_length
        parameters["plastid_floor_source"] = SEGMENT_POLICY_SOURCE

    # ── Auto-locate backend ───────────────────────────────────────────
    from .install import check_backend, install_hint

    loc = check_backend(method, scan_envs=True)
    backend_path = loc.get("path") if loc.get("installed") else None
    backend_env = loc.get("env")
    warning = ""
    if not backend_path:
        warning = "\n  ⚠ " + install_hint(method)
    bin_name = backend_path or method

    if method == "pantools":
        from .pantools_builder import build_pantools

        try:
            return build_pantools(
                genomes,
                out,
                executable=bin_name,
                k=k,
                threads=threads,
                memory_mb=pantools_memory_mb,
            )
        except (OSError, ValueError, RuntimeError) as error:
            return _graph_failure(
                genomes,
                code="pangenome.pantools_build_failed",
                message=str(error),
                details={"reason": str(error)},
                parameters=parameters,
                argv=[],
                started_at=started_at,
                backend="pantools",
            )

    # ── Per-method argv ───────────────────────────────────────────────
    gfa_out = str(out / "pangenome.gfa")
    if method == "minigraph":
        # Real execution through the controlled runner: -xggs graph-build preset,
        # reference first then samples, GFA streamed from stdout into the
        # staging file (never a shell ">" token).
        ref = fastas[reference_index]
        samples = [f for i, f in enumerate(fastas) if i != reference_index]
        argv = [bin_name, "-xggs", "-t", str(threads), ref, *samples]
        result = _build_minigraph_graph(
            genomes,
            fastas=fastas,
            argv=argv,
            gfa_out=Path(gfa_out),
            out=out,
            k=k,
            parameters=parameters,
            started_at=started_at,
            backend_env=backend_env,
            warning=warning,
            runner=run_command,
        )
        return record_backend_version(result, bin_name, out, run_command)
    if method == "pggb":
        # pggb is reference-free: every record of every genome goes into one
        # deterministic PAN-SN FASTA, and its output tree lives in a private
        # subdirectory so the discovered GFA cannot collide with staging files.
        staged_project = PangenomeProject.from_genomes(genomes).materialize(out / "pggb-input")
        pansn_fa = Path(staged_project.fasta_path)
        pggb_out = out / "pggb"
        pggb_out.mkdir(parents=True, exist_ok=True)
        samtools = _companion_executable("samtools", bin_name)
        index_argv = [samtools, "faidx", str(pansn_fa)]
        nhap = n_haplotypes or len(fastas)
        argv = [
            bin_name,
            "-i",
            str(pansn_fa),
            "-o",
            str(pggb_out),
            "-n",
            str(nhap),
            "-s",
            str(segment_length),
            "-p",
            f"{identity:g}",
            "-t",
            str(threads),
        ]
        result = _build_pggb_graph(
            genomes,
            fastas=fastas,
            index_argv=index_argv,
            argv=argv,
            pggb_out=pggb_out,
            k=k,
            parameters=parameters,
            started_at=started_at,
            backend_env=backend_env,
            warning=warning,
            runner=run_command,
        )
        return record_backend_version(result, bin_name, out, run_command)
    raise AssertionError(f"validated backend was not dispatched: {method}")


def classify_sequences(
    genomes: list[OrganelleGenome],
    *,
    k: int = 31,
    core_threshold: float = 0.9,
    overlap_threshold: float = 0.5,
) -> OrganelleResult:
    """Classify each contig/sequence as core / variable / private.

    A sequence is considered 'present' in another genome when its k-mer
    overlap with that genome is >= ``overlap_threshold`` (default 0.5). It is
    'core' if present in >= ``core_threshold`` fraction of the *other*
    genomes, 'private' if present in none of the others, else 'variable'.
    This sequence-level k-mer criterion defaults to 0.9 and excludes the focal
    genome. Graph node PAV instead counts all samples (or paths), defaults to
    strict 1.0, and supports an explicit soft core threshold such as 0.95.
    """
    started_at = utc_now()
    all_seqs = []
    for g in genomes:
        path = sequence_path(g)
        if path is not None:
            all_seqs.append([(sid, s) for sid, s in read_fasta(path)])
    classifications = []
    n_genomes = len(all_seqs)
    for gi, genome_seqs in enumerate(all_seqs):
        for sid, seq in genome_seqs:
            presence = _sequence_presence(seq, all_seqs, gi, k, overlap_threshold)
            frac = presence / max(1, n_genomes - 1)
            label = (
                "core" if frac >= core_threshold else ("private" if presence == 0 else "variable")
            )
            classifications.append(
                {"genome": gi, "seqid": sid, "label": label, "presence": presence}
            )
    core = sum(1 for c in classifications if c["label"] == "core")
    var = sum(1 for c in classifications if c["label"] == "variable")
    priv = sum(1 for c in classifications if c["label"] == "private")
    return OrganelleResult(
        operation_id="pangenome.classify_sequences",
        operation_version=OPERATION_VERSION,
        scope=result_scope(genomes),
        status="ok",
        summary_text=f"core={core}, variable={var}, private={priv}.",
        metrics={
            "core": core,
            "variable": var,
            "private": priv,
            "total": len(classifications),
            "core_threshold": core_threshold,
            "overlap_threshold": overlap_threshold,
            "frequency_denominator": "other_genomes",
            "presence_definition": "k-mer overlap >= overlap_threshold",
        },
        findings=(
            finding("core", core, unit="sequences"),
            finding("variable", var, unit="sequences"),
            finding("private", priv, unit="sequences"),
        ),
        provenance=make_provenance(
            operation_id="pangenome.classify_sequences",
            parameters={
                "k": k,
                "core_threshold": core_threshold,
                "overlap_threshold": overlap_threshold,
            },
            input_object_ids=input_object_ids(genomes),
            input_artifact_hashes=input_artifact_hashes(genomes),
            actual_backend="organelleverse",
            attempted_backends=("organelleverse",),
            started_at=started_at,
            finished_at=utc_now(),
        ),
    )


def gene_pav(
    genomes: list[OrganelleGenome],
) -> OrganelleResult:
    """Presence of named GenBank CDS features across supplied annotations.

    This legacy operation selects CDS features only (not gene/RNA features),
    using the lowercased first /gene qualifier. Missing annotation or unnamed
    CDS features fail explicitly. A zero records no supplied named CDS, never
    independently verified biological absence. Core, shell and cloud are
    mutually exclusive; a one-genome input contains core genes only.

    Matrix columns retain species metadata labels (or g{i}); write_pav writes
    the TSV. For gene-feature/copy-aware graph analysis use annotation_projection.
    """
    started_at = utc_now()
    metrics = _gene_pav_metrics(genomes)
    core, shell, cloud = metrics["core"], metrics["shell"], metrics["cloud"]
    return OrganelleResult(
        operation_id="pangenome.gene_pav",
        operation_version=OPERATION_VERSION,
        scope=result_scope(genomes),
        status="ok",
        summary_text=f"PAV: core={core}, shell={shell}, cloud={cloud}.",
        metrics=metrics,
        findings=(
            finding("core_genes", core, unit="genes"),
            finding("shell_genes", shell, unit="genes"),
            finding("cloud_genes", cloud, unit="genes"),
        ),
        provenance=make_provenance(
            operation_id="pangenome.gene_pav",
            parameters={},
            input_object_ids=input_object_ids(genomes),
            input_artifact_hashes=input_artifact_hashes(genomes),
            actual_backend="organelleverse",
            attempted_backends=("organelleverse",),
            started_at=started_at,
            finished_at=utc_now(),
        ),
    )


def write_pav(
    result: OrganelleResult | Mapping[str, Any],
    output: str | Path,
) -> Path:
    """Write a gene presence/absence matrix from ``gene_pav()``.

    Accepts either a canonical result (whose ``metrics`` carry the matrix) or the
    plain mapping returned by :func:`~organelleverse.pangenome.compute_gene_pav`.
    """
    metrics: Mapping[str, Any] = result.metrics if isinstance(result, OrganelleResult) else result
    matrix = dict(metrics.get("matrix", {}))
    accessions = list(metrics.get("accessions", ()))
    path = _resolve_output_path(output, "pav.tsv")
    lines = ["gene\t" + "\t".join(str(a) for a in accessions)]
    for gene in sorted(matrix):
        lines.append(str(gene) + "\t" + "\t".join(map(str, matrix[gene])))
    path.write_text("\n".join(lines) + "\n")
    return path


def pan_repeats(
    genomes: list[OrganelleGenome],
    *,
    k: int = 21,
    min_shared: int = 2,
) -> OrganelleResult:
    """Detect repeats shared across genomes (k-mer based)."""
    started_at = utc_now()
    kmer_sets = []
    for g in genomes:
        path = sequence_path(g)
        if path is not None:
            seq = "".join(s.upper() for _, s in read_fasta(path))
            kmer_sets.append(
                {seq[i : i + k] for i in range(len(seq) - k + 1) if "N" not in seq[i : i + k]}
            )
    # repeats = kmers appearing multiple times within AND across genomes
    from collections import Counter

    counter: Counter = Counter()
    for ks in kmer_sets:
        for km in ks:
            counter[km] += 1
    shared = {km: c for km, c in counter.items() if c >= min_shared}
    return OrganelleResult(
        operation_id="pangenome.pan_repeats",
        operation_version=OPERATION_VERSION,
        scope=result_scope(genomes),
        status="ok",
        summary_text=f"{len(shared)} shared repeat k-mers (k={k}) across {len(genomes)} genomes.",
        metrics={"shared_kmers": len(shared), "k": k},
        findings=(finding("shared_repeat_kmers", len(shared), unit="kmers"),),
        provenance=make_provenance(
            operation_id="pangenome.pan_repeats",
            parameters={"k": k, "min_shared": min_shared},
            input_object_ids=input_object_ids(genomes),
            input_artifact_hashes=input_artifact_hashes(genomes),
            actual_backend="organelleverse",
            attempted_backends=("organelleverse",),
            started_at=started_at,
            finished_at=utc_now(),
        ),
    )


# -- helpers --------------------------------------------------------------


def _build_minigraph_graph(
    genomes: list[OrganelleGenome],
    *,
    fastas: list[str],
    argv: list[str],
    gfa_out: Path,
    out: Path,
    k: int,
    parameters: dict[str, Any],
    started_at: datetime,
    backend_env: str,
    warning: str,
    runner: Callable[..., CommandRecord],
) -> OrganelleResult:
    """Run minigraph through the controlled runner and publish a validated GFA.

    Fail closed on spawn errors, non-zero exit, or a missing/empty/invalid
    GFA; ``graph_built`` is only reported once the produced graph carries at
    least one segment.
    """
    try:
        record = runner(list(argv), stdout_path=str(gfa_out), cwd=str(out))
    except OSError as exc:
        return _graph_failure(
            genomes,
            code="pangenome.backend_spawn_failed",
            message=f"minigraph failed to start: {exc}{warning}",
            details={"argv": argv, "error": str(exc)},
            parameters=parameters,
            argv=argv,
            started_at=started_at,
        )
    if record.returncode != 0:
        return _graph_failure(
            genomes,
            code="pangenome.backend_failed",
            message=f"minigraph exited with code {record.returncode}.",
            details={
                "argv": argv,
                "returncode": record.returncode,
                "resource_usage": record.resource_usage,
                "stderr": record.stderr,
            },
            parameters=parameters,
            argv=argv,
            started_at=started_at,
        )
    summary = _validated_gfa(gfa_out)
    if summary is None:
        return _graph_failure(
            genomes,
            code="pangenome.invalid_gfa",
            message=f"minigraph did not produce a valid GFA at {gfa_out}.",
            details={"argv": argv, "path": str(gfa_out)},
            parameters=parameters,
            argv=argv,
            started_at=started_at,
        )
    clusters = _kmer_clusters([read_fasta(Path(f)) for f in fastas], k)
    graph_artifact = artifact_from_path(
        gfa_out,
        kind="pangenome_graph",
        format="gfa",
        media_type="text/plain",
    )
    artifacts: tuple[ArtifactRef, ...] = (graph_artifact,) if graph_artifact is not None else ()
    return OrganelleResult(
        operation_id="pangenome.build_graph",
        operation_version=OPERATION_VERSION,
        scope=result_scope(genomes),
        status="ok",
        summary_text=(
            f"Pangenome graph (minigraph); built by minigraph, "
            f"{summary.segment_count} segments, {len(clusters)} k-mer clusters."
        )
        + warning,
        metrics={
            "method": "minigraph",
            "kmer_clusters": len(clusters),
            "genome_count": len(fastas),
            "gfa_segments": summary.segment_count,
            "gfa_edges": summary.edge_count,
            "backend_env": backend_env,
            "output_path": str(gfa_out),
            "returncode": record.returncode,
            "resource_usage": record.resource_usage,
        },
        findings=(
            finding("method", "minigraph"),
            finding("gfa_segments", summary.segment_count, unit="segments"),
            finding("kmer_clusters", len(clusters), unit="clusters"),
            *((finding("backend_env", backend_env),) if backend_env else ()),
        ),
        flags=("graph_built",),
        artifacts=artifacts,
        provenance=make_provenance(
            operation_id="pangenome.build_graph",
            parameters=parameters,
            input_object_ids=input_object_ids(genomes),
            input_artifact_hashes=input_artifact_hashes(genomes),
            requested_backend="minigraph",
            actual_backend="minigraph",
            attempted_backends=("minigraph",),
            argv=tuple(str(item) for item in argv),
            started_at=started_at,
            finished_at=utc_now(),
        ),
    )


def _build_pggb_graph(
    genomes: list[OrganelleGenome],
    *,
    fastas: list[str],
    index_argv: list[str],
    argv: list[str],
    pggb_out: Path,
    k: int,
    parameters: dict[str, Any],
    started_at: datetime,
    backend_env: str,
    warning: str,
    runner: Callable[..., CommandRecord],
) -> OrganelleResult:
    """Run pggb through the controlled runner and publish the smoothed GFA.

    pggb writes its whole output tree itself (no stdout handover): the runner
    spawns with the pggb output directory as ``cwd``, and success requires
    exit 0 plus exactly one discoverable ``*.smooth.final.gfa`` that validates.
    """
    index_log = pggb_out / "faidx.stdout.log"
    try:
        index_record = runner(
            list(index_argv),
            stdout_path=str(index_log),
            cwd=str(pggb_out),
        )
    except OSError as exc:
        return _graph_failure(
            genomes,
            code="pangenome.index_spawn_failed",
            message=f"samtools faidx failed to start: {exc}",
            details={"argv": index_argv, "error": str(exc)},
            parameters=parameters,
            argv=index_argv,
            started_at=started_at,
            backend="pggb",
        )
    fasta_index = Path(f"{argv[argv.index('-i') + 1]}.fai")
    if index_record.returncode != 0 or not fasta_index.is_file() or fasta_index.stat().st_size == 0:
        return _graph_failure(
            genomes,
            code="pangenome.index_failed",
            message="samtools faidx did not produce a usable PGGB input index.",
            details={
                "argv": index_argv,
                "returncode": index_record.returncode,
                "resource_usage": index_record.resource_usage,
                "stderr": index_record.stderr,
                "stdout_tail": _read_text_tail(index_record.stdout_path),
                "path": str(fasta_index),
            },
            parameters=parameters,
            argv=index_argv,
            started_at=started_at,
            backend="pggb",
        )
    try:
        record = runner(
            list(argv),
            stdout_path=str(pggb_out / "pggb.stdout.log"),
            cwd=str(pggb_out),
        )
    except OSError as exc:
        return _graph_failure(
            genomes,
            code="pangenome.backend_spawn_failed",
            message=f"pggb failed to start: {exc}{warning}",
            details={"argv": argv, "error": str(exc)},
            parameters=parameters,
            argv=argv,
            started_at=started_at,
            backend="pggb",
        )
    if record.returncode != 0:
        return _graph_failure(
            genomes,
            code="pangenome.backend_failed",
            message=failure_message("pggb", record.returncode, record.stderr),
            details={
                "argv": argv,
                "returncode": record.returncode,
                "resource_usage": record.resource_usage,
                "stderr": record.stderr,
                "stdout_tail": _read_text_tail(record.stdout_path),
            },
            parameters=parameters,
            argv=argv,
            started_at=started_at,
            backend="pggb",
        )
    candidates = sorted(pggb_out.rglob("*.smooth.final.gfa"))
    if not candidates:
        return _graph_failure(
            genomes,
            code="pangenome.final_gfa_missing",
            message=f"pggb produced no *.smooth.final.gfa under {pggb_out}.",
            details={"argv": argv, "search_root": str(pggb_out)},
            parameters=parameters,
            argv=argv,
            started_at=started_at,
            backend="pggb",
        )
    if len(candidates) > 1:
        listed = ", ".join(str(match) for match in candidates)
        return _graph_failure(
            genomes,
            code="pangenome.final_gfa_ambiguous",
            message=(
                f"pggb left {len(candidates)} *.smooth.final.gfa files under "
                f"{pggb_out}; expected exactly one: {listed}"
            ),
            details={
                "argv": argv,
                "search_root": str(pggb_out),
                "matches": [str(match) for match in candidates],
            },
            parameters=parameters,
            argv=argv,
            started_at=started_at,
            backend="pggb",
        )
    final_gfa = candidates[0]
    summary = _validated_gfa(final_gfa)
    if summary is None:
        return _graph_failure(
            genomes,
            code="pangenome.invalid_gfa",
            message=f"pggb did not produce a valid GFA at {final_gfa}.",
            details={"argv": argv, "path": str(final_gfa)},
            parameters=parameters,
            argv=argv,
            started_at=started_at,
            backend="pggb",
        )
    try:
        related, related_validation = related_artifacts(pggb_out)
    except (OrganelleInputError, OSError, UnicodeError, ValueError, EOFError) as error:
        return _graph_failure(
            genomes,
            code="pangenome.invalid_related_artifact",
            message=f"PGGB produced an invalid related artifact: {error}",
            details={"reason": str(error)},
            parameters=parameters,
            argv=argv,
            started_at=started_at,
            backend="pggb",
        )
    clusters = _kmer_clusters([read_fasta(Path(f)) for f in fastas], k)
    graph_artifact = artifact_from_path(
        final_gfa,
        kind="pangenome_graph",
        format="gfa",
        media_type="text/plain",
    )
    artifacts: tuple[ArtifactRef, ...] = (
        (graph_artifact,) if graph_artifact is not None else ()
    ) + related
    return OrganelleResult(
        operation_id="pangenome.build_graph",
        operation_version=OPERATION_VERSION,
        scope=result_scope(genomes),
        status="ok",
        summary_text=(
            f"Pangenome graph (pggb); built by pggb, "
            f"{summary.segment_count} segments, {len(clusters)} k-mer clusters."
        )
        + warning,
        metrics={
            "method": "pggb",
            "segment_length": parameters["segment_length"],
            "requested_segment_length": parameters["requested_segment_length"],
            "segment_length_policy_source": parameters["plastid_floor_source"],
            "related_artifact_validation": related_validation,
            "kmer_clusters": len(clusters),
            "genome_count": len(fastas),
            "gfa_segments": summary.segment_count,
            "gfa_edges": summary.edge_count,
            "backend_env": backend_env,
            "output_path": str(final_gfa),
            "returncode": record.returncode,
            "resource_usage": record.resource_usage,
        },
        findings=(
            finding("method", "pggb"),
            finding("gfa_segments", summary.segment_count, unit="segments"),
            finding("kmer_clusters", len(clusters), unit="clusters"),
            *((finding("backend_env", backend_env),) if backend_env else ()),
        ),
        flags=("graph_built",),
        artifacts=artifacts,
        provenance=make_provenance(
            operation_id="pangenome.build_graph",
            parameters=parameters,
            input_object_ids=input_object_ids(genomes),
            input_artifact_hashes=input_artifact_hashes(genomes),
            requested_backend="pggb",
            actual_backend="pggb",
            attempted_backends=("pggb",),
            argv=tuple(str(item) for item in argv),
            started_at=started_at,
            finished_at=utc_now(),
        ),
    )


def _graph_failure(
    genomes: list[OrganelleGenome],
    *,
    code: str,
    message: str,
    details: dict[str, Any],
    parameters: dict[str, Any],
    argv: list[str],
    started_at: datetime,
    backend: str = "minigraph",
) -> OrganelleResult:
    """Assemble the fail-closed result for one graph-build attempt."""
    return OrganelleResult(
        operation_id="pangenome.build_graph",
        operation_version=OPERATION_VERSION,
        scope=result_scope(genomes),
        status="failed",
        summary_text=message,
        errors=(ErrorDetail(code=code, message=message, details=details),),
        provenance=make_provenance(
            operation_id="pangenome.build_graph",
            parameters=parameters,
            input_object_ids=input_object_ids(genomes),
            input_artifact_hashes=input_artifact_hashes(genomes),
            requested_backend=backend,
            actual_backend="",
            attempted_backends=(backend,),
            argv=tuple(str(item) for item in argv),
            started_at=started_at,
            finished_at=utc_now(),
        ),
    )


def _validated_gfa(path: Path) -> GfaSummary | None:
    """Return the GfaSummary of a usable graph file, or ``None`` when invalid.

    Reuses the QC parser: the file must be a non-empty regular file whose
    records parse cleanly and which carries at least one segment. A read that
    fails at the I/O or decoding layer (missing/unreadable file, non-UTF-8
    bytes) is an unusable graph, not an exception for the caller to handle.
    """
    if not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        load_gfa(path)
        summary = validate_gfa(path)
    except (OrganelleInputError, OSError, UnicodeError, ValueError):
        return None
    return summary if summary.segment_count >= 1 else None


def _read_text_tail(path: str | None, *, max_bytes: int = 16_384) -> str:
    """Read a bounded UTF-8 diagnostic tail from a process log."""
    if path is None:
        return ""
    try:
        target = Path(path)
        with target.open("rb") as handle:
            size = target.stat().st_size
            handle.seek(max(0, size - max_bytes))
            return handle.read(max_bytes).decode("utf-8", errors="replace")
    except OSError:
        return ""


def _companion_executable(name: str, backend: str) -> str:
    """Prefer a real sibling tool, then PATH, without assuming Conda layout."""
    executable = Path(backend).expanduser()
    if executable.is_absolute():
        sibling = executable.with_name(name)
        if sibling.is_file():
            return str(sibling)
    return shutil.which(name) or name


def _resolve_output_path(output: str | Path, default_name: str) -> Path:
    path = Path(output)
    if path.suffix:
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    path.mkdir(parents=True, exist_ok=True)
    return path / default_name


def _kmer_clusters(genome_seqs: list[list[tuple[str, str]]], k: int) -> list[set[str]]:
    """Cluster sequences by shared k-mer content (proxy for graph components).

    Uses fixpoint merge: repeatedly merge overlapping clusters until stable,
    so transitively-connected sequences end up in one cluster.
    """
    seq_kmers = []
    for recs in genome_seqs:
        for _, seq in recs:
            seq_up = seq.upper()
            seq_kmers.append(
                {
                    seq_up[i : i + k]
                    for i in range(len(seq_up) - k + 1)
                    if "N" not in seq_up[i : i + k]
                }
            )
    clusters: list[set[str]] = [set(kms) for kms in seq_kmers]
    changed = True
    while changed:
        changed = False
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                if clusters[i] and clusters[j] and (clusters[i] & clusters[j]):
                    clusters[i] |= clusters[j]
                    clusters[j] = set()
                    changed = True
        clusters = [c for c in clusters if c]
    return clusters


def _sequence_presence(
    seq: str,
    all_seqs: list[list[tuple[str, str]]],
    self_idx: int,
    k: int,
    overlap_threshold: float = 0.5,
) -> int:
    """Count other genomes where this sequence's k-mer overlap >= threshold."""
    kmers = {
        seq.upper()[i : i + k] for i in range(len(seq) - k + 1) if "N" not in seq.upper()[i : i + k]
    }
    presence = 0
    for gi, recs in enumerate(all_seqs):
        if gi == self_idx:
            continue
        for _, s in recs:
            s_up = s.upper()
            other = {
                s_up[i : i + k] for i in range(len(s_up) - k + 1) if "N" not in s_up[i : i + k]
            }
            if len(kmers & other) / max(1, len(kmers)) >= overlap_threshold:
                presence += 1
                break
    return presence


def _gene_pav_metrics(genomes: list[OrganelleGenome]) -> dict[str, Any]:
    """One definition shared by canonical and legacy dict PAV entry points."""
    if not genomes:
        raise OrganelleInputError(
            code="pangenome.empty_inputs",
            message="Gene PAV requires at least one annotated genome.",
        )
    gene_sets = [_gene_set_from_genome(genome) for genome in genomes]
    all_genes = sorted(set.union(*gene_sets))
    matrix = {gene: [int(gene in genes) for genes in gene_sets] for gene in all_genes}
    counts = {gene: sum(row) for gene, row in matrix.items()}
    classes = {
        gene: "core" if count == len(genomes) else "cloud" if count == 1 else "shell"
        for gene, count in counts.items()
    }
    return {
        "core": sum(label == "core" for label in classes.values()),
        "shell": sum(label == "shell" for label in classes.values()),
        "cloud": sum(label == "cloud" for label in classes.values()),
        "total_genes": len(all_genes),
        "accessions": [
            genome.metadata.species or f"g{index}" for index, genome in enumerate(genomes)
        ],
        "matrix": matrix,
        "counts": counts,
        "classes": classes,
        "feature_type": "CDS",
        "identifier_policy": "first GenBank /gene qualifier, lowercased",
        "classification": {
            "core": "count = number of supplied genomes",
            "shell": "1 < count < number of supplied genomes",
            "cloud": "count = 1 and count < number of supplied genomes",
        },
        "absence_interpretation": (
            "Zero means no supplied named CDS annotation, not verified biological absence. "
            "Annotation completeness and homology are not inferred."
        ),
    }


def _gene_set_from_genome(g: OrganelleGenome) -> set[str]:
    path = annotation_path(g)
    if path is None:
        raise OrganelleInputError(
            code="pangenome.missing_annotation",
            message="Every genome must supply a GenBank annotation for gene PAV.",
            details={"sample": g.metadata.accession or g.metadata.species or g.object_id},
        )
    document = parse_genbank(path)
    genes: set[str] = set()
    for record in document.records:
        for feature in record.features:
            if feature.type.casefold() != "cds":
                continue
            values = feature.qualifier_values("gene")
            if not values or not values[0].strip():
                raise OrganelleInputError(
                    code="pangenome.unnamed_cds",
                    message="Every selected CDS requires a /gene qualifier for gene PAV.",
                    details={"annotation": str(path)},
                )
            genes.add(values[0].lower())
    return genes
