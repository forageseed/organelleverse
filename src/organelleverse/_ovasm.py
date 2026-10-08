"""Resolution and invocation of the ``ovasm`` binary (https://github.com/forageseed/ovasm).

``ovasm recruit`` scans sequencing reads once and writes the reads of each seeded
organelle target (plus NUMT candidates and a JSON report); ``ovasm discover`` does
the same without seeds, by k-mer depth. Assembly backends then work on a small,
depth-normalised read set instead of the whole-genome library. ``ovasm components``
drops the connected components of an assembly graph that are not the target organelle
(for reads found by depth, which bring nuclear repeats along). ``ovasm evidence``
scores an assembly graph against long reads: support for every link, the minority
share at each branch point and the pairings across two-copy repeats. ``ovasm
linearize`` turns that report into one representative sequence through the
read-supported phased paths and says whether the reads single it out. ``ovasm unify``
rewrites any backend's graph as OV-GFA v1 (blunt links, PanSN paths, common tags) for
the pangenome module. ``ovasm run`` is the whole flow in one call: recruitment, assembly
with the k-ladder retry, evidence, linearization, unification and, in standard mode,
downsampled replicates.
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from .core.errors import OrganelleDependencyError
from .core.external import run_external

__all__ = [
    "OvasmPreset",
    "resolve_ovasm",
    "run_assemble",
    "run_components",
    "run_correct",
    "run_discover",
    "run_evidence",
    "run_identify",
    "run_panel",
    "run_linearize",
    "run_pipeline",
    "run_recruit",
    "run_unify",
]

OvasmPreset = Literal["hifi", "ont", "sr"]

#: ``ovasm evidence`` settings per read type: k, diagonal tolerance, hits per side.
_EVIDENCE_PRESETS: dict[str, tuple[int, int, int]] = {
    "hifi": (21, 30, 10),
    "ont": (15, 150, 5),
    # short reads: 100-150 bp can hold few hits per side (use with min_anchor 40)
    "sr": (21, 30, 5),
}

#: Explicit opt-out tokens for ``ORG_VERSE_OVASM_BIN``.
_DISABLED_TOKENS = {"none", "off", "disable", "0"}


def resolve_ovasm() -> str | None:
    """Resolve the ovasm binary: env override, then PATH."""
    env_value = os.environ.get("ORG_VERSE_OVASM_BIN")
    if env_value:
        if env_value.strip().casefold() in _DISABLED_TOKENS:
            return None
        resolved = shutil.which(env_value) or (
            str(Path(env_value)) if Path(env_value).is_file() else None
        )
        if resolved:
            return resolved
    return shutil.which("ovasm")


def _require() -> str:
    binary = resolve_ovasm()
    if binary is None:
        raise OrganelleDependencyError(
            code="dependency_missing",
            message="ovasm is not available; install it from https://github.com/forageseed/ovasm "
            "and put it on PATH or set ORG_VERSE_OVASM_BIN",
            details={"missing": ["ovasm"]},
        )
    return binary


def run_recruit(
    reads: Sequence[str | Path],
    *,
    seeds: Mapping[str, Sequence[str | Path]],
    out_dir: str | Path,
    preset: OvasmPreset = "hifi",
    target_depth: float | None = None,
    salt: int = 0,
    iterations: int | None = None,
    threads: int | None = None,
    seed_reads: Mapping[str, Sequence[str | Path]] | None = None,
    bootstrap: bool = False,
    saturation: float | None = None,
    extend_gate: bool = False,
    numt_min_block: int | None = None,
) -> dict[str, Any]:
    """Run ``ovasm recruit`` and return its parsed ``recruit.json`` report.

    ``iterations`` defaults to the preset's own: 20 for hifi/sr, 1 for noisy ont reads.
    ``seed_reads`` seeds targets from reads (their solid k-mers) instead of, or besides, a
    reference, e.g. corrected long reads of the same sample to recruit its short reads.
    ``bootstrap`` (hifi only) seeds a target its seed fails to recruit from partially matching
    reads that are organelle-deep along their whole length; seed every organelle of the sample.
    ``saturation`` (default 0.05) stops extending a target whose reads grew by less; 0 extends
    until no bait grows. ``extend_gate`` (hifi only) extends only from organelle-deep reads.
    ``numt_min_block`` is the shortest organelle-like block (bp) that makes a partially matching
    read a NUMT candidate (default: the preset's); a value longer than any read writes none.
    """
    if (bootstrap or extend_gate) and preset != "hifi":
        raise ValueError("OVASM recruitment bootstrap and extension gate need the hifi preset")
    if not reads:
        raise ValueError("at least one read file is required")
    if not seeds and not seed_reads:
        raise ValueError("at least one seed target is required (use run_discover without seeds)")
    argv: list[str] = [
        _require(),
        "recruit",
        "--out",
        str(out_dir),
        "--preset",
        preset,
        "--salt",
        str(salt),
    ]
    if iterations is not None:
        argv += ["--iterations", str(iterations)]
    for path in reads:
        argv += ["--reads", str(path)]
    for name, paths in seeds.items():
        if not paths:
            raise ValueError(f"seed target {name!r} has no files")
        for path in paths:
            argv += ["--seed", f"{name}={path}"]
    for name, paths in (seed_reads or {}).items():
        if not paths:
            raise ValueError(f"seed-reads target {name!r} has no files")
        for path in paths:
            argv += ["--seed-reads", f"{name}={path}"]
    if target_depth is not None:
        argv += ["--target-depth", str(target_depth)]
    if threads is not None:
        argv += ["--threads", str(threads)]
    if bootstrap:
        argv.append("--bootstrap")
    if saturation is not None:
        argv += ["--saturation", str(saturation)]
    if extend_gate:
        argv.append("--extend-gate")
    if numt_min_block is not None:
        argv += ["--numt-min-block", str(numt_min_block)]
    run_external(argv, tool="ovasm recruit", code="assembly.recruitment_failed")
    return json.loads((Path(out_dir) / "recruit.json").read_text(encoding="utf-8"))


def run_discover(
    reads: Sequence[str | Path],
    *,
    out_dir: str | Path,
    target_depth: float | None = None,
    salt: int = 0,
    threads: int | None = None,
) -> dict[str, Any]:
    """Run seed-free ``ovasm discover`` and return its parsed ``discover.json`` report."""
    if not reads:
        raise ValueError("at least one read file is required")
    argv: list[str] = [_require(), "discover", "--out", str(out_dir), "--salt", str(salt)]
    for path in reads:
        argv += ["--reads", str(path)]
    if target_depth is not None:
        argv += ["--target-depth", str(target_depth)]
    if threads is not None:
        argv += ["--threads", str(threads)]
    run_external(argv, tool="ovasm discover", code="assembly.recruitment_failed")
    return json.loads((Path(out_dir) / "discover.json").read_text(encoding="utf-8"))


def run_label(
    clusters: Sequence[str | Path],
    *,
    genes: str | Path,
    out_dir: str | Path,
    threads: int | None = None,
) -> dict[str, Any]:
    """Run ``ovasm label``: assemble each read cluster, label unitigs by organelle genes.

    Returns the parsed ``labels.json``; its ``seeds`` hold one FASTA per organelle (an empty
    string when no unitig of that organelle was found).
    """
    if not clusters:
        raise ValueError("at least one read cluster is required")
    argv: list[str] = [_require(), "label", "--genes", str(genes), "--out", str(out_dir)]
    for path in clusters:
        argv += ["--reads", str(path)]
    if threads is not None:
        argv += ["--threads", str(threads)]
    run_external(argv, tool="ovasm label", code="assembly.label_failed")
    return json.loads((Path(out_dir) / "labels.json").read_text(encoding="utf-8"))


def run_identify(
    query: str | Path,
    *,
    references: Mapping[str, str | Path],
    out_json: str | Path,
    k: int | None = None,
) -> dict[str, Any]:
    """Run ``ovasm identify``: k-mer containment of ``query`` in each reference genome.

    ``references`` maps a set name (e.g. ``mitochondrion``) to a FASTA with one genome per
    record. Returns the parsed report, references ranked by containment.
    """
    if not references:
        raise ValueError("at least one reference set is required")
    argv: list[str] = [_require(), "identify", "--query", str(query), "--out", str(out_json)]
    for name, path in references.items():
        argv += ["--reference", f"{name}={path}"]
    if k is not None:
        argv += ["--k", str(k)]
    run_external(argv, tool="ovasm identify", code="assembly.identify_failed")
    return json.loads(Path(out_json).read_text(encoding="utf-8"))


def run_evidence(
    gfa: str | Path,
    reads: Sequence[str | Path],
    *,
    out_json: str | Path,
    preset: Literal["hifi", "ont", "sr"] = "hifi",
    min_anchor: int = 500,
    bootstrap: int = 1000,
    threads: int | None = None,
) -> dict[str, Any]:
    """Run ``ovasm evidence`` on an assembly graph and return the parsed JSON report."""
    if not reads:
        raise ValueError("at least one read file is required")
    if preset not in _EVIDENCE_PRESETS:
        raise ValueError(f"unknown evidence preset {preset!r}")
    k, diag_tol, side_hits = _EVIDENCE_PRESETS[preset]
    argv: list[str] = [
        _require(),
        "evidence",
        "--gfa",
        str(gfa),
        "--out",
        str(out_json),
        "--k",
        str(k),
        "--diag-tol",
        str(diag_tol),
        "--min-side-hits",
        str(side_hits),
        "--min-anchor",
        str(min_anchor),
        "--bootstrap",
        str(bootstrap),
    ]
    for path in reads:
        argv += ["--reads", str(path)]
    if threads is not None:
        argv += ["--threads", str(threads)]
    run_external(argv, tool="ovasm evidence", code="assembly.evidence_failed")
    return json.loads(Path(out_json).read_text(encoding="utf-8"))


def run_linearize(
    gfa: str | Path,
    evidence_json: str | Path,
    *,
    out_fasta: str | Path,
    out_json: str | Path,
    min_reads: int = 2,
) -> dict[str, Any]:
    """Run ``ovasm linearize`` on a graph and its evidence report; return the parsed report.

    ``out_fasta`` is written only when some path visits every unique segment (``best`` set).
    """
    argv: list[str] = [
        _require(),
        "linearize",
        "--gfa",
        str(gfa),
        "--evidence",
        str(evidence_json),
        "--out-fasta",
        str(out_fasta),
        "--out-json",
        str(out_json),
        "--min-reads",
        str(min_reads),
    ]
    run_external(argv, tool="ovasm linearize", code="assembly.evidence_failed")
    return json.loads(Path(out_json).read_text(encoding="utf-8"))


def run_unify(
    gfa: str | Path,
    *,
    sample: str,
    out_gfa: str | Path,
    out_json: str | Path,
    backend: str = "unknown",
    organelle: Literal["mitochondrion", "plastid"] = "mitochondrion",
    evidence_json: str | Path | None = None,
    linearize_json: str | Path | None = None,
    molecule: str | None = None,
    max_alternatives: int = 3,
) -> dict[str, Any]:
    """Run ``ovasm unify`` and return its parsed JSON report."""
    argv: list[str] = [
        _require(),
        "unify",
        "--gfa",
        str(gfa),
        "--sample",
        sample,
        "--backend",
        backend,
        "--organelle",
        organelle,
        "--max-alternatives",
        str(max_alternatives),
        "--out",
        str(out_gfa),
        "--out-json",
        str(out_json),
    ]
    if evidence_json is not None:
        argv += ["--evidence", str(evidence_json)]
    if linearize_json is not None:
        argv += ["--linearization", str(linearize_json)]
    if molecule is not None:
        argv += ["--molecule", molecule]
    run_external(argv, tool="ovasm unify", code="assembly.unify_failed")
    return json.loads(Path(out_json).read_text(encoding="utf-8"))


def run_assemble(
    reads: Sequence[str | Path],
    *,
    out_gfa: str | Path,
    out_json: str | Path,
    k: int | None = None,
    dense: bool = False,
    min_count: int | None = None,
    threads: int | None = None,
) -> dict[str, Any]:
    """Run ``ovasm assemble`` and return its parsed JSON report.

    ``dense`` makes every k-mer a node (short reads); ``k`` then defaults to the median read
    length minus 23, otherwise to the largest of 1001, 701, ... 101 whose node depth is enough,
    the reads self-corrected first when none is (the binary's defaults).
    """
    if not reads:
        raise ValueError("at least one read file is required")
    argv: list[str] = [_require(), "assemble", "--out", str(out_gfa), "--out-json", str(out_json)]
    for path in reads:
        argv += ["--reads", str(path)]
    if k is not None:
        argv += ["--k", str(k)]
    if dense:
        argv += ["--s", "0"]
    if min_count is not None:
        argv += ["--min-count", str(min_count)]
    if threads is not None:
        argv += ["--threads", str(threads)]
    run_external(argv, tool="ovasm assemble", code="assembly.assemble_failed")
    return json.loads(Path(out_json).read_text(encoding="utf-8"))


def run_correct(
    reads: Sequence[str | Path],
    *,
    out_fasta: str | Path,
    out_json: str | Path,
    ks: Sequence[int] = (15, 21, 25),
    short_reads: Sequence[str | Path] = (),
    threads: int | None = None,
) -> dict[str, Any]:
    """Run ``ovasm correct`` (noisy long reads; hybrid when ``short_reads`` are given, which
    then also runs self rounds for stretches the short reads miss) and return its report."""
    if not reads:
        raise ValueError("at least one read file is required")
    argv: list[str] = [
        _require(),
        "correct",
        "--out",
        str(out_fasta),
        "--out-json",
        str(out_json),
    ]
    for path in reads:
        argv += ["--reads", str(path)]
    for k in ks:
        argv += ["--k", str(k)]
    for path in short_reads:
        argv += ["--short-reads", str(path)]
    if threads is not None:
        argv += ["--threads", str(threads)]
    run_external(argv, tool="ovasm correct", code="assembly.correct_failed")
    return json.loads(Path(out_json).read_text(encoding="utf-8"))


def run_components(
    gfa: str | Path,
    *,
    organelle: Literal["mitochondrion", "plastid"],
    out_gfa: str | Path,
    out_json: str | Path,
    depth_factor: float | None = None,
    gene_depth_factor: float | None = None,
    min_other_genes: int | None = None,
    panel: str | Path | None = None,
) -> dict[str, Any]:
    """Run ``ovasm components`` on an assembly graph and return its report.

    ``out_gfa`` gets the connected components that are the target organelle: the one with the
    most target genes (conserved-protein panel, as ``ovasm panel``), those with target
    genes unless far shallower (``gene_depth_factor``), and those without genes whose depth is
    within ``depth_factor`` of it. ``report["components"]`` lists every component, kept or
    removed, with its length, depth, genes and reason. Without a gene-bearing component
    nothing is removed. Defaults are the binary's.
    """
    argv: list[str] = [
        _require(),
        "components",
        "--gfa",
        str(gfa),
        "--organelle",
        organelle,
        "--out",
        str(out_gfa),
        "--out-json",
        str(out_json),
    ]
    if depth_factor is not None:
        argv += ["--depth-factor", str(depth_factor)]
    if gene_depth_factor is not None:
        argv += ["--gene-depth-factor", str(gene_depth_factor)]
    if min_other_genes is not None:
        argv += ["--min-other-genes", str(min_other_genes)]
    if panel is not None:
        argv += ["--panel", str(panel)]
    run_external(argv, tool="ovasm components", code="assembly.components_failed")
    return json.loads(Path(out_json).read_text(encoding="utf-8"))


def run_panel(
    sets: Sequence[str | Path],
    *,
    out_json: str | Path,
    panel: str | Path | None = None,
    exclude_species: Sequence[str] = (),
    threads: int | None = None,
) -> dict[str, Any]:
    """Run ``ovasm panel`` and return its report.

    Each file of ``sets`` (e.g. a ``discover`` cluster) is called on its own as
    ``mitochondrion``, ``plastid``, ``mixed`` or ``none`` from the conserved organelle proteins
    its reads encode; ``report["sets"]`` follows the order of ``sets``.
    """
    if not sets:
        raise ValueError("at least one read set is required")
    argv: list[str] = [_require(), "panel", "--out-json", str(out_json)]
    for path in sets:
        argv += ["--reads", str(path)]
    if panel is not None:
        argv += ["--panel", str(panel)]
    for name in exclude_species:
        argv += ["--exclude-species", name]
    if threads is not None:
        argv += ["--threads", str(threads)]
    run_external(argv, tool="ovasm panel", code="assembly.identify_failed")
    return json.loads(Path(out_json).read_text(encoding="utf-8"))


def run_pipeline(
    reads: Sequence[str | Path],
    *,
    organelle: Literal["mitochondrion", "plastid", "both"],
    out_dir: str | Path,
    read_type: Literal["hifi", "sr"] = "hifi",
    read_set: Literal["whole-genome", "target"] = "whole-genome",
    seeds: Mapping[str, Sequence[str | Path]] | None = None,
    recruit: Literal["seeds", "discover", "both"] = "seeds",
    recruit_depth: float | None = None,
    single_seed: bool = False,
    mode: Literal["fast", "standard"] = "fast",
    replicates: int | None = None,
    max_replicates: int | None = None,
    replicate_depth: float | None = None,
    salt: int = 0,
    sample: str | None = None,
    component_panel: str | Path | None = None,
    threads: int | None = None,
) -> dict[str, Any]:
    """Run ``ovasm run`` (reads to organelle graph and molecules) and return ``summary.json``.

    The binary does what the desktop runner does: recruit the organelle's reads from
    whole-genome reads (``seeds`` maps ``mitochondrion``/``plastid`` to seed FASTA files, e.g.
    from ``organelleverse._ovasm_seeddb.bundled_seed``; seed both so that each organelle claims
    its own reads), stop when fewer than 100 kb are recruited, assemble at the automatic k and
    retry down the k ladder until the molecules explain half of the graph, then write ``assembly.gfa``,
    ``evidence.json``, ``linearization.json``, ``molecules.fasta``, ``organelle.gfa``/``.json``
    and ``summary.json`` into ``out_dir``. ``mode="standard"`` adds downsampled replicates and
    each junction's share of replicates that reproduce it (``summary["replicates"]``).
    With ``recruit="discover"``/``"both"`` each graph first loses the components that are not the
    target organelle (``ovasm components``; ``component_panel`` swaps its protein panel).
    ``organelle="both"`` writes each organelle into its own subdirectory and returns
    ``{"organelles": {...}}``; a failed run raises with the binary's message.
    """
    if not reads:
        raise ValueError("at least one read file is required")
    argv: list[str] = [
        _require(),
        "run",
        "--organelle",
        organelle,
        "--out",
        str(out_dir),
        "--read-type",
        read_type,
        "--read-set",
        read_set,
        "--recruit",
        recruit,
        "--mode",
        mode,
        "--salt",
        str(salt),
    ]
    for path in reads:
        argv += ["--reads", str(path)]
    for name, paths in (seeds or {}).items():
        if not paths:
            raise ValueError(f"seed target {name!r} has no files")
        for path in paths:
            argv += ["--seeds", f"{name}={path}"]
    if recruit_depth is not None:
        argv += ["--recruit-depth", str(recruit_depth)]
    if single_seed:
        argv.append("--single-seed")
    if replicates is not None:
        argv += ["--replicates", str(replicates)]
    if max_replicates is not None:
        argv += ["--max-replicates", str(max_replicates)]
    if replicate_depth is not None:
        argv += ["--replicate-depth", str(replicate_depth)]
    if sample is not None:
        argv += ["--sample", sample]
    if component_panel is not None:
        argv += ["--component-panel", str(component_panel)]
    if threads is not None:
        argv += ["--threads", str(threads)]
    run_external(argv, tool="ovasm run", code="assembly.pipeline_failed")
    return json.loads((Path(out_dir) / "summary.json").read_text(encoding="utf-8"))
