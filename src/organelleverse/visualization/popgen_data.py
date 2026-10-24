"""Data behind the population / pangenome comparison figures (no drawing here).

Everything is reference-anchored and 1-based inclusive:

* :func:`fst_windows` — fixed-grid F_ST windows from a VCF with the per-site
  estimators of :mod:`organelleverse.population` (same SNP admission rule).
* :func:`pi_ratio_windows` — π per group in shared windows with the Nei (1987)
  estimator of :mod:`organelleverse.diversity`, from an alignment FASTA or a
  reference + VCF pseudo-alignment, and their ratio.
* :func:`reference_coverage_from_gfa` / :func:`coverage_from_paf` — which
  reference bases each sample carries: graph paths sharing the reference
  path's nodes, or per-sample alignments to the reference.
* :func:`presence_svs` — polymorphic presence/absence segments of the
  reference from that coverage.
* :func:`sv_schematic_rows` — homologous vs variable blocks of a locus for a
  few samples, read from their graph paths.
"""

from __future__ import annotations

import gzip
import itertools
import math
import re
from bisect import bisect_left, bisect_right
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ..core.errors import OrganelleInputError, OrganelleParameterError

Interval = tuple[int, int]


# ---------------------------------------------------------------------------
# groups
# ---------------------------------------------------------------------------


def group_order(groups: Mapping[str, str], order: Sequence[str] | None = None) -> list[str]:
    """Group labels in ``order`` if given, else first-appearance order."""
    labels = list(dict.fromkeys(str(g) for g in groups.values()))
    if order is None:
        return labels
    missing = [g for g in order if g not in labels]
    if missing:
        raise OrganelleParameterError(
            code="visualization.popgen.unknown_group",
            message=f"Groups not present in the sample assignment: {', '.join(missing)}.",
        )
    return [str(g) for g in order]


def _pair(groups: Mapping[str, str], pair: Sequence[str] | None) -> tuple[str, str]:
    labels = group_order(groups, pair)
    if pair is None and len(labels) != 2:
        raise OrganelleParameterError(
            code="visualization.popgen.group_pair",
            message=f"{len(labels)} groups found; pass the two to compare (pair=[A, B]).",
        )
    if len(labels) < 2:
        raise OrganelleParameterError(
            code="visualization.popgen.group_pair", message="Two groups are required."
        )
    return labels[0], labels[1]


# ---------------------------------------------------------------------------
# VCF
# ---------------------------------------------------------------------------


def _open_text(path: str | Path):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return path.open()


def vcf_contigs(vcf: str | Path) -> dict[str, int | None]:
    """``##contig`` IDs and lengths, plus record CHROMs not declared in the header."""
    contigs: dict[str, int | None] = {}
    with _open_text(vcf) as handle:
        for line in handle:
            if line.startswith("##contig=<"):
                ident = re.search(r"ID=([^,>]+)", line)
                length = re.search(r"length=(\d+)", line)
                if ident:
                    contigs[ident.group(1)] = int(length.group(1)) if length else None
            elif not line.startswith("#"):
                contigs.setdefault(line.split("\t", 1)[0], None)
    return contigs


def _single_contig(vcf: str | Path, contig: str | None) -> tuple[str, int | None]:
    contigs = vcf_contigs(vcf)
    if contig is not None:
        if contig not in contigs:
            raise OrganelleInputError(
                code="visualization.popgen.vcf_contig",
                message=f"Contig {contig!r} is not in the VCF ({', '.join(contigs) or 'none'}).",
            )
        return contig, contigs[contig]
    if len(contigs) != 1:
        raise OrganelleInputError(
            code="visualization.popgen.vcf_contig",
            message=f"The VCF has {len(contigs)} contigs; choose one with contig=.",
        )
    return next(iter(contigs.items()))


def vcf_records(vcf: str | Path, contig: str):
    """(pos, REF, [ALT...], {sample: GT}) for every record on ``contig``."""
    import vcfpy

    reader = vcfpy.Reader.from_path(str(vcf))
    try:
        for record in reader:
            if contig != record.CHROM:
                continue
            alts = [str(getattr(a, "value", a)).upper() for a in record.ALT]
            calls = {str(c.sample): str(c.data.get("GT") or ".") for c in record.calls}
            yield int(record.POS), str(record.REF).upper(), alts, calls
    finally:
        reader.close()


def biallelic_snps(vcf: str | Path, contig: str) -> dict[int, dict[str, str]]:
    """``{pos: {sample: GT}}`` with the admission rule of ``population.fst_scan``.

    Only records with one concrete ACGT REF base and one concrete ACGT ALT base
    count (a symbolic ``*``/``<NON_REF>`` is not an allele); unlike the
    population parser the contig is respected, so a multi-contig VCF cannot
    merge two molecules' positions.
    """
    snps: dict[int, dict[str, str]] = {}
    for pos, ref, alts, calls in vcf_records(vcf, contig):
        if len(ref) == 1 and ref in "ACGT" and len(alts) == 1 and len(alts[0]) == 1 and alts[0] in "ACGT":
            snps[pos] = calls
    return snps


def _grid(length: int, window_size: int, step: int) -> list[Interval]:
    if window_size < 1 or step < 1:
        raise OrganelleParameterError(
            code="visualization.popgen.window", message="window_size and step must be >= 1."
        )
    starts = range(1, max(2, length - window_size + 2), step)
    return [(s, min(length, s + window_size - 1)) for s in starts]


def fst_windows(
    vcf: str | Path,
    groups: Mapping[str, str],
    *,
    pair: Sequence[str] | None = None,
    estimator: str = "heterozygosity",
    window_size: int = 5000,
    step: int = 1000,
    min_snps: int = 3,
    contig: str | None = None,
    genome_length: int | None = None,
    reference_sample: str | None = None,
) -> dict[str, Any]:
    """Fixed-grid F_ST windows between two groups.

    Per-site values come from ``population.FST_ESTIMATORS[estimator]`` and a
    window is their mean, as in ``population.fst_scan``; windows with fewer than
    ``min_snps`` SNPs have ``fst = None`` (drawn as gaps, not zeros). Unlike
    ``fst_scan`` the windows tile the whole molecule from position 1, so two
    figures with the same grid line up.

    ``reference_sample`` names the sample whose sequence *is* the VCF reference
    (``vg deconstruct -P`` leaves the reference path out of the sample columns):
    it is counted as carrying REF (``0``) at every site.
    """
    from ..population.population import FST_ESTIMATORS

    if estimator not in FST_ESTIMATORS:
        raise OrganelleParameterError(
            code="visualization.popgen.estimator",
            message=f"Unsupported F_ST estimator {estimator!r}; choose one of "
            + ", ".join(sorted(FST_ESTIMATORS))
            + ".",
        )
    name_a, name_b = _pair(groups, pair)
    contig, declared = _single_contig(vcf, contig)
    snps = biallelic_snps(vcf, contig)
    if reference_sample is not None:
        for calls in snps.values():
            calls.setdefault(reference_sample, "0")
    length = genome_length or declared or (max(snps) if snps else 0)
    if not length:
        raise OrganelleInputError(
            code="visualization.popgen.length",
            message="Genome length unknown: the VCF declares no contig length; pass genome_length.",
        )
    vcf_samples = set().union(*snps.values()) if snps else set()
    pop_a = [s for s, g in groups.items() if g == name_a and s in vcf_samples]
    pop_b = [s for s, g in groups.items() if g == name_b and s in vcf_samples]
    missing = sorted(s for s, g in groups.items() if g in (name_a, name_b) and s not in vcf_samples)
    if not pop_a or not pop_b:
        raise OrganelleInputError(
            code="visualization.popgen.empty_group",
            message="Both groups need samples present in the VCF.",
            details={"missing_from_vcf": missing},
        )
    site = FST_ESTIMATORS[estimator]
    positions = sorted(snps)
    values = [site(snps[p], pop_a, pop_b) for p in positions]
    windows = []
    for start, end in _grid(length, window_size, step):
        lo, hi = bisect_left(positions, start), bisect_right(positions, end)
        n = hi - lo
        fst = round(sum(values[lo:hi]) / n, 4) if n >= min_snps else None
        windows.append({"start": start, "end": end, "fst": fst, "n_snps": n})
    scored = [w["fst"] for w in windows if w["fst"] is not None]
    return {
        "contig": contig,
        "genome_length": length,
        "estimator": estimator,
        "groups": [name_a, name_b],
        "group_sizes": {name_a: len(pop_a), name_b: len(pop_b)},
        "samples_missing_from_vcf": missing,
        "window_size": window_size,
        "step": step,
        "min_snps": min_snps,
        "n_snps": len(positions),
        "mean_site_fst": round(sum(values) / len(values), 4) if values else None,
        "windows": windows,
        "scored_windows": len(scored),
        "max_fst": max(scored) if scored else None,
    }


# ---------------------------------------------------------------------------
# π (diversity estimator) and the π ratio
# ---------------------------------------------------------------------------


def _called(gt: str) -> str | None:
    """Haploid organelle allele index of a GT; None when missing or heterozygous."""
    alleles = gt.replace("|", "/").split("/")
    if any(a in (".", "") for a in alleles) or len(set(alleles)) != 1:
        return None
    return alleles[0]


def pseudo_alignment_from_vcf(
    vcf: str | Path,
    reference_fasta: str | Path,
    samples: Iterable[str],
    *,
    contig: str | None = None,
    reference_sample: str | None = None,
) -> tuple[str, dict[str, str], dict[str, int]]:
    """Reference-length sequences carrying each sample's SNP alleles.

    A sample gets the ALT base where its (haploid) call is ALT, ``N`` where its
    call is missing or heterozygous, and the reference base elsewhere. Where a
    sample carries a non-SNP ALT allele (indel / SV record) the REF span is set
    to ``N``: that sequence is absent or different in the sample, so it must not
    count as identical to the reference. ``reference_sample`` (absent from the
    VCF columns, as with ``vg deconstruct -P``) is the reference sequence itself.
    Returns (reference, sequences, counts).
    """
    from .._bio import read_fasta

    contig, _ = _single_contig(vcf, contig)
    records = read_fasta(Path(reference_fasta))
    chosen = [seq for name, seq in records if name.split()[0] == contig]
    if not chosen:
        if len(records) != 1:
            raise OrganelleInputError(
                code="visualization.popgen.reference",
                message=f"Reference FASTA has no record named {contig!r}.",
            )
        chosen = [records[0][1]]
    reference = chosen[0].upper()
    names = list(samples)
    seqs = {s: bytearray(reference, "ascii") for s in names}
    # masked bases stay N even when an overlapping SNP record comes later
    masked = {s: bytearray(len(reference)) for s in names}
    counts = {"snps": 0, "masked_records": 0, "masked_bases": 0}

    def mask(sample: str, start: int, length: int) -> None:
        seqs[sample][start : start + length] = b"N" * length
        masked[sample][start : start + length] = b"" * length

    for pos, ref, alts, calls in vcf_records(vcf, contig):
        start = pos - 1
        if start < 0 or start + len(ref) > len(reference):
            continue
        is_snp = len(ref) == 1 and all(len(a) == 1 and a in "ACGT" for a in alts)
        counts["snps" if is_snp else "masked_records"] += 1
        for sample in names:
            default = "0" if sample == reference_sample else "."
            allele = _called(calls.get(sample, default))
            if allele is None:
                mask(sample, start, len(ref))
                continue
            if allele == "0":
                continue
            index = int(allele) - 1
            if is_snp and 0 <= index < len(alts):
                if not masked[sample][start]:
                    seqs[sample][start] = ord(alts[index])
            else:
                mask(sample, start, len(ref))
                counts["masked_bases"] += len(ref)
    return reference, {s: seqs[s].decode() for s in names}, counts


def pi_ratio_windows(
    groups: Mapping[str, str],
    *,
    alignment: str | Path | None = None,
    vcf: str | Path | None = None,
    reference_fasta: str | Path | None = None,
    reference: str | None = None,
    pair: Sequence[str] | None = None,
    window_size: int = 5000,
    step: int = 1000,
    gap_mode: str = "pairwise",
    min_sites: int = 500,
    pseudocount: float | None = None,
    contig: str | None = None,
    reference_sample: str | None = None,
) -> dict[str, Any]:
    """π of two groups in the same reference windows and π_A / π_B.

    π is ``diversity._window_pi_s`` (Nei 1987 mean pairwise differences per
    comparable site) over the window's columns, separately for each group.
    Coordinates are on ``reference`` (alignment input; the first sequence by
    default) or on the VCF reference. A window whose either group has fewer
    than ``min_sites`` comparable sites has no ratio. ``pseudocount`` is added
    to both π before dividing so a monomorphic denominator does not explode;
    by default it is half the smallest non-zero π of either group.
    ``reference_sample`` is passed to :func:`pseudo_alignment_from_vcf`.
    """
    from ..diversity.diversity import _window_pi_s

    if (alignment is None) == (vcf is None):
        raise OrganelleParameterError(
            code="visualization.popgen.pi_input",
            message="Pass exactly one of alignment= (aligned FASTA) or vcf= (+ reference_fasta=).",
        )
    if gap_mode not in ("exclude", "pairwise"):
        raise OrganelleParameterError(
            code="visualization.popgen.gap_mode", message="gap_mode must be 'exclude' or 'pairwise'."
        )
    name_a, name_b = _pair(groups, pair)
    source: dict[str, Any]
    if vcf is not None:
        if reference_fasta is None:
            raise OrganelleParameterError(
                code="visualization.popgen.pi_input",
                message="vcf= needs reference_fasta= to build the pseudo-alignment.",
            )
        wanted = [s for s, g in groups.items() if g in (name_a, name_b)]
        ref_seq, seqs, counts = pseudo_alignment_from_vcf(
            vcf, reference_fasta, wanted, contig=contig, reference_sample=reference_sample
        )
        source = {"input": "vcf", "reference_length": len(ref_seq), **counts}
    else:
        from .._bio import read_fasta

        records = [(name.split()[0], seq.upper()) for name, seq in read_fasta(Path(alignment))]
        if len({len(s) for _, s in records}) != 1:
            raise OrganelleInputError(
                code="visualization.popgen.unaligned",
                message="Aligned sequences must have equal lengths; align the input first.",
            )
        by_name = dict(records)
        ref_name = reference or records[0][0]
        if ref_name not in by_name:
            raise OrganelleInputError(
                code="visualization.popgen.reference",
                message=f"Reference {ref_name!r} is not in the alignment.",
            )
        ref_seq = by_name[ref_name]
        seqs = {s: by_name[s] for s in groups if s in by_name}
        source = {"input": "alignment", "reference": ref_name, "alignment_length": len(ref_seq)}
    members = {
        g: [seqs[s] for s, grp in groups.items() if grp == g and s in seqs] for g in (name_a, name_b)
    }
    if any(len(m) < 2 for m in members.values()):
        raise OrganelleInputError(
            code="visualization.popgen.empty_group",
            message="Each group needs at least two sequences for π.",
            details={g: len(m) for g, m in members.items()},
        )
    # reference coordinate of every alignment column (gaps map to the previous base)
    ref_pos, called = [], 0
    for base in ref_seq:
        if base not in "-.":
            called += 1
        ref_pos.append(called)
    ref_length = called
    windows = []
    for start, end in _grid(ref_length, window_size, step):
        lo = bisect_left(ref_pos, start)
        hi = bisect_right(ref_pos, end)
        row: dict[str, Any] = {"start": start, "end": end}
        for key, seq_list in (("pi_a", members[name_a]), ("pi_b", members[name_b])):
            if gap_mode == "exclude":
                columns = [
                    [s[i] for s in seq_list]
                    for i in range(lo, hi)
                    if all(s[i] in "ACGT" for s in seq_list)
                ]
            else:
                columns = [[s[i] for s in seq_list if s[i] in "ACGT"] for i in range(lo, hi)]
            pi, segregating, comparable = _window_pi_s(columns)
            row[key] = round(pi, 7)
            row[f"{key}_sites"] = comparable
            row[f"{key}_segregating"] = segregating
        windows.append(row)
    nonzero = [w[k] for w in windows for k in ("pi_a", "pi_b") if w[k] > 0]
    if pseudocount is None:
        pseudocount = (min(nonzero) / 2) if nonzero else 1e-6
    for w in windows:
        ok = w["pi_a_sites"] >= min_sites and w["pi_b_sites"] >= min_sites
        w["ratio"] = round((w["pi_a"] + pseudocount) / (w["pi_b"] + pseudocount), 4) if ok else None
    scored = [w["ratio"] for w in windows if w["ratio"] is not None]
    mean = lambda key: round(sum(w[key] for w in windows) / len(windows), 7) if windows else None  # noqa: E731
    return {
        **source,
        "genome_length": ref_length,
        "groups": [name_a, name_b],
        "group_sizes": {g: len(m) for g, m in members.items()},
        "window_size": window_size,
        "step": step,
        "gap_mode": gap_mode,
        "min_sites": min_sites,
        "pseudocount": pseudocount,
        "mean_pi_a": mean("pi_a"),
        "mean_pi_b": mean("pi_b"),
        "windows": windows,
        "scored_windows": len(scored),
        "max_ratio": max(scored) if scored else None,
    }


# ---------------------------------------------------------------------------
# coverage of the reference
# ---------------------------------------------------------------------------


def merge_intervals(intervals: Iterable[Interval], gap: int = 0) -> list[Interval]:
    """Sorted, merged 1-based inclusive intervals (``gap`` bp apart still merge)."""
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1] + 1 + gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged]


def _segment_length(segment) -> int:
    if segment.length is not None:
        return int(segment.length)
    if segment.sequence is not None:
        return len(segment.sequence)
    raise OrganelleInputError(
        code="visualization.popgen.gfa_length",
        message=f"GFA segment {segment.name} has neither sequence nor LN tag.",
    )


def _path_sample(path) -> str:
    return path.sample or path.name.split("#", 1)[0]


def _reference_path(graph, reference: str):
    if reference in graph.paths:
        return graph.paths[reference]
    hits = [p for p in graph.paths.values() if _path_sample(p) == reference]
    if len(hits) != 1:
        raise OrganelleInputError(
            code="visualization.popgen.reference_path",
            message=f"Reference {reference!r} must name one GFA path or a sample with one path "
            f"({len(hits)} found).",
        )
    return hits[0]


def _load_graph(gfa):
    if hasattr(gfa, "paths") and hasattr(gfa, "segments"):
        return gfa
    from ..pangenome._gfa import load_gfa

    return load_gfa(gfa)


def reference_coverage_from_gfa(
    gfa: str | Path | Any,
    reference: str,
    *,
    samples: Sequence[str] | None = None,
    merge_gap: int = 0,
) -> dict[str, Any]:
    """Reference bases each sample's graph paths traverse.

    A reference base is carried by a sample when the graph node it lies on is
    also on one of the sample's paths (PanSN ``sample#hap#contig``; all of a
    sample's paths are pooled). This is node-sharing presence, the same
    definition PGGB/odgi PAV use: it says the sequence is present, not where.
    """
    graph = _load_graph(gfa)
    ref_path = _reference_path(graph, reference)
    nodes_by_sample: dict[str, set[str]] = defaultdict(set)
    for path in graph.paths.values():
        nodes_by_sample[_path_sample(path)].update(node for node, _ in path.steps)
    names = list(samples) if samples is not None else sorted(nodes_by_sample)
    unknown = [s for s in names if s not in nodes_by_sample]
    if unknown:
        raise OrganelleInputError(
            code="visualization.popgen.unknown_sample",
            message=f"Samples without a GFA path: {', '.join(unknown)}.",
        )
    steps = []
    offset = 0
    for node, _ in ref_path.steps:
        length = _segment_length(graph.segments[node])
        steps.append((node, offset + 1, offset + length))
        offset += length
    coverage = {}
    for sample in names:
        carried = nodes_by_sample[sample]
        coverage[sample] = merge_intervals(
            ((a, b) for node, a, b in steps if node in carried), gap=merge_gap
        )
    return {
        "source": "gfa",
        "reference": ref_path.name,
        "reference_sample": _path_sample(ref_path),
        "genome_length": offset,
        "coverage": coverage,
    }


def coverage_from_paf(
    paf: str | Path,
    *,
    reference: str | None = None,
    min_identity: float = 0.0,
    min_length: int = 0,
    sample_of: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Reference intervals covered by each query's alignments in a PAF.

    The query name gives the sample (PanSN prefix before ``#``) unless
    ``sample_of`` maps query names. ``min_identity`` is matches / block length.
    """
    coverage: dict[str, list[Interval]] = defaultdict(list)
    target_length = None
    target = reference
    with _open_text(paf) as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 12:
                continue
            if target is None:
                target = fields[5]
            if fields[5] != target:
                continue
            target_length = int(fields[6])
            matches, block = int(fields[9]), int(fields[10])
            if block < min_length or (block and matches / block < min_identity):
                continue
            query = fields[0]
            sample = (sample_of or {}).get(query, query.split("#", 1)[0])
            coverage[sample].append((int(fields[7]) + 1, int(fields[8])))
    if target is None:
        raise OrganelleInputError(code="visualization.popgen.paf", message="PAF has no alignments.")
    return {
        "source": "paf",
        "reference": target,
        "reference_sample": target,
        "genome_length": target_length,
        "coverage": {s: merge_intervals(v) for s, v in coverage.items()},
    }


def covered_fraction(intervals: Sequence[Interval], start: int, end: int) -> float:
    """Fraction of [start, end] covered by sorted, merged ``intervals``."""
    total = 0
    for a, b in intervals:
        if b < start:
            continue
        if a > end:
            break
        total += min(b, end) - max(a, start) + 1
    return total / (end - start + 1) if end >= start else 0.0


def presence_svs(
    coverage: Mapping[str, Sequence[Interval]],
    genome_length: int,
    *,
    min_length: int = 500,
    min_absent: int = 1,
    min_present: int = 1,
    merge_gap: int = 100,
    present_min: float = 0.8,
    absent_max: float = 0.2,
) -> list[dict[str, Any]]:
    """Reference segments present in some samples and absent from others.

    The reference is cut at every coverage boundary; each piece has the set of
    samples carrying it. Adjacent pieces with the same carrier set (or separated
    by fewer than ``merge_gap`` bp) are joined, and segments of at least
    ``min_length`` bp missing from ``min_absent`` or more samples while carried
    by ``min_present`` or more are reported. Sequence absent from the reference
    itself (insertions in other samples) is outside this reference-anchored scan.
    Per sample, ``fraction`` is the covered fraction of the segment and
    ``presence`` is ``True`` when it is at least ``present_min``, ``False`` at
    most ``absent_max`` and ``None`` (not called, a missing value) in between.
    """
    if not 0 <= absent_max < present_min <= 1:
        raise OrganelleParameterError(
            code="visualization.popgen.presence_thresholds",
            message="Need 0 <= absent_max < present_min <= 1.",
        )
    samples = list(coverage)
    cuts = {1, genome_length + 1}
    for intervals in coverage.values():
        for a, b in intervals:
            cuts.update((a, b + 1))
    edges = sorted(c for c in cuts if 1 <= c <= genome_length + 1)
    starts = {s: [a for a, _ in coverage[s]] for s in samples}

    def carries(sample: str, position: int) -> bool:
        index = bisect_right(starts[sample], position) - 1
        return index >= 0 and coverage[sample][index][1] >= position

    pieces = []
    for a, nxt in itertools.pairwise(edges):
        b = nxt - 1
        carriers = frozenset(s for s in samples if carries(s, (a + b) // 2))
        pieces.append([a, b, carriers])
    merged: list[list[Any]] = []
    for piece in pieces:
        if merged and merged[-1][2] == piece[2]:
            merged[-1][1] = piece[1]
        else:
            merged.append(piece)
    candidates = [
        p
        for p in merged
        if len(samples) - len(p[2]) >= min_absent and len(p[2]) >= min_present
    ]
    joined: list[list[Any]] = []
    for piece in candidates:
        if joined and joined[-1][2] == piece[2] and piece[0] - joined[-1][1] - 1 <= merge_gap:
            joined[-1][1] = piece[1]
        else:
            joined.append(list(piece))
    svs = []
    for a, b, _ in joined:
        if b - a + 1 < min_length:
            continue
        fraction = {s: round(covered_fraction(coverage[s], a, b), 4) for s in samples}
        presence = {
            s: True if f >= present_min else False if f <= absent_max else None
            for s, f in fraction.items()
        }
        svs.append(
            {
                "id": f"SV{len(svs) + 1}",
                "start": a,
                "end": b,
                "length": b - a + 1,
                "present": sum(v is True for v in presence.values()),
                "absent": sum(v is False for v in presence.values()),
                "missing": sum(v is None for v in presence.values()),
                "presence": presence,
                "fraction": fraction,
            }
        )
    return svs


# ---------------------------------------------------------------------------
# SV schematic rows from graph paths
# ---------------------------------------------------------------------------


def _flip_walk(steps):
    return [(node, "-" if orient == "+" else "+") for node, orient in reversed(steps)]


def _heaviest_chain(hits, forward, region_nodes, ref_starts, seg_len, max_insert):
    """Heaviest (bp) collinear chain of locus-node hits on one sample path.

    ``hits`` are (ref_rank, path_index, path_offset); a forward chain has path
    index increasing with rank, a reverse one decreasing. Consecutive hits may
    be at most ``max_insert`` bp further apart on the sample than on the
    reference, so a chain cannot jump to a distant copy. Returns (weight, first
    path index, last path index, first rank, last rank) or None. O(n^2), n = locus-node hits.
    """
    if not hits:
        return None
    hits = sorted(hits)
    n = len(hits)
    weight = [seg_len[region_nodes[r]] for r, _, _ in hits]
    best = weight[:]
    back = [-1] * n
    for j in range(n):
        rj, ij, oj = hits[j]
        for i in range(j):
            ri, ii, oi = hits[i]
            if ri >= rj or (ii >= ij if forward else ii <= ij):
                continue
            if abs(oj - oi) - (ref_starts[rj] - ref_starts[ri]) > max_insert:
                continue
            if best[i] + weight[j] > best[j]:
                best[j] = best[i] + weight[j]
                back[j] = i
    end = max(range(n), key=best.__getitem__)
    first = end
    while back[first] != -1:
        first = back[first]
    return best[end], hits[first][1], hits[end][1], hits[first][0], hits[end][0]


def sv_schematic_rows(
    gfa: str | Path | Any,
    reference: str,
    start: int,
    end: int,
    samples: Sequence[str],
    *,
    min_sv_length: int = 100,
    max_insert: int | None = None,
    homology_fraction: float = 1.0,
    min_span: float = 0.8,
    labels: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Homologous (shared) vs structural-variant blocks of one locus per sample.

    The locus is ``start..end`` on the reference path; the reference row is its
    walk. For every other sample the row is the longest stretch of its path that
    is collinear with the locus: the heaviest chain (in bp) of locus nodes
    (single-copy within the locus) in reference order and
    orientation (or all reversed, then the walk is flipped); a node repeated in
    the sample enters the chain at most once, and consecutive chain nodes may be
    at most ``max_insert`` bp (default: the locus length, at least 20 kb)
    further apart on the sample than on the reference. Sequence of a
    rearranged sample outside that chain is not drawn (``collinear_fraction``
    is the share of the locus the chain spans). Rows spanning at least
    ``min_span`` of the locus vote on homology (the reference always does):
    nodes carried by at least ``homology_fraction`` of the voters (default all)
    are homologous; runs of other nodes at least ``min_sv_length`` bp long are SV
    blocks, shorter ones (SNP/indel bubbles) stay inside the homologous block.
    SV blocks between the same pair of homologous nodes share an id (SV1, SV2,
    ... in reference order). ``align`` = (row x, reference-row x) of the row's first
    homologous node, so rows can be drawn under the reference; ``edge_positions`` are the
    homologous-block edges used for the dashed connectors.
    """
    graph = _load_graph(gfa)
    ref_path = _reference_path(graph, reference)
    seg_len = {name: _segment_length(seg) for name, seg in graph.segments.items()}
    sample_paths: dict[str, list] = defaultdict(list)
    for path in graph.paths.values():
        sample_paths[_path_sample(path)].append(path)
    rows_in = list(dict.fromkeys(samples))
    ref_sample = _path_sample(ref_path)
    if ref_sample in rows_in:
        rows_in.remove(ref_sample)
    rows_in.insert(0, ref_sample)
    missing = [s for s in rows_in if s not in sample_paths]
    if missing:
        raise OrganelleInputError(
            code="visualization.popgen.unknown_sample",
            message=f"Samples without a GFA path: {', '.join(missing)}.",
        )
    region, offset = [], 0
    for node, orient in ref_path.steps:
        a, b = offset + 1, offset + seg_len[node]
        if b >= start and a <= end:
            region.append((node, orient, a, b))
        offset = b
    if len(region) < 2:
        raise OrganelleParameterError(
            code="visualization.popgen.locus", message="The locus is outside the reference path."
        )
    region_nodes = [n for n, _, _, _ in region]
    counts_in_region: dict[str, int] = defaultdict(int)
    for n in region_nodes:
        counts_in_region[n] += 1
    ref_rank = {n: i for i, n in enumerate(region_nodes) if counts_in_region[n] == 1}
    ref_orient = {n: o for n, o, _, _ in region}
    locus_bp = region[-1][3] - region[0][2] + 1
    ref_starts = [a for _, _, a, _ in region]
    if max_insert is None:
        max_insert = max(locus_bp, 20000)
    walks: dict[str, list] = {ref_sample: [(n, o) for n, o, _, _ in region]}
    collinear: dict[str, float] = {ref_sample: 1.0}
    for sample in rows_in[1:]:
        best = None
        for path in sample_paths[sample]:
            steps = list(path.steps)
            hits: dict[bool, list] = {True: [], False: []}
            pos = 0
            for i, (node, orient) in enumerate(steps):
                if node in ref_rank:
                    hits[orient == ref_orient[node]].append((ref_rank[node], i, pos))
                pos += seg_len[node]
            for forward in (True, False):
                chain = _heaviest_chain(
                    hits[forward], forward, region_nodes, ref_starts, seg_len, max_insert
                )
                if chain and (best is None or chain[0] > best[0]):
                    best = (chain[0], steps, chain[1], chain[2], forward, chain[3], chain[4])
        if best is None:
            raise OrganelleInputError(
                code="visualization.popgen.no_anchor",
                message=f"{sample} carries no single-copy node of the locus.",
            )
        _weight, steps, first, last, forward, rank_a, rank_b = best
        lo, hi = min(first, last), max(first, last)
        walk = steps[lo : hi + 1]
        walks[sample] = walk if forward else _flip_walk(walk)
        span = region[rank_b][3] - region[rank_a][2] + 1
        collinear[sample] = round(span / locus_bp, 4)
    if not 0 < homology_fraction <= 1:
        raise OrganelleParameterError(
            code="visualization.popgen.homology_fraction",
            message="homology_fraction must be in (0, 1].",
        )
    voters = [w for s, w in walks.items() if s == ref_sample or collinear[s] >= min_span]
    carriers_per_node: dict[str, int] = defaultdict(int)
    for walk in voters:
        for node in {n for n, _ in walk}:
            carriers_per_node[node] += 1
    need = homology_fraction * len(voters) - 1e-9
    shared = {n for n, c in carriers_per_node.items() if c >= need}
    order_rank: dict[str, int] = {}
    for i, n in enumerate(region_nodes):
        order_rank.setdefault(n, i)
    rows, sv_keys = [], {}
    for sample in rows_in:
        walk = walks[sample]
        # runs of shared / variable nodes in sample coordinates (0-based start)
        runs, pos = [], 0
        for node, _ in walk:
            kind = "homologous" if node in shared else "sv"
            length = seg_len[node]
            if runs and runs[-1]["kind"] == kind:
                runs[-1]["end"] += length
                runs[-1]["nodes"].append(node)
            else:
                runs.append({"kind": kind, "start": pos, "end": pos + length, "nodes": [node]})
            pos += length
        for run in runs:  # short variable runs are SNP/indel bubbles, not SVs
            if run["kind"] == "sv" and run["end"] - run["start"] < min_sv_length:
                run["kind"] = "homologous"
        blocks: list[dict[str, Any]] = []
        for run in runs:
            if blocks and blocks[-1]["kind"] == run["kind"]:
                blocks[-1]["end"] = run["end"]
                blocks[-1]["nodes"].extend(run["nodes"])
            else:
                blocks.append({**run, "nodes": list(run["nodes"])})
        for index, block in enumerate(blocks):
            if block["kind"] != "sv":
                continue
            before = next(
                (n for b in reversed(blocks[:index]) for n in reversed(b["nodes"]) if n in shared),
                None,
            )
            after = next(
                (n for b in blocks[index + 1 :] for n in b["nodes"] if n in shared), None
            )
            key = (before, after)
            sv_keys.setdefault(key, order_rank.get(before, -1))
            block["key"] = key
        rows.append({"sample": sample, "length": pos, "blocks": blocks})
    order = sorted(sv_keys, key=lambda k: sv_keys[k])
    sv_ids = {key: f"SV{i + 1}" for i, key in enumerate(order)}
    # reference coordinates of each SV site: between its flanking homologous nodes
    ref_node_span: dict[str, tuple[int, int]] = {}
    for n, _, a, b in region:
        ref_node_span.setdefault(n, (a, b))
    svs = []
    for key in order:
        before, after = key
        left_edge = ref_node_span.get(before, (start, start - 1))[1]
        right_edge = ref_node_span.get(after, (end + 1, end))[0]
        ref_present = right_edge - left_edge - 1 > 0
        carriers = [r["sample"] for r in rows if any(b.get("key") == key for b in r["blocks"])]
        svs.append(
            {
                "id": sv_ids[key],
                "ref_start": left_edge + 1,
                "ref_end": right_edge - 1 if ref_present else left_edge,
                "in_reference": any(b.get("key") == key for b in rows[0]["blocks"]),
                "carriers": carriers,
            }
        )
    # node edge positions per row for connectors between homologous blocks
    for row in rows:
        node_pos, pos = {}, 0
        for node, _ in walks[row["sample"]]:
            node_pos.setdefault(node, (pos, pos + seg_len[node]))
            pos += seg_len[node]
        edges = []
        for block in row["blocks"]:
            if block["kind"] == "homologous":
                first = next((n for n in block["nodes"] if n in shared), None)
                last = next((n for n in reversed(block["nodes"]) if n in shared), None)
                if first is not None:
                    edges.append(("start", first))
                if last is not None:
                    edges.append(("end", last))
        row["edges"] = edges
        row["node_pos"] = node_pos
    ref_pos = rows[0]["node_pos"]
    for row in rows:
        first = next((n for n, _ in walks[row["sample"]] if n in ref_pos and n in shared), None)
        # x of this row's first homologous node, and of the same node on the reference row
        row["align"] = [row["node_pos"][first][0], ref_pos[first][0]] if first else [0, 0]
        row["collinear_fraction"] = collinear[row["sample"]]
    edge_set = {e for row in rows for e in row["edges"]}
    for row in rows:
        positions = {}
        for side, node in edge_set:
            if node in row["node_pos"]:
                a, b = row["node_pos"][node]
                positions[f"{side}:{node}"] = a if side == "start" else b
        row["edge_positions"] = positions
    for row in rows:
        for block in row["blocks"]:
            key = block.pop("key", None)
            block["sv"] = sv_ids.get(key) if key is not None else None
            block["nodes"] = len(block["nodes"])
        del row["node_pos"], row["edges"]
        row["label"] = (labels or {}).get(row["sample"], row["sample"])
    return {
        "reference": ref_path.name,
        "locus": {"start": start, "end": end},
        "reference_anchor_start": region[0][2],
        "min_sv_length": min_sv_length,
        "max_insert": max_insert,
        "homology_fraction": homology_fraction,
        "min_span": min_span,
        "voting_rows": len(voters),
        "rows": rows,
        "svs": svs,
    }


def sv_presence_table(
    svs: Sequence[Mapping[str, Any]],
    groups: Mapping[str, str],
    units: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Per SV and group: present / absent / missing counts of independent units.

    ``units`` maps sample -> unit id (e.g. :func:`identical_path_groups`);
    samples of one unit count once, with the majority call of its members
    (a tie or no call is missing). Without ``units`` every sample is a unit.
    Samples without a group are skipped; a sample absent from an SV's
    ``presence`` (or with ``None``) is missing.
    """
    labels = group_order(groups)
    unit_of = {s: (units or {}).get(s, s) for s in groups}
    members: dict[tuple[str, str], list[str]] = defaultdict(list)
    for sample, group in groups.items():
        members[(group, unit_of[sample])].append(sample)
    table = []
    for sv in svs:
        presence = sv["presence"]
        counts = {g: {"present": 0, "absent": 0, "missing": 0} for g in labels}
        for (group, _unit), samples in members.items():
            calls = [presence.get(s) for s in samples]
            yes, no = sum(c is True for c in calls), sum(c is False for c in calls)
            state = "present" if yes > no else "absent" if no > yes else "missing"
            counts[group][state] += 1
        table.append({"id": sv["id"], "counts": counts})
    return table


def identical_path_groups(gfa: str | Path | Any, samples: Sequence[str] | None = None) -> dict[str, str]:
    """Unit id per sample: samples whose graph paths spell the same molecule share one.

    A sample's paths are compared as circular oriented walks, invariant to the
    start node and to reading direction (rotation + reverse complement), so two
    accessions of one identical mitogenome deposited with different origins
    collapse. Unit ids are ``U1, U2, ...`` in first-seen order.
    """
    graph = _load_graph(gfa)
    walks: dict[str, list[tuple]] = defaultdict(list)
    for path in graph.paths.values():
        walks[_path_sample(path)].append(tuple(path.steps))
    names = list(samples) if samples is not None else sorted(walks)
    keys: dict[tuple, str] = {}
    units = {}
    for sample in names:
        key = tuple(sorted(_canonical_walk(w) for w in walks.get(sample, [])))
        units[sample] = keys.setdefault(key, f"U{len(keys) + 1}")
    return units


def _canonical_walk(steps: tuple) -> tuple:
    """Least rotation of the walk or of its reverse-complement walk."""
    if not steps:
        return ()
    flipped = tuple((n, "-" if o == "+" else "+") for n, o in reversed(steps))
    best = None
    for walk in (steps, flipped):
        k = _least_rotation(walk)
        rotated = walk[k:] + walk[:k]
        if best is None or rotated < best:
            best = rotated
    return best


def _least_rotation(seq: Sequence) -> int:
    """Booth's algorithm: start index of the lexicographically least rotation."""
    doubled = list(seq) + list(seq)
    failure = [-1] * len(doubled)
    k = 0
    for j in range(1, len(doubled)):
        item = doubled[j]
        i = failure[j - k - 1]
        while i != -1 and item != doubled[k + i + 1]:
            if item < doubled[k + i + 1]:
                k = j - i - 1
            i = failure[i]
        if item != doubled[k + i + 1]:
            if item < doubled[k]:
                k = j
            failure[j - k] = -1
        else:
            failure[j - k] = i + 1
    return k


def exact_presence_test(present: Sequence[int], absent: Sequence[int]) -> float | None:
    """Exact p for a k x 2 (group x present/absent) table with fixed margins.

    Two-sided Fisher's exact test for k = 2 and its Freeman-Halton extension
    for k > 2: the sum of the hypergeometric probabilities of every table with
    the observed margins that is no more probable than the observed one
    (relative tolerance 1e-7, as scipy). ``None`` when the table has fewer
    than two non-empty groups or no variation (all present or all absent).
    """
    rows = [(int(p), int(a)) for p, a in zip(present, absent, strict=True) if p + a > 0]
    total_present = sum(p for p, _ in rows)
    sizes = [p + a for p, a in rows]
    n = sum(sizes)
    if len(rows) < 2 or total_present in (0, n):
        return None

    def log_comb(a: int, b: int) -> float:
        return math.lgamma(a + 1) - math.lgamma(b + 1) - math.lgamma(a - b + 1)

    base = log_comb(n, total_present)
    observed = sum(log_comb(size, p) for size, (p, _) in zip(sizes, rows, strict=True)) - base
    threshold = observed + math.log1p(1e-7)
    tail = [sum(sizes[i:]) for i in range(len(sizes))] + [0]
    total = 0.0

    def walk(i: int, remaining: int, log_p: float) -> None:
        nonlocal total
        if i == len(sizes) - 1:
            if remaining <= sizes[i]:
                value = log_p + log_comb(sizes[i], remaining) - base
                if value <= threshold:
                    total += math.exp(value)
            return
        low = max(0, remaining - tail[i + 1])
        for x in range(low, min(sizes[i], remaining) + 1):
            walk(i + 1, remaining - x, log_p + log_comb(sizes[i], x))

    walk(0, total_present, 0.0)
    return min(1.0, total)


def benjamini_hochberg(p_values: Sequence[float | None]) -> list[float | None]:
    """BH-adjusted q-values; None entries stay None."""
    indexed = [(p, i) for i, p in enumerate(p_values) if p is not None and not math.isnan(p)]
    m = len(indexed)
    q: list[float | None] = [None] * len(p_values)
    running = 1.0
    for rank, (p, i) in reversed(list(enumerate(sorted(indexed), start=1))):
        running = min(running, p * m / rank)
        q[i] = min(1.0, running)
    return q


__all__ = [
    "benjamini_hochberg",
    "biallelic_snps",
    "coverage_from_paf",
    "covered_fraction",
    "exact_presence_test",
    "fst_windows",
    "group_order",
    "identical_path_groups",
    "merge_intervals",
    "pi_ratio_windows",
    "presence_svs",
    "pseudo_alignment_from_vcf",
    "reference_coverage_from_gfa",
    "sv_presence_table",
    "sv_schematic_rows",
    "vcf_contigs",
    "vcf_records",
]
