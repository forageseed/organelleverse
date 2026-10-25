"""Population / pangenome comparison figures for organelle genomes.

Five compute-only plots (plus a stacked A+B panel); each returns an
:class:`~organelleverse.visualization.plot_object.OrganellePlot` whose
``metrics`` hold every number drawn. ``ov.write(plot, "fig.pdf")`` renders it at
printed size (``width`` = ``"single"`` 89 mm / ``"onehalf"`` 120 mm /
``"double"`` 183 mm or millimetres; heights in mm, at most 170) with a
``<stem>.tsv`` data sidecar, and ``plot.panel`` lets
:func:`~organelleverse.visualization.compose.compose_figure` place it as one
lettered panel of a composite. ``title`` is the panel header in a composite;
stand-alone figures carry no title (journal style, see :mod:`.style`).

* :func:`plot_fst_scan` — sliding-window F_ST along the molecule, optional
  dashed boxes over regions of interest;
* :func:`plot_pi_ratio` — π_A / π_B in the same windows, peaks marked with a red
  triangle and the overlapping gene in italics;
* :func:`plot_fst_pi_stack` — the two above stacked on one shared x axis;
* :func:`plot_sv_schematic` — a few taxa, one row each: homologous blocks in
  blue, structural-variant blocks in red, dashed lines joining homologous edges,
  SV sites with reference coordinates underneath;
* :func:`plot_sv_frequency` — stacked present/absent bars per group for each SV,
  with Fisher / chi-square tests and BH-adjusted stars;
* :func:`plot_presence_tracks` — dozens of samples over one SV interval:
  covered blocks coloured by group, sample annotation strips on the left, SV
  sites and gene arrows underneath.

The numbers come from :mod:`.popgen_data` (F_ST estimators of
``population``, the π estimator of ``diversity``, graph-path / PAF coverage of
the reference); drawing lives in :mod:`.popgen_render`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from organelleverse._public_validation import validated_public

from ..core.errors import OrganelleInputError, OrganelleParameterError
from . import popgen_data as data
from . import popgen_render
from .plot_object import OrganellePlot, jsonable, plot_result

#: Okabe-Ito, colour-blind safe; group colours are taken in this order.
PALETTE = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#882255",
    "#999933",
)
SOURCE_COLORS = {"Mt": "#4C72B0", "Pt": "#55A868", "Nuclear": "#C44E52"}

POPGEN_OPERATIONS = frozenset(
    {
        "visualization.plot_fst_scan",
        "visualization.plot_pi_ratio",
        "visualization.plot_fst_pi_stack",
        "visualization.plot_sv_schematic",
        "visualization.plot_sv_frequency",
        "visualization.plot_presence_tracks",
    }
)


# ---------------------------------------------------------------------------
# shared style
# ---------------------------------------------------------------------------


def _group_colors(labels: Sequence[str], colors: Mapping[str, str] | None = None) -> dict[str, str]:
    chosen = dict(colors or {})
    free = [c for c in PALETTE if c not in chosen.values()]
    for label in labels:
        if label not in chosen:
            chosen[label] = free.pop(0) if free else PALETTE[len(chosen) % len(PALETTE)]
    return chosen










def _stars(q: float | None) -> str:
    if q is None:
        return "n.t."
    return "***" if q < 0.001 else "**" if q < 0.01 else "*" if q < 0.05 else "ns"


def _merge_runs(windows: Sequence[Mapping[str, Any]], key: str, cut: float) -> list[dict[str, Any]]:
    """Merge overlapping/adjacent windows with ``key >= cut`` into regions (peak kept)."""
    regions: list[dict[str, Any]] = []
    for w in windows:
        value = w.get(key)
        if value is None or value < cut:
            continue
        if regions and w["start"] <= regions[-1]["end"] + 1:
            region = regions[-1]
            region["end"] = max(region["end"], w["end"])
            if value > region["peak_value"]:
                region.update(peak_value=value, peak_start=w["start"], peak_end=w["end"])
        else:
            regions.append(
                {
                    "start": w["start"],
                    "end": w["end"],
                    "peak_value": value,
                    "peak_start": w["start"],
                    "peak_end": w["end"],
                }
            )
    return regions


def _top_cut(values: Sequence[float], top_fraction: float) -> float | None:
    ranked = sorted((v for v in values if v is not None), reverse=True)
    if not ranked or top_fraction <= 0:
        return None
    return ranked[max(1, math.ceil(len(ranked) * top_fraction)) - 1]


def _genes_in_region(
    genbank: str | Path | None, start: int, end: int, organelle: str = "mito"
) -> list[dict[str, Any]]:
    """Gene/tRNA/rRNA parts overlapping ``start..end`` (1-based inclusive) of one record."""
    if genbank is None:
        return []
    from Bio import SeqIO

    from .features import annotation_rows
    from .ogdraw import GENE_COLORS

    records = list(SeqIO.parse(str(genbank), "genbank"))
    if len(records) != 1:
        raise OrganelleInputError(
            code="visualization.popgen.genbank",
            message=f"Expected one GenBank record for the reference, found {len(records)}.",
        )
    genes = []
    for row in annotation_rows(records[0], organelle, strict=False):
        for exon in row["exons"]:
            if exon["end"] < start or exon["start"] > end:
                continue
            genes.append(
                {
                    "name": row["gene"],
                    "start": exon["start"],
                    "end": exon["end"],
                    "strand": int(exon["strand"]),
                    "feature_type": row["feature_type"],
                    "category": row["category"],
                    "color": GENE_COLORS.get(row["category"], "#9ca3af"),
                }
            )
    # one entry per (gene, part): CDS supersede gene spans in annotation_rows already
    unique = {(g["name"], g["start"], g["end"]): g for g in genes}
    return sorted(unique.values(), key=lambda g: (g["start"], g["end"]))


def _gene_features(genbank: str | Path | None) -> list[tuple[int, int, str]]:
    if genbank is None:
        return []
    from ..diversity.diversity import _feature_intervals

    return _feature_intervals(genbank)


def _peak_label(start: int, end: int, features: list[tuple[int, int, str]]) -> str | None:
    if not features:
        return None
    from ..diversity.diversity import _label_region

    return _label_region(start, end, features)




# ---------------------------------------------------------------------------
# A. F_ST scan
# ---------------------------------------------------------------------------


@validated_public('visualization.plot_fst_scan')
def plot_fst_scan(
    vcf: str | Path | None = None,
    *,
    groups: Mapping[str, str] | None = None,
    pair: Sequence[str] | None = None,
    windows: Sequence[Mapping[str, Any]] | None = None,
    estimator: str = "heterozygosity",
    window_size: int = 5000,
    step: int = 1000,
    min_snps: int = 3,
    contig: str | None = None,
    genome_length: int | None = None,
    reference_sample: str | None = None,
    highlight: Sequence[Mapping[str, int | str | None]] | str | None = None,
    top_fraction: float = 0.05,
    group_colors: Mapping[str, str] | None = None,
    title: str | None = None,
    organelle: str = "mito",
    width: str | float = "double",
    height: float | None = None,
    dpi: int = 600,
) -> OrganellePlot:
    """Sliding-window F_ST between two groups along the organelle genome.

    From a VCF, ``groups`` maps sample → group label (exactly two groups, or name
    them with ``pair``); per-site values use ``population``'s estimator
    (``heterozygosity`` default as ``fst_scan``, ``weir_cockerham`` or
    ``hudson`` — they are not interchangeable) and windows tile the molecule
    from position 1 (``window_size``/``step``; windows with < ``min_snps`` SNPs
    are gaps). ``reference_sample`` is the sample whose sequence is the VCF
    reference when it has no VCF column (``vg deconstruct -P``): all REF. Alternatively pass precomputed ``windows`` (start, end, fst) and
    ``genome_length``.

    ``highlight`` draws dashed boxes: a list of ``{start, end, label}`` or
    ``"auto"`` for the merged top-``top_fraction`` windows. The dashed horizontal
    line is the top-``top_fraction`` cut.
    """
    if windows is None:
        if vcf is None or groups is None:
            raise OrganelleParameterError(
                code="visualization.popgen.fst_input",
                message="Pass vcf= with groups=, or precomputed windows= with genome_length=.",
            )
        scan = data.fst_windows(
            vcf,
            groups,
            pair=pair,
            estimator=estimator,
            window_size=window_size,
            step=step,
            min_snps=min_snps,
            contig=contig,
            genome_length=genome_length,
            reference_sample=reference_sample,
        )
    else:
        rows = [dict(w) for w in windows]
        length = genome_length or max((int(w["end"]) for w in rows), default=0)
        if not length:
            raise OrganelleParameterError(
                code="visualization.popgen.length", message="Pass genome_length with windows=."
            )
        scan = {
            "contig": contig,
            "genome_length": length,
            "estimator": estimator,
            "groups": list(pair or ("group A", "group B")),
            "group_sizes": {},
            "window_size": None,
            "step": None,
            "min_snps": None,
            "n_snps": None,
            "windows": rows,
            "scored_windows": sum(w.get("fst") is not None for w in rows),
            "max_fst": max((w["fst"] for w in rows if w.get("fst") is not None), default=None),
        }
    cut = _top_cut([w["fst"] for w in scan["windows"]], top_fraction)
    boxes = _highlights(highlight, scan["windows"], "fst", cut)
    metrics = jsonable(
        {
            **scan,
            "top_fraction": top_fraction,
            "top_cut": cut,
            "highlights": boxes,
            "colors": _group_colors(scan["groups"], group_colors),
            "title": title,
            "width": width,
            "height": height,
            "dpi": dpi,
            "coordinate_system": "1-based inclusive reference coordinates",
        }
    )
    return _plot("plot_fst_scan", metrics, organelle, f"F_ST scan prepared ({scan['scored_windows']} scored windows).")


def _highlights(highlight, windows, key, cut) -> list[dict[str, Any]]:
    if highlight is None:
        return []
    if highlight == "auto":
        if cut is None:
            return []
        return [
            {"start": r["start"], "end": r["end"], "label": None}
            for r in _merge_runs(windows, key, cut)
        ]
    if isinstance(highlight, str):
        raise OrganelleParameterError(
            code="visualization.popgen.highlight",
            message="highlight must be 'auto' or a list of {start, end, label}.",
        )
    boxes = []
    for item in highlight:
        start, end = int(item["start"]), int(item["end"])
        if end < start:
            raise OrganelleParameterError(
                code="visualization.popgen.highlight", message="highlight end must be >= start."
            )
        boxes.append({"start": start, "end": end, "label": item.get("label")})
    return boxes












# ---------------------------------------------------------------------------
# B. π ratio
# ---------------------------------------------------------------------------


@validated_public('visualization.plot_pi_ratio')
def plot_pi_ratio(
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
    genbank: str | Path | None = None,
    top_fraction: float = 0.05,
    label_top: int = 6,
    log_scale: bool = False,
    group_colors: Mapping[str, str] | None = None,
    title: str | None = None,
    organelle: str = "mito",
    width: str | float = "double",
    height: float | None = None,
    dpi: int = 600,
) -> OrganellePlot:
    """π ratio between two groups (π_A / π_B, ``pair`` order) along the genome.

    π per window and group uses the Nei (1987) estimator of ``diversity``
    (mean pairwise differences per comparable site), from an aligned FASTA
    (``alignment``; coordinates on ``reference``, the first record by default)
    or from a ``vcf`` + ``reference_fasta`` pseudo-alignment (SNP alleles, N for
    missing calls and for samples carrying a non-SNP ALT). Windows where either
    group has < ``min_sites`` comparable sites have no ratio.

    Peaks are the merged top-``top_fraction`` windows; the ``label_top`` highest
    get a red triangle and, with ``genbank`` (the reference record), the gene
    (or ``a-b`` spacer) they overlap in italics.
    """
    scan = data.pi_ratio_windows(
        groups,
        alignment=alignment,
        vcf=vcf,
        reference_fasta=reference_fasta,
        reference=reference,
        pair=pair,
        window_size=window_size,
        step=step,
        gap_mode=gap_mode,
        min_sites=min_sites,
        pseudocount=pseudocount,
        contig=contig,
        reference_sample=reference_sample,
    )
    cut = _top_cut([w["ratio"] for w in scan["windows"]], top_fraction)
    features = _gene_features(genbank)
    peaks = []
    if cut is not None:
        regions = sorted(
            _merge_runs(scan["windows"], "ratio", cut), key=lambda r: r["peak_value"], reverse=True
        )
        for region in regions[: max(0, label_top)]:
            label = _peak_label(region["peak_start"], region["peak_end"], features)
            peaks.append({**region, "label": label})
        peaks.sort(key=lambda p: p["peak_start"])
    if vcf is not None:
        scan["contig"] = data._single_contig(vcf, contig)[0]
    else:
        scan["contig"] = scan.get("reference")
    metrics = jsonable(
        {
            **scan,
            "top_fraction": top_fraction,
            "top_cut": cut,
            "peaks": peaks,
            "log_scale": log_scale,
            "colors": _group_colors(scan["groups"], group_colors),
            "title": title,
            "width": width,
            "height": height,
            "dpi": dpi,
            "coordinate_system": "1-based inclusive reference coordinates",
            "estimator": "Nei 1987 pi (diversity._window_pi_s)",
        }
    )
    return _plot(
        "plot_pi_ratio", metrics, organelle, f"π ratio prepared ({scan['scored_windows']} windows, {len(peaks)} labelled peaks)."
    )








# ---------------------------------------------------------------------------
# A + B stacked
# ---------------------------------------------------------------------------


def plot_fst_pi_stack(
    fst: Any,
    pi_ratio: Any,
    *,
    title: str | None = None,
    panel_labels: Sequence[str] = ("a", "b"),
    organelle: str = "mito",
    width: str | float = "double",
    height: float | None = None,
    dpi: int = 600,
) -> OrganellePlot:
    """Stack an F_ST scan over a π-ratio scan on one shared x axis.

    Both inputs are the Results of :func:`plot_fst_scan` and :func:`plot_pi_ratio`
    (live or deserialized); their genome lengths must match (same reference).
    """
    fst_m, pi_m = _metrics_of(fst, "plot_fst_scan"), _metrics_of(pi_ratio, "plot_pi_ratio")
    if fst_m["genome_length"] != pi_m["genome_length"]:
        raise OrganelleParameterError(
            code="visualization.popgen.stack_length",
            message=f"F_ST ({fst_m['genome_length']} bp) and π ({pi_m['genome_length']} bp) "
            "scans are on different references.",
        )
    metrics = jsonable(
        {
            "fst": fst_m,
            "pi_ratio": pi_m,
            "genome_length": fst_m["genome_length"],
            "panel_labels": list(panel_labels),
            "title": title,
            "width": width,
            "height": height,
            "dpi": dpi,
        }
    )
    return _plot("plot_fst_pi_stack", metrics, organelle, "Stacked F_ST / π-ratio panels prepared.")


def _metrics_of(result: Any, kind: str) -> dict[str, Any]:
    from ..core.frozen import thaw_json

    metrics = thaw_json(result.metrics) if hasattr(result, "metrics") else dict(result)
    if metrics.get("plot_kind") != kind:
        raise OrganelleParameterError(
            code="visualization.popgen.stack_input",
            message=f"Expected a {kind} Result, got {metrics.get('plot_kind')!r}.",
        )
    return metrics




# ---------------------------------------------------------------------------
# C. SV schematic
# ---------------------------------------------------------------------------


@validated_public('visualization.plot_sv_schematic')
def plot_sv_schematic(
    gfa: str | Path | None = None,
    *,
    reference: str | None = None,
    start: int | None = None,
    end: int | None = None,
    samples: Sequence[str] | None = None,
    labels: Mapping[str, str] | None = None,
    min_sv_length: int = 100,
    homology_fraction: float = 1.0,
    min_span: float = 0.8,
    max_insert: int | None = None,
    schematic: Mapping[str, Any] | None = None,
    title: str | None = None,
    organelle: str = "mito",
    width: str | float = "double",
    row_height: float | None = None,
    dpi: int = 600,
) -> OrganellePlot:
    """Homologous (blue) vs SV (red) blocks of one locus, one row per taxon.

    From a pangenome ``gfa`` (PGGB/odgi paths; ``reference`` path or sample,
    ``start..end`` on it, the ``samples`` to show — the reference row is
    added first): see :func:`popgen_data.sv_schematic_rows` for the block rules.
    Dashed lines join the same homologous edge on neighbouring rows; SV sites
    are listed under the rows with reference coordinates (``ins.`` where the
    reference has no sequence there). ``schematic`` accepts that function's
    output directly.
    """
    if schematic is None:
        if gfa is None or reference is None or start is None or end is None or not samples:
            raise OrganelleParameterError(
                code="visualization.popgen.schematic_input",
                message="Pass gfa=, reference=, start=, end= and samples=, or schematic=.",
            )
        schematic = data.sv_schematic_rows(
            gfa,
            reference,
            start,
            end,
            samples,
            min_sv_length=min_sv_length,
            max_insert=max_insert,
            homology_fraction=homology_fraction,
            min_span=min_span,
            labels=labels,
        )
    metrics = jsonable(
        {
            **dict(schematic),
            "title": title,
            "width": width,
            "row_height": row_height,
            "dpi": dpi,
            "coordinate_system": "row blocks: 0-based offsets from the left anchor; SV sites: 1-based reference",
        }
    )
    return _plot(
        "plot_sv_schematic",
        metrics,
        organelle,
        f"SV schematic prepared ({len(metrics['rows'])} rows, {len(metrics['svs'])} SV sites).",
    )










# ---------------------------------------------------------------------------
# D. SV presence/absence frequency
# ---------------------------------------------------------------------------


@validated_public('visualization.plot_sv_frequency')
def plot_sv_frequency(
    svs: Sequence[Mapping[str, str | int | Mapping[str, bool | None] | Mapping[str, float] | None]] | None = None,
    *,
    groups: Mapping[str, str],
    gfa: str | Path | None = None,
    reference: str | None = None,
    coverage: Mapping[str, Any] | None = None,
    units: Mapping[str, str] | None = None,
    collapse_identical: bool = True,
    group_order: Sequence[str] | None = None,
    test: str = "exact",
    min_length: int = 500,
    max_svs: int = 12,
    sort: str = "significance",
    group_colors: Mapping[str, str] | None = None,
    title: str | None = None,
    organelle: str = "mito",
    width: str | float = "double",
    height: float | None = None,
    dpi: int = 600,
) -> OrganellePlot:
    """Stacked present/absent bars per group for each SV, with a group-difference test.

    ``svs`` rows carry ``id``, ``start``, ``end`` and ``presence`` (sample ->
    True / False / None), e.g. :func:`popgen_data.presence_svs`; or pass ``gfa``
    + ``reference`` (or a ``coverage`` dict) to call SVs >= ``min_length`` bp
    here (a sample covering 20-80 % of an SV is not called: missing).

    **Independence.** The unit counted is an independent molecule, not a file:
    ``units`` maps sample -> unit; with ``gfa`` and ``collapse_identical``
    (default) samples whose graph paths spell the identical mitogenome
    (rotation / reverse-complement invariant) are one unit
    (:func:`popgen_data.identical_path_groups`). A unit's call is the majority
    of its members. Units still share ancestry (organelle genomes do not
    recombine), so the test compares frequencies among the sampled
    haplotypes, not lineages.

    **Test** per SV on the groups x (present, absent) table of units, missing
    units excluded (the denominator of each group is its called units):
    ``exact`` (default; two-sided Fisher's exact for two groups, its
    Freeman-Halton k x 2 extension for more) or ``chi2`` (Pearson, no
    continuity correction; the minimum expected count is reported because the
    approximation needs >= 5). p-values are BH-adjusted across every SV tested;
    stars are on q (* < 0.05, ** < 0.01, *** < 0.001); an SV with no variation
    among called units is not tested (``n.t.``). ``metrics["svs"]`` has, per SV
    and group, present / absent / missing units, the denominators, p, q and
    stars; ``metrics["caption"]`` states the method for the figure legend.
    """
    if svs is None:
        if coverage is None:
            if gfa is None or reference is None:
                raise OrganelleParameterError(
                    code="visualization.popgen.sv_input",
                    message="Pass svs=, coverage=, or gfa= with reference=.",
                )
            coverage = data.reference_coverage_from_gfa(gfa, reference, samples=list(groups))
        svs = data.presence_svs(coverage["coverage"], coverage["genome_length"], min_length=min_length)
    svs = [dict(sv) for sv in svs]
    if not svs:
        raise OrganelleInputError(code="visualization.popgen.no_svs", message="No polymorphic SVs to draw.")
    labels = data.group_order(groups, group_order)
    if test == "auto":
        test = "exact"
    if test == "fisher":
        if len(labels) != 2:
            raise OrganelleParameterError(
                code="visualization.popgen.test",
                message="Fisher's exact test is the 2-group case; use test='exact' (Freeman-Halton) for more.",
            )
        test = "exact"
    if test not in ("exact", "chi2"):
        raise OrganelleParameterError(
            code="visualization.popgen.test", message="test must be 'exact' or 'chi2'."
        )
    grouped = {s: g for s, g in groups.items() if g in labels}
    if units is not None:
        unit_of = {s: str(units.get(s, s)) for s in grouped}
        independence = "units given by the caller"
    elif gfa is not None and collapse_identical:
        found = data.identical_path_groups(gfa, list(grouped))
        unit_of = {s: found[s] for s in grouped}
        independence = "samples with identical mitogenome sequence (graph paths, rotation/strand invariant) counted once"
    else:
        unit_of = {s: s for s in grouped}
        independence = "every sample counted as independent (identity not checked)"
    collapsed = []
    by_unit: dict[str, list[str]] = {}
    for sample, unit in unit_of.items():
        by_unit.setdefault(unit, []).append(sample)
    for unit, members in by_unit.items():
        if len(members) > 1:
            collapsed.append({"unit": unit, "samples": sorted(members), "groups": sorted({grouped[s] for s in members})})
    table = data.sv_presence_table(svs, grouped, units=unit_of)
    rows = []
    for sv, counts in zip(svs, table, strict=True):
        cells = {g: counts["counts"][g] for g in labels}
        p, stat, min_expected = _sv_test([cells[g] for g in labels], test)
        rows.append(
            {
                "id": sv["id"],
                "start": sv.get("start"),
                "end": sv.get("end"),
                "length": sv.get("length") or (sv["end"] - sv["start"] + 1 if sv.get("end") else None),
                "counts": cells,
                "denominator": {g: cells[g]["present"] + cells[g]["absent"] for g in labels},
                "missing": sum(cells[g]["missing"] for g in labels),
                "statistic": stat,
                "min_expected": min_expected,
                "p": p,
            }
        )
    for row, q in zip(rows, data.benjamini_hochberg([r["p"] for r in rows]), strict=True):
        row["q"] = q
        row["stars"] = _stars(q)
    if sort == "significance":
        rows.sort(key=lambda r: (r["p"] is None, r["p"] if r["p"] is not None else 1.0, r["start"] or 0))
    elif sort != "position":
        raise OrganelleParameterError(
            code="visualization.popgen.sort", message="sort must be 'significance' or 'position'."
        )
    tested_total = len(rows)
    tested = sum(r["p"] is not None for r in rows)
    rows = rows[:max_svs] if max_svs else rows
    rows.sort(key=lambda r: r["start"] or 0)
    n_samples = {g: sum(1 for x in grouped.values() if x == g) for g in labels}
    n_units = {g: len({unit_of[s] for s, x in grouped.items() if x == g}) for g in labels}
    method = (
        "two-sided Fisher's exact test" if test == "exact" and len(labels) == 2
        else "Freeman\u2013Halton exact test" if test == "exact"
        else "Pearson \u03c7\u00b2 test (no continuity correction)"
    )
    missing_total = sum(r["missing"] for r in rows)
    units_text = ", ".join(
        f"{g} {n_units[g]}" + (f" of {n_samples[g]}" if n_units[g] != n_samples[g] else "") for g in labels
    )
    caption = (
        f"{method} per SV on group \u00d7 presence counts of independent mitogenomes "
        f"(n = {units_text}; {sum(len(c['samples']) - 1 for c in collapsed)} identical duplicate(s) collapsed). "
        f"Denominator: genomes with a called state; missing (20\u201380% covered) excluded: {missing_total} calls "
        f"across the SVs shown. BH-adjusted over {tested} tested SVs: * q < 0.05, ** q < 0.01, *** q < 0.001; "
        "n.t., no variation among called genomes."
    )
    low_expected = [r["id"] for r in rows if r["min_expected"] is not None and r["min_expected"] < 5]
    metrics = jsonable(
        {
            "groups": labels,
            "group_sizes": n_samples,
            "group_units": n_units,
            "unit": independence,
            "collapsed_units": collapsed,
            "missing_rule": "covered fraction strictly between absent_max (0.2) and present_min (0.8) of the SV",
            "test": test,
            "test_description": method,
            "multiple_testing": "Benjamini\u2013Hochberg across every SV tested (svs_tested)",
            "svs_called": tested_total,
            "svs_tested": tested,
            "low_expected_svs": low_expected,
            "svs": rows,
            "caption": caption,
            "caption_short": (
                f"{method}; n = " + " / ".join(f"{n_units[g]}" for g in labels)
                + " genomes; BH q (full method in the legend table)"
            ),
            "colors": _group_colors(labels, group_colors),
            "title": title,
            "width": width,
            "height": height,
            "dpi": dpi,
        }
    )
    significant = sum(1 for r in rows if r["q"] is not None and r["q"] < 0.05)
    return _plot(
        "plot_sv_frequency",
        metrics,
        organelle,
        f"SV frequency prepared ({len(rows)} SVs, {significant} with q < 0.05, {method}).",
        flags=("chi2_low_expected_counts",) if low_expected else (),
    )


def _sv_test(
    counts: Sequence[Mapping[str, int]], method: str
) -> tuple[float | None, float | None, float | None]:
    """(p, statistic, minimum expected count) on the called units of each group."""
    present = [c["present"] for c in counts]
    absent = [c["absent"] for c in counts]
    if method == "exact":
        return data.exact_presence_test(present, absent), None, None
    keep = [(p, a) for p, a in zip(present, absent, strict=True) if p + a]
    if len(keep) < 2 or sum(p for p, _ in keep) in (0, sum(p + a for p, a in keep)):
        return None, None, None
    from scipy import stats

    chi2, p, _dof, expected = stats.chi2_contingency([list(x) for x in keep], correction=False)
    return float(p), float(chi2), float(expected.min())


# ---------------------------------------------------------------------------
# E. multi-sample presence tracks
# ---------------------------------------------------------------------------


@validated_public('visualization.plot_presence_tracks')
def plot_presence_tracks(
    coverage: Mapping[str, Any] | None = None,
    *,
    gfa: str | Path | None = None,
    paf: str | Path | None = None,
    reference: str | None = None,
    start: int,
    end: int,
    groups: Mapping[str, str],
    group_order: Sequence[str] | None = None,
    annotations: Mapping[str, Mapping[str, str]] | None = None,
    annotation_colors: Mapping[str, Mapping[str, str]] | None = None,
    sample_labels: Mapping[str, str] | None = None,
    svs: Sequence[Mapping[str, str | int]] | str | None = "auto",
    min_sv_length: int = 300,
    genbank: str | Path | None = None,
    min_block: int = 50,
    group_colors: Mapping[str, str] | None = None,
    title: str | None = None,
    organelle: str = "mito",
    width: str | float = "double",
    row_height: float | None = None,
    dpi: int = 600,
) -> OrganellePlot:
    """Which samples carry each part of one reference interval (SV region view).

    One row per sample of ``groups``; each block is reference sequence that
    sample carries (graph node sharing from ``gfa``, alignments from ``paf``, or
    a ``coverage`` dict from :mod:`popgen_data`), coloured by group. Rows are
    ordered by group, then by carried pattern. On the left, a square per row
    for the group and for each ``annotations`` column (e.g. ``{"DNA source":
    {sample: "Mt"}}``; ``Mt``/``Pt``/``Nuclear`` have fixed colours). Underneath,
    SV sites (``"auto"`` calls polymorphic presence segments ≥ ``min_sv_length``
    in the interval; or pass ``{id, start, end}`` rows) and the reference
    ``genbank`` genes as arrows. Gaps shorter than ``min_block`` are closed.
    """
    if coverage is None:
        if gfa is not None:
            coverage = data.reference_coverage_from_gfa(gfa, reference, samples=list(groups))
        elif paf is not None:
            coverage = data.coverage_from_paf(paf, reference=reference)
        else:
            raise OrganelleParameterError(
                code="visualization.popgen.coverage_input",
                message="Pass coverage=, gfa= (with reference=) or paf=.",
            )
    length = int(coverage["genome_length"])
    if not 1 <= start < end <= length:
        raise OrganelleParameterError(
            code="visualization.popgen.region",
            message=f"Region {start}..{end} must lie within 1..{length}.",
        )
    labels = data.group_order(groups, group_order)
    samples = [s for s in groups if groups[s] in labels]
    missing = [s for s in samples if s not in coverage["coverage"]]
    if missing:
        raise OrganelleInputError(
            code="visualization.popgen.unknown_sample",
            message=f"Samples without coverage: {', '.join(missing[:10])}.",
        )
    blocks = {}
    for sample in samples:
        clipped = [
            (max(a, start), min(b, end))
            for a, b in coverage["coverage"][sample]
            if b >= start and a <= end
        ]
        blocks[sample] = data.merge_intervals(clipped, gap=min_block)
    if svs == "auto":
        local = {s: blocks[s] for s in samples}
        shifted = {s: [(a - start + 1, b - start + 1) for a, b in v] for s, v in local.items()}
        called = data.presence_svs(shifted, end - start + 1, min_length=min_sv_length)
        sv_rows = [
            {"id": sv["id"], "start": sv["start"] + start - 1, "end": sv["end"] + start - 1}
            for sv in called
        ]
    else:
        sv_rows = [
            {"id": str(sv["id"]), "start": int(sv["start"]), "end": int(sv["end"])}
            for sv in (svs or [])
        ]
    patterns = {
        s: tuple(data.covered_fraction(blocks[s], sv["start"], sv["end"]) >= 0.5 for sv in sv_rows)
        for s in samples
    }
    carried = {s: sum(b - a + 1 for a, b in blocks[s]) for s in samples}
    order = sorted(
        samples,
        key=lambda s: (labels.index(groups[s]), tuple(not x for x in patterns[s]), -carried[s], s),
    )
    columns = {"Group": {s: groups[s] for s in samples}}
    for name, values in (annotations or {}).items():
        columns[str(name)] = {s: str(values.get(s, "NA")) for s in samples}
    palette = {"Group": _group_colors(labels, group_colors)}
    for name, values in columns.items():
        if name == "Group":
            continue
        cats = list(dict.fromkeys(values[s] for s in order))
        given = dict((annotation_colors or {}).get(name, {}))
        auto = [c for c in ("#8DA0CB", "#FC8D62", "#66C2A5", "#E78AC3", "#A6D854", "#FFD92F", "#B3B3B3") if c not in given.values()]
        for cat in cats:
            if cat not in given:
                given[cat] = SOURCE_COLORS.get(cat) or (auto.pop(0) if auto else "#B3B3B3")
        palette[name] = given
    genes = _genes_in_region(genbank, start, end, organelle)
    metrics = jsonable(
        {
            "source": coverage.get("source"),
            "reference": coverage.get("reference"),
            "genome_length": length,
            "region": {"start": start, "end": end},
            "groups": labels,
            "group_sizes": {g: sum(1 for s in samples if groups[s] == g) for g in labels},
            "rows": [
                {
                    "sample": s,
                    "label": (sample_labels or {}).get(s, s),
                    "group": groups[s],
                    "blocks": blocks[s],
                    "carried_bp": carried[s],
                    "annotations": {name: columns[name][s] for name in columns},
                }
                for s in order
            ],
            "annotation_columns": list(columns),
            "palette": palette,
            "svs": sv_rows,
            "genes": genes,
            "min_block": min_block,
            "title": title,
            "width": width,
            "row_height": row_height,
            "dpi": dpi,
            "coordinate_system": "1-based inclusive reference coordinates",
        }
    )
    return _plot(
        "plot_presence_tracks",
        metrics,
        organelle,
        f"Presence tracks prepared ({len(order)} samples, {len(sv_rows)} SV sites, {len(genes)} gene parts).",
    )




# ---------------------------------------------------------------------------
# plumbing
# ---------------------------------------------------------------------------

def _sidecar(kind: str, m: Mapping[str, Any]):
    """The ``<stem>.tsv`` table written next to every rendered figure."""
    write_tsv = popgen_render.write_tsv

    def write(path: Path) -> None:
        if kind == "plot_fst_scan":
            write_tsv(path, m["windows"], ["start", "end", "fst", "n_snps"])
        elif kind == "plot_pi_ratio":
            write_tsv(
                path,
                m["windows"],
                ["start", "end", "pi_a", "pi_b", "ratio", "pi_a_sites", "pi_b_sites",
                 "pi_a_segregating", "pi_b_segregating"],
            )
        elif kind == "plot_fst_pi_stack":
            by_start = {w["start"]: dict(w) for w in m["pi_ratio"]["windows"]}
            rows = [
                {**w, **{k: v for k, v in by_start.get(w["start"], {}).items() if k not in ("start", "end")}}
                for w in m["fst"]["windows"]
            ]
            write_tsv(path, rows, ["start", "end", "fst", "n_snps", "pi_a", "pi_b", "ratio"])
        elif kind == "plot_sv_schematic":
            rows = [{"row": r["label"], "sample": r["sample"], **b} for r in m["rows"] for b in r["blocks"]]
            write_tsv(path, rows, ["row", "sample", "kind", "sv", "start", "end", "nodes"])
            sv_rows = [{**sv, "carriers": ",".join(sv.get("carriers", []))} for sv in m["svs"]]
            write_tsv(path, sv_rows, ["id", "ref_start", "ref_end", "in_reference", "carriers"], ".svs.tsv")
        elif kind == "plot_sv_frequency":
            labels = m["groups"]
            rows = []
            for sv in m["svs"]:
                keys = ("id", "start", "end", "length", "statistic", "min_expected", "p", "q", "stars")
                row = {k: sv.get(k) for k in keys}
                for g in labels:
                    for state in ("present", "absent", "missing"):
                        row[f"{g} {state}"] = sv["counts"][g][state]
                    row[f"{g} denominator"] = sv["denominator"][g]
                rows.append(row)
            fields = ["id", "start", "end", "length"]
            for g in labels:
                fields += [f"{g} present", f"{g} absent", f"{g} missing", f"{g} denominator"]
            write_tsv(path, rows, [*fields, "statistic", "min_expected", "p", "q", "stars"])
        elif kind == "plot_presence_tracks":
            rows = [
                {"row": i, "label": r["label"], "sample": r["sample"], "group": r["group"], "start": a, "end": b}
                for i, r in enumerate(m["rows"], 1)
                for a, b in r["blocks"]
            ]
            write_tsv(path, rows, ["row", "label", "sample", "group", "start", "end"])
            index = [
                {
                    "row": i,
                    "label": r["label"],
                    "sample": r["sample"],
                    "group": r["group"],
                    "carried_bp": r["carried_bp"],
                    **{f"annotation:{k}": v for k, v in r["annotations"].items()},
                }
                for i, r in enumerate(m["rows"], 1)
            ]
            extra = sorted({k for r in index for k in r if k.startswith("annotation:")})
            write_tsv(path, index, ["row", "label", "sample", "group", "carried_bp", *extra], ".rows.tsv")
            svs = [{**sv, "length": sv["end"] - sv["start"] + 1} for sv in m["svs"]]
            write_tsv(path, svs, ["id", "start", "end", "length"], ".svs.tsv")

    return write


def _attach(plot: OrganellePlot, kind: str, metrics: Mapping[str, Any]) -> OrganellePlot:
    """Stand-alone renderer plus the panel drawer used by ``compose_figure``."""
    from .plot_object import PanelDrawer

    def render(path):
        return popgen_render.render(kind, Path(path), metrics, _sidecar(kind, metrics))

    def draw(ax, legend):
        return popgen_render.DRAWERS[kind](ax, metrics, legend)

    plot._renderer = render
    plot._panel = PanelDrawer(draw, popgen_render.aspect(kind, metrics), popgen_render.panel_header(kind, metrics))
    return plot


def _plot(
    op: str, metrics: dict[str, Any], organelle: str, summary: str, flags: tuple[str, ...] = ()
) -> OrganellePlot:
    try:
        popgen_render.figure_width_mm(metrics["width"])
    except (TypeError, ValueError) as exc:
        raise OrganelleParameterError(code="visualization.popgen.width", message=str(exc)) from exc
    plot = plot_result(
        op, lambda path: path, organelle=organelle, metrics=metrics, summary=summary, flags=flags,
        method="popgen_plots",
    )
    return _attach(plot, op, {**metrics, "plot_kind": op})


def _restore_popgen_plot(result) -> OrganellePlot:
    """Reattach renderer and panel drawer to a serialized popgen Result (metrics hold all drawn data)."""
    from ..core.frozen import thaw_json

    plot = OrganellePlot.model_validate(result.model_dump(mode="python"))
    metrics = thaw_json(plot.metrics)
    kind = metrics.get("plot_kind")
    if kind not in popgen_render.DRAWERS:
        raise OrganelleInputError(
            code="visualization.unrenderable_result",
            message=f"Not a popgen plot: {kind!r}.",
        )
    return _attach(plot, kind, metrics)


__all__ = [
    "POPGEN_OPERATIONS",
    "plot_fst_pi_stack",
    "plot_fst_scan",
    "plot_pi_ratio",
    "plot_presence_tracks",
    "plot_sv_frequency",
    "plot_sv_schematic",
]
