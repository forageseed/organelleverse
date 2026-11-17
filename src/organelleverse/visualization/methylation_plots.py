"""Plant organelle DNA methylation and NUMT/NUPT read-separation figures.

``plot_methylation_profile`` (alias ``methylation_profile``) renders the outputs
of ``ovasm check methylation`` or ``ovasm run ... --methylation`` at printed
journal size (:mod:`.style`):

* **Panel a** — sliding-window base-modification rate along each assembled
  organelle contig, contrasting the unfiltered modBAM pileup
  (``methylation.unfiltered.bed``, carrying nuclear NUMT/NUPT spikes) against
  the clean single-molecule organelle profile (``methylation.bed``), with
  optional high-identity nuclear transfer intervals shaded from ``transfer_paf``;
* **Panel b** — per-read ``CG`` 5mC bimodal histogram from
  ``reads.methylation.tsv`` separating clean organelle reads from hypermethylated
  and chimeric NUMT/NUPT reads, annotated with the auto-calibrated threshold and
  ROC AUC;
* **Panel c** — plant four-context modification summary (``CG``, ``CHG``,
  ``CHH``, ``6mA``) comparing unfiltered versus clean organelle read pools.

Inputs may be an ``ovasm`` methylation directory, a single-organelle ``ovasm run``
directory, a dual-organelle ``ovasm run --organelle both`` directory, or an
:class:`~organelleverse.core.data.OrganelleData` carrying a ``methylation``
artifact. The call is compute-only; ``ov.write(plot, "fig.pdf" | ".png" | ".svg")``
renders the figure and writes ``<stem>.windows.tsv`` alongside it.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..core.data import OrganelleData
from ..core.errors import OrganelleInputError, OrganelleParameterError
from .plot_object import OrganellePlot, PanelDrawer, plot_result
from .style import FONT_PT, MAX_HEIGHT_MM, add_panel_label, publication_style, width_inches

VALID_CONTEXTS = ("CG", "CHG", "CHH", "6mA")
CLEAN_COLOR = "#0072B2"
UNFILTERED_COLOR = "#D55E00"
THRESHOLD_COLOR = "#009E73"
CONTEXT_ALT_COLOR = "#CC79A7"
TRANSFER_FILL = "#FDE0D2"


@dataclass(frozen=True)
class _OrganelleMethylationBundle:
    label: str
    organelle: str
    directory: Path
    report: dict[str, Any]
    clean_bed: Path
    unfiltered_bed: Path | None
    reads_tsv: Path | None


def _resolve_methylation_bundles(
    source: str | Path | OrganelleData,
    organelle: str,
) -> list[_OrganelleMethylationBundle]:
    if isinstance(source, OrganelleData):
        art = source.artifacts.get("methylation") or source.artifacts.get("methylation_json")
        if art is None:
            raise OrganelleInputError(
                code="visualization.methylation.missing_artifact",
                message="OrganelleData has no 'methylation' or 'methylation_json' artifact.",
            )
        base = Path(art.uri)
        if base.is_file():
            base = base.parent
    else:
        base = Path(source)

    if base.is_file() and base.name == "methylation.json":
        base = base.parent

    if not base.exists():
        raise OrganelleInputError(
            code="visualization.methylation.not_found",
            message=f"Methylation path does not exist: {base}",
            details={"path": str(base)},
        )

    candidates: list[tuple[str, str, Path]] = []
    req_org = organelle.strip().lower()

    if (base / "methylation.json").is_file():
        inferred = "plastid" if "plast" in str(base).lower() else "mitochondrion"
        if req_org not in ("", "auto"):
            inferred = "plastid" if req_org in ("plastid", "plastome", "chloroplast", "chloro") else "mitochondrion"
        candidates.append((inferred.capitalize(), inferred, base))
    elif (base / "methylation" / "methylation.json").is_file():
        inferred = "plastid" if base.name.lower().startswith("plast") else "mitochondrion"
        if req_org not in ("", "auto"):
            inferred = "plastid" if req_org in ("plastid", "plastome", "chloroplast", "chloro") else "mitochondrion"
        candidates.append((inferred.capitalize(), inferred, base / "methylation"))
    else:
        for sub_name, org_key, title in (
            ("mitochondrion", "mitochondrion", "Mitochondrion"),
            ("plastid", "plastid", "Plastid"),
        ):
            if req_org not in ("", "auto", "both", "mixed") and not req_org.startswith(sub_name[:5]):
                continue
            mdir = base / sub_name / "methylation"
            if not (mdir / "methylation.json").is_file():
                mdir = base / sub_name
            if (mdir / "methylation.json").is_file():
                candidates.append((title, org_key, mdir))

    if not candidates:
        raise OrganelleInputError(
            code="visualization.methylation.invalid_directory",
            message=(
                f"No methylation.json found in {base} (expected an ovasm check methylation "
                "directory or an ovasm run --methylation output directory)."
            ),
            details={"path": str(base)},
        )

    bundles: list[_OrganelleMethylationBundle] = []
    for label, org_key, mdir in candidates:
        report = json.loads((mdir / "methylation.json").read_text(encoding="utf-8"))
        clean_bed = mdir / "methylation.bed"
        if not clean_bed.is_file():
            raise OrganelleInputError(
                code="visualization.methylation.missing_bed",
                message=f"Missing methylation.bed in {mdir}",
                details={"directory": str(mdir)},
            )
        unfilt = mdir / "methylation.unfiltered.bed"
        rtsv = mdir / "reads.methylation.tsv"
        bundles.append(
            _OrganelleMethylationBundle(
                label=label,
                organelle=org_key,
                directory=mdir,
                report=report,
                clean_bed=clean_bed,
                unfiltered_bed=unfilt if unfilt.is_file() else None,
                reads_tsv=rtsv if rtsv.is_file() else None,
            )
        )
    return bundles


def _window_bed_rates(
    bed_path: Path,
    contig_lengths: dict[str, int],
    *,
    context: str,
    window_bp: int,
) -> list[dict[str, Any]]:
    offsets: dict[str, int] = {}
    total_len = 0
    for cname, clen in contig_lengths.items():
        offsets[cname] = total_len
        total_len += clen
    if total_len <= 0:
        return []

    n_bins = max(1, int(math.ceil(total_len / window_bp)))
    mod_sum = np.zeros(n_bins, dtype=float)
    cov_sum = np.zeros(n_bins, dtype=float)
    site_cnt = np.zeros(n_bins, dtype=int)

    with open(bed_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip().split("\t")
            if len(parts) < 9:
                continue
            chrom, pos_s, _, ctx = parts[0], parts[1], parts[2], parts[3]
            if ctx != context or chrom not in offsets:
                continue
            pos = offsets[chrom] + int(pos_s)
            b = min(max(pos // window_bp, 0), n_bins - 1)
            mc = float(parts[7])
            uc = float(parts[8])
            mod_sum[b] += mc
            cov_sum[b] += mc + uc
            site_cnt[b] += 1

    rows: list[dict[str, Any]] = []
    for idx in range(n_bins):
        start = idx * window_bp
        end = min((idx + 1) * window_bp, total_len)
        rate = float(mod_sum[idx] / cov_sum[idx]) if cov_sum[idx] > 0 else None
        rows.append(
            {
                "start": start,
                "end": end,
                "mid_kb": round(0.5 * (start + end) / 1000.0, 4),
                "sites": int(site_cnt[idx]),
                "coverage": float(cov_sum[idx]),
                "mod_rate": rate,
            }
        )
    return rows


def _parse_reads_tsv(tsv_path: Path | None, min_read_cpgs: int) -> dict[str, Any]:
    clean_cgs: list[float] = []
    numt_cgs: list[float] = []
    clean_ctx = {c: [0.0, 0] for c in VALID_CONTEXTS}
    unfilt_ctx = {c: [0.0, 0] for c in VALID_CONTEXTS}

    if tsv_path is None or not tsv_path.is_file():
        return {
            "clean_cgs": clean_cgs,
            "numt_cgs": numt_cgs,
            "clean_context_rates": {},
            "unfiltered_context_rates": {},
        }

    with open(tsv_path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            cls = row.get("class", row.get("classification", ""))
            is_clean = cls in ("organelle", "clean_organelle")
            cpg_calls = int(row.get("cg_calls", 0))
            cg_frac = float(row.get("cg_frac", row.get("cg_5mc_frac", 0.0)))
            if cpg_calls >= min_read_cpgs:
                if is_clean:
                    clean_cgs.append(cg_frac)
                else:
                    numt_cgs.append(cg_frac)

            for ctx, ck, fk in (
                ("CG", "cg_calls", "cg_frac"),
                ("CHG", "chg_calls", "chg_frac"),
                ("CHH", "chh_calls", "chh_frac"),
                ("6mA", "m6a_calls", "m6a_frac"),
            ):
                if ck in row and fk in row:
                    calls = int(row[ck])
                    frac = float(row[fk])
                    unfilt_ctx[ctx][0] += calls * frac
                    unfilt_ctx[ctx][1] += calls
                    if is_clean:
                        clean_ctx[ctx][0] += calls * frac
                        clean_ctx[ctx][1] += calls

    def _rates(acc: dict[str, list[Any]]) -> dict[str, float]:
        return {
            k: round(v[0] / v[1], 6)
            for k, v in acc.items()
            if v[1] > 0
        }

    return {
        "clean_cgs": clean_cgs,
        "numt_cgs": numt_cgs,
        "clean_context_rates": _rates(clean_ctx),
        "unfiltered_context_rates": _rates(unfilt_ctx),
    }


def _parse_transfer_paf(
    paf_path: Path | None,
    contig_lengths: dict[str, int],
    *,
    min_length: int,
    min_identity: float,
) -> list[tuple[float, float]]:
    if paf_path is None:
        return []
    if not paf_path.is_file():
        raise OrganelleInputError(
            code="visualization.methylation.transfer_paf_not_found",
            message=f"transfer_paf file not found: {paf_path}",
            details={"path": str(paf_path)},
        )
    offsets: dict[str, int] = {}
    acc = 0
    for cname, clen in contig_lengths.items():
        offsets[cname] = acc
        acc += clen

    intervals: list[tuple[float, float]] = []
    with open(paf_path, "r", encoding="utf-8") as fh:
        for line in fh:
            cols = line.rstrip().split("\t")
            if len(cols) < 11:
                continue
            qname, qstart, qend = cols[0], int(cols[2]), int(cols[3])
            matches, blen = int(cols[9]), int(cols[10])
            ident = matches / blen if blen > 0 else 0.0
            if qname in offsets and (qend - qstart) >= min_length and ident >= min_identity:
                off = offsets[qname]
                intervals.append(((off + qstart) / 1000.0, (off + qend) / 1000.0))
    return intervals


def plot_methylation_profile(
    methylation: str | Path | OrganelleData,
    *,
    organelle: str = "auto",
    context: str = "CG",
    window_bp: int = 2000,
    min_read_cpgs: int = 5,
    transfer_paf: str | Path | None = None,
    transfer_min_length: int = 2000,
    transfer_min_identity: float = 0.98,
    width: str | float = "double",
    dpi: int = 600,
) -> OrganellePlot:
    """Prepare a multi-panel plant organelle methylation and NUMT/NUPT separation figure.

    Parameters
    ----------
    methylation:
        Output directory of ``ovasm check methylation``, an ``ovasm run --methylation``
        directory (single organelle or ``--organelle both``), a ``methylation.json``
        path, or an :class:`OrganelleData` carrying a ``methylation`` artifact.
    organelle:
        ``"auto"``, ``"mitochondrion"``, ``"plastid"``, or ``"both"``.
    context:
        Modification context for the genome-wide sliding-window track:
        ``"CG"``, ``"CHG"``, ``"CHH"``, or ``"6mA"``.
    window_bp:
        Sliding-window width in base pairs for Panel a (>= 200).
    min_read_cpgs:
        Minimum called ``CG`` sites on a read to include in Panel b (>= 1).
    transfer_paf:
        Optional PAF alignment of the target organelle assembly against nuclear
        chromosomes; intervals >= ``transfer_min_length`` bp at >=
        ``transfer_min_identity`` are shaded behind Panel a.
    width:
        Printed journal width: ``"single"`` (89 mm), ``"onehalf"`` (120 mm),
        ``"double"`` (183 mm), or millimetres.
    dpi:
        Rasterization DPI for PNG/TIFF output.
    """
    try:
        w_in = width_inches(width)
    except ValueError as exc:
        raise OrganelleParameterError(
            code="visualization.methylation.invalid_width",
            message=str(exc),
            details={"width": width},
        ) from exc

    if context not in VALID_CONTEXTS:
        raise OrganelleParameterError(
            code="visualization.methylation.invalid_context",
            message=f"context must be one of {VALID_CONTEXTS}, got {context!r}",
            details={"context": context},
        )
    if window_bp < 200:
        raise OrganelleParameterError(
            code="visualization.methylation.invalid_window",
            message="window_bp must be at least 200 bp.",
            details={"window_bp": window_bp},
        )
    if min_read_cpgs < 1 or dpi <= 0:
        raise OrganelleParameterError(
            code="visualization.methylation.invalid_parameters",
            message="min_read_cpgs >= 1 and dpi > 0 are required.",
            details={"min_read_cpgs": min_read_cpgs, "dpi": dpi},
        )

    bundles = _resolve_methylation_bundles(methylation, organelle)
    paf_path = Path(transfer_paf) if transfer_paf is not None else None

    prepared_tracks: list[dict[str, Any]] = []
    all_clean_cgs: list[float] = []
    all_numt_cgs: list[float] = []
    combined_clean_ctx: dict[str, list[float]] = {c: [] for c in VALID_CONTEXTS}
    combined_unfilt_ctx: dict[str, list[float]] = {c: [] for c in VALID_CONTEXTS}
    thresholds: list[float] = []
    aucs: list[float] = []

    for b in bundles:
        contigs = b.report.get("contigs") or []
        contig_lengths = {str(c["name"]): int(c["length"]) for c in contigs if "name" in c and "length" in c}
        total_bp = sum(contig_lengths.values())
        clean_win = _window_bed_rates(b.clean_bed, contig_lengths, context=context, window_bp=window_bp)
        unfilt_win = (
            _window_bed_rates(b.unfiltered_bed, contig_lengths, context=context, window_bp=window_bp)
            if b.unfiltered_bed is not None
            else []
        )
        transfers = _parse_transfer_paf(
            paf_path,
            contig_lengths,
            min_length=transfer_min_length,
            min_identity=transfer_min_identity,
        )
        read_stats = _parse_reads_tsv(b.reads_tsv, min_read_cpgs=min_read_cpgs)
        all_clean_cgs.extend(read_stats["clean_cgs"])
        all_numt_cgs.extend(read_stats["numt_cgs"])

        for ctx in VALID_CONTEXTS:
            if ctx in read_stats["clean_context_rates"]:
                combined_clean_ctx[ctx].append(read_stats["clean_context_rates"][ctx])
            if ctx in read_stats["unfiltered_context_rates"]:
                combined_unfilt_ctx[ctx].append(read_stats["unfiltered_context_rates"][ctx])

        params_meta = b.report.get("parameters") or {}
        reads_meta = b.report.get("reads") or {}
        eff_thr = params_meta.get("effective_max_cg_methylation")
        if isinstance(eff_thr, (int, float)):
            thresholds.append(float(eff_thr))
        auc_val = reads_meta.get("roc_auc_organelle_vs_numt")
        if isinstance(auc_val, (int, float)):
            aucs.append(float(auc_val))

        prepared_tracks.append(
            {
                "label": b.label,
                "organelle": b.organelle,
                "total_bp": total_bp,
                "contig_count": len(contig_lengths),
                "clean_windows": clean_win,
                "unfiltered_windows": unfilt_win,
                "transfers": transfers,
                "reads_summary": reads_meta,
                "clean_context_rates": read_stats["clean_context_rates"],
                "unfiltered_context_rates": read_stats["unfiltered_context_rates"],
            }
        )

    scope_org = "mixed" if len(prepared_tracks) > 1 else prepared_tracks[0]["organelle"]
    mean_thr = float(np.mean(thresholds)) if thresholds else None
    mean_auc = float(np.mean(aucs)) if aucs else None

    metrics = {
        "context": context,
        "window_bp": window_bp,
        "organelle_count": len(prepared_tracks),
        "clean_reads_plotted": len(all_clean_cgs),
        "numt_nupt_reads_plotted": len(all_numt_cgs),
        "effective_max_cg_methylation": mean_thr,
        "roc_auc_organelle_vs_numt": mean_auc,
        "tracks": [
            {
                "organelle": t["organelle"],
                "total_bp": t["total_bp"],
                "contig_count": t["contig_count"],
                "clean_context_rates": t["clean_context_rates"],
                "unfiltered_context_rates": t["unfiltered_context_rates"],
            }
            for t in prepared_tracks
        ],
    }

    def _draw_track_panel(ax, legend: bool = True) -> Sequence[Any]:
        t0 = prepared_tracks[0]
        xs = [w["mid_kb"] for w in t0["clean_windows"]]
        ys_clean = [w["mod_rate"] * 100.0 if w["mod_rate"] is not None else np.nan for w in t0["clean_windows"]]
        handles = []
        for s_kb, e_kb in t0["transfers"]:
            ax.axvspan(s_kb, e_kb, color=TRANSFER_FILL, alpha=0.65, lw=0, zorder=0)
        if t0["unfiltered_windows"]:
            ys_unf = [
                w["mod_rate"] * 100.0 if w["mod_rate"] is not None else np.nan
                for w in t0["unfiltered_windows"]
            ]
            (h_u,) = ax.plot(xs, ys_unf, color=UNFILTERED_COLOR, lw=0.85, label="Unfiltered modBAM")
            ax.fill_between(xs, 0, np.nan_to_num(ys_unf), color=UNFILTERED_COLOR, alpha=0.14)
            handles.append(h_u)
        (h_c,) = ax.plot(xs, ys_clean, color=CLEAN_COLOR, lw=1.1, label="Clean organelle")
        handles.append(h_c)
        ax.set_xlim(0, max(t0["total_bp"] / 1000.0, 0.1))
        ax.set_xlabel(f"{t0['label']} position (kb)", fontsize=FONT_PT["body"])
        ax.set_ylabel(f"{context} mod. (%)", fontsize=FONT_PT["body"])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if legend and handles:
            ax.legend(handles=handles, loc="upper right", fontsize=FONT_PT["tick"])
        return handles

    def _render(output_path: str | Path) -> Path:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        out_p = Path(output_path)
        h_mm = min(MAX_HEIGHT_MM, 112.0 if w_in * 25.4 >= 140.0 else 135.0)

        with publication_style():
            fig = plt.figure(figsize=(w_in, h_mm / 25.4), dpi=dpi)
            gs = GridSpec(
                2,
                2,
                figure=fig,
                height_ratios=[1.08, 1.0],
                width_ratios=[1.25, 1.0],
                hspace=0.44,
                wspace=0.28,
                left=0.085,
                right=0.985,
                top=0.93,
                bottom=0.10,
            )

            # Panel a: Genome-wide sliding-window track(s)
            n_tr = len(prepared_tracks)
            ratios = [max(t["total_bp"], 1) for t in prepared_tracks]
            gs_a = gs[0, :].subgridspec(1, n_tr, width_ratios=ratios, wspace=0.16)
            shared_ax = None
            for ti, tr in enumerate(prepared_tracks):
                ax_t = fig.add_subplot(gs_a[0, ti], sharey=shared_ax)
                if ti == 0:
                    shared_ax = ax_t
                    add_panel_label(ax_t, "a", x_pt=-14.0, y_pt=4.0)
                    ax_t.set_ylabel(f"{context} modification (%)", fontsize=FONT_PT["body"])
                xs = [w["mid_kb"] for w in tr["clean_windows"]]
                yc = [w["mod_rate"] * 100.0 if w["mod_rate"] is not None else np.nan for w in tr["clean_windows"]]
                for s_kb, e_kb in tr["transfers"]:
                    ax_t.axvspan(s_kb, e_kb, color=TRANSFER_FILL, alpha=0.65, lw=0, zorder=0)
                if tr["unfiltered_windows"]:
                    yu = [
                        w["mod_rate"] * 100.0 if w["mod_rate"] is not None else np.nan
                        for w in tr["unfiltered_windows"]
                    ]
                    ax_t.plot(xs, yu, color=UNFILTERED_COLOR, lw=0.85, alpha=0.9, label="Unfiltered (with NUMTs/NUPTs)")
                    ax_t.fill_between(xs, 0, np.nan_to_num(yu), color=UNFILTERED_COLOR, alpha=0.14)
                ax_t.plot(xs, yc, color=CLEAN_COLOR, lw=1.1, label="Clean organelle")
                ax_t.set_xlim(0, max(tr["total_bp"] / 1000.0, 0.1))
                ax_t.set_xlabel(
                    f"{tr['label']} (kb; {tr['total_bp'] / 1000.0:.1f} kb)",
                    fontsize=FONT_PT["body"],
                )
                ax_t.spines["top"].set_visible(False)
                ax_t.spines["right"].set_visible(False)
                if ti == 0:
                    ax_t.legend(loc="upper right", fontsize=FONT_PT["tick"], handlelength=1.4)

            # Panel b: Read-level CG bimodal histogram
            ax_b = fig.add_subplot(gs[1, 0])
            add_panel_label(ax_b, "b", x_pt=-14.0, y_pt=4.0)
            bins = np.linspace(0.0, 100.0, 36)
            if all_clean_cgs:
                ax_b.hist(
                    np.asarray(all_clean_cgs) * 100.0,
                    bins=bins,
                    density=True,
                    color=CLEAN_COLOR,
                    alpha=0.68,
                    label=f"Clean organelle (n={len(all_clean_cgs):,})",
                )
            if all_numt_cgs:
                ax_b.hist(
                    np.asarray(all_numt_cgs) * 100.0,
                    bins=bins,
                    density=True,
                    color=UNFILTERED_COLOR,
                    alpha=0.68,
                    label=f"NUMT/NUPT (n={len(all_numt_cgs):,})",
                )
            if mean_thr is not None:
                lbl = f"Auto threshold ({mean_thr * 100.0:.1f}%)"
                if mean_auc is not None:
                    lbl += f" · AUC={mean_auc:.3f}"
                ax_b.axvline(mean_thr * 100.0, color=THRESHOLD_COLOR, ls="--", lw=1.0, label=lbl)
            ax_b.set_xlim(0.0, 100.0)
            ax_b.set_xlabel("Per-read CG 5mC (%)", fontsize=FONT_PT["body"])
            ax_b.set_ylabel("Density", fontsize=FONT_PT["body"])
            ax_b.spines["top"].set_visible(False)
            ax_b.spines["right"].set_visible(False)
            if all_clean_cgs or all_numt_cgs:
                ax_b.legend(loc="upper right", fontsize=5.6, handlelength=1.3)

            # Panel c: Plant 4-context summary (unfiltered vs clean)
            ax_c = fig.add_subplot(gs[1, 1])
            add_panel_label(ax_c, "c", x_pt=-12.0, y_pt=4.0)
            present_ctxs = [
                ctx for ctx in VALID_CONTEXTS if combined_clean_ctx[ctx] or combined_unfilt_ctx[ctx]
            ] or ["CG"]
            xc = np.arange(len(present_ctxs))
            bw = 0.34
            unf_means = [
                float(np.mean(combined_unfilt_ctx[c]) * 100.0) if combined_unfilt_ctx[c] else 0.0
                for c in present_ctxs
            ]
            cln_means = [
                float(np.mean(combined_clean_ctx[c]) * 100.0) if combined_clean_ctx[c] else 0.0
                for c in present_ctxs
            ]
            ax_c.bar(xc - bw / 2, unf_means, width=bw * 0.9, color=UNFILTERED_COLOR, alpha=0.82, label="Unfiltered pool")
            ax_c.bar(xc + bw / 2, cln_means, width=bw * 0.9, color=CLEAN_COLOR, alpha=0.88, label="Clean organelle")
            ax_c.set_xticks(xc)
            ax_c.set_xticklabels(present_ctxs)
            ax_c.set_ylabel("Mean modification (%)", fontsize=FONT_PT["body"])
            ax_c.spines["top"].set_visible(False)
            ax_c.spines["right"].set_visible(False)
            ax_c.legend(loc="upper right", fontsize=5.6, handlelength=1.2)

            fig.savefig(out_p, dpi=dpi)
            plt.close(fig)

        tsv_sidecar = out_p.with_suffix(".windows.tsv")
        with open(tsv_sidecar, "w", encoding="utf-8", newline="") as tf:
            writer = csv.writer(tf, delimiter="\t")
            writer.writerow(
                ["organelle", "start", "end", "mid_kb", "sites", "coverage", "clean_mod_rate", "unfiltered_mod_rate"]
            )
            for tr in prepared_tracks:
                unf_map = {(w["start"], w["end"]): w["mod_rate"] for w in tr["unfiltered_windows"]}
                for w in tr["clean_windows"]:
                    u_rate = unf_map.get((w["start"], w["end"]))
                    writer.writerow(
                        [
                            tr["organelle"],
                            w["start"],
                            w["end"],
                            w["mid_kb"],
                            w["sites"],
                            round(w["coverage"], 1),
                            "" if w["mod_rate"] is None else round(w["mod_rate"], 6),
                            "" if u_rate is None else round(u_rate, 6),
                        ]
                    )

        return out_p

    key_findings = [
        {"metric": "clean_reads_plotted", "value": len(all_clean_cgs), "unit": "reads"},
        {"metric": "numt_nupt_reads_plotted", "value": len(all_numt_cgs), "unit": "reads"},
    ]
    if mean_auc is not None:
        key_findings.append({"metric": "roc_auc_organelle_vs_numt", "value": round(mean_auc, 4)})

    plot = plot_result(
        "plot_methylation_profile",
        _render,
        organelle=scope_org,
        metrics=metrics,
        key_findings=tuple(key_findings),
        summary=(
            f"Methylation profile prepared across {len(prepared_tracks)} organelle track(s) "
            f"({len(all_clean_cgs)} clean reads, {len(all_numt_cgs)} NUMT/NUPT reads)."
        ),
    )
    plot._panel = PanelDrawer(_draw_track_panel, aspect=0.45, header=f"{context} methylation profile")
    return plot


methylation_profile = plot_methylation_profile

__all__ = ["methylation_profile", "plot_methylation_profile"]
