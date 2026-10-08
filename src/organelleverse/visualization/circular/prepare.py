from __future__ import annotations

import csv
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _write_tsv(path: Path, header: Sequence[str], rows: Sequence[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["\t".join(header)]
    for row in rows:
        lines.append("\t".join(str(row.get(h, "")) for h in header))
    path.write_text("\n".join(lines) + "\n")


def _read_tsv(path: Path) -> list[dict[str, str]]:
    text = Path(path).read_text().splitlines()
    if not text:
        return []
    reader = csv.DictReader(text, delimiter="\t")
    return [dict(r) for r in reader]


@dataclass(frozen=True)
class TrackData:
    """Base for prepared track data. Subclasses implement to_tsv/from_tsv."""


def _make_bins(genome_length: int, *, bins: int, window_size: int | None):
    if window_size is None:
        window_size = max(1, math.ceil(genome_length / max(1, bins)))
    starts = list(range(0, genome_length, window_size))
    ends = [min(genome_length, s + window_size) for s in starts]
    return starts, ends


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(float(str(v).strip()))
    except (TypeError, ValueError):
        return default


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class DensityData(TrackData):
    bin_starts: tuple[int, ...]
    bin_ends: tuple[int, ...]
    density: tuple[float, ...]
    n_samples: int

    def to_tsv(self, path: Path) -> None:
        rows = [
            {"start": s, "end": e, "density": d}
            for s, e, d in zip(self.bin_starts, self.bin_ends, self.density, strict=False)
        ]
        _write_tsv(path, ["start", "end", "density"], rows)

    @classmethod
    def from_tsv(cls, path: Path) -> DensityData:
        rows = _read_tsv(path)
        return cls(
            tuple(_safe_int(r["start"]) for r in rows),
            tuple(_safe_int(r["end"]) for r in rows),
            tuple(float(r["density"]) for r in rows),
            n_samples=0,
        )


def prepare_density(variants, genome_length, *, bins=120, window_size=None) -> DensityData:
    rows = _read_tsv(Path(variants))
    cols = {c.lower() for c in (rows[0].keys() if rows else ())}
    # Native windowed output of variation.snp.snp_density(): start/end/snp_count.
    if {"start", "end"} <= cols and ("snp_count" in cols or "density_per_kb" in cols):
        col = {c.lower(): c for c in rows[0]}
        value_key = col.get("snp_count") or col.get("density_per_kb")
        bin_s = tuple(_safe_int(r[col["start"]]) for r in rows)
        bin_e = tuple(_safe_int(r[col["end"]]) for r in rows)
        density = tuple(_safe_float(r.get(value_key)) for r in rows)
        return DensityData(bin_s, bin_e, density, n_samples=0)
    starts, ends = _make_bins(genome_length, bins=bins, window_size=window_size)
    width = max(1, ends[0] - starts[0])
    density = [0.0] * len(starts)
    samples: set[str] = set()
    for r in rows:
        pos = _safe_int(r.get("position") or r.get("pos") or r.get("start"), 0)
        idx = min(len(starts) - 1, max(0, pos // width))
        density[idx] += 1.0
        samples.add(str(r.get("accession") or r.get("sample") or "sample"))
    return DensityData(tuple(starts), tuple(ends), tuple(density), len(samples))


@dataclass(frozen=True)
class SVData(TrackData):
    bin_starts: tuple[int, ...]
    bin_ends: tuple[int, ...]
    density: tuple[float, ...]
    sv_count: int

    def to_tsv(self, path: Path) -> None:
        rows = [
            {"start": s, "end": e, "sv": d}
            for s, e, d in zip(self.bin_starts, self.bin_ends, self.density, strict=False)
        ]
        _write_tsv(path, ["start", "end", "sv"], rows)

    @classmethod
    def from_tsv(cls, path: Path) -> SVData:
        rows = _read_tsv(path)
        dens = tuple(float(r["sv"]) for r in rows)
        return cls(
            tuple(_safe_int(r["start"]) for r in rows),
            tuple(_safe_int(r["end"]) for r in rows),
            dens,
            int(sum(dens)),
        )


def prepare_sv(sv, genome_length, *, bins=120, window_size=None) -> SVData:
    rows = _read_tsv(Path(sv))
    starts, ends = _make_bins(genome_length, bins=bins, window_size=window_size)
    cols = {c.lower() for c in (rows[0].keys() if rows else [])}
    if {"sv", "start"} <= cols:  # precomputed density
        density = [0.0] * len(starts)
        for i, r in enumerate(rows[: len(starts)]):
            density[i] = float(r.get("sv") or 0.0)
        return SVData(tuple(starts), tuple(ends), tuple(density), int(sum(density)))
    width = max(1, ends[0] - starts[0])
    density = [0.0] * len(starts)
    total = 0
    for r in rows:
        pos = _safe_int(r.get("position") or r.get("pos") or r.get("start"), -1)
        if pos < 0:
            continue
        idx = min(len(starts) - 1, max(0, (pos - 1) // width))
        density[idx] += 1.0
        total += 1
    return SVData(tuple(starts), tuple(ends), tuple(density), total)


@dataclass(frozen=True)
class DiversityData(TrackData):
    bin_starts: tuple[int, ...]
    bin_ends: tuple[int, ...]
    snp: tuple[float, ...]
    indel: tuple[float, ...]
    is_true_pi: bool

    def to_tsv(self, path: Path) -> None:
        rows = [
            {"start": s, "end": e, "snp": a, "indel": b}
            for s, e, a, b in zip(
                self.bin_starts, self.bin_ends, self.snp, self.indel, strict=False
            )
        ]
        _write_tsv(path, ["start", "end", "snp", "indel"], rows)

    @classmethod
    def from_tsv(cls, path: Path) -> DiversityData:
        rows = _read_tsv(path)
        return cls(
            tuple(_safe_int(r["start"]) for r in rows),
            tuple(_safe_int(r["end"]) for r in rows),
            tuple(_safe_float(r["snp"]) for r in rows),
            tuple(_safe_float(r["indel"]) for r in rows),
            is_true_pi=True,
        )


def prepare_diversity(source, genome_length, *, bins=120, window_size=None) -> DiversityData:
    rows = _read_tsv(Path(source))
    cols = {c.lower(): c for c in (rows[0].keys() if rows else {})}
    # Native windowed output of diversity.nucleotide_diversity(): start/end/pi.
    # Single-hue π feeds the SNP-diversity curve; indel diversity is left empty
    # (that estimator does not compute it).
    if (
        "pi" in cols
        and not (cols.get("snp") or cols.get("indel"))
        and {"start", "end"} <= set(cols)
    ):
        bin_s = tuple(_safe_int(r[cols["start"]]) for r in rows)
        bin_e = tuple(_safe_int(r[cols["end"]]) for r in rows)
        snp_pi = tuple(_safe_float(r.get(cols["pi"])) for r in rows)
        return DiversityData(bin_s, bin_e, snp_pi, tuple(0.0 for _ in rows), True)
    starts, ends = _make_bins(genome_length, bins=bins, window_size=window_size)
    snp_col = cols.get("snp") or cols.get("pi_snp")
    indel_col = cols.get("indel") or cols.get("pi_indel")
    if snp_col and indel_col:
        snp = [0.0] * len(starts)
        indel = [0.0] * len(starts)
        for i, r in enumerate(rows[: len(starts)]):
            snp[i] = _safe_float(r.get(snp_col))
            indel[i] = _safe_float(r.get(indel_col))
        return DiversityData(tuple(starts), tuple(ends), tuple(snp), tuple(indel), True)
    # derive normalized counts (caveat: not true pi)
    width = max(1, ends[0] - starts[0])
    snp = [0.0] * len(starts)
    indel = [0.0] * len(starts)
    samples: set[str] = set()
    for r in rows:
        pos = _safe_int(r.get("position") or r.get("pos") or r.get("start"), 1) - 1
        idx = min(len(starts) - 1, max(0, pos // width))
        kind = str(r.get("type") or r.get("variant_type") or "SNP").lower()
        if "ind" in kind or "ins" in kind or "del" in kind:
            indel[idx] += 1.0
        else:
            snp[idx] += 1.0
        samples.add(str(r.get("accession") or "s"))
    denom = max(1, len(samples))
    return DiversityData(
        tuple(starts),
        tuple(ends),
        tuple(x / denom for x in snp),
        tuple(x / denom for x in indel),
        False,
    )


@dataclass(frozen=True)
class SegmentRow:
    start: int
    end: int
    label: str
    segment_class: str
    coverage: float | None


@dataclass(frozen=True)
class SegmentData(TrackData):
    rows: tuple[SegmentRow, ...]

    def to_tsv(self, path: Path) -> None:
        _write_tsv(
            path,
            ["start", "end", "label", "class", "coverage"],
            [
                {
                    "start": r.start,
                    "end": r.end,
                    "label": r.label,
                    "class": r.segment_class,
                    "coverage": "" if r.coverage is None else r.coverage,
                }
                for r in self.rows
            ],
        )

    @classmethod
    def from_tsv(cls, path: Path) -> SegmentData:
        return prepare_segments(path, genome_length=0)


def prepare_segments(segments, genome_length) -> SegmentData:
    raw = _read_tsv(Path(segments))
    out: list[SegmentRow] = []
    for i, r in enumerate(raw):
        cov = r.get("coverage")
        out.append(
            SegmentRow(
                _safe_int(r.get("start")),
                _safe_int(r.get("end") or r.get("stop")),
                str(r.get("label") or r.get("name") or f"s{i + 1}"),
                str(r.get("class") or r.get("type") or "core").lower(),
                None if cov in (None, "") else _safe_float(cov),
            )
        )
    return SegmentData(tuple(out))


@dataclass(frozen=True)
class GraphImage(TrackData):
    path: Path | None
    status: str

    def to_tsv(self, path: Path) -> None:  # image, not tabular
        _write_tsv(path, ["path", "status"], [{"path": self.path or "", "status": self.status}])

    @classmethod
    def from_tsv(cls, path: Path) -> GraphImage:
        r = _read_tsv(path)[0]
        return cls(Path(r["path"]) if r["path"] else None, r["status"])


def prepare_graph(gfa, output, *, executable=None, runner=None, layout="spread") -> GraphImage:
    from ..gfa_graph import plot_gfa_graph
    from ..plot_object import OrganellePlot

    result = plot_gfa_graph(
        gfa,
        executable=executable,
        runner=runner,
        color_by="uniform",
        height=1900,
        font_size=40,
        node_width=13,
        edge_width=2.6,
        outline=0.0,
        linear=(layout == "spread"),
        extra_args=(
            "--unicolpos",
            "#4FA6A0",
            "--edgecol",
            "#AAB4BD",
            "--textcol",
            "#2B2B2B",
            "--outcol",
            "#3C736E",
        ),
    )
    out = Path(output)
    if isinstance(result, OrganellePlot) and result.status == "ok":
        try:
            written = result.save(out)
        except Exception:
            return GraphImage(None, "failed")
        return GraphImage(written, "ok")
    return GraphImage(None, getattr(result, "status", "unavailable"))
