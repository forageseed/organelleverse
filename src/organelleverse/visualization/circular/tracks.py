from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import prepare as _prep


@dataclass(frozen=True)
class Slot:
    r_in: float
    r_out: float


def allocate_rings(
    weights: Sequence[float], *, outer: float = 0.80, inner: float = 0.46, gap: float = 0.018
) -> list[Slot]:
    if not weights:
        raise ValueError("no ring weights to allocate")
    if inner >= outer:
        raise ValueError(f"inner ({inner}) must be < outer ({outer})")
    n = len(weights)
    total_gap = gap * (n - 1)
    usable = (outer - inner) - total_gap
    if usable <= 0:
        raise ValueError("not enough radial room for the requested rings")
    wsum = float(sum(weights)) or 1.0
    slots: list[Slot] = []
    r = outer
    for w in weights:
        h = usable * (w / wsum)
        slots.append(Slot(r_in=r - h, r_out=r))
        r = r - h - gap
    return slots


@dataclass(frozen=True)
class GeneRing:
    strand: str = "both"
    minus_label: str = "outer_gray"
    label: str | None = None
    kind: str = "genes"
    is_center: bool = False
    weight: float = 0.0

    def resolve(self, genome_length: int, work_dir: Path) -> None:
        return None


@dataclass(frozen=True)
class DensityRing:
    variants: Any = None
    data: _prep.DensityData | None = None
    weight: float = 1.0
    label: str | None = None
    kind: str = "density"
    is_center: bool = False

    def resolve(self, genome_length: int, work_dir: Path) -> _prep.DensityData:
        if self.data is not None:
            return self.data
        return _prep.prepare_density(self.variants, genome_length)


@dataclass(frozen=True)
class SVRing:
    sv: Any = None
    data: _prep.SVData | None = None
    weight: float = 1.0
    label: str | None = None
    kind: str = "sv"
    is_center: bool = False

    def resolve(self, genome_length: int, work_dir: Path) -> _prep.SVData:
        if self.data is not None:
            return self.data
        return _prep.prepare_sv(self.sv, genome_length)


@dataclass(frozen=True)
class DiversityRing:
    diversity: Any = None
    data: _prep.DiversityData | None = None
    weight: float = 1.0
    label: str | None = None
    kind: str = "diversity"
    is_center: bool = False

    def resolve(self, genome_length: int, work_dir: Path) -> _prep.DiversityData:
        if self.data is not None:
            return self.data
        return _prep.prepare_diversity(self.diversity, genome_length)


@dataclass(frozen=True)
class SegmentRing:
    segments: Any = None
    data: _prep.SegmentData | None = None
    weight: float = 1.0
    label: str | None = None
    kind: str = "segments"
    is_center: bool = False

    def resolve(self, genome_length: int, work_dir: Path) -> _prep.SegmentData:
        if self.data is not None:
            return self.data
        return _prep.prepare_segments(self.segments, genome_length)


@dataclass(frozen=True)
class GraphCenter:
    gfa: Any = None
    data: _prep.GraphImage | None = None
    layout: str = "spread"
    executable: Any = None
    runner: Any = None
    label: str | None = None
    kind: str = "graph"
    is_center: bool = True
    weight: float = 0.0

    def resolve(self, genome_length: int, work_dir: Path) -> _prep.GraphImage:
        if self.data is not None:
            return self.data
        if self.gfa is None:
            return _prep.GraphImage(None, "none")
        return _prep.prepare_graph(
            self.gfa,
            Path(work_dir) / "center_graph.png",
            executable=self.executable,
            runner=self.runner,
            layout=self.layout,
        )


@dataclass(frozen=True)
class SchematicCenter:
    segments: Any = None
    data: _prep.SegmentData | None = None
    label: str | None = None
    kind: str = "schematic"
    is_center: bool = True
    weight: float = 0.0

    def resolve(self, genome_length: int, work_dir: Path) -> _prep.SegmentData:
        if self.data is not None:
            return self.data
        return _prep.prepare_segments(self.segments, genome_length)
