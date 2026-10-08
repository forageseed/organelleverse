"""Dataclasses for the native covariance-model engine."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CMState:
    """One Infernal covariance-model state.

    ``transitions`` are child-transition log-odds (length ``cnum``, except
    bifurcation ``B`` states which carry none). ``emissions`` are singlet (4) or
    pair (16) emission log-odds, empty for non-emitting states.
    """

    type: str
    v: int
    plast: int
    pnum: int
    cfirst: int
    cnum: int
    transitions: list[float] = field(default_factory=list)
    emissions: list[float] = field(default_factory=list)
    node: int = -1  # index into CovarianceModel.nodes


@dataclass
class CMNode:
    type: str
    idx: int
    lcol: int  # MSA/reference column of the left consensus position (-1 if none)
    rcol: int  # MSA/reference column of the right consensus position (-1 if none)


@dataclass
class CovarianceModel:
    name: str
    clen: int
    states: list[CMState] = field(default_factory=list)
    nodes: list[CMNode] = field(default_factory=list)
    null: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])


@dataclass
class CMHit:
    """A covariance-model hit with 1-based inclusive sequence coordinates."""

    start: int
    end: int
    strand: int
    score: float
    cm_from: int = 0
    cm_to: int = 0
    # Mature exons (1-based, forward genome coordinates, ascending) of a spliced
    # intron tRNA; empty for an ordinary contiguous hit.
    exons: tuple[tuple[int, int], ...] = ()
