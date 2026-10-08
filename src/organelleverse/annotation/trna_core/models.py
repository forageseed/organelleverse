"""Dataclasses shared by the clean-room tRNA engine."""

from __future__ import annotations

from dataclasses import dataclass, field

from .anticodon import normalize_anticodon


@dataclass(frozen=True)
class TRNACandidate:
    sequence: str
    start: int
    end: int
    strand: int
    anticodon: str
    amino_acid: str
    source: str = "native_cleanroom"
    organelle: str = "mitochondrion"
    anticodon_offset: int | None = None
    segments: tuple[tuple[int, int], ...] = field(default_factory=tuple)
    intron_start: int | None = None
    intron_end: int | None = None

    def __post_init__(self) -> None:
        if self.start < 1:
            raise ValueError("tRNA start coordinate must be one-based and positive")
        if self.end < self.start:
            raise ValueError("tRNA end coordinate must be greater than or equal to start")
        if self.strand not in {-1, 1}:
            raise ValueError("tRNA strand must be 1 or -1")
        if not self.sequence:
            raise ValueError("tRNA candidate sequence must not be empty")
        object.__setattr__(self, "sequence", self.sequence.upper().replace("U", "T"))
        object.__setattr__(self, "anticodon", normalize_anticodon(self.anticodon))
        segments = self.segments or ((self.start, self.end),)
        if any(start < 1 or end < start for start, end in segments):
            raise ValueError("tRNA candidate segments must use positive one-based inclusive spans")
        object.__setattr__(self, "segments", segments)

    @property
    def length(self) -> int:
        return len(self.sequence)

    @property
    def gene_name(self) -> str:
        if self.amino_acid == "fM":
            return f"trnfM({self.anticodon})"
        return f"trn{self.amino_acid}({self.anticodon})"


@dataclass(frozen=True)
class StemPair:
    left_offset: int
    right_offset: int
    left_base: str
    right_base: str
    pair_class: str


@dataclass(frozen=True)
class StemFeature:
    name: str
    left: tuple[int, int]
    right: tuple[int, int]
    pairs: tuple[StemPair, ...] = field(default_factory=tuple)
    mismatches: int = 0
    gaps: int = 0
    bulges: int = 0
    score: float = 0.0


@dataclass(frozen=True)
class TRNAFeatureSet:
    anticodon_offset: int | None
    stems: tuple[StemFeature, ...] = field(default_factory=tuple)
    variable_region: tuple[int, int] | None = None
    intron: tuple[int, int] | None = None
    model_name: str = "cleanroom_v1"
    structure_score: float = 0.0

    def stem(self, name: str) -> StemFeature | None:
        return next((stem for stem in self.stems if stem.name == name), None)

    @property
    def accepted_stem_count(self) -> int:
        return len(self.stems)


@dataclass(frozen=True)
class TRNAScore:
    total: float
    sequence_score: float
    structure_score: float
    covariance_score: float
    anticodon_score: float
    intron_score: float
    pseudogene_penalty: float
    overcall_penalty: float

    @property
    def components(self) -> dict[str, float]:
        return {
            "total": self.total,
            "sequence_score": self.sequence_score,
            "structure_score": self.structure_score,
            "covariance_score": self.covariance_score,
            "anticodon_score": self.anticodon_score,
            "intron_score": self.intron_score,
            "pseudogene_penalty": self.pseudogene_penalty,
            "overcall_penalty": self.overcall_penalty,
        }


@dataclass(frozen=True)
class TRNAFilterDecision:
    passed: bool
    reason: str
    confidence: str


@dataclass(frozen=True)
class TRNACall:
    candidate: TRNACandidate
    features: TRNAFeatureSet
    score: TRNAScore
    decision: TRNAFilterDecision

    @property
    def source(self) -> str:
        return self.candidate.source

    @property
    def confidence(self) -> str:
        return self.decision.confidence

    @property
    def gene_name(self) -> str:
        return self.candidate.gene_name

    @property
    def score_components(self) -> dict[str, float]:
        return self.score.components
