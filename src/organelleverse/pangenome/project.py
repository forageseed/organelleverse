"""Pangenome project contract: validated samples with stable PanSN paths.

This module owns L1 input validation and internal staging: it turns a
deterministic list of canonical genomes into a frozen project whose samples
carry unique, PanSN-safe identities, and materializes that project as a
PanSN FASTA plus JSON manifest inside a caller-supplied staging directory.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from .._bio import read_fasta, write_fasta
from ..core.artifacts import ArtifactRef
from ..core.base import StrictFrozenModel
from ..core.errors import OrganelleInputError
from ..core.genome import OrganelleGenome

#: Minimum number of samples a pangenome project must contain.
MIN_SAMPLES = 2

#: Runs of characters outside the PanSN-safe identity alphabet.
_IDENTITY_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]+")

__all__ = [
    "MIN_SAMPLES",
    "PangenomeProject",
    "PangenomeSample",
    "PangenomeStagingManifest",
    "StagedSample",
    "normalize_identity",
]


def normalize_identity(value: str) -> str:
    """Normalize a sample identity to the ``[A-Za-z0-9_.-]`` alphabet."""
    return _IDENTITY_UNSAFE.sub("_", value).strip("_")


class PangenomeSample(StrictFrozenModel[Literal["pangenome_sample"]]):
    """One project sample: a genome plus its stable PanSN identity."""

    kind: Literal["pangenome_sample"] = "pangenome_sample"
    name: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    haplotype: str = Field(default="1", min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    genome: OrganelleGenome

    @field_validator("genome")
    @classmethod
    def require_sequence(cls, genome: OrganelleGenome) -> OrganelleGenome:
        if genome.sequence is None:
            raise ValueError("project sample requires a sequence artifact")
        return genome

    @property
    def pansn_path(self) -> str:
        """PanSN path of this sample's first emitted FASTA record."""
        return f"{self.name}#{self.haplotype}#1"


class StagedSample(StrictFrozenModel[Literal["pangenome_staged_sample"]]):
    """One staged sample: its source SHA256 and emitted PanSN headers."""

    kind: Literal["pangenome_staged_sample"] = "pangenome_staged_sample"
    sample: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    headers: tuple[str, ...] = Field(min_length=1)


class PangenomeStagingManifest(StrictFrozenModel[Literal["pangenome_staging_manifest"]]):
    """Traceability manifest for a staged PanSN FASTA."""

    kind: Literal["pangenome_staging_manifest"] = "pangenome_staging_manifest"
    fasta_path: str = Field(min_length=1)
    manifest_path: str = Field(min_length=1)
    samples: tuple[StagedSample, ...]


class PangenomeProject(StrictFrozenModel[Literal["pangenome_project"]]):
    """A frozen, validated set of pangenome samples in deterministic order."""

    kind: Literal["pangenome_project"] = "pangenome_project"
    samples: tuple[PangenomeSample, ...]

    @model_validator(mode="after")
    def validate_samples(self) -> Self:
        if len(self.samples) < MIN_SAMPLES:
            raise ValueError(f"PangenomeProject requires at least {MIN_SAMPLES} samples")
        names = [sample.name for sample in self.samples]
        if len(set(names)) != len(names):
            raise ValueError("sample identities must be unique within a project")
        paths = [sample.pansn_path for sample in self.samples]
        if len(set(paths)) != len(paths):
            raise ValueError("PanSN paths must be unique within a project")
        return self

    @classmethod
    def from_genomes(cls, genomes: Sequence[OrganelleGenome]) -> Self:
        """Build a project from genomes, preserving input order.

        Sample identity prefers the metadata accession, then the species, and is
        normalized to the PanSN-safe alphabet; empty or duplicate identities fail
        validation.
        """
        samples = tuple(
            PangenomeSample(
                name=normalize_identity(genome.metadata.accession or genome.metadata.species),
                genome=genome,
            )
            for genome in genomes
        )
        return cls(samples=samples)

    def materialize(self, staging_dir: str | Path) -> PangenomeStagingManifest:
        """Write the staged PanSN FASTA and JSON manifest into ``staging_dir``.

        Each source FASTA record becomes one PanSN record headed
        ``sample#haplotype#molecule``; the molecule segment is the 1-based
        record index within its source file, so multi-record molecules are
        preserved as distinct segments instead of concatenated.
        """
        target = Path(staging_dir)
        fasta_path = target / "pansn.fa"
        manifest_path = target / "manifest.json"
        records: list[tuple[str, str]] = []
        staged: list[StagedSample] = []
        for sample in self.samples:
            source = sample.genome.sequence
            assert source is not None  # guaranteed by PangenomeSample validation
            source_path = self._verified_source_path(source)
            headers: list[str] = []
            for index, (_, sequence) in enumerate(read_fasta(source_path), start=1):
                header = f"{sample.name}#{sample.haplotype}#{index}"
                headers.append(header)
                records.append((header, sequence))
            staged.append(
                StagedSample(sample=sample.name, sha256=source.sha256, headers=tuple(headers))
            )
        write_fasta(fasta_path, records)
        manifest = PangenomeStagingManifest(
            fasta_path=str(fasta_path),
            manifest_path=str(manifest_path),
            samples=tuple(staged),
        )
        manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return manifest

    def verify_sources(self) -> None:
        """Re-hash every source before execution or managed-cache reuse."""
        for sample in self.samples:
            source = sample.genome.sequence
            assert source is not None
            self._verified_source_path(source)

    @staticmethod
    def _verified_source_path(source: ArtifactRef) -> Path:
        source_path = source.resolve()
        current = ArtifactRef.from_path(
            source_path,
            kind=source.kind,
            format=source.format,
            media_type=source.media_type,
        )
        if current.sha256 != source.sha256 or current.size_bytes != source.size_bytes:
            raise OrganelleInputError(
                code="pangenome.source_digest_mismatch",
                message="pangenome source bytes changed after artifact capture",
                details={"uri": source.uri, "expected_sha256": source.sha256},
            )
        return source_path
