"""Closed requests and durable stage records for the pangenome workflow."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MoleculeNormalization(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    orientation: Literal["+", "-"] = "+"
    origin: int = Field(default=0, ge=0)


class WorkflowRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_path: str | None = None
    backend: Literal["existing", "minigraph", "pggb", "pantools"] = "pggb"
    gfa_path: str | None = None
    annotations_path: str | None = None
    organelle: Literal["mitochondrion", "plastid"] = "mitochondrion"
    threads: int = Field(default=4, ge=1, le=256)
    cloud_threshold: float = Field(default=0.05, ge=0, lt=0.5)
    core_threshold: float = Field(default=1.0, gt=0, le=1)
    bootstrap_replicates: int = Field(default=100, ge=0, le=10000)
    seed: int = Field(default=0, ge=0)
    segment_length: int = Field(default=5000, ge=1)
    identity: float = Field(default=90, gt=0, le=100)
    k: int = Field(default=31, ge=6, le=255)
    pantools_memory_mb: int = Field(default=4096, ge=128, le=1048576)
    normalization: dict[str, MoleculeNormalization] = Field(default_factory=dict)
    molecule_topologies: dict[str, Literal["linear", "circular", "unknown"]] = Field(
        default_factory=dict
    )
    gene_synonyms: dict[str, str] = Field(default_factory=dict)
    expected_genes: tuple[str, ...] = ()
    annotation_reference_paths: tuple[str, ...] = ()
    annotation_untangle: Literal["exact", "odgi"] = "exact"
    tree_mode: Literal["pav", "msa", "none"] = "pav"
    bootstrap_method: Literal["felsenstein", "adaptive_pav"] = "felsenstein"
    adaptive_min_replicates: int = Field(default=1000, ge=1, le=100000)
    adaptive_max_replicates: int = Field(default=10000, ge=1, le=100000)
    adaptive_batch_size: int = Field(default=100, ge=1, le=10000)
    adaptive_convergence_threshold: float = Field(default=0.99, gt=0, le=1)
    msa_path: str | None = None
    msa_format: Literal["fasta", "maf"] = "fasta"
    msa_seed: int = Field(default=1, ge=1)
    msa_taxon_mapping: dict[str, str] = Field(default_factory=dict)
    msa_taxon_unit: Literal["sample", "path"] = "sample"
    raxml_model: str = "GTR+G"
    tree_parsimony_starts: int = Field(default=10, ge=0, le=100)
    tree_random_starts: int = Field(default=10, ge=0, le=100)
    recommend_parameters: bool = False
    auto_adopt_recommendation: bool = False
    run_repeatmasker: bool = False
    repeatmasker_species: str = Field(default="", max_length=100)
    generate_overview: bool = True
    window_bp: int = Field(default=10000, ge=1)
    formats: tuple[Literal["svg", "pdf", "png"], ...] = ("svg", "pdf", "png")

    @model_validator(mode="after")
    def require_input(self) -> Self:
        if self.cloud_threshold >= self.core_threshold:
            raise ValueError("cloud_threshold must be smaller than core_threshold")
        if self.annotation_untangle == "odgi" and not self.annotation_reference_paths:
            raise ValueError("ODGI annotation untangling requires explicit reference paths")
        if self.adaptive_min_replicates > self.adaptive_max_replicates:
            raise ValueError("Adaptive bootstrap minimum cannot exceed maximum")
        if self.tree_mode == "msa":
            if not self.msa_path:
                raise ValueError("MSA phylogeny requires an explicit aligned FASTA or MAF file")
            if self.tree_parsimony_starts + self.tree_random_starts < 1:
                raise ValueError("MSA phylogeny requires at least one starting tree")
            if self.bootstrap_method != "felsenstein":
                raise ValueError("Adaptive PAV bootstrap is specific to a PAV tree")
        elif self.msa_path or self.msa_taxon_mapping:
            raise ValueError("Alignment input and taxon mapping require MSA tree mode")
        if self.backend == "existing" and (self.normalization or self.molecule_topologies):
            raise ValueError(
                "Input normalization precedes construction; an imported graph retains its coordinates"
            )
        if self.run_repeatmasker and not self.recommend_parameters:
            raise ValueError("run_repeatmasker requires explicit recommend_parameters opt-in")
        if self.auto_adopt_recommendation and (
            not self.recommend_parameters or self.backend != "pggb"
        ):
            raise ValueError("Automatic adoption requires PGGB and recommend_parameters opt-in")
        if self.repeatmasker_species and not self.run_repeatmasker:
            raise ValueError("repeatmasker_species requires explicit run_repeatmasker opt-in")
        if self.recommend_parameters and self.backend == "existing":
            raise ValueError("Parameter recommendation requires an input sequence dataset")
        if self.recommend_parameters and self.threads > 64:
            raise ValueError("Parameter recommendation supports at most 64 threads")
        if self.backend == "existing":
            if not self.gfa_path or self.dataset_path:
                raise ValueError("existing mode requires gfa_path and no dataset_path")
        elif not self.dataset_path or self.gfa_path:
            raise ValueError("graph construction requires dataset_path and no gfa_path")
        return self
