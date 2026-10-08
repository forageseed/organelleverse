"""Mitochondrial annotation pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from ..execution import CommandRunner
from .boundary import correct_boundaries
from .cds import partition_valid_cds, validate_cds
from .db import DBManager
from .fasta import load_fasta, validate_fasta
from .gff import write_genbank, write_gff3
from .pcg import PCGConfig, annotate_pcg
from .result import MitochondrialAnnotationStats
from .rrna import annotate_rrna
from .sequences import extract_all
from .trans_splicing import validate_trans_spliced_genes
from .trna import annotate_trna


class MitochondrialAnnotationPipeline:
    def __init__(
        self,
        input_fasta: str | Path,
        output_dir: str | Path,
        name: str,
        *,
        threads: int = 4,
        db_path: str | Path | None = None,
        call_pcg: bool = True,
        call_trna: bool = False,
        call_rrna: bool = False,
        trna_engine: str | None = None,
        rrna_engine: str | None = None,
        tool_paths: Mapping[str, str] | None = None,
        command_runner: CommandRunner | None = None,
        pcg_evalue: float = 1e-5,
        pcg_min_score: float = 30.0,
    ):
        self.input_fasta = Path(input_fasta)
        self.output_dir = Path(output_dir)
        self.name = name
        self.threads = threads
        self.db_manager = DBManager(Path(db_path) if db_path else None)
        self.call_pcg = call_pcg
        self.call_trna = call_trna
        self.call_rrna = call_rrna
        self.trna_engine = trna_engine
        self.rrna_engine = rrna_engine
        self.tool_paths = dict(tool_paths) if tool_paths is not None else None
        self.command_runner = command_runner
        self.pcg_evalue = pcg_evalue
        self.pcg_min_score = pcg_min_score
        self.gff_dir = self.output_dir / "gff"
        self.genbank_dir = self.output_dir / "genbank"
        self.fasta_dir = self.output_dir / "fasta"
        self.tmp_dir = self.output_dir / "tmp"
        self.dependency_issues: list[str] = []
        self.annotation_result: MitochondrialAnnotationStats | None = None

        for directory in (self.gff_dir, self.genbank_dir, self.fasta_dir, self.tmp_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def verify_dependencies(self) -> bool:
        self.dependency_issues = []
        if not self.input_fasta.exists():
            self.dependency_issues.append("input_fasta")
        self.dependency_issues.extend(self.db_manager.verify())
        return not self.dependency_issues

    def _solver_models(self, genome, annotations, rejected_cds):
        from .pcg import CORE_GENES_FORCE_BLAST
        from .solver_models import model_missing_genes

        # Core respiratory genes only: ribosomal-protein and sdh genes are often
        # lost or pseudogenised, and modelling their rejected remnants turned
        # them into plausible but false ORFs (maize rps3, Medicago sdh4, Salix sdh3).
        del rejected_cds
        found = {a.gene_name.lower() for a in annotations if a.gene_type == "CDS"}
        targets = [g for g in CORE_GENES_FORCE_BLAST if g.lower() not in found]
        if not targets:
            return [], []
        models, displaced = model_missing_genes(
            genome,
            self.db_manager,
            targets,
            annotations,
            tool_paths=self.tool_paths,
            command_runner=self.command_runner,
        )
        valid, _ = partition_valid_cds(models, genome, self.db_manager)
        kept = {id(m) for m in valid}
        # only fragments under a model that survived validation are replaced
        displaced = [
            d
            for d in displaced
            if any(
                id(m) in kept
                and any(
                    min(me.end, de.end) >= max(me.start, de.start)
                    for me in m.exons
                    for de in d.exons
                )
                for m in models
            )
        ]
        return valid, displaced

    def run_pipeline(self) -> MitochondrialAnnotationStats | None:
        if not self.verify_dependencies():
            return None

        genome = load_fasta(self.input_fasta)
        warnings = validate_fasta(genome)
        config = PCGConfig(
            evalue=self.pcg_evalue,
            min_score=self.pcg_min_score,
            threads=self.threads,
            transl_table=1,
        )
        annotations = []
        if self.call_pcg:
            annotations = annotate_pcg(
                genome,
                self.db_manager,
                config,
                tool_paths=self.tool_paths,
                command_runner=self.command_runner,
            )
            warnings.extend(validate_trans_spliced_genes(annotations, self.db_manager))

        trna_annotations = []
        if self.call_trna:
            trna_annotations = annotate_trna(
                self.input_fasta,
                self.tmp_dir,
                self.threads,
                db_manager=self.db_manager,
                engine=self.trna_engine,
                tool_paths=self.tool_paths,
                command_runner=self.command_runner,
            )

        rrna_annotations = []
        if self.call_rrna:
            rrna_annotations = annotate_rrna(
                self.input_fasta,
                self.tmp_dir,
                db_manager=self.db_manager,
                engine=self.rrna_engine,
            )

        annotations = correct_boundaries(
            annotations,
            genome,
            self.db_manager,
            tool_paths=self.tool_paths,
            command_runner=self.command_runner,
        )
        annotations, rejected_cds = partition_valid_cds(
            annotations,
            genome,
            self.db_manager,
        )
        # The primary path encodes angiosperm exon structures; genes it built
        # wrong (rejected above) or never found are modelled from a reference
        # protein by the reading-frame DP (moss cox1 has five exons, atp9 four).
        models, displaced = self._solver_models(genome, annotations, rejected_cds)
        annotations = [a for a in annotations if all(a is not d for d in displaced)] + models
        warnings.extend(
            "Rejected invalid CDS candidate "
            f"{candidate.gene_name}:"
            f"{candidate.start if candidate.start is not None else '?'}-"
            f"{candidate.end if candidate.end is not None else '?'} "
            f"({','.join(candidate.issue_codes)})"
            for candidate in rejected_cds
        )
        cds_result = validate_cds(annotations, genome, self.db_manager)

        # RNA-editing annotation: record C-to-U start/stop gain sites so the
        # writers emit the edited /translation= and the exception qualifier.
        from .editing_gains import annotate_rna_edits

        annotations = [annotate_rna_edits(ann, genome, self.db_manager) for ann in annotations]

        gff_path = self.gff_dir / f"{self.name}.gff"
        gb_path = self.genbank_dir / f"{self.name}.gb"
        write_gff3(annotations, trna_annotations, rrna_annotations, genome, gff_path)
        write_genbank(
            annotations,
            trna_annotations,
            rrna_annotations,
            genome,
            gb_path,
            organism=self.name,
        )
        extracted = extract_all(
            annotations,
            trna_annotations,
            rrna_annotations,
            genome,
            self.fasta_dir,
            self.name,
        )

        cds_count = sum(1 for ann in annotations if ann.gene_type == "CDS")
        stats = MitochondrialAnnotationStats(
            sample_name=self.name,
            output_paths=tuple([gff_path, gb_path, *extracted.values()]),
            warnings=tuple(warnings),
            contig_count=1 if not genome.contig_map else len(genome.contig_map),
            cds_predicted=cds_count,
            trna_predicted=len(trna_annotations),
            rrna_predicted=len(rrna_annotations),
            missing_core_genes=tuple(cds_result.missing_core),
            invalid_cds=len(rejected_cds),
            rejected_cds_candidates=rejected_cds,
        )
        self.annotation_result = stats
        return stats
