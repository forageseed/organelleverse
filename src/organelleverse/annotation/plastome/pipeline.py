"""OrganelleVerse plastome annotation pipeline."""

from __future__ import annotations

import logging
import re
import statistics
from dataclasses import replace
from itertools import combinations
from pathlib import Path

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqFeature import CompoundLocation, FeatureLocation, Location, SeqFeature
from Bio.SeqRecord import SeqRecord

from .blast import detect_ir_regions, parse_blast_hit_lists, run_plastome_blasts
from .db import default_plastome_reference_dir, load_product_map
from .exon_models import ExonModel, assemble_exon_models, select_mature_model_copies
from .exon_models import protein_extended_parts as _protein_extended_parts
from .gene_features import rebuild_gene_features
from .ir_mirror import cohere_ir_exons, mirror_ir_copies
from .models import BlastHit, PlastomeAnnotationStats, ReferenceQuery, ReferenceRecord
from .references import (
    build_exon_query_panel,
    build_plastome_queries,
    choose_reference,
    clean_seq,
    iter_fasta_inputs,
    load_references,
    rank_references,
)
from .ssc_orientation import evaluate_ssc_correction, flip_ssc_segment
from .splice_refine import _san, _score
from .tools import find_blast_tools

logger = logging.getLogger(__name__)


_AA3_TO_1 = {
    "ala": "A",
    "arg": "R",
    "asn": "N",
    "asp": "D",
    "cys": "C",
    "gln": "Q",
    "glu": "E",
    "gly": "G",
    "his": "H",
    "ile": "I",
    "leu": "L",
    "lys": "K",
    "met": "M",
    "fmet": "fM",
    "phe": "F",
    "pro": "P",
    "ser": "S",
    "thr": "T",
    "trp": "W",
    "tyr": "Y",
    "val": "V",
}


def _no_features_message(
    reference: str,
    *,
    queries: int,
    hits: int,
    min_identity: float,
    qcoverage: object,
    target_length: int,
) -> str:
    """Why nothing was transferred, so a caller can tell a wrong input from a weak reference."""

    head = (
        f"No plastome features could be transferred from {reference}: {queries} reference "
        f"gene queries were searched on a {target_length:,} bp target at >= {min_identity:g}% "
        f"identity (query coverage {qcoverage})"
    )
    if hits == 0:
        return (
            head + "; nothing matched. The sequence is probably not a plastid genome of a lineage "
            "this reference set covers, or the identity threshold is too strict."
        )
    return (
        head + f"; {hits} alignment hits were found but none could be placed as a gene "
        "(none reached the reference coverage after splice and frame checks)."
    )


class PlastomeAnnotationPipeline:
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        reference_dir: str | None = None,
        legacy_plastome_path: str | None = None,
        blastn_path: str | None = None,
        makeblastdb_path: str | None = None,
        tblastn_path: str | None = None,
        losat_path: str | None = None,
        organism_file: str | None = None,
        min_ir_length: int = 1000,
        min_pidentity: int = 40,
        qcoverage_range: str = "0.5,2",
        genome_form: str = "circular",
        skip_existing: bool = True,
        log_level: str = "INFO",
        native_trna_rrna: bool = True,
        refine_cds: bool = True,
        detect_ir: bool = True,
        correct_ssc_orientation: bool = True,
        threads: int = 1,
        exclude_reference_files: tuple[str, ...] = (),
        rotate_to_trnh: bool = False,
    ):
        del legacy_plastome_path
        self.rotate_to_trnh = rotate_to_trnh
        self.native_trna_rrna = native_trna_rrna
        self.refine_cds = refine_cds
        self.detect_ir = detect_ir
        self.correct_ssc_orientation = correct_ssc_orientation
        self.threads = max(1, int(threads))
        self.exclude_reference_files = set(exclude_reference_files)
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.reference_dir = (
            Path(reference_dir) if reference_dir else default_plastome_reference_dir()
        )
        self.tools = find_blast_tools(blastn_path, makeblastdb_path, tblastn_path, losat_path)
        self.organism_file = Path(organism_file) if organism_file else None
        self.min_ir_length = min_ir_length
        self.min_pidentity = min_pidentity
        self.qcoverage_range = qcoverage_range
        self.genome_form = genome_form
        self.skip_existing = skip_existing
        self.genbank_dir = self.output_dir / "Annotated_GenBank"
        self.logs_dir = self.output_dir / "Annotation_Logs"
        self.reports_dir = self.output_dir / "Reports"
        self.blast_dir = self.output_dir / "BLAST"
        self.annotation_results: dict[str, PlastomeAnnotationStats] = {}
        self.failed_samples: dict[str, str] = {}
        self.reference_files: list[Path] = []
        self.dependency_issues: list[str] = []
        logger.setLevel(getattr(logging, str(log_level).upper(), logging.INFO))

        for directory in (self.genbank_dir, self.logs_dir, self.reports_dir, self.blast_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def verify_dependencies(self) -> bool:
        self.dependency_issues = []
        if not self.input_dir.exists():
            self.dependency_issues.append("input_dir")
        if not self.reference_dir.exists() or not self.reference_dir.is_dir():
            self.dependency_issues.append("reference_dir")
        else:
            self.reference_files = sorted(self.reference_dir.glob("*.gb")) + sorted(
                self.reference_dir.glob("*.gbk")
            )
            if self.exclude_reference_files:
                self.reference_files = [
                    path
                    for path in self.reference_files
                    if path.name not in self.exclude_reference_files
                ]
            if not self.reference_files:
                self.dependency_issues.append("reference_genbank")
        # LOSAT covers every search group in -subject mode (no database build),
        # so the NCBI BLAST+ executables are only required when it is absent.
        losat_available = self.tools.losat is not None
        if self.tools.blastn is None and not losat_available:
            self.dependency_issues.append("blastn")
        if self.tools.makeblastdb is None and not losat_available:
            self.dependency_issues.append("makeblastdb")
        if self.tools.tblastn is None and not losat_available:
            self.dependency_issues.append("tblastn")
        return not self.dependency_issues

    def run_pipeline(self) -> None:
        if not self.verify_dependencies():
            return
        references = load_references(self.reference_files)
        product_map = load_product_map(self.reference_dir)
        for fasta_file in iter_fasta_inputs(self.input_dir):
            try:
                logger.info("Annotating plastome sample %s", fasta_file.stem)
                self._annotate_one(fasta_file, references, product_map)
                logger.info("Finished plastome sample %s", fasta_file.stem)
            except Exception as exc:
                self.failed_samples[fasta_file.stem] = str(exc)
                logger.exception("Plastome sample %s failed", fasta_file.stem)
        self.generate_reports()

    def generate_reports(self) -> None:
        summary_file = self.reports_dir / "00_ANNOTATION_SUMMARY.tsv"
        with summary_file.open("w") as handle:
            handle.write(
                "Sample\tInput_File\tReference_Genes\tFeatures_Annotated\t"
                "Warnings\tUnannotated_Genes\tOutput_GenBank\tSSC_Orientation\n"
            )
            for sample_name, stats in sorted(self.annotation_results.items()):
                handle.write(
                    f"{sample_name}\t{stats.input_fasta.name}\t"
                    f"{stats.total_genes_reference}\t{stats.total_genes_annotated}\t"
                    f"{len(stats.warnings)}\t{len(stats.unannotated_genes)}\t"
                    f"{stats.output_genbank.name if stats.output_genbank else 'NA'}\t"
                    f"{stats.ssc_orientation}\n"
                )
        if self.failed_samples:
            failed_file = self.reports_dir / "00_FAILED_SAMPLES.tsv"
            with failed_file.open("w") as handle:
                handle.write("Sample\tError\n")
                for sample, error in sorted(self.failed_samples.items()):
                    handle.write(f"{sample}\t{error}\n")

    def _supplement_genes(
        self,
        target_seq: str,
        primary: ReferenceRecord,
        references: tuple[ReferenceRecord, ...],
        annotated: list[SeqFeature],
        work_dir: Path,
        qcov: tuple[float, float],
        product_map: dict[str, str],
    ) -> list[SeqFeature]:
        """CDS genes most references carry but the primary transfer lacks, from the closest reference that has them.

        Transfer from one reference cannot annotate a gene that reference lost:
        the closest reference to Ricinus has no rps16, so OV (and PGA) never
        searched for it, while the RNA-curated Ricinus rps16 is intact. Each
        missing gene is searched with the nearest reference carrying it; the
        result must become an open reading frame after refinement or it is
        dropped (remnants of genes this lineage lost too).
        """
        have = {_name_of(f) for f in annotated if f.type == "CDS"}
        carriers: dict[str, int] = {}
        for ref in references:
            for gene in {f.gene for f in ref.features if f.feature_type == "CDS" and f.gene}:
                carriers[gene] = carriers.get(gene, 0) + 1
        wanted = {
            g
            for g, n in carriers.items()
            if g not in have and n >= _SUPPLEMENT_MIN_SHARE * len(references)
        }
        if not wanted:
            return []
        by_donor: dict[str, tuple[ReferenceRecord, set[str]]] = {}
        ranked = rank_references(target_seq, references)
        for gene in sorted(wanted):
            donor = next(
                (
                    r
                    for r in ranked
                    if r is not primary
                    and any(f.feature_type == "CDS" and f.gene == gene for f in r.features)
                ),
                None,
            )
            if donor is not None:
                by_donor.setdefault(donor.path.name, (donor, set()))[1].add(gene)
        added: list[SeqFeature] = []
        for name, (donor, genes) in sorted(by_donor.items()):
            queries = tuple(
                q for q in build_plastome_queries(donor) if q.reference_feature.gene in genes
            )
            if not queries:
                continue
            hits = run_plastome_blasts(
                target_seq,
                queries,
                work_dir=work_dir / "supplement" / Path(name).stem,
                tools=self.tools,
                min_identity=self.min_pidentity / 100,
                qcoverage_range=qcov,
                threads=self.threads,
            )
            features, _ = _build_features_from_hits(
                queries,
                hits,
                product_map=product_map,
                ir_regions=None,
                record_length=len(target_seq),
            )
            # a gene the primary transfer annotated under another name (ycf3/pafI)
            # already occupies the locus
            taken = {
                name
                for name in genes
                if any(
                    f.type == "CDS" and _name_of(f) == name and _overlaps_cds(f, annotated)
                    for f in features
                )
            }
            donor_nt = {
                f.gene: len(f.sequence)
                for f in donor.features
                if f.feature_type == "CDS" and f.gene in genes and f.sequence
            }
            for feature in features:
                if feature.type in ("CDS", "gene") and _name_of(feature) in genes - taken:
                    feature.qualifiers.setdefault("note", []).append(_SUPPLEMENT_NOTE)
                    feature.reference_length = donor_nt.get(_name_of(feature), 0)
                    added.append(feature)
        if added:
            logger.info(
                "Genes searched with secondary references: %s", sorted({_name_of(f) for f in added})
            )
        return added

    def _annotate_one(
        self,
        fasta_file: Path,
        references: tuple[ReferenceRecord, ...],
        product_map: dict[str, str],
    ) -> None:
        target_records = list(SeqIO.parse(fasta_file, "fasta"))
        if not target_records:
            raise ValueError(f"No FASTA records in {fasta_file}")
        target = target_records[0]
        target_seq = clean_seq(str(target.seq))
        reference = choose_reference(target_seq, references)
        queries = build_plastome_queries(reference)
        if not queries:
            raise ValueError(f"No plastome query sequences built from {reference.path.name}")
        qcov = _parse_qcoverage(self.qcoverage_range)
        sample_blast_dir = self.blast_dir / fasta_file.stem
        hits = run_plastome_blasts(
            target_seq,
            queries,
            work_dir=sample_blast_dir,
            tools=self.tools,
            min_identity=self.min_pidentity / 100,
            qcoverage_range=qcov,
            threads=self.threads,
        )
        run_warnings: list[str] = []
        ssc_status = "disabled"
        if self.correct_ssc_orientation:
            decision = evaluate_ssc_correction(target_seq, reference, references, hits, queries)
            ssc_status = decision.reason
            if decision.flip:
                # Flip before anything is placed on the sequence and rerun the
                # search: junction-spanning genes (ycf1, ndhF) must be found
                # on the corrected presentation, not coordinate-remapped.
                target_seq = flip_ssc_segment(target_seq, decision.ssc_start, decision.ssc_length)
                logger.info(
                    "SSC arc reverse-complemented (%d bp) to follow the reference-panel convention",
                    decision.ssc_length,
                )
                run_warnings.append(
                    f"ssc_orientation: reverse-complemented a {decision.ssc_length} bp SSC arc "
                    f"({decision.reason})"
                )
                ssc_status = "flipped"
                hits = run_plastome_blasts(
                    target_seq,
                    queries,
                    work_dir=sample_blast_dir,
                    tools=self.tools,
                    min_identity=self.min_pidentity / 100,
                    qcoverage_range=qcov,
                    threads=self.threads,
                )
        if self.rotate_to_trnh and self.genome_form == "circular":
            trnh = _trnh_hit(queries, hits)
            if trnh is None:
                run_warnings.append("rotate_to_trnh: no trnH-GUG found; the sequence is left as given")
            else:
                target_seq, flipped, shift = rotate_to_trnh(target_seq, trnh)
                if flipped or shift:
                    run_warnings.append(
                        "rotate_to_trnh: sequence "
                        + ("reverse-complemented and " if flipped else "")
                        + f"rotated by {shift} bp so that it starts at trnH-GUG on the reverse strand"
                    )
                    # Same as the SSC flip: search again on the new presentation.
                    hits = run_plastome_blasts(
                        target_seq,
                        queries,
                        work_dir=sample_blast_dir,
                        tools=self.tools,
                        min_identity=self.min_pidentity / 100,
                        qcoverage_range=qcov,
                        threads=self.threads,
                    )
        ir_regions = None
        if self.detect_ir:
            ir_regions = detect_ir_regions(
                target_seq,
                work_dir=sample_blast_dir / "ir",
                tools=self.tools,
                min_ir_length=self.min_ir_length,
                threads=self.threads,
            )
        annotated, warnings = _build_features_from_hits(
            queries,
            hits,
            product_map=product_map,
            ir_regions=ir_regions,
            record_length=len(target_seq),
        )
        warnings = run_warnings + warnings
        annotated.extend(
            self._supplement_genes(
                target_seq, reference, references, annotated, sample_blast_dir, qcov, product_map
            )
        )
        cohere_ir_exons(annotated, target_seq, ir_regions)
        if self.native_trna_rrna:
            # Replace BLAST-transferred tRNA/rRNA with the native covariance-model
            # tRNA engine and the native pyhmmer rRNA detector (no external program).
            from ..cmsearch.parallel import cyk_threads
            from .native_features import native_trna_rrna_features

            # Transferred gene spans guide splicing of intron tRNAs whose short
            # exons the native HMM filter cannot anchor (trnK/trnG/trnL).
            exon_lengths = _intron_trna_exon_lengths(references)
            exon_seeds = _intron_trna_exon_seeds(references)
            hints = _intron_trna_hints(annotated, exon_lengths)
            annotated = [f for f in annotated if f.type not in ("tRNA", "rRNA")]
            # The CYK kernel releases the GIL: score independent windows on the run's threads (same results).
            with cyk_threads(self.threads):
                annotated.extend(
                    native_trna_rrna_features(
                        target_seq,
                        intron_hints=hints,
                        intron_exon_lengths=exon_lengths,
                        intron_exon_seeds=exon_seeds,
                    )
                )

        # Transferred CDS spans before refinement: a remnant retired below keeps its transferred extent.
        transferred = {id(f): f.location for f in annotated if f.type == "CDS"}
        if self.refine_cds:
            from .cds_refine import refine_cds_features
            from .group_ii import suspect_references
            from .micro_exon import recover_micro_exons, recover_terminal_micro_exons
            from .splice_flanks import reference_junction_flanks

            reference_files = {reference.path.name for reference in references}
            # Reference audit: a reference whose junction its own domain VI
            # contradicts is no evidence for that gene's splice sites.
            suspect = suspect_references(reference_files)
            if suspect:
                logger.info("Reference junctions set aside by the audit: %s", sorted(suspect))
            ref_proteins = _reference_protein_map(references, skip=suspect)
            # Recover micro first exons (petB/petD/rpl16) before refinement, so the
            # spliced reading frame is right when the terminal stop is snapped.
            recover_micro_exons(target_seq, annotated, ref_proteins, _micro_exon_map(references))
            recover_terminal_micro_exons(
                target_seq, annotated, ref_proteins, _terminal_micro_exon_map(references)
            )
            from ..cmsearch.parallel import cyk_threads

            with cyk_threads(self.threads):  # features are refined on the run's threads (same result)
                refine_cds_features(
                    target_seq,
                    annotated,
                    ref_proteins=ref_proteins,
                    junction_flanks=reference_junction_flanks(references, skip=suspect),
                    reference_files=reference_files,
                )

            # Search complete reference architectures independently of the one
            # genome-wide transfer reference, retaining eligible copy hits.
            exon_queries = build_exon_query_panel(references)
            if exon_queries:
                exon_dir = sample_blast_dir / "complete_exons"
                run_plastome_blasts(
                    target_seq,
                    exon_queries,
                    work_dir=exon_dir,
                    tools=self.tools,
                    min_identity=self.min_pidentity / 100,
                    qcoverage_range=qcov,
                    threads=self.threads,
                    retain_exon_hsps=True,
                )
                exon_hits = parse_blast_hit_lists(
                    exon_dir / "reference4.tsv",
                    self.min_pidentity / 100,
                    qcov,
                    list(exon_queries),
                )
                reference_sequences = {
                    feature.reference_name: str(reference.record.seq)
                    for reference in references
                    for feature in reference.features
                }
                models = assemble_exon_models(
                    target_seq,
                    exon_queries,
                    exon_hits,
                    reference_sequences,
                )
                _retranslate_cds_features(annotated, target_seq)
                warnings.extend(
                    _promote_complete_exon_models(
                        annotated,
                        models,
                        ref_proteins,
                        product_map,
                    )
                )


        _merge_abutting_parts(annotated)
        # A supplemented gene must end up an open reading frame: remnants of
        # genes lost in this lineage (pseudogenes) are dropped with their gene feature.
        broken = {
            _name_of(f)
            for f in annotated
            if f.type == "CDS"
            and _SUPPLEMENT_NOTE in f.qualifiers.get("note", [])
            and not _is_open_frame(f, target_seq)
        }
        if broken:
            logger.info("Supplemented genes dropped (no open reading frame): %s", sorted(broken))
            annotated = [
                f
                for f in annotated
                if not (_name_of(f) in broken and _SUPPLEMENT_NOTE in f.qualifiers.get("note", []))
            ]
        retired = _retire_remnants(annotated, references, transferred, target_seq)
        if retired:
            logger.info("CDS retired as pseudogene remnants: %s", retired)
            warnings.append("pseudogene remnants (no CDS of reference length): " + ", ".join(retired))
        # One best hit per query leaves IR genes with a single copy; mirror the
        # refined copy into the other inverted repeat.
        mirror_ir_copies(annotated, target_seq, ir_regions)
        annotated = _drop_identical_features(annotated)
        from .cds_refine import mark_edited_starts

        mark_edited_starts(annotated, target_seq)
        # Gene features were transferred before CDS refinement and native RNA
        # detection; derive them from the final products instead.
        annotated = rebuild_gene_features(annotated, len(target_seq))
        # Checks of a manual curation workflow (GeSeq/PGA curation): reported only, the annotation is unchanged.
        warnings.extend(_short_cds_warnings(annotated))
        warnings.extend(_overlap_warnings(annotated))

        if not annotated:
            raise ValueError(
                _no_features_message(
                    reference.path.name,
                    queries=len(queries),
                    hits=len(hits),
                    min_identity=self.min_pidentity,
                    qcoverage=self.qcoverage_range,
                    target_length=len(target_seq),
                )
            )
        organism = _organism_for_sample(fasta_file.stem, target.description)
        _retranslate_cds_features(annotated, target_seq)
        out_record = _make_output_record(
            fasta_file.stem,
            organism,
            target_seq,
            genome_form=self.genome_form,
            features=annotated,
        )
        output_path = self.genbank_dir / f"{fasta_file.stem}.gb"
        SeqIO.write(out_record, output_path, "genbank")
        reference_genes = _reference_gene_set(reference)
        annotated_genes = _annotated_gene_set(annotated)
        missing_genes = sorted(reference_genes - annotated_genes)
        stats = PlastomeAnnotationStats(
            sample_name=fasta_file.stem,
            input_fasta=fasta_file,
            total_genes_reference=len(reference_genes),
            total_genes_annotated=len(annotated),
            annotation_success=bool(annotated),
            warnings=warnings,
            unannotated_genes=missing_genes,
            output_genbank=output_path,
            ssc_orientation=ssc_status,
        )
        self.annotation_results[fasta_file.stem] = stats
        _write_warning_log(self.logs_dir / f"{fasta_file.stem}.warning.log", stats)


def _reference_protein_map(
    references: tuple[ReferenceRecord, ...], skip: set[tuple[str, str]] | None = None
) -> dict[str, list[str]]:
    """Map gene name to candidate reference proteins (from CDS ``translation``).

    Used by CDS refinement to correct two-exon internal splice boundaries.
    ``skip``: (reference file, gene) pairs whose junctions the audit rejected.
    """
    proteins: dict[str, list[str]] = {}
    for reference in references:
        for feat in reference.features:
            if feat.feature_type != "CDS" or not feat.gene:
                continue
            if skip and (reference.path.name, feat.gene) in skip:
                continue
            translation = feat.feature.qualifiers.get("translation", [None])[0]
            if translation:
                proteins.setdefault(feat.gene, []).append(translation)
    return proteins


_MIN_INTRON_TRNA_SPAN = 150


def _trna_family(gene: str) -> str | None:
    """Amino-acid key of a tRNA gene name (trnK-UUU -> K, trnfM -> fM)."""
    if gene.startswith("trnfM"):
        return "fM"
    match = re.match(r"trn([A-Z])", gene)
    return match.group(1) if match else None


def _intron_trna_exon_lengths(
    references: tuple[ReferenceRecord, ...],
) -> dict[str, tuple[int, int]]:
    """Median (5', 3') exon lengths of the references' intron tRNAs, by amino acid.

    Land-plant plastids carry at most one intron tRNA per amino acid (trnK-UUU,
    trnG-UCC, trnL-UAA, trnV-UAC, trnI-GAU, trnA-UGC; trnT in some ferns), so
    the amino acid identifies the gene even when a record omits the anticodon.
    Preprocessed references keep no exon structure and contribute nothing.
    """
    pairs: dict[str, list[tuple[int, int]]] = {}
    for reference in references:
        for feat in reference.features:
            parts = list(getattr(feat.feature.location, "parts", []))
            if feat.feature_type != "tRNA" or len(parts) != 2:
                continue
            family = _trna_family(_standardize_gene_qualifier("tRNA", feat.feature.qualifiers))
            if family:
                pairs.setdefault(family, []).append((len(parts[0]), len(parts[1])))
    return {
        family: (
            int(statistics.median(five for five, _ in values)),
            int(statistics.median(three for _, three in values)),
        )
        for family, values in pairs.items()
    }


def _intron_trna_exon_seeds(
    references: tuple[ReferenceRecord, ...],
) -> dict[str, list[tuple[str, str]]]:
    """Distinct (5' exon, 3' exon) sequences of the references' intron tRNAs, by amino acid."""
    seeds: dict[str, set[tuple[str, str]]] = {}
    for reference in references:
        for feat in reference.features:
            parts = list(getattr(feat.feature.location, "parts", []))
            if feat.feature_type != "tRNA" or len(parts) != 2:
                continue
            family = _trna_family(_standardize_gene_qualifier("tRNA", feat.feature.qualifiers))
            if family:
                five, three = (str(part.extract(reference.record.seq)).upper() for part in parts)
                seeds.setdefault(family, set()).add((five, three))
    return {family: sorted(pairs) for family, pairs in seeds.items()}


def _intron_trna_hints(features: list[SeqFeature], exon_lengths: dict[str, tuple[int, int]]):
    """Reference-transferred gene spans of intron tRNAs, as native splice hints."""
    from .native_features import IntronTRNAHint

    hints = []
    for feature in features:
        if feature.type != "gene" or len(feature.location) < _MIN_INTRON_TRNA_SPAN:
            continue
        family = _trna_family(feature.qualifiers.get("gene", [""])[0])
        if family not in exon_lengths:
            continue
        hints.append(
            IntronTRNAHint(
                start=int(feature.location.start) + 1,
                end=int(feature.location.end),
                strand=feature.location.strand or 1,
                exon_lengths=exon_lengths[family],
                amino_acid=family,
            )
        )
    return hints


def _terminal_micro_exon_map(
    references: tuple[ReferenceRecord, ...],
) -> dict[str, tuple[int, int, int]]:
    """Map gene -> (exon count, penultimate exon length, terminal exon length) for CDS whose last exon is tiny.

    rps12's 26-30 nt exon 3 is below the blastn word size, so transfer ends the
    gene in exon 2 read on into the intron. Taken from the per-exon ``CDS_exon``
    features (most common structure across references).
    """
    from collections import Counter

    exons: dict[tuple[str, str], dict[int, int]] = {}
    meta: dict[tuple[str, str], tuple[str, int]] = {}
    copies: Counter = Counter()  # CDS copies per gene, any structure
    for reference in references:
        for feat in reference.features:
            if feat.feature_type == "CDS" and feat.gene:
                copies[feat.gene] += 1
            if feat.feature_type == "CDS_exon" and feat.gene and feat.exon_count >= 2:
                key = (feat.reference_name, feat.feature_id.rsplit(":exon", 1)[0])
                exons.setdefault(key, {})[feat.exon_index] = len(feat.sequence)
                meta[key] = (feat.gene, feat.exon_count)
    shapes: dict[str, Counter] = {}
    for key, lengths in exons.items():
        gene, n = meta[key]
        if sorted(lengths) != list(range(1, n + 1)) or lengths[n] > 40:
            continue
        shapes.setdefault(gene, Counter())[(n, lengths[n - 1], lengths[n])] += 1
    # The structure must be the gene's usual one: a few records give single-exon
    # rpl22 a stray short second exon, and following them truncates real genes.
    out = {}
    for gene, counts in shapes.items():
        shape, n = counts.most_common(1)[0]
        if n * 2 > copies[gene]:
            out[gene] = shape
    return out


def _micro_exon_map(references: tuple[ReferenceRecord, ...]) -> dict[str, int]:
    """Map gene -> reference 5' exon length for two-exon CDS with a micro first exon.

    Genes like petB/petD/rpl16 have a 6-9 bp first exon that blastn cannot transfer;
    the length lets the micro-exon finder recover it downstream. Derived from the
    per-exon ``CDS_exon`` features (exon_index 1 = transcription-first) so it works
    whether references are parsed from GenBank or loaded from the preprocessed cache.
    """
    micro: dict[str, int] = {}
    for reference in references:
        for feat in reference.features:
            if (
                feat.feature_type == "CDS_exon"
                and feat.exon_count == 2
                and feat.exon_index == 1
                and feat.gene
                and feat.gene not in micro
                and 3 <= len(feat.sequence) <= 15
            ):
                micro[feat.gene] = len(feat.sequence)
    return micro


def _build_features_from_hits(
    queries: tuple[ReferenceQuery, ...],
    hits: dict[str, BlastHit],
    *,
    product_map: dict[str, str],
    ir_regions: tuple[tuple[int, int, int], tuple[int, int, int]] | None,
    record_length: int | None = None,
) -> tuple[list[SeqFeature], list[str]]:
    query_by_id = {query.query_id: query for query in queries}
    features: list[SeqFeature] = []
    warnings: list[str] = []
    occupied: set[tuple[str, str, int, int, int]] = set()
    if ir_regions is not None:
        labels = ("inverted repeat B", "inverted repeat A")
        for label, (start, end, strand) in zip(labels, ir_regions, strict=True):
            quals = {"note": [label], "rpt_type": ["inverted"]}
            # IRs come from a doubled-sequence self-search; an IR crossing
            # the circular origin maps past the linear record's end. Emit it
            # as two bounded segments instead of an out-of-bounds feature.
            if record_length is not None and end > record_length:
                head_end = min(end, record_length)
                if start < head_end:
                    features.append(
                        SeqFeature(
                            FeatureLocation(start, head_end, strand=strand),
                            type="repeat_region",
                            qualifiers={
                                **quals,
                                "note": [f"{label} (origin-wrapped, part 1/2)"],
                            },
                        )
                    )
                tail_end = end - record_length
                if tail_end > 0:
                    features.append(
                        SeqFeature(
                            FeatureLocation(0, tail_end, strand=strand),
                            type="repeat_region",
                            qualifiers={
                                **quals,
                                "note": [f"{label} (origin-wrapped, part 2/2)"],
                            },
                        )
                    )
                continue
            features.append(
                SeqFeature(
                    FeatureLocation(start, end, strand=strand),
                    type="repeat_region",
                    qualifiers=dict(quals),
                )
            )
    exon_hits: dict[str, list[tuple[ReferenceQuery, BlastHit]]] = {}
    for query_id, hit in hits.items():
        query = query_by_id.get(query_id)
        if query is None:
            continue
        ref_feature = query.reference_feature
        if ref_feature.feature_type == "CDS_exon":
            root_id = ref_feature.feature_id.rsplit(":exon", 1)[0]
            exon_hits.setdefault(root_id, []).append((query, hit))
            continue
        location = FeatureLocation(hit.start, hit.end, strand=hit.strand)
        feature = _copy_feature(ref_feature.feature, location, product_map)
        key = (
            feature.type,
            feature.qualifiers.get("gene", [""])[0],
            int(location.start),
            int(location.end),
            hit.strand,
        )
        if key in occupied:
            continue
        occupied.add(key)
        features.append(feature)
        if hit.qcov < 0.95:
            warnings.append(
                f"{ref_feature.gene or ref_feature.feature_type} query coverage {hit.qcov:.2f}"
            )
    for _root_id, items in exon_hits.items():
        items.sort(key=lambda item: item[0].reference_feature.exon_index)
        template = items[0][0].reference_feature.feature
        parts = _protein_extended_parts(items)
        location = parts[0] if len(parts) == 1 else CompoundLocation(parts)
        feature = _copy_feature(template, location, product_map)
        gene = feature.qualifiers.get("gene", [""])[0]
        key = (feature.type, gene, int(feature.location.start), int(feature.location.end), 0)
        if key not in occupied:
            occupied.add(key)
            features.append(feature)
        expected = items[0][0].reference_feature.exon_count
        if len(items) < expected:
            warnings.append(f"{gene} annotated with {len(items)}/{expected} coding exons")
        if len(items) == 1 and expected > 1:
            warnings.append(f"{gene} possible intron loss")
    _sync_gene_feature_names(features)
    features.sort(key=lambda feat: (int(feat.location.start), int(feat.location.end), feat.type))
    return features, warnings


# Qualifiers that describe the gene itself and stay valid on the target genome.
# Everything else in a reference feature belongs to the reference record --
# locus_tag, protein_id, db_xref, inference, translation -- or encodes its
# coordinates (anticodon, transl_except, rpt_unit_range), so it is dropped.
_TRANSFERABLE_QUALIFIERS = frozenset(
    {
        "gene",
        "gene_synonym",
        "product",
        "function",
        "note",
        "number",
        "pseudo",
        "exception",
        "trans_splicing",
        "codon_recognized",
        "transl_table",
        "codon_start",
        "rpt_type",
        "rpt_family",
        "regulatory_class",
    }
)


def _copy_feature(feature: SeqFeature, location, product_map: dict[str, str]) -> SeqFeature:
    qualifiers = {
        key: list(value)
        for key, value in feature.qualifiers.items()
        if key in _TRANSFERABLE_QUALIFIERS
    }
    gene = _standardize_gene_qualifier(feature.type, qualifiers)
    if gene:
        qualifiers["gene"] = [gene]
    if feature.type in {"CDS", "tRNA", "rRNA"} and gene and "product" not in qualifiers:
        product = product_map.get(gene)
        if product:
            qualifiers["product"] = [product]
    if feature.type == "CDS":
        qualifiers.setdefault("transl_table", ["11"])
        qualifiers.setdefault("codon_start", ["1"])
    return SeqFeature(location=location, type=feature.type, qualifiers=qualifiers)


def _sync_gene_feature_names(features: list[SeqFeature]) -> None:
    typed = [
        feature
        for feature in features
        if feature.type in {"CDS", "tRNA", "rRNA"} and feature.qualifiers.get("gene")
    ]
    for gene_feature in features:
        if gene_feature.type != "gene":
            continue
        gene = gene_feature.qualifiers.get("gene", [""])[0]
        if not gene:
            continue
        for child in typed:
            child_gene = child.qualifiers.get("gene", [""])[0]
            if _is_more_specific_gene_name(gene, child_gene) and _features_overlap(
                gene_feature,
                child,
            ):
                gene_feature.qualifiers["gene"] = [child_gene]
                break


def _features_overlap(a: SeqFeature, b: SeqFeature) -> bool:
    start = max(int(a.location.start), int(b.location.start))
    end = min(int(a.location.end), int(b.location.end))
    overlap = max(0, end - start)
    shortest = min(
        int(a.location.end) - int(a.location.start),
        int(b.location.end) - int(b.location.start),
    )
    return shortest > 0 and overlap / shortest >= 0.8


def _is_more_specific_gene_name(gene: str, candidate: str) -> bool:
    if not gene or not candidate or gene == candidate:
        return False
    return candidate.startswith(f"{gene}-") or candidate.startswith(f"{gene}(")


_THREE_TO_ONE = {"Gln": "Q", "Arg": "R", "Trp": "W", "Met": "M"}


def _apply_transl_except(feature: SeqFeature, protein: str) -> str:
    """Substitute ``transl_except`` residues (edited stops) into ``protein``."""
    excepts = feature.qualifiers.get("transl_except", [])
    if not excepts:
        return protein
    positions = []
    for part in feature.location.parts:
        span = range(int(part.start), int(part.end))
        positions.extend(reversed(span) if part.strand == -1 else span)
    index_of = {pos: i for i, pos in enumerate(positions)}
    residues = list(protein)
    for value in excepts:
        match = re.fullmatch(r"\(pos:(.+),aa:(\w+)\)", value)
        if not match or match[2] not in _THREE_TO_ONE:
            continue
        edited_location = Location.fromstring(match[1], length=max(positions) + 1)
        edited_bases = [index_of.get(pos) for pos in edited_location]
        if len(edited_bases) != 3 or any(i is None for i in edited_bases):
            continue
        first = edited_bases[0]
        if (
            first % 3 == 0
            and first // 3 < len(residues)
            and edited_bases == [first, first + 1, first + 2]
        ):
            residues[first // 3] = _THREE_TO_ONE[match[2]]
    return "".join(residues)


#: Share of loaded references carrying a CDS above which a gene missing from the
#: primary reference's transfer is searched for with another reference.
_SUPPLEMENT_MIN_SHARE = 0.5
_SUPPLEMENT_NOTE = "transferred from a secondary reference"

# A refined CDS shorter than this share of the gene's median reference CDS is retired as a remnant of a gene lost
# in this lineage (orchid ndhJ/ndhC, parasite rps16) when the transfer itself shows it is one; see _retire_remnants.
_REMNANT_FRACTION = 0.4
_REMNANT_NOTE = "pseudogene remnant: no open reading frame of reference length"
_READ_THROUGH_NOTE = "read through as U-to-C"


def _longest_open_share(location: Location, genome: str) -> float:
    """Longest stop-free run of codons in any frame of the span, as a share of the span length.

    A transfer often runs past the gene's own stop codon, so a stop near an end is no evidence; a lost gene's
    span is broken by stops into runs well short of it.
    """
    nt = str(location.extract(Seq(genome))).upper()
    best = 0
    for frame in range(3):
        run = 0
        for i in range(frame, len(nt) - 2, 3):
            run = 0 if nt[i : i + 3] in ("TAA", "TAG", "TGA") else run + 3
            best = max(best, run)
    return best / max(1, len(nt))


def _open_run_nearby(location: Location, genome: str, flank: float) -> int:
    """Longest stop-free run (nt) in any frame of the span's strand, the span extended by ``flank`` on both sides."""
    strand = location.strand or 1
    lo = max(0, int(location.start) - int(flank))
    hi = min(len(genome), int(location.end) + int(flank))
    nt = genome[lo:hi].upper()
    if strand == -1:
        nt = str(Seq(nt).reverse_complement())
    best = 0
    for frame in range(3):
        run = 0
        for i in range(frame, len(nt) - 2, 3):
            run = 0 if nt[i : i + 3] in ("TAA", "TAG", "TGA") else run + 3
            best = max(best, run)
    return best


def _retire_remnants(
    features: list[SeqFeature],
    references: tuple[ReferenceRecord, ...],
    transferred: dict[int, Location],
    genome: str,
) -> list[str]:
    """Turn CDS remnants of lost genes into pseudo gene features in place; return "gene N bp" for each.

    A short refined CDS alone is no evidence: refinement also cuts real genes (psbA, rps4, fern ycf1 in the benchmark),
    and those stay CDS. It is retired only when the transfer shows a lost gene: the transferred span is itself shorter
    than the remnant share of the reference, or, in a genome without U-to-C editing, the transferred span is short of
    80% of the reference and stops break every frame into runs below 80% of it (orchid ndhJ/ndhC). A full-length
    transfer is never retired: a fern whose editing went undetected would otherwise lose real genes.
    Kept as CDS also: genes without reference proteins, trans-spliced rps12 parts, and a CDS
    within 3 nt of either sequence end (a gene cut by the end of a linear sequence is partial, not a remnant). The
    pseudo gene feature takes the transferred span, which shows how much of the gene is left.
    """
    length = len(genome)
    editing = any(
        _READ_THROUGH_NOTE in note for f in features if f.type == "CDS" for note in f.qualifiers.get("note", [])
    )
    medians = {
        gene: statistics.median(3 * len(p.rstrip("*")) + 3 for p in proteins)
        for gene, proteins in _reference_protein_map(references).items()
        if proteins
    }
    retired: list[str] = []
    for i, feature in enumerate(features):
        if feature.type != "CDS":
            continue
        gene = _name_of(feature)
        median = medians.get(gene)
        if not median or gene == "rps12" or "trans_splicing" in feature.qualifiers:
            continue
        n = len(feature.location)
        if n >= _REMNANT_FRACTION * median:
            continue
        if int(feature.location.start) < 3 or int(feature.location.end) > length - 3:
            continue
        span = transferred.get(id(feature), feature.location)
        lost = len(span) < _REMNANT_FRACTION * median or (
            not editing and len(span) < 0.8 * median and _longest_open_share(span, genome) < 0.8
        )
        # A transfer can also hit only a piece of an intact gene (legume rps2, Pinus rps4): an open reading frame
        # of near-reference length beside the span means the gene is there, so it is not called a pseudogene.
        if not lost or _open_run_nearby(span, genome, median) >= 0.8 * median:
            continue
        features[i] = SeqFeature(
            span,
            type="gene",
            qualifiers={"gene": [gene], "pseudo": [""], "note": [f"{_REMNANT_NOTE} ({n} of ~{median:.0f} bp)"]},
        )
        retired.append(f"{gene} {n} bp")
    return retired


def _name_of(feature: SeqFeature) -> str:
    return feature.qualifiers.get("gene", [""])[0]


def _trnh_hit(queries: tuple[ReferenceQuery, ...], hits: dict[str, BlastHit]) -> BlastHit | None:
    """Best-scoring search hit of a reference trnH-GUG, or None."""
    best = None
    for query in queries:
        feature = query.reference_feature
        if feature.feature_type != "tRNA" or not (feature.gene or "").lower().startswith("trnh"):
            continue
        hit = hits.get(query.query_id)
        if hit is not None and (best is None or hit.bitscore > best.bitscore):
            best = hit
    return best


def rotate_to_trnh(sequence: str, hit: BlastHit) -> tuple[str, bool, int]:
    """Present a circular plastome with trnH-GUG on the reverse strand at its start (the usual GenBank layout).

    ``hit`` is trnH's search hit on ``sequence`` (0-based start, end exclusive, strand). Returns the new sequence,
    whether it was reverse-complemented, and the rotation in bp.
    """
    n = len(sequence)
    start, end = hit.start, hit.end
    flipped = hit.strand == 1
    if flipped:
        sequence = str(Seq(sequence).reverse_complement())
        start, end = n - end, n - start
    shift = start % n
    return sequence[shift:] + sequence[:shift], flipped, shift


# The shortest plastid CDS (petN) has 29 codons; a CDS below 33 bp is a fragment (manual curation threshold).
_SHORT_CDS_NT = 33


def _short_cds_warnings(features: list[SeqFeature]) -> list[str]:
    out = []
    for f in features:
        if f.type != "CDS" or "trans_splicing" in f.qualifiers or _name_of(f) == "rps12":
            continue
        n = len(f.location)
        if n < _SHORT_CDS_NT:
            out.append(f"short CDS (< {_SHORT_CDS_NT} bp): {_name_of(f)} {n} bp at {_span_text(f)}")
    return out


# Neighbouring genes that overlap in most plastomes; within this many bp they are not reported.
_KNOWN_OVERLAPS = frozenset(
    frozenset(pair) for pair in (("psbC", "psbD"), ("atpB", "atpE"), ("ndhC", "ndhK"), ("rpl22", "rps3"))
)
_KNOWN_OVERLAP_NT = 100
_OVERLAP_MIN_NT = 20


def _overlap_warnings(features: list[SeqFeature]) -> list[str]:
    """Product features whose exons overlap (introns do not count: matK inside the trnK-UUU intron is fine).

    Reported: an overlap of at least 20 bp, except the usual neighbours (psbC/psbD, atpB/atpE, ndhC/ndhK,
    rpl22/rps3) within 100 bp, and always one feature lying 90% or more inside another.
    """
    products = [f for f in features if f.type in ("CDS", "tRNA", "rRNA") and "trans_splicing" not in f.qualifiers]
    parts = [[(int(p.start), int(p.end)) for p in f.location.parts] for f in products]
    spans = [(min(s for s, _ in ps), max(e for _, e in ps)) for ps in parts]
    order = sorted(range(len(products)), key=lambda i: spans[i])
    out = []
    for x, i in enumerate(order):
        for j in order[x + 1 :]:
            if spans[j][0] >= spans[i][1]:
                break
            shared = sum(max(0, min(e1, e2) - max(s1, s2)) for s1, e1 in parts[i] for s2, e2 in parts[j])
            if not shared:
                continue
            a, b = products[i], products[j]
            shorter = min(len(a.location), len(b.location))
            contained = shared >= 0.9 * shorter
            known = frozenset((_name_of(a), _name_of(b))) in _KNOWN_OVERLAPS and shared <= _KNOWN_OVERLAP_NT
            if contained or (shared >= _OVERLAP_MIN_NT and not known):
                kind = "inside" if contained else "overlap"
                out.append(
                    f"{kind}: {a.type} {_name_of(a) or '?'} {_span_text(a)} and {b.type} {_name_of(b) or '?'} "
                    f"{_span_text(b)} share {shared} bp"
                )
    return out


def _span_text(feature: SeqFeature) -> str:
    ps = feature.location.parts
    lo = min(int(p.start) for p in ps) + 1
    hi = max(int(p.end) for p in ps)
    return f"{lo}-{hi}({'+' if (feature.location.strand or 1) == 1 else '-'})"


def _overlaps_cds(feature: SeqFeature, others: list[SeqFeature]) -> bool:
    """True when over half of ``feature``'s bases lie in a CDS of ``others`` on the same strand."""

    def bases(f):
        return {(p.strand, i) for p in f.location.parts for i in range(int(p.start), int(p.end))}

    mine = bases(feature)
    if not mine:
        return False
    return any(f.type == "CDS" and len(mine & bases(f)) > len(mine) / 2 for f in others)


#: A supplemented CDS shorter than this share of the donor reference's CDS is a
#: remnant (Cephalotaxus rps16 43 of ~85 aa, Lagenaria infA 49 of ~77 aa).
_SUPPLEMENT_MIN_LENGTH = 0.8


def _is_open_frame(feature: SeqFeature, target_seq: str) -> bool:
    try:
        cds = str(feature.extract(Seq(target_seq))).upper()
    except Exception:
        return False
    if len(cds) % 3 or len(cds) < 90:
        return False
    if len(cds) < _SUPPLEMENT_MIN_LENGTH * getattr(feature, "reference_length", 0):
        return False
    protein = str(Seq(cds).translate(table=11))
    return "*" not in protein[:-1]


def _drop_identical_features(features: list[SeqFeature]) -> list[SeqFeature]:
    """Keep the first of CDS/tRNA/rRNA features with the same type, gene and location.

    ndhB came out twice at the same coordinates in six development species
    (a second query hit the same locus); no tool reports an identical copy twice.
    """
    seen: set[tuple] = set()
    out: list[SeqFeature] = []
    for feature in features:
        if feature.type in ("CDS", "tRNA", "rRNA") and _name_of(feature):
            key = (
                feature.type,
                _name_of(feature),
                tuple((int(p.start), int(p.end), p.strand) for p in feature.location.parts),
            )
            if key in seen:
                continue
            seen.add(key)
        out.append(feature)
    return out


def _merge_abutting_parts(features: list[SeqFeature]) -> int:
    """Join consecutive same-strand CDS parts that touch or overlap; return how many joins.

    A reference exon split transferred onto a genome that lost the intron
    (legume clpP intron 1) leaves two blocks with no bases between them, an
    intron of length zero. Parts across the circular origin are not touching
    in coordinates and stay separate.
    """
    merged = 0
    for feature in features:
        if feature.type != "CDS" or not isinstance(feature.location, CompoundLocation):
            continue
        parts = list(feature.location.parts)
        out = [parts[0]]
        for part in parts[1:]:
            prev = out[-1]
            if part.strand == prev.strand == 1 and prev.start <= part.start <= prev.end:
                out[-1] = FeatureLocation(prev.start, max(prev.end, part.end), strand=1)
            elif part.strand == prev.strand == -1 and prev.start <= part.end <= prev.end:
                out[-1] = FeatureLocation(min(prev.start, part.start), prev.end, strand=-1)
            else:
                out.append(part)
                continue
            merged += 1
        if len(out) < len(parts):
            feature.location = out[0] if len(out) == 1 else CompoundLocation(out)
    return merged


def _retranslate_cds_features(features: list[SeqFeature], target_seq: str) -> None:
    """Rewrite CDS ``translation`` qualifiers from the spliced target sequence.

    Reference transfer copies the reference protein into the qualifier, but a
    partial transfer leaves the qualifier describing a protein the emitted CDS
    does not encode. GenBank semantics (and the released document validator)
    require ``translation`` to be the translation of the spliced CDS itself, so
    it is recomputed here; when the transferred boundary is not codon-exact
    (frame-shifted or internal stops), the qualifier is dropped rather than
    asserting a false protein.
    """
    record_seq = Seq(target_seq)
    for feature in features:
        if feature.type != "CDS":
            continue
        try:
            cds = feature.extract(record_seq)
            if len(cds) % 3 != 0:
                feature.qualifiers.pop("translation", None)
                continue
            protein = _apply_transl_except(feature, str(cds.translate(table=11)).removesuffix("*"))
        except Exception:
            feature.qualifiers.pop("translation", None)
            continue
        if "*" in protein:
            feature.qualifiers.pop("translation", None)
            continue
        feature.qualifiers["translation"] = [protein]


def _make_output_record(
    sample_name: str,
    organism: str,
    target_seq: str,
    *,
    genome_form: str,
    features: list[SeqFeature],
) -> SeqRecord:
    record = SeqRecord(
        Seq(target_seq),
        id=sample_name[:16],
        name=sample_name[:16],
        description=f"{organism} chloroplast genome annotated by OrganelleVerse",
    )
    record.annotations["molecule_type"] = "DNA"
    record.annotations["topology"] = genome_form
    record.annotations["organism"] = organism
    record.features = [
        SeqFeature(
            FeatureLocation(0, len(target_seq), strand=1),
            type="source",
            qualifiers={
                "organism": [organism],
                "mol_type": ["genomic DNA"],
                "organelle": ["plastid:chloroplast"],
            },
        ),
        *features,
    ]
    return record


def _reference_gene_set(reference: ReferenceRecord) -> set[str]:
    genes = set()
    for ref_feature in reference.features:
        if ref_feature.feature_type == "CDS_exon":
            continue
        if ref_feature.feature_type in {"gene", "CDS", "tRNA", "rRNA"}:
            gene = _standardize_gene_qualifier(
                ref_feature.feature_type,
                ref_feature.feature.qualifiers,
            )
            if gene:
                genes.add(_canonical_gene_for_set(gene))
    return genes


def _annotated_gene_set(features: list[SeqFeature]) -> set[str]:
    genes = set()
    for feature in features:
        gene = _standardize_gene_qualifier(feature.type, feature.qualifiers)
        if gene:
            genes.add(_canonical_gene_for_set(gene))
    return genes


def _canonical_gene_for_set(gene: str) -> str:
    return gene.replace("rps12+1", "rps12").replace("rps12+2", "rps12")


def _standardize_gene_qualifier(
    feature_type: str,
    qualifiers: dict[str, list[str]],
) -> str:
    gene = qualifiers.get("gene", [""])[0]
    if standard := _standardize_trna_gene(gene):
        return standard
    if standard := _infer_edited_trna_gene(gene, qualifiers):
        return standard
    if (feature_type == "rRNA" or gene.lower().startswith("rrn")) and (
        standard := _standardize_rrna_gene(gene)
    ):
        return standard
    if feature_type == "rRNA":
        product = qualifiers.get("product", [""])[0]
        if standard := _standardize_rrna_gene(product):
            return standard
    return gene


def _standardize_trna_gene(gene: str) -> str | None:
    compact = gene.strip()
    match = re.fullmatch(r"trn([A-Za-z]{1,2})[-(]([ACGTUacgtu]{3})\)?", compact)
    if match:
        amino_acid = match.group(1)
        anticodon = match.group(2).upper().replace("T", "U")
        return f"trn{amino_acid}-{anticodon}"
    match = re.fullmatch(r"tRNA-([A-Za-z]+)\s*\(([ACGTUacgtu]{3})\)", compact)
    if not match:
        return None
    amino_acid = _AA3_TO_1.get(match.group(1).lower())
    if not amino_acid:
        return None
    anticodon = match.group(2).upper().replace("T", "U")
    return f"trn{amino_acid}-{anticodon}"


def _infer_edited_trna_gene(gene: str, qualifiers: dict[str, list[str]]) -> str | None:
    if gene != "trnI":
        return None
    product = qualifiers.get("product", [""])[0].lower()
    note = " ".join(qualifiers.get("note", [])).lower()
    if product == "trna-ile" and "lysidine" in note:
        return "trnI-CAU"
    return None


def _standardize_rrna_gene(gene: str) -> str | None:
    compact = gene.strip()
    match = re.search(r"(?:rrn)?(4\.5|5|16|23)S?(?:\s+ribosomal\s+RNA)?$", compact, re.I)
    if not match:
        return None
    return f"rrn{match.group(1)}"


def _write_warning_log(path: Path, stats: PlastomeAnnotationStats) -> None:
    with path.open("w") as handle:
        handle.write(f"{stats.sample_name}\n")
        for warning in stats.warnings:
            handle.write(f"Warning: {warning}\n")
        handle.write(
            f"Total number of genes in the reference plastome(s): {stats.total_genes_reference}.\n"
        )
        annotated_gene_count = stats.total_genes_reference - len(stats.unannotated_genes)
        handle.write(
            f"Total number of genes annotated in the target plastome: {annotated_gene_count}.\n"
        )
        handle.write(
            "All gene names from the reference plastome(s) that were not annotated in the target plastome:\n"
        )
        if stats.unannotated_genes:
            handle.write("\t".join(stats.unannotated_genes))
        handle.write("\n")


def _parse_qcoverage(value: str) -> tuple[float, float]:
    try:
        lo, hi = value.split(",", 1)
        return float(lo), float(hi)
    except Exception:
        return 0.5, 2.0


def _organism_for_sample(sample_name: str, description: str) -> str:
    words = description.replace("_", " ").split()
    if len(words) >= 2 and words[0][0].isupper() and words[1][0].islower():
        return f"{words[0]} {words[1]}"
    return sample_name.replace("_", " ")


def _promote_complete_exon_models(
    features: list[SeqFeature],
    models: list[ExonModel],
    ref_proteins: dict[str, list[str]],
    product_map: dict[str, str],
) -> list[str]:
    """Promote strictly better complete products under a shared protein bank.

    Equal scores retain the current model. Different-DNA score ties, and
    overlapping alternative placements of one mature DNA, stay unresolved.
    Identical shared exons are allowed between distinct trans-spliced copies.
    """
    warnings = []
    for gene in sorted({m.gene for m in models}):
        bank = ref_proteins.get(gene, [])
        if not bank:
            warnings.append(f"{gene} complete candidates lack a reference protein bank")
            continue
        scored = [
            replace(m, protein_score=_common_protein_score(m.protein, bank))
            for m in models
            if m.gene == gene
        ]
        selected = select_mature_model_copies(scored)
        if not selected:
            warnings.append(f"{gene} complete candidates have unresolved mature-DNA score ties")
            continue
        # Boundary alternatives at the same physical locus are not extra copies.
        overlaps = any(
            a != b and max(a.start, b.start) < min(a.end, b.end)
            for left, right in combinations(selected, 2)
            for a in left.location.parts
            for b in right.location.parts
        )
        if overlaps:
            warnings.append(f"{gene} complete candidates have unresolved overlapping placements")
            continue
        current = [
            f for f in features if f.type == "CDS" and f.qualifiers.get("gene", [""])[0] == gene
        ]
        # A missing translation is outside the protein objective's domain,
        # not evidence of a lower score (for example, an RNA-edited CDS).
        if any(not f.qualifiers.get("translation") for f in current):
            warnings.append(f"{gene} complete candidates cannot compare an unscored current product")
            continue
        current_scores = [
            _common_protein_score(f.qualifiers["translation"][0], bank)
            for f in current
            if f.qualifiers.get("translation")
        ]
        if current_scores and selected[0].protein_score <= max(current_scores):
            continue
        features[:] = [f for f in features if f not in current]
        features.extend(_copy_feature(m.template, m.location, product_map) for m in selected)
    return warnings


def _common_protein_score(protein: str, bank: list[str]) -> float:
    return max(_score(_san(p.rstrip("*")), _san(protein.rstrip("*"))) for p in bank)
