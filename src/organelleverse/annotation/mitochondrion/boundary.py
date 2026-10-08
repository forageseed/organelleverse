"""Gene boundary correction.

Translates PMGA v1 logic from 02.editBoundary.py:
- Exon-intron boundary adjustment via sliding window
- Start codon correction (ACG->AUG RNA editing)
- Stop codon correction (CAA->UAA RNA editing for stop-gain genes)
- Short intron removal (<150 bp)
- rpl16 truncation handling
- Multi-exon gene processing
- Fixed offset correction for genes with systematic position errors (Round 2 improvement)
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from collections.abc import Mapping
from itertools import pairwise
from pathlib import Path

from organelleverse._losat import (
    losat_from_tool_paths,
    run_losat_with_query_coverage,
)
from organelleverse.core.errors import OrganelleDependencyError, OrganelleExecutionError
from organelleverse.core.external import run_external

from ..execution import CommandRunner
from ._tuning import tuned
from .boundary_db import refine_exon_boundaries
from .codon_contract import allowed_start_codons
from .db import DBManager
from .models.gene import ExonRecord, GeneAnnotation, Strand
from .models.genome import GenomeSequence
from .pcg import START_CODONS, STOP_CODONS, translate_sequence
from .trans_splicing import TRANS_SPLICED_CONFIG

#: Reverse-complement translation table.
#:
#: Module level because it was previously bound inside a conditional branch and
#: read outside it: ``_handle_special_genes`` defined ``_rc`` only on the
#: minus-strand path of one check, then used it unconditionally 200 lines later,
#: so any gene reaching the second site without having taken the first branch
#: raised UnboundLocalError and failed the whole annotation. Ginkgo biloba does
#: exactly that.
_RC_TABLE = str.maketrans("ATGCatgcNn", "TACGtacgNn")

logger = logging.getLogger(__name__)

# Stop-gain RNA editing: C->U editing creates stop codons
STOP_GAIN_CODONS = {"CAA", "CAG", "CGA"}
_TRUE_TRANS_SPLICED_GENES = {"nad1", "nad2", "nad5"}


def _has_trans_spliced_layout(ann: GeneAnnotation) -> bool:
    if ann.gene_name.casefold() not in _TRUE_TRANS_SPLICED_GENES or len(ann.exons) < 2:
        return False
    starts = [exon.start for exon in ann.exons]
    monotonic = (
        all(first <= second for first, second in pairwise(starts))
        if ann.strand == Strand.PLUS
        else all(first >= second for first, second in pairwise(starts))
    )
    scattered = any(
        abs(first.start - second.start) > 10_000 for first, second in pairwise(ann.exons)
    )
    return not monotonic or scattered


def _in_genomic_order(step, ann: GeneAnnotation, *args) -> GeneAnnotation:
    """Run a start/stop codon step on a minus-strand cis gene with exons in ascending order.

    The minus-strand branches of the start/stop corrections take ``exons[-1]``
    as the 5' exon and ``exons[0]`` as the 3' exon, i.e. they expect genomic
    order. Minus-strand genes built by BLAST arrive in transcription order
    (high to low), so the stop search ran from exon 1's donor and moved that
    splice site instead of the stop (rpl2 exon 1 lost 21 nt in Arabidopsis and
    12 of 13 PMGA Table S2 species). The original exon order is restored and
    exons are renumbered in transcription order.
    """
    exons = list(ann.exons)
    if (
        ann.strand != Strand.MINUS
        or len(exons) < 2
        or _has_trans_spliced_layout(ann)
        or len({exon.strand for exon in exons}) > 1
    ):
        return step(ann, *args)
    descending = exons[0].start > exons[-1].start
    ascending = sorted(exons, key=lambda e: e.start)
    out = step(ann.model_copy(update={"exons": ascending}), *args)
    unchanged = [(e.start, e.end) for e in out.exons] == [(e.start, e.end) for e in ascending]
    if unchanged and out.notes == ann.notes and out.exceptions == ann.exceptions:
        return ann
    ordered = sorted(out.exons, key=lambda e: e.start, reverse=descending)
    by_tx = sorted(ordered, key=lambda e: e.start, reverse=True)
    number = {id(e): i for i, e in enumerate(by_tx, 1)}
    return out.model_copy(
        update={"exons": [e.model_copy(update={"number": number[id(e)]}) for e in ordered]}
    )


# After editing: CAA->UAA, CAG->UAG, CGA->UGA.
SHORT_INTRON_THRESHOLD = 150  # bp, introns shorter than this are removed

# Fixed offset correction for genes with systematic position errors
# Based on Round 1 validation analysis
FIXED_OFFSET_GENES = {
    # cox2: ~1400bp fixed offset (reference boundary definition issue)
    "cox2": {"start_offset": -1400, "end_offset": 0, "reason": "reference_boundary"},
    # rps10: ~900bp offset (RNA editing STGE - start-gain error)
    "rps10": {"start_offset": -900, "end_offset": 0, "reason": "RNA_editing_STGE"},
    # nad7: ~75bp offset (start codon boundary detection)
    "nad7": {"start_offset": -75, "end_offset": 0, "reason": "start_codon_detection"},
    # rps14: ~84bp offset (start codon boundary detection)
    "rps14": {"start_offset": -84, "end_offset": 0, "reason": "start_codon_detection"},
}


def correct_boundaries(
    annotations: list[GeneAnnotation],
    genome: GenomeSequence,
    db_manager: DBManager,
    search_range: int = 30,  # Very conservative: only 30bp adjustment
    *,
    tool_paths: Mapping[str, str] | None = None,
    command_runner: CommandRunner | None = None,
) -> list[GeneAnnotation]:
    """Apply all boundary corrections to annotations.

    Steps:
    1. Remove short introns (merge exons separated by <150 bp)
    2. Apply fixed offset correction for known systematic errors
    3. Correct start codons (search upstream for ATG/ACG) - CONSERVATIVE
    4. Correct stop codons (search downstream for stop) - CONSERVATIVE
    5. Handle rpl16 truncation
    6. Handle special start codons (mttB ATA, etc.)

    Args:
        annotations: List of gene annotations
        genome: Genome sequence
        db_manager: Database manager for gene metadata
        search_range: How far upstream/downstream to search (bp, default 30)
            Very conservative to prevent over-extension into adjacent genes.
            Only adjust if start/stop codon is very close to HMM boundary.

    Returns:
        Corrected annotations
    """
    corrected = []
    for ann in annotations:
        # Use gene-specific search range if available
        gene_search_range = _get_gene_search_range(ann.gene_name, search_range)

        ann = _remove_short_introns(ann, genome)
        # Data-driven internal exon/intron boundary refinement against the
        # reference boundary-flank database (replaces hardcoded offsets for
        # multi-exon genes; handles non-canonical group II boundaries).
        ann = _refine_exon_intron_boundaries(ann, genome)
        # Phase 3: adaptive tblastn boundary refinement (replaces fixed offsets where possible)
        ann = _refine_boundary_by_tblastn(
            ann,
            genome,
            db_manager,
            tool_paths=tool_paths,
            command_runner=command_runner,
        )
        # Phase 3: apply fixed offset correction only if tblastn did not refine the boundary
        if ann.source_method != "tblastn":
            ann = _apply_fixed_offset_correction(ann, genome)
        else:
            logger.info(f"Skipping fixed offset for {ann.gene_name} (tblastn refined)")
        # Only do minimal boundary correction - trust HMM hit more
        ann = _in_genomic_order(
            _correct_start_codon_conservative, ann, genome, db_manager, gene_search_range
        )
        ann = _handle_special_genes(ann, genome, db_manager)
        # Run stop codon correction AFTER _handle_special_genes (_LENGTH_LIMITED
        # may trim the start, changing the reading frame and gene length, which
        # affects which stop-gain codons are acceptable)
        ann = _in_genomic_order(
            _correct_stop_codon_conservative, ann, genome, db_manager, gene_search_range
        )
        ann = _validate_gene_length(ann, db_manager)
        # Phase 3: restore codon phase continuity across exons after boundary shifts
        ann = _restore_phase_continuity(ann, genome)
        ann = repair_complete_cds_terminals(
            ann,
            genome,
            db_manager,
            max(gene_search_range, 150),
        )
        ann = pmga_frame_repair(ann, genome, db_manager)
        corrected.append(ann)
    return corrected


def _refine_exon_intron_boundaries(ann: GeneAnnotation, genome: GenomeSequence) -> GeneAnnotation:
    """Pin internal exon/intron junctions of a multi-exon CDS against the
    reference boundary-flank PWM database (data-driven, species-general).

    Only internal boundaries move; the 5' start and 3' end are left for
    start/stop-codon refinement. Non-canonical group II boundaries (3' AY, not
    AG) are handled because the database encodes the real conserved flanks.
    """
    if ann.gene_type != "CDS" or len(ann.exons) < 2:
        return ann
    strand = int(ann.strand)
    # True trans-spliced exon numbers define biological order; genomic sorting
    # silently changes their protein sequence.
    if ann.gene_name.casefold() in _TRUE_TRANS_SPLICED_GENES:
        ordered_exons = list(ann.exons)
    else:
        ordered_exons = sorted(ann.exons, key=lambda e: e.start, reverse=(strand == -1))
    coords = [(e.start, e.end) for e in ordered_exons]
    refined = refine_exon_boundaries(ann.gene_name, coords, genome.sequence, strand)
    if refined is None:
        return ann
    # Phases follow the new lengths: _restore_phase_continuity trusts the stored
    # phase, and a stale one made it move a just-pinned donor by 1 nt (ccmFC in
    # all 29 PMGA Table S2 species, whose acceptor the DB had moved 20 nt).
    new_exons = []
    cumulative = 0
    for (s, e), old in zip(refined, ordered_exons, strict=True):
        new_exons.append(
            ExonRecord(start=s, end=e, strand=ann.strand, number=old.number, phase=cumulative % 3)
        )
        cumulative += e - s + 1
    logger.info(f"Boundary-DB refined internal junctions for {ann.gene_name}")
    return ann.model_copy(update={"exons": new_exons})


def _has_internal_stop(ann: GeneAnnotation, genome: GenomeSequence) -> bool:
    """True when the spliced CDS (transcription order, read from its start) has a stop before its last codon."""
    minus = ann.strand == Strand.MINUS
    parts = []
    for exon in sorted(ann.exons, key=lambda e: e.start, reverse=minus):
        seq = genome.get_sequence_for_range(exon.start, exon.end).upper()
        parts.append(seq.translate(str.maketrans("ACGT", "TGCA"))[::-1] if minus else seq)
    cds = "".join(parts)
    codons = [cds[k : k + 3] for k in range(0, len(cds) - 2, 3)]
    return any(codon in STOP_CODONS for codon in codons[:-1])


def _restore_phase_continuity(ann: GeneAnnotation, genome: GenomeSequence) -> GeneAnnotation:
    """Micro-adjust exon boundaries to maintain codon phase across exons.

    After start/stop codon correction, an exon's length may change and break
    the reading frame for subsequent exons. This function attempts ±1 bp or
    ±2 bp shifts of the preceding exon's boundary to restore phase continuity.
    Adjustments are only accepted if they keep the exon ≥3 bp and do not
    overlap the next exon.
    """
    if len(ann.exons) <= 1 or _has_trans_spliced_layout(ann):
        return ann
    # The stored phases come from the hit that built the gene and go stale when
    # boundaries move; an open reading frame is the real test. Moving a donor of
    # a CDS without internal stops only broke it (rps3 donor +1 nt in 10 PMGA
    # Table S2 species, whose 3' end was merely short).
    if not _has_internal_stop(ann, genome):
        return ann

    exons = list(ann.exons)
    modified = False

    for i in range(1, len(exons)):
        cumulative_len = sum(e.end - e.start + 1 for e in exons[:i])
        expected_phase = cumulative_len % 3
        actual_phase = exons[i].phase

        if expected_phase == actual_phase:
            continue

        # Determine required length change for previous exon
        delta = (actual_phase - expected_phase) % 3
        if delta == 1:
            shifts = [+1, -2]
        elif delta == 2:
            shifts = [+2, -1]
        else:
            continue

        prev = exons[i - 1]
        nxt = exons[i]

        for shift in shifts:
            if ann.strand == Strand.PLUS:
                new_end = prev.end + shift
                if new_end < prev.start + 2:
                    continue
                if new_end >= nxt.start:
                    continue
                exons[i - 1] = ExonRecord(
                    start=prev.start,
                    end=new_end,
                    strand=prev.strand,
                    number=prev.number,
                    phase=prev.phase,
                )
                modified = True
                break
            else:
                # Minus strand: transcription is high->low coords.
                # To change prev exon length by +shift, move start down.
                new_start = prev.start - shift
                if new_start > prev.end - 2:
                    continue
                if new_start <= nxt.end:
                    continue
                exons[i - 1] = ExonRecord(
                    start=new_start,
                    end=prev.end,
                    strand=prev.strand,
                    number=prev.number,
                    phase=prev.phase,
                )
                modified = True
                break

    if modified:
        notes = list(ann.notes)
        notes.append("phase continuity restored by micro-adjustment")
        return ann.model_copy(update={"exons": exons, "notes": notes})

    return ann.model_copy(update={"exons": exons})


#: PMGA 03.CDSCheck.py default slide extent (its main() falls back to 250).
_PMGA_SLIDE_EXTENT = 250


def pmga_frame_repair(
    ann: GeneAnnotation,
    genome: GenomeSequence,
    db_manager: DBManager,
) -> GeneAnnotation:
    """Repair reading frames and terminal codons the way PMGA does.

    Direct translation of PMGA 03.CDSCheck.py:
    - ``internal_edit``: per-exon ±1 base shifts at the transcript 5' end,
      accepted greedily when the cumulative translation loses its internal
      stop; the gene is left untouched if no shift fixes it.
    - ``check_reading2``: when internal stops cluster in one half, trim that
      side (steps of 3) until the translation is clean.
    - ``start_edit``: slide the 5' boundary outward in steps of 3 up to
      ``_PMGA_SLIDE_EXTENT`` looking for ATG (or ACG with RNA editing, or GTG
      when the gene's contract allows it); a stop codon aborts the slide.
    - ``stop_edit``: slide the 3' boundary outward in steps of 3 looking for
      a stop codon; stop-gain codons (CAA/CGA/CAG) are accepted only for
      genes the database lists as stop-gain, marked with an RNA editing
      exception.

    Convergent by construction: every edit is kept only when it removes the
    defect it targets, and the final annotation is returned only when the
    spliced CDS is in frame with accepted terminal codons and no unexplained
    internal stop. Valid genes pass through untouched.
    """

    if ann.gene_type != "CDS" or ann.is_pseudo or not ann.exons:
        return ann

    from .cds import _extract_cds_sequence
    from .pcg import STOP_GAIN_CODONS

    def spliced(exons: list[ExonRecord]) -> str:
        return _extract_cds_sequence(ann.model_copy(update={"exons": exons}), genome)

    exons = list(ann.exons)
    notes = list(ann.notes)
    exceptions = list(ann.exceptions)
    seq = spliced(exons)
    if not seq:
        return ann
    protein = translate_sequence(seq)

    def _internal_edit(exons: list[ExonRecord], seq: str) -> tuple[list[ExonRecord], str] | None:
        """Per-exon ±1 base 5' shifts until no internal stop (PMGA internal_edit).

        Returns the repaired exons and spliced sequence, or None when the gene
        cannot be adjusted (in which case the caller must leave it untouched).
        """
        working = list(exons)
        working_seq = seq
        for index in range(len(working)):
            exon = working[index]
            prefix_len = sum(
                len(genome.get_sequence_for_range(e.start, e.end, e.strand))
                for e in working[:index]
            )
            prefix = working_seq[:prefix_len]
            exon_seq = genome.get_sequence_for_range(exon.start, exon.end, exon.strand)
            cumulative = prefix + exon_seq
            if len(cumulative) < 6:
                continue
            if "*" not in translate_sequence(cumulative)[:-1]:
                continue
            shifted_options: list[ExonRecord] = []
            if exon.strand == Strand.PLUS:
                if exon.start > 1:
                    shifted_options.append(
                        exon.model_copy(update={"start": exon.start - 1})
                    )
                if exon.start < exon.end:
                    shifted_options.append(
                        exon.model_copy(update={"start": exon.start + 1})
                    )
            else:
                if exon.end < genome.length:
                    shifted_options.append(
                        exon.model_copy(update={"end": exon.end + 1})
                    )
                if exon.end > exon.start:
                    shifted_options.append(
                        exon.model_copy(update={"end": exon.end - 1})
                    )
            for shifted in shifted_options:
                trial = list(working)
                trial[index] = shifted
                shifted_seq = genome.get_sequence_for_range(
                    shifted.start, shifted.end, shifted.strand
                )
                if "*" not in translate_sequence(prefix + shifted_seq)[:-1]:
                    working = trial
                    working_seq = spliced(trial)
                    break  # PMGA continues with the next exon
            else:
                return None  # PMGA: cannot adjust this gene
        return working, working_seq

    # PMGA classifies problems independently: internal-stop genes go through
    # internal_edit (then check_reading2 when internal_edit gives up), start
    # problems through start_edit, stop problems through stop_edit. A gene
    # with both an internal stop and a bad start (e.g. atp8) must still reach
    # start_edit after internal_edit fails.
    if "*" in protein[:-1]:
        repaired = _internal_edit(exons, seq)
        if repaired is not None:
            exons, seq = repaired
            protein = translate_sequence(seq)
            if "*" not in protein[:-1]:
                notes.append("reading frame adjusted by internal_edit (PMGA)")

    if "*" in protein[:-1]:
        # PMGA check_reading2: internal stops cluster in one half — trim that
        # side in steps of 3 until the translation is clean.
        stop_positions = [i + 1 for i, aa in enumerate(protein[:-1]) if aa == "*"]
        code_num = len(protein)
        candidate: list[ExonRecord] | None = None
        if max(stop_positions) - (code_num - 1) / 2 <= 0:
            trim = max(stop_positions) * 3
            first = exons[0]
            if first.strand == Strand.PLUS:
                new_first = first.model_copy(update={"start": first.start + trim})
            else:
                new_first = first.model_copy(update={"end": first.end - trim})
            if new_first.start <= new_first.end:
                trial = list(exons)
                trial[0] = new_first
                candidate = trial
        else:
            stop_long = code_num - min(stop_positions)
            encumbrance = len(seq) % 3
            trim = (stop_long - 1) * 3 + encumbrance
            last = exons[-1]
            if last.strand == Strand.PLUS:
                new_last = last.model_copy(update={"end": last.end - trim})
            else:
                new_last = last.model_copy(update={"start": last.start + trim})
            if new_last.start <= new_last.end:
                trial = list(exons)
                trial[-1] = new_last
                candidate = trial
        if candidate is not None:
            trial_seq = spliced(candidate)
            trial_protein = translate_sequence(trial_seq)
            if (
                len(trial_seq) % 3 == 0
                and len(trial_seq) >= 0.5 * len(seq)
                and "*" not in trial_protein[:-1]
            ):
                exons = candidate
                seq = trial_seq
                protein = trial_protein
                notes.append("terminal trim by check_reading2 (PMGA)")

    is_start_gain = db_manager.is_start_gain_gene(ann.gene_name)
    valid_starts = allowed_start_codons(
        ann.gene_name,
        allow_rna_editing=is_start_gain,
    )
    first_codon = seq[:3].upper()
    if len(seq) >= 3 and first_codon not in valid_starts:
        first = exons[0]
        slid = 0
        while slid <= _PMGA_SLIDE_EXTENT:
            slid += 3
            if first.strand == Strand.PLUS:
                new_start = first.start - slid
                if new_start < 1:
                    break
                trial_first = first.model_copy(update={"start": new_start})
            else:
                new_end = first.end + slid
                if new_end > genome.length:
                    break
                trial_first = first.model_copy(update={"end": new_end})
            trial = list(exons)
            trial[0] = trial_first
            trial_seq = spliced(trial)
            codon = trial_seq[:3].upper()
            if codon == "ATG":
                exons = trial
                seq = trial_seq
                break
            if codon == "ACG" and is_start_gain:
                exons = trial
                seq = trial_seq
                if "RNA editing" not in exceptions:
                    exceptions.append("RNA editing")
                notes.append("start codon is created by C to U RNA editing (ACG)")
                break
            if codon == "GTG" and "GTG" in valid_starts:
                exons = trial
                seq = trial_seq
                notes.append("start codon is not determined (GTG)")
                break
            if codon in STOP_CODONS:
                break  # PMGA: stop searching, start stays undetermined

    def terminal_codon(seq: str) -> str:
        """In-frame terminal codon position (PMGA check_each_genes: the check
        depends on len%3, so a frame-shifted CDS is treated as a stop problem
        even when its trailing triplet happens to be a stop)."""
        frame = len(seq) % 3
        if frame == 0:
            return seq[-3:].upper()
        return seq[-(3 + frame) : -frame].upper()

    last_codon = terminal_codon(seq)
    has_exception = any(
        exception.strip().casefold() == "rna editing" for exception in exceptions
    )
    is_stop_gain = db_manager.is_stop_gain_gene(ann.gene_name)
    terminal_ok = last_codon in STOP_CODONS or (
        is_stop_gain and has_exception and last_codon in STOP_GAIN_CODONS
    )
    if len(seq) >= 3 and not terminal_ok:
        last = exons[-1]
        if (
            len(seq) % 3 == 0
            and is_stop_gain
            and last_codon in STOP_GAIN_CODONS
        ):
            # The terminal already sits on an editable stop-gain codon; only
            # the exception qualifier is missing.
            if "RNA editing" not in exceptions:
                exceptions.append("RNA editing")
            notes.append("stop codon is created by C to U RNA editing")
        else:
            slid = 0
            while slid <= _PMGA_SLIDE_EXTENT:
                # The first extension aligns the reading frame; afterwards the
                # slide keeps the frame by stepping 3 (PMGA stop_edit behaviour).
                step = 3 - ((len(seq) + slid) % 3) or 3
                slid += step
                if last.strand == Strand.PLUS:
                    new_end = last.end + slid
                    if new_end > genome.length:
                        break
                    trial_last = last.model_copy(update={"end": new_end})
                else:
                    new_start = last.start - slid
                    if new_start < 1:
                        break
                    trial_last = last.model_copy(update={"start": new_start})
                trial = list(exons)
                trial[-1] = trial_last
                trial_seq = spliced(trial)
                codon = terminal_codon(trial_seq)
                if codon in STOP_CODONS:
                    exons = trial
                    seq = trial_seq
                    break
                if is_stop_gain and codon in STOP_GAIN_CODONS:
                    exons = trial
                    seq = trial_seq
                    if "RNA editing" not in exceptions:
                        exceptions.append("RNA editing")
                    notes.append("stop codon is created by C to U RNA editing")
                    break

    if exons == ann.exons and exceptions == ann.exceptions:
        return ann

    # Convergent guard: only publish the repaired annotation when the spliced
    # CDS now satisfies the same terminal/frame contract validation applies.
    candidate = ann.model_copy(
        update={"exons": exons, "notes": notes, "exceptions": exceptions}
    )
    final_seq = spliced(exons)
    if len(final_seq) < 3 or len(final_seq) % 3 != 0:
        return ann
    final_first = final_seq[:3].upper()
    final_last = terminal_codon(final_seq)
    final_exception = any(
        exception.strip().casefold() == "rna editing"
        for exception in candidate.exceptions
    )
    if final_first not in allowed_start_codons(
        candidate.gene_name,
        allow_rna_editing=db_manager.is_start_gain_gene(candidate.gene_name),
    ):
        return ann
    if final_last not in STOP_CODONS and not (
        db_manager.is_stop_gain_gene(candidate.gene_name)
        and final_exception
        and final_last in STOP_GAIN_CODONS
    ):
        return ann
    return candidate


def repair_complete_cds_terminals(
    ann: GeneAnnotation,
    genome: GenomeSequence,
    db_manager: DBManager,
    search_range: int,
) -> GeneAnnotation:
    """Select a nearby start/stop pair that yields one complete open reading frame.

    Boundary evidence can be a few bases inside or outside the true CDS. This
    final bounded search changes only the biological first and last exon, keeps
    exon order intact, and accepts a pair only when the complete spliced CDS is
    in frame and contains no unexplained internal stop.
    """

    if ann.gene_type != "CDS" or ann.is_pseudo or not ann.exons:
        return ann

    from .cds import _extract_cds_sequence

    radius = max(1, search_range)
    first = ann.exons[0]
    last = ann.exons[-1]
    current_start = first.start if first.strand == Strand.PLUS else first.end
    current_stop = last.end if last.strand == Strand.PLUS else last.start
    allowed_starts = _get_allowed_start_codons(ann.gene_name, db_manager)
    allow_edited_stop = db_manager.is_stop_gain_gene(ann.gene_name)

    start_candidates: list[int] = []
    for coordinate in range(
        max(1, current_start - radius),
        min(genome.length, current_start + radius) + 1,
    ):
        if first.strand == Strand.PLUS:
            if coordinate > first.end:
                continue
            codon = genome.get_sequence_for_range(coordinate, coordinate + 2, Strand.PLUS)
        else:
            if coordinate < first.start:
                continue
            codon = genome.get_sequence_for_range(coordinate - 2, coordinate, Strand.MINUS)
        if len(codon) == 3 and codon.upper() in allowed_starts:
            start_candidates.append(coordinate)

    stop_candidates: list[tuple[int, bool]] = []
    for coordinate in range(
        max(1, current_stop - radius),
        min(genome.length, current_stop + radius) + 1,
    ):
        if last.strand == Strand.PLUS:
            if coordinate < last.start:
                continue
            codon = genome.get_sequence_for_range(coordinate - 2, coordinate, Strand.PLUS)
        else:
            if coordinate > last.end:
                continue
            codon = genome.get_sequence_for_range(coordinate, coordinate + 2, Strand.MINUS)
        codon = codon.upper()
        if len(codon) != 3:
            continue
        if codon in STOP_CODONS:
            stop_candidates.append((coordinate, False))
        elif allow_edited_stop and codon in STOP_GAIN_CODONS:
            stop_candidates.append((coordinate, True))

    current_length = ann.total_exon_length
    minimum_length = max(3, min(0.65 * current_length, current_length - 6))
    maximum_length = max(1.35 * current_length, current_length + 6)
    choices: list[tuple[tuple[int, int, int], GeneAnnotation, bool]] = []
    for start_coordinate in start_candidates:
        for stop_coordinate, edited_stop in stop_candidates:
            exons = list(ann.exons)
            first_exon = exons[0]
            if first_exon.strand == Strand.PLUS:
                exons[0] = first_exon.model_copy(update={"start": start_coordinate})
            else:
                exons[0] = first_exon.model_copy(update={"end": start_coordinate})
            last_index = len(exons) - 1
            last_exon = exons[last_index]
            if last_exon.strand == Strand.PLUS:
                exons[last_index] = last_exon.model_copy(update={"end": stop_coordinate})
            else:
                exons[last_index] = last_exon.model_copy(update={"start": stop_coordinate})
            if any(exon.start > exon.end for exon in exons):
                continue
            candidate = ann.model_copy(update={"exons": exons})
            sequence = _extract_cds_sequence(candidate, genome)
            if not sequence or len(sequence) % 3:
                continue
            if not minimum_length <= len(sequence) <= maximum_length:
                continue
            protein = translate_sequence(sequence)
            internal = protein if edited_stop else protein[:-1]
            if "*" in internal:
                continue
            score = (
                abs(start_coordinate - current_start) + abs(stop_coordinate - current_stop),
                int(edited_stop),
                abs(len(sequence) - current_length),
            )
            choices.append((score, candidate, edited_stop))

    if not choices:
        return ann
    _score, repaired, edited_stop = min(choices, key=lambda item: item[0])
    sequence = _extract_cds_sequence(repaired, genome)
    edited_start = sequence[:3].upper() == "ACG"
    if edited_start or edited_stop:
        exceptions = tuple(dict.fromkeys((*ann.exceptions, "RNA editing")))
        notes = list(ann.notes)
        if edited_start:
            notes.append("RNA editing: ACG->AUG start codon")
        if edited_stop:
            notes.append(f"RNA editing: {sequence[-3:].upper()}->stop (C-to-U)")
        repaired = repaired.model_copy(update={"exceptions": list(exceptions), "notes": notes})
    if repaired.exons != ann.exons:
        notes = [*repaired.notes, "bounded terminal codon repair"]
        repaired = repaired.model_copy(update={"notes": notes})
    return repaired


def _refine_boundary_by_tblastn(
    ann: GeneAnnotation,
    genome: GenomeSequence,
    db_manager: DBManager,
    *,
    tool_paths: Mapping[str, str] | None = None,
    command_runner: CommandRunner | None = None,
) -> GeneAnnotation:
    """Adaptive boundary refinement using tblastn against reference protein.

    Runs a local tblastn search using the gene's Protein.fasta reference
    against the genome. If a high-quality hit overlaps the current
    annotation substantially, the tblastn boundaries are adopted.

    Skips trans-spliced/multi-exon genes where tblastn aligns the full
    protein and contiguous hits are not meaningful.
    """
    gene_name_lower = ann.gene_name.lower()

    # Skip trans-spliced genes and any gene with more than one exon
    if gene_name_lower in {k.lower() for k in TRANS_SPLICED_CONFIG} or len(ann.exons) > 1:
        return ann

    # Skip genes with unreliable tblastn references (truncated reference proteins
    # cause false shortening of the annotation)
    _TBLASTN_SHORTEN_ONLY_GENES = {"matr"}
    _skip_shorten = gene_name_lower in _TBLASTN_SHORTEN_ONLY_GENES

    losat = losat_from_tool_paths(tool_paths)
    tblastn = tool_paths.get("tblastn") if tool_paths is not None else shutil.which("tblastn")
    makeblastdb = (
        tool_paths.get("makeblastdb") if tool_paths is not None else shutil.which("makeblastdb")
    )
    if losat is None and (not tblastn or not makeblastdb):
        raise OrganelleDependencyError(
            code="dependency_missing",
            message="Install LOSAT or NCBI BLAST+ (tblastn and makeblastdb) for boundary refinement",
            details={
                "missing": [
                    name
                    for name, path in (("tblastn", tblastn), ("makeblastdb", makeblastdb))
                    if not path
                ]
            },
        )

    ref_dir = db_manager.blast_ref_dir
    ref_file = ref_dir / f"{ann.gene_name}.Protein.fasta"
    if not ref_file.exists():
        return ann

    with tempfile.TemporaryDirectory() as tmpdir:
        genome_fa = Path(tmpdir) / "genome.fasta"
        genome_fa.write_text(f">{genome.seqid}\n{genome.sequence}\n")

        if losat is not None:
            raw_rows = run_losat_with_query_coverage(
                "tblastn",
                ref_file,
                genome_fa,
                evalue=1e-10,
                max_target_seqs=5,
                executable=losat,
                command_runner=command_runner,
                timeout=120,
                stage=f"boundary_refinement:{ann.gene_name}:tblastn",
            )
            rows = [
                [r[0], r[1], r[8], r[9], r[10], r[11], r[2], r[12]]
                for r in raw_rows
                if len(r) >= 13
            ]
        else:
            db_path = Path(tmpdir) / "genome_db"
            make_db_cmd = (
                makeblastdb,
                "-in",
                str(genome_fa),
                "-dbtype",
                "nucl",
                "-out",
                str(db_path),
            )
            if command_runner is not None:
                command_runner.run(
                    make_db_cmd,
                    stage=f"boundary_refinement:{ann.gene_name}:makeblastdb",
                    timeout=60,
                )
            else:
                try:
                    run_external(make_db_cmd, timeout=60)
                except OrganelleExecutionError as e:
                    logger.warning(f"makeblastdb failed for {ann.gene_name}: {e.message}")
                    return ann

            out_file = Path(tmpdir) / f"tblastn_{ann.gene_name}.tsv"
            cmd = [
                tblastn,
                "-query",
                str(ref_file),
                "-db",
                str(db_path),
                "-out",
                str(out_file),
                "-outfmt",
                "6 qseqid sseqid sstart send evalue bitscore pident qcovs",
                "-evalue",
                "1e-10",
                "-max_target_seqs",
                "5",
            ]

            if command_runner is not None:
                command_runner.run(
                    tuple(cmd),
                    stage=f"boundary_refinement:{ann.gene_name}:tblastn",
                    timeout=120,
                )
            else:
                try:
                    run_external(cmd, timeout=120)
                except OrganelleExecutionError as e:
                    logger.warning(f"tblastn failed for {ann.gene_name}: {e.message}")
                    return ann

            rows = (
                [line.split("\t") for line in out_file.read_text().splitlines() if line.strip()]
                if out_file.exists()
                else []
            )

        hmm_start = ann.genomic_start
        hmm_end = ann.genomic_end
        hmm_len = hmm_end - hmm_start + 1

        best_hit = None
        best_score = -1.0

        for parts in rows:
            if len(parts) < 8:
                continue
            try:
                sstart = int(parts[2])
                send = int(parts[3])
                bitscore = float(parts[5])
                pident = float(parts[6])
                qcovs = float(parts[7]) if len(parts) > 7 else 0.0
            except (ValueError, IndexError):
                continue

            if pident < 80 or qcovs < 70:
                continue

            hit_start = min(sstart, send)
            hit_end = max(sstart, send)
            hit_len = hit_end - hit_start + 1

            overlap_start = max(hmm_start, hit_start)
            overlap_end = min(hmm_end, hit_end)
            overlap_len = max(0, overlap_end - overlap_start + 1)

            if hmm_len > 0 and overlap_len / hmm_len < 0.8:
                continue

            # Reject hits that would shrink the annotation by >10%.
            # A shorter tblastn hit likely comes from a truncated reference
            # protein and should not override a more accurate boundary.
            if hit_len < hmm_len * 0.9:
                continue

            # Prefer hits closest to the HMM region
            proximity = min(abs(hit_start - hmm_start), abs(hit_end - hmm_end))
            score = bitscore + (100 - proximity) + pident

            if score > best_score:
                best_score = score
                best_hit = (hit_start, hit_end)

    if best_hit:
        new_start, new_end = best_hit
        # For genes with unreliable tblastn references, only accept extensions
        # (not shortenings) to avoid false truncation from truncated reference proteins
        if _skip_shorten and new_end - new_start + 1 < hmm_len * 0.98:
            logger.info(
                f"tblastn: skipping shortening for {ann.gene_name} "
                f"({new_end - new_start + 1} < {hmm_len})"
            )
            return ann
        logger.info(
            f"tblastn refined {ann.gene_name}: {hmm_start}-{hmm_end} -> {new_start}-{new_end}"
        )
        new_exons = [
            ExonRecord(
                start=new_start,
                end=new_end,
                strand=ann.strand,
                number=1,
                phase=0,
            )
        ]
        notes = list(ann.notes)
        notes.append(f"tblastn boundary refinement: {hmm_start}-{hmm_end} -> {new_start}-{new_end}")
        return ann.model_copy(
            update={
                "exons": new_exons,
                "notes": notes,
                "source_method": "tblastn",
            }
        )

    return ann


def _apply_fixed_offset_correction(ann: GeneAnnotation, genome: GenomeSequence) -> GeneAnnotation:
    """Apply fixed offset correction for genes with systematic position errors.

    Some genes have consistent position offsets across multiple species,
    indicating systematic issues in the annotation pipeline. This function
    applies known corrections.

    Args:
        ann: Gene annotation
        genome: Genome sequence

    Returns:
        Corrected annotation (if gene is in FIXED_OFFSET_GENES)
    """
    if not tuned("offsets"):
        return ann

    gene_name_lower = ann.gene_name.lower()

    if gene_name_lower not in FIXED_OFFSET_GENES:
        return ann

    # Guard: skip annotations already resolved by trans-splicing
    # (BLAST or HMM-merged multi-exon — offsets would destroy accuracy)
    if ann.source_method in ("BLAST", "HMM-merged"):
        return ann

    if len(ann.exons) >= 2:
        return ann

    offset_config = FIXED_OFFSET_GENES[gene_name_lower]
    start_offset = offset_config["start_offset"]
    reason = offset_config["reason"]

    if not ann.exons:
        return ann

    # Find the transcription start exon
    if ann.strand == Strand.PLUS:
        # Plus strand: first exon has lowest coordinates
        start_exon_idx = 0
        old_start = ann.exons[0].start
        new_start = old_start + start_offset
        new_start = max(1, new_start)

        ann.exons[start_exon_idx].start = new_start
        logger.info(
            f"Fixed offset for {ann.gene_name}: "
            f"start {old_start} -> {new_start} "
            f"(offset={start_offset}, reason={reason})"
        )
    else:
        # Minus strand: transcription start is the exon with highest coordinates
        # Find exon with highest start (first in transcription order)
        start_exon_idx = max(range(len(ann.exons)), key=lambda i: ann.exons[i].start)
        old_start = ann.exons[start_exon_idx].end
        new_start = old_start - start_offset
        new_start = min(len(genome.sequence), new_start)

        ann.exons[start_exon_idx].end = new_start
        logger.info(
            f"Fixed offset for {ann.gene_name}: "
            f"end {old_start} -> {new_start} "
            f"(offset={start_offset}, reason={reason})"
        )

    return ann


def _get_gene_search_range(gene_name: str, default_range: int) -> int:
    """Get appropriate search range for a gene.

    Some genes are known to have issues with boundary prediction
    and need more conservative search ranges.
    """
    if not tuned("offsets"):
        return default_range

    # Genes that are prone to over-extension
    CONSERVATIVE_GENES = {
        "atp1",
        "atp6",
        "atp8",
        "atp9",
        "cox1",
        "cox2",
        "cox3",
        "cob",
        "ccmB",
        "ccmC",
        "nad6",
        "matR",
        "mttB",
    }

    # Genes that need WIDER search because they use non-standard start codons
    # and the real ATG is often further upstream than the HMM-predicted boundary
    WIDE_SEARCH_GENES = {
        "mttB": 75,  # ATA/GTG used as start, but real ATG can be 51-63bp upstream
    }

    if gene_name in WIDE_SEARCH_GENES:
        return WIDE_SEARCH_GENES[gene_name]

    if gene_name in CONSERVATIVE_GENES:
        return min(default_range, 100)  # Very conservative

    return default_range


def _validate_gene_length(ann: GeneAnnotation, db_manager: DBManager) -> GeneAnnotation:
    """Validate gene length against known reference lengths.

    If gene is abnormally long or short, add a warning and potentially trim.
    """
    if not tuned("length_gates"):
        return ann

    # Known approximate lengths for core genes (in bp)
    EXPECTED_LENGTHS = {
        "atp1": (1400, 1600),
        "atp4": (500, 650),
        "atp6": (900, 1200),
        "atp8": (400, 550),
        "atp9": (200, 280),
        "ccmB": (550, 700),
        "ccmC": (700, 850),
        "ccmFC": (2000, 2500),
        "ccmFN1": (1000, 1300),
        "ccmFN2": (550, 700),
        "cob": (1100, 1300),
        "cox1": (1500, 1650),
        "cox2": (700, 900),
        "cox3": (750, 900),
        "matR": (1800, 2100),
        "mttB": (750, 900),
        "nad1": (900, 1100),  # Total exon length, not span
        "nad2": (1100, 1400),
        "nad3": (300, 400),
        "nad4": (1300, 1600),
        "nad4L": (250, 350),
        "nad5": (1800, 2300),
        "nad6": (550, 700),
        "nad7": (1100, 1400),
        "nad9": (500, 650),
        "rpl2": (900, 1200),
        "rpl5": (500, 650),
        "rpl10": (450, 600),
        "rpl16": (480, 580),
        "rps1": (900, 1200),
        "rps3": (1100, 1500),
        "rps4": (950, 1150),
        "rps7": (400, 550),
        "rps10": (250, 350),
        "rps12": (350, 450),
        "rps14": (280, 380),
        "rps19": (250, 350),
    }

    if ann.gene_name not in EXPECTED_LENGTHS:
        return ann

    min_len, max_len = EXPECTED_LENGTHS[ann.gene_name]
    current_len = ann.total_exon_length

    # If gene is abnormally long, it may have been over-extended
    if current_len > max_len * 1.5:
        logger.warning(
            f"{ann.gene_name}: Abnormal length {current_len}bp "
            f"(expected {min_len}-{max_len}bp). May be over-extended."
        )
        # Don't auto-trim, just flag for now
        notes = [*ann.notes, f"Warning: length {current_len}bp exceeds expected range"]
        return ann.model_copy(update={"notes": notes})

    # If gene is abnormally short, it may be fragmented
    if current_len < min_len * 0.5:
        logger.warning(
            f"{ann.gene_name}: Abnormal length {current_len}bp "
            f"(expected {min_len}-{max_len}bp). May be fragmented."
        )
        notes = [*ann.notes, f"Warning: length {current_len}bp below expected range"]
        return ann.model_copy(update={"notes": notes})

    return ann


def _remove_short_introns(ann: GeneAnnotation, genome: GenomeSequence) -> GeneAnnotation:
    """Merge exons of a cis gene separated by gaps too short to be introns.

    Plant mitochondrial introns are hundreds of bp long, so a gap below
    ``SHORT_INTRON_THRESHOLD`` -- including abutting or overlapping exon hits --
    means the reference exon split is absent from this genome (e.g. cox1
    without its group I intron). Exons are compared in genomic order, so
    minus-strand genes (stored in transcription order, high to low) merge too.
    """
    if len(ann.exons) <= 1 or _has_trans_spliced_layout(ann):
        return ann
    if len({exon.strand for exon in ann.exons}) > 1:
        return ann

    by_position = sorted(ann.exons, key=lambda exon: exon.start)
    merged: list[tuple[int, int]] = [(by_position[0].start, by_position[0].end)]
    for exon in by_position[1:]:
        prev_start, prev_end = merged[-1]
        if exon.start - prev_end - 1 < SHORT_INTRON_THRESHOLD:
            merged[-1] = (prev_start, max(prev_end, exon.end))
        else:
            merged.append((exon.start, exon.end))
    if len(merged) == len(ann.exons):
        return ann

    if ann.strand == Strand.MINUS:
        merged.reverse()
    exons = []
    cumulative_len = 0
    for number, (start, end) in enumerate(merged, 1):
        exons.append(
            ExonRecord(
                start=start,
                end=end,
                strand=ann.exons[0].strand,
                number=number,
                phase=cumulative_len % 3,
            )
        )
        cumulative_len += end - start + 1
    logger.debug(f"  {ann.gene_name}: merged {len(ann.exons)} exons -> {len(exons)}")
    notes = [
        *ann.notes,
        f"merged {len(ann.exons)} exon hits separated by <{SHORT_INTRON_THRESHOLD} bp",
    ]
    return ann.model_copy(update={"exons": exons, "notes": notes})


def _correct_start_codon_conservative(
    ann: GeneAnnotation,
    genome: GenomeSequence,
    db_manager: DBManager,
    search_range: int,
) -> GeneAnnotation:
    """Conservative start codon correction - only adjust if very close to HMM boundary.

    Strategy: Trust the HMM hit boundaries more, only make small adjustments
    if a clear start codon is found very close to the boundary.
    """
    if ann.is_pseudo or len(ann.exons) == 0:
        return ann

    # Define expected lengths locally
    EXPECTED_LENGTHS = {
        "atp1": (1400, 1600),
        "atp4": (500, 650),
        "atp6": (900, 1200),
        "atp8": (400, 550),
        "atp9": (200, 280),
        "ccmB": (550, 700),
        "ccmC": (700, 850),
        "ccmFC": (2000, 2500),
        "ccmFN1": (1000, 1300),
        "ccmFN2": (550, 700),
        "cob": (1100, 1300),
        "cox1": (1500, 1650),
        "cox2": (700, 900),
        "cox3": (750, 900),
        "matR": (1800, 2100),
        "mttB": (750, 900),
        "nad1": (900, 1100),
        "nad2": (1100, 1400),
        "nad3": (300, 400),
        "nad4": (1300, 1600),
        "nad4L": (250, 350),
        "nad5": (1800, 2300),
        "nad6": (550, 700),
        "nad7": (1100, 1400),
        "nad9": (500, 650),
        "rpl2": (900, 1200),
        "rpl5": (500, 650),
        "rpl10": (450, 600),
        "rpl16": (400, 500),
        "rps3": (1100, 1500),
        "rps4": (950, 1150),
        "rps7": (400, 550),
        "rps12": (350, 450),
        "rps14": (280, 380),
    }
    if not tuned("length_gates"):
        EXPECTED_LENGTHS = {}

    # Don't extend genes that are already reasonable length
    current_len = ann.total_exon_length
    logger.debug(f"Boundary correction check for {ann.gene_name}: len={current_len}")

    # Genes with non-standard start codons that need correction even at reasonable length
    _FORCE_START_CORRECTION = {"mttB"}

    if ann.gene_name in EXPECTED_LENGTHS and ann.gene_name not in _FORCE_START_CORRECTION:
        min_exp, max_exp = EXPECTED_LENGTHS[ann.gene_name]
        if min_exp * 0.8 <= current_len <= max_exp * 1.2:
            # Length is reasonable, skip correction
            logger.debug(f"{ann.gene_name}: length {current_len} is reasonable, skipping")
            return ann

    first_exon = ann.exons[0]
    allowed = _get_allowed_start_codons(ann.gene_name, db_manager)

    if ann.strand == Strand.PLUS:
        # For forward strand, look for start codon within small window
        start = first_exon.start
        search_from = max(1, start - search_range)

        # Get sequence
        seq = genome.get_sequence_for_range(search_from, first_exon.end)
        if len(seq) < 3:
            return ann

        offset_in_seq = start - search_from

        # Scan backward from HMM start position
        best_start = None
        best_non_atg = None
        for i in range(offset_in_seq, -1, -3):
            if i + 2 < len(seq):
                codon = seq[i : i + 3].upper()
                if codon in allowed:
                    genome_start = search_from + i
                    # Only accept if within search_range and upstream
                    if genome_start <= start + 3 and abs(genome_start - start) <= search_range:
                        if codon == "ATG":
                            best_start = genome_start
                            break
                        elif best_non_atg is None:
                            best_non_atg = genome_start
        # Use ATG if found, otherwise fall back to first non-ATG match
        if best_start is None:
            best_start = best_non_atg

        if best_start and best_start != start:
            new_exons = [
                ExonRecord(
                    start=best_start,
                    end=first_exon.end,
                    strand=ann.strand,
                    number=1,
                )
            ]
            new_exons.extend(ann.exons[1:])

            note = ""
            codon_at_start = genome.get_sequence_for_range(best_start, best_start + 2).upper()
            if codon_at_start == "ACG":
                note = "RNA editing: ACG->AUG start codon"
            elif codon_at_start not in START_CODONS:
                note = f"non-standard start codon: {codon_at_start}"

            updates = {"exons": new_exons}
            if note:
                updates["notes"] = [*ann.notes, note]
                updates["exceptions"] = (
                    [*ann.exceptions, "RNA editing"] if "ACG" in note else ann.exceptions
                )
            return ann.model_copy(update=updates)

    else:
        # For reverse strand
        end = ann.exons[-1].end
        search_to = min(genome.length, end + search_range)

        fwd_seq = genome.get_sequence_for_range(ann.exons[-1].start, search_to)
        rc_seq = fwd_seq.translate(str.maketrans("ATGCatgcNn", "TACGtacgNn"))[::-1]

        if len(rc_seq) < 3:
            return ann

        # HMM end position in RC
        hmm_pos_rc = len(fwd_seq) - 1
        best_offset = None
        for i in range(hmm_pos_rc, -1, -3):
            if i + 2 < len(rc_seq):
                codon = rc_seq[i : i + 3].upper()
                if codon in allowed:
                    if i <= hmm_pos_rc + 3:  # Within small range
                        best_offset = i
                    break

        if best_offset:
            new_end = ann.exons[-1].start + (len(fwd_seq) - 1 - best_offset)
            if abs(new_end - end) <= search_range:
                new_exons = list(ann.exons[:-1])
                new_exons.append(
                    ExonRecord(
                        start=ann.exons[-1].start,
                        end=new_end,
                        strand=ann.strand,
                        number=ann.exons[-1].number,
                    )
                )

                codon_at_start = genome.get_sequence_for_range(new_end - 2, new_end)
                codon_at_start = codon_at_start.translate(
                    str.maketrans("ATGCatgcNn", "TACGtacgNn")
                )[::-1].upper()
                note = ""
                if codon_at_start == "ACG":
                    note = "RNA editing: ACG->AUG start codon"

                updates = {"exons": new_exons}
                if note:
                    updates["notes"] = [*ann.notes, note]
                return ann.model_copy(update=updates)

    return ann


def _correct_stop_codon_conservative(
    ann: GeneAnnotation,
    genome: GenomeSequence,
    db_manager: DBManager,
    search_range: int,
) -> GeneAnnotation:
    """Conservative stop codon correction - only adjust if very close to HMM boundary."""
    if ann.is_pseudo or len(ann.exons) == 0:
        return ann

    # Define expected lengths locally (all lowercase to match gene_name)
    EXPECTED_LENGTHS = {
        "atp1": (1400, 1600),
        "atp4": (500, 650),
        "atp6": (900, 1200),
        "atp8": (400, 550),
        "atp9": (200, 280),
        "ccmb": (550, 700),
        "ccmc": (700, 850),
        "ccmfc": (1300, 1450),
        "ccmfn": (1200, 1900),
        "cob": (1100, 1300),
        "cox1": (1500, 1650),
        "cox2": (700, 900),
        "cox3": (750, 900),
        "matr": (1800, 2100),
        "mttb": (700, 900),
        "nad1": (900, 1100),
        "nad2": (1100, 1400),
        "nad3": (300, 400),
        "nad4": (1300, 1600),
        "nad4l": (250, 350),
        "nad5": (1800, 2300),
        "nad6": (550, 700),
        "nad7": (1100, 1400),
        "nad9": (500, 650),
        "rpl2": (900, 1200),
        "rpl5": (500, 650),
        "rpl10": (450, 600),
        "rpl16": (400, 500),
        "rps1": (450, 650),
        "rps2": (600, 680),
        "rps3": (1100, 1500),
        "rps4": (950, 1150),
        "rps7": (400, 550),
        "rps10": (320, 440),
        "rps12": (350, 450),
        "rps13": (340, 360),
        "rps14": (280, 380),
        "rps19": (270, 360),
        "sdh3": (280, 360),
        "sdh4": (340, 480),
    }
    if not tuned("length_gates"):
        EXPECTED_LENGTHS = {}

    # Note: no early return based on current length — rely on acceptance-time
    # length guard to reject bad corrections instead.
    current_len = ann.total_exon_length
    gene_key = ann.gene_name.lower()

    is_stop_gain = db_manager.is_stop_gain_gene(ann.gene_name)
    last_exon = ann.exons[-1]

    if ann.strand == Strand.PLUS:
        end = last_exon.end
        # Circular-aware: fetch last exon plus downstream search_range
        downstream_end = ((end + search_range - 1) % genome.length) + 1
        downstream = genome.subsequence(end + 1, downstream_end)
        seq = genome.get_sequence_for_range(last_exon.start, last_exon.end) + downstream

        if len(seq) < 3:
            return ann

        # Calculate frame offset
        prior_len = sum((e.end - e.start + 1) for e in ann.exons[:-1])
        frame_offset = prior_len % 3

        # Look for stop codon
        # For stop-gain genes, collect all valid stop-gain candidates and pick
        # the one closest to current end (longest gene length), since internal
        # CAA/CAG/CGA codons are common and not all are real editing sites.
        best_stop_gain = None  # (new_end, codon, new_len)
        for i in range(frame_offset, len(seq) - 2, 3):
            codon = seq[i : i + 3].upper()
            if codon in STOP_CODONS:
                new_end_raw = last_exon.start + i + 2
                new_end = ((new_end_raw - 1) % genome.length) + 1
                end_diff = genome.circular_span(end, new_end)
                is_downstream = end_diff <= search_range + 3
                if len(ann.exons) == 1:
                    new_len = new_end - last_exon.start + 1
                    if gene_key in EXPECTED_LENGTHS:
                        min_exp, _ = EXPECTED_LENGTHS[gene_key]
                        is_upstream = (new_end <= end) and (new_len >= min_exp)
                    else:
                        is_upstream = (new_end <= end) and (end - new_end <= 3)
                else:
                    is_upstream = (new_end <= end) and (end - new_end <= 3)
                if is_downstream or is_upstream:
                    # Before accepting this standard stop, check if a
                    # stop-gain (RNA editing) candidate upstream is closer
                    # to the expected length — it may be the real terminus.
                    if best_stop_gain is not None:
                        sg_end, sg_codon, sg_len = best_stop_gain
                        if sg_end < new_end and gene_key in EXPECTED_LENGTHS:
                            min_exp, max_exp = EXPECTED_LENGTHS[gene_key]
                            # Prefer stop-gain if it brings length closer to max_exp
                            cur_dev = abs(new_len - max_exp)
                            sg_dev = abs(sg_len - max_exp)
                            if sg_dev < cur_dev:
                                notes = [*ann.notes, f"RNA editing: {sg_codon}->stop (C-to-U)"]
                                exceptions = list(set([*ann.exceptions, "RNA editing"]))
                                new_exons = list(ann.exons[:-1])
                                new_exons.append(
                                    ExonRecord(
                                        start=last_exon.start,
                                        end=sg_end,
                                        strand=ann.strand,
                                        number=last_exon.number,
                                    )
                                )
                                return ann.model_copy(
                                    update={
                                        "exons": new_exons,
                                        "notes": notes,
                                        "exceptions": exceptions,
                                    }
                                )
                    new_exons = list(ann.exons[:-1])
                    new_exons.append(
                        ExonRecord(
                            start=last_exon.start,
                            end=new_end,
                            strand=ann.strand,
                            number=last_exon.number,
                        )
                    )
                    return ann.model_copy(update={"exons": new_exons})
                break
            if is_stop_gain and codon in STOP_GAIN_CODONS:
                new_end_raw = last_exon.start + i + 2
                new_end = ((new_end_raw - 1) % genome.length) + 1
                end_diff = genome.circular_span(end, new_end)
                is_downstream = end_diff <= search_range + 3
                if len(ann.exons) == 1:
                    new_len = new_end - last_exon.start + 1
                    if gene_key in EXPECTED_LENGTHS:
                        min_exp, _ = EXPECTED_LENGTHS[gene_key]
                        is_upstream = (new_end <= end) and (new_len >= min_exp * 0.7)
                    else:
                        is_upstream = (new_end <= end) and (end - new_end <= 3)
                else:
                    new_len = 0
                    is_upstream = (new_end <= end) and (end - new_end <= 3)
                if (
                    (is_downstream or is_upstream)
                    and is_upstream
                    and (best_stop_gain is None or new_end > best_stop_gain[0])
                ):
                    best_stop_gain = (new_end, codon, new_len)
                # Don't break — keep scanning for stop-gain codons

        # Apply best stop-gain candidate if found
        if best_stop_gain:
            sg_end, sg_codon, sg_len = best_stop_gain
            if sg_end != end:
                notes = [*ann.notes, f"RNA editing: {sg_codon}->stop (C-to-U)"]
                exceptions = list(set([*ann.exceptions, "RNA editing"]))
                new_exons = list(ann.exons[:-1])
                new_exons.append(
                    ExonRecord(
                        start=last_exon.start,
                        end=sg_end,
                        strand=ann.strand,
                        number=last_exon.number,
                    )
                )
                return ann.model_copy(
                    update={"exons": new_exons, "notes": notes, "exceptions": exceptions}
                )

    else:
        # Reverse strand
        start = ann.exons[0].start
        # For single-exon genes, search the full exon for stop codons
        # to handle over-extension at the 3' end (low-coordinate side)
        if len(ann.exons) == 1:
            search_from = max(1, start - (ann.exons[0].end - ann.exons[0].start))
        else:
            search_from = max(1, start - search_range)
        fwd_seq = genome.get_sequence_for_range(search_from, start + 2)
        rc_seq = fwd_seq.translate(str.maketrans("ATGCatgcNn", "TACGtacgNn"))[::-1]

        if len(rc_seq) < 3:
            return ann

        # Codons are counted from the CDS start, as in the plus-strand branch:
        # rc_seq[0] is transcript position total-3, so codons begin at
        # i = -total mod 3. Counting from the current 3' end searched the wrong
        # frame whenever that end was off by a non-multiple of 3 (BLAST put rps3
        # 4 nt short in 10 PMGA Table S2 species).
        frame = -ann.total_exon_length % 3
        for i in range(frame, len(rc_seq) - 2, 3):
            codon = rc_seq[i : i + 3].upper()
            is_stop = codon in STOP_CODONS
            is_edited_stop = is_stop_gain and codon in STOP_GAIN_CODONS
            if is_stop or is_edited_stop:
                new_start = start - i
                if len(ann.exons) == 1:
                    new_len = ann.exons[0].end - new_start + 1
                    if gene_key in EXPECTED_LENGTHS:
                        min_exp, max_exp_local = EXPECTED_LENGTHS[gene_key]
                        # For stop-gain RNA editing, use relaxed min threshold
                        eff_min = min_exp * 0.7 if is_edited_stop else min_exp
                        accept = new_start >= 1 and (eff_min <= new_len <= max_exp_local)
                    else:
                        accept = new_start >= 1 and (new_len <= current_len + 50)
                else:
                    accept = abs(new_start - start) <= search_range and new_start <= start + 3
                if accept:
                    notes = list(ann.notes)
                    exceptions = list(ann.exceptions)
                    if is_edited_stop:
                        notes.append(f"RNA editing: {codon}->stop (C-to-U)")
                        exceptions = list(set([*exceptions, "RNA editing"]))
                    new_exons = [
                        ExonRecord(
                            start=new_start,
                            end=ann.exons[0].end,
                            strand=ann.strand,
                            number=1,
                        )
                    ]
                    new_exons.extend(ann.exons[1:])
                    updates = {"exons": new_exons}
                    if notes != ann.notes:
                        updates["notes"] = notes
                    if exceptions != list(ann.exceptions):
                        updates["exceptions"] = exceptions
                    return ann.model_copy(update=updates)
                if is_stop:
                    break

    return ann


def _correct_start_codon(
    ann: GeneAnnotation,
    genome: GenomeSequence,
    db_manager: DBManager,
    search_range: int,
) -> GeneAnnotation:
    """Find the best start codon for each gene.

    For RNA editing genes (cox1, nad1, nad4L, rps10):
    ACG is accepted as a start codon (edited to AUG).

    For mttB: ATA is accepted.
    For rpl16: GTG is accepted.
    """
    if ann.is_pseudo or len(ann.exons) == 0:
        return ann

    allowed = _get_allowed_start_codons(ann.gene_name, db_manager)

    if ann.strand == Strand.PLUS:
        return _find_start_forward(ann, genome, allowed, search_range)
    else:
        return _find_start_reverse(ann, genome, allowed, search_range)


def _get_allowed_start_codons(gene_name: str, db_manager: DBManager) -> set[str]:
    """Get allowed start codons for a gene."""
    return allowed_start_codons(
        gene_name,
        allow_rna_editing=db_manager.is_start_gain_gene(gene_name),
    )


def _find_start_forward(
    ann: GeneAnnotation,
    genome: GenomeSequence,
    allowed: set[str],
    search_range: int,
) -> GeneAnnotation:
    """Find start codon on forward strand with correct reading frame."""
    start = ann.exons[0].start
    search_from = max(1, start - search_range)
    # Limit search to not extend beyond reasonable range
    max_search_end = min(genome.length, start + 50)  # Don't extend too far downstream

    seq = genome.get_sequence_for_range(search_from, max_search_end)
    if len(seq) < 3:
        return ann

    # Calculate the correct reading frame offset
    # The HMM hit start should be at position (start - search_from) in seq
    # We need to find which offset (0, 1, or 2) puts it in the correct frame
    # Search for start codon upstream of the HMM hit, in the correct frame
    best_start = None
    # Start from the HMM hit position and scan backwards in frame
    hmm_pos = start - search_from  # 0-based position of HMM start in seq

    # Scan from HMM start backwards to search_from, stepping by 3
    for i in range(hmm_pos, -1, -3):
        if i + 2 < len(seq):
            codon = seq[i : i + 3].upper()
            if codon in allowed:
                genome_start = search_from + i
                # Only accept if it's upstream or close to original start
                if genome_start <= start + 3:  # Allow very small downstream adjustment
                    best_start = genome_start
                    break

    # Also scan a short distance downstream for alternative start
    if best_start is None:
        for i in range(hmm_pos, min(hmm_pos + 9, len(seq) - 2), 3):
            codon = seq[i : i + 3].upper()
            if codon in allowed:
                genome_start = search_from + i
                best_start = genome_start
                break

    if best_start is not None and abs(best_start - start) <= search_range:
        new_exons = [
            ExonRecord(
                start=best_start,
                end=ann.exons[0].end,
                strand=ann.strand,
                number=1,
            )
        ]
        new_exons.extend(ann.exons[1:])

        note = ""
        codon_at_start = genome.get_sequence_for_range(best_start, best_start + 2).upper()
        if codon_at_start == "ACG":
            note = "RNA editing: ACG->AUG start codon"
        elif codon_at_start not in START_CODONS:
            note = f"non-standard start codon: {codon_at_start}"

        updates = {"exons": new_exons}
        if note:
            updates["notes"] = [*ann.notes, note]
            updates["exceptions"] = (
                [*ann.exceptions, "RNA editing"] if "ACG" in note else ann.exceptions
            )
        return ann.model_copy(update=updates)

    return ann


def _find_start_reverse(
    ann: GeneAnnotation,
    genome: GenomeSequence,
    allowed: set[str],
    search_range: int,
) -> GeneAnnotation:
    """Find start codon on reverse strand with correct reading frame.

    On reverse strand, 'start' means the high-coordinate end in genome
    (which is the 5' end of the transcribed gene on the reverse strand).
    """
    end = ann.exons[-1].end  # This is the 5' end on reverse strand
    start = ann.exons[0].start
    search_to = min(genome.length, end + search_range)

    # Limit downstream search
    search_to = min(search_to, end + 50)

    # Get forward strand sequence covering the HMM hit and downstream region
    fwd_seq = genome.get_sequence_for_range(start, search_to)
    rc_seq = fwd_seq.translate(str.maketrans("ATGCatgcNn", "TACGtacgNn"))[::-1]

    if len(rc_seq) < 3:
        return ann

    # Calculate correct reading frame
    # HMM end is at position (end - start) in fwd_seq
    # In RC, this is at position len(fwd_seq) - 1 - (end - start)
    hmm_pos_in_rc = len(fwd_seq) - 1 - (end - start)
    hmm_offset = hmm_pos_in_rc % 3

    # Adjust hmm_pos_in_rc to be at the start of its codon
    hmm_pos_in_rc -= hmm_offset

    # Scan from HMM position backwards (towards higher genome coords = 5' end)
    best_offset = None
    for i in range(hmm_pos_in_rc, -1, -3):
        if i + 2 < len(rc_seq):
            codon = rc_seq[i : i + 3].upper()
            if codon in allowed:
                # Convert RC position back to genome coordinate
                # RC offset i corresponds to genome position start + (len(fwd_seq) - 1 - i)
                genome_pos = start + (len(fwd_seq) - 1 - i)
                if genome_pos >= end - 3:  # Allow small upstream adjustment
                    best_offset = i
                    break

    # If not found, scan a short distance downstream (towards lower genome coords)
    if best_offset is None:
        for i in range(hmm_pos_in_rc, min(hmm_pos_in_rc + 9, len(rc_seq) - 2), 3):
            codon = rc_seq[i : i + 3].upper()
            if codon in allowed:
                best_offset = i
                break

    if best_offset is not None:
        # Convert RC offset back to genome coordinate
        genome_end = start + (len(fwd_seq) - 1 - best_offset)
        if abs(genome_end - end) <= search_range:
            new_exons = list(ann.exons[:-1])
            new_exons.append(
                ExonRecord(
                    start=ann.exons[-1].start,
                    end=genome_end,
                    strand=ann.strand,
                    number=ann.exons[-1].number,
                )
            )
            # Read the codon at the new start (reverse complement it)
            codon_at_start = genome.get_sequence_for_range(genome_end - 2, genome_end)
            codon_at_start = codon_at_start.translate(str.maketrans("ATGCatgcNn", "TACGtacgNn"))[
                ::-1
            ].upper()
            note = ""
            if codon_at_start == "ACG":
                note = "RNA editing: ACG->AUG start codon"

            updates = {"exons": new_exons}
            if note:
                updates["notes"] = [*ann.notes, note]
            return ann.model_copy(update=updates)

    return ann


def _correct_stop_codon(
    ann: GeneAnnotation,
    genome: GenomeSequence,
    db_manager: DBManager,
    search_range: int,
) -> GeneAnnotation:
    """Find the stop codon for each gene.

    For stop-gain genes (ccmFC, rps10, atp9, atp6, rps11):
    CAA/CAG/CGA codons are accepted as "premature" stops that
    will be edited (C->U) to create proper stop codons.
    """
    if ann.is_pseudo or len(ann.exons) == 0:
        return ann

    is_stop_gain = db_manager.is_stop_gain_gene(ann.gene_name)
    last_exon = ann.exons[-1]

    if ann.strand == Strand.PLUS:
        end = last_exon.end
        search_to = min(genome.length, end + search_range)
        seq = genome.get_sequence_for_range(last_exon.start, search_to)

        # Determine reading frame offset within last exon
        # Total coding length before this exon determines the frame
        prior_coding_len = sum((e.end - e.start + 1) for e in ann.exons[:-1])
        frame_offset = prior_coding_len % 3  # 0, 1, or 2

        # Scan for stop codon in frame, starting at the correct offset
        for i in range(frame_offset, len(seq) - 2, 3):
            codon = seq[i : i + 3].upper()
            if codon in STOP_CODONS:
                new_end = last_exon.start + i + 2  # Include stop codon
                if new_end != end:
                    new_exons = list(ann.exons[:-1])
                    new_exons.append(
                        ExonRecord(
                            start=last_exon.start,
                            end=new_end,
                            strand=ann.strand,
                            number=last_exon.number,
                        )
                    )
                    return ann.model_copy(update={"exons": new_exons})
                break
            # Stop-gain: CAA/CAG/CGA treated as edited stop
            if is_stop_gain and codon in STOP_GAIN_CODONS:
                new_end = last_exon.start + i + 2
                notes = [*ann.notes, f"RNA editing: {codon}->stop (C-to-U)"]
                exceptions = list(set([*ann.exceptions, "RNA editing"]))
                new_exons = list(ann.exons[:-1])
                new_exons.append(
                    ExonRecord(
                        start=last_exon.start,
                        end=new_end,
                        strand=ann.strand,
                        number=last_exon.number,
                    )
                )
                return ann.model_copy(
                    update={
                        "exons": new_exons,
                        "notes": notes,
                        "exceptions": exceptions,
                    }
                )
    else:
        # Reverse strand: stop codon is at the low-coordinate end
        # On reverse strand, the gene reads from high coord to low coord.
        # The stop is near the first exon's start (low coord).
        # We scan from start outward (lower coords) on the reverse complement.
        start = ann.exons[0].start
        search_from = max(1, start - search_range)

        # Get forward strand sequence, then reverse complement
        fwd_seq = genome.get_sequence_for_range(search_from, start + 2)
        rc_seq = fwd_seq.translate(str.maketrans("ATGCatgcNn", "TACGtacgNn"))[::-1]

        # In RC coords: offset 0 = genome position start+2 (5' of gene on RC)
        # Scan toward 3' (which is decreasing genome coords)
        for i in range(0, len(rc_seq) - 2, 3):
            codon = rc_seq[i : i + 3].upper()
            is_stop = codon in STOP_CODONS
            is_edited_stop = is_stop_gain and codon in STOP_GAIN_CODONS
            if is_stop or is_edited_stop:
                # RC offset i corresponds to genome end: start+2-i
                # The codon covers genome [start+2-i-2, start+2-i]
                new_start = start - i
                if new_start != start:
                    notes = list(ann.notes)
                    exceptions = list(ann.exceptions)
                    if is_edited_stop:
                        notes.append(f"RNA editing: {codon}->stop (C-to-U)")
                        exceptions = list(set([*exceptions, "RNA editing"]))
                    new_exons = [
                        ExonRecord(
                            start=new_start,
                            end=ann.exons[0].end,
                            strand=ann.strand,
                            number=1,
                        )
                    ]
                    new_exons.extend(ann.exons[1:])
                    updates = {"exons": new_exons}
                    if notes != ann.notes:
                        updates["notes"] = notes
                    if exceptions != list(ann.exceptions):
                        updates["exceptions"] = exceptions
                    return ann.model_copy(update=updates)
                break

    return ann


def _handle_special_genes(
    ann: GeneAnnotation,
    genome: GenomeSequence,
    db_manager: DBManager,
) -> GeneAnnotation:
    """Handle gene-specific quirks."""
    name = ann.gene_name

    # rpl16: truncate if first exon is very long AND no valid start codon
    # on the coding strand.  Must be strand-aware: for minus-strand genes
    # the start codon sits at the 3' end (in genome coords).
    # NOTE: even if a start codon exists at the current boundary, if the
    # gene is over-extended (>480bp), fall through to _LENGTH_LIMITED below
    # which will trim it to the expected range.
    if name == "rpl16" and len(ann.exons) >= 1:
        first_len = ann.exons[0].end - ann.exons[0].start + 1
        if first_len > 330:  # >110 aa
            has_start = False
            if ann.strand == Strand.PLUS:
                codon = genome.get_sequence_for_range(
                    ann.exons[0].start, ann.exons[0].start + 2
                ).upper()
            else:
                raw = genome.get_sequence_for_range(ann.exons[0].end - 2, ann.exons[0].end).upper()
                codon = raw.translate(_RC_TABLE)[::-1]
            if codon in {"ATG", "GTG"}:
                has_start = True

            if has_start and first_len <= 480:
                logger.debug(f"rpl16: valid start codon {codon} and length OK ({first_len}bp)")
                return ann

            # If over-extended with a start codon, fall through to _LENGTH_LIMITED
            if has_start and first_len > 480:
                logger.debug(f"rpl16: has start codon but over-extended ({first_len}bp > 480bp)")
                pass  # fall through to _LENGTH_LIMITED below
            elif not has_start:
                # Scan for nearest ATG/GTG in-frame on the coding strand
                if ann.strand == Strand.PLUS:
                    for offset in range(0, min(150, first_len), 3):
                        pos = ann.exons[0].start + offset
                        c = genome.get_sequence_for_range(pos, pos + 2).upper()
                        if c in {"ATG", "GTG"}:
                            if offset == 0:
                                return ann
                            new_exons = [
                                ExonRecord(
                                    start=pos,
                                    end=ann.exons[0].end,
                                    strand=ann.strand,
                                    number=1,
                                )
                            ]
                            new_exons.extend(ann.exons[1:])
                            logger.info(f"rpl16: trimmed {offset}bp to reach start codon")
                            return ann.model_copy(
                                update={
                                    "exons": new_exons,
                                    "notes": [
                                        *ann.notes,
                                        f"rpl16: trimmed {offset}bp to start codon",
                                    ],
                                }
                            )
                else:
                    for offset in range(0, min(150, first_len), 3):
                        pos = ann.exons[0].end - offset
                        raw = genome.get_sequence_for_range(pos - 2, pos).upper()
                        c = raw.translate(_RC_TABLE)[::-1]
                        if c in {"ATG", "GTG"}:
                            if offset == 0:
                                return ann
                            new_exons = [
                                ExonRecord(
                                    start=ann.exons[0].start,
                                    end=pos,
                                    strand=ann.strand,
                                    number=1,
                                )
                            ]
                            new_exons.extend(ann.exons[1:])
                            logger.info(
                                f"rpl16: trimmed {offset}bp to reach start codon (minus strand)"
                            )
                            return ann.model_copy(
                                update={
                                    "exons": new_exons,
                                    "notes": [
                                        *ann.notes,
                                        f"rpl16: trimmed {offset}bp to start codon",
                                    ],
                                }
                            )

    # Genes with known N-terminal over-extension from reference proteins.
    # If the gene is longer than the expected maximum, scan forward for an
    # in-frame ATG/GTG that brings the length into the expected range.
    _LENGTH_LIMITED = {
        "rps14": (280, 310),  # NCBI typically 303bp, reference gives ~360bp
        "rps19": (240, 280),  # NCBI typically ~264bp
        "nad6": (600, 650),  # NCBI typically 618bp, reference gives ~669bp
        "rps7": (420, 480),  # NCBI typically 447bp, reference gives ~516bp
        "rps13": (300, 370),  # NCBI typically 351bp, reference gives ~423bp
        "nad9": (520, 600),  # NCBI typically 573bp, reference gives ~660bp
        "rpl5": (480, 590),  # NCBI typically 549-564bp, reference gives ~636bp
        "rpl10": (470, 530),  # NCBI typically 481-532bp, reference can give ~598bp
        "rpl16": (380, 480),  # NCBI typically 434bp, HMM over-extends to ~557bp
        "atp6": (670, 820),  # NCBI typically 719-816bp, HMM over-extends to ~840-890bp
    }
    if name in _LENGTH_LIMITED and len(ann.exons) >= 1:
        min_len, max_len = _LENGTH_LIMITED[name]
        first_len = ann.exons[0].end - ann.exons[0].start + 1
        if first_len > max_len:
            allowed = _get_allowed_start_codons(name, db_manager)
            if ann.strand == Strand.PLUS:
                for offset in range(3, min(150, first_len - min_len), 3):
                    pos = ann.exons[0].start + offset
                    new_len = first_len - offset
                    c = genome.get_sequence_for_range(pos, pos + 2).upper()
                    if c in allowed and min_len <= new_len <= max_len:
                        new_exons = [
                            ExonRecord(
                                start=pos,
                                end=ann.exons[0].end,
                                strand=ann.strand,
                                number=1,
                            )
                        ]
                        new_exons.extend(ann.exons[1:])
                        logger.info(f"{name}: trimmed {offset}bp, now {new_len}bp")
                        return ann.model_copy(
                            update={
                                "exons": new_exons,
                                "notes": [*ann.notes, f"{name}: trimmed {offset}bp to {new_len}bp"],
                            }
                        )
            else:
                for offset in range(3, min(150, first_len - min_len), 3):
                    pos = ann.exons[0].end - offset
                    new_len = first_len - offset
                    raw = genome.get_sequence_for_range(pos - 2, pos).upper()
                    c = raw.translate(_RC_TABLE)[::-1]
                    if c in allowed and min_len <= new_len <= max_len:
                        new_exons = [
                            ExonRecord(
                                start=ann.exons[0].start,
                                end=pos,
                                strand=ann.strand,
                                number=1,
                            )
                        ]
                        new_exons.extend(ann.exons[1:])
                        logger.info(f"{name}: trimmed {offset}bp (minus strand), now {new_len}bp")
                        return ann.model_copy(
                            update={
                                "exons": new_exons,
                                "notes": [*ann.notes, f"{name}: trimmed {offset}bp to {new_len}bp"],
                            }
                        )

    # nad7: first exon is systematically over-extended by ~75bp (BLASTn reference
    # includes upstream sequence). NCBI exon1 is consistently 144bp. Scan forward
    # for an in-frame ATG that trims the first exon to a reasonable size.
    if name == "nad7" and len(ann.exons) >= 2:
        first_len = ann.exons[0].end - ann.exons[0].start + 1
        if first_len > 165:
            allowed = _get_allowed_start_codons(name, db_manager)
            if ann.strand == Strand.PLUS:
                for offset in range(3, min(225, first_len - 100), 3):
                    pos = ann.exons[0].start + offset
                    new_len = first_len - offset
                    c = genome.get_sequence_for_range(pos, pos + 2).upper()
                    if c == "ATG" and 100 <= new_len <= 300:
                        new_exons = list(ann.exons)
                        new_exons[0] = ExonRecord(
                            start=pos,
                            end=ann.exons[0].end,
                            strand=ann.strand,
                            number=1,
                        )
                        logger.info(f"nad7: trimmed exon1 {offset}bp, now {new_len}bp")
                        return ann.model_copy(
                            update={
                                "exons": new_exons,
                                "notes": [
                                    *ann.notes,
                                    f"nad7: trimmed exon1 {offset}bp to {new_len}bp",
                                ],
                            }
                        )
            else:
                for offset in range(3, min(225, first_len - 100), 3):
                    pos = ann.exons[0].end - offset
                    new_len = first_len - offset
                    raw = genome.get_sequence_for_range(pos - 2, pos).upper()
                    c = raw.translate(_RC_TABLE)[::-1]
                    if c == "ATG" and 100 <= new_len <= 300:
                        new_exons = list(ann.exons)
                        new_exons[0] = ExonRecord(
                            start=ann.exons[0].start,
                            end=pos,
                            strand=ann.strand,
                            number=1,
                        )
                        logger.info(f"nad7: trimmed exon1 {offset}bp (minus), now {new_len}bp")
                        return ann.model_copy(
                            update={
                                "exons": new_exons,
                                "notes": [
                                    *ann.notes,
                                    f"nad7: trimmed exon1 {offset}bp to {new_len}bp",
                                ],
                            }
                        )

    # nad5: trim over-extended short exon (exon 3).
    # The HMM finds an 82-116bp region where the real exon is only ~22bp.
    # Use the highly conserved 22bp motif to locate the correct boundaries.
    if name == "nad5" and len(ann.exons) >= 3:
        _NAD5_SHORT_EXON_MOTIF = "GATATGATGATTGGTTTAGGTA"
        _NAD5_SHORT_EXON_LEN = 22
        new_exons = list(ann.exons)
        changed = False
        for idx, exon in enumerate(new_exons):
            exon_len = exon.end - exon.start + 1
            if exon_len < 50 or exon_len > 160:
                continue
            # Get exon sequence from genome
            if exon.strand == Strand.PLUS or exon.strand is None:
                seq = genome.get_sequence_for_range(exon.start, exon.end).upper()
            else:
                raw = genome.get_sequence_for_range(exon.start, exon.end).upper()
                seq = raw.translate(_RC_TABLE)[::-1]
            # Search for the conserved 22bp motif (allow 1-2 mismatches)
            best_pos = -1
            best_mm = len(_NAD5_SHORT_EXON_MOTIF)
            for i in range(len(seq) - _NAD5_SHORT_EXON_LEN + 1):
                window = seq[i : i + _NAD5_SHORT_EXON_LEN]
                mm = sum(1 for a, b in zip(window, _NAD5_SHORT_EXON_MOTIF, strict=True) if a != b)
                if mm < best_mm:
                    best_mm = mm
                    best_pos = i
            if best_pos < 0 or best_mm > 2:
                continue
            # Compute new boundaries
            if exon.strand == Strand.PLUS or exon.strand is None:
                new_start = exon.start + best_pos
                new_end = new_start + _NAD5_SHORT_EXON_LEN - 1
            else:
                new_end = exon.end - best_pos
                new_start = new_end - _NAD5_SHORT_EXON_LEN + 1
            old_len = exon.end - exon.start + 1
            trim = old_len - _NAD5_SHORT_EXON_LEN
            if trim < 30:
                continue
            new_exons[idx] = ExonRecord(
                start=new_start,
                end=new_end,
                strand=exon.strand,
                number=exon.number,
            )
            changed = True
            logger.info(
                f"nad5: trimmed exon {idx + 1} from {old_len}bp to "
                f"{_NAD5_SHORT_EXON_LEN}bp ({trim}bp removed, "
                f"{best_mm} mismatches to motif)"
            )
        if changed:
            return ann.model_copy(
                update={
                    "exons": new_exons,
                    "notes": [*ann.notes, "nad5: trimmed over-extended short exon"],
                }
            )

    return ann
