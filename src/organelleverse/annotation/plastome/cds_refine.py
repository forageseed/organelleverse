"""Codon-precise start/stop refinement for plastome CDS features.

Reference transfer (BLAST) finds the right genes but not codon-precise
boundaries — the transferred CDS rarely starts at an in-frame start codon or
ends at the stop codon. This module snaps each single-exon CDS to the nearest
in-frame start codon and extends the 3' end to include the first in-frame stop
codon, which is what standard GenBank CDS coordinates use.

Start codons include the edited/alternative organellar starts (ACG/GTG, edited
to AUG in vivo). Multi-exon (intron/trans-spliced) CDS are left unchanged.
"""

from __future__ import annotations

from Bio.Seq import Seq
from Bio.SeqFeature import CompoundLocation, FeatureLocation, SeqFeature

from .cds_terminals import _refine_multi_forward

_STOP = {"TAA", "TAG", "TGA"}
_START = ("ATG", "ACG", "GTG")  # ACG/GTG are edited (C->U) organellar starts
# Genomic codons that RNA editing (C->U) turns into a stop: CAA->UAA, CGA->UGA, CAG->UAG.
_EDITED_STOP = {"CAA", "CGA", "CAG"}
# A first in-frame stop this far (or 10% of the gene) before the reference-
# anchored end is not the gene's end in a plastome with U-to-C editing.
_EDIT_MARGIN_NT = 90
# Ferns, lycophytes and hornworts edit U to C and so carry genomic stops inside
# genes (52 of 165 RefSeq CDS in Adiantum and Pteridium); seed plants and
# mosses do not, and one or two early stops there are pseudogenes. Read-through
# is enabled only when at least this many genes show an early stop.
_MIN_EDITED_GENES = 5
# U-to-C editing of the first base: UAA/UAG -> CAA/CAG (Gln), UGA -> CGA (Arg).
_EDITED_STOP_AA = {"TAA": "Gln", "TAG": "Gln", "TGA": "Arg"}


def _refine_forward(
    seq: str,
    start: int,
    end: int,
    start_window: int,
    max_len: int,
    *,
    tolerate_edits: bool = False,
) -> tuple[int, int, list[int]]:
    L = len(seq)
    # Best in-frame start codon: the one CLOSEST to the reference-anchored position
    # (which is ~correct), with codon type only as a tiebreak. This keeps an edited
    # ACG/GTG start at the anchor rather than jumping to a nearby ATG.
    best = None
    best_rank = None
    for off in range(-start_window, start_window + 1):
        s = start + off
        if s < 1 or s + 2 > L:
            continue
        codon = seq[s - 1 : s + 2]
        if codon in _START:
            rank = (abs(off), _START.index(codon))
            if best_rank is None or rank < best_rank:
                best_rank, best = rank, s
    s = best if best is not None else start

    # Extend to the first in-frame genomic stop codon (inclusive).
    e = end
    genomic = None
    edited = None
    p = s
    while p + 2 <= L and p < s + max_len:
        codon = seq[p - 1 : p + 2]
        if codon in _STOP:
            genomic = p + 2
            break
        if edited is None and codon in _EDITED_STOP:
            edited = p + 2
        p += 3
    if genomic is not None:
        e = genomic
    elif edited is not None:
        # No genomic stop in-frame within range: fall back to an RNA-edited stop.
        e = edited
    edits: list[int] = []
    margin = max(_EDIT_MARGIN_NT, (end - s) // 10)
    if tolerate_edits and genomic is not None and genomic < end - margin:
        # The first stop sits well inside the reference-anchored gene: in a
        # plastome with U-to-C editing it is an edited site, so read through
        # every stop before the anchored end and stop at the first one near it.
        p = s
        while p + 2 <= L and p < s + max_len:
            if seq[p - 1 : p + 2] in _STOP:
                if p + 2 >= end - margin:
                    return s, p + 2, edits
                edits.append(p)
            p += 3
        edits = []  # no terminal stop near the anchor: keep the genomic stop
    return s, e, edits


def refine_cds_boundaries(
    seq: str,
    start: int,
    end: int,
    strand: int,
    *,
    start_window: int = 15,
    max_len: int = 6000,
) -> tuple[int, int]:
    """Return codon-precise (start, end) 1-based inclusive genome coordinates."""
    ns, ne, _ = refine_cds_boundaries_with_edits(
        seq, start, end, strand, start_window=start_window, max_len=max_len
    )
    return ns, ne


def refine_cds_boundaries_with_edits(
    seq: str,
    start: int,
    end: int,
    strand: int,
    *,
    start_window: int = 15,
    max_len: int = 6000,
    tolerate_edits: bool = False,
) -> tuple[int, int, list[tuple[int, int]]]:
    """Like :func:`refine_cds_boundaries`, plus the genome spans of read-through stops."""
    seq = seq.upper()
    L = len(seq)
    if strand == -1:
        rc = str(Seq(seq).reverse_complement())
        rs, re = L - end + 1, L - start + 1
        ns, ne, edits = _refine_forward(
            rc, rs, re, start_window, max_len, tolerate_edits=tolerate_edits
        )
        return L - ne + 1, L - ns + 1, [(L - q - 1, L - q + 1) for q in edits]
    ns, ne, edits = _refine_forward(
        seq, start, end, start_window, max_len, tolerate_edits=tolerate_edits
    )
    return ns, ne, [(q, q + 2) for q in edits]


def refine_cds_features(
    genome: str,
    features: list[SeqFeature],
    *,
    window: int = 15,
    max_cds_len: int = 8000,
    multi_ext: int = 150,
    ref_proteins: dict[str, list[str]] | None = None,
    junction_flanks: dict | None = None,
    reference_files: set[str] | None = None,
) -> list[SeqFeature]:
    """Refine boundaries of CDS features in place (single- and multi-exon).

    ``max_cds_len`` bounds the 3' stop search for single-exon CDS (must cover the
    largest genes, e.g. ycf2 ~6.8 kb). ``multi_ext`` bounds how far past a
    multi-exon 3' exon to look for the stop.

    ``ref_proteins`` maps gene name to candidate reference proteins. When given,
    a two-exon CDS first has its internal splice boundary corrected against a
    reference protein (see :mod:`splice_refine`) so the reading frame across the
    splice is right — otherwise ``_refine_multi_forward`` reads an off-frame last
    exon and halts at a premature stop.

    ``junction_flanks`` (see :mod:`splice_flanks`) resolves splice placements
    that differ only at an RNA editing site, which protein scores cannot.
    ``reference_files`` (names of the loaded references) calibrates the group
    II domain V-VI 3' splice site prediction (see :mod:`group_ii`).
    """
    genome = genome.upper()
    L = len(genome)
    single = [f for f in features if f.type == "CDS" and len(f.location.parts) == 1]
    read_through = {
        id(f): refine_cds_boundaries_with_edits(
            genome,
            int(f.location.start) + 1,
            int(f.location.end),
            f.location.strand or 1,
            start_window=window,
            max_len=max_cds_len,
            tolerate_edits=True,
        )
        for f in single
    }
    editing_genome = sum(1 for r in read_through.values() if r[2]) >= _MIN_EDITED_GENES
    for feat in features:
        if feat.type != "CDS":
            continue
        strand = feat.location.strand or 1
        parts = sorted((int(p.start) + 1, int(p.end)) for p in feat.location.parts)
        if len(parts) == 1:
            if editing_genome:
                ns, ne, edits = read_through[id(feat)]
            else:
                ns, ne = refine_cds_boundaries(
                    genome,
                    parts[0][0],
                    parts[0][1],
                    strand,
                    start_window=window,
                    max_len=max_cds_len,
                )
                edits = []
            if ne > ns:
                feat.location = FeatureLocation(ns - 1, ne, strand=strand)
                if edits:
                    _annotate_edited_stops(feat, genome, edits, strand)
            continue

        # Multi-exon: refine terminal exons in the reading frame. Skip pathological
        # spans (trans-spliced genes like rps12, or IR-spanning artifacts) where a
        # simple spliced read is meaningless.
        if parts[-1][1] - parts[0][0] > 20000:
            # Trans-spliced: refine only the joins of its cis-spliced blocks
            # (rps12 exons 2-3); the trans joins keep their transfer.
            _refine_cis_blocks(genome, feat, ref_proteins, junction_flanks, reference_files)
            continue

        # Correct internal splice boundaries first (BLAST places them approximately,
        # throwing off the frame and thus the terminal stop / producing internal
        # stops). Two-exon refinement permits the group II AY acceptor as well
        # as AG, with protein and reading-frame constraints. More-exon genes
        # retain the joint protein-guided boundary search.
        gene = feat.qualifiers.get("gene", [""])[0]
        if len(parts) >= 2 and ref_proteins and gene in ref_proteins:
            reading = parts if strand == 1 else list(reversed(parts))
            from .splice_refine import _san, _score

            coding = "".join(
                genome[s - 1 : e]
                if strand == 1
                else str(Seq(genome[s - 1 : e]).reverse_complement())
                for s, e in reading
            )
            query = _san(str(Seq(coding[: len(coding) - len(coding) % 3]).translate(table=11)))
            # Reference files have an arbitrary order. Rank the candidate
            # proteins by their match to the transferred gene before refining
            # its boundaries, so a distant first file cannot dictate its splice.
            ranked_proteins = sorted(
                ref_proteins[gene],
                key=lambda protein: _score(_san(protein.rstrip("*")), query),
                reverse=True,
            )
            if len(parts) == 2:
                from .splice_refine import refine_two_exon_cds

                for prot in ranked_proteins:
                    spliced = refine_two_exon_cds(
                        genome, reading, prot, strand, window=window, terminal_ext=multi_ext,
                        junction_flanks=(junction_flanks or {}).get(gene),
                        structure_offsets=_structure_offsets(gene, len(parts) - 1, reference_files),
                    )
                    if spliced:
                        parts = sorted(spliced)
                        break
            else:
                from .splice_refine import refine_internal_boundaries

                for prot in ranked_proteins:
                    spliced = refine_internal_boundaries(
                        genome, reading, prot, strand, junction_flanks=(junction_flanks or {}).get(gene),
                        structure_offsets=_structure_offsets(gene, len(parts) - 1, reference_files),
                    )
                    if spliced:
                        parts = sorted(spliced)
                        break
        if strand == -1:
            rc = str(Seq(genome).reverse_complement())
            rparts = sorted((L - e + 1, L - s + 1) for s, e in parts)
            refined = _refine_multi_forward(rc, rparts, window, multi_ext)
            back = sorted((L - e + 1, L - s + 1) for s, e in refined)
            # -strand CompoundLocation parts are listed 5'->3' (descending genome pos).
            new_parts = [FeatureLocation(s - 1, e, strand=-1) for s, e in reversed(back)]
        else:
            refined = _refine_multi_forward(genome, parts, window, multi_ext)
            new_parts = [FeatureLocation(s - 1, e, strand=1) for s, e in refined]
        # Phase guard: canonical validation rejects any spliced CDS whose
        # length is not divisible by three. Prefer the refined bounds; fall
        # back to the pre-refine bounds when those were in frame; as a last
        # resort trim the final exon to phase and mark the feature partial.
        spliced_nt = sum(int(q.end) - int(q.start) for q in new_parts)
        if spliced_nt % 3:
            # These are 1-based inclusive bounds after internal splice
            # refinement, before the terminal start/stop search. Preserve
            # that valid model in biological reading order if the terminal
            # search cannot produce an in-frame replacement.
            reading = parts if strand == 1 else list(reversed(parts))
            fallback = [FeatureLocation(s - 1, e, strand=strand) for s, e in reading]
            if sum(int(q.end) - int(q.start) for q in fallback) % 3 == 0:
                new_parts = fallback
        spliced_nt = sum(int(q.end) - int(q.start) for q in new_parts)
        if spliced_nt % 3:
            drop = spliced_nt % 3
            last = new_parts[-1]
            trimmed_start = int(last.start) + drop if strand == -1 else int(last.start)
            trimmed_end = int(last.end) - drop if strand == 1 else int(last.end)
            if trimmed_end - trimmed_start >= 3:
                new_parts = [
                    *list(new_parts[:-1]),
                    FeatureLocation(trimmed_start, trimmed_end, strand=strand),
                ]
                feat.qualifiers.setdefault("partial", ["true"])
            else:
                # Cannot form an in-frame feature from these bounds: leave
                # the location untouched and let validation see the honest
                # original rather than a broken refinement.
                continue
        feat.location = new_parts[0] if len(new_parts) == 1 else CompoundLocation(new_parts)
    return features


_CIS_MAX_INTRON = 10000


def _structure_offsets(gene: str, n_joins: int, reference_files: set[str] | None) -> list | None:
    """(offset, reliable) calibration of each join's domain V-VI prediction, or None."""
    if not reference_files:
        return None
    from .group_ii import reference_calibration

    return [reference_calibration(gene, k, reference_files) for k in range(1, n_joins + 1)]


def _refine_cis_blocks(genome: str, feat: SeqFeature, ref_proteins, junction_flanks, reference_files=None) -> None:
    gene = feat.qualifiers.get("gene", [""])[0]
    if not ref_proteins or gene not in ref_proteins:
        return
    reading = [(int(p.start) + 1, int(p.end), p.strand or 1) for p in feat.location.parts]
    blocks, current = [], [0]
    for i in range(1, len(reading)):
        (s0, e0, st0), (s1, e1, st1) = reading[i - 1], reading[i]
        gap = s1 - e0 - 1 if st0 == 1 else s0 - e1 - 1
        if st0 == st1 and 0 < gap <= _CIS_MAX_INTRON:
            current.append(i)
        else:
            blocks.append(current)
            current = [i]
    blocks.append(current)
    blocks = [b for b in blocks if len(b) > 1]
    if not blocks:
        return
    from .splice_refine import refine_cis_block

    n_joins = len(reading) - 1
    flanks = [ref for ref in (junction_flanks or {}).get(gene, []) if len(ref) == n_joins]
    from .splice_refine import _san, _score

    changed = False
    for block in blocks:
        upstream = sum(e - s + 1 for s, e, _ in reading[: block[0]])
        phase_skip = (3 - upstream % 3) % 3
        first, last = block[0], block[-1]
        coords = [(reading[i][0], reading[i][1]) for i in block]
        block_flanks = [ref[first:last] for ref in flanks] or None
        strand = reading[first][2]
        block_seq = "".join(
            genome[a - 1 : b] if strand == 1 else str(Seq(genome[a - 1 : b]).reverse_complement()) for a, b in coords
        )[phase_skip:]
        query = _san(str(Seq(block_seq[: len(block_seq) - len(block_seq) % 3]).translate(table=11)))
        # As for whole genes: the reference protein closest to the transferred block decides.
        proteins = sorted(ref_proteins[gene], key=lambda p: _score(_san(p.rstrip("*")), query), reverse=True)
        for protein in proteins:
            new = refine_cis_block(
                genome,
                coords,
                protein,
                strand,
                phase_skip=phase_skip,
                final=last == len(reading) - 1,
                junction_flanks=block_flanks,
                structure_offsets=(_structure_offsets(gene, n_joins, reference_files) or [None] * n_joins)[first:last],
            )
            if new:
                for i, (a, b) in zip(block, new, strict=True):
                    if (a, b) != (reading[i][0], reading[i][1]):
                        changed = True
                    reading[i] = (a, b, reading[i][2])
                break
    if changed:
        feat.location = CompoundLocation([FeatureLocation(s - 1, e, strand=st) for s, e, st in reading])


def _annotate_edited_stops(
    feature: SeqFeature, genome: str, edits: list[tuple[int, int]], strand: int
) -> None:
    """Record read-through stops as RNA-edited codons (``transl_except``)."""
    feature.qualifiers["exception"] = ["RNA editing"]
    excepts = []
    for a, b in edits:
        codon = genome[a - 1 : b]
        if strand == -1:
            codon = str(Seq(codon).reverse_complement())
        aa = _EDITED_STOP_AA.get(codon, "OTHER")
        pos = f"{a}..{b}" if strand == 1 else f"complement({a}..{b})"
        excepts.append(f"(pos:{pos},aa:{aa})")
    feature.qualifiers["transl_except"] = excepts
    feature.qualifiers.setdefault("note", []).append(
        f"{len(edits)} in-frame genomic stop codon(s) read through as U-to-C edited sites"
    )


def reannotate_edited_stops(feature: SeqFeature, genome: str) -> None:
    """Recompute ``transl_except`` for a CDS placed elsewhere (e.g. its IR mirror).

    Every internal in-frame stop of the feature's own coding sequence is an
    edited site; the terminal stop is not.
    """
    for key in ("transl_except", "exception"):
        feature.qualifiers.pop(key, None)
    notes = [n for n in feature.qualifiers.get("note", []) if "read through as U-to-C" not in n]
    if notes:
        feature.qualifiers["note"] = notes
    else:
        feature.qualifiers.pop("note", None)
    genome = genome.upper()
    positions: list[tuple[int, int]] = []  # 1-based positions in coding order
    for part in feature.location.parts:
        span = range(int(part.start) + 1, int(part.end) + 1)
        positions.extend(
            (q, part.strand or 1) for q in (reversed(span) if part.strand == -1 else span)
        )
    offset = int(feature.qualifiers.get("codon_start", ["1"])[0]) - 1
    positions = positions[offset:]
    coding = str(feature.extract(Seq(genome)))[offset:]
    excepts = []
    for i in range(0, len(coding) - 3, 3):  # the terminal codon is not an edited stop
        codon = coding[i : i + 3]
        if codon not in _STOP:
            continue
        runs: list[list[tuple[int, int]]] = []
        for position in positions[i : i + 3]:
            if (
                runs
                and position[1] == runs[-1][-1][1]
                and position[0] == runs[-1][-1][0] + position[1]
            ):
                runs[-1].append(position)
            else:
                runs.append([position])
        spans = []
        for run in runs:
            a, b = sorted((run[0][0], run[-1][0]))
            span = str(a) if a == b else f"{a}..{b}"
            spans.append(span if run[0][1] == 1 else f"complement({span})")
        pos = spans[0] if len(spans) == 1 else f"join({','.join(spans)})"
        excepts.append(f"(pos:{pos},aa:{_EDITED_STOP_AA.get(codon, 'OTHER')})")
    if excepts:
        feature.qualifiers["exception"] = ["RNA editing"]
        feature.qualifiers["transl_except"] = excepts
        feature.qualifiers.setdefault("note", []).append(
            f"{len(excepts)} in-frame genomic stop codon(s) read through as U-to-C edited sites"
        )
