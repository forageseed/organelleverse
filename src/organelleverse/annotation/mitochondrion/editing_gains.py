"""Initiators and terminators that exist only after C-to-U editing.

A plant mitochondrial gene is frequently not a gene in its own DNA. C-to-U
editing of the transcript creates codons the genome does not carry, so a CDS may
begin at a triplet that reads ACG on the chromosome and AUG in the message, or
end at one that reads CAA and becomes UAA. Annotating from DNA alone puts the
start in the wrong place or runs the reading frame past its real end.

The rule is chemical and needs no editing prediction to apply, because the
substitution only ever goes one way:

* **terminators are only created, never destroyed.** UAA, UAG and UGA contain no
  editable C, so no amount of editing can remove a stop. CAA, CAG and CGA each
  become one. A stop present in the DNA is therefore real, and the candidates for
  a gained stop are exactly those three codons in frame.
* **initiators are created from ACG**, and from ACA/ACU/ACC only via a second
  edit, so ACG is the candidate worth taking seriously.

That asymmetry is what lets this run before any editing model: the search space
is fixed by the genetic code, and a predictor -- Deepred-mt or otherwise -- can
only rank candidates inside it, never widen it. Gymnosperms make the separation
matter, at roughly 913 editing sites per species against 400-600 in angiosperms,
which is outside the range those predictors were trained on.

The gene lists in the database say which genes are *known* to need this. They
narrow the search; they do not define it, so a gene absent from the list is still
checked and simply tends to find nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Codons one C-to-U edit away from a terminator, with the terminator produced.
STOP_GAIN_CODONS = {"CAA": "TAA", "CAG": "TAG", "CGA": "TGA"}

#: Codons one C-to-U edit away from AUG.
START_GAIN_CODONS = {"ACG": "ATG"}

#: Terminators, which editing can create but never remove.
STOP_CODONS = frozenset({"TAA", "TAG", "TGA"})


@dataclass(frozen=True)
class Gain:
    """A codon that becomes an initiator or terminator once edited."""

    kind: str           # "start" | "stop"
    position: int       # 1-based genome coordinate of the codon's first base
    dna_codon: str
    edited_codon: str
    offset_codons: int  # signed distance from the annotated boundary, in codons

    @property
    def description(self) -> str:
        return (f"{self.kind} gain: {self.dna_codon}->{self.edited_codon} "
                f"at {self.position} ({self.offset_codons:+d} codons)")


def _codons(seq: str) -> list[str]:
    return [seq[i : i + 3] for i in range(0, len(seq) - 2, 3)]


def find_stop_gain(cds: str, cds_start: int) -> Gain | None:
    """The first in-frame codon that editing would turn into a terminator.

    Searched from the 5' end, because a terminator gained early truncates the
    protein there however much open frame follows -- the DNA's own stop is never
    reached. A CDS that already ends in a terminator is left alone: editing
    cannot remove one, so that stop is real and nothing here applies.
    """
    codons = _codons(cds)
    if not codons or codons[-1] in STOP_CODONS:
        return None

    for index, codon in enumerate(codons[:-1]):
        edited = STOP_GAIN_CODONS.get(codon)
        if edited:
            return Gain(
                kind="stop",
                position=cds_start + 3 * index,
                dna_codon=codon,
                edited_codon=edited,
                offset_codons=index - len(codons),
            )
    return None


def find_start_gain(upstream: str, cds_start: int, *, max_upstream_codons: int = 30) -> Gain | None:
    """The nearest upstream ACG that editing would turn into an initiator.

    ``upstream`` is the in-frame sequence immediately 5' of the annotated start,
    on the coding strand. The nearest candidate wins: reaching further than the
    evidence supports invents N-terminal residues, and a genuine initiator beyond
    an in-frame terminator cannot belong to this reading frame at all, which is
    what bounds the search.
    """
    codons = _codons(upstream)
    if not codons:
        return None

    for back, codon in enumerate(reversed(codons[-max_upstream_codons:]), 1):
        if codon in STOP_CODONS:
            return None
        if codon in START_GAIN_CODONS or codon == "ATG":
            return Gain(
                kind="start",
                position=cds_start - 3 * back,
                dna_codon=codon,
                edited_codon=START_GAIN_CODONS.get(codon, codon),
                offset_codons=-back,
            )
    return None


def scan_gains(cds: str, cds_start: int, upstream: str = "") -> list[Gain]:
    """Both gains for one gene, in genome order.

    Reported rather than applied. Whether to move a boundary needs the profile
    alignment as well; this states only what the sequence permits.
    """
    gains: list[Gain] = []
    start = find_start_gain(upstream, cds_start) if upstream else None
    if start:
        gains.append(start)
    stop = find_stop_gain(cds, cds_start)
    if stop:
        gains.append(stop)
    return gains


def edited_protein(cds: str, edits: list[int], table: int = 1) -> str:
    """Translate a spliced CDS with C-to-U editing applied at the given
    codon indices (0-based). Each edited codon's first-base C becomes U."""
    if not cds:
        return ""
    bases = list(cds.upper())
    for index in edits:
        position = index * 3
        if position + 2 >= len(bases):
            continue
        codon = "".join(bases[position : position + 3])
        if codon == "ACG":
            bases[position + 1] = "T"   # A(C)G -> A(U)G
        elif codon in STOP_GAIN_CODONS and bases[position] == "C":
            bases[position] = "T"       # (C)AA -> (U)AA, ...

    from .pcg import translate_sequence

    protein = translate_sequence("".join(bases), table=table)
    return protein[:-1] if protein.endswith("*") else protein


def annotate_rna_edits(
    ann: "GeneAnnotation",
    genome: "GenomeSequence",
    db_manager: "DBManager | None" = None,
) -> "GeneAnnotation":
    """Detect and record RNA-editing sites on a CDS annotation.

    Editing sites are found by the chemical rule (C-to-U only): the start is an
    ACG initiator gain and the stop is a CAA/CGA/CAG terminator gain. Internal
    premature stops are never editing sites — a stop contains no editable C, so
    editing cannot remove one; an internal stop is always a boundary or frame
    artifact and is out of scope for this DNA-only pass.

    Convergent: the annotation changes only when at least one site is found,
    and the exception qualifier is added together with the sites.
    """
    from .cds import _extract_cds_sequence

    if ann.gene_type != "CDS" or ann.is_pseudo or not ann.exons:
        return ann

    cds = _extract_cds_sequence(ann, genome)
    if len(cds) < 3 or len(cds) % 3 != 0:
        return ann
    codons = _codons(cds)
    if not codons:
        return ann

    has_exception = any(
        exception.strip().casefold() == "rna editing" for exception in ann.exceptions
    )
    is_start_gain = db_manager.is_start_gain_gene(ann.gene_name) if db_manager else False
    is_stop_gain = db_manager.is_stop_gain_gene(ann.gene_name) if db_manager else False

    edits: list[int] = []
    # start gain: ACG that editing turns into AUG
    if codons[0] == "ACG" and (is_start_gain or has_exception):
        edits.append(0)
    # stop gain at the annotated terminus
    if codons[-1] in STOP_GAIN_CODONS and (is_stop_gain or has_exception):
        edits.append(len(codons) - 1)
    edits = sorted(set(edits))
    if not edits:
        return ann

    notes = list(ann.notes)
    notes.append(
        "RNA editing: "
        + ", ".join(
            f"{codons[i]}->{STOP_GAIN_CODONS.get(codons[i], 'ATG' if i == 0 and codons[i] == 'ACG' else '?')}"
            for i in edits
        )
    )
    exceptions = list(ann.exceptions)
    if "RNA editing" not in exceptions:
        exceptions.append("RNA editing")
    return ann.model_copy(update={"rna_edits": edits, "notes": notes, "exceptions": exceptions})
