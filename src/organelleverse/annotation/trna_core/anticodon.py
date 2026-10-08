"""Organelle tRNA anticodon normalization and isotype lookup."""

from __future__ import annotations

_MITOCHONDRION_ANTICODON_TO_AA: dict[str, tuple[str, ...]] = {
    "aaa": ("F",),
    "acg": ("R",),
    "caa": ("L",),
    "cat": ("M", "fM", "I"),
    "cca": ("W",),
    "ccg": ("R",),
    "cct": ("R",),
    "cga": ("S",),
    "cgt": ("T",),
    "ctt": ("K",),
    "gaa": ("F",),
    "gac": ("V",),
    "gat": ("I",),
    "gca": ("C",),
    "gcc": ("G",),
    "gct": ("S",),
    "gga": ("S",),
    "ggt": ("T",),
    "gta": ("Y",),
    "gtc": ("D",),
    "gtg": ("H",),
    "gtt": ("N",),
    "taa": ("L",),
    "tac": ("V",),
    "tag": ("L",),
    "tcc": ("G",),
    "tct": ("R",),
    "tga": ("S",),
    "tgc": ("A",),
    "tgg": ("P",),
    "tgt": ("T",),
    "ttc": ("E",),
    "ttg": ("Q",),
    "ttt": ("K",),
}

_PLASTOME_ANTICODON_TO_AA: dict[str, tuple[str, ...]] = {
    anticodon: tuple(aa for aa in amino_acids if aa != "fM")
    for anticodon, amino_acids in _MITOCHONDRION_ANTICODON_TO_AA.items()
}


def normalize_anticodon(value: str) -> str:
    """Return a three-base DNA anticodon in lowercase."""
    cleaned = value.strip().lower().replace("u", "t")
    if len(cleaned) != 3 or any(base not in {"a", "c", "g", "t", "n"} for base in cleaned):
        raise ValueError(f"Invalid anticodon: {value!r}")
    return cleaned


def isotypes_for_anticodon(anticodon: str, organelle: str = "mitochondrion") -> tuple[str, ...]:
    normalized = normalize_anticodon(anticodon)
    if "n" in normalized:
        return ()
    if organelle == "mitochondrion":
        return _MITOCHONDRION_ANTICODON_TO_AA.get(normalized, ())
    if organelle == "plastome":
        return _PLASTOME_ANTICODON_TO_AA.get(normalized, ())
    raise ValueError(f"Unsupported organelle for tRNA anticodon lookup: {organelle!r}")


def known_anticodons(organelle: str = "mitochondrion") -> tuple[str, ...]:
    if organelle == "mitochondrion":
        return tuple(sorted(_MITOCHONDRION_ANTICODON_TO_AA))
    if organelle == "plastome":
        return tuple(sorted(_PLASTOME_ANTICODON_TO_AA))
    raise ValueError(f"Unsupported organelle for tRNA anticodon lookup: {organelle!r}")
