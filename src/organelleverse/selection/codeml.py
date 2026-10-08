"""CodeML selection-pressure analysis preprocessing. Self-contained.

Re-implements the bash CDS-preprocessing pipeline (batch PAL2NAL + PAML format)
in pure Python, with optional MAFFT for protein alignment.

Pipeline (6 steps, mirroring the standard codeml-prep bash script):
  1. validate_cds()    — check CDS integrity (multiple of 3, ATG start, no stops)
  2. translate()       — CDS → protein (standard genetic code)
  3. align_protein()   — protein MSA (MAFFT subprocess, or self-contained)
  4. pal2nal()         — back-align protein MSA → codon alignment (pure Python)
  5. to_paml()         — FASTA → PAML format (T→U conversion)
  6. prepare_codeml()  — one-call pipeline wrapping 1-5

The PAL2NAL back-alignment maps protein-alignment gaps back onto the codon
sequences, preserving the reading frame. This is the original PAL2NAL algorithm
(Suyama et al. 2006) reimplemented without the Perl dependency.
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import _contract
from ..core.external import run_external
from ..core.result import OrganelleResult
from .._bio import read_fasta
from ..selection.kaks import _CODONS as STANDARD_CODONS

__all__ = [
    "validate_cds",
    "translate_cds",
    "align_protein",
    "pal2nal",
    "to_paml",
    "prepare_codeml",
    "write_codeml_inputs",
]


# =========================================================================
# 1. CDS validation
# =========================================================================


def validate_cds(
    cds_fasta: str | Path,
    *,
    require_start: bool = True,
    allow_internal_stop: bool = False,
) -> OrganelleResult:
    """Validate CDS sequences for codeml analysis.

    Checks: multiple of 3, valid codons, ATG start (optional), no internal
    stop codons (optional). Returns per-sequence pass/fail + failure reasons.
    """
    parameters = {"require_start": require_start, "allow_internal_stop": allow_internal_stop}
    seqs = read_fasta(Path(cds_fasta))
    if not seqs:
        return _contract.failed(
            "validate_cds",
            summary_text="No sequences found.",
            anomalies=["empty"],
            parameters=parameters,
        )

    results: list[dict] = []
    n_pass = 0
    for name, seq in seqs:
        seq = seq.upper().replace("U", "T")
        issues: list[str] = []
        # check length
        if len(seq) % 3 != 0:
            issues.append("not_multiple_of_3")
        # check valid characters
        if any(c not in "ACGTN" for c in seq):
            issues.append("invalid_characters")
        # check start codon
        if require_start and len(seq) >= 3 and seq[:3] != "ATG":
            issues.append("no_start_codon")
        # check internal stops
        if not allow_internal_stop:
            for i in range(0, len(seq) - 2, 3):
                codon = seq[i : i + 3]
                if STANDARD_CODONS.get(codon) == "*" and i < len(seq) - 3:
                    issues.append("internal_stop")
                    break
        passed = len(issues) == 0
        if passed:
            n_pass += 1
        results.append({"name": name, "length": len(seq), "passed": passed, "issues": issues})

    return _contract.ok(
        "validate_cds",
        artifacts=(),
        metrics={
            "n_sequences": len(seqs),
            "n_passed": n_pass,
            "n_failed": len(seqs) - n_pass,
        },
        findings=_contract.findings(
            "validate_cds",
            (
                ("validated", n_pass),
                ("failed", len(seqs) - n_pass),
            ),
        ),
        flags=("all_valid",) if n_pass == len(seqs) else (),
        summary_text=f"CDS validation: {n_pass}/{len(seqs)} passed.",
        method="organelleverse",
        parameters=parameters,
    )


# =========================================================================
# 2. CDS translation (standard genetic code, table 1)
# =========================================================================


def translate_cds(
    cds_fasta: str | Path,
    *,
    strip_stop: bool = True,
) -> OrganelleResult:
    """Translate CDS sequences to protein (NCBI table 1).

    Return named protein sequences in ``metrics["records"]`` (``name``,
    ``seq``), as in :func:`align_protein` and :func:`pal2nal`.
    """
    parameters = {"strip_stop": strip_stop}
    seqs = read_fasta(Path(cds_fasta))
    if not seqs:
        return _contract.failed(
            "translate_cds",
            summary_text="No sequences.",
            anomalies=["empty"],
            parameters=parameters,
        )

    proteins: list[tuple[str, str]] = []
    for name, seq in seqs:
        seq = seq.upper().replace("U", "T")
        aa = _translate(seq, strip_stop)
        proteins.append((name, aa))

    return _contract.ok(
        "translate_cds",
        artifacts=(),
        metrics={
            "n_sequences": len(proteins),
            "records": [{"name": name, "seq": seq} for name, seq in proteins],
        },
        findings=_contract.findings("translate_cds", (("translated", len(proteins)),)),
        flags=(),
        summary_text=f"Translated {len(proteins)} CDS → protein.",
        method="organelleverse",
        parameters=parameters,
    )


def _translate(dna: str, strip_stop: bool = True) -> str:
    """Translate a DNA sequence to protein."""
    aa: list[str] = []
    for i in range(0, len(dna) - 2, 3):
        codon = dna[i : i + 3]
        residue = STANDARD_CODONS.get(codon, "X")
        if residue == "*":
            if strip_stop:
                break  # stop at first stop codon
            aa.append("*")
        else:
            aa.append(residue)
    return "".join(aa)


# =========================================================================
# 3. Protein alignment (MAFFT subprocess, or self-contained pairwise)
# =========================================================================


def align_protein(
    protein_fasta: str | Path,
    *,
    method: str = "mafft",
    executor: Any = None,
) -> OrganelleResult:
    """Align protein sequences (MAFFT subprocess or self-contained).

    With ``method="mafft"`` and MAFFT installed, runs ``mafft --auto``.
    Otherwise falls back to a self-contained ungapped concatenation.

    Compute-only: returns the aligned FASTA records in
    ``metrics["records"]`` and writes no user file. Materialize the
    alignment with :func:`_materialize_aligned_protein`,
    :func:`ov.selection.write`, or the generic :func:`ov.write`.
    """
    parameters = {"method": method}
    seqs = read_fasta(Path(protein_fasta))
    if not seqs:
        return _contract.failed(
            "align_protein",
            summary_text="No sequences.",
            anomalies=["empty"],
            parameters=parameters,
        )

    aligned: list[tuple[str, str]]
    used_method = "ungapped"
    # Priority: Rust MAFFT (pure-Rust, no external dep) > external MAFFT > ungapped
    from .. import accel

    max_len = max(len(s) for _, s in seqs)
    if accel.HAS_RUST and method in ("mafft", "rust-mafft"):
        try:
            raw = accel.mafft_align(list(seqs), "fft-ns-2")
            # validate: all sequences must have the same length and be non-empty
            if raw and all(len(s) > 0 for _, s in raw) and len({len(s) for _, s in raw}) == 1:
                aligned = raw
                used_method = "rust-mafft"
            else:
                raise ValueError("Rust MAFFT produced empty/invalid alignment")
        except Exception:
            # fall through to external MAFFT or ungapped
            if shutil.which("mafft"):
                try:
                    aligned = _run_mafft(Path(protein_fasta))
                    used_method = "external-mafft"
                except Exception:
                    aligned = [(name, seq.ljust(max_len, "-")) for name, seq in seqs]
            else:
                aligned = [(name, seq.ljust(max_len, "-")) for name, seq in seqs]
    elif method == "mafft" and shutil.which("mafft"):
        try:
            aligned = _run_mafft(Path(protein_fasta))
            used_method = "external-mafft"
        except Exception:
            aligned = [(name, seq.ljust(max_len, "-")) for name, seq in seqs]
    else:
        aligned = [(name, seq.ljust(max_len, "-")) for name, seq in seqs]

    return _contract.ok(
        "align_protein",
        artifacts=(),
        metrics={
            "n_sequences": len(aligned),
            "method": used_method,
            "mafft_used": used_method != "ungapped",
            "records": [{"name": name, "seq": seq} for name, seq in aligned],
        },
        findings=_contract.findings(
            "align_protein",
            (
                ("method", used_method),
                ("n_sequences", len(aligned)),
            ),
        ),
        flags=("aligned", f"method:{used_method}"),
        summary_text=f"Aligned {len(aligned)} proteins via {used_method}.",
        method=used_method,
        parameters=parameters,
    )


def _run_mafft(input_path: Path) -> list[tuple[str, str]]:
    """Run MAFFT and parse the aligned output."""
    result = run_external(["mafft", "--auto", str(input_path)], tool="MAFFT")
    return _read_fasta_str(result.stdout)


def _read_fasta_str(text: str) -> list[tuple[str, str]]:
    """Parse FASTA from a string."""
    seqs: list[tuple[str, str]] = []
    name: str | None = None
    chunks: list[str] = []
    for line in text.strip().splitlines():
        if line.startswith(">"):
            if name is not None:
                seqs.append((name, "".join(chunks)))
            name = line[1:].split()[0] if len(line) > 1 else ""
            chunks = []
        else:
            chunks.append(line.strip())
    if name is not None:
        seqs.append((name, "".join(chunks)))
    return seqs


# =========================================================================
# 4. PAL2NAL back-alignment (protein alignment → codon alignment)
#    Pure Python reimplementation of Suyama et al. 2006.
# =========================================================================


def pal2nal(
    protein_alignment_fasta: str | Path,
    cds_fasta: str | Path,
) -> OrganelleResult:
    """Back-align a protein MSA onto CDS sequences to produce a codon alignment.

    Maps each gap ('-') in the protein alignment to a gap codon ('---') in the
    codon alignment, preserving reading frame. This is the PAL2NAL algorithm
    (Suyama et al. 2006) reimplemented in pure Python — no Perl dependency.

    Both inputs must have matching sequence names. The CDS sequences must be
    ungapped (raw CDS); the protein alignment provides the gap structure.

    Compute-only: returns the codon FASTA records in
    ``metrics["records"]`` and writes no user file. Materialize the
    codon alignment with :func:`_materialize_pal2nal`,
    :func:`ov.selection.write`, or the generic :func:`ov.write`.
    """
    prot_aln = dict(read_fasta(Path(protein_alignment_fasta)))
    cds_seqs = dict(read_fasta(Path(cds_fasta)))

    if not prot_aln or not cds_seqs:
        return _contract.failed(
            "pal2nal",
            summary_text="Empty input.",
            anomalies=["empty"],
        )

    codon_aligned: list[tuple[str, str]] = []
    n_aligned = 0
    for name, prot_seq in prot_aln.items():
        cds = cds_seqs.get(name, "").upper().replace("U", "T")
        if not cds:
            continue
        codon_seq = _back_align(prot_seq, cds)
        codon_aligned.append((name, codon_seq))
        n_aligned += 1

    aln_len = len(codon_aligned[0][1]) // 3 if codon_aligned else 0
    return _contract.ok(
        "pal2nal",
        artifacts=(),
        metrics={
            "n_sequences": n_aligned,
            "alignment_codons": aln_len,
            "records": [{"name": name, "seq": seq} for name, seq in codon_aligned],
        },
        findings=_contract.findings(
            "pal2nal",
            (
                ("aligned_sequences", n_aligned),
                ("alignment_length_codons", aln_len),
            ),
        ),
        flags=("codon_aligned",),
        summary_text=f"Back-aligned {n_aligned} sequences to {aln_len} codons.",
        method="organelleverse",
    )


def _back_align(prot_aln: str, cds: str) -> str:
    """Map protein-alignment gaps back onto CDS to create a codon alignment.

    For each residue in the protein alignment:
      - if it's a gap '-', insert '---' in the codon alignment
      - otherwise, take the next 3 nucleotides from the CDS as a codon
    """
    result: list[str] = []
    cds_idx = 0
    for residue in prot_aln:
        if residue == "-":
            result.append("---")
        elif residue == "X":
            # unknown residue — take next codon if available
            if cds_idx + 3 <= len(cds):
                result.append(cds[cds_idx : cds_idx + 3])
                cds_idx += 3
            else:
                result.append("---")
        else:
            if cds_idx + 3 <= len(cds):
                result.append(cds[cds_idx : cds_idx + 3])
                cds_idx += 3
            else:
                result.append("---")  # CDS shorter than protein
    return "".join(result)


# =========================================================================
# 5. FASTA → PAML format conversion
# =========================================================================


def to_paml(
    alignment_fasta: str | Path,
    *,
    convert_t_to_u: bool = True,
) -> OrganelleResult:
    """Convert a FASTA alignment to PAML (sequential) format.

    PAML format:
        n_sequences  alignment_length
        seq_name
        sequence      (T → U if convert_t_to_u)
        ...

    Compute-only: returns the PAML text in ``metrics["paml_text"]``
    and writes no user file. Materialize it with :func:`_materialize_paml`,
    :func:`ov.selection.write`, or the generic :func:`ov.write`.
    """
    parameters = {"convert_t_to_u": convert_t_to_u}
    seqs = read_fasta(Path(alignment_fasta))
    if not seqs:
        return _contract.failed(
            "to_paml",
            summary_text="No sequences.",
            anomalies=["empty"],
            parameters=parameters,
        )

    n_seqs = len(seqs)
    aln_len = len(seqs[0][1])

    lines: list[str] = [f"{n_seqs}  {aln_len}"]
    for name, seq in seqs:
        seq_out = seq.replace("T", "U").replace("t", "u") if convert_t_to_u else seq
        lines.append(name)
        lines.append(seq_out)

    content = "\n".join(lines) + "\n"
    return _contract.ok(
        "to_paml",
        artifacts=(),
        metrics={
            "n_sequences": n_seqs,
            "alignment_length": aln_len,
            "t_to_u": convert_t_to_u,
            "paml_text": content,
        },
        findings=_contract.findings(
            "to_paml",
            (
                ("n_sequences", n_seqs),
                ("alignment_length", aln_len),
            ),
        ),
        flags=("paml_format",),
        summary_text=f"Converted {n_seqs} sequences ({aln_len} nt) to PAML format.",
        method="organelleverse",
        parameters=parameters,
    )


# =========================================================================
# 5b. Private typed materializers for the compute-only payloads above.
# =========================================================================


def _resolve_destination(destination: str | Path, default_name: str) -> Path:
    """An explicit file destination (with suffix) wins; otherwise append a name.

    A destination with a suffix is treated as an exact file path. A destination
    without a suffix is treated as a directory and joined with ``default_name``.
    """
    path = Path(destination)
    if path.suffix:
        return path
    return path / default_name


def _records_to_fasta(records: Sequence[Mapping[str, Any]]) -> str:
    return "\n".join(f">{rec['name']}\n{rec['seq']}" for rec in records) + "\n"


def _materialize_aligned_protein(result: OrganelleResult, destination: str | Path) -> Path:
    """Materialize an ``align_protein`` result as the aligned protein FASTA."""
    path = _resolve_destination(destination, "protein_aln.fasta")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_records_to_fasta(_records(result)))
    return path


def _materialize_pal2nal(result: OrganelleResult, destination: str | Path) -> Path:
    """Materialize a ``pal2nal`` result as the codon-alignment FASTA."""
    path = _resolve_destination(destination, "codon_aln.fasta")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_records_to_fasta(_records(result)))
    return path


def _materialize_paml(result: OrganelleResult, destination: str | Path) -> Path:
    """Materialize a ``to_paml`` result as the PAML sequential-format text."""
    path = _resolve_destination(destination, "codon_aln.paml")
    path.parent.mkdir(parents=True, exist_ok=True)
    text = result.metrics["paml_text"]
    if not isinstance(text, str):
        raise ValueError("to_paml result is missing its PAML text payload")
    path.write_text(text)
    return path


def _records(result: OrganelleResult) -> Sequence[Mapping[str, Any]]:
    """Read the frozen FASTA record payload out of a compute-only result."""
    records = result.metrics["records"]
    if not isinstance(records, tuple):
        raise ValueError("result is missing its FASTA record payload")
    return [record for record in records if isinstance(record, Mapping)]


# =========================================================================
# 6. One-call pipeline: prepare_codeml()
# =========================================================================


def prepare_codeml(
    cds_fasta: str | Path,
    *,
    require_start: bool = True,
    use_mafft: bool = True,
) -> OrganelleResult:
    """Prepare codeml-ready metrics without writing user-facing outputs."""
    with tempfile.TemporaryDirectory() as tmp:
        result = write_codeml_inputs(
            cds_fasta,
            tmp,
            require_start=require_start,
            use_mafft=use_mafft,
        )
    if result.status != "ok":
        return result
    return result.model_copy(
        update={
            "artifacts": (),
            "flags": tuple(f for f in result.flags if f != "codeml_ready")
            + ("codeml_ready", "output_dir_unbound"),
            "summary_text": (
                result.summary_text
                + " This is a compute-only result; call ov.selection.write(..., kind='codeml_inputs') to create files."
            ),
        }
    )


def write_codeml_inputs(
    cds_fasta: str | Path,
    output_dir: str | Path,
    *,
    require_start: bool = True,
    use_mafft: bool = True,
) -> OrganelleResult:
    """One-call pipeline: CDS → validate → translate → align → PAL2NAL → PAML.

    This mirrors the standard batch codeml-prep bash script but in pure Python.
    Produces in ``output_dir``:
      - validated_cds.fasta   (CDS passing validation)
      - protein.fasta         (translated proteins)
      - protein_aln.fasta     (protein alignment)
      - codon_aln.fasta       (PAL2NAL codon alignment)
      - codon_aln.paml        (PAML format, ready for codeml)
    """
    parameters = {"require_start": require_start, "use_mafft": use_mafft}
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    steps_done: list[str] = []
    step_results: dict[str, Any] = {}

    # Step 1: validate
    val = validate_cds(cds_fasta, require_start=require_start)
    step_results["validate"] = val.metrics
    steps_done.append("validate")
    if val.metrics["n_passed"] == 0:
        return _contract.failed(
            "prepare_codeml",
            summary_text="No CDS passed validation.",
            anomalies=["no_valid_cds"],
            parameters=parameters,
        )

    # filter to passing sequences
    all_seqs = read_fasta(Path(cds_fasta))
    valid_seqs = []
    for name, seq in all_seqs:
        seq_u = seq.upper().replace("U", "T")
        passed = len(seq_u) % 3 == 0
        if require_start and len(seq_u) >= 3 and seq_u[:3] != "ATG":
            passed = False
        if passed:
            valid_seqs.append((name, seq_u))
    valid_cds_path = out / "validated_cds.fasta"
    valid_cds_path.write_text("\n".join(f">{n}\n{s}" for n, s in valid_seqs) + "\n")
    steps_done.append("filter")

    # Step 2: translate
    prot_seqs = [(name, _translate(seq, strip_stop=True)) for name, seq in valid_seqs]
    prot_path = out / "protein.fasta"
    prot_path.write_text("\n".join(f">{n}\n{s}" for n, s in prot_seqs) + "\n")
    steps_done.append("translate")

    # Step 3: protein alignment (compute-only, then materialize for the next step)
    prot_aln_path = out / "protein_aln.fasta"
    aln = align_protein(prot_path, method="mafft" if use_mafft else "ungapped")
    step_results["align"] = aln.metrics
    _materialize_aligned_protein(aln, prot_aln_path)
    steps_done.append("align")

    # Step 4: PAL2NAL back-alignment (reads the materialized protein alignment)
    codon_aln_path = out / "codon_aln.fasta"
    p2n = pal2nal(prot_aln_path, valid_cds_path)
    step_results["pal2nal"] = p2n.metrics
    _materialize_pal2nal(p2n, codon_aln_path)
    steps_done.append("pal2nal")

    # Step 5: PAML format (reads the materialized codon alignment)
    paml_path = out / "codon_aln.paml"
    paml_result = to_paml(codon_aln_path)
    _materialize_paml(paml_result, paml_path)
    steps_done.append("paml")

    output_files = [valid_cds_path, prot_path, prot_aln_path, codon_aln_path, paml_path]

    return _contract.ok(
        "prepare_codeml",
        artifacts=_contract.artifacts(output_files),
        metrics={
            "steps_completed": len(steps_done),
            "n_valid_cds": len(valid_seqs),
            "alignment_codons": p2n.metrics.get("alignment_codons", 0),
            "mafft_used": aln.metrics.get("mafft_used", False),
        },
        findings=_contract.findings(
            "prepare_codeml",
            (
                ("valid_cds", len(valid_seqs)),
                ("alignment_codons", p2n.metrics.get("alignment_codons", 0)),
                ("steps", len(steps_done)),
            ),
        ),
        flags=("codeml_ready", "mafft_used")
        if aln.metrics.get("mafft_used")
        else ("codeml_ready",),
        summary_text=f"Codeml prep complete: {len(valid_seqs)} CDS → "
        f"{p2n.metrics.get('alignment_codons', 0)} codons, "
        f"{'MAFFT' if aln.metrics.get('mafft_used') else 'ungapped'}, "
        f"PAML format ready.",
        method="organelleverse",
        parameters=parameters,
    )
