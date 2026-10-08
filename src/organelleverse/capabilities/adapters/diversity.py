"""Adapter decisions for the ``diversity`` domain's restored capabilities.

Ledger records for this domain (``docs/operations/restored-capabilities.toml``):

* ``diversity.nucleotide_diversity`` / ``diversity.neutral_tests`` -
  ``canonical`` result_shape, first parameter ``alignment_fasta: str |
  Path`` (not a bare ``OrganelleGenome``), so neither matches
  ``canonical_core``. Resolved below: ``alignment_fasta`` gets the ``path``
  codec (a real file ``read_fasta`` opens); both already return a real
  ``OrganelleResult`` they built themselves, so ``result_codec =
  "canonical"``. ``nucleotide_diversity``'s ``window_size``/``step`` keep
  their own defaults.
* ``diversity.compute_nucleotide_diversity`` / ``.compute_neutral_tests`` -
  the ``json`` twins (``diversity/diversity_core.py``, no
  ``organelleverse.core`` types): same ``alignment_fasta: str | Path`` first
  parameter gets ``path``; ``json_metric`` under ``"diversity_metrics"`` /
  ``"neutral_tests"`` respectively. **Caveat, discovered while proving this
  domain end to end**: unlike its ``OrganelleResult``-returning sibling
  ``diversity.neutral_tests`` (``diversity.py:278-279`` explicitly converts
  a NaN ``theta_w``/``tajima_d`` to JSON ``null`` before returning),
  ``compute_neutral_tests``'s raw dict does not sanitize NaN - Tajima's D is
  mathematically undefined below 3 segregating sites, a real and common
  alignment shape, and ``json_metric``'s ``_is_finite_json`` check correctly
  rejects it at invocation time (``capability.result_codec_invalid``) rather
  than silently coercing NaN to anything. This is not a wrong codec choice -
  ``json_metric`` is still the honest binding for the capability's general
  case - it is a genuine, data-dependent gap in the *implementation*, left
  exactly as found; the generator does not paper over implementation bugs.
* ``diversity.write_nucleotide`` / ``.write_neutral_tests`` - both take
  ``result: OrganelleResult | Mapping[str, Any]`` as their required first
  parameter. No implemented ``ParameterCodec`` decodes an ``OrganelleResult``
  as an Agent-facing JSON argument (``LEGACY_RESULT`` is declared but
  unimplemented - ``python_binding._UNIMPLEMENTED_PARAMETER_CODECS``), so
  neither has an entry here - the generator fails closed for both with
  ``capability.no_adapter_override``.
Capability Plan 04 Task 1 adds ``FIXTURES`` for all four bindable
capabilities, sharing one 6-sequence, 40 bp alignment
(``_ALIGNMENT_FASTA``) with 4 segregating sites - enough for
``neutral_tests``/``compute_neutral_tests`` to compute a *finite* Tajima's D
(this environment has scikit-allel installed, so both take the
``allel.tajima_d`` path; a 3-sequence alignment was tried first and
produced a real but non-finite ``nan`` from scikit-allel's own variance
formula degenerating at that sample size - confirmed interactively, not
guessed - so it was replaced rather than special-cased). No timestamp, no
randomness: ``diversity.py``'s own ``_provenance`` never sets
``started_at``/``finished_at``.
"""

from __future__ import annotations

from dataclasses import dataclass

from organelleverse.capabilities.adapters import FixtureCase, FixtureFile
from organelleverse.operations.spec import ParameterCodec, ResultCodec, SideEffect

# A 41 bp base sequence repeated across 6 sequences, each carrying one
# distinct substitution (positions 3, 8, 15, 22, 30) -> 4 real segregating
# sites (one substitution happens to coincide with the reference base at
# that position and is dropped by read_fasta's own upper-casing, verified
# interactively) and a finite Tajima's D via scikit-allel.
_BASE_SEQ = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"
_SUBSTITUTIONS = ((3, "A"), (8, "T"), (15, "C"), (22, "G"), (30, "A"))


def _substituted(position: int, base: str) -> str:
    chars = list(_BASE_SEQ)
    chars[position] = base if chars[position] != base else "N"
    return "".join(chars)


_ALIGNMENT_SEQS = (_BASE_SEQ, *(_substituted(position, base) for position, base in _SUBSTITUTIONS))
_ALIGNMENT_FASTA = FixtureFile(
    relative_path="alignment.fasta",
    content="".join(f">s{index}\n{seq}\n" for index, seq in enumerate(_ALIGNMENT_SEQS)),
)


@dataclass(frozen=True)
class ParameterOverride:
    """One named, Agent-facing parameter the generator could not derive alone."""

    name: str
    codec: ParameterCodec
    path_role: str | None = None


@dataclass(frozen=True)
class NamedParameterOverride:
    """The full ``named_parameters`` binding plan for one restored capability."""

    parameters: tuple[ParameterOverride, ...]
    result_codec: ResultCodec
    result_key: str | None
    side_effects: tuple[SideEffect, ...]


OVERRIDES: dict[str, NamedParameterOverride] = {
    "diversity.sliding_window_diversity": NamedParameterOverride(
        parameters=tuple(
            ParameterOverride(name=name, codec=codec)
            for name, codec in (
                ("alignment_fasta", ParameterCodec.PATH),
                ("window_size", ParameterCodec.JSON),
                ("step", ParameterCodec.JSON),
                ("gap_mode", ParameterCodec.JSON),
                ("normalize_orientation", ParameterCodec.JSON),
                ("orientation_reference", ParameterCodec.PATH),
                ("alignment_method", ParameterCodec.JSON),
                ("single_sample_drop", ParameterCodec.JSON),
            )
        ),
        result_codec=ResultCodec.CANONICAL,
        result_key=None,
        side_effects=(SideEffect.READ_FILES, SideEffect.SUBPROCESS, SideEffect.WRITE_FILES),
    ),
    "diversity.nucleotide_diversity": NamedParameterOverride(
        parameters=(ParameterOverride(name="alignment_fasta", codec=ParameterCodec.PATH),),
        result_codec=ResultCodec.CANONICAL,
        result_key=None,
        side_effects=(SideEffect.READ_FILES,),
    ),
    "diversity.neutral_tests": NamedParameterOverride(
        parameters=(ParameterOverride(name="alignment_fasta", codec=ParameterCodec.PATH),),
        result_codec=ResultCodec.CANONICAL,
        result_key=None,
        side_effects=(SideEffect.READ_FILES,),
    ),
    "diversity.compute_nucleotide_diversity": NamedParameterOverride(
        parameters=(ParameterOverride(name="alignment_fasta", codec=ParameterCodec.PATH),),
        result_codec=ResultCodec.JSON_METRIC,
        result_key="diversity_metrics",
        side_effects=(SideEffect.READ_FILES,),
    ),
    "diversity.compute_neutral_tests": NamedParameterOverride(
        parameters=(ParameterOverride(name="alignment_fasta", codec=ParameterCodec.PATH),),
        result_codec=ResultCodec.JSON_METRIC,
        result_key="neutral_tests",
        side_effects=(SideEffect.READ_FILES,),
    ),
}
"""``diversity.write_nucleotide`` and ``.write_neutral_tests`` are
deliberately absent - see the module docstring for why no codec can decode
their ``OrganelleResult | Mapping`` parameter honestly.
"""

FIXTURES: dict[str, tuple[FixtureCase, ...]] = {
    capability_id: (
        FixtureCase(
            case="basic",
            files=(_ALIGNMENT_FASTA,),
            parameters={"alignment_fasta": _ALIGNMENT_FASTA.relative_path},
        ),
    )
    for capability_id in (
        "diversity.nucleotide_diversity",
        "diversity.neutral_tests",
        "diversity.compute_nucleotide_diversity",
        "diversity.compute_neutral_tests",
    )
}

FIXTURES["diversity.sliding_window_diversity"] = (
    FixtureCase(
        case="basic",
        files=(
            FixtureFile(
                relative_path="alignment.fasta",
                content=">s0\nGCTAAAGACAATTACATAACATACACGTCAGCACGAAACTTGTTGGCCCAGTGTGAATCG\nCTTAAGGGTTAAGTAAGTGTGATGCATACGCCTTTACTTGCTGTGTCCACCCCATCGGAC\nTGGCATTTTTATTACACTCAGAAACAGAACTCGGGTAATTTTGACAGGTCACGCAGAGGC\nGCGCCCTCCTGAAGTGCGTGGACACTCGCTATGAATCTCTGATTTACCCACTCTGCCAAA\nCTCCAGCGCGGTCAGTTCCATCACCCTAAGTAACCGAATAATGCGTTCGCTCTATTGACT\n>s1\nGCTAAAGACAATTACATAACATACACGTCAGCACGAAACTTGTTGGCCCAGTGTGAATCG\nCTTAAGGGTTAAGTAAGTGTGATGCATACGCCTTTACTTGCTGTGTCCACCCCATCGGAC\nAACTTCTCAG-GCGCTCACGGATTACCGTGACTTCCGATACTGGGGCTGAAAGAGGTCAA\nGCGCCCTCCTGAAGTGCGTGGACACTCGCTATGAATCTCTGATTTACCCACTCTGCCAAA\nCTCCAGCGCGGTCAGTTCCATCACCCTAAGTAACCGAATAATGCGTTCGCTCTATTGACT\n>s2\nGCTAAAGACAATTACATAACATACACGTCAGCACGAAACTTGTTGGCCCAGTGTGAATCG\nCTTAAGGGTTAAGTAAGTGTGATGCATACGCCTTTACTTGCTGTGTCCACCCCATCGGAC\nCCGCTCGGGTTAAGGGTGTAAGAGCGCCCAATATCCAACTGCTGCCAACACTGCTCCACT\nGCGCCCTCCTGAAGTGCGTGGACACTCGCTATGAATCTCTGATTTACCCACTCTGCCAAA\nCTCCAGCGCGGTCAGTTCCATCACCCTAAGTAACCGAATAATGCGTTCGCTCTATTGACT\n>s3\nGCTAAAGACAATTACATAACATACACGTCAGCACGAAACTTGTTGGCCCAGTGTGAATCG\nCTTAAGGGTTAAGTAAGTGTGATGCATACGCCTTTACTTGCTGTGTCCACCCCATCGGAC\nGGCACTTGATGAAACTGAAACAGCAAGGGGACTGGTCGAGACAGAGGTCGTACTCAAAGT\nGCGCCCTCCTGAAGTGCGTGGACACTCGCTATGAATCTCTGATTTACCCACTCTGCCAAA\nCTCCAGCGCGGTCAGTTCCATCACCCTAAGTAACCGAATAATGCGTTCGCTCTATTGACT\n",
            ),
        ),
        parameters={
            "alignment_fasta": "alignment.fasta",
            "window_size": 100,
            "step": 50,
            "gap_mode": "exclude",
        },
    ),
)

__all__ = ["FIXTURES", "OVERRIDES", "NamedParameterOverride", "ParameterOverride"]
