"""Versioned six-frame organelle marker HMM evidence."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import pyhmmer
from Bio.Seq import Seq

from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.errors import OrganelleInputError
from organelleverse.quality_control.fasta import read_fasta

from .contracts import (
    MarkerEvidence,
    MarkerHit,
    MarkerProfileSet,
    QcCheck,
    ResolvedAssemblyEvidence,
    SequenceIdentity,
)
from .policy import QcPolicy


@dataclass(frozen=True)
class _Frame:
    sequence_id: str
    strand: Literal["+", "-"]
    frame: int
    nucleotide_length: int
    protein: str


def collect_marker_evidence(
    evidence: ResolvedAssemblyEvidence,
    profile_set: MarkerProfileSet | tuple[MarkerProfileSet, ...],
    policy: QcPolicy,
) -> MarkerEvidence:
    """Search one or more pinned amino or nucleotide HMM profile sets."""

    profile_sets = (profile_set,) if isinstance(profile_set, MarkerProfileSet) else profile_set
    if not profile_sets:
        raise OrganelleInputError(
            code="qc.input_contract_violation",
            message="at least one marker profile set is required",
        )
    sequences = tuple(read_fasta(evidence.primary_fasta_path))
    raw_hits: list[MarkerHit] = []
    target_organelle = evidence.primary_genome.organelle
    expected_target_profiles: set[tuple[str, str]] = set()
    for profiles in profile_sets:
        hmms, alphabet = _load_profiles(profiles)
        expected_target_profiles.update(
            (profiles.hmm_artifact.object_id, profile_id)
            for profile_id in (_decode_name(hmm.name) for hmm in hmms)
            if _profile_target(profile_id, profiles.target) == target_organelle
        )
        if alphabet.is_amino():
            if profiles.genetic_code is None:
                raise OrganelleInputError(
                    code="qc.input_contract_violation",
                    message="amino marker profiles require a genetic code",
                    details={"profile_set_id": profiles.profile_set_id},
                )
            raw_hits.extend(_search_amino_profiles(sequences, hmms, alphabet, profiles, policy))
        elif alphabet.is_nucleotide():
            raw_hits.extend(
                _search_nucleotide_profiles(sequences, hmms, alphabet, profiles, policy)
            )
        else:
            raise OrganelleInputError(
                code="qc.input_contract_violation",
                message="marker profile alphabet is unsupported",
                details={"profile_set_id": profiles.profile_set_id},
            )

    hits = _consolidate_hits(raw_hits)
    sequence_ids = tuple(sequence_id for sequence_id, _ in sequences)
    identities = tuple(
        SequenceIdentity(
            sequence_id=sequence_id,
            identity_evidence_status=(
                "target_supported"
                if any(
                    hit.sequence_id == sequence_id and hit.target == target_organelle
                    for hit in hits
                )
                else "unresolved"
            ),
        )
        for sequence_id in sequence_ids
    )
    direct_count = sum(hit.target == target_organelle for hit in hits)
    conflicting_count = sum(bool(hit.target) and hit.target != target_organelle for hit in hits)
    complete_target_hits = tuple(
        hit for hit in hits if hit.target == target_organelle and hit.complete
    )
    recovered_target_profiles = {
        (artifact_id, hit.profile_id)
        for hit in complete_target_hits
        for artifact_id in hit.evidence_artifact_ids
        if (artifact_id, hit.profile_id) in expected_target_profiles
    }
    complete_target_profile_count = len(recovered_target_profiles)
    expected_target_profile_count = len(expected_target_profiles)
    target_profile_recovery_fraction = (
        complete_target_profile_count / expected_target_profile_count
        if expected_target_profile_count
        else None
    )
    duplicate_complete_target_profile_count = sum(
        max(
            0,
            sum(
                hit.profile_id == profile_id and artifact_id in hit.evidence_artifact_ids
                for hit in complete_target_hits
            )
            - 1,
        )
        for artifact_id, profile_id in expected_target_profiles
    )
    artifact_ids = tuple(profiles.hmm_artifact.object_id for profiles in profile_sets)
    checks = (
        QcCheck(
            check_id="qc.target_marker_support",
            category="identity",
            status="pass" if direct_count else "warn",
            value=direct_count,
            unit="marker_hits",
            message=(
                "Target-organelle marker profiles were detected."
                if direct_count
                else "No direct target-organelle marker profile was detected."
            ),
            finding_code="" if direct_count else "qc.target_marker_absent",
            evidence_artifact_ids=artifact_ids,
        ),
        QcCheck(
            check_id="qc.conflicting_marker_context",
            category="identity",
            status="warn" if conflicting_count else "pass",
            value=conflicting_count,
            unit="marker_hits",
            message=(
                "Conflicting-organelle markers require structural-context interpretation."
                if conflicting_count
                else "No conflicting-organelle marker profile was detected."
            ),
            finding_code="qc.conflicting_organelle_marker" if conflicting_count else "",
            evidence_artifact_ids=artifact_ids,
        ),
        QcCheck(
            check_id="qc.target_marker_profile_recovery",
            category="identity",
            status=(
                "pass"
                if complete_target_profile_count
                else "warn"
                if expected_target_profile_count
                else "not_assessed"
            ),
            value=target_profile_recovery_fraction,
            unit="fraction" if target_profile_recovery_fraction is not None else "",
            message=(
                "Reports complete target HMM profile recovery; this is not genome completeness."
                if expected_target_profile_count
                else "No target-organelle HMM profile was present in the supplied profile sets."
            ),
            finding_code=(
                "qc.target_marker_profile_absent"
                if expected_target_profile_count and not complete_target_profile_count
                else ""
            ),
            evidence_artifact_ids=artifact_ids,
        ),
    )
    profile_ids = tuple(dict.fromkeys(item.profile_set_id for item in profile_sets))
    versions = tuple(dict.fromkeys(item.version for item in profile_sets))
    genetic_codes = {item.genetic_code for item in profile_sets}
    combined_sha256 = (
        profile_sets[0].sha256
        if len(profile_sets) == 1
        else hashlib.sha256("\n".join(item.sha256 for item in profile_sets).encode()).hexdigest()
    )
    return MarkerEvidence(
        profiles_assessed=True,
        profile_set_id="+".join(profile_ids),
        profile_version="+".join(versions),
        profile_sha256=combined_sha256,
        genetic_code=genetic_codes.pop() if len(genetic_codes) == 1 else None,
        expected_target_profile_count=expected_target_profile_count,
        complete_target_profile_count=complete_target_profile_count,
        target_profile_recovery_fraction=target_profile_recovery_fraction,
        duplicate_complete_target_profile_count=duplicate_complete_target_profile_count,
        marker_hits=hits,
        sequence_identities=identities,
        checks=checks,
    )


def _load_profiles(
    profile_set: MarkerProfileSet,
) -> tuple[list[Any], Any]:
    profile_path = _verify_profile(profile_set)
    hmms: list[Any] = []
    with pyhmmer.plan7.HMMFile(profile_path) as source:
        hmms.extend(source)
    if not hmms:
        raise OrganelleInputError(
            code="qc.input_contract_violation",
            message="marker profile set contains no HMM profiles",
            details={"profile_set_id": profile_set.profile_set_id},
        )
    alphabet = hmms[0].alphabet
    if any(hmm.alphabet != alphabet for hmm in hmms):
        raise OrganelleInputError(
            code="qc.input_contract_violation",
            message="marker profile set mixes incompatible alphabets",
            details={"profile_set_id": profile_set.profile_set_id},
        )
    return hmms, alphabet


def _search_amino_profiles(
    sequences: tuple[tuple[str, str], ...],
    hmms: list[Any],
    alphabet: Any,
    profile_set: MarkerProfileSet,
    policy: QcPolicy,
) -> list[MarkerHit]:
    assert profile_set.genetic_code is not None
    frames: dict[str, _Frame] = {}
    targets: list[Any] = []
    for sequence_id, nucleotide in sequences:
        for strand, frame, protein in _six_frames(nucleotide, profile_set.genetic_code):
            if not protein:
                continue
            target_name = f"frame:{len(frames)}"
            frames[target_name] = _Frame(
                sequence_id=sequence_id,
                strand=strand,
                frame=frame,
                nucleotide_length=len(nucleotide),
                protein=protein,
            )
            targets.append(
                pyhmmer.easel.TextSequence(
                    name=target_name.encode(),
                    sequence=protein.encode(),
                ).digitize(alphabet)
            )
    if not targets:
        return []
    results = cast(
        Iterable[Any],
        pyhmmer.hmmer.hmmsearch(
            hmms,
            targets,
            cpus=1,
            E=policy.marker_evalue,
        ),
    )
    hits: list[MarkerHit] = []
    for top_hits in results:
        query = top_hits.query
        profile_id = _decode_name(query.name)
        target = _profile_target(profile_id, profile_set.target)
        hmm_length = int(query.M)
        for hit in top_hits.included:
            frame_record = frames.get(_decode_name(hit.name))
            if frame_record is None:
                continue
            for domain in hit.domains.included:
                env_from = int(domain.env_from)
                env_to = int(domain.env_to)
                start, end = _domain_coordinates(frame_record, env_from, env_to)
                hmm_covered = int(domain.alignment.hmm_to) - int(domain.alignment.hmm_from) + 1
                uninterrupted = "X" not in frame_record.protein[env_from - 1 : env_to]
                hits.append(
                    MarkerHit(
                        profile_id=profile_id,
                        sequence_id=frame_record.sequence_id,
                        start=start,
                        end=end,
                        strand=frame_record.strand,
                        score=max(0.0, float(domain.score)),
                        complete=(hmm_covered / max(1, hmm_length))
                        >= policy.marker_complete_fraction
                        and uninterrupted,
                        target=target,
                        evidence_artifact_ids=(profile_set.hmm_artifact.object_id,),
                    )
                )
    return hits


def _search_nucleotide_profiles(
    sequences: tuple[tuple[str, str], ...],
    hmms: list[Any],
    alphabet: Any,
    profile_set: MarkerProfileSet,
    policy: QcPolicy,
) -> list[MarkerHit]:
    targets = [
        pyhmmer.easel.TextSequence(
            name=sequence_id.encode(),
            sequence=nucleotide.upper().encode(),
        ).digitize(alphabet)
        for sequence_id, nucleotide in sequences
    ]
    if not targets:
        return []
    results = cast(
        Iterable[Any],
        pyhmmer.hmmer.nhmmer(  # pyright: ignore[reportUnknownMemberType]
            hmms,
            targets,
            cpus=1,
            E=policy.marker_evalue,
        ),
    )
    hits: list[MarkerHit] = []
    for top_hits in results:
        query = top_hits.query
        profile_id = _decode_name(query.name)
        target = _profile_target(profile_id, profile_set.target)
        hmm_length = int(query.M)
        for hit in top_hits.included:
            sequence_id = _decode_name(hit.name)
            for domain in hit.domains.included:
                left = min(int(domain.env_from), int(domain.env_to))
                right = max(int(domain.env_from), int(domain.env_to))
                hmm_covered = int(domain.alignment.hmm_to) - int(domain.alignment.hmm_from) + 1
                hits.append(
                    MarkerHit(
                        profile_id=profile_id,
                        sequence_id=sequence_id,
                        start=left - 1,
                        end=right,
                        strand=cast(Literal["+", "-"], str(domain.strand)),
                        score=max(0.0, float(domain.score)),
                        complete=(hmm_covered / max(1, hmm_length))
                        >= policy.marker_complete_fraction,
                        target=target,
                        evidence_artifact_ids=(profile_set.hmm_artifact.object_id,),
                    )
                )
    return hits


def _verify_profile(profile_set: MarkerProfileSet) -> Path:
    artifact = profile_set.hmm_artifact
    path = Path(artifact.uri)
    if not path.is_file():
        raise OrganelleInputError(
            code="qc.artifact_missing",
            message=f"marker HMM artifact is missing: {path}",
            details={"profile_set_id": profile_set.profile_set_id, "uri": artifact.uri},
        )
    current = ArtifactRef.from_path(
        path,
        kind=artifact.kind,
        format=artifact.format,
        media_type=artifact.media_type,
    )
    if current.sha256 != profile_set.sha256 or current.size_bytes != artifact.size_bytes:
        raise OrganelleInputError(
            code="qc.artifact_digest_mismatch",
            message=f"marker HMM artifact digest mismatch: {path}",
            details={"profile_set_id": profile_set.profile_set_id, "uri": artifact.uri},
        )
    return path


def _six_frames(
    nucleotide: str,
    genetic_code: int,
) -> tuple[tuple[Literal["+", "-"], int, str], ...]:
    sequence = nucleotide.upper()
    reverse = _reverse_complement(sequence)
    frames: list[tuple[Literal["+", "-"], int, str]] = []
    strand_sources: tuple[tuple[Literal["+", "-"], str], ...] = (
        ("+", sequence),
        ("-", reverse),
    )
    for strand, source in strand_sources:
        for frame in range(3):
            usable = len(source) - frame
            usable -= usable % 3
            translated = (
                str(
                    Seq(source[frame : frame + usable]).translate(
                        table=genetic_code  # pyright: ignore[reportArgumentType]
                    )
                )
                if usable
                else ""
            )
            frames.append((strand, frame, translated.replace("*", "X")))
    return tuple(frames)


def _reverse_complement(sequence: str) -> str:
    return sequence.translate(str.maketrans("ACGTN", "TGCAN"))[::-1]


def _domain_coordinates(frame: _Frame, env_from: int, env_to: int) -> tuple[int, int]:
    translated_start = frame.frame + (env_from - 1) * 3
    translated_end = min(frame.nucleotide_length, frame.frame + env_to * 3)
    if frame.strand == "+":
        return (translated_start, translated_end)
    return (
        frame.nucleotide_length - translated_end,
        frame.nucleotide_length - translated_start,
    )


def _decode_name(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return value.decode("utf-8", errors="strict")


def _profile_target(
    profile_id: str,
    declared_target: Literal["mitochondrion", "plastid"] | None = None,
) -> str:
    if declared_target is not None:
        return declared_target
    prefix = profile_id.split(":", 1)[0].split("|", 1)[0].lower()
    if prefix in {"plastid", "chloroplast"}:
        return "plastid"
    if prefix in {"mito", "mitochondrion", "mitochondrial"}:
        return "mitochondrion"
    return ""


def _consolidate_hits(hits: list[MarkerHit]) -> tuple[MarkerHit, ...]:
    ordered = sorted(
        hits,
        key=lambda item: (
            item.profile_id,
            item.target,
            item.sequence_id,
            item.strand,
            item.start,
            item.end,
            -(item.score or 0.0),
        ),
    )
    consolidated: list[MarkerHit] = []
    for hit in ordered:
        if not consolidated:
            consolidated.append(hit)
            continue
        previous = consolidated[-1]
        same_profile = (
            previous.profile_id == hit.profile_id
            and previous.target == hit.target
            and previous.sequence_id == hit.sequence_id
            and previous.strand == hit.strand
        )
        overlaps = hit.start < previous.end and previous.start < hit.end
        if same_profile and overlaps:
            if (hit.score or 0.0) > (previous.score or 0.0):
                consolidated[-1] = hit
        else:
            consolidated.append(hit)
    return tuple(consolidated)


__all__ = ["collect_marker_evidence"]
