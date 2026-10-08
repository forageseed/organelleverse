from __future__ import annotations

import random
from pathlib import Path

import pyhmmer
import pytest
from pydantic import ValidationError

from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.genome import OrganelleGenome
from organelleverse.quality_control.contracts import MarkerProfileSet, ResolvedAssemblyEvidence
from organelleverse.quality_control.markers import collect_marker_evidence
from organelleverse.quality_control.policy import QC_POLICY_V5

_PROTEIN = "M" + "ACDEFGHIKLMNPQRSTVWY" * 3
_CODONS = {
    "A": "GCT",
    "C": "TGT",
    "D": "GAT",
    "E": "GAA",
    "F": "TTT",
    "G": "GGT",
    "H": "CAT",
    "I": "ATT",
    "K": "AAA",
    "L": "CTG",
    "M": "ATG",
    "N": "AAT",
    "P": "CCT",
    "Q": "CAA",
    "R": "CGT",
    "S": "TCT",
    "T": "ACT",
    "V": "GTT",
    "W": "TGG",
    "Y": "TAT",
}


def _profile(path: Path, *, name: bytes = b"plastid:rbcL") -> ArtifactRef:
    alphabet = pyhmmer.easel.Alphabet.amino()
    msa = pyhmmer.easel.TextMSA(
        name=name,
        sequences=[
            pyhmmer.easel.TextSequence(
                name=f"reference-{index}".encode(),
                sequence=_PROTEIN.encode(),
            )
            for index in range(3)
        ],
    )
    builder = pyhmmer.plan7.Builder(alphabet)
    background = pyhmmer.plan7.Background(alphabet)
    hmm, _, _ = builder.build_msa(msa.digitize(alphabet), background)
    with path.open("wb") as handle:
        hmm.write(handle)
    return ArtifactRef.from_path(path, kind="hmm_profile", format="hmm")


def _dna_profile(path: Path, *, name: bytes, sequence: str) -> ArtifactRef:
    alphabet = pyhmmer.easel.Alphabet.dna()
    variants = [sequence]
    for offset in (37, 83):
        replacement = "A" if sequence[offset] != "A" else "C"
        variants.append(sequence[:offset] + replacement + sequence[offset + 1 :])
    msa = pyhmmer.easel.TextMSA(
        name=name,
        sequences=[
            pyhmmer.easel.TextSequence(
                name=f"reference-{index}".encode(),
                sequence=variant.encode(),
            )
            for index, variant in enumerate(variants, start=1)
        ],
    )
    builder = pyhmmer.plan7.Builder(alphabet)
    background = pyhmmer.plan7.Background(alphabet)
    hmm, _, _ = builder.build_msa(msa.digitize(alphabet), background)
    with path.open("wb") as handle:
        hmm.write(handle)
    return ArtifactRef.from_path(path, kind="hmm_profile", format="hmm")


def _resolved(fasta: Path, *, organelle: str) -> ResolvedAssemblyEvidence:
    artifact = ArtifactRef.from_path(fasta, kind="sequence", format="fasta")
    genome = OrganelleGenome(organelle=organelle, sequence=artifact)  # type: ignore[arg-type]
    return ResolvedAssemblyEvidence.model_construct(
        kind="resolved_assembly_evidence",
        source_result=None,
        run_manifest=None,
        primary_genome=genome,
        primary_fasta_path=fasta,
        assembly_graph_path=None,
        input_libraries=(),
    )


def test_marker_profile_identity_must_match_artifact_digest(tmp_path: Path) -> None:
    artifact = _profile(tmp_path / "markers.hmm")
    with pytest.raises(ValidationError, match="sha256"):
        MarkerProfileSet(
            profile_set_id="plant-organelles",
            version="1",
            hmm_artifact=artifact,
            sha256="0" * 64,
            genetic_code=1,
        )


def test_six_frame_hmm_search_records_direct_target_marker(tmp_path: Path) -> None:
    nucleotide = "".join(_CODONS[amino_acid] for amino_acid in _PROTEIN)
    fasta = tmp_path / "assembly.fa"
    fasta.write_text(f">ctg\n{nucleotide}\n")
    artifact = _profile(tmp_path / "markers.hmm")
    profiles = MarkerProfileSet(
        profile_set_id="plant-organelles",
        version="1",
        hmm_artifact=artifact,
        sha256=artifact.sha256,
        genetic_code=1,
    )

    evidence = collect_marker_evidence(
        _resolved(fasta, organelle="plastid"),
        profiles,
        QC_POLICY_V5,
    )

    hit = next(item for item in evidence.marker_hits if item.strand == "+")
    assert hit.profile_id == "plastid:rbcL"
    assert hit.sequence_id == "ctg"
    assert hit.strand == "+"
    assert hit.start == 0
    assert hit.end == len(nucleotide)
    assert hit.complete is True
    assert hit.target == "plastid"
    assert evidence.profile_set_id == "plant-organelles"
    assert evidence.profile_version == "1"
    assert evidence.profile_sha256 == artifact.sha256
    assert evidence.genetic_code == 1
    assert evidence.profiles_assessed is True
    assert evidence.expected_target_profile_count == 1
    assert evidence.complete_target_profile_count == 1
    assert evidence.target_profile_recovery_fraction == 1.0
    assert evidence.duplicate_complete_target_profile_count == 0
    assert evidence.sequence_identities[0].identity_evidence_status == "target_supported"
    assert evidence.checks[0].status == "pass"
    recovery = next(
        item for item in evidence.checks if item.check_id == "qc.target_marker_profile_recovery"
    )
    assert recovery.value == 1.0
    assert "genome completeness" in recovery.message


def test_oatk_dna_profile_bundle_labels_target_and_conflicting_hits(
    tmp_path: Path,
) -> None:
    source = random.Random(17)
    mito_sequence = "".join(source.choice("ACGT") for _ in range(600))
    plastid_sequence = "".join(source.choice("ACGT") for _ in range(600))
    fasta = tmp_path / "assembly.fa"
    fasta.write_text(f">mito_contig\n{mito_sequence}\n>plastid_insert\n{plastid_sequence}\n")
    mito_artifact = _dna_profile(
        tmp_path / "mito.fam",
        name=b"atp1",
        sequence=mito_sequence,
    )
    plastid_artifact = _dna_profile(
        tmp_path / "plastid.fam",
        name=b"rbcL",
        sequence=plastid_sequence,
    )
    profiles = (
        MarkerProfileSet(
            profile_set_id="oatkdb-embryophyta",
            version="v20230921",
            hmm_artifact=mito_artifact,
            sha256=mito_artifact.sha256,
            genetic_code=None,
            target="mitochondrion",
        ),
        MarkerProfileSet(
            profile_set_id="oatkdb-embryophyta",
            version="v20230921",
            hmm_artifact=plastid_artifact,
            sha256=plastid_artifact.sha256,
            genetic_code=None,
            target="plastid",
        ),
    )

    evidence = collect_marker_evidence(
        _resolved(fasta, organelle="mitochondrion"),
        profiles,
        QC_POLICY_V5,
    )

    hits = {(hit.profile_id, hit.sequence_id, hit.target) for hit in evidence.marker_hits}
    assert ("atp1", "mito_contig", "mitochondrion") in hits
    assert ("rbcL", "plastid_insert", "plastid") in hits
    assert evidence.profile_set_id == "oatkdb-embryophyta"
    assert evidence.profile_version == "v20230921"
    assert evidence.genetic_code is None
    assert evidence.expected_target_profile_count == 1
    assert evidence.complete_target_profile_count == 1
    assert evidence.target_profile_recovery_fraction == 1.0
    assert evidence.checks[0].status == "pass"
    assert evidence.checks[1].status == "warn"
    assert set(evidence.checks[0].evidence_artifact_ids) == {
        mito_artifact.object_id,
        plastid_artifact.object_id,
    }


def test_overlapping_same_name_profiles_preserve_both_organelle_sources(
    tmp_path: Path,
) -> None:
    source = random.Random(29)
    sequence = "".join(source.choice("ACGT") for _ in range(600))
    fasta = tmp_path / "assembly.fa"
    fasta.write_text(f">shared_region\n{sequence}\n")
    profiles = []
    for target in ("mitochondrion", "plastid"):
        artifact = _dna_profile(
            tmp_path / f"{target}.fam",
            name=b"rrn16",
            sequence=sequence,
        )
        profiles.append(
            MarkerProfileSet(
                profile_set_id="oatkdb-embryophyta",
                version="v20230921",
                hmm_artifact=artifact,
                sha256=artifact.sha256,
                target=target,
            )
        )

    evidence = collect_marker_evidence(
        _resolved(fasta, organelle="mitochondrion"),
        tuple(profiles),
        QC_POLICY_V5,
    )

    assert {
        hit.target
        for hit in evidence.marker_hits
        if hit.profile_id == "rrn16" and hit.sequence_id == "shared_region"
    } == {"mitochondrion", "plastid"}
