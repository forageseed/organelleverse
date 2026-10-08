"""Strict PMAT2 parameter contract and input-strategy tests.

Covers the parameter-contract half of the PMAT2 backend plan Task 1:

* :class:`PmatParameters` is a closed ``OperationParameterModel`` carrying every
  non-path ``autoMito`` flag with PMAT2 v2.1.5 defaults and numeric boundaries.
* :func:`validate_pmat_assembly_data` gates which sequencing-read payloads PMAT2
  may consume: long-read only, one shared profile across libraries, no type
  guessing, and only PMAT-supported auxiliary roles.
* :func:`effective_pmat_parameters` resolves the closed, recorded effective
  parameter block: correction auto/override matrix, impossible HiFi correction,
  non-plant plastid rejection, optional genome size, and the mt/pt and
  plant/animal/fungi mappings.

No adapter, argv, runtime, or service is exercised here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest
from pydantic import ValidationError

from organelleverse.assembly.contracts import (
    AssemblyRequest,
    PmatParameters,
    effective_pmat_parameters,
)
from organelleverse.assembly.data_contract import validate_pmat_assembly_data
from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.data import OrganelleData
from organelleverse.core.errors import OrganelleInputError

# Every (technology, quality_state) profile PMAT2 v2.1.5 supports.
PMAT_PROFILES: tuple[tuple[str, str], ...] = (
    ("pacbio_hifi", "ccs"),
    ("pacbio_clr", "raw"),
    ("pacbio_clr", "corrected"),
    ("ont", "raw"),
    ("ont", "corrected"),
    ("ont", "hq"),
    ("ont", "duplex"),
)

_SEQTYPE: dict[str, str] = {
    "pacbio_hifi": "hifi",
    "pacbio_clr": "clr",
    "ont": "ont",
}


# ---------------------------------------------------------------------------
# data + request builders
# ---------------------------------------------------------------------------


def _reads_artifact(tmp_path: Path, role: str) -> ArtifactRef:
    path = tmp_path / f"{role}.fastq"
    path.write_bytes(b"@r\nACGT\n+\nI\n")
    return ArtifactRef.from_path(
        path, kind="long_read", format="fastq", media_type="application/x-fastq"
    )


def _data(
    tmp_path: Path,
    libraries: tuple[tuple[str, str], ...],
    *,
    auxiliary: dict[str, object] | None = None,
    extra_artifacts: dict[str, ArtifactRef] | None = None,
) -> OrganelleData:
    """Build sequencing-read data from one or more same-profile long libraries."""
    artifacts: dict[str, ArtifactRef] = dict(extra_artifacts or {})
    long_libraries: list[dict[str, object]] = []
    for index, (technology, quality_state) in enumerate(libraries):
        role = f"long_{index}_reads"
        if role not in artifacts:
            artifacts[role] = _reads_artifact(tmp_path, role)
        long_libraries.append(
            {"technology": technology, "quality_state": quality_state, "reads_artifact": role}
        )
    payload: dict[str, object] = {
        "contract_version": "organelleverse.assembly-input.v1",
        "long_libraries": long_libraries,
    }
    if auxiliary is not None:
        payload["auxiliary"] = auxiliary
    return OrganelleData.model_validate(
        {"modality": "sequencing_reads", "artifacts": artifacts, "payload": payload}
    )


def _request(
    tmp_path: Path,
    data: OrganelleData,
    *,
    organelle: Literal["mitochondrion", "plastid"] = "mitochondrion",
    taxon_group: Literal["plant", "animal", "fungi"] = "plant",
) -> AssemblyRequest:
    return AssemblyRequest(
        data=data,
        organelle=organelle,
        taxon_group=taxon_group,
    )


# ---------------------------------------------------------------------------
# PmatParameters: closed strict model
# ---------------------------------------------------------------------------


def test_pmat_parameters_defaults_match_pmat2_v215() -> None:
    params = PmatParameters()
    assert params.backend == "pmat"
    assert params.kmer_size == 31
    assert params.correction_task == "auto"
    assert params.correction_software == "nextdenovo"
    assert params.subsample_factor == 1.0
    assert params.random_seed == 6
    assert params.long_read_break_length == 20000
    assert params.minimum_overlap_identity == 90
    assert params.minimum_overlap_length == 40
    assert params.keep_sequences_in_memory is False


@pytest.mark.parametrize(
    ("field", "good"),
    [
        ("kmer_size", 1),
        ("kmer_size", 31),
        ("subsample_factor", 0.001),
        ("subsample_factor", 1.0),
        ("random_seed", 0),
        ("long_read_break_length", 100),
        ("minimum_overlap_identity", 1),
        ("minimum_overlap_identity", 100),
        ("minimum_overlap_length", 1),
    ],
)
def test_pmat_parameters_accepts_numeric_boundaries(field: str, good: object) -> None:
    PmatParameters.model_validate({field: good})


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("kmer_size", 0),
        ("kmer_size", 32),
        ("subsample_factor", 0.0),
        ("subsample_factor", 1.5),
        ("random_seed", -1),
        ("long_read_break_length", 99),
        ("minimum_overlap_identity", 0),
        ("minimum_overlap_identity", 101),
        ("minimum_overlap_length", 0),
    ],
)
def test_pmat_parameters_rejects_out_of_range(field: str, bad: object) -> None:
    with pytest.raises(ValidationError):
        PmatParameters.model_validate({field: bad})


@pytest.mark.parametrize(
    ("field", "bad"),
    [("correction_task", "force"), ("correction_software", "flye")],
)
def test_pmat_parameters_rejects_unknown_enum(field: str, bad: object) -> None:
    with pytest.raises(ValidationError):
        PmatParameters.model_validate({field: bad})


def test_pmat_parameters_rejects_non_pmat_discriminator() -> None:
    with pytest.raises(ValidationError):
        PmatParameters(backend="oatk")  # type: ignore[arg-type]


def test_pmat_parameters_rejects_extra_args_channel() -> None:
    with pytest.raises(ValidationError) as raised:
        PmatParameters(extra_args=["--foo"])  # type: ignore[call-arg]
    assert raised.value.errors()[0]["type"] == "extra_forbidden"


def test_pmat_parameters_is_frozen() -> None:
    params = PmatParameters()
    with pytest.raises(ValidationError):
        params.kmer_size = 21


# ---------------------------------------------------------------------------
# validate_pmat_assembly_data: input strategy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("technology", "quality_state"), PMAT_PROFILES)
def test_validate_pmat_accepts_every_supported_profile(
    tmp_path: Path, technology: str, quality_state: str
) -> None:
    data = _data(tmp_path, ((technology, quality_state),))
    payload = validate_pmat_assembly_data(data)
    assert len(payload.long_libraries) == 1


@pytest.mark.parametrize(("technology", "quality_state"), PMAT_PROFILES)
def test_validate_pmat_accepts_multiple_same_profile_libraries(
    tmp_path: Path, technology: str, quality_state: str
) -> None:
    data = _data(tmp_path, ((technology, quality_state),) * 3)
    payload = validate_pmat_assembly_data(data)
    assert len(payload.long_libraries) == 3
    assert {lib.technology for lib in payload.long_libraries} == {technology}
    assert {lib.quality_state for lib in payload.long_libraries} == {quality_state}


def test_validate_pmat_rejects_mixed_quality_profiles(tmp_path: Path) -> None:
    data = _data(tmp_path, (("ont", "raw"), ("ont", "corrected")))
    with pytest.raises(OrganelleInputError) as raised:
        validate_pmat_assembly_data(data)
    assert raised.value.code == "assembly.unsupported_data_profile"


def test_validate_pmat_rejects_mixed_technologies(tmp_path: Path) -> None:
    data = _data(tmp_path, (("pacbio_hifi", "ccs"), ("ont", "raw")))
    with pytest.raises(OrganelleInputError) as raised:
        validate_pmat_assembly_data(data)
    assert raised.value.code == "assembly.unsupported_data_profile"


def test_validate_pmat_rejects_short_reads(tmp_path: Path) -> None:
    r1 = tmp_path / "r1.fastq"
    r1.write_bytes(b"@r\nACGT\n+\nI\n")
    artifact = ArtifactRef.from_path(
        r1, kind="short_read", format="fastq", media_type="application/x-fastq"
    )
    data = OrganelleData.model_validate(
        {
            "modality": "sequencing_reads",
            "artifacts": {"r1": artifact},
            "payload": {
                "contract_version": "organelleverse.assembly-input.v1",
                "short_libraries": [
                    {
                        "technology": "illumina",
                        "layout": "single_end",
                        "read1_artifact": "r1",
                        "read_length": 150,
                    }
                ],
            },
        }
    )
    with pytest.raises(OrganelleInputError) as raised:
        validate_pmat_assembly_data(data)
    assert raised.value.code == "assembly.unsupported_data_profile"


def test_validate_pmat_rejects_contig_inputs(tmp_path: Path) -> None:
    contigs = tmp_path / "c.fasta"
    contigs.write_bytes(b">c\nACGT\n")
    artifact = ArtifactRef.from_path(
        contigs, kind="contigs", format="fasta", media_type="text/x-fasta"
    )
    data = OrganelleData.model_validate(
        {
            "modality": "sequencing_reads",
            "artifacts": {"c": artifact},
            "payload": {
                "contract_version": "organelleverse.assembly-input.v1",
                "contig_inputs": [{"fasta_artifact": "c"}],
            },
        }
    )
    with pytest.raises(OrganelleInputError) as raised:
        validate_pmat_assembly_data(data)
    assert raised.value.code == "assembly.unsupported_data_profile"


def test_validate_pmat_rejects_disallowed_auxiliary(tmp_path: Path) -> None:
    seed = tmp_path / "seed.fasta"
    seed.write_bytes(b">s\nACGT\n")
    seed_artifact = ArtifactRef.from_path(
        seed, kind="sequence", format="fasta", media_type="text/x-fasta"
    )
    data = _data(
        tmp_path,
        (("ont", "raw"),),
        extra_artifacts={"seed_fasta": seed_artifact},
        auxiliary={"seed_fasta_artifact": "seed_fasta"},
    )
    with pytest.raises(OrganelleInputError) as raised:
        validate_pmat_assembly_data(data)
    assert raised.value.code == "assembly.unsupported_data_profile"


def test_validate_pmat_accepts_correction_auxiliary(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.ini"
    cfg.write_bytes(b"[correct]\n")
    cfg_artifact = ArtifactRef.from_path(cfg, kind="config", format="ini", media_type="text/plain")
    data = _data(
        tmp_path,
        (("ont", "raw"),),
        extra_artifacts={"correction_config": cfg_artifact},
        auxiliary={"correction_config_artifact": "correction_config"},
    )
    payload = validate_pmat_assembly_data(data)
    assert payload.auxiliary.correction_config_artifact == "correction_config"


def test_validate_pmat_accepts_supplied_genome_size(tmp_path: Path) -> None:
    data = _data(tmp_path, (("ont", "raw"),), auxiliary={"genome_size": 540_000_000})
    payload = validate_pmat_assembly_data(data)
    assert payload.auxiliary.genome_size == 540_000_000


def test_validate_pmat_does_not_require_genome_size(tmp_path: Path) -> None:
    data = _data(tmp_path, (("ont", "raw"),))
    payload = validate_pmat_assembly_data(data)
    assert payload.auxiliary.genome_size is None


# ---------------------------------------------------------------------------
# effective_pmat_parameters: policy resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("technology", "quality_state", "expected"),
    [
        ("pacbio_hifi", "ccs", "skip"),
        ("pacbio_clr", "raw", "run"),
        ("pacbio_clr", "corrected", "skip"),
        ("ont", "raw", "run"),
        ("ont", "corrected", "skip"),
        ("ont", "hq", "skip"),
        ("ont", "duplex", "skip"),
    ],
)
def test_correction_auto_resolves_by_profile(
    tmp_path: Path, technology: str, quality_state: str, expected: str
) -> None:
    data = _data(tmp_path, ((technology, quality_state),))
    request = _request(tmp_path, data)
    payload = validate_pmat_assembly_data(data)
    block = effective_pmat_parameters(request, payload, None)
    assert block["correction_task"] == expected
    assert block["seqtype"] == _SEQTYPE[technology]


def test_correction_explicit_skip_overrides_raw_default(tmp_path: Path) -> None:
    data = _data(tmp_path, (("ont", "raw"),))
    request = _request(tmp_path, data)
    payload = validate_pmat_assembly_data(data)
    block = effective_pmat_parameters(request, payload, PmatParameters(correction_task="skip"))
    assert block["correction_task"] == "skip"


def test_correction_explicit_run_overrides_corrected_default(tmp_path: Path) -> None:
    data = _data(tmp_path, (("ont", "corrected"),))
    request = _request(tmp_path, data)
    payload = validate_pmat_assembly_data(data)
    block = effective_pmat_parameters(request, payload, PmatParameters(correction_task="run"))
    assert block["correction_task"] == "run"


def test_explicit_hifi_correction_is_rejected(tmp_path: Path) -> None:
    data = _data(tmp_path, (("pacbio_hifi", "ccs"),))
    request = _request(tmp_path, data)
    payload = validate_pmat_assembly_data(data)
    with pytest.raises(OrganelleInputError) as raised:
        effective_pmat_parameters(request, payload, PmatParameters(correction_task="run"))
    assert raised.value.code == "assembly.pmat_hifi_correction_unsupported"


@pytest.mark.parametrize("taxon_group", ["animal", "fungi"])
def test_non_plant_plastid_is_rejected(
    tmp_path: Path, taxon_group: Literal["animal", "fungi"]
) -> None:
    data = _data(tmp_path, (("ont", "raw"),))
    request = _request(tmp_path, data, organelle="plastid", taxon_group=taxon_group)
    payload = validate_pmat_assembly_data(data)
    with pytest.raises(OrganelleInputError) as raised:
        effective_pmat_parameters(request, payload, None)
    assert raised.value.code == "assembly.pmat_non_plant_plastid_unsupported"


def test_effective_maps_organelle_and_taxon(tmp_path: Path) -> None:
    data = _data(tmp_path, (("ont", "raw"),))
    request = _request(tmp_path, data, organelle="plastid", taxon_group="plant")
    payload = validate_pmat_assembly_data(data)
    block = effective_pmat_parameters(request, payload, None)
    assert block["target"] == "pt"
    assert block["taxon"] == 0


@pytest.mark.parametrize(("taxon_group", "taxon"), [("animal", 1), ("fungi", 2)])
def test_effective_maps_animal_and_fungi_taxa(
    tmp_path: Path, taxon_group: Literal["animal", "fungi"], taxon: int
) -> None:
    data = _data(tmp_path, (("ont", "raw"),))
    request = _request(tmp_path, data, taxon_group=taxon_group)
    payload = validate_pmat_assembly_data(data)
    block = effective_pmat_parameters(request, payload, None)
    assert block["target"] == "mt"
    assert block["taxon"] == taxon


def test_effective_records_supplied_genome_size(tmp_path: Path) -> None:
    data = _data(tmp_path, (("ont", "raw"),), auxiliary={"genome_size": 540_000_000})
    request = _request(tmp_path, data)
    payload = validate_pmat_assembly_data(data)
    block = effective_pmat_parameters(request, payload, None)
    assert block["genome_size"] == 540_000_000


def test_effective_records_absent_genome_size_as_none(tmp_path: Path) -> None:
    data = _data(tmp_path, (("ont", "raw"),))
    request = _request(tmp_path, data)
    payload = validate_pmat_assembly_data(data)
    block = effective_pmat_parameters(request, payload, None)
    assert block["genome_size"] is None


def test_effective_records_correction_software_selection(tmp_path: Path) -> None:
    data = _data(tmp_path, (("ont", "raw"),))
    request = _request(tmp_path, data)
    payload = validate_pmat_assembly_data(data)
    default_block = effective_pmat_parameters(request, payload, None)
    assert default_block["correction_software"] == "nextdenovo"
    canu_block = effective_pmat_parameters(
        request, payload, PmatParameters(correction_software="canu")
    )
    assert canu_block["correction_software"] == "canu"
    assert canu_block["correction_task"] == "run"


def test_effective_block_is_closed_and_path_free(tmp_path: Path) -> None:
    data = _data(tmp_path, (("ont", "raw"),), auxiliary={"genome_size": 540_000_000})
    request = _request(tmp_path, data)
    payload = validate_pmat_assembly_data(data)
    block = effective_pmat_parameters(request, payload, None)
    assert set(block) == {
        "backend",
        "kmer_size",
        "correction_task",
        "correction_software",
        "subsample_factor",
        "random_seed",
        "long_read_break_length",
        "minimum_overlap_identity",
        "minimum_overlap_length",
        "keep_sequences_in_memory",
        "seqtype",
        "target",
        "taxon",
        "genome_size",
    }
