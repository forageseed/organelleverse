import gzip
import json
from collections.abc import Mapping
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

import organelleverse as ov
from organelleverse.assembly.contracts import (
    AssemblyAuxiliary,
    AssemblyInputPayload,
    AssemblyRequest,
    AuxiliaryContract,
    ContigInput,
    GenomeSizeEvidence,
    LongReadLibrary,
    ShortReadLibrary,
    validate_assembly_data,
)
from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.data import OrganelleData
from organelleverse.core.errors import OrganelleInputError
from organelleverse.io import read_reads


def _fastq(path: Path, name: str) -> Path:
    path.write_text(f"@{name}\nACGT\n+\n!!!!\n")
    return path


def _fasta(path: Path, name: str = "seed") -> Path:
    path.write_text(f">{name}\nACGTACGT\n")
    return path


def _assert_frozen_and_extra_forbid(model: BaseModel) -> None:
    payload = model.model_dump(mode="python", round_trip=True)
    payload["unexpected"] = True
    with pytest.raises(ValidationError) as extra:
        type(model).model_validate(payload)
    assert extra.value.errors()[0]["type"] == "extra_forbidden"

    field_name = next(iter(type(model).model_fields))
    current: object = getattr(model, field_name)
    with pytest.raises(ValidationError) as frozen:
        setattr(model, field_name, current)
    assert frozen.value.errors()[0]["type"] == "frozen_instance"


def test_read_reads_builds_content_addressed_illumina_contract(tmp_path: Path) -> None:
    read1 = _fastq(tmp_path / "sample_R1.fastq", "r1")
    read2 = _fastq(tmp_path / "sample_R2.fastq", "r2")

    data = ov.io.read_reads(
        short_libraries=(
            ShortReadLibrary(
                technology="illumina",
                layout="paired_end",
                read1=read1,
                read2=read2,
                read_length=150,
                insert_size=350,
            ),
        ),
    )

    assert type(data) is OrganelleData
    assert data.modality == "sequencing_reads"
    assert tuple(data.artifacts) == ("short_0_read1", "short_0_read2")
    payload = validate_assembly_data(data)
    assert payload.contract_version == "organelleverse.assembly-input.v1"
    assert payload.short_libraries[0].read1_artifact == "short_0_read1"
    assert (
        data.artifacts["short_0_read1"].sha256
        == ArtifactRef.from_path(read1, kind="short_read", format="fastq").sha256
    )
    assert data.dimensions == {"short_libraries": 1, "long_libraries": 0, "contig_inputs": 0}


def test_read_reads_detects_format_after_dotted_sample_name(tmp_path: Path) -> None:
    reads = tmp_path / "Malus_domestica.540Mb.fasta.gz"
    with gzip.open(reads, "wt") as handle:
        handle.write(">read\nACGT\n")

    data = read_reads(
        long_libraries=(
            LongReadLibrary(
                technology="pacbio_hifi",
                quality_state="ccs",
                reads=reads,
            ),
        )
    )

    assert data.artifacts["long_0_reads"].format == "fasta"


def test_hifi_requires_ccs_quality_state(tmp_path: Path) -> None:
    reads = _fastq(tmp_path / "sample.fastq", "hifi")

    with pytest.raises(ValidationError, match="pacbio_hifi requires quality_state='ccs'"):
        LongReadLibrary(technology="pacbio_hifi", quality_state="hq", reads=reads)


def test_paired_end_requires_read2(tmp_path: Path) -> None:
    read1 = _fastq(tmp_path / "sample_R1.fastq", "r1")

    with pytest.raises(ValidationError, match="paired_end requires read2"):
        ShortReadLibrary(
            technology="illumina",
            layout="paired_end",
            read1=read1,
            read_length=150,
            insert_size=350,
        )


def test_auxiliary_artifact_from_fetch_style_data_is_preserved(tmp_path: Path) -> None:
    reads = _fastq(tmp_path / "sample.fastq", "hifi")
    seed = ArtifactRef.from_path(_fasta(tmp_path / "seed.fasta"), kind="records", format="fasta")

    data = read_reads(
        long_libraries=(
            LongReadLibrary(
                technology="pacbio_hifi",
                quality_state="ccs",
                reads=reads,
            ),
        ),
        auxiliary=AssemblyAuxiliary(seed_fasta=seed),
    )

    assert data.artifacts["seed_fasta"] == seed
    assert validate_assembly_data(data).auxiliary.seed_fasta_artifact == "seed_fasta"


def test_validate_assembly_data_rejects_dangling_artifact_role() -> None:
    data = OrganelleData.model_validate(
        {
            "modality": "sequencing_reads",
            "payload": {
                "contract_version": "organelleverse.assembly-input.v1",
                "short_libraries": [],
                "long_libraries": [
                    {
                        "technology": "pacbio_hifi",
                        "quality_state": "ccs",
                        "reads_artifact": "missing_reads",
                    }
                ],
                "contig_inputs": [],
                "auxiliary": {},
            },
        }
    )

    with pytest.raises(OrganelleInputError) as raised:
        validate_assembly_data(data)

    assert raised.value.code == "assembly.missing_artifact_role"
    assert raised.value.as_dict()["details"]["artifact_role"] == "missing_reads"


def test_validate_assembly_data_rejects_unused_artifact_role() -> None:
    reads = ArtifactRef(
        kind="long_read",
        uri="reads.fastq",
        format="fastq",
        sha256="a" * 64,
        size_bytes=10,
    )
    extra = ArtifactRef(
        kind="reference",
        uri="extra.fasta",
        format="fasta",
        sha256="b" * 64,
        size_bytes=10,
    )
    data = OrganelleData.model_validate(
        {
            "modality": "sequencing_reads",
            "artifacts": {"reads": reads, "unreferenced": extra},
            "payload": {
                "contract_version": "organelleverse.assembly-input.v1",
                "short_libraries": [],
                "long_libraries": [
                    {
                        "technology": "pacbio_hifi",
                        "quality_state": "ccs",
                        "reads_artifact": "reads",
                    }
                ],
                "contig_inputs": [],
                "auxiliary": {},
            },
        }
    )

    with pytest.raises(OrganelleInputError) as raised:
        validate_assembly_data(data)

    assert raised.value.code == "assembly.unused_artifact_role"
    assert raised.value.as_dict()["details"]["artifact_role"] == "unreferenced"


def test_public_input_models_are_frozen_and_reject_unknown_fields(tmp_path: Path) -> None:
    short = ShortReadLibrary(
        technology="illumina",
        layout="single_end",
        read1=_fastq(tmp_path / "short.fastq", "short"),
        read_length=150,
    )
    long = LongReadLibrary(
        technology="pacbio_hifi",
        quality_state="ccs",
        reads=_fastq(tmp_path / "hifi.fastq", "hifi"),
    )
    contig = ContigInput(fasta=_fasta(tmp_path / "contig.fasta"))
    auxiliary = AssemblyAuxiliary(genome_size=500_000_000)
    data = read_reads(short_libraries=(short,))
    payload: AssemblyInputPayload = validate_assembly_data(data)

    for model in (short, long, contig, auxiliary, payload):
        _assert_frozen_and_extra_forbid(model)


def test_validate_assembly_data_rejects_wrong_modality() -> None:
    with pytest.raises(OrganelleInputError) as raised:
        validate_assembly_data(OrganelleData(modality="alignment"))

    assert raised.value.code == "assembly.unsupported_data_profile"


def test_direct_payload_cannot_disguise_ont_hq_as_hifi() -> None:
    data = OrganelleData.model_validate(
        {
            "modality": "sequencing_reads",
            "artifacts": {
                "reads": ArtifactRef(
                    kind="long_read",
                    uri="reads.fastq",
                    format="fastq",
                    sha256="a" * 64,
                    size_bytes=10,
                )
            },
            "payload": {
                "contract_version": "organelleverse.assembly-input.v1",
                "short_libraries": [],
                "long_libraries": [
                    {
                        "technology": "pacbio_hifi",
                        "quality_state": "hq",
                        "reads_artifact": "reads",
                    }
                ],
                "contig_inputs": [],
                "auxiliary": {},
            },
        }
    )

    with pytest.raises(OrganelleInputError) as raised:
        validate_assembly_data(data)

    assert raised.value.code == "assembly.unsupported_data_profile"


_REPORT_SHA256 = "a" * 64
_READS_SHA256 = "b" * 64


def _report_artifact(sha256: str = _REPORT_SHA256) -> ArtifactRef:
    return ArtifactRef(
        kind="genome_size_report",
        uri="inputs/genome_size_report.json",
        format="json",
        media_type="application/json",
        sha256=sha256,
        size_bytes=128,
    )


def _reads_artifact() -> ArtifactRef:
    return ArtifactRef(
        kind="long_read",
        uri="inputs/reads.fastq",
        format="fastq",
        sha256=_READS_SHA256,
        size_bytes=64,
    )


def _user_evidence(genome_size_bp: int = 500_000_000) -> GenomeSizeEvidence:
    return GenomeSizeEvidence(source="user", genome_size_bp=genome_size_bp)


def _ncbi_evidence(
    *,
    genome_size_bp: int = 500_000_000,
    report_sha256: str = _REPORT_SHA256,
    report_artifact_role: str = "genome_size_report",
) -> GenomeSizeEvidence:
    return GenomeSizeEvidence(
        source="ncbi_assembly",
        genome_size_bp=genome_size_bp,
        taxon_id=4530,
        scientific_name="Oryza sativa",
        assembly_accession="GCF_001433935.1",
        assembly_level="Chromosome",
        refseq_category="reference",
        report_uri=(
            "https://api.ncbi.nlm.nih.gov/datasets/v2/genome/accession/"
            "GCF_001433935.1/dataset_report"
        ),
        report_sha256=report_sha256,
        report_artifact_role=report_artifact_role,
    )


def _data_with_auxiliary(
    auxiliary: Mapping[str, object],
    *,
    report: ArtifactRef | None = None,
) -> OrganelleData:
    artifacts: dict[str, ArtifactRef] = {"long_0_reads": _reads_artifact()}
    if report is not None:
        artifacts["genome_size_report"] = report
    return OrganelleData.model_validate(
        {
            "modality": "sequencing_reads",
            "artifacts": artifacts,
            "payload": {
                "contract_version": "organelleverse.assembly-input.v1",
                "short_libraries": [],
                "long_libraries": [
                    {
                        "technology": "pacbio_hifi",
                        "quality_state": "ccs",
                        "reads_artifact": "long_0_reads",
                    }
                ],
                "contig_inputs": [],
                "auxiliary": auxiliary,
            },
        }
    )


def test_user_genome_size_evidence_is_accepted_with_matching_size() -> None:
    auxiliary = {
        "genome_size": 500_000_000,
        "genome_size_evidence": _user_evidence().model_dump(mode="json"),
    }
    payload = validate_assembly_data(_data_with_auxiliary(auxiliary))

    evidence = payload.auxiliary.genome_size_evidence
    assert evidence is not None
    assert evidence.source == "user"
    assert evidence.genome_size_bp == 500_000_000
    assert evidence.report_sha256 is None


def test_user_evidence_rejects_report_fields() -> None:
    with pytest.raises(ValidationError, match="report"):
        GenomeSizeEvidence(source="user", genome_size_bp=500_000_000, report_sha256=_REPORT_SHA256)


def test_complete_ncbi_evidence_is_accepted_with_matching_report() -> None:
    report = _report_artifact(sha256=_REPORT_SHA256)
    auxiliary = {
        "genome_size": 500_000_000,
        "genome_size_evidence": _ncbi_evidence(report_sha256=_REPORT_SHA256).model_dump(
            mode="json"
        ),
        "genome_size_report_artifact": "genome_size_report",
    }
    payload = validate_assembly_data(_data_with_auxiliary(auxiliary, report=report))

    evidence = payload.auxiliary.genome_size_evidence
    assert evidence is not None
    assert evidence.source == "ncbi_assembly"
    assert evidence.report_artifact_role == "genome_size_report"
    assert evidence.report_sha256 == report.sha256


def test_ncbi_evidence_requires_complete_metadata() -> None:
    complete = _ncbi_evidence().model_dump(mode="json")
    for missing in ("taxon_id", "assembly_accession", "assembly_level", "report_uri"):
        partial = {**complete}
        partial.pop(missing)
        with pytest.raises(ValidationError, match=missing):
            GenomeSizeEvidence.model_validate(partial)


def test_evidence_genome_size_bp_must_equal_auxiliary_genome_size() -> None:
    with pytest.raises(ValidationError, match="genome_size must equal"):
        AuxiliaryContract.model_validate(
            {
                "genome_size": 400_000_000,
                "genome_size_evidence": _user_evidence(500_000_000).model_dump(mode="json"),
            }
        )


def test_ncbi_evidence_report_artifact_role_must_be_genome_size_report() -> None:
    with pytest.raises(ValidationError, match="genome_size_report"):
        AuxiliaryContract.model_validate(
            {
                "genome_size": 500_000_000,
                "genome_size_evidence": _ncbi_evidence(
                    report_artifact_role="other_role"
                ).model_dump(mode="json"),
                "genome_size_report_artifact": "genome_size_report",
            }
        )


def test_ncbi_evidence_report_artifact_role_must_equal_report_artifact_role() -> None:
    """The auxiliary contract's report-artifact role and the evidence's claimed
    role must name the same artifact: a mismatched pair must be rejected even
    when each value is individually well-formed."""
    with pytest.raises(ValidationError, match="report_artifact_role"):
        AuxiliaryContract.model_validate(
            {
                "genome_size": 500_000_000,
                "genome_size_evidence": _ncbi_evidence(
                    report_artifact_role="genome_size_report"
                ).model_dump(mode="json"),
                "genome_size_report_artifact": "mismatched_report_role",
            }
        )


def test_ncbi_evidence_report_sha256_must_match_report_artifact() -> None:
    report = _report_artifact(sha256="c" * 64)
    auxiliary = {
        "genome_size": 500_000_000,
        "genome_size_evidence": _ncbi_evidence(report_sha256=_REPORT_SHA256).model_dump(
            mode="json"
        ),
        "genome_size_report_artifact": "genome_size_report",
    }
    with pytest.raises(OrganelleInputError) as raised:
        validate_assembly_data(_data_with_auxiliary(auxiliary, report=report))

    assert raised.value.code == "assembly.genome_size_evidence_mismatch"
    details = raised.value.as_dict()["details"]
    assert details["artifact_role"] == "genome_size_report"


def test_ncbi_evidence_requires_the_report_artifact_role() -> None:
    auxiliary = {
        "genome_size": 500_000_000,
        "genome_size_evidence": _ncbi_evidence().model_dump(mode="json"),
        "genome_size_report_artifact": "genome_size_report",
    }
    with pytest.raises(OrganelleInputError) as raised:
        validate_assembly_data(_data_with_auxiliary(auxiliary, report=None))

    assert raised.value.code == "assembly.missing_artifact_role"


def test_report_artifact_without_evidence_is_unused() -> None:
    auxiliary = {"genome_size_report_artifact": "genome_size_report"}
    with pytest.raises(OrganelleInputError) as raised:
        validate_assembly_data(_data_with_auxiliary(auxiliary, report=_report_artifact()))

    assert raised.value.code == "assembly.unused_artifact_role"


def test_optional_executable_artifacts_are_referenced_when_present(tmp_path: Path) -> None:
    canu = ArtifactRef(
        kind="executable",
        uri="bin/canu",
        format="executable",
        sha256="d" * 64,
        size_bytes=4096,
    )
    data = OrganelleData.model_validate(
        {
            "modality": "sequencing_reads",
            "artifacts": {"long_0_reads": _reads_artifact(), "canu_executable": canu},
            "payload": {
                "contract_version": "organelleverse.assembly-input.v1",
                "short_libraries": [],
                "long_libraries": [
                    {
                        "technology": "pacbio_hifi",
                        "quality_state": "ccs",
                        "reads_artifact": "long_0_reads",
                    }
                ],
                "contig_inputs": [],
                "auxiliary": {"canu_executable_artifact": "canu_executable"},
            },
        }
    )
    payload = validate_assembly_data(data)
    assert payload.auxiliary.canu_executable_artifact == "canu_executable"


def test_genome_size_evidence_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError) as raised:
        GenomeSizeEvidence.model_validate(
            {"source": "user", "genome_size_bp": 500_000_000, "unexpected": True}
        )
    assert raised.value.errors()[0]["type"] == "extra_forbidden"


@pytest.mark.parametrize(
    ("value", "fragment"),
    [
        ("500000000", "int_type"),
        (0, "greater_than"),
        (-1, "greater_than"),
        (5.5, "int_type"),
    ],
)
def test_genome_size_evidence_rejects_non_integer_or_non_positive_size(
    value: object, fragment: str
) -> None:
    with pytest.raises(ValidationError) as raised:
        GenomeSizeEvidence.model_validate({"source": "user", "genome_size_bp": value})  # type: ignore[arg-type]
    assert fragment in str(raised.value)


def test_genome_size_evidence_is_frozen_and_rejects_unknown_fields() -> None:
    _assert_frozen_and_extra_forbid(_user_evidence())


def test_read_reads_round_trips_genome_size_evidence_report_and_executables(
    tmp_path: Path,
) -> None:
    """``ov.io.read_reads`` must carry traceable genome-size evidence and the
    PMAT2 companion-executable artifacts through to the v1 contract instead of
    silently dropping them on the auxiliary boundary."""
    report_path = tmp_path / "dataset_report.json"
    report_path.write_text('{"total_sequence_length":500000000}')
    report_artifact = ArtifactRef.from_path(report_path, kind="genome_size_report", format="json")
    canu_path = tmp_path / "canu"
    canu_path.write_text("#!/bin/sh\n")
    nextdenovo_path = tmp_path / "NextDenovo"
    nextdenovo_path.write_text("#!/bin/sh\n")

    evidence = _ncbi_evidence(report_sha256=report_artifact.sha256)
    data = ov.io.read_reads(
        long_libraries=(
            LongReadLibrary(
                technology="pacbio_hifi",
                quality_state="ccs",
                reads=_fastq(tmp_path / "hifi.fastq", "hifi"),
            ),
        ),
        auxiliary=AssemblyAuxiliary(
            genome_size=500_000_000,
            genome_size_evidence=evidence,
            genome_size_report=report_path,
            canu_executable=canu_path,
            nextdenovo_executable=nextdenovo_path,
        ),
    )

    payload = validate_assembly_data(data)
    # The evidence object survives the auxiliary -> contract boundary.
    assert payload.auxiliary.genome_size_evidence == evidence
    # The NCBI report artifact is bound under its canonical role and its
    # content address still matches the evidence recorded in the contract.
    assert payload.auxiliary.genome_size_report_artifact == "genome_size_report"
    assert "genome_size_report" in data.artifacts
    assert data.artifacts["genome_size_report"].sha256 == evidence.report_sha256
    assert data.artifacts["genome_size_report"] == report_artifact
    # Both companion executables are bound under their canonical roles.
    assert payload.auxiliary.canu_executable_artifact == "canu_executable"
    assert payload.auxiliary.nextdenovo_executable_artifact == "nextdenovo_executable"
    assert data.artifacts["canu_executable"] == ArtifactRef.from_path(
        canu_path, kind="executable", format="executable"
    )
    assert data.artifacts["nextdenovo_executable"] == ArtifactRef.from_path(
        nextdenovo_path, kind="executable", format="executable"
    )


def test_assembly_request_semantic_payload_includes_evidence_not_report_paths(
    tmp_path: Path,
) -> None:
    report = _report_artifact(sha256=_REPORT_SHA256)
    evidence = _ncbi_evidence(report_sha256=_REPORT_SHA256)
    data = _data_with_auxiliary(
        {
            "genome_size": 500_000_000,
            "genome_size_evidence": evidence.model_dump(mode="json"),
            "genome_size_report_artifact": "genome_size_report",
        },
        report=report,
    )
    request = AssemblyRequest(data=data, organelle="mitochondrion")

    payload = request.semantic_payload()

    assert payload["genome_size_evidence"] == evidence.model_dump(mode="json")
    flattened = json.dumps(payload, allow_nan=False)
    # The content hash is part of scientific identity; the local report path is not.
    assert report.sha256 in flattened
    assert report.uri not in flattened
