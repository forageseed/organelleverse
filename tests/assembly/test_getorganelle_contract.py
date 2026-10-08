"""Strict GetOrganelle parameter contract and short-read input-strategy tests.

Covers the GetOrganelle backend plan Task 1:

* :class:`GetOrganelleParameters` is a closed ``OperationParameterModel`` carrying
  every non-path GetOrganelle option as an explicit, schema-faithful field with
  numeric boundaries and string-valued ``Literal`` modes. No raw argv channel,
  no dependency path field, no schema hooks, no custom validators.
* :func:`validate_getorganelle_assembly_data` gates which sequencing-read payloads
  GetOrganelle may consume: exactly one Illumina short-read library (paired or
  single end), no long reads, no contigs, and only the four artifact-backed
  auxiliary roles ``seed_fasta``, ``anti_seed``, ``label_genes``, ``exclude_genes``.
* :func:`effective_backend_parameters` resolves the closed, path-free effective
  parameter block for backend ``getorganelle``: the organelle/taxon target mapping
  plus the cross-field constraints (single memory preset, clean prefix/option
  strings, unique odd SPAdes kmers, expected_min <= expected_max).

No adapter, argv, runtime, or service is exercised here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest
from pydantic import ValidationError

from organelleverse.assembly.contracts import (
    GetOrganelleParameters,
    effective_backend_parameters,
)
from organelleverse.assembly.data_contract import validate_getorganelle_assembly_data
from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.data import OrganelleData
from organelleverse.core.errors import OrganelleInputError

# Every (organelle, taxon_group) -> GetOrganelle organelle-mode target.
_GETORGANELLE_ORGANELLE = Literal["mitochondrion", "plastid"]
_GETORGANELLE_TAXON = Literal["plant", "animal", "fungi"]
GETORGANELLE_TARGETS: dict[tuple[_GETORGANELLE_ORGANELLE, _GETORGANELLE_TAXON], str] = {
    ("plastid", "plant"): "embplant_pt",
    ("plastid", "animal"): "other_pt",
    ("plastid", "fungi"): "other_pt",
    ("mitochondrion", "plant"): "embplant_mt",
    ("mitochondrion", "animal"): "animal_mt",
    ("mitochondrion", "fungi"): "fungus_mt",
}

_AUXILIARY_ROLES: tuple[tuple[str, str], ...] = (
    ("seed_fasta_artifact", "seed_fasta"),
    ("anti_seed_artifact", "anti_seed"),
    ("label_genes_artifact", "label_genes"),
    ("exclude_genes_artifact", "exclude_genes"),
)


# ---------------------------------------------------------------------------
# data + request builders
# ---------------------------------------------------------------------------


def _short_artifact(tmp_path: Path, role: str) -> ArtifactRef:
    path = tmp_path / f"{role}.fastq"
    path.write_bytes(b"@r\nACGT\n+\nI\n")
    return ArtifactRef.from_path(
        path, kind="short_read", format="fastq", media_type="application/x-fastq"
    )


def _seq_artifact(tmp_path: Path, role: str) -> ArtifactRef:
    path = tmp_path / f"{role}.fasta"
    path.write_bytes(b">g\nACGT\n")
    return ArtifactRef.from_path(path, kind="sequence", format="fasta", media_type="text/x-fasta")


def _short_data(
    tmp_path: Path,
    *,
    layout: Literal["paired_end", "single_end"] = "paired_end",
    libraries: int = 1,
    auxiliary: dict[str, object] | None = None,
    extra_artifacts: dict[str, ArtifactRef] | None = None,
    long_libraries: int = 0,
    contig_inputs: int = 0,
) -> OrganelleData:
    """Build Illumina short-read sequencing data for GetOrganelle."""
    artifacts: dict[str, ArtifactRef] = dict(extra_artifacts or {})
    short_libraries: list[dict[str, object]] = []
    for index in range(libraries):
        r1_role = f"r1_{index}"
        if r1_role not in artifacts:
            artifacts[r1_role] = _short_artifact(tmp_path, r1_role)
        library: dict[str, object] = {
            "technology": "illumina",
            "layout": layout,
            "read1_artifact": r1_role,
            "read_length": 150,
        }
        if layout == "paired_end":
            r2_role = f"r2_{index}"
            if r2_role not in artifacts:
                artifacts[r2_role] = _short_artifact(tmp_path, r2_role)
            library["read2_artifact"] = r2_role
        short_libraries.append(library)
    for index in range(long_libraries):
        role = f"long_{index}_reads"
        if role not in artifacts:
            path = tmp_path / f"{role}.fastq"
            path.write_bytes(b"@r\nACGT\n+\nI\n")
            artifacts[role] = ArtifactRef.from_path(
                path, kind="long_read", format="fastq", media_type="application/x-fastq"
            )
    long_payload = [
        {"technology": "ont", "quality_state": "raw", "reads_artifact": f"long_{index}_reads"}
        for index in range(long_libraries)
    ]
    for index in range(contig_inputs):
        role = f"contig_{index}"
        if role not in artifacts:
            path = tmp_path / f"{role}.fasta"
            path.write_bytes(b">c\nACGT\n")
            artifacts[role] = ArtifactRef.from_path(
                path, kind="contigs", format="fasta", media_type="text/x-fasta"
            )
    contig_payload = [{"fasta_artifact": f"contig_{index}"} for index in range(contig_inputs)]
    payload: dict[str, object] = {
        "contract_version": "organelleverse.assembly-input.v1",
        "short_libraries": short_libraries,
        "long_libraries": long_payload,
        "contig_inputs": contig_payload,
    }
    if auxiliary is not None:
        payload["auxiliary"] = auxiliary
    return OrganelleData.model_validate(
        {"modality": "sequencing_reads", "artifacts": artifacts, "payload": payload}
    )


def _long_only_data(tmp_path: Path) -> OrganelleData:
    return _short_data(tmp_path, libraries=0, long_libraries=1)


# ---------------------------------------------------------------------------
# GetOrganelleParameters: closed strict model
# ---------------------------------------------------------------------------


def test_getorganelle_parameters_defaults_omit_upstream_options() -> None:
    params = GetOrganelleParameters()
    assert params.backend == "getorganelle"
    # ``None`` means the upstream option is omitted so GetOrganelle presets apply.
    for field in (
        "max_reads",
        "reduce_reads_for_coverage",
        "max_ignore_percent",
        "phred_offset",
        "min_quality_score",
        "output_prefix",
        "word_size",
        "pregroup_word_size",
        "max_rounds",
        "max_words",
        "jump_step",
        "mesh_size",
        "bowtie2_options",
        "target_genome_size",
        "max_extending_length",
        "spades_kmers",
        "spades_options",
        "ignore_kmer",
        "disentangle_depth_factor",
        "contamination_depth",
        "contamination_similarity",
        "degenerate_depth",
        "degenerate_similarity",
        "disentangle_time_limit",
        "expected_max_size",
        "expected_min_size",
        "max_paths",
        "pregrouped_reads",
        "remove_duplicates",
        "flush_step",
        "random_seed",
    ):
        assert getattr(params, field) is None, field
    # Boolean flags default to ``False`` (option absent, not inverted).
    for field in (
        "output_per_round",
        "zip_files",
        "keep_temp",
        "fast",
        "memory_save",
        "memory_unlimited",
        "larger_auto_word_size",
        "no_spades",
        "no_degenerate",
        "reverse_lsc",
        "index_in_memory",
        "verbose",
    ):
        assert getattr(params, field) is False, field


@pytest.mark.parametrize(
    ("field", "good"),
    [
        ("max_reads", 1),
        ("reduce_reads_for_coverage", 10.0001),
        ("reduce_reads_for_coverage", "inf"),
        ("max_ignore_percent", 0),
        ("max_ignore_percent", 1),
        ("phred_offset", 33),
        ("phred_offset", 64),
        ("word_size", 0.0001),
        ("pregroup_word_size", 21),
        ("max_rounds", 2),
        ("max_rounds", "inf"),
        ("max_words", 1),
        ("jump_step", 1),
        ("mesh_size", 1),
        ("target_genome_size", 1),
        ("max_extending_length", 0),
        ("max_extending_length", "auto"),
        ("max_extending_length", "inf"),
        ("spades_kmers", (21, 33, 55)),
        ("spades_kmers", (31,)),
        ("ignore_kmer", 0),
        ("disentangle_depth_factor", 0.1),
        ("contamination_depth", 0.1),
        ("contamination_similarity", 0),
        ("contamination_similarity", 1),
        ("degenerate_depth", 0.1),
        ("degenerate_similarity", 0.5),
        ("disentangle_time_limit", 1),
        ("expected_max_size", 1),
        ("expected_min_size", 1),
        ("max_paths", 1),
        ("pregrouped_reads", 0),
        ("remove_duplicates", 0),
        ("flush_step", 1),
        ("flush_step", "inf"),
        ("random_seed", 0),
        ("output_prefix", "sample"),
        ("bowtie2_options", "--very-sensitive"),
        ("spades_options", "--careful"),
        ("min_quality_score", 2),
    ],
)
def test_getorganelle_parameters_accepts_numeric_boundaries(field: str, good: object) -> None:
    GetOrganelleParameters.model_validate({field: good})


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("max_reads", 0),
        ("reduce_reads_for_coverage", 10),
        ("max_ignore_percent", -0.1),
        ("max_ignore_percent", 1.1),
        ("phred_offset", 32),
        ("phred_offset", 34),
        ("phred_offset", 50),
        ("phred_offset", 63),
        ("phred_offset", 65),
        ("word_size", 0),
        ("pregroup_word_size", 0),
        ("max_rounds", 1),
        ("max_words", 0),
        ("jump_step", 0),
        ("mesh_size", 0),
        ("target_genome_size", 0),
        ("max_extending_length", -1),
        ("ignore_kmer", -1),
        ("disentangle_depth_factor", 0),
        ("contamination_depth", 0),
        ("contamination_similarity", -0.1),
        ("contamination_similarity", 1.1),
        ("degenerate_depth", 0),
        ("degenerate_similarity", -0.1),
        ("disentangle_time_limit", 0),
        ("expected_max_size", 0),
        ("expected_min_size", 0),
        ("max_paths", 0),
        ("pregrouped_reads", -1),
        ("remove_duplicates", -1),
        ("flush_step", 0),
    ],
)
def test_getorganelle_parameters_rejects_out_of_range(field: str, bad: object) -> None:
    with pytest.raises(ValidationError):
        GetOrganelleParameters.model_validate({field: bad})


def test_getorganelle_parameters_rejects_non_getorganelle_discriminator() -> None:
    with pytest.raises(ValidationError):
        GetOrganelleParameters(backend="oatk")  # type: ignore[arg-type]


def test_getorganelle_parameters_rejects_extra_args_channel() -> None:
    with pytest.raises(ValidationError) as raised:
        GetOrganelleParameters(extra_args=["--foo"])  # type: ignore[call-arg]
    assert raised.value.errors()[0]["type"] == "extra_forbidden"


def test_getorganelle_parameters_rejects_string_coercion_for_integer() -> None:
    with pytest.raises(ValidationError):
        GetOrganelleParameters.model_validate({"max_reads": "15"})


def test_getorganelle_parameters_rejects_dependency_path_field() -> None:
    with pytest.raises(ValidationError):
        GetOrganelleParameters(dependency_path="/opt/getorganelle")  # type: ignore[call-arg]


def test_getorganelle_parameters_is_frozen() -> None:
    params = GetOrganelleParameters()
    with pytest.raises(ValidationError):
        params.max_reads = 100


def test_getorganelle_parameters_schema_is_closed_with_getorganelle_discriminator() -> None:
    schema = GetOrganelleParameters.model_json_schema()
    assert schema["additionalProperties"] is False
    properties = schema["properties"]
    assert properties["backend"]["const"] == "getorganelle"
    # No free-form argv, raw extra-args, or dependency path channel is exposed.
    forbidden = {
        "extra_args",
        "argv",
        "raw_args",
        "extra_argv",
        "dependency_path",
        "executable_artifact",
    }
    assert not (forbidden & set(properties))


def test_getorganelle_parameters_schema_phred_offset_admits_only_33_and_64() -> None:
    schema = GetOrganelleParameters.model_json_schema()
    assert schema["additionalProperties"] is False
    phred = schema["properties"]["phred_offset"]
    branches = phred.get("anyOf") or [phred]
    admitted: set[int] = set()
    for branch in branches:
        if branch.get("type") != "integer":
            continue
        # The public contract admits exactly the two real phred offsets. Each
        # integer branch must therefore be an exact singleton; a range branch
        # (minimum < maximum) would admit other offsets and is rejected here.
        if "const" in branch:
            admitted.add(branch["const"])
        elif "enum" in branch:
            admitted.update(branch["enum"])
        else:
            assert branch.get("minimum") is not None and branch.get("maximum") is not None, branch
            assert branch["minimum"] == branch["maximum"], branch
            admitted.add(branch["minimum"])
    assert admitted == {33, 64}


# ---------------------------------------------------------------------------
# validate_getorganelle_assembly_data: input strategy
# ---------------------------------------------------------------------------


def test_validate_getorganelle_accepts_paired_end_library(tmp_path: Path) -> None:
    pe_data = _short_data(tmp_path, layout="paired_end")
    assert validate_getorganelle_assembly_data(pe_data).short_libraries[0].layout == "paired_end"


def test_validate_getorganelle_accepts_single_end_library(tmp_path: Path) -> None:
    se_data = _short_data(tmp_path, layout="single_end")
    assert validate_getorganelle_assembly_data(se_data).short_libraries[0].layout == "single_end"


def test_validate_getorganelle_rejects_no_short_library(tmp_path: Path) -> None:
    with pytest.raises(OrganelleInputError) as raised:
        validate_getorganelle_assembly_data(_long_only_data(tmp_path))
    assert raised.value.code == "assembly.unsupported_data_profile"


def test_validate_getorganelle_rejects_more_than_one_short_library(tmp_path: Path) -> None:
    data = _short_data(tmp_path, libraries=2)
    with pytest.raises(OrganelleInputError) as raised:
        validate_getorganelle_assembly_data(data)
    assert raised.value.code == "assembly.unsupported_data_profile"


def test_validate_getorganelle_rejects_any_long_library(tmp_path: Path) -> None:
    data = _short_data(tmp_path, libraries=1, long_libraries=1)
    with pytest.raises(OrganelleInputError) as raised:
        validate_getorganelle_assembly_data(data)
    assert raised.value.code == "assembly.unsupported_data_profile"


def test_validate_getorganelle_rejects_contig_inputs(tmp_path: Path) -> None:
    data = _short_data(tmp_path, libraries=1, contig_inputs=1)
    with pytest.raises(OrganelleInputError) as raised:
        validate_getorganelle_assembly_data(data)
    assert raised.value.code == "assembly.unsupported_data_profile"


@pytest.mark.parametrize(
    ("auxiliary", "artifact_role"),
    [
        ({"reference_fasta_artifact": "ref"}, "ref"),
        ({"genome_size": 540_000_000}, None),
        ({"correction_config_artifact": "cfg"}, "cfg"),
    ],
)
def test_validate_getorganelle_rejects_disallowed_auxiliary(
    tmp_path: Path, auxiliary: dict[str, object], artifact_role: str | None
) -> None:
    extra: dict[str, ArtifactRef] = {}
    if artifact_role is not None:
        path = tmp_path / f"{artifact_role}.fasta"
        path.write_bytes(b">x\nACGT\n")
        extra[artifact_role] = ArtifactRef.from_path(
            path, kind="sequence", format="fasta", media_type="text/x-fasta"
        )
    data = _short_data(tmp_path, extra_artifacts=extra, auxiliary=auxiliary)
    with pytest.raises(OrganelleInputError) as raised:
        validate_getorganelle_assembly_data(data)
    assert raised.value.code == "assembly.unsupported_data_profile"


@pytest.mark.parametrize(("field", "role"), _AUXILIARY_ROLES)
def test_validate_getorganelle_accepts_each_auxiliary_role(
    tmp_path: Path, field: str, role: str
) -> None:
    data = _short_data(
        tmp_path,
        extra_artifacts={role: _seq_artifact(tmp_path, role)},
        auxiliary={field: role},
    )
    payload = validate_getorganelle_assembly_data(data)
    assert getattr(payload.auxiliary, field) == role


@pytest.mark.parametrize(("field", "role"), _AUXILIARY_ROLES)
def test_validate_getorganelle_rejects_missing_auxiliary_role(
    tmp_path: Path, field: str, role: str
) -> None:
    data = _short_data(tmp_path, auxiliary={field: "ghost_role"})
    with pytest.raises(OrganelleInputError) as raised:
        validate_getorganelle_assembly_data(data)
    assert raised.value.code == "assembly.missing_artifact_role"
    assert role  # sanity: keep the parameter referenced


# ---------------------------------------------------------------------------
# effective_backend_parameters: target mapping + cross-field policy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("organelle_taxon", "target"),
    sorted(GETORGANELLE_TARGETS.items()),
)
def test_effective_getorganelle_maps_organelle_and_taxon(
    organelle_taxon: tuple[_GETORGANELLE_ORGANELLE, _GETORGANELLE_TAXON], target: str
) -> None:
    organelle, taxon_group = organelle_taxon
    block = effective_backend_parameters(
        "getorganelle", None, organelle=organelle, taxon_group=taxon_group
    )
    assert block["target"] == target


def test_effective_getorganelle_block_is_closed_and_path_free() -> None:
    block = effective_backend_parameters(
        "getorganelle",
        None,
        organelle="mitochondrion",
        taxon_group="plant",
    )
    expected_keys = set(GetOrganelleParameters().model_dump(mode="json")) | {"target"}
    assert set(block) == expected_keys
    # No raw argv or dependency path leaks into the recorded effective block.
    assert "extra_args" not in block
    assert "argv" not in block
    assert "dependency_path" not in block


def test_effective_getorganelle_records_provided_overrides() -> None:
    block = effective_backend_parameters(
        "getorganelle",
        GetOrganelleParameters(max_reads=42, word_size=21.0),
        organelle="plastid",
        taxon_group="plant",
    )
    assert block["max_reads"] == 42
    assert block["word_size"] == 21.0
    assert block["target"] == "embplant_pt"


def test_effective_getorganelle_rejects_conflicting_memory_presets() -> None:
    with pytest.raises(OrganelleInputError) as raised:
        effective_backend_parameters(
            "getorganelle",
            GetOrganelleParameters(fast=True, memory_save=True),
            organelle="mitochondrion",
            taxon_group="plant",
        )
    assert raised.value.code == "assembly.getorganelle_conflicting_memory_preset"


def test_effective_getorganelle_accepts_single_memory_preset() -> None:
    block = effective_backend_parameters(
        "getorganelle",
        GetOrganelleParameters(memory_unlimited=True),
        organelle="mitochondrion",
        taxon_group="plant",
    )
    assert block["memory_unlimited"] is True
    assert block["fast"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("output_prefix", ""),
        ("output_prefix", "bad\nname"),
        ("output_prefix", "bad\0name"),
        ("bowtie2_options", ""),
        ("bowtie2_options", "--x\nrm"),
        ("spades_options", ""),
        ("spades_options", "--x\0rm"),
    ],
)
def test_effective_getorganelle_rejects_unclean_option_strings(field: str, value: str) -> None:
    with pytest.raises(OrganelleInputError) as raised:
        effective_backend_parameters(
            "getorganelle",
            GetOrganelleParameters.model_validate({field: value}),
            organelle="mitochondrion",
            taxon_group="plant",
        )
    assert raised.value.code == "assembly.getorganelle_invalid_option_string"


@pytest.mark.parametrize(
    ("kmers", "reason"),
    [
        ((21, 21), "duplicate"),
        ((21, 33, 33), "duplicate"),
        ((21, 22), "even"),
        ((31, 32, 33), "even"),
    ],
)
def test_effective_getorganelle_rejects_invalid_spades_kmers(
    kmers: tuple[int, ...], reason: str
) -> None:
    with pytest.raises(OrganelleInputError) as raised:
        effective_backend_parameters(
            "getorganelle",
            GetOrganelleParameters(spades_kmers=kmers),
            organelle="mitochondrion",
            taxon_group="plant",
        )
    assert raised.value.code == "assembly.getorganelle_invalid_spades_kmers"
    assert reason  # keep parameter referenced


def test_effective_getorganelle_accepts_unique_odd_spades_kmers() -> None:
    block = effective_backend_parameters(
        "getorganelle",
        GetOrganelleParameters(spades_kmers=(21, 33, 55, 77, 99, 127)),
        organelle="mitochondrion",
        taxon_group="plant",
    )
    assert block["spades_kmers"] == [21, 33, 55, 77, 99, 127]


def test_effective_getorganelle_rejects_inverted_expected_size_range() -> None:
    with pytest.raises(OrganelleInputError) as raised:
        effective_backend_parameters(
            "getorganelle",
            GetOrganelleParameters(expected_min_size=200_000, expected_max_size=100_000),
            organelle="mitochondrion",
            taxon_group="plant",
        )
    assert raised.value.code == "assembly.getorganelle_invalid_size_range"


def test_effective_getorganelle_accepts_ordered_expected_size_range() -> None:
    block = effective_backend_parameters(
        "getorganelle",
        GetOrganelleParameters(expected_min_size=100_000, expected_max_size=200_000),
        organelle="mitochondrion",
        taxon_group="plant",
    )
    assert block["expected_min_size"] == 100_000
    assert block["expected_max_size"] == 200_000


def test_effective_getorganelle_requires_organelle_and_taxon_group() -> None:
    with pytest.raises(ValueError):
        effective_backend_parameters("getorganelle", None)


def test_effective_getorganelle_rejects_mismatched_backend_parameters() -> None:
    from organelleverse.assembly.contracts import OatkParameters

    with pytest.raises(ValueError):
        effective_backend_parameters(
            "getorganelle",
            OatkParameters(),  # type: ignore[arg-type]
            organelle="mitochondrion",
            taxon_group="plant",
        )
