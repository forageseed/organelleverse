"""Explicit read-QC bindings and small, hand-checkable FASTQ inputs.

Only statistics has an exact serialized bundle fixture: filtering creates
managed paths. Its fixture inputs are also exercised by the domain tests,
including a real-fastp integration test; failures are never replaced by mocks
in real-data validation.
"""

from organelleverse.capabilities.adapters import FixtureCase, FixtureFile
from organelleverse.capabilities.adapters.variation import NamedParameterOverride, ParameterOverride
from organelleverse.operations.spec import ParameterCodec, ResultCodec, SideEffect

_READS = FixtureFile(
    relative_path="reads.fastq",
    content="@short\nACGT\n+\nIIII\n@long\nACGTAC\n+\n555555\n@low\nAC\n+\n!!\n",
)

OVERRIDES = {
    "qc." + name: NamedParameterOverride(
        parameters=tuple(
            ParameterOverride(name=parameter, codec=codec) for parameter, codec in parameters
        ),
        result_codec=ResultCodec.CANONICAL,
        result_key=None,
        side_effects=effects,
    )
    for name, parameters, effects in (
        ("read_statistics", (("reads", ParameterCodec.PATH),), (SideEffect.READ_FILES,)),
        (
            "filter_long_reads",
            (
                ("reads", ParameterCodec.PATH),
                ("min_length", ParameterCodec.JSON),
                ("min_mean_quality", ParameterCodec.JSON),
                ("target_bases", ParameterCodec.JSON),
            ),
            (SideEffect.READ_FILES, SideEffect.WRITE_FILES),
        ),
        (
            "filter_short_reads",
            (
                ("reads", ParameterCodec.PATH),
                ("reads2", ParameterCodec.PATH),
                ("min_length", ParameterCodec.JSON),
                ("min_mean_quality", ParameterCodec.JSON),
                ("threads", ParameterCodec.JSON),
                ("adapter_sequence", ParameterCodec.JSON),
                ("adapter_sequence_r2", ParameterCodec.JSON),
            ),
            (SideEffect.READ_FILES, SideEffect.WRITE_FILES, SideEffect.SUBPROCESS),
        ),
    )
}
FIXTURES = {
    "qc.read_statistics": (
        FixtureCase(case="basic", files=(_READS,), parameters={"reads": "reads.fastq"}),
    ),
    "qc.filter_long_reads": (
        FixtureCase(
            case="basic",
            files=(_READS,),
            parameters={
                "reads": "reads.fastq",
                "min_length": 3,
                "min_mean_quality": 10.0,
                "target_bases": 5,
            },
        ),
    ),
}

OVERRIDES["qc.compare_assembly_to_reference"] = NamedParameterOverride(
    parameters=(
        ParameterOverride(name="input_fasta", codec=ParameterCodec.PATH),
        ParameterOverride(name="reference_fasta", codec=ParameterCodec.PATH),
        ParameterOverride(name="organelle", codec=ParameterCodec.JSON),
    ),
    result_codec=ResultCodec.CANONICAL,
    result_key=None,
    side_effects=(SideEffect.READ_FILES,),
)


def _reference_fixture():
    # Deterministic nonrepetitive DNA with an exact reverse-complement candidate.
    import random

    from Bio.Seq import reverse_complement

    sequence = "".join(random.Random(812).choices("ACGT", k=3000))
    return FixtureCase(
        case="basic",
        files=(
            FixtureFile(relative_path="reference.fa", content=f">reference\n{sequence}\n"),
            FixtureFile(
                relative_path="candidate.fa",
                content=f">candidate\n{reverse_complement(sequence)}\n",
            ),
        ),
        parameters={"input_fasta": "candidate.fa", "reference_fasta": "reference.fa"},
    )


FIXTURES["qc.compare_assembly_to_reference"] = (_reference_fixture(),)
