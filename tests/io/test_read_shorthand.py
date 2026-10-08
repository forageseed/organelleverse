"""ov.read("reads.fastq.gz", technology=...): the short way to declare a reads file."""

from __future__ import annotations

import gzip
from pathlib import Path

import pytest

import organelleverse as ov
from organelleverse.assembly import LongReadLibrary, ShortReadLibrary
from organelleverse.core.errors import OrganelleInputError


def _fastq(path: Path, names: list[str], length: int) -> Path:
    body = "".join(f"@{name}\n{'A' * length}\n+\n{'I' * length}\n" for name in names)
    if path.name.endswith(".gz"):
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write(body)
    else:
        path.write_text(body, encoding="utf-8")
    return path


def _hifi(tmp_path: Path, name: str = "hifi.fastq") -> Path:
    movie = "m64011_190830_220126"
    return _fastq(tmp_path / name, [f"{movie}/{i}/ccs" for i in range(5)], 5000)


def _explicit(library: LongReadLibrary | ShortReadLibrary) -> object:
    if isinstance(library, LongReadLibrary):
        return ov.read(long_libraries=(library,))
    return ov.read(short_libraries=(library,))


def test_hifi_shorthand_equals_the_structured_library(tmp_path: Path) -> None:
    reads = _hifi(tmp_path)

    short = ov.read(reads, technology="hifi")
    explicit = _explicit(
        LongReadLibrary(technology="pacbio_hifi", quality_state="ccs", reads=reads)
    )

    assert short.payload == explicit.payload  # type: ignore[attr-defined]
    assert short.artifacts == explicit.artifacts  # type: ignore[attr-defined]


def test_full_technology_names_are_accepted(tmp_path: Path) -> None:
    reads = _hifi(tmp_path)

    assert ov.read(reads, technology="pacbio_hifi").payload == ov.read(  # type: ignore[attr-defined]
        reads, technology="hifi"
    ).payload


def test_ont_and_clr_defaults_and_quality_override(tmp_path: Path) -> None:
    reads = _fastq(tmp_path / "long.fastq.gz", ["r1", "r2"], 3000)

    for tech, tech_full, state in (("ont", "ont", "raw"), ("clr", "pacbio_clr", "raw")):
        short = ov.read(reads, technology=tech)
        explicit = _explicit(LongReadLibrary(technology=tech_full, quality_state=state, reads=reads))  # type: ignore[arg-type]
        assert short.payload == explicit.payload  # type: ignore[attr-defined]

    duplex = ov.read(reads, technology="ont", quality_state="duplex")
    assert duplex.payload == _explicit(  # type: ignore[attr-defined]
        LongReadLibrary(technology="ont", quality_state="duplex", reads=reads)
    ).payload


def test_quality_state_is_checked_against_the_technology(tmp_path: Path) -> None:
    reads = _hifi(tmp_path)

    with pytest.raises(Exception, match="ccs"):
        ov.read(reads, technology="hifi", quality_state="raw")


def test_illumina_single_and_paired(tmp_path: Path) -> None:
    r1 = _fastq(tmp_path / "a_1.fastq", ["x/1", "y/1"], 150)
    r2 = _fastq(tmp_path / "a_2.fastq", ["x/2", "y/2"], 150)

    single = ov.read(r1, technology="illumina")
    paired = ov.read([r1, r2], technology="illumina")

    assert single.payload == _explicit(  # type: ignore[attr-defined]
        ShortReadLibrary(technology="illumina", layout="single_end", read1=r1, read_length=150)
    ).payload
    assert paired.payload == _explicit(  # type: ignore[attr-defined]
        ShortReadLibrary(
            technology="illumina", layout="paired_end", read1=r1, read2=r2, read_length=150
        )
    ).payload
    assert ov.read(r1, technology="illumina", read_length=151).payload != single.payload  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("names", "length", "expected"),
    [
        ([f"m64011_190830_220126/{i}/ccs" for i in range(4)], 8000, "hifi"),
        ([f"m64011_190830_220126/{i}/ccs/fwd" for i in range(4)], 8000, "hifi"),
        ([f"m54006_160520_172103/{i}/0_9000" for i in range(4)], 9000, "clr"),
        (
            [
                f"0a1b2c3d-1111-2222-3333-44445555666{i} runid=abc read={i} ch=7"
                for i in range(4)
            ],
            9000,
            "ont",
        ),
        ([f"A00123:45:HXXXX:1:1101:{i}:1000 1:N:0:ACGT" for i in range(4)], 150, "illumina"),
    ],
)
def test_auto_identifies_the_platform_from_the_file(
    tmp_path: Path, names: list[str], length: int, expected: str
) -> None:
    reads = _fastq(tmp_path / "auto.fastq.gz", names, length)

    auto = ov.read(reads, technology="auto")
    chosen = ov.read(reads, technology=expected)

    assert auto.payload == chosen.payload  # type: ignore[attr-defined]


def test_auto_refuses_when_the_names_prove_nothing(tmp_path: Path) -> None:
    reads = _fastq(tmp_path / "unknown.fastq", ["read1", "read2", "read3"], 8000)

    with pytest.raises(OrganelleInputError) as caught:
        ov.read(reads, technology="auto")

    assert caught.value.code == "input.technology_not_identifiable"


def test_auto_refuses_mixed_platforms(tmp_path: Path) -> None:
    reads = _fastq(
        tmp_path / "mixed.fastq",
        ["m64011_190830_220126/1/ccs", "0a1b2c3d-1111-2222-3333-444455556666 runid=x"],
        8000,
    )

    with pytest.raises(OrganelleInputError) as caught:
        ov.read(reads, technology="auto")

    assert caught.value.code == "input.technology_not_identifiable"


def test_auto_reads_fasta_too(tmp_path: Path) -> None:
    fasta = tmp_path / "reads.fasta"
    fasta.write_text("".join(f">m64011_190830_220126/{i}/ccs\n{'A' * 4000}\n" for i in range(3)))

    assert ov.read(fasta, technology="auto").payload == ov.read(fasta, technology="hifi").payload  # type: ignore[attr-defined]


def test_unknown_technology_and_misuse_are_rejected(tmp_path: Path) -> None:
    reads = _hifi(tmp_path)

    with pytest.raises(OrganelleInputError) as unknown:
        ov.read(reads, technology="sanger")
    assert unknown.value.code == "input.invalid_technology"

    with pytest.raises(OrganelleInputError) as several:
        ov.read([reads, reads], technology="hifi")
    assert several.value.code == "input.too_many_read_files"

    with pytest.raises(OrganelleInputError) as length:
        ov.read(reads, technology="hifi", read_length=150)
    assert length.value.code == "input.read_length_not_applicable"

    with pytest.raises(OrganelleInputError) as mixed:
        ov.read(reads, technology="hifi", organelle="mitochondrion")
    assert mixed.value.code == "input.reads_shorthand_conflict"

    with pytest.raises(OrganelleInputError) as no_tech:
        ov.read(reads, quality_state="ccs")
    assert no_tech.value.code == "input.technology_required"


def test_missing_reads_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(OrganelleInputError) as caught:
        ov.read(tmp_path / "absent.fastq", technology="auto")

    assert caught.value.code == "input.missing_artifact"
