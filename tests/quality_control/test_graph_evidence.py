from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import pytest

try:
    import pysam
except ImportError:
    pytest.skip("pysam is required for graph evidence tests", allow_module_level=True)

from organelleverse.core.artifacts import ArtifactRef
from organelleverse.quality_control.contracts import (
    InputLibraryEvidence,
    LibraryAlignmentEvidence,
    MappingEvidence,
    ResolvedAssemblyEvidence,
)
from organelleverse.quality_control.graph import collect_graph_evidence
from organelleverse.quality_control.policy import QC_POLICY_V5


def _artifact(path: Path, *, kind: str, fmt: str) -> ArtifactRef:
    return ArtifactRef.from_path(path, kind=kind, format=fmt)


def _write_bam(
    path: Path,
    *,
    reference_length: int,
    records: tuple[tuple[str, int, str, int], ...],
) -> tuple[ArtifactRef, ArtifactRef]:
    header = {
        "HD": {"VN": "1.6", "SO": "coordinate"},
        "SQ": [{"SN": "ctg", "LN": reference_length}],
    }
    unsorted = path.with_suffix(".unsorted.bam")
    with pysam.AlignmentFile(str(unsorted), "wb", header=header) as output:
        for name, start, cigar, flag in records:
            aligned = pysam.AlignedSegment()
            aligned.query_name = name
            query_length = sum(
                int(token[:-1])
                for token in _cigar_tokens(cigar)
                if token[-1] in {"M", "I", "S", "=", "X"}
            )
            aligned.query_sequence = "A" * query_length
            aligned.flag = flag
            aligned.reference_id = 0
            aligned.reference_start = start
            aligned.mapping_quality = 60
            aligned.cigarstring = cigar
            aligned.query_qualities = pysam.qualitystring_to_array("I" * query_length)
            aligned.set_tag("NM", 0)
            output.write(aligned)
    pysam.sort("-o", str(path), str(unsorted))
    unsorted.unlink()
    pysam.index(str(path))
    bai = Path(f"{path}.bai")
    return (
        _artifact(path, kind="alignment_bam", fmt="bam"),
        _artifact(bai, kind="alignment_index", fmt="bai"),
    )


def _cigar_tokens(cigar: str) -> tuple[str, ...]:
    tokens: list[str] = []
    start = 0
    for index, character in enumerate(cigar):
        if character.isalpha() or character == "=":
            tokens.append(cigar[start : index + 1])
            start = index + 1
    return tuple(tokens)


def _evidence(
    tmp_path: Path,
    *,
    path_steps: str,
    sequence: str,
    records: tuple[tuple[str, int, str, int], ...],
    segment_sequences: dict[str, str] | None = None,
) -> tuple[ResolvedAssemblyEvidence, MappingEvidence]:
    segments = segment_sequences or {
        "A": sequence[0:400],
        "B": sequence[400:800],
        "C": sequence[800:1200],
    }
    names = [step[:-1] for step in path_steps.split(",")]
    path_sequence = "".join(segments[name] for name in names)
    fasta = tmp_path / "assembly.fa"
    fasta.write_text(f">ctg\n{path_sequence}\n")
    links = [f"L\t{left}\t+\t{right}\t+\t0M" for left, right in pairwise(names)]
    gfa = tmp_path / "assembly.gfa"
    gfa.write_text(
        "H\tVN:Z:1.0\n"
        + "".join(f"S\t{name}\t{segment}\n" for name, segment in segments.items())
        + "\n".join(links)
        + f"\nP\tctg\t{path_steps}\t*\n"
    )
    reads = tmp_path / "reads.fastq"
    reads.write_text("@r\n" + "A" * 600 + "\n+\n" + "I" * 600 + "\n")
    library = InputLibraryEvidence(
        role="hifi",
        technology="pacbio_hifi",
        quality_state="ccs",
        reads=_artifact(reads, kind="long_read", fmt="fastq"),
    )
    bam, bai = _write_bam(
        tmp_path / "reads.bam",
        reference_length=len(path_sequence),
        records=records,
    )
    resolved = ResolvedAssemblyEvidence.model_construct(
        kind="resolved_assembly_evidence",
        source_result=None,
        run_manifest=None,
        primary_genome=None,
        primary_fasta_path=fasta,
        assembly_graph_path=gfa,
        input_libraries=(library,),
    )
    mapping = MappingEvidence(
        library_alignments=(
            LibraryAlignmentEvidence(
                library_role="hifi",
                bam_artifact=bam,
                index_artifact=bai,
            ),
        )
    )
    return resolved, mapping


def test_long_reads_support_reported_path_junction(tmp_path: Path) -> None:
    sequence = "A" * 400 + "C" * 400 + "G" * 400
    records = tuple((f"span-{index}", 100, "600M", 0) for index in range(3))
    evidence, mapping = _evidence(
        tmp_path,
        path_steps="A+,B+",
        sequence=sequence,
        records=records,
    )

    graph = collect_graph_evidence(evidence, mapping, QC_POLICY_V5)

    assert graph.gfa_summary is not None
    assert graph.gfa_summary.segment_count == 3
    assert graph.gfa_summary.component_count == 2
    assert len(graph.junction_support) == 1
    assert graph.junction_support[0].supporting_reads == 3
    assert graph.junction_support[0].status == "supported"


def test_suitable_nonspanning_reads_make_junction_unsupported(tmp_path: Path) -> None:
    sequence = "A" * 400 + "C" * 400 + "G" * 400
    records = tuple((f"left-{index}", 0, "600M", 0) for index in range(3))
    evidence, mapping = _evidence(
        tmp_path,
        path_steps="A+,B+",
        sequence=sequence,
        records=records,
    )

    graph = collect_graph_evidence(evidence, mapping, QC_POLICY_V5)

    junction = graph.junction_support[0]
    assert junction.supporting_reads == 0
    assert junction.status == "unsupported"
    candidate = next(
        item
        for item in graph.structural_error_candidates
        if item.error_type == "unsupported_junction"
    )
    assert candidate.supporting_observations == 0


def test_soft_clipped_reads_can_contradict_a_claimed_junction(tmp_path: Path) -> None:
    sequence = "A" * 400 + "C" * 400 + "G" * 400
    records = tuple((f"clip-{index}", 150, "250M350S", 0) for index in range(3))
    evidence, mapping = _evidence(
        tmp_path,
        path_steps="A+,B+",
        sequence=sequence,
        records=records,
    )

    graph = collect_graph_evidence(evidence, mapping, QC_POLICY_V5)

    junction = graph.junction_support[0]
    assert junction.contradicting_reads == 3
    assert junction.status == "contradicted"
    candidate = graph.structural_error_candidates[0]
    assert candidate.supporting_observations == 3


def test_reverse_complement_path_accepts_the_reverse_of_a_declared_link(
    tmp_path: Path,
) -> None:
    segments = {"A": "AT" * 200, "B": "CG" * 200, "C": "GC" * 200}
    records = tuple((f"span-{index}", 100, "600M", 0) for index in range(3))
    evidence, mapping = _evidence(
        tmp_path,
        path_steps="B-,A-",
        sequence="",
        records=records,
        segment_sequences=segments,
    )
    assert evidence.assembly_graph_path is not None
    graph_text = evidence.assembly_graph_path.read_text()
    evidence.assembly_graph_path.write_text(
        graph_text.replace("L\tB\t+\tA\t+\t0M", "L\tA\t+\tB\t+\t0M")
    )

    graph = collect_graph_evidence(evidence, mapping, QC_POLICY_V5)

    assert graph.junction_support[0].status == "supported"


def test_explicit_path_repeat_reports_copy_and_spanning_evidence(tmp_path: Path) -> None:
    segment_a = "A" * 400
    segment_b = "C" * 400
    sequence = segment_a + segment_b + segment_a
    records = tuple(
        [(f"left-{index}", 100, "600M", 0) for index in range(3)]
        + [(f"right-{index}", 500, "600M", 0) for index in range(3)]
    )
    evidence, mapping = _evidence(
        tmp_path,
        path_steps="A+,B+,A+",
        sequence=sequence,
        records=records,
    )

    graph = collect_graph_evidence(evidence, mapping, QC_POLICY_V5)

    repeat = graph.repeat_support[0]
    assert repeat.repeat_id == "A"
    assert repeat.path_occurrences == 2
    assert not hasattr(repeat, "inferred_copies")
    assert not hasattr(repeat, "median_depth")
    assert repeat.spanning_reads == 6
    assert repeat.status == "resolved"
