from __future__ import annotations

import random
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest

try:
    import pysam
except ImportError:
    pytest.skip("pysam is required for mapping evidence tests", allow_module_level=True)

from organelleverse.assembly.execution import CommandOutcome, CommandRunner
from organelleverse.core.artifacts import ArtifactRef
from organelleverse.quality_control.contracts import (
    InputLibraryEvidence,
    ResolvedAssemblyEvidence,
)
from organelleverse.quality_control.environment import (
    QcEnvironment,
    QcExecutable,
    resolve_qc_environment,
)
from organelleverse.quality_control.evidence import collect_mapping_evidence


def _artifact(path: Path, *, kind: str, fmt: str) -> ArtifactRef:
    return ArtifactRef.from_path(
        path,
        kind=kind,
        format=fmt,
        media_type="application/octet-stream",
    )


def _resolved(fasta: Path, libraries: tuple[InputLibraryEvidence, ...]) -> ResolvedAssemblyEvidence:
    return ResolvedAssemblyEvidence.model_construct(
        kind="resolved_assembly_evidence",
        source_result=None,
        run_manifest=None,
        primary_genome=None,
        primary_fasta_path=fasta,
        assembly_graph_path=None,
        input_libraries=libraries,
    )


def _fake_environment(tmp_path: Path, *, with_meryl: bool = False) -> QcEnvironment:
    return QcEnvironment(
        environment_id="qc-environment:test",
        source="installed",
        minimap2=QcExecutable(
            name="minimap2",
            path=tmp_path / "minimap2",
            version="test",
            sha256="a" * 64,
            source="installed",
        ),
        samtools=QcExecutable(
            name="samtools",
            path=tmp_path / "samtools",
            version="test",
            sha256="b" * 64,
            source="installed",
        ),
        meryl=(
            QcExecutable(
                name="meryl",
                path=tmp_path / "meryl",
                version="test",
                sha256="c" * 64,
                source="installed",
            )
            if with_meryl
            else None
        ),
    )


def _real_environment() -> QcEnvironment:
    minimap2 = shutil.which("minimap2")
    samtools = shutil.which("samtools")
    if minimap2 is None or samtools is None:
        pytest.skip("tiny real mapping integration requires minimap2 and samtools")
    return resolve_qc_environment(
        installed={"minimap2": Path(minimap2), "samtools": Path(samtools)}
    )


def _outcome(
    argv: tuple[str, ...],
    *,
    stage: str,
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
) -> CommandOutcome:
    now = datetime.now(UTC)
    return CommandOutcome(
        stage=stage,
        argv=argv,
        cwd=str(cwd),
        started=True,
        started_at=now,
        finished_at=now,
        duration_seconds=0.0,
        returncode=0,
        termination="exit",
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
    )


class _DeterministicRunner(CommandRunner):
    def __init__(
        self,
        *,
        meryl_failure_stage: str | None = None,
        sam_records: tuple[str, ...] | None = None,
    ) -> None:
        super().__init__()
        self.calls: list[dict[str, object]] = []
        self.selected_reads = ""
        self.meryl_failure_stage = meryl_failure_stage
        self.sam_records = sam_records

    def run(
        self,
        argv: tuple[str, ...],
        *,
        stage: str,
        cwd: Path,
        timeout_seconds: float,
        stdout_path: Path,
        stderr_path: Path,
        env: dict[str, str] | None = None,
        stdin_path: Path | None = None,
    ) -> CommandOutcome:
        self.calls.append(
            {
                "argv": argv,
                "stage": stage,
                "cwd": cwd,
                "timeout_seconds": timeout_seconds,
                "stdout_path": stdout_path,
                "stderr_path": stderr_path,
                "stdin_path": stdin_path,
            }
        )
        stderr_path.write_text("")
        if stage == self.meryl_failure_stage:
            stdout_path.write_text("")
            stderr_path.write_text("meryl failed")
            return CommandOutcome(
                stage=stage,
                argv=argv,
                cwd=str(cwd),
                started=True,
                started_at=datetime.now(UTC),
                finished_at=datetime.now(UTC),
                duration_seconds=0.0,
                returncode=2,
                termination="exit",
                stdout_path=str(stdout_path),
                stderr_path=str(stderr_path),
            )
        if stage.endswith(".minimap2"):
            if self.sam_records is not None:
                stdout_path.write_text(
                    "@HD\tVN:1.6\tSO:unsorted\n"
                    "@SQ\tSN:ctg1\tLN:200\n" + "".join(f"{record}\n" for record in self.sam_records)
                )
            else:
                mismatch = stage.startswith("library_002")
                sequence = "ACGT" * 25
                stdout_path.write_text(
                    "@HD\tVN:1.6\tSO:unsorted\n"
                    "@SQ\tSN:ctg1\tLN:200\n"
                    f"r1\t0\tctg1\t1\t60\t100M\t*\t0\t0\t{sequence}\t{'I' * 100}"
                    f"\tNM:i:{10 if mismatch else 0}\n"
                )
        elif stage.endswith(".samtools_sort"):
            assert stdin_path is not None
            output = Path(argv[argv.index("-o") + 1])
            pysam.sort("-o", str(output), str(stdin_path))
            stdout_path.write_text("")
        elif stage.endswith(".samtools_index"):
            pysam.index(argv[-2], argv[-1])
            stdout_path.write_text("")
        elif stage == "kmer.count_target_reads":
            self.selected_reads = Path(argv[-3]).read_text()
            Path(argv[-1]).mkdir()
            stdout_path.write_text("")
        elif stage in {"kmer.count_assembly", "kmer.intersect"}:
            Path(argv[-1]).mkdir()
            stdout_path.write_text("")
        elif stage == "kmer.histogram_assembly":
            stdout_path.write_text("1\t8\n2\t4\n")
        elif stage == "kmer.histogram_shared":
            stdout_path.write_text("1\t9\n")
        return _outcome(
            argv,
            stage=stage,
            cwd=cwd,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )


def test_mapping_uses_exact_shell_free_argv_and_keeps_libraries_separate(
    tmp_path: Path,
) -> None:
    fasta = tmp_path / "assembly.fa"
    fasta.write_text(">ctg1\n" + "ACGT" * 50 + "\n")
    read_a = tmp_path / "a.fastq"
    read_b = tmp_path / "b.fastq"
    fastq = "@r1\n" + "ACGT" * 25 + "\n+\n" + "I" * 100 + "\n"
    read_a.write_text(fastq)
    read_b.write_text(fastq)
    libraries = (
        InputLibraryEvidence(
            role="hifi_a",
            technology="pacbio_hifi",
            quality_state="ccs",
            reads=_artifact(read_a, kind="long_read", fmt="fastq"),
        ),
        InputLibraryEvidence(
            role="hifi_b",
            technology="pacbio_hifi",
            quality_state="ccs",
            reads=_artifact(read_b, kind="long_read", fmt="fastq"),
        ),
    )
    environment = _fake_environment(tmp_path)
    runner = _DeterministicRunner()

    evidence = collect_mapping_evidence(
        _resolved(fasta, libraries),
        environment,
        workspace=tmp_path / "mapping",
        threads=3,
        timeout_seconds=60,
        runner=runner,
    )

    minimap_calls = [call for call in runner.calls if str(call["stage"]).endswith(".minimap2")]
    assert [call["argv"] for call in minimap_calls] == [
        (
            str(environment.minimap2.path),
            "-a",
            "-x",
            "map-hifi",
            "-t",
            "3",
            str(fasta),
            str(read_a),
        ),
        (
            str(environment.minimap2.path),
            "-a",
            "-x",
            "map-hifi",
            "-t",
            "3",
            str(fasta),
            str(read_b),
        ),
    ]
    sort_calls = [call for call in runner.calls if str(call["stage"]).endswith(".samtools_sort")]
    assert all(call["stdin_path"] is not None for call in sort_calls)
    assert [item.library_role for item in evidence.library_mapping_summaries] == [
        "hifi_a",
        "hifi_b",
    ]
    first = evidence.library_mapping_summaries[0]
    second = evidence.library_mapping_summaries[1]
    # The per-library technology/quality_state are reported alongside the
    # honest agreement metric; the value is not a Merqury consensus QV.
    assert first.technology == "pacbio_hifi"
    assert first.quality_state == "ccs"
    assert first.read_assembly_agreement_qv is not None
    assert second.read_assembly_agreement_qv is not None
    assert first.read_assembly_agreement_qv > second.read_assembly_agreement_qv
    assert not hasattr(first, "consensus_qv")
    kmer_check = next(item for item in evidence.checks if item.check_id == "qc.kmer_evidence")
    assert kmer_check.status == "not_assessed"
    assert (
        len([item for item in evidence.coordinate_artifacts if item.kind == "alignment_bam"]) == 2
    )


@pytest.mark.parametrize(
    ("cigar", "query_sequence", "edit_count", "expected_type"),
    [
        ("50M3I50M", "A" * 103, 3, "small_collapse"),
        ("50M3D50M", "A" * 100, 3, "small_expansion"),
        ("100M", "A" * 50 + "C" + "A" * 49, 1, "substitution"),
    ],
)
def test_high_fraction_support_promotes_directional_assembly_base_error_candidate(
    tmp_path: Path,
    cigar: str,
    query_sequence: str,
    edit_count: int,
    expected_type: str,
) -> None:
    fasta = tmp_path / "assembly.fa"
    fasta.write_text(">ctg1\n" + "A" * 200 + "\n")
    reads = tmp_path / "reads.fastq"
    reads.write_text(f"@r1\n{query_sequence}\n+\n{'I' * len(query_sequence)}\n")
    library = InputLibraryEvidence(
        role="hifi",
        technology="pacbio_hifi",
        quality_state="ccs",
        reads=_artifact(reads, kind="long_read", fmt="fastq"),
    )
    records = tuple(
        f"support-{index}\t0\tctg1\t1\t60\t{cigar}\t*\t0\t0\t"
        f"{query_sequence}\t{'I' * len(query_sequence)}\tNM:i:{edit_count}"
        for index in range(1, 6)
    )

    evidence = collect_mapping_evidence(
        _resolved(fasta, (library,)),
        _fake_environment(tmp_path),
        workspace=tmp_path / "mapping",
        threads=1,
        timeout_seconds=60,
        runner=_DeterministicRunner(sam_records=records),
    )

    assert len(evidence.base_error_candidates) == 1
    candidate = evidence.base_error_candidates[0]
    assert candidate.error_type == expected_type
    assert candidate.supporting_observations == 5
    assert candidate.assessed_depth == 5
    assert candidate.support_fraction == 1.0
    assert candidate.supporting_library_count == 1
    assert candidate.confidence is None
    support_artifact = next(
        item for item in evidence.coordinate_artifacts if item.kind == "base_error_evidence"
    )
    assert candidate.evidence_artifact_ids == (support_artifact.object_id,)
    support_text = Path(support_artifact.uri).read_text()
    assert evidence.library_alignments[0].bam_artifact.object_id in support_text
    assert f"\t{expected_type}\t5\t5\t1.0\t1\t" in support_text
    check = next(item for item in evidence.checks if item.check_id == "qc.base_error_candidates")
    assert check.status == "warn"
    assert check.value == 1
    assert check.finding_code == "qc.base_error_candidate"


def test_low_fraction_read_error_is_not_promoted_to_assembly_error_candidate(
    tmp_path: Path,
) -> None:
    fasta = tmp_path / "assembly.fa"
    fasta.write_text(">ctg1\n" + "A" * 200 + "\n")
    reads = tmp_path / "reads.fastq"
    reads.write_text("@r1\n" + "A" * 100 + "\n+\n" + "I" * 100 + "\n")
    library = InputLibraryEvidence(
        role="hifi",
        technology="pacbio_hifi",
        quality_state="ccs",
        reads=_artifact(reads, kind="long_read", fmt="fastq"),
    )
    mismatch = "A" * 50 + "C" + "A" * 49
    records = tuple(
        (
            f"read-{index}\t0\tctg1\t1\t60\t100M\t*\t0\t0\t"
            f"{mismatch if index <= 3 else 'A' * 100}\t{'I' * 100}\t"
            f"NM:i:{1 if index <= 3 else 0}"
        )
        for index in range(1, 11)
    )

    evidence = collect_mapping_evidence(
        _resolved(fasta, (library,)),
        _fake_environment(tmp_path),
        workspace=tmp_path / "mapping",
        threads=1,
        timeout_seconds=60,
        runner=_DeterministicRunner(sam_records=records),
    )

    assert evidence.base_error_candidates == ()
    check = next(item for item in evidence.checks if item.check_id == "qc.base_error_candidates")
    assert check.status == "pass"


def test_deletion_candidate_fraction_counts_reference_supporting_reads(
    tmp_path: Path,
) -> None:
    fasta = tmp_path / "assembly.fa"
    fasta.write_text(">ctg1\n" + "A" * 200 + "\n")
    reads = tmp_path / "reads.fastq"
    reads.write_text("@r1\n" + "A" * 100 + "\n+\n" + "I" * 100 + "\n")
    library = InputLibraryEvidence(
        role="hifi",
        technology="pacbio_hifi",
        quality_state="ccs",
        reads=_artifact(reads, kind="long_read", fmt="fastq"),
    )
    records = tuple(
        (
            f"read-{index}\t0\tctg1\t1\t60\t"
            f"{'50M3D50M' if index <= 5 else '100M'}\t*\t0\t0\t"
            f"{'A' * 100}\t{'I' * 100}\tNM:i:{3 if index <= 5 else 0}"
        )
        for index in range(1, 8)
    )

    evidence = collect_mapping_evidence(
        _resolved(fasta, (library,)),
        _fake_environment(tmp_path),
        workspace=tmp_path / "mapping",
        threads=1,
        timeout_seconds=60,
        runner=_DeterministicRunner(sam_records=records),
    )

    assert evidence.base_error_candidates == ()


def test_single_read_indel_is_not_promoted_to_assembly_error_candidate(
    tmp_path: Path,
) -> None:
    fasta = tmp_path / "assembly.fa"
    fasta.write_text(">ctg1\n" + "A" * 200 + "\n")
    reads = tmp_path / "reads.fastq"
    reads.write_text("@r1\n" + "A" * 103 + "\n+\n" + "I" * 103 + "\n")
    library = InputLibraryEvidence(
        role="hifi",
        technology="pacbio_hifi",
        quality_state="ccs",
        reads=_artifact(reads, kind="long_read", fmt="fastq"),
    )
    record = f"noise\t0\tctg1\t1\t60\t50M3I50M\t*\t0\t0\t{'A' * 103}\t{'I' * 103}\tNM:i:3"

    evidence = collect_mapping_evidence(
        _resolved(fasta, (library,)),
        _fake_environment(tmp_path),
        workspace=tmp_path / "mapping",
        threads=1,
        timeout_seconds=60,
        runner=_DeterministicRunner(sam_records=(record,)),
    )

    assert evidence.base_error_candidates == ()
    check = next(item for item in evidence.checks if item.check_id == "qc.base_error_candidates")
    assert check.status == "pass"
    assert check.value == 0


def test_fasta_reads_without_base_qualities_contribute_coverage(
    tmp_path: Path,
) -> None:
    fasta = tmp_path / "assembly.fa"
    fasta.write_text(">ctg1\n" + "A" * 200 + "\n")
    reads = tmp_path / "reads.fa"
    reads.write_text(">r1\n" + "A" * 100 + "\n")
    library = InputLibraryEvidence(
        role="hifi",
        technology="pacbio_hifi",
        quality_state="ccs",
        reads=_artifact(reads, kind="long_read", fmt="fasta"),
    )
    records = tuple(
        f"support-{index}\t0\tctg1\t1\t60\t100M\t*\t0\t0\t{'A' * 100}\t*\tNM:i:0"
        for index in range(1, 4)
    )

    evidence = collect_mapping_evidence(
        _resolved(fasta, (library,)),
        _fake_environment(tmp_path),
        workspace=tmp_path / "mapping",
        threads=1,
        timeout_seconds=60,
        runner=_DeterministicRunner(sam_records=records),
    )

    window = evidence.coverage_windows[0]
    # High-MAPQ primary alignments feed both the repeat-aware any-alignment
    # track and the high-confidence primary track.
    assert window.any_alignment_mean_depth == 1.5
    assert window.confident_mean_depth == 1.5


def test_fastq_bases_below_quality_threshold_do_not_contribute_coverage(
    tmp_path: Path,
) -> None:
    fasta = tmp_path / "assembly.fa"
    fasta.write_text(">ctg1\n" + "A" * 200 + "\n")
    reads = tmp_path / "reads.fastq"
    reads.write_text(f"@r1\n{'A' * 100}\n+\n{'!' * 100}\n")
    library = InputLibraryEvidence(
        role="hifi",
        technology="pacbio_hifi",
        quality_state="ccs",
        reads=_artifact(reads, kind="long_read", fmt="fastq"),
    )
    record = f"low-quality\t0\tctg1\t1\t60\t100M\t*\t0\t0\t{'A' * 100}\t{'!' * 100}\tNM:i:0"

    evidence = collect_mapping_evidence(
        _resolved(fasta, (library,)),
        _fake_environment(tmp_path),
        workspace=tmp_path / "mapping",
        threads=1,
        timeout_seconds=60,
        runner=_DeterministicRunner(sam_records=(record,)),
    )

    window = evidence.coverage_windows[0]
    assert window.any_alignment_mean_depth == 0.0
    assert window.confident_mean_depth == 0.0


def test_ambiguous_primary_and_secondary_alignments_contribute_repeat_coverage(
    tmp_path: Path,
) -> None:
    fasta = tmp_path / "assembly.fa"
    fasta.write_text(">ctg1\n" + "A" * 200 + "\n")
    reads = tmp_path / "reads.fastq"
    reads.write_text("@repeat\n" + "A" * 100 + "\n+\n" + "I" * 100 + "\n")
    library = InputLibraryEvidence(
        role="illumina_pe",
        technology="illumina",
        layout="single_end",
        read1=_artifact(reads, kind="short_read", fmt="fastq"),
        read_length=100,
    )
    # A low-MAPQ primary and its secondary placement: both feed the
    # repeat-aware any-alignment track, neither feeds the confident track.
    records = (
        f"repeat\t0\tctg1\t1\t0\t100M\t*\t0\t0\t{'A' * 100}\t{'I' * 100}\tNM:i:0",
        f"repeat\t256\tctg1\t101\t0\t100M\t*\t0\t0\t{'A' * 100}\t{'I' * 100}\tNM:i:0",
    )

    evidence = collect_mapping_evidence(
        _resolved(fasta, (library,)),
        _fake_environment(tmp_path),
        workspace=tmp_path / "mapping",
        threads=1,
        timeout_seconds=60,
        runner=_DeterministicRunner(sam_records=records),
    )

    interval_artifact = next(
        item for item in evidence.coordinate_artifacts if item.kind == "coverage_intervals"
    )
    assert "zero_coverage" not in Path(interval_artifact.uri).read_text()
    confident_artifact = next(
        item
        for item in evidence.coordinate_artifacts
        if item.kind == "coverage_intervals_confident"
    )
    # The repeat-aware track covers the window; the confident track does not.
    window = evidence.coverage_windows[0]
    assert window.any_alignment_mean_depth == 1.0
    assert window.confident_mean_depth == 0.0
    assert Path(confident_artifact.uri).read_text().splitlines() == [
        "ctg1\t0\t200\tzero_coverage",
    ]
    assert evidence.base_error_candidates == ()


def test_meryl_reports_target_read_kmer_support_with_shell_free_commands(
    tmp_path: Path,
) -> None:
    fasta = tmp_path / "assembly.fa"
    fasta.write_text(">ctg1\n" + "ACGT" * 50 + "\n")
    reads = tmp_path / "reads.fastq"
    reads.write_text("@r1\n" + "ACGT" * 25 + "\n+\n" + "I" * 100 + "\n")
    library = InputLibraryEvidence(
        role="hifi",
        technology="pacbio_hifi",
        quality_state="ccs",
        reads=_artifact(reads, kind="long_read", fmt="fastq"),
    )
    environment = _fake_environment(tmp_path, with_meryl=True)
    runner = _DeterministicRunner()
    workspace = tmp_path / "mapping"

    evidence = collect_mapping_evidence(
        _resolved(fasta, (library,)),
        environment,
        workspace=workspace,
        threads=3,
        timeout_seconds=60,
        runner=runner,
    )

    kmer_check = next(item for item in evidence.checks if item.check_id == "qc.kmer_evidence")
    assert kmer_check.status == "pass"
    assert kmer_check.value == 0.75
    assert kmer_check.unit == "fraction"
    assert "9 of 12" in kmer_check.message
    assert runner.selected_reads.startswith(">library_001_read_000000001\n")
    meryl_calls = [
        (call["stage"], call["argv"])
        for call in runner.calls
        if str(call["stage"]).startswith("kmer.")
    ]
    meryl = str(environment.meryl.path) if environment.meryl is not None else ""
    assert meryl_calls == [
        (
            "kmer.count_target_reads",
            (
                meryl,
                "count",
                "k=21",
                "threads=3",
                str(workspace / "target_reads.fasta"),
                "output",
                str(workspace / "target_reads.meryl"),
            ),
        ),
        (
            "kmer.count_assembly",
            (
                meryl,
                "count",
                "k=21",
                "threads=3",
                str(fasta),
                "output",
                str(workspace / "assembly.meryl"),
            ),
        ),
        (
            "kmer.intersect",
            (
                meryl,
                "intersect",
                "threads=3",
                str(workspace / "assembly.meryl"),
                str(workspace / "target_reads.meryl"),
                "output",
                str(workspace / "shared.meryl"),
            ),
        ),
        (
            "kmer.histogram_assembly",
            (
                meryl,
                "histogram",
                "threads=3",
                str(workspace / "assembly.meryl"),
            ),
        ),
        (
            "kmer.histogram_shared",
            (
                meryl,
                "histogram",
                "threads=3",
                str(workspace / "shared.meryl"),
            ),
        ),
    ]
    metrics = next(item for item in evidence.coordinate_artifacts if item.kind == "kmer_metrics")
    assert Path(metrics.uri).read_text() == (
        "metric\tvalue\n"
        "kmer_size\t21\n"
        "assembly_distinct_kmers\t12\n"
        "target_read_supported_assembly_kmers\t9\n"
        "assembly_kmer_support_fraction\t0.75\n"
    )
    assert not (workspace / "target_reads.fasta").exists()
    assert not (workspace / "assembly.meryl").exists()
    assert not (workspace / "target_reads.meryl").exists()
    assert not (workspace / "shared.meryl").exists()


def test_meryl_failure_is_explicit_without_discarding_mapping_evidence(
    tmp_path: Path,
) -> None:
    fasta = tmp_path / "assembly.fa"
    fasta.write_text(">ctg1\n" + "ACGT" * 50 + "\n")
    reads = tmp_path / "reads.fastq"
    reads.write_text("@r1\n" + "ACGT" * 25 + "\n+\n" + "I" * 100 + "\n")
    library = InputLibraryEvidence(
        role="hifi",
        technology="pacbio_hifi",
        quality_state="ccs",
        reads=_artifact(reads, kind="long_read", fmt="fastq"),
    )
    runner = _DeterministicRunner(meryl_failure_stage="kmer.count_target_reads")
    workspace = tmp_path / "mapping"

    evidence = collect_mapping_evidence(
        _resolved(fasta, (library,)),
        _fake_environment(tmp_path, with_meryl=True),
        workspace=workspace,
        threads=1,
        timeout_seconds=60,
        runner=runner,
    )

    mapping_check = next(item for item in evidence.checks if item.check_id == "qc.read_mapping")
    assert mapping_check.status == "pass"
    kmer_check = next(item for item in evidence.checks if item.check_id == "qc.kmer_evidence")
    assert kmer_check.status == "not_assessed"
    assert kmer_check.value is None
    assert kmer_check.finding_code == "qc.kmer_tool_failed"
    assert "kmer.count_target_reads" in kmer_check.message
    assert any(
        item.kind == "execution_log" and Path(item.uri).read_text() == "meryl failed"
        for item in evidence.coordinate_artifacts
    )
    assert not (workspace / "target_reads.fasta").exists()
    assert not (workspace / "target_reads.meryl").exists()


def test_tiny_real_mapping_reports_exact_zero_coverage_intervals(tmp_path: Path) -> None:
    random_source = random.Random(17)
    sequence = "".join(random_source.choice("ACGT") for _ in range(1_600))
    fasta = tmp_path / "assembly.fa"
    fasta.write_text(f">ctg1\n{sequence}\n")
    reads = tmp_path / "reads.fastq"
    read_sequences = (sequence[100:500], sequence[1100:1500])
    reads.write_text(
        "".join(
            f"@r{index}\n{read}\n+\n{'I' * len(read)}\n"
            for index, read in enumerate(read_sequences, start=1)
        )
    )
    library = InputLibraryEvidence(
        role="hifi",
        technology="pacbio_hifi",
        quality_state="ccs",
        reads=_artifact(reads, kind="long_read", fmt="fastq"),
    )
    environment = _real_environment()

    evidence = collect_mapping_evidence(
        _resolved(fasta, (library,)),
        environment,
        workspace=tmp_path / "real-mapping",
        threads=1,
        timeout_seconds=60,
    )

    summary = evidence.library_mapping_summaries[0]
    assert summary.mapped_reads == 2
    assert summary.target_fraction == 1.0
    assert summary.read_assembly_agreement_qv is not None
    interval_artifact = next(
        item for item in evidence.coordinate_artifacts if item.kind == "coverage_intervals"
    )
    assert Path(interval_artifact.uri).read_text().splitlines() == [
        "ctg1\t0\t100\tzero_coverage",
        "ctg1\t100\t500\tlow_coverage",
        "ctg1\t500\t1100\tzero_coverage",
        "ctg1\t1100\t1500\tlow_coverage",
        "ctg1\t1500\t1600\tzero_coverage",
    ]


@pytest.mark.parametrize(("termination", "returncode"), [("timeout", None), ("exit", 2)])
def test_mapping_process_failure_is_typed(
    tmp_path: Path, termination: str, returncode: int | None
) -> None:
    fasta = tmp_path / "assembly.fa"
    fasta.write_text(">ctg1\n" + "ACGT" * 50 + "\n")
    reads = tmp_path / "reads.fastq"
    reads.write_text("@r\n" + "ACGT" * 25 + "\n+\n" + "I" * 100 + "\n")
    library = InputLibraryEvidence(
        role="hifi",
        technology="pacbio_hifi",
        quality_state="ccs",
        reads=_artifact(reads, kind="long_read", fmt="fastq"),
    )
    environment = _fake_environment(tmp_path)

    class _FailingRunner(CommandRunner):
        def run(
            self,
            argv: tuple[str, ...],
            *,
            stage: str,
            cwd: Path,
            timeout_seconds: float,
            stdout_path: Path,
            stderr_path: Path,
            env: dict[str, str] | None = None,
            stdin_path: Path | None = None,
        ) -> CommandOutcome:
            now = datetime.now(UTC)
            stdout_path.write_text("")
            stderr_path.write_text("failed")
            return CommandOutcome(
                stage=stage,
                argv=argv,
                cwd=str(cwd),
                started=True,
                started_at=now,
                finished_at=now,
                duration_seconds=0.0,
                returncode=returncode,
                termination=termination,  # type: ignore[arg-type]
                stdout_path=str(stdout_path),
                stderr_path=str(stderr_path),
            )

    from organelleverse.core.errors import OrganelleExecutionError

    with pytest.raises(OrganelleExecutionError) as captured:
        collect_mapping_evidence(
            _resolved(fasta, (library,)),
            environment,
            workspace=tmp_path / "failed-mapping",
            threads=1,
            timeout_seconds=1,
            runner=_FailingRunner(),
        )
    assert captured.value.code == "qc.mapping_failed"
