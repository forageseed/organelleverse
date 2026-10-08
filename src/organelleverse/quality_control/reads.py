"""FASTQ QC using the shared assembly parser, numpy and the real fastp CLI.

Quality means are arithmetic means of Phred+33 scores, weighted by bases.
Long-read selection ranks by mean quality, then length, then input position.
It keeps the shortest ranked prefix reaching target_bases (whole reads), and
writes selected records in input order. It is not Filtlong's composite score.
Only lengths/quality sums/indices are retained, never a dataset of sequences.
"""

from __future__ import annotations

import json
import math
import os
import shutil
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from organelleverse.assembly.read_stats import iter_fastq_records, open_record_text
from organelleverse.core.errors import (
    OrganelleDependencyError,
    OrganelleExecutionError,
    OrganelleInputError,
    OrganelleParameterError,
)
from organelleverse.core.external import run_external
from organelleverse.core.result import OrganelleResult


def _records(path: Path) -> Iterator[tuple[str, str, str]]:
    try:
        with open_record_text(path) as handle:
            yield from iter_fastq_records(handle)
    except (OSError, UnicodeError, EOFError) as error:
        raise OrganelleInputError(
            code="qc.invalid_fastq", message=f"Cannot read FASTQ {path}: {error}"
        ) from error


def _quality_sum(quality: str) -> int:
    try:
        # numpy's buffer parameter is untyped in the installed stubs.
        values = np.frombuffer(quality.encode("ascii"), dtype=np.uint8)  # pyright: ignore[reportUnknownMemberType]
    except UnicodeEncodeError as error:
        raise OrganelleInputError(
            code="qc.invalid_quality", message="FASTQ quality must use Phred+33 ASCII scores."
        ) from error
    if values.min() < 33 or values.max() > 126:
        raise OrganelleInputError(
            code="qc.invalid_quality", message="FASTQ quality must use Phred+33 scores 0..93."
        )
    return int(values.sum()) - 33 * len(quality)


class _Statistics:
    def __init__(self) -> None:
        self.lengths: Counter[int] = Counter()
        self.quality_sum = 0

    def add(self, length: int, quality_sum: int) -> None:
        self.lengths[length] += 1
        self.quality_sum += quality_sum

    def result(self) -> dict[str, Any]:
        count = sum(self.lengths.values())
        bases = sum(length * count for length, count in self.lengths.items())
        n50 = 0
        cumulative_bases = cumulative_reads = 0
        middle: list[int] = []
        for length, frequency in sorted(self.lengths.items()):
            for rank in ((count - 1) // 2, count // 2):
                if cumulative_reads <= rank < cumulative_reads + frequency:
                    middle.append(length)
            cumulative_reads += frequency
        for length, frequency in sorted(self.lengths.items(), reverse=True):
            cumulative_bases += length * frequency
            if cumulative_bases * 2 >= bases:
                n50 = length
                break
        return {
            "reads": count,
            "bases": bases,
            "n50": n50,
            "mean_length": bases / count if count else None,
            "median_length": sum(middle) / 2 if count else None,
            "mean_quality": self.quality_sum / bases if bases else None,
        }


def _statistics(paths: list[Path]) -> dict[str, Any]:
    stats = _Statistics()
    for path in paths:
        for _header, sequence, quality in _records(path):
            stats.add(len(sequence), _quality_sum(quality))
    return stats.result()


def _result(operation: str, metrics: dict[str, Any], *, empty: bool = False) -> OrganelleResult:
    return OrganelleResult.model_validate(
        {
            "operation_id": f"qc.{operation}",
            "scope": "none",
            "status": "warning" if empty else "ok",
            "summary_text": f"Read QC: {operation}",
            "metrics": metrics,
            "flags": ("qc.no_reads_retained",) if empty else (),
        }
    )


def read_statistics(reads: Path) -> OrganelleResult:
    """Count four-line FASTQ (plain/gzip); mean quality is base-weighted Phred+33.

    Empty files have zero counts/N50 and null mean/median values.
    """
    return _result("read_statistics", _statistics([Path(reads)]))


def _run_directory(operation: str) -> Path:
    from organelleverse.runtime import managed_run_path

    output = managed_run_path(f"qc.{operation}", uuid4().hex)
    output.mkdir(parents=True, exist_ok=False)
    return output


def _validate_filters(min_length: int, min_mean_quality: float, target_bases: int | None) -> None:
    if type(min_length) is not int or min_length < 1:
        raise OrganelleParameterError(code="qc.invalid_length", message="min_length must be >= 1.")
    if not math.isfinite(min_mean_quality) or not 0 <= min_mean_quality <= 93:
        raise OrganelleParameterError(
            code="qc.invalid_quality", message="min_mean_quality must be finite and in 0..93."
        )
    if target_bases is not None and (type(target_bases) is not int or target_bases < 1):
        raise OrganelleParameterError(
            code="qc.invalid_target", message="target_bases must be a positive integer."
        )


def filter_long_reads(
    reads: Path,
    *,
    min_length: int = 1000,
    min_mean_quality: float = 0.0,
    target_bases: int | None = None,
) -> OrganelleResult:
    """Filter ONT/HiFi FASTQ; optionally retain the highest mean-quality reads.

    Rank: descending arithmetic mean Phred, descending length, input position.
    Keep whole reads until reaching the target, or all passing reads if fewer
    bases are available. Overshoot is strictly less than the last ranked read
    length. Output remains in input order. No platform-specific quality model
    is inferred. Memory is O(number of passing reads) with a target, otherwise
    O(number of distinct lengths). Inputs must remain unchanged during the run.
    """
    _validate_filters(min_length, min_mean_quality, target_bases)
    reads = Path(reads)
    before, eligible, after = _Statistics(), _Statistics(), _Statistics()
    output = _run_directory("filter_long_reads") / "filtered.fastq"
    candidates: list[tuple[float, int, int]] = []
    with output.open("w", encoding="utf-8") as handle:
        for index, (header, sequence, quality) in enumerate(_records(reads)):
            length, quality_sum = len(sequence), _quality_sum(quality)
            before.add(length, quality_sum)
            mean = quality_sum / length
            if length < min_length or mean < min_mean_quality:
                continue
            eligible.add(length, quality_sum)
            if target_bases is not None:
                candidates.append((-mean, -length, index))
            else:
                handle.write(f"@{header}\n{sequence}\n+\n{quality}\n")
                after.add(length, quality_sum)
        last_length = None
        if target_bases is not None:
            candidates.sort()
            selected: set[int] = set()
            retained_bases = 0
            for _mean, negative_length, index in candidates:
                selected.add(index)
                last_length = -negative_length
                retained_bases += last_length
                if retained_bases >= target_bases:
                    break
            del candidates
            for index, (header, sequence, quality) in enumerate(_records(reads)):
                if index in selected:
                    handle.write(f"@{header}\n{sequence}\n+\n{quality}\n")
                    after.add(len(sequence), _quality_sum(quality))
    final = after.result()
    return _result(
        "filter_long_reads",
        {
            "before": before.result(),
            "eligible": eligible.result(),
            "after": final,
            "min_length": min_length,
            "min_mean_quality": min_mean_quality,
            "target_bases": target_bases,
            "target_delta_bases": final["bases"] - target_bases
            if target_bases is not None
            else None,
            "last_ranked_read_length": last_length,
            "ranking": "mean_phred_desc,length_desc,input_order",
            "quality_definition": "arithmetic_phred33_base_weighted",
            "output_paths": [str(output)],
        },
        empty=final["reads"] == 0,
    )


def filter_short_reads(
    reads: Path,
    *,
    reads2: Path | None = None,
    min_length: int = 15,
    min_mean_quality: int = 0,
    threads: int = 2,
    adapter_sequence: str | None = None,
    adapter_sequence_r2: str | None = None,
) -> OrganelleResult:
    """Run fastp on single/paired Phred+33 FASTQ and parse its JSON report.

    fastp's default per-base quality and N filters remain enabled, as does
    adapter trimming (including paired overlap detection). Reads count mates
    individually. Returned duplication rate describes the input, not deduplication.
    Set ORGANELLEVERSE_FASTP to an executable or put fastp on PATH.
    Output FASTQ and complete JSON/HTML reports live in managed run storage.
    """
    _validate_filters(min_length, min_mean_quality, None)
    if type(min_mean_quality) is not int:
        raise OrganelleParameterError(
            code="qc.invalid_quality", message="fastp min_mean_quality must be an integer."
        )
    if type(threads) is not int or not 1 <= threads <= 16:
        raise OrganelleParameterError(
            code="qc.invalid_threads", message="threads must be in 1..16."
        )
    if adapter_sequence_r2 is not None and reads2 is None:
        raise OrganelleParameterError(
            code="qc.invalid_adapter", message="adapter_sequence_r2 requires paired reads."
        )
    for adapter in (adapter_sequence, adapter_sequence_r2):
        if adapter is not None and (len(adapter) < 4 or set(adapter.upper()) - set("ACGT")):
            raise OrganelleParameterError(
                code="qc.invalid_adapter",
                message="Adapters must contain at least four A/C/G/T bases.",
            )
    binary = shutil.which(os.environ.get("ORGANELLEVERSE_FASTP", "fastp"))
    if binary is None:
        raise OrganelleDependencyError(
            code="qc.fastp_missing",
            message="Install fastp: micromamba create -n ov-fastp -c conda-forge -c bioconda fastp; "
            "put fastp on PATH or set ORGANELLEVERSE_FASTP=/path/to/fastp.",
        )
    inputs = [Path(reads)] + ([Path(reads2)] if reads2 is not None else [])
    before = _statistics(inputs)
    if not before["reads"]:
        raise OrganelleInputError(code="qc.empty_fastq", message="fastp requires nonempty reads.")
    out = _run_directory("filter_short_reads")
    outputs = [out / "filtered_1.fastq"]
    report, html = out / "fastp.json", out / "fastp.html"
    argv = [
        binary,
        "--in1",
        str(inputs[0]),
        "--out1",
        str(outputs[0]),
        "--json",
        str(report),
        "--html",
        str(html),
        "--thread",
        str(threads),
        "--length_required",
        str(min_length),
        "--average_qual",
        str(min_mean_quality),
    ]
    if reads2 is not None:
        outputs.append(out / "filtered_2.fastq")
        argv += ["--in2", str(inputs[1]), "--out2", str(outputs[1]), "--detect_adapter_for_pe"]
    for option, adapter in (
        ("--adapter_sequence", adapter_sequence),
        ("--adapter_sequence_r2", adapter_sequence_r2),
    ):
        if adapter is not None:
            argv += [option, adapter.upper()]
    completed = run_external(argv, tool="fastp", code="qc.fastp_failed")
    (out / "fastp.log").write_text(completed.stderr, encoding="utf-8")
    try:
        raw = json.loads(report.read_text(encoding="utf-8"))
        summaries = raw["summary"]
        fastp_before = summaries["before_filtering"]
        fastp_after = summaries["after_filtering"]
        after = _statistics(outputs)
        for stats, reported in ((before, fastp_before), (after, fastp_after)):
            if (stats["reads"], stats["bases"]) != (
                reported["total_reads"],
                reported["total_bases"],
            ):
                raise ValueError("fastp report counts disagree with FASTQ files")
            for key in ("q20_bases", "q30_bases", "q20_rate", "q30_rate", "gc_content"):
                stats[key] = reported[key]
        duplication = raw["duplication"]["rate"]
        # fastp disables SE adapter cutting when autodetection finds no adapter;
        # its reporter then omits the section (jsonreporter.cpp).
        if "adapter_cutting" not in raw and reads2 is None and adapter_sequence is None:
            adapter_stats = {
                "enabled": False,
                "adapter_trimmed_reads": 0,
                "adapter_trimmed_bases": 0,
            }
        else:
            adapter_stats = {
                key: raw["adapter_cutting"][key]
                for key in ("adapter_trimmed_reads", "adapter_trimmed_bases")
            }
            adapter_stats["enabled"] = True
        filtering_result = raw["filtering_result"]
        version = summaries["fastp_version"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise OrganelleExecutionError(
            code="qc.invalid_fastp_report", message=f"Invalid fastp outputs: {error}"
        ) from error
    return _result(
        "filter_short_reads",
        {
            "before": before,
            "after": after,
            "paired": reads2 is not None,
            "duplication_rate": duplication,
            "adapter_cutting": adapter_stats,
            "filtering_result": filtering_result,
            "fastp_version": version,
            "quality_definition": "arithmetic_phred33_base_weighted",
            "output_paths": [str(path) for path in outputs],
            "json_report": str(report),
            "html_report": str(html),
            "log": str(out / "fastp.log"),
        },
        empty=after["reads"] == 0,
    )
