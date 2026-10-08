"""Organelle read recruitment before assembly (``ovasm recruit``).

Whole-genome long-read libraries hold organelle DNA at a few percent of the bases but
hundreds to thousands of fold depth. Recruiting it once and downsampling to a target
depth lets every assembly backend run on a small read set.

Typical use::

    recruited, report = recruit_long_reads(
        Path("sample.hifi.fastq.gz"),
        technology="pacbio_hifi",
        quality_state="ccs",
        seeds={"mitochondrion": [Path("related_mt.fa")], "plastid": [Path("related_cp.fa")]},
        out_dir=Path("recruit"),
    )
    result = assemble(recruited, organelle="mitochondrion", method="himt")
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from .._ovasm import OvasmPreset, run_recruit
from ..core.data import OrganelleData
from ..io_reads import read_long_reads

__all__ = ["preset_for", "recruit_long_reads"]

Technology = Literal["pacbio_hifi", "pacbio_clr", "ont"]
QualityState = Literal["raw", "corrected", "duplex", "hq", "ccs"]


def preset_for(technology: Technology, quality_state: QualityState) -> OvasmPreset:
    """HiFi-like accuracy uses exact 21-mers; noisier long reads use the ONT preset."""
    if technology == "pacbio_hifi" or quality_state in {"ccs", "duplex", "hq"}:
        return "hifi"
    return "ont"


def _merge_unique(sources: Sequence[Path], target: Path) -> int:
    """Concatenate FASTQs, keeping the first copy of each read id.

    A read tied between targets (identical MTPT/NUPT sequence) is written to each target's
    file by ovasm; merging them must not double its depth.
    """
    seen: set[bytes] = set()
    kept = 0
    with target.open("wb") as out:
        for source in sources:
            with source.open("rb") as fh:
                while True:
                    header = fh.readline()
                    if not header:
                        break
                    record = header + fh.readline() + fh.readline() + fh.readline()
                    read_id = header[1:].split(maxsplit=1)[0] if header[1:].strip() else header
                    if read_id in seen:
                        continue
                    seen.add(read_id)
                    out.write(record)
                    kept += 1
    return kept


def recruit_long_reads(
    reads: Path | Sequence[Path],
    *,
    technology: Technology,
    quality_state: QualityState,
    seeds: Mapping[str, Sequence[Path]],
    out_dir: Path,
    target_depth: float | None = 150.0,
    include_targets: Sequence[str] | None = None,
    salt: int = 0,
    iterations: int | None = None,
    threads: int | None = None,
) -> tuple[OrganelleData, dict[str, Any]]:
    """Recruit organelle reads with ovasm and return them as a long-read library.

    ``seeds`` maps a target name (e.g. ``"mitochondrion"``, ``"plastid"``) to seed FASTA
    files from the same or a related species. Seeding both organelles lets tied MTPT reads
    be recognised. ``include_targets`` selects which targets go into the returned library
    (default: all), each downsampled to ``target_depth``; tied reads appear once.
    """
    inputs = [p.resolve() for p in ([reads] if isinstance(reads, Path) else list(reads))]
    # Absolute paths: assembly backends run in their own working directories, so a relative
    # path recorded in the returned library would not resolve there.
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report = run_recruit(
        inputs,
        seeds={name: [str(p) for p in paths] for name, paths in seeds.items()},
        out_dir=out_dir,
        preset=preset_for(technology, quality_state),
        target_depth=target_depth,
        salt=salt,
        iterations=iterations,
        threads=threads,
    )
    wanted = (
        list(include_targets)
        if include_targets is not None
        else [o["name"] for o in report["outputs"]]
    )
    unknown = sorted(set(wanted) - {o["name"] for o in report["outputs"]})
    if unknown:
        raise ValueError(f"include_targets not among seeded targets: {unknown}")
    outputs = {o["name"]: Path(o["output"]) for o in report["outputs"]}
    merged = out_dir / "recruited.fastq"
    report["merged_reads"] = _merge_unique([outputs[name] for name in wanted], merged)
    report["merged_output"] = str(merged)
    report["merged_targets"] = wanted
    return read_long_reads(merged, technology=technology, quality_state=quality_state), report
