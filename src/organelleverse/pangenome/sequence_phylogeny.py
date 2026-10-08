"""Validated aligned DNA/MAF input and real RAxML-NG execution.

MAF blocks retain input order and gaps for missing taxa; no block reorientation,
realignment, silent paralog choice, or automatic model selection is performed.
"""

from __future__ import annotations

import csv
import json
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from Bio import AlignIO, Phylo, SeqIO

from ..core.artifacts import ArtifactRef
from ._runner import run_command


@dataclass(frozen=True)
class RaxmlOptions:
    model: str = "GTR+G"
    bootstrap_replicates: int = 100
    seed: int = 1
    threads: int = 1
    parsimony_starts: int = 10
    random_starts: int = 10

    def __post_init__(self):
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9]*(?:\+[A-Za-z][A-Za-z0-9]*)*", self.model):
            raise ValueError(
                "Model must be an explicit RAxML model expression, not a partition filename"
            )
        for name, value in asdict(self).items():
            if name == "model":
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("RAxML numerical options must be nonnegative integers")
        if self.threads < 1 or self.seed < 1 or self.parsimony_starts + self.random_starts < 1:
            raise ValueError("RAxML requires positive threads, seed and at least one starting tree")


def alignment_records(path: str | Path, *, format: str = "fasta", source_mapping=None):
    """Return aligned records and coordinate provenance for each MAF block row."""
    source_mapping = source_mapping or {}
    coordinate_rows = []
    if format == "fasta":
        records = [
            (source_mapping.get(r.id, r.id), str(r.seq).upper()) for r in SeqIO.parse(path, "fasta")
        ]
    elif format == "maf":
        # The MAF iterator streams blocks; final MSA materialization is necessary
        # for the external likelihood engine, not duplicated into Result metrics.
        blocks = []
        taxa = []
        offset = 0
        for block_index, alignment in enumerate(AlignIO.parse(path, "maf")):
            block = {}
            width = alignment.get_alignment_length()
            for record in alignment:
                label = source_mapping.get(record.id, record.id)
                if label in block:
                    raise ValueError(
                        "Multiple MAF rows map to one taxon in a block; paralog choice must be explicit"
                    )
                sequence = str(record.seq).upper()
                start, size, strand, source_size = (
                    record.annotations[k] for k in ("start", "size", "strand", "srcSize")
                )
                if (
                    size != len(sequence.replace("-", ""))
                    or start < 0
                    or start + size > source_size
                    or strand not in {1, -1}
                ):
                    raise ValueError("MAF row coordinates or ungapped size are invalid")
                forward_start = start if strand == 1 else source_size - start - size
                coordinate_rows.append(
                    {
                        "block": block_index,
                        "taxon": label,
                        "source_id": record.id,
                        "source_start": forward_start,
                        "source_end": forward_start + size,
                        "source_size": source_size,
                        "strand": "+" if strand == 1 else "-",
                        "alignment_start": offset,
                        "alignment_end": offset + width,
                    }
                )
                block[label] = sequence
                if label not in taxa:
                    taxa.append(label)
            blocks.append((width, block))
            offset += width
        records = [
            (taxon, "".join(block.get(taxon, "-" * width) for width, block in blocks))
            for taxon in taxa
        ]
    else:
        raise ValueError("Alignment format must be fasta or maf")
    names = [name for name, _seq in records]
    if not names or len(set(names)) != len(names):
        raise ValueError("Alignment taxon labels must be unique and nonempty")
    if any(not re.fullmatch(r"[A-Za-z0-9_.#|-]+", name) for name in names):
        raise ValueError(
            "Alignment labels contain unsupported Newick characters; provide explicit source_mapping"
        )
    lengths = {len(sequence) for _, sequence in records}
    if len(lengths) != 1 or 0 in lengths:
        raise ValueError("DNA alignment must have equal nonzero sequence lengths")
    if any(set(sequence) - set("ACGTRYSWKMBDHVN?-") for _, sequence in records):
        raise ValueError("Alignment must contain IUPAC DNA, gap or missing-data symbols")
    if any(not set(sequence) & set("ACGT") for _, sequence in records):
        raise ValueError("Every aligned taxon must have observed nucleotide data")
    return records, coordinate_rows


def run_raxml_ng(
    alignment_path: str | Path,
    output_directory: str | Path,
    *,
    format: str = "fasta",
    options: RaxmlOptions | None = None,
    source_mapping=None,
    executable: str | None = None,
) -> dict:
    """Execute a real likelihood search in an empty stage directory.

    Required outputs and tip identities are verified before returning. The
    caller's managed runtime publishes the returned regular files as artifacts.
    """
    options = options or RaxmlOptions()
    records, coordinates = alignment_records(
        alignment_path, format=format, source_mapping=source_mapping
    )
    if len(records) < 4:
        raise ValueError("RAxML-NG phylogeny requires at least four aligned taxa")
    tool = executable or shutil.which("raxml-ng")
    if tool is None or not Path(tool).is_file():
        raise FileNotFoundError(
            "Optional sequence phylogeny requires an installed raxml-ng executable"
        )
    tool = str(Path(tool).resolve())
    directory = Path(output_directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError("RAxML output directory must be empty to avoid stale result reuse")
    msa = directory / "alignment.fasta"
    msa.write_text("".join(f">{name}\n{sequence}\n" for name, sequence in records))
    coordinate_path = directory / "alignment-coordinates.tsv"
    if coordinates:
        with coordinate_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(coordinates[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(coordinates)
    version_path = directory / "raxml-version.txt"
    version_record = run_command([tool, "--version"], cwd=directory, stdout_path=version_path)
    version_text = version_path.read_text()
    if not version_record.ok or "RAxML-NG" not in version_text:
        raise RuntimeError("RAxML-NG version probe failed or returned an unexpected program")
    starts = []
    if options.parsimony_starts:
        starts.append(f"pars{{{options.parsimony_starts}}}")
    if options.random_starts:
        starts.append(f"rand{{{options.random_starts}}}")
    prefix = directory / "sequence-tree"
    argv = [
        tool,
        "--all" if options.bootstrap_replicates else "--search",
        "--msa",
        str(msa),
        "--msa-format",
        "FASTA",
        "--data-type",
        "DNA",
        "--model",
        options.model,
        "--prefix",
        str(prefix),
        "--seed",
        str(options.seed),
        "--threads",
        str(options.threads),
        "--tree",
        ",".join(starts),
    ]
    if options.bootstrap_replicates:
        argv.extend(["--bs-trees", str(options.bootstrap_replicates), "--bs-metric", "fbp"])
    command = run_command(argv, cwd=directory, stdout_path=directory / "raxml-stdout.txt")
    (directory / "raxml-stderr.txt").write_text(command.stderr)
    execution = {
        "argv": list(command.argv),
        "returncode": command.returncode,
        "resource_usage": command.resource_usage,
    }
    (directory / "raxml-command.json").write_text(json.dumps(execution, indent=2) + "\n")
    if not command.ok:
        raise RuntimeError(
            f"RAxML-NG failed with exit {command.returncode}: {command.stderr[-2000:]}"
        )
    tree_path = Path(
        str(prefix) + (".raxml.support" if options.bootstrap_replicates else ".raxml.bestTree")
    )
    required = [tree_path, Path(str(prefix) + ".raxml.bestModel"), Path(str(prefix) + ".raxml.log")]
    if options.bootstrap_replicates:
        required.append(Path(str(prefix) + ".raxml.bootstraps"))
    if any(not path.is_file() or path.stat().st_size == 0 for path in required):
        raise RuntimeError("RAxML-NG did not create its required nonempty result files")
    tree = Phylo.read(tree_path, "newick")
    if sorted(t.name for t in tree.get_terminals()) != sorted(name for name, _ in records):
        raise RuntimeError("RAxML-NG tree taxon identities differ from the input alignment")
    if options.bootstrap_replicates:
        trees = list(Phylo.parse(Path(str(prefix) + ".raxml.bootstraps"), "newick"))
        if len(trees) != options.bootstrap_replicates:
            raise RuntimeError("RAxML-NG bootstrap replicate count differs from requested count")
    source = ArtifactRef.from_path(
        Path(alignment_path).resolve(), kind="sequence_alignment", format=format
    )
    result = {
        "method": "RAxML-NG maximum likelihood",
        "mode": "msa",
        "newick": tree_path.read_text().strip(),
        "paths": [name for name, _ in records],
        "alignment_sites": len(records[0][1]),
        "model": options.model,
        "fitted_model": required[1].read_text().strip(),
        "tool_version": version_text.strip(),
        "executable": tool,
        "parameters": asdict(options),
        "source_alignment": source.model_dump(mode="json"),
        "source_mapping": source_mapping or {},
        "alignment_policy": "MAF input block order; aligned orientation retained; missing taxa gap-filled; one row per taxon per block",
        "bootstrap_method": "Felsenstein nonparametric alignment-site bootstrap"
        if options.bootstrap_replicates
        else None,
        "resource_usage": command.resource_usage,
        "command": execution,
    }
    (directory / "sequence-phylogeny.json").write_text(json.dumps(result, indent=2) + "\n")
    result["files"] = sorted(str(path) for path in directory.iterdir() if path.is_file())
    return result
