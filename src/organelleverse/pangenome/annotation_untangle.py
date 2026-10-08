"""Optional real ODGI annotation injection and reference-relative untangling."""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import asdict
from pathlib import Path

from ..core.artifacts import ArtifactRef
from ._gfa import _exact_overlap
from ._runner import run_command
from .annotation_projection import Annotation, normalize_annotations, project_annotations
from .graph import load_gfa, path_sequences, path_steps
from .graph_formats import _check_encoding
from .odgi_names import numeric_gfa, write_name_table


def annotations_from_result(result):
    """Reconstruct already-normalized compound loci from the gene-arrow table."""
    grouped = {}
    for row in result["gene_arrows"]:
        grouped.setdefault((row["path"], row["locus_id"]), []).append(row)
    features = []
    for (path, locus), parts in grouped.items():
        parts = sorted(parts, key=lambda row: row["part"])
        if len({row["gene"] for row in parts}) != 1:
            raise ValueError("Gene-arrow rows disagree about locus gene identity")
        features.append(
            Annotation(
                path=path,
                gene=parts[0]["gene"],
                locus_id=locus,
                parts=tuple((row["start"], row["end"]) for row in parts),
                strand=".",
                part_strands=tuple(row["strand"] for row in parts),
                source_locus_tag=parts[0].get("source_locus_tag"),
            )
        )
    return features


def run_odgi_untangle(
    gfa_path,
    annotations,
    reference_paths,
    output_directory,
    *,
    threads=1,
    min_jaccard=0.0,
    n_best=1,
    executable=None,
):
    """Inject each compound-feature part separately, preserving an identity map.

    ODGI's reference-relative segmentation is recorded as raw gggenes TSV.
    This optional algorithm does not establish complete gene copy counts. The
    exact shared-node mapper remains available independently of ODGI.
    """
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError("ODGI threads must be a positive integer")
    if (
        isinstance(n_best, bool)
        or not isinstance(n_best, int)
        or n_best < 1
        or not 0 <= min_jaccard <= 1
    ):
        raise ValueError("ODGI mapping count or Jaccard threshold is invalid")
    graph = load_gfa(gfa_path)
    if any(path.kind != "P" for path in graph.paths.values()) or any(
        _exact_overlap(edge.overlap) != 0 for edge in graph.links
    ):
        raise ValueError("ODGI annotation injection requires explicit zero-overlap P paths")
    if (
        not reference_paths
        or len(set(reference_paths)) != len(reference_paths)
        or set(reference_paths) - graph.paths.keys()
    ):
        raise ValueError("ODGI references must be unique named graph paths")
    features = normalize_annotations(annotations)
    if not features:
        raise ValueError("ODGI injection requires supplied annotation features")
    project_annotations(features, path_steps(gfa_path))
    sequences = path_sequences(gfa_path)
    tool = executable or shutil.which("odgi")
    if tool is None or not Path(tool).is_file():
        raise FileNotFoundError("ODGI annotation untangle requires an installed odgi executable")
    tool = str(Path(tool).resolve())
    directory = Path(output_directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError("ODGI output directory must be empty")
    commands = []

    def execute(arguments, stdout_name):
        record = run_command([tool, *arguments], cwd=directory, stdout_path=directory / stdout_name)
        commands.append(asdict(record))
        (directory / (stdout_name + ".stderr")).write_text(record.stderr)
        (directory / "odgi-commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        if not record.ok:
            raise RuntimeError(f"ODGI {arguments[0]} failed: {record.stderr[-2000:]}")

    execute(["version"], "odgi-version.txt")
    version = (directory / "odgi-version.txt").read_text().strip()
    bed = directory / "annotation-parts.bed"
    expected = {}
    identities = []
    for index, feature in enumerate(features):
        for part, (start, end) in enumerate(feature.parts):
            name = f"ov_annotation_{index}_part_{part}"
            if name in graph.paths:
                raise ValueError(
                    "Injected annotation identity collides with an existing graph path"
                )
            expected[name] = sequences[feature.path][start:end]
            identities.append(
                {
                    "injected_path": name,
                    "source_path": feature.path,
                    "gene": feature.gene,
                    "locus_id": feature.locus_id,
                    "part": part,
                    "start": start,
                    "end": end,
                    "biological_strand": feature.strand_for_part(part),
                }
            )
    bed.write_text(
        "".join(
            f"{row['source_path']}\t{row['start']}\t{row['end']}\t{row['injected_path']}\t0\t+\n"
            for row in identities
        )
    )
    # Inject forward path intervals; biological strand remains in the identity
    # table rather than relying on undocumented treatment of BED column six.
    (directory / "annotation-identities.json").write_text(json.dumps(identities, indent=2) + "\n")
    gene_paths = directory / "annotation-paths.txt"
    gene_paths.write_text("\n".join(expected) + "\n")
    references = directory / "reference-paths.txt"
    references.write_text("\n".join(reference_paths) + "\n")
    og = directory / "annotated.og"
    # ODGI reads segment names as integers; path names and sequences are unchanged
    odgi_input = Path(gfa_path).resolve()
    numeric = directory / "graph.numeric.gfa"
    renamed = numeric_gfa(odgi_input, numeric)
    if renamed is not None:
        write_name_table(renamed, directory / "node_names.tsv")
        odgi_input = numeric
    execute(
        [
            "inject",
            "-i",
            str(odgi_input),
            "-b",
            str(bed),
            "-o",
            str(og),
            "-t",
            str(threads),
        ],
        "odgi-inject.txt",
    )
    _check_encoding(og, "og")
    injected_gfa = directory / "annotated.gfa"
    execute(["view", "-i", str(og), "-g"], injected_gfa.name)
    mapped = path_sequences(injected_gfa)
    if mapped != {**sequences, **expected}:
        raise RuntimeError(
            "ODGI injection did not preserve source paths and exact annotation intervals"
        )
    execute(
        [
            "untangle",
            "-i",
            str(og),
            "-R",
            str(gene_paths),
            "-Q",
            str(references),
            "-j",
            str(min_jaccard),
            "-n",
            str(n_best),
            "-t",
            str(threads),
            "-g",
        ],
        "untangle.tsv",
    )
    output = directory / "untangle.tsv"
    with output.open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != ["molecule", "gene", "start", "end", "strand"]:
            raise RuntimeError("ODGI untangle did not emit a gggenes table header")
        row_count = 0
        for row in reader:
            if row["molecule"] not in reference_paths or row["gene"] not in expected:
                raise RuntimeError("ODGI untangle table refers to unknown paths")
            start, end = int(row["start"]), int(row["end"])
            if not 0 <= start < end <= len(sequences[row["molecule"]]) or row["strand"] not in {
                "0",
                "1",
            }:
                raise RuntimeError("ODGI untangle emitted invalid coordinates or strand")
            row_count += 1
    result = {
        "method": "odgi_inject_untangle",
        "tool_version": version,
        "executable": tool,
        "source_graph": ArtifactRef.from_path(
            Path(gfa_path).resolve(), kind="pangenome_graph", format="gfa"
        ).model_dump(mode="json"),
        "reference_paths": list(reference_paths),
        "annotation_part_count": len(identities),
        "mapping_row_count": row_count,
        "parameters": {"threads": threads, "min_jaccard": min_jaccard, "n_best": n_best},
        "commands": commands,
        "interpretation": "ODGI segmented mappings to injected forward annotation-part paths; biological strands and compound identities are preserved separately. Mapping rows are not complete gene copy counts.",
    }
    (directory / "odgi-untangle.json").write_text(json.dumps(result, indent=2) + "\n")
    result["files"] = sorted(str(path) for path in directory.iterdir() if path.is_file())
    return result
