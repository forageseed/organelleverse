"""Full-graph ODGI views with retained layouts and external-command evidence.

The 1D axis is sorted graph sequence and 2D positions are ODGI layout coordinates,
not genome coordinates. Only explicit, sequenced, zero-overlap P graphs are
supported; W intervals need a normalization policy before ODGI conversion.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from xml.etree import ElementTree

from ..core.artifacts import ArtifactRef
from ..core.errors import OrganelleDependencyError, OrganelleExecutionError, OrganelleInputError
from ._contract import utc_now
from ._gfa import _exact_overlap
from ._runner import run_command
from .graph import graph_statistics, load_gfa, path_sequences
from .graph_formats import _check_encoding, _fingerprint, circular_paths
from .odgi_names import numeric_gfa, write_name_table


def _validate_image(path: Path) -> None:
    if path.suffix == ".png":
        with path.open("rb") as handle:
            header = handle.read(24)
        valid = (
            len(header) == 24
            and header[:8] == b"\x89PNG\r\n\x1a\n"
            and header[12:16] == b"IHDR"
            and int.from_bytes(header[16:20], "big") > 0
            and int.from_bytes(header[20:24], "big") > 0
        )
    else:
        try:
            root = ElementTree.parse(path).getroot()
            valid = root.tag == "{http://www.w3.org/2000/svg}svg" and len(root) > 0
        except ElementTree.ParseError:
            valid = False
    if not valid:
        raise OrganelleExecutionError(
            code="pangenome.invalid_overview_image",
            message=f"ODGI output is not a valid {path.suffix[1:].upper()} image: {path.name}",
        )


def render_odgi_overview(
    graph_path: str | Path, output_dir: str | Path, threads: int = 4
) -> list[Path]:
    """Render all nodes and paths, retaining OG, layouts and provenance files.

    ``output_dir`` must be empty. Source conversion is roundtrip-checked before
    sorting. Sorting compacts node IDs, so its decoded GFA is retained as the
    node-ID reference for the exported layout. Byte-identical decoded GFA files
    reuse the source artifact instead of publishing duplicate source bytes.
    Command resources retain the runner's measured CPU/RSS and their scope.
    """
    if isinstance(threads, bool) or not isinstance(threads, int) or not 1 <= threads <= 256:
        raise OrganelleInputError(
            code="pangenome.invalid_threads", message="threads must be an integer in [1,256]"
        )
    source = Path(graph_path).expanduser().resolve()
    graph = load_gfa(source)
    if not graph.paths or any(path.kind != "P" for path in graph.paths.values()):
        raise OrganelleInputError(
            code="pangenome.overview_requires_p_paths",
            message="ODGI overview requires explicit P paths; W input needs an explicit normalization policy",
        )
    if any(_exact_overlap(link.overlap) != 0 for link in graph.links) or any(
        _exact_overlap(overlap) != 0
        for path in graph.paths.values()
        for overlap in (path.overlaps or ())
        if overlap != "*"
    ):
        raise OrganelleInputError(
            code="pangenome.overview_overlaps",
            message="ODGI overview supports zero-overlap graphs only",
        )
    expected = _fingerprint(source)  # Also rejects absent stored segment sequence.
    executable = shutil.which("odgi")
    if not executable:
        raise OrganelleDependencyError(
            code="pangenome.missing_odgi", message="ODGI is required to generate graph overviews"
        )
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise OrganelleInputError(
            code="pangenome.overview_output_exists",
            message="ODGI overview output directory must be empty",
        )
    # ODGI reads segment names as integers: give it a renumbered copy (with the name table)
    # and hold its roundtrip to that copy.
    numeric = output / "graph.numeric.gfa"
    renamed = numeric_gfa(source, numeric)
    odgi_input = source
    if renamed is not None:
        write_name_table(renamed, output / "node_names.tsv")
        odgi_input = numeric
    # ODGI keeps no path tags, so circularity is compared on the source only (recorded below)
    expected = _fingerprint(odgi_input, circularity=False)
    source_ref = ArtifactRef.from_path(source, kind="pangenome_graph", format="gfa")
    evidence: dict = {
        "source": source_ref.model_dump(mode="json"),
        "artifact_uri_base": "directory containing overview.evidence.json; reused source references are absolute",
        "started_at": utc_now().isoformat(),
        "status": "running",
        "software": {"executable": executable},
        "resources": {"requested_threads": threads},
        "commands": [],
        "validation": {},
        "node_names": {
            "renumbered": renamed is not None,
            "table": "node_names.tsv" if renamed is not None else None,
        },
        "circular_paths": circular_paths(source),
        "interpretation": {
            "one_dimensional": "Sorted graph sequence axis; rows are stored graph paths, not an alignment coordinate system.",
            "two_dimensional": "ODGI path-guided SGD layout; positions and pixel distances are not genomic coordinates or phylogenetic distances.",
            "node_ids": "Sorting compacts graph node IDs; use decoded_graphs.sorted with the retained layouts. TSV idx is ODGI's native layout endpoint index, not a source GFA node ID.",
            "determinism": "ODGI layout uses its default stochastic algorithm; multithreaded layouts need not be pixel-identical across runs.",
        },
    }
    names = [
        "odgi.version.txt",
        "graph.og",
        "graph.roundtrip.gfa",
        "graph.sorted.og",
        "graph.sorted.gfa",
        "graph_1d.png",
        "graph_2d.lay",
        "graph_2d.tsv",
        "graph_2d.png",
        "graph_2d.svg",
    ]
    paths = {name: output / name for name in names}
    evidence_path = output / "overview.evidence.json"
    started = perf_counter()

    def run(args: list[str], stdout: Path | None = None) -> None:
        record = run_command([executable, *args], cwd=output, stdout_path=stdout)
        evidence["commands"].append({**asdict(record), "resource_usage": record.resource_usage})
        if not record.ok:
            raise OrganelleExecutionError(
                code="pangenome.overview_command_failed",
                message=f"ODGI {args[0]} failed",
                details={
                    "argv": list(record.argv),
                    "returncode": record.returncode,
                    "stderr": record.stderr,
                },
            )

    try:
        run(["version"], paths["odgi.version.txt"])
        evidence["software"]["version"] = paths["odgi.version.txt"].read_text().strip()
        run(["build", "-g", str(odgi_input), "-o", str(paths["graph.og"]), "-t", str(threads)])
        _check_encoding(paths["graph.og"], "og")
        run(["view", "-i", str(paths["graph.og"]), "-g"], paths["graph.roundtrip.gfa"])
        if _fingerprint(paths["graph.roundtrip.gfa"], circularity=False) != expected:
            raise OrganelleExecutionError(
                code="pangenome.overview_conversion_changed_graph",
                message="ODGI conversion changed source nodes, topology, paths or sequences",
            )
        evidence["validation"]["source_roundtrip_equivalent"] = True
        run(
            [
                "sort",
                "-i",
                str(paths["graph.og"]),
                "-o",
                str(paths["graph.sorted.og"]),
                "-b",
                "-O",
                "-t",
                str(threads),
            ]
        )
        _check_encoding(paths["graph.sorted.og"], "og")
        run(["view", "-i", str(paths["graph.sorted.og"]), "-g"], paths["graph.sorted.gfa"])
        sorted_graph = load_gfa(paths["graph.sorted.gfa"])
        if set(sorted_graph.segments) != {str(i) for i in range(1, len(graph.segments) + 1)}:
            raise OrganelleExecutionError(
                code="pangenome.overview_not_compacted",
                message="ODGI sorting did not compact graph node IDs",
            )
        if path_sequences(source) != path_sequences(paths["graph.sorted.gfa"]):
            raise OrganelleExecutionError(
                code="pangenome.overview_sort_changed_paths",
                message="ODGI sorting changed path names or spelled sequences",
            )
        evidence["validation"].update(
            {"sorted_ids_compact": True, "sorted_path_sequences_equivalent": True}
        )
        stats = graph_statistics(paths["graph.sorted.gfa"])
        evidence["graph_statistics"] = {
            key: stats[key] for key in ("nodes", "edges", "paths", "components", "total_bp")
        }
        run(
            [
                "viz",
                "-i",
                str(paths["graph.sorted.og"]),
                "-o",
                str(paths["graph_1d.png"]),
                "-x",
                "1500",
                "-y",
                "500",
                "-t",
                str(threads),
            ]
        )
        run(
            [
                "layout",
                "-i",
                str(paths["graph.sorted.og"]),
                "-o",
                str(paths["graph_2d.lay"]),
                "-T",
                str(paths["graph_2d.tsv"]),
                "-t",
                str(threads),
            ]
        )
        run(
            [
                "draw",
                "-i",
                str(paths["graph.sorted.og"]),
                "-c",
                str(paths["graph_2d.lay"]),
                "-p",
                str(paths["graph_2d.png"]),
                "-s",
                str(paths["graph_2d.svg"]),
                "-t",
                str(threads),
            ]
        )
        for path in paths.values():
            if not path.is_file() or not path.stat().st_size:
                raise OrganelleExecutionError(
                    code="pangenome.empty_overview_output",
                    message=f"ODGI produced no output: {path.name}",
                )
            if path.suffix in {".png", ".svg"}:
                _validate_image(path)
        evidence["validation"]["image_encodings_valid"] = True
        evidence["artifacts"] = []
        evidence["decoded_graphs"] = {}
        for path in list(paths.values()):
            artifact = ArtifactRef.from_path(
                path,
                kind="pangenome_overview",
                format=path.suffix[1:],
                media_type={
                    ".png": "image/png",
                    ".svg": "image/svg+xml",
                    ".tsv": "text/tab-separated-values",
                }.get(path.suffix, "application/octet-stream"),
            )
            if path.suffix == ".gfa":
                reuse_source = artifact.sha256 == source_ref.sha256
                evidence["decoded_graphs"]["sorted" if "sorted" in path.name else "roundtrip"] = {
                    "artifact": (
                        source_ref
                        if reuse_source
                        else artifact.model_copy(update={"uri": path.name})
                    ).model_dump(mode="json"),
                    "reuses_source": reuse_source,
                }
                if reuse_source:
                    path.unlink()
                    del paths[path.name]
                    continue
            evidence["artifacts"].append(
                artifact.model_copy(update={"uri": path.name}).model_dump(mode="json")
            )
        evidence["status"] = "succeeded"
    except Exception as error:
        evidence["status"] = "failed"
        evidence["error"] = str(error)
        raise
    finally:
        evidence["finished_at"] = utc_now().isoformat()
        evidence["resources"]["wall_seconds"] = perf_counter() - started
        evidence_path.write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n")
    return [*paths.values(), evidence_path]
