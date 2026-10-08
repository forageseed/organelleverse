"""Managed, validated GFA/ODGI/VG conversions through real external tools.

Only explicit zero-overlap P graphs are converted to binary graph formats.
Roundtrip validation requires unchanged segment identities, topology, traversals,
and path sequences. General overlap graphs and W coordinate conversion require
an explicit normalization policy and are rejected rather than silently changed.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Literal
from uuid import uuid4

from ..core.artifacts import ArtifactRef
from ..core.errors import OrganelleDependencyError, OrganelleExecutionError, OrganelleInputError
from ..core.result import OrganelleResult
from ..runtime import create_staged_run, managed_run_path, publish_staged_result
from ._contract import make_provenance, utc_now
from ._gfa import _exact_overlap, _flip
from ._runner import CommandRecord, run_command
from .graph import graph_statistics, load_gfa, path_sequences
from .odgi_names import numeric_gfa

_OPERATION = "pangenome.convert_graph"


def _run(
    argv: list[str], cwd: Path, records: list[dict], stdout: Path | None = None
) -> CommandRecord:
    record = run_command(argv, cwd=cwd, stdout_path=stdout)
    records.append(asdict(record))
    if not record.ok:
        raise OrganelleExecutionError(
            code="pangenome.conversion_failed",
            message="Graph conversion backend failed",
            details={"argv": argv, "returncode": record.returncode, "stderr": record.stderr},
        )
    return record


def _fingerprint(path: Path, circularity: bool = True) -> tuple:
    """Segments, bidirected adjacency, path traversals and sequences.

    ``circularity`` off drops each path's circular flag. ODGI keeps no path tags, so
    ``TP:Z:circular`` does not survive an OG roundtrip: the overview, which only draws the
    graph, compares without it and lists the circular paths in its evidence; a conversion,
    whose OG is the product, still fails rather than lose the flag."""
    graph = load_gfa(path)
    if any(_exact_overlap(link.overlap) != 0 for link in graph.links):
        raise OrganelleInputError(
            code="pangenome.conversion_overlaps",
            message="Binary conversion supports zero-overlap graphs only",
        )
    if any(segment.sequence is None for segment in graph.segments.values()):
        raise OrganelleInputError(
            code="pangenome.conversion_missing_sequence",
            message="Binary conversion requires stored segment sequences",
        )
    # Parallel L records may be deduplicated by handlegraph formats; topology
    # here is the bidirected adjacency set, not redundant record multiplicity.
    edges = {
        min((link.source, link.target), (_flip(link.target), _flip(link.source)))
        for link in graph.links
    }
    sequences = path_sequences(path) if graph.paths else {}
    identities = _path_identities(graph, sequences)
    return (
        {name: segment.sequence.upper() for name, segment in graph.segments.items()},
        edges,
        {
            identities[name]: (
                record.steps,
                circularity and "TP:Z:circular" in record.raw.split("\t"),
            )
            for name, record in graph.paths.items()
        },
        {identities[name]: sequence for name, sequence in sequences.items()},
    )


def circular_paths(path: Path) -> list[str]:
    """Names of the paths a GFA marks circular (``TP:Z:circular``)."""
    return sorted(
        name
        for name, record in load_gfa(path).paths.items()
        if "TP:Z:circular" in record.raw.split("\t")
    )


def _path_identities(graph, sequences: dict[str, str]) -> dict[str, tuple]:
    identities = {}
    for name, record in graph.paths.items():
        if record.haplotype is None:
            identities[name] = ("path", name)
        else:
            start = 0 if record.kind == "P" else record.start
            end = len(sequences[name]) if record.kind == "P" else record.end
            if start is None or end is None:
                raise OrganelleInputError(
                    code="pangenome.conversion_walk_coordinates",
                    message="Walk interval bounds are required for conversion validation",
                )
            identities[name] = (
                "haplotype",
                record.sample,
                record.haplotype,
                record.molecule,
                start,
                end,
            )
    if len(set(identities.values())) != len(identities):
        raise OrganelleInputError(
            code="pangenome.ambiguous_path_identity",
            message="Graph paths have duplicate sample/haplotype/molecule interval identity",
        )
    return identities


def _check_encoding(path: Path, fmt: str) -> None:
    """Check ODGI network-order magic or VG tagged protobuf stream framing."""
    with path.open("rb") as stream:
        if fmt == "og":
            # odgi graph_t::get_magic_number (v0.9.2 source), uint32 network order.
            valid = stream.read(4) == (1988148666).to_bytes(4, "big")
        elif fmt == "vg":

            def varint() -> int:
                number = 0
                for shift in range(0, 70, 7):
                    byte = stream.read(1)
                    if not byte:
                        return -1
                    number |= (byte[0] & 127) << shift
                    if byte[0] < 128:
                        return number
                return -1

            group_size = varint()
            tag_length = varint()
            valid = group_size > 0 and tag_length == 2 and stream.read(2) == b"VG"
        else:
            return
    if not valid:
        raise OrganelleInputError(
            code="pangenome.graph_encoding_mismatch",
            message=f"Graph bytes do not match declared {fmt} encoding",
        )


def convert_graph(
    graph: ArtifactRef, *, target_format: Literal["gfa", "og", "vg"], threads: int = 1
) -> OrganelleResult:
    """Convert GFA→OG, OG→GFA or GFA→VG protobuf in a managed output run.

    Dependencies are resolved from PATH; no executable injection or destination
    path is accepted. Binary outputs must load and survive a GFA roundtrip.
    """
    route = (graph.format, target_format)
    if route not in {("gfa", "og"), ("og", "gfa"), ("gfa", "vg")}:
        raise OrganelleInputError(
            code="pangenome.unsupported_conversion",
            message="Supported conversions are GFA→OG, OG→GFA and GFA→VG",
        )
    if isinstance(threads, bool) or not isinstance(threads, int) or not 1 <= threads <= 256:
        raise OrganelleInputError(
            code="pangenome.invalid_threads", message="threads must be an integer in [1,256]"
        )
    source = graph.resolve().resolve()
    actual = ArtifactRef.from_path(source, kind=graph.kind, format=graph.format)
    if actual.sha256 != graph.sha256 or actual.size_bytes != graph.size_bytes:
        raise OrganelleInputError(
            code="pangenome.source_digest_mismatch", message="Graph input changed after capture"
        )
    program = "vg" if target_format == "vg" else "odgi"
    executable = shutil.which(program)
    if not executable:
        raise OrganelleDependencyError(
            code="pangenome.missing_converter",
            message=f"Required graph converter is unavailable: {program}",
            details={"program": program},
        )
    if graph.format == "gfa" and any(p.kind != "P" for p in load_gfa(source).paths.values()):
        raise OrganelleInputError(
            code="pangenome.conversion_walk_coordinates",
            message="Binary conversion requires explicit P paths; W input needs an explicit normalization policy",
        )
    _check_encoding(source, graph.format)
    expected = _fingerprint(source) if graph.format == "gfa" else None
    started = utc_now()
    records: list[dict] = []
    # Decoder output is validation evidence, not a duplicate source artifact.
    with tempfile.TemporaryDirectory(prefix="organelleverse-conversion-") as temporary:
        scratch = Path(temporary)
        version_output = scratch / "version.txt"
        version_record = _run([executable, "version"], scratch, records, version_output)
        version = version_output.read_text().strip() or version_record.stderr.strip()
        candidate = scratch / f"graph.{target_format}"
        roundtrip = scratch / "roundtrip.gfa"
        renamed = None
        if route == ("gfa", "og"):
            # ODGI reads segment names as integers; the roundtrip is held to the renumbered copy
            numeric = scratch / "numeric.gfa"
            renamed = numeric_gfa(source, numeric)
            odgi_input = source
            expected = _fingerprint(numeric if renamed is not None else source)
            if renamed is not None:
                odgi_input = numeric
            _run(
                [
                    executable,
                    "build",
                    "-g",
                    str(odgi_input),
                    "-o",
                    str(candidate),
                    "-t",
                    str(threads),
                ],
                scratch,
                records,
            )
            _run([executable, "view", "-i", str(candidate), "-g"], scratch, records, roundtrip)
        elif route == ("og", "gfa"):
            _run([executable, "view", "-i", str(source), "-g"], scratch, records, candidate)
            roundtrip = candidate
        else:
            _run(
                [executable, "convert", "-g", "-v", "-t", str(threads), str(source)],
                scratch,
                records,
                candidate,
            )
            _run([executable, "convert", "-f", str(candidate)], scratch, records, roundtrip)
        if not candidate.is_file() or not candidate.stat().st_size:
            raise OrganelleExecutionError(
                code="pangenome.empty_conversion", message="Converter returned no graph artifact"
            )
        _check_encoding(candidate, target_format)
        observed = _fingerprint(roundtrip)
        if expected is not None and expected != observed:
            raise OrganelleExecutionError(
                code="pangenome.conversion_changed_graph",
                message="Conversion changed segment identities, topology, paths or sequences",
            )
        statistics = graph_statistics(roundtrip)
        mapping = {}
        if expected is not None:
            source_identities = (
                _path_identities(load_gfa(source), path_sequences(source))
                if load_gfa(source).paths
                else {}
            )
            target_identities = (
                _path_identities(load_gfa(roundtrip), path_sequences(roundtrip))
                if load_gfa(roundtrip).paths
                else {}
            )
            target_names = {identity: name for name, identity in target_identities.items()}
            mapping = {name: target_names[identity] for name, identity in source_identities.items()}
        run_id = uuid4().hex
        staging = create_staged_run(_OPERATION, run_id)
        destination_graph = staging / candidate.name
        shutil.copyfile(candidate, destination_graph)
        evidence = staging / "conversion.json"
        evidence.write_text(
            json.dumps(
                {
                    "input": graph.model_dump(mode="json"),
                    "target_format": target_format,
                    "encoding": "VG Protobuf" if target_format == "vg" else target_format,
                    "backend": program,
                    "version": version,
                    "commands": records,
                    "path_mapping": mapping,
                    # ODGI node number -> source segment name, when the names were not numbers
                    "node_names": renamed,
                    "validation": {
                        "method": "backend GFA decoding and strict graph parsing",
                        "source_equivalence": expected is not None,
                        "compared": [
                            "segments",
                            "bidirected adjacency",
                            "path traversals",
                            "path sequences",
                        ]
                        if expected is not None
                        else [],
                        "statistics": statistics,
                    },
                },
                indent=2,
            )
            + "\n"
        )
        artifacts = tuple(
            ArtifactRef.from_path(
                file,
                kind="pangenome_graph" if file == destination_graph else "conversion_evidence",
                format=file.suffix.lstrip("."),
                media_type="text/plain"
                if file.suffix == ".gfa"
                else "application/json"
                if file.suffix == ".json"
                else "application/octet-stream",
            ).model_copy(update={"uri": file.name})
            for file in (destination_graph, evidence)
        )
        result = OrganelleResult(
            operation_id=_OPERATION,
            operation_version="1.0",
            scope="mixed",
            status="ok",
            summary_text=f"Converted {graph.format} to {target_format} with verified graph structure.",
            artifacts=artifacts,
            metrics={"nodes": statistics["nodes"], "paths": statistics["paths"]},
            provenance=make_provenance(
                operation_id=_OPERATION,
                parameters={"target_format": target_format, "threads": threads},
                input_artifact_hashes=(graph.sha256,),
                requested_backend=program,
                actual_backend=program,
                attempted_backends=(program,),
                started_at=started,
                finished_at=utc_now(),
            ).model_copy(update={"operation_version": "1.0"}),
        )
        destination = managed_run_path(_OPERATION, run_id)
        published = publish_staged_result(
            result,
            staging,
            destination,
            trusted_input_hashes=frozenset({graph.sha256}),
        )
        (destination / "result.json").write_text(published.model_dump_json(indent=2) + "\n")
        return published
