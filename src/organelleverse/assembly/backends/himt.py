"""HiMT long-read organelle assembly adapter."""

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING

from organelleverse.assembly.backends.base import (
    AdapterContext,
    AssemblyCommand,
    BackendOutput,
    ExpectedBackendResources,
    NormalizedAssemblyOutputs,
    PreparedBackendResources,
    RawAssemblyOutputs,
)
from organelleverse.assembly.environment_specs import AssemblyEnvironmentSpec
from organelleverse.assembly.environments import EnvironmentManager, PreparedEnvironment
from organelleverse.assembly.normalization import (
    FastaStats,
    GfaStats,
    normalize_fasta,
    normalize_gfa,
    validate_fasta_against_gfa,
)
from organelleverse.core.errors import OrganelleExecutionError

if TYPE_CHECKING:
    from organelleverse.assembly.contracts import (
        AssemblyInputPayload,
        AssemblyRequest,
    )

__all__ = ["EmptyAssemblyResourceProvider", "HimtAdapter"]


_RAW_MITO_FASTA = "himt_mitochondrial_raw.fa"
_RAW_MITO_GRAPH = "himt_mitochondrial.gfa"
_RAW_PLASTID_GRAPH = "himt_chloroplast.gfa"
_RAW_PLASTID_PATH1 = "chloroplast_path1.fa"
_RAW_PLASTID_PATH2 = "chloroplast_path2.fa"


class EmptyAssemblyResourceProvider:
    """Resource provider for backends with no managed database or model files."""

    def expected(
        self,
        manager: EnvironmentManager,
        environment_spec: AssemblyEnvironmentSpec,
        request: AssemblyRequest,
        payload: AssemblyInputPayload,
    ) -> ExpectedBackendResources:
        return ExpectedBackendResources()

    def prepare(
        self,
        manager: EnvironmentManager,
        environment_spec: AssemblyEnvironmentSpec,
        request: AssemblyRequest,
        payload: AssemblyInputPayload,
        environment: PreparedEnvironment,
    ) -> PreparedBackendResources:
        return PreparedBackendResources()


class HimtAdapter:
    """Adapter for the HiMT long-read organelle assembler."""

    backend_id = "himt"

    def preflight(self, context: AdapterContext) -> None:
        if context.route.selected_backend != self.backend_id:
            raise _unsupported(
                "HiMT adapter selected for a non-HiMT route",
                selected_backend=context.route.selected_backend,
            )
        if context.environment.backend_id != self.backend_id:
            raise _unsupported(
                "HiMT adapter requires a HiMT managed environment",
                environment_backend=context.environment.backend_id,
            )
        if len(context.payload.long_libraries) != 1:
            raise _unsupported(
                "HiMT requires exactly one declared long-read library",
                long_library_count=len(context.payload.long_libraries),
            )
        if context.payload.short_libraries:
            raise _unsupported(
                "HiMT does not consume short-read libraries",
                short_library_count=len(context.payload.short_libraries),
            )
        if context.payload.contig_inputs:
            raise _unsupported("HiMT does not accept preassembled contig input")

        unsupported_auxiliary = sorted(
            name
            for name, value in context.payload.auxiliary.model_dump(mode="python").items()
            if name.endswith("_artifact") and value is not None
        )
        if unsupported_auxiliary:
            raise _unsupported(
                "HiMT does not consume declared auxiliary inputs",
                unsupported_auxiliary=unsupported_auxiliary,
            )

        library = context.payload.long_libraries[0]
        allowed = {
            ("pacbio_hifi", "ccs"),
            ("pacbio_clr", "raw"),
            ("pacbio_clr", "corrected"),
            ("ont", "raw"),
            ("ont", "corrected"),
            ("ont", "hq"),
            ("ont", "duplex"),
        }
        if (library.technology, library.quality_state) not in allowed:
            raise _unsupported(
                "HiMT does not support this technology/quality combination",
                technology=library.technology,
                quality_state=library.quality_state,
            )

        artifact = context.input_artifacts[library.reads_artifact]
        if not _is_accepted_sequence_path(Path(artifact.uri)):
            raise _unsupported(
                "HiMT accepts only FASTA or FASTQ reads, optionally gzip-compressed",
                artifact_format=artifact.format,
                uri=artifact.uri,
            )

        params = context.effective_backend_parameters
        filter_depth = params.get("filter_depth", 0)
        proportion = params.get("proportion", 0.0)
        if (
            isinstance(filter_depth, int)
            and filter_depth > 0
            and (
                (isinstance(proportion, float) and proportion == 0.0)
                or (isinstance(proportion, int) and proportion == 0)
            )
        ):
            raise _unsupported(
                "HiMT filter_depth > 0 requires proportion > 0",
                filter_depth=filter_depth,
                proportion=proportion,
            )

        if params.get("species") == "animal" and context.request.organelle == "plastid":
            raise _unsupported(
                "HiMT animal mode cannot assemble plastids",
                species=params.get("species"),
                organelle=context.request.organelle,
            )

        provided = context.request.backend_parameters
        if provided is not None and provided.backend != self.backend_id:
            raise _unsupported(
                "backend_parameters discriminator does not match HiMT",
                discriminator=provided.backend,
            )

    def build_command(self, context: AdapterContext) -> AssemblyCommand:
        self.preflight(context)
        params = context.effective_backend_parameters
        input_role = context.payload.long_libraries[0].reads_artifact
        input_artifact = context.input_artifacts[input_role]

        backend_prefix = context.workspace / "backend" / self.backend_id

        stable: list[str] = [
            "himt",
            "assemble",
            "-i",
            f"role://artifact/{input_role}",
            "-o",
            "role://workspace/assembly",
            "-s",
            str(params["species"]),
            "-d",
            _data_type(context),
            "-k",
            str(params["kmer_length"]),
            "-n",
            str(params["head_number"]),
            "-t",
            str(context.request.threads),
            "-e",
            str(params["extract_parallel"]),
            "-b",
            str(params["base_number"]),
            "-fd",
            str(params["filter_depth"]),
            "-fp",
            str(params["filter_percentage"]),
            "-p",
            str(params["proportion"]),
            "-c",
            str(params["accuracy"]),
            "-x",
            str(params["normalize_depth"]),
        ]
        if params["no_flye_meta"]:
            stable.append("--no_flye_meta")

        resolved = [
            str(context.environment.require_executable("himt")),
            "assemble",
            "-i",
            str(Path(input_artifact.uri)),
            "-o",
            str(backend_prefix),
            "-s",
            str(params["species"]),
            "-d",
            _data_type(context),
            "-k",
            str(params["kmer_length"]),
            "-n",
            str(params["head_number"]),
            "-t",
            str(context.request.threads),
            "-e",
            str(params["extract_parallel"]),
            "-b",
            str(params["base_number"]),
            "-fd",
            str(params["filter_depth"]),
            "-fp",
            str(params["filter_percentage"]),
            "-p",
            str(params["proportion"]),
            "-c",
            str(params["accuracy"]),
            "-x",
            str(params["normalize_depth"]),
        ]
        if params["no_flye_meta"]:
            resolved.append("--no_flye_meta")

        return AssemblyCommand(
            stable_argv=tuple(stable),
            resolved_argv=tuple(resolved),
        )

    def collect_outputs(self, context: AdapterContext) -> RawAssemblyOutputs:
        backend_prefix = context.workspace / "backend" / self.backend_id
        requested = context.request.organelle

        outputs: list[BackendOutput] = []
        primary_sequence_role = "assembly_fasta"
        primary_graph_role = "assembly_graph"

        if requested == "mitochondrion":
            mito_fasta = backend_prefix / _RAW_MITO_FASTA
            mito_graph = backend_prefix / _RAW_MITO_GRAPH
            outputs.append(BackendOutput(role="assembly_fasta", path=mito_fasta, format="fasta"))
            outputs.append(BackendOutput(role="assembly_graph", path=mito_graph, format="gfa"))

            plastid_graph = backend_prefix / _RAW_PLASTID_GRAPH
            plastid_path1 = backend_prefix / _RAW_PLASTID_PATH1
            plastid_path2 = backend_prefix / _RAW_PLASTID_PATH2
            if plastid_graph.is_file() and plastid_graph.stat().st_size > 0:
                outputs.append(
                    BackendOutput(role="detected_plastid_graph", path=plastid_graph, format="gfa")
                )
            if plastid_path1.is_file() and plastid_path1.stat().st_size > 0:
                outputs.append(
                    BackendOutput(role="detected_plastid_path1", path=plastid_path1, format="fasta")
                )
            if plastid_path2.is_file() and plastid_path2.stat().st_size > 0:
                outputs.append(
                    BackendOutput(role="detected_plastid_path2", path=plastid_path2, format="fasta")
                )

            if not mito_fasta.is_file() or mito_fasta.stat().st_size == 0:
                raise _output_incomplete(
                    "HiMT did not produce the requested mitochondrial output",
                    requested_organelle=requested,
                )
        else:
            plastid_graph = backend_prefix / _RAW_PLASTID_GRAPH
            plastid_path1 = backend_prefix / _RAW_PLASTID_PATH1
            plastid_path2 = backend_prefix / _RAW_PLASTID_PATH2
            mito_fasta = backend_prefix / _RAW_MITO_FASTA
            mito_graph = backend_prefix / _RAW_MITO_GRAPH

            outputs.append(BackendOutput(role="assembly_fasta", path=plastid_path1, format="fasta"))
            outputs.append(BackendOutput(role="assembly_graph", path=plastid_graph, format="gfa"))

            if plastid_path2.is_file() and plastid_path2.stat().st_size > 0:
                outputs.append(
                    BackendOutput(
                        role="alternate_assembly_fasta", path=plastid_path2, format="fasta"
                    )
                )

            if mito_fasta.is_file() and mito_fasta.stat().st_size > 0:
                outputs.append(
                    BackendOutput(
                        role="detected_mitochondrion_fasta", path=mito_fasta, format="fasta"
                    )
                )
            if mito_graph.is_file() and mito_graph.stat().st_size > 0:
                outputs.append(
                    BackendOutput(
                        role="detected_mitochondrion_graph", path=mito_graph, format="gfa"
                    )
                )

            if not plastid_path1.is_file() or plastid_path1.stat().st_size == 0:
                raise _output_incomplete(
                    "HiMT did not produce the requested plastid path 1 output",
                    requested_organelle=requested,
                )

        return RawAssemblyOutputs(
            outputs=tuple(outputs),
            primary_sequence_role=primary_sequence_role,
            primary_graph_role=primary_graph_role,
        )

    def normalize(
        self,
        context: AdapterContext,
        raw: RawAssemblyOutputs,
        output_dir: Path,
    ) -> NormalizedAssemblyOutputs:
        output_dir.mkdir(parents=True, exist_ok=True)

        normalized_outputs: list[BackendOutput] = []
        fasta_stats_by_role: dict[str, FastaStats] = {}
        gfa_stats_by_role: dict[str, GfaStats] = {}

        _CANONICAL_NAMES = {
            "assembly_fasta": "assembly",
            "assembly_graph": "assembly",
        }

        def normalized_path(role: str, suffix: str) -> Path:
            return output_dir / f"{_CANONICAL_NAMES.get(role, role)}{suffix}"

        for output in raw.outputs:
            if output.role in {
                "assembly_graph",
                "detected_plastid_graph",
                "detected_mitochondrion_graph",
            }:
                destination = normalized_path(output.role, ".gfa")
                gfa_stats_by_role[output.role] = normalize_gfa(output.path, destination)
                normalized_outputs.append(
                    BackendOutput(role=output.role, path=destination, format="gfa")
                )
            elif output.format == "fasta" or output.path.suffix in {".fa", ".fasta"}:
                destination = normalized_path(output.role, ".fasta")
                fasta_stats_by_role[output.role] = normalize_fasta(output.path, destination)
                normalized_outputs.append(
                    BackendOutput(role=output.role, path=destination, format="fasta")
                )
            else:
                destination = normalized_path(output.role, output.path.suffix or ".bin")
                _copy_file(output.path, destination)
                normalized_outputs.append(
                    BackendOutput(
                        role=output.role,
                        path=destination,
                        format=output.path.suffix.lstrip(".") or "bin",
                    )
                )

        primary_sequence = raw.require(raw.primary_sequence_role)
        primary_graph = raw.primary_graph_role
        validation_pairs: list[tuple[str, str, bool]] = []
        if primary_graph is not None:
            validation_pairs.append(
                (
                    primary_sequence.role,
                    primary_graph,
                    context.request.organelle == "plastid",
                )
            )
        if context.request.organelle == "mitochondrion":
            for role in ("detected_plastid_path1", "detected_plastid_path2"):
                if role in fasta_stats_by_role and "detected_plastid_graph" in gfa_stats_by_role:
                    validation_pairs.append((role, "detected_plastid_graph", True))
        elif (
            "alternate_assembly_fasta" in fasta_stats_by_role
            and "assembly_graph" in gfa_stats_by_role
        ):
            validation_pairs.append(("alternate_assembly_fasta", "assembly_graph", True))
        if (
            "detected_mitochondrion_fasta" in fasta_stats_by_role
            and "detected_mitochondrion_graph" in gfa_stats_by_role
        ):
            validation_pairs.append(
                ("detected_mitochondrion_fasta", "detected_mitochondrion_graph", False)
            )

        for sequence_role, graph_role, repair_plastid in validation_pairs:
            fasta_stats = fasta_stats_by_role[sequence_role]
            graph_stats = gfa_stats_by_role[graph_role]
            try:
                validate_fasta_against_gfa(fasta_stats, graph_stats)
            except OrganelleExecutionError:
                if not repair_plastid:
                    raise
                destination = next(
                    output.path for output in normalized_outputs if output.role == sequence_role
                )
                repaired = _repair_himt_plastid_path(fasta_stats, graph_stats, destination)
                if repaired is None:
                    raise
                validate_fasta_against_gfa(repaired, graph_stats)
                fasta_stats_by_role[sequence_role] = repaired

        primary_fasta_stats = fasta_stats_by_role[primary_sequence.role]

        alternate_sequence_roles: tuple[str, ...] = ()
        if context.request.organelle == "plastid" and any(
            output.role == "alternate_assembly_fasta" for output in raw.outputs
        ):
            alternate_sequence_roles = ("alternate_assembly_fasta",)

        return NormalizedAssemblyOutputs(
            outputs=tuple(normalized_outputs),
            primary_sequence_role=raw.primary_sequence_role,
            primary_graph_role=raw.primary_graph_role,
            alternate_sequence_roles=alternate_sequence_roles,
            record_count=primary_fasta_stats.record_count,
            total_bases=primary_fasta_stats.total_bases,
            gene_markers=(),
        )


def _data_type(context: AdapterContext) -> str:
    library = context.payload.long_libraries[0]
    mapping = {
        ("pacbio_hifi", "ccs"): "HiFi",
        ("pacbio_clr", "raw"): "CLR",
        ("pacbio_clr", "corrected"): "CLR",
        ("ont", "raw"): "ONT",
        ("ont", "corrected"): "ONT",
        ("ont", "hq"): "ONT",
        ("ont", "duplex"): "ONT",
    }
    return mapping[(library.technology, library.quality_state)]


def _repair_himt_plastid_path(
    fasta: FastaStats,
    graph: GfaStats,
    destination: Path,
) -> FastaStats | None:
    """Repair HiMT 1.1.3's three-segment plastid orientation error.

    HiMT's ``output_two_chl_hap`` concatenates the GFA segment strings without
    applying link orientations. The raw files remain in the published backend
    workspace. For the one exact topology emitted by that routine (three unique
    segments, one repeat used twice, zero-overlap links), recover the upstream
    segment order and choose the closest orientation assignment that forms a
    closed GFA walk. Any other topology fails closed.
    """
    if fasta.record_count != 1 or len(graph.segment_names) != 3:
        return None

    oriented = _oriented_segment_sequences(graph)
    raw_tokens = _decompose_himt_path(fasta.sequences[0], oriented)
    if raw_tokens is None:
        return None
    names = tuple(name for name, _orientation in raw_tokens)
    if len(names) != 4 or set(names) != set(graph.segment_names):
        return None
    counts = sorted(names.count(name) for name in set(names))
    if counts != [1, 1, 2]:
        return None

    edges = _zero_overlap_oriented_edges(graph)
    candidates: list[tuple[int, tuple[str, ...], str]] = []
    raw_orientations = tuple(orientation for _name, orientation in raw_tokens)
    for orientations in product(("+", "-"), repeat=len(names)):
        nodes = tuple(zip(names, orientations, strict=True))
        circular_pairs = tuple(zip(nodes, (*nodes[1:], nodes[0]), strict=True))
        if not all(target in edges.get(source, ()) for source, target in circular_pairs):
            continue
        sequence = "".join(oriented[node] for node in nodes)
        if len(sequence) != len(fasta.sequences[0]):
            continue
        distance = sum(
            observed != corrected
            for observed, corrected in zip(raw_orientations, orientations, strict=True)
        )
        candidates.append((distance, orientations, sequence))
    if not candidates:
        return None

    _distance, _orientations, sequence = min(candidates)
    header = fasta.record_ids[0]
    if fasta.descriptions[0]:
        header = f"{header} {fasta.descriptions[0]}"
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f">{header}\n")
        for offset in range(0, len(sequence), 80):
            handle.write(sequence[offset : offset + 80])
            handle.write("\n")
    return normalize_fasta(destination, destination)


def _oriented_segment_sequences(graph: GfaStats) -> dict[tuple[str, str], str]:
    complement = str.maketrans("ACGTRYSWKMBDHVN", "TGCAYRSWMKVHDBN")
    oriented: dict[tuple[str, str], str] = {}
    for name, sequence in zip(graph.segment_names, graph.segment_sequences, strict=True):
        oriented[(name, "+")] = sequence
        oriented[(name, "-")] = sequence.translate(complement)[::-1]
    return oriented


def _decompose_himt_path(
    sequence: str,
    oriented: dict[tuple[str, str], str],
) -> tuple[tuple[str, str], ...] | None:
    ordered_nodes = tuple(sorted(oriented))

    def search(
        offset: int, tokens: tuple[tuple[str, str], ...]
    ) -> tuple[tuple[str, str], ...] | None:
        if offset == len(sequence):
            return tokens
        if len(tokens) == 4:
            return None
        for node in ordered_nodes:
            segment = oriented[node]
            if sequence.startswith(segment, offset):
                found = search(offset + len(segment), (*tokens, node))
                if found is not None:
                    return found
        return None

    return search(0, ())


def _zero_overlap_oriented_edges(
    graph: GfaStats,
) -> dict[tuple[str, str], set[tuple[str, str]]]:
    edges: dict[tuple[str, str], set[tuple[str, str]]] = {}
    for from_name, from_orientation, to_name, to_orientation, overlap in graph.links:
        if overlap != 0:
            continue
        source = (from_name, from_orientation)
        target = (to_name, to_orientation)
        edges.setdefault(source, set()).add(target)
        reverse_source = (to_name, _flip_orientation(to_orientation))
        reverse_target = (from_name, _flip_orientation(from_orientation))
        edges.setdefault(reverse_source, set()).add(reverse_target)
    return edges


def _flip_orientation(orientation: str) -> str:
    return "-" if orientation == "+" else "+"


def _is_accepted_sequence_path(path: Path) -> bool:
    suffixes = {".fa", ".fasta", ".fq", ".fastq", ".fa.gz", ".fasta.gz", ".fq.gz", ".fastq.gz"}
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in suffixes)


def _unsupported(message: str, **details: object) -> OrganelleExecutionError:
    return OrganelleExecutionError(
        code="assembly.unsupported_data_profile", message=message, details=details
    )


def _output_incomplete(message: str, **details: object) -> OrganelleExecutionError:
    return OrganelleExecutionError(
        code="assembly.output_incomplete", message=message, details=details
    )


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(source.read_bytes())
