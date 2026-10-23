"""Native ovasm assembly adapter.

ovasm (rust/ovasm) is one self-contained binary, found by ``_ovasm.resolve_ovasm``: there is
no conda environment to create, lock or resolve. :class:`NativeOvasmEnvironment` stands in
for the environment manager and gives the service the binary and its identity (content hash
and reported version).

One assembly is several ovasm commands with decisions between them (recruit, correct noisy
reads, assemble, evidence, linearize, retry at a smaller k, unify). That pipeline is
``desktop.ovasm_runner.run_ovasm``, the function the desktop assembly tools call; the backend
command runs it in a child Python process (:func:`main`) so that the service starts, times
out and logs it like any other backend.
"""

from __future__ import annotations

from organelleverse._layout import import_root

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from organelleverse._ovasm import resolve_ovasm
from organelleverse.assembly.backends.base import (
    AdapterContext,
    AssemblyCommand,
    BackendOutput,
    NormalizedAssemblyOutputs,
    RawAssemblyOutputs,
)
from organelleverse.assembly.environment_specs import AssemblyEnvironmentSpec, normalize_platform
from organelleverse.assembly.environments import PreparedEnvironment, PreparedExecutable
from organelleverse.assembly.normalization import normalize_fasta, normalize_gfa, validate_fasta_against_gfa
from organelleverse.core.errors import (
    OrganelleDependencyError,
    OrganelleExecutionError,
    OrganelleInputError,
)

__all__ = ["NativeOvasmEnvironment", "OvasmAdapter", "main"]

# Public component evidence was absent from earlier cached output bundles.
OVASM_REPORT_PROTOCOL_VERSION = "2"

#: The child puts the package root first on its path: the service strips PYTHONPATH from
#: backend processes, and the pipeline must be the code of the parent, not another install.
_BOOTSTRAP = (
    "import sys; sys.path.insert(0, sys.argv.pop(1)); "
    "from organelleverse.assembly.backends.ovasm import main; raise SystemExit(main())"
)

#: Pipeline files published besides the sequence and the graph: role -> (file, format, type).
_REPORTS: dict[str, tuple[str, str, str]] = {
    "assigned_graph": ("assembly.assigned.gfa", "gfa", "text/plain"),
    "unassigned_graph": ("assembly.unassigned.gfa", "gfa", "text/plain"),
    "unassigned_sequences": ("unassigned.fasta", "fasta", "text/x-fasta"),
    "unassigned_report": ("unassigned.json", "json", "application/json"),
    "organelle_graph": ("organelle.gfa", "gfa", "text/plain"),
    "graph_metadata": ("organelle.json", "json", "application/json"),
    "evidence": ("evidence.json", "json", "application/json"),
    "configuration": ("configuration.json", "json", "application/json"),
    "linearization": ("linearization.json", "json", "application/json"),
    "summary": ("summary.json", "json", "application/json"),
    "assembly_report": ("assembly.json", "json", "application/json"),
    "recruitment": ("recruit/recruit.json", "json", "application/json"),
    "seed_database": ("seed_database.json", "json", "application/json"),
    "discovery": ("discover/discover.json", "json", "application/json"),
    "identification": ("discover/identify.json", "json", "application/json"),
    "correction": ("correct/correct.json", "json", "application/json"),
    "nuclear_coverage": ("nuclear-coverage/coverage-profile.json", "json", "application/json"),
    "component_filter": ("components.json", "json", "application/json"),
}
_REQUIRED_REPORTS = ("organelle_graph", "graph_metadata", "evidence", "linearization", "summary")


def _unsupported(message: str, **details: object) -> OrganelleInputError:
    return OrganelleInputError(
        code="assembly.unsupported_data_profile",
        message=message,
        details=details,
    )


def _output_incomplete(message: str, **details: object) -> OrganelleExecutionError:
    return OrganelleExecutionError(
        code="assembly.output_incomplete",
        message=message,
        details=details,
    )


class NativeOvasmEnvironment:
    """The ovasm binary as a prepared environment: nothing is created or installed."""

    def __init__(self, parameters: Mapping[str, object] | None = None) -> None:
        self._multichrom_input_protocol = "2" if parameters and parameters.get("graph_mode") == "multichrom" else None
        self._coverage_tools = None
        self._coverage_provenance = None
        if parameters and parameters.get("graph_min_count") == "auto":
            from organelleverse.assembly.nuclear_coverage import resolve_tools

            self._coverage_tools = resolve_tools()
            self._coverage_provenance = self._coverage_tools.provenance()

    def _binary(self) -> Path:
        binary = resolve_ovasm()
        if binary is None:
            raise OrganelleDependencyError(
                code="assembly.environment_unavailable",
                message="ovasm is not available; build it with `cargo build --release` in "
                "rust/ovasm or set ORG_VERSE_OVASM_BIN",
                details={"missing": ["ovasm"]},
            )
        return Path(binary).resolve()

    def expected_environment_digest(
        self,
        spec: AssemblyEnvironmentSpec,
        *,
        platform: str | None = None,
    ) -> str:
        """Identity of the binary that would run: a rebuilt binary is another environment."""
        binary = self._binary()
        digest = hashlib.sha256()
        with binary.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        payload = {
            "contract_version": spec.contract_version,
            "backend_id": spec.backend_id,
            "carrier": spec.carrier,
            "platform": platform if platform is not None else normalize_platform(),
            "binary_sha256": digest.hexdigest(),
            "report_protocol_version": OVASM_REPORT_PROTOCOL_VERSION,
        }
        if self._coverage_provenance is not None:
            payload["nuclear_coverage_provider"] = self._coverage_provenance
        if self._multichrom_input_protocol is not None:
            payload["multichrom_input_protocol"] = self._multichrom_input_protocol
        return (
            "sha256:"
            + hashlib.sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
        )

    def prepare(
        self,
        spec: AssemblyEnvironmentSpec,
        *,
        policy: Literal["ensure", "require"],
        platform: str | None = None,
    ) -> PreparedEnvironment:
        del policy  # nothing to create: the binary is there or `_binary` has raised
        binary = self._binary()
        try:
            reported = subprocess.run(
                [str(binary), "--version"], capture_output=True, text=True, timeout=60, check=True
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError) as error:
            raise OrganelleDependencyError(
                code="assembly.environment_unavailable",
                message=f"ovasm at {binary} did not report its version: {error}",
                details={"executable": str(binary)},
            ) from error
        if not reported.startswith("ovasm "):
            raise OrganelleDependencyError(
                code="assembly.environment_unavailable",
                message=f"{binary} is not ovasm: `--version` printed {reported!r}",
                details={"executable": str(binary)},
            )
        executables = [PreparedExecutable(name="ovasm", path=binary)]
        if self._coverage_tools is not None:
            executables.extend([
                PreparedExecutable(name="jellyfish", path=self._coverage_tools.jellyfish),
                PreparedExecutable(name="Rscript", path=self._coverage_tools.rscript),
                PreparedExecutable(name="genomescope.R", path=self._coverage_tools.genomescope),
            ])
        return PreparedEnvironment(
            backend_id=spec.backend_id,
            carrier="native",
            platform=platform if platform is not None else normalize_platform(),
            digest=self.expected_environment_digest(spec, platform=platform),
            prefix=binary.parent,
            executables=tuple(executables),
            version=reported,
            software_version=reported.removeprefix("ovasm ").strip(),
        )


@dataclass(frozen=True)
class _OvasmInputs:
    """The read files of one run: the main file, a paired library's second file, the Illumina
    file(s) of a hybrid run, and the custom seed."""

    reads_role: str
    reads: Path
    mate_role: str | None
    mate: Path | None
    short_roles: tuple[str, ...]
    short_files: tuple[Path, ...]
    seed_role: str | None
    seed: Path | None
    origins_role: str | None
    origins: Path | None


class OvasmAdapter:
    """Run the ovasm pipeline on one read library and publish its graph and evidence.

    The primary sequence is the representative molecule set of ``ovasm linearize``; the
    evidence and linearization reports say how well the reads single it out and are
    published with it, as is the unified graph (OV-GFA) for the pangenome module.
    """

    backend_id = "ovasm"

    def preflight(self, context: AdapterContext) -> None:
        if context.route.selected_backend != self.backend_id:
            raise _unsupported("ovasm adapter selected for a non-ovasm route")
        if context.environment.backend_id != self.backend_id:
            raise _unsupported("ovasm requires the native ovasm environment")
        request = context.request
        if request.taxon_group != "plant":
            raise _unsupported(
                "ovasm's seed database and protein panel are land plants only",
                taxon_group=request.taxon_group,
            )
        if request.backend_version not in {"tested", context.environment.software_version}:
            raise _unsupported(
                "ovasm runs the binary that is installed; backend_version cannot select another",
                backend_version=request.backend_version,
                installed=context.environment.software_version,
            )
        if request.environment_hint is not None:
            raise _unsupported(
                "ovasm takes no environment hint; point ORG_VERSE_OVASM_BIN at the binary"
            )
        provided = request.backend_parameters
        if provided is not None and provided.backend != self.backend_id:
            raise _unsupported("backend_parameters discriminator does not match ovasm")
        payload = context.payload
        if (
            payload.contig_inputs
            or len(payload.long_libraries) > 1
            or len(payload.short_libraries) > 1
            or not payload.long_libraries + payload.short_libraries
        ):
            raise _unsupported(
                "ovasm takes one long-read library, one Illumina library, or one of each"
            )

    def _inputs(self, context: AdapterContext) -> _OvasmInputs:
        """Roles and paths of the read files and of the custom seed, if any."""
        payload = context.payload

        def local(role: str) -> Path:
            path = Path(context.input_artifacts[role].uri)
            if not path.is_file():
                raise _unsupported(
                    "ovasm read artifact is not a readable local file", uri=str(path)
                )
            return path

        short = payload.short_libraries[0] if payload.short_libraries else None
        short_roles = (
            tuple(r for r in (short.read1_artifact, short.read2_artifact) if r is not None)
            if short is not None
            else ()
        )
        if payload.long_libraries:
            # long reads, with the Illumina file(s) of a hybrid run beside them
            reads_role, mate_role = payload.long_libraries[0].reads_artifact, None
            hybrid_roles = short_roles
        else:
            # Illumina only: the first file is the reads, the second file of a pair its mate
            reads_role = short_roles[0]
            mate_role = short_roles[1] if len(short_roles) > 1 else None
            hybrid_roles = ()
        seed_role = payload.auxiliary.seed_fasta_artifact
        return _OvasmInputs(
            reads_role=reads_role,
            reads=local(reads_role),
            mate_role=mate_role,
            mate=local(mate_role) if mate_role is not None else None,
            short_roles=hybrid_roles,
            short_files=tuple(local(role) for role in hybrid_roles),
            seed_role=seed_role,
            seed=Path(context.input_artifacts[seed_role].uri) if seed_role is not None else None,
            origins_role=payload.auxiliary.read_origins_artifact,
            origins=local(payload.auxiliary.read_origins_artifact)
            if payload.auxiliary.read_origins_artifact is not None else None,
        )

    def build_command(self, context: AdapterContext) -> AssemblyCommand:
        self.preflight(context)
        inputs = self._inputs(context)
        role, reads, seed_role, seed = (
            inputs.reads_role,
            inputs.reads,
            inputs.seed_role,
            inputs.seed,
        )
        params = context.effective_backend_parameters
        options = [
            "--organelle",
            context.request.organelle,
            "--read-type",
            str(params["read_type"]),
            "--read-set",
            str(params["read_set"]),
            "--seed-source",
            str(params["seed_source"]),
            "--sample",
            str(params["sample"]),
            "--threads",
            str(context.request.threads),
            "--k-ladder-start",
            str(params["k_ladder_start"]),
            "--max-molecules",
            str(params["max_molecules"]),
        ]
        if params["also_discover"]:
            options.append("--also-discover")
        if not params.get("fallback", True):
            options.append("--no-fallback")
        if params.get("graph_mode", "sparse") == "multichrom":
            options += ["--graph-mode", "multichrom", "--graph-min-count", str(params["graph_min_count"])]
            if params["graph_min_count"] == "auto":
                options += ["--nuclear-ploidy", str(params["nuclear_ploidy"])]
        # a paired library's second file (--mate) or a hybrid run's Illumina files
        more_stable: list[str] = []
        more_resolved: list[str] = []
        if inputs.origins is not None:
            read_sha = context.input_artifacts[inputs.reads_role].sha256
            document = json.loads(inputs.origins.read_text(encoding="utf-8"))
            if document.get("original_reads_sha256") != read_sha:
                raise _unsupported("read_origins does not match the canonical original read artifact")
            more_stable += ["--read-origins", f"role://artifact/{inputs.origins_role}",
                            "--original-read-artifact-sha256", read_sha]
            more_resolved += ["--read-origins", str(inputs.origins),
                              "--original-read-artifact-sha256", read_sha]
        if inputs.mate is not None:
            more_stable += ["--mate", f"role://artifact/{inputs.mate_role}"]
            more_resolved += ["--mate", str(inputs.mate)]
        for flag, short_role, short_file in zip(
            ("--short-reads", "--short-reads-2"), inputs.short_roles, inputs.short_files, strict=False
        ):
            more_stable += [flag, f"role://artifact/{short_role}"]
            more_resolved += [flag, str(short_file)]
        stable = ["ovasm-pipeline", "--reads", f"role://artifact/{role}", *options, *more_stable]
        resolved = [
            sys.executable,
            "-c",
            _BOOTSTRAP,
            str(import_root()),
            "--ovasm",
            str(context.environment.require_executable("ovasm")),
            "--reads",
            str(reads),
            *options,
            *more_resolved,
        ]
        if params.get("graph_min_count") == "auto":
            for flag, name in (("--coverage-jellyfish", "jellyfish"),
                               ("--coverage-rscript", "Rscript"),
                               ("--coverage-genomescope", "genomescope.R")):
                resolved += [flag, str(context.environment.require_executable(name))]
        if seed is not None:
            stable += ["--seed", f"role://artifact/{seed_role}"]
            resolved += ["--seed", str(seed)]
        stable += ["--out", "role://workspace/output"]
        resolved += ["--out", str(self._output_root(context))]
        return AssemblyCommand(stable_argv=tuple(stable), resolved_argv=tuple(resolved))

    def _output_root(self, context: AdapterContext) -> Path:
        return context.workspace / "backend" / self.backend_id

    def collect_outputs(self, context: AdapterContext) -> RawAssemblyOutputs:
        root = self._output_root(context)
        required_reports = set(_REQUIRED_REPORTS)
        if context.payload.auxiliary.read_origins_artifact is not None:
            required_reports.update({"assigned_graph", "unassigned_graph",
                                     "unassigned_sequences", "unassigned_report"})
        summary_path = root / "summary.json"
        if not summary_path.is_file():
            raise _output_incomplete(
                "the ovasm pipeline wrote no summary.json; see stderr.log",
                output_root=str(root),
            )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        fasta, graph = root / "molecules.fasta", root / "assembly.gfa"
        if not summary["molecules"] or not fasta.is_file():
            # A graph without a representative sequence is a real outcome, but the result
            # contract needs a primary sequence. The graph, evidence and linearization
            # report stay in the published run directory, where this message points.
            raise _output_incomplete(
                "ovasm built a graph but no representative sequence (see linearization.json: "
                "unbridged anchors, unsolved components); the graph and evidence are kept "
                "under backend/ovasm in the run directory",
                k_tried=summary["k_tried"],
                status=summary.get("status"),
                reasons=summary.get("reasons"),
                unitigs=summary["unitigs"],
                unsolved_components=summary["unsolved_components"],
                unbridged_anchors=summary["unbridged_anchors"],
            )
        outputs = [
            BackendOutput(
                role="assembly_fasta", path=fasta, format="fasta", media_type="text/x-fasta"
            ),
            BackendOutput(role="assembly_graph", path=graph, format="gfa", media_type="text/plain"),
        ]
        for report_role, (name, file_format, media_type) in _REPORTS.items():
            path = root / name
            if path.is_file():
                outputs.append(
                    BackendOutput(
                        role=report_role, path=path, format=file_format, media_type=media_type
                    )
                )
            elif report_role in required_reports:
                raise _output_incomplete(f"the ovasm pipeline wrote no {name}", missing=name)
        return RawAssemblyOutputs(
            outputs=tuple(outputs),
            primary_sequence_role="assembly_fasta",
            primary_graph_role="assembly_graph",
        )

    def normalize(
        self,
        context: AdapterContext,
        raw: RawAssemblyOutputs,
        output_dir: Path,
    ) -> NormalizedAssemblyOutputs:
        del context
        output_dir.mkdir(parents=True, exist_ok=True)
        stats = normalize_fasta(raw.require("assembly_fasta").path, output_dir / "assembly.fasta")
        gfa_stats = normalize_gfa(raw.require("assembly_graph").path, output_dir / "assembly.gfa")
        validate_fasta_against_gfa(stats, gfa_stats)
        outputs = [
            BackendOutput(
                role="assembly_fasta",
                path=output_dir / "assembly.fasta",
                format="fasta",
                media_type="text/x-fasta",
            ),
            BackendOutput(
                role="assembly_graph",
                path=output_dir / "assembly.gfa",
                format="gfa",
                media_type="text/plain",
            ),
        ]
        for item in raw.outputs:
            if item.role in {"assembly_fasta", "assembly_graph"}:
                continue
            # reports are published as written; their flat names are the pipeline's own
            destination = output_dir / Path(_REPORTS[item.role][0]).name
            shutil.copyfile(item.path, destination)
            outputs.append(
                BackendOutput(
                    role=item.role, path=destination, format=item.format, media_type=item.media_type
                )
            )
        return NormalizedAssemblyOutputs(
            outputs=tuple(outputs),
            primary_sequence_role="assembly_fasta",
            primary_graph_role="assembly_graph",
            record_count=stats.record_count,
            total_bases=stats.total_bases,
        )


def main(argv: Sequence[str] | None = None) -> int:
    """The backend command: run the ovasm pipeline once; progress and errors go to stderr."""
    parser = argparse.ArgumentParser(prog="ovasm-pipeline", description=main.__doc__)
    parser.add_argument("--ovasm", required=True, help="the ovasm binary")
    parser.add_argument("--reads", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--organelle", required=True, choices=("mitochondrion", "plastid"))
    parser.add_argument(
        "--read-type",
        required=True,
        choices=("hifi_only", "ont_only", "clr_only", "short_read", "hybrid"),
    )
    parser.add_argument("--read-set", required=True, choices=("whole_genome", "target_reads"))
    parser.add_argument("--seed-source", required=True, choices=("builtin", "custom", "discover"))
    parser.add_argument("--seed", type=Path)
    parser.add_argument("--read-origins", type=Path)
    parser.add_argument("--original-read-artifact-sha256")
    parser.add_argument("--mate", type=Path, help="second file of a paired Illumina library")
    parser.add_argument("--short-reads", type=Path, help="Illumina file of a hybrid run")
    parser.add_argument("--short-reads-2", type=Path, help="its mate")
    parser.add_argument("--also-discover", action="store_true")
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="run once as requested (no escalation to discovery when the run is not accepted)",
    )
    parser.add_argument("--k-ladder-start", choices=("auto", "top"), default="auto")
    parser.add_argument("--max-molecules", type=int, default=4)
    parser.add_argument("--graph-mode", choices=("sparse", "multichrom"), default="sparse")
    parser.add_argument("--graph-min-count")
    parser.add_argument("--nuclear-ploidy", type=int, default=2)
    parser.add_argument("--coverage-jellyfish", type=Path)
    parser.add_argument("--coverage-rscript", type=Path)
    parser.add_argument("--coverage-genomescope", type=Path)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--threads", required=True, type=int)
    args = parser.parse_args(argv)

    os.environ["ORG_VERSE_OVASM_BIN"] = args.ovasm
    from organelleverse.assembly.ovasm_fallback import run_ovasm_resilient

    params = {
        "strategy": args.read_type,
        "read_set": args.read_set,
        "seed_source": args.seed_source,
        "seed": str(args.seed) if args.seed is not None else "",
        "also_discover": args.also_discover,
        "fallback": not args.no_fallback,
        "k_ladder_start": args.k_ladder_start,
        "max_molecules": args.max_molecules,
        "graph_mode": args.graph_mode,
        "graph_min_count": args.graph_min_count,
        "nuclear_ploidy": args.nuclear_ploidy,
        "sample": args.sample,
        "threads": args.threads,
    }
    if args.short_reads is not None:
        params["short_reads"] = str(args.short_reads)
    if args.short_reads_2 is not None:
        params["short_reads_2"] = str(args.short_reads_2)
    more = {"mate": args.mate} if args.mate is not None else {}
    if args.read_origins is not None:
        more["read_origins"] = args.read_origins
        more["original_read_artifact_sha256"] = args.original_read_artifact_sha256
    if args.graph_min_count == "auto":
        from organelleverse.assembly.nuclear_coverage import CoverageTools

        if args.coverage_jellyfish is not None:
            more["coverage_tools"] = CoverageTools(
                args.coverage_jellyfish, args.coverage_rscript, args.coverage_genomescope,
            )
    try:
        result = run_ovasm_resilient(
            params,
            organelle=args.organelle,
            reads=args.reads,
            output_dir=args.out,
            log=lambda message: print(message, file=sys.stderr, flush=True),
            **more,
        )
    except Exception as error:
        # the cause in one line at the end of stderr.log, where a failed Result points
        print(f"ovasm pipeline failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(result["summary_text"])
    return 0
