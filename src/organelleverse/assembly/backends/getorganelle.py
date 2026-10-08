"""GetOrganelle Illumina adapter and managed database preparation."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from organelleverse.assembly.backends.base import (
    AdapterContext,
    AssemblyCommand,
    BackendOutput,
    ExpectedBackendResources,
    NormalizedAssemblyOutputs,
    PreparedBackendResources,
    RawAssemblyOutputs,
)
from organelleverse.assembly.contracts import (
    GETORGANELLE_TARGETS,
    GetOrganelleParameters,
)
from organelleverse.assembly.environment_specs import (
    GETORGANELLE_VERSION,
    AssemblyEnvironmentSpec,
)
from organelleverse.assembly.execution import managed_prefix_environment
from organelleverse.assembly.normalization import normalize_fasta, normalize_gfa
from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.errors import OrganelleDependencyError, OrganelleExecutionError
from organelleverse.core.external import MESSAGE_TAIL_LINES, run_external, tail_lines
from organelleverse.core.frozen import FrozenMap

if TYPE_CHECKING:
    from organelleverse.assembly.contracts import AssemblyInputPayload, AssemblyRequest
    from organelleverse.assembly.environment_specs import ManagedDatabaseSpec, ManagedFileSpec
    from organelleverse.assembly.environments import EnvironmentManager, PreparedEnvironment

__all__ = ["GetOrganelleAdapter", "GetOrganelleResourceProvider"]


_METADATA_NAME = "organelleverse_getorganelledb.json"
_PRESET_FLAGS = (
    ("fast", "--fast"),
    ("memory_save", "--memory-save"),
    ("memory_unlimited", "--memory-unlimited"),
)
_OPTION_FLAGS = (
    ("max_reads", "--max-reads", "value"),
    ("reduce_reads_for_coverage", "--reduce-reads-for-coverage", "value"),
    ("max_ignore_percent", "--max-ignore-percent", "value"),
    ("phred_offset", "--phred-offset", "value"),
    ("min_quality_score", "--min-quality-score", "value"),
    ("output_prefix", "--prefix", "value"),
    ("output_per_round", "--out-per-round", "flag"),
    ("zip_files", "--zip-files", "flag"),
    ("keep_temp", "--keep-temp", "flag"),
    ("word_size", "-w", "value"),
    ("pregroup_word_size", "--pre-w", "value"),
    ("max_rounds", "-R", "value"),
    ("max_words", "--max-n-words", "value"),
    ("jump_step", "-J", "value"),
    ("mesh_size", "-M", "value"),
    ("bowtie2_options", "--bowtie2-options", "value"),
    ("larger_auto_word_size", "--larger-auto-ws", "flag"),
    ("target_genome_size", "--target-genome-size", "value"),
    ("max_extending_length", "--max-extending-len", "value"),
    ("spades_kmers", "-k", "kmers"),
    ("spades_options", "--spades-options", "value"),
    ("no_spades", "--no-spades", "flag"),
    ("ignore_kmer", "--ignore-k", "value"),
    ("disentangle_depth_factor", "--disentangle-df", "value"),
    ("contamination_depth", "--contamination-depth", "value"),
    ("contamination_similarity", "--contamination-similarity", "value"),
    ("no_degenerate", "--no-degenerate", "flag"),
    ("degenerate_depth", "--degenerate-depth", "value"),
    ("degenerate_similarity", "--degenerate-similarity", "value"),
    ("disentangle_time_limit", "--disentangle-time-limit", "value"),
    ("expected_max_size", "--expected-max-size", "value"),
    ("expected_min_size", "--expected-min-size", "value"),
    ("reverse_lsc", "--reverse-lsc", "flag"),
    ("max_paths", "--max-paths-num", "value"),
    ("pregrouped_reads", "-P", "value"),
    ("index_in_memory", "--index-in-memory", "flag"),
    ("remove_duplicates", "--remove-duplicates", "value"),
    ("flush_step", "--flush-step", "value"),
    ("random_seed", "--random-seed", "value"),
    ("verbose", "--verbose", "flag"),
)


def _unsupported(message: str, **details: object) -> OrganelleExecutionError:
    return OrganelleExecutionError(
        code="assembly.unsupported_data_profile", message=message, details=details
    )


def _output_incomplete(message: str, **details: object) -> OrganelleExecutionError:
    return OrganelleExecutionError(
        code="assembly.output_incomplete", message=message, details=details
    )


def _environment_error(message: str, **details: object) -> OrganelleDependencyError:
    return OrganelleDependencyError(
        code="assembly.environment_unavailable", message=message, details=details
    )


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _database_source(database: ManagedDatabaseSpec) -> ManagedFileSpec:
    if len(database.files) != 1:
        raise _environment_error(
            "GetOrganelleDB must declare exactly one managed archive",
            file_count=len(database.files),
        )
    return database.files[0]


def _database_targets(target: str) -> tuple[str, ...]:
    # Upstream plant defaults label both organelles, in target-priority order.
    if target == "embplant_mt":
        return (target, "embplant_pt")
    if target == "embplant_pt":
        return (target, "embplant_mt")
    return (target,)


def _resource_identity(database: ManagedDatabaseSpec, *, target: str) -> tuple[dict[str, str], str]:
    source = _database_source(database)
    identity = {
        "archive_sha256": source.sha256,
        "database_id": database.database_id,
        "database_version": database.version,
        "source_commit": database.source_commit,
        "source_url": source.url,
        "target": target,
        "configured_targets": ",".join(_database_targets(target)),
        "tool_version": GETORGANELLE_VERSION,
    }
    return identity, hashlib.sha256(_canonical_bytes(identity)).hexdigest()


def _prepared_artifact(root: Path, identity: dict[str, str]) -> ArtifactRef:
    metadata_path = root / _METADATA_NAME
    if not root.is_dir() or not metadata_path.is_file():
        raise _environment_error(
            "prepared GetOrganelleDB metadata is missing", prepared_root=str(root)
        )
    try:
        raw_metadata: object = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise _environment_error(
            "prepared GetOrganelleDB metadata is invalid", prepared_root=str(root)
        ) from error
    if not isinstance(raw_metadata, dict):
        raise _environment_error(
            "prepared GetOrganelleDB metadata is invalid", prepared_root=str(root)
        )
    metadata = cast(dict[str, object], raw_metadata)
    if metadata.get("identity") != identity:
        raise _environment_error(
            "prepared GetOrganelleDB identity does not match the request",
            prepared_root=str(root),
        )
    raw_records = metadata.get("files")
    if not isinstance(raw_records, list) or not raw_records:
        raise _environment_error(
            "prepared GetOrganelleDB file inventory is invalid", prepared_root=str(root)
        )
    expected_names: set[str] = set()
    for raw_record in cast(list[object], raw_records):
        if not isinstance(raw_record, dict):
            raise _environment_error("prepared GetOrganelleDB file inventory is invalid")
        record = cast(dict[str, object], raw_record)
        name, sha256, size = record.get("name"), record.get("sha256"), record.get("size_bytes")
        if (
            not isinstance(name, str)
            or name in expected_names
            or not isinstance(sha256, str)
            or not isinstance(size, int)
        ):
            raise _environment_error("prepared GetOrganelleDB file inventory is invalid")
        candidate = root / name
        try:
            candidate.resolve(strict=False).relative_to(root.resolve())
        except ValueError:
            raise _environment_error("prepared GetOrganelleDB inventory escapes its root") from None
        if candidate.is_symlink() or not candidate.is_file() or candidate.stat().st_size != size:
            raise _environment_error(
                "prepared GetOrganelleDB file is missing or changed", file=name
            )
        actual_sha256 = _sha256_file(candidate)
        if actual_sha256 != sha256:
            raise _environment_error(
                "prepared GetOrganelleDB file hash changed",
                file=name,
                expected_sha256=sha256,
                actual_sha256=actual_sha256,
            )
        expected_names.add(name)
    actual_names = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path != metadata_path
    }
    if actual_names != expected_names:
        raise _environment_error(
            "prepared GetOrganelleDB directory contains unexpected files",
            missing=sorted(expected_names - actual_names),
            extra=sorted(actual_names - expected_names),
        )
    metadata_bytes = _canonical_bytes(metadata)
    if metadata_path.read_bytes() != metadata_bytes:
        raise _environment_error("prepared GetOrganelleDB metadata is not canonical")
    return ArtifactRef.from_path(
        metadata_path,
        kind="getorganelledb_config",
        format="json",
        media_type="application/json",
    )


def _config_root(artifact: ArtifactRef) -> Path:
    return Path(artifact.uri).parent


def _write_prepared_metadata(root: Path, identity: dict[str, str]) -> None:
    files: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path != root / _METADATA_NAME:
            files.append(
                {
                    "name": str(path.relative_to(root)),
                    "sha256": _sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    if not files:
        raise _environment_error("GetOrganelleDB initialization produced no files")
    metadata: dict[str, object] = {
        "schema_version": "organelleverse.getorganelledb-prepared.v1",
        "identity": identity,
        "files": files,
    }
    (root / _METADATA_NAME).write_bytes(_canonical_bytes(metadata))


class GetOrganelleResourceProvider:
    """Prepare the target and companion databases required by upstream defaults."""

    def expected(
        self,
        manager: EnvironmentManager,
        environment_spec: AssemblyEnvironmentSpec,
        request: AssemblyRequest,
        payload: AssemblyInputPayload,
    ) -> ExpectedBackendResources:
        del manager, payload
        target = GETORGANELLE_TARGETS[(request.organelle, request.taxon_group)]
        database = environment_spec.require_database()
        _, digest = _resource_identity(database, target=target)
        return ExpectedBackendResources(
            artifact_roles=("getorganelledb_config",),
            database_hashes=FrozenMap(
                {
                    "getorganelledb_archive": _database_source(database).sha256,
                    "getorganelledb_identity": digest,
                }
            ),
        )

    def prepare(
        self,
        manager: EnvironmentManager,
        environment_spec: AssemblyEnvironmentSpec,
        request: AssemblyRequest,
        payload: AssemblyInputPayload,
        environment: PreparedEnvironment,
    ) -> PreparedBackendResources:
        del payload
        if (
            re.search(r"(?<!\d)" + re.escape(GETORGANELLE_VERSION) + r"(?!\d)", environment.version)
            is None
        ):
            raise _environment_error(
                "prepared GetOrganelle version differs from the tested version",
                observed_version=environment.version,
                expected_version=GETORGANELLE_VERSION,
            )
        target = GETORGANELLE_TARGETS[(request.organelle, request.taxon_group)]
        database = environment_spec.require_database()
        identity, digest = _resource_identity(database, target=target)
        prepared_root = (
            manager.cache_root
            / "prepared-databases"
            / f"{database.database_id}-{database.version}"
            / digest
        )
        if prepared_root.exists():
            artifact = _prepared_artifact(prepared_root, identity)
            return self._resources(database, digest, artifact)

        archive_dir = manager.materialize_database_files(database)
        source = _database_source(database)
        archive_path = archive_dir / source.name
        parent = prepared_root.parent
        parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{digest}.", suffix=".tmp", dir=parent))
        extracted = temporary / "archive"
        candidate = temporary / "prepared"
        try:
            manager.safe_extract_tar(archive_path.read_bytes(), extracted)
            archive_root = extracted / f"GetOrganelleDB-{database.source_commit}" / database.version
            if not archive_root.is_dir():
                raise _environment_error(
                    "GetOrganelleDB commit archive is missing the declared full database",
                    source_commit=database.source_commit,
                    database_version=database.version,
                )
            candidate.mkdir()
            executable = environment.require_executable("get_organelle_config.py")
            argv = (
                str(executable),
                "-a",
                ",".join(_database_targets(target)),
                "--use-local",
                str(archive_root),
                "--config-dir",
                str(candidate),
            )
            env = managed_prefix_environment(environment.prefix)
            try:
                run_external(
                    argv,
                    timeout=300,
                    env=env,
                    tool="get_organelle_config.py",
                    code="assembly.environment_unavailable",
                    error_class=OrganelleDependencyError,
                    extra_details={"target": target},
                )
            except OrganelleDependencyError as error:
                # Launch failures and timeouts reported no exit code; keep their
                # original message distinct from a started process that exited
                # non-zero.
                details = cast(Mapping[str, Any], error.details)
                if details.get("returncode") is None:
                    prefix = "GetOrganelleDB initialization could not complete"
                else:
                    prefix = "get_organelle_config.py failed to initialize GetOrganelleDB"
                tail = tail_lines(str(details.get("stderr_tail", "")), MESSAGE_TAIL_LINES)
                raise OrganelleDependencyError(
                    code=error.code,
                    message=f"{prefix}: {tail}" if tail else prefix,
                    details=details,
                ) from error
            _write_prepared_metadata(candidate, identity)
            try:
                os.replace(candidate, prepared_root)
            except OSError:
                if not prepared_root.exists():
                    raise
            artifact = _prepared_artifact(prepared_root, identity)
        finally:
            shutil.rmtree(temporary, ignore_errors=True)
        return self._resources(database, digest, artifact)

    @staticmethod
    def _resources(
        database: ManagedDatabaseSpec, digest: str, artifact: ArtifactRef
    ) -> PreparedBackendResources:
        return PreparedBackendResources(
            artifacts=FrozenMap.from_items({"getorganelledb_config": artifact}),
            artifact_roles=("getorganelledb_config",),
            database_hashes=FrozenMap(
                {
                    "getorganelledb_archive": _database_source(database).sha256,
                    "getorganelledb_identity": digest,
                }
            ),
        )


class GetOrganelleAdapter:
    """Build and normalize one GetOrganelle Illumina assembly run."""

    backend_id = "getorganelle"

    def preflight(self, context: AdapterContext) -> None:
        if context.route.selected_backend != self.backend_id:
            raise _unsupported(
                "GetOrganelle adapter selected for a non-GetOrganelle route",
                selected_backend=context.route.selected_backend,
            )
        if context.environment.backend_id != self.backend_id:
            raise _unsupported("GetOrganelle adapter requires a GetOrganelle managed environment")
        if len(context.payload.short_libraries) != 1:
            raise _unsupported(
                "GetOrganelle accepts exactly one short-read library",
                short_library_count=len(context.payload.short_libraries),
            )
        library = context.payload.short_libraries[0]
        if library.technology != "illumina" or library.layout not in {"paired_end", "single_end"}:
            raise _unsupported(
                "GetOrganelle requires one explicit Illumina PE or SE library",
                technology=library.technology,
                layout=library.layout,
            )
        if library.layout == "paired_end" and library.read2_artifact is None:
            raise _unsupported("GetOrganelle paired-end input requires read2")
        if context.payload.long_libraries or context.payload.contig_inputs:
            raise _unsupported("GetOrganelle accepts short reads only")
        provided = context.request.backend_parameters
        if provided is not None and provided.backend != self.backend_id:
            raise _unsupported(
                "backend_parameters discriminator does not match GetOrganelle",
                discriminator=provided.backend,
            )

    def build_command(self, context: AdapterContext) -> AssemblyCommand:
        self.preflight(context)
        library = context.payload.short_libraries[0]
        stable = ["get_organelle_from_reads.py"]
        resolved = [str(context.environment.require_executable("get_organelle_from_reads.py"))]

        self._append_artifact(
            stable,
            resolved,
            "-1" if library.layout == "paired_end" else "-u",
            library.read1_artifact,
            context,
        )
        if library.layout == "paired_end":
            read2_role = library.read2_artifact
            if read2_role is None:
                raise _unsupported("GetOrganelle paired-end input requires read2")
            self._append_artifact(stable, resolved, "-2", read2_role, context)

        target_value = context.effective_backend_parameters.get("target")
        target = str(target_value)
        if target not in set(GETORGANELLE_TARGETS.values()):
            raise _unsupported("GetOrganelle target is not supported", target=target)
        output_root = context.workspace / "backend" / self.backend_id / "output"
        config = context.resources.artifacts.get("getorganelledb_config")
        if config is None:
            raise _environment_error("prepared GetOrganelleDB config is unavailable")
        stable.extend(
            (
                "-o",
                "role://workspace/output",
                "-F",
                target,
                "--config-dir",
                "role://artifact/getorganelledb_config",
                "-t",
                str(context.request.threads),
            )
        )
        resolved.extend(
            (
                "-o",
                str(output_root),
                "-F",
                target,
                "--config-dir",
                str(_config_root(config)),
                "-t",
                str(context.request.threads),
            )
        )

        provided = context.request.backend_parameters
        parameters = (
            provided if isinstance(provided, GetOrganelleParameters) else GetOrganelleParameters()
        )
        for field, flag in _PRESET_FLAGS:
            if getattr(parameters, field):
                stable.append(flag)
                resolved.append(flag)
        for field, flag, kind in _OPTION_FLAGS:
            value = getattr(parameters, field)
            if kind == "flag":
                if value:
                    stable.append(flag)
                    resolved.append(flag)
                continue
            if value is None:
                continue
            token = ",".join(str(item) for item in value) if kind == "kmers" else str(value)
            stable.extend((flag, _stable_value(token)))
            resolved.extend((flag, token))

        auxiliary = context.payload.auxiliary
        if auxiliary.seed_fasta_artifact is not None:
            self._append_artifact(stable, resolved, "-s", auxiliary.seed_fasta_artifact, context)
        if auxiliary.anti_seed_artifact is not None:
            self._append_artifact(stable, resolved, "-a", auxiliary.anti_seed_artifact, context)
        if auxiliary.label_genes_artifact is not None:
            self._append_artifact(
                stable, resolved, "--genes", auxiliary.label_genes_artifact, context
            )
        if auxiliary.exclude_genes_artifact is not None:
            self._append_artifact(
                stable, resolved, "--ex-genes", auxiliary.exclude_genes_artifact, context
            )
        return AssemblyCommand(stable_argv=tuple(stable), resolved_argv=tuple(resolved))

    @staticmethod
    def _append_artifact(
        stable: list[str],
        resolved: list[str],
        flag: str,
        role: str,
        context: AdapterContext,
    ) -> None:
        artifact = context.input_artifacts[role]
        stable.extend((flag, f"role://artifact/{role}"))
        resolved.extend((flag, artifact.uri))

    def collect_outputs(self, context: AdapterContext) -> RawAssemblyOutputs:
        output_root = context.workspace / "backend" / self.backend_id / "output"
        if not output_root.is_dir():
            raise _output_incomplete(
                "GetOrganelle output directory is missing", output_root=str(output_root)
            )
        fastas = sorted(
            path
            for path in output_root.glob("*.path_sequence.fasta")
            if path.is_file() and path.stat().st_size > 0
        )
        graphs = sorted(
            {
                *output_root.glob("*.selected_graph.gfa"),
                *output_root.glob("*.path_sequence.gfa"),
            }
        )
        graphs = [path for path in graphs if path.is_file() and path.stat().st_size > 0]
        if not fastas:
            raise _output_incomplete(
                "GetOrganelle produced no non-empty path FASTA", output_root=str(output_root)
            )
        if len(graphs) != 1:
            raise _output_incomplete(
                "GetOrganelle must produce exactly one selected path GFA",
                output_root=str(output_root),
                graph_count=len(graphs),
            )
        outputs = [BackendOutput(role="selected_graph", path=graphs[0], format="gfa")]
        for index, path in enumerate(fastas):
            outputs.append(BackendOutput(role=f"path_fasta_{index}", path=path, format="fasta"))
        log_path = output_root / "get_org.log.txt"
        if log_path.is_file() and log_path.stat().st_size > 0:
            outputs.append(
                BackendOutput(
                    role="backend_log",
                    path=log_path,
                    format="log",
                    media_type="text/plain",
                )
            )
        return RawAssemblyOutputs(
            outputs=tuple(outputs),
            primary_sequence_role="path_fasta_0",
            primary_graph_role="selected_graph",
        )

    def normalize(
        self,
        context: AdapterContext,
        raw: RawAssemblyOutputs,
        output_dir: Path,
    ) -> NormalizedAssemblyOutputs:
        del context
        output_dir.mkdir(parents=True, exist_ok=True)
        fasta_stats = normalize_fasta(
            raw.require(raw.primary_sequence_role).path, output_dir / "assembly.fasta"
        )
        normalize_gfa(raw.require("selected_graph").path, output_dir / "assembly.gfa")
        outputs = [
            BackendOutput(
                role="assembly_fasta", path=output_dir / "assembly.fasta", format="fasta"
            ),
            BackendOutput(role="assembly_graph", path=output_dir / "assembly.gfa", format="gfa"),
        ]
        alternate_roles: list[str] = []
        for item in raw.outputs:
            if item.role.startswith("path_fasta_") and item.role != raw.primary_sequence_role:
                index = item.role.removeprefix("path_fasta_")
                role = f"alternate_fasta_{index}"
                destination = output_dir / f"alternate_{index}.fasta"
                normalize_fasta(item.path, destination)
                alternate_roles.append(role)
                outputs.append(BackendOutput(role=role, path=destination, format="fasta"))
            elif item.role == "backend_log":
                destination = output_dir / "get_org.log.txt"
                shutil.copyfile(item.path, destination)
                outputs.append(
                    BackendOutput(
                        role="backend_log",
                        path=destination,
                        format="log",
                        media_type="text/plain",
                    )
                )
        return NormalizedAssemblyOutputs(
            outputs=tuple(outputs),
            primary_sequence_role="assembly_fasta",
            primary_graph_role="assembly_graph",
            alternate_sequence_roles=tuple(alternate_roles),
            record_count=fasta_stats.record_count,
            total_bases=fasta_stats.total_bases,
        )


def _stable_value(value: str) -> str:
    if re.fullmatch(r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)", value):
        return value
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_.+-]*", value):
        return value
    return "sha256_" + hashlib.sha256(value.encode()).hexdigest()
