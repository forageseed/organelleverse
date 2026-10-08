"""Managed Mash/RepeatMasker recommendation orchestration."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Annotated, Any, NoReturn, cast

from pydantic import Field

from .._bio import read_fasta, write_fasta
from ..core.artifacts import ArtifactRef
from ..core.errors import OrganelleExecutionError, OrganelleInputError
from ..core.external import failure_message
from ..core.genome import OrganelleGenome
from ..core.result import OrganelleResult
from ..runtime import (
    create_staged_run,
    managed_run_path,
    publish_staged_result,
    verify_staged_artifacts,
)
from ._contract import (
    OPERATION_VERSION,
    input_object_ids,
    make_provenance,
    result_scope,
)
from ._runner import CommandRecord, run_command
from .project import PangenomeProject
from .tuning import (
    SEGMENT_POLICY_SOURCE,
    Recommendation,
    estimate_identity,
    estimate_segment_length,
    parse_mash_distances,
    parse_repeatmasker_output,
    protected_segment_length,
    sequence_input_hashes,
    validate_recommendation,
)

_OPERATION_ID = "pangenome.recommend_parameters"
_POLICY_VERSION = "sample-mash-plastid-floor.v3"
_run_command = run_command


def recommend_parameters(
    genomes: list[OrganelleGenome],
    *,
    threads: Annotated[int, Field(ge=1, le=64)] = 4,
    run_repeatmasker: bool = True,
    species: str = "",
    identity_margin: Annotated[float, Field(ge=0, le=10)] = 2,
    repeat_multiplier: Annotated[float, Field(ge=0.1, le=10)] = 1.2,
    round_to: Annotated[int, Field(ge=1, le=100_000)] = 10,
    no_repeat_fallback: Annotated[int, Field(ge=1, le=10_000_000)] = 5000,
    include_rna: bool = True,
) -> OrganelleResult:
    """Recommend PGGB identity and segment length from verified inputs."""
    if (
        len(species) > 100
        or not 1 <= threads <= 64
        or not 0 <= identity_margin <= 10
        or not 0.1 <= repeat_multiplier <= 10
        or not 1 <= round_to <= 100_000
        or not 1 <= no_repeat_fallback <= 10_000_000
    ):
        raise OrganelleInputError(
            code="pangenome.invalid_tuning_parameter",
            message="pangenome tuning parameters are outside supported ranges",
        )
    project = PangenomeProject.from_genomes(genomes)
    project.verify_sources()
    hashes = list(sequence_input_hashes(genomes))
    mash = _find_executable("mash")
    mash_version = _tool_version(mash, "--version")
    repeatmasker = _find_executable("RepeatMasker") if run_repeatmasker else ""
    repeatmasker_version = _tool_version(repeatmasker, "-v") if run_repeatmasker else ""
    repeat_library, library_identity, default_database_label = (
        _repeatmasker_library(repeatmasker, species) if run_repeatmasker else ("", "", "")
    )
    database_label = os.environ.get("REPEATMASKER_DATABASE_LABEL", default_database_label)
    policy = {
        # Scope changes the segment policy even for identical sequence bytes.
        "policy_version": _POLICY_VERSION
        if any(g.organelle == "plastid" for g in genomes)
        else "sample-mash-mitochondrial.v3",
        "input_hashes": hashes,
        "threads": threads,
        "run_repeatmasker": run_repeatmasker,
        "species": species,
        "identity_margin": identity_margin,
        "repeat_multiplier": repeat_multiplier,
        "round_to": round_to,
        "no_repeat_fallback": no_repeat_fallback,
        "include_rna": include_rna,
        "mash_executable": mash,
        "mash_version": mash_version,
        "repeatmasker_executable": repeatmasker,
        "repeatmasker_version": repeatmasker_version,
        "repeatmasker_library_identity": library_identity,
        "repeatmasker_database_label": database_label,
    }
    run_digest = _digest(policy)
    run_id = f"sha256-{run_digest}"
    completed = managed_run_path(_OPERATION_ID, run_id)
    if completed.exists():
        return _load_cached(completed, genomes, run_digest=run_digest, policy=policy)

    staging = create_staged_run(_OPERATION_ID, run_id)
    workspace = staging / "workspace"
    try:
        workspace.mkdir()
        manifest = project.materialize(workspace / "input")
        pansn = Path(manifest.fasta_path)
        sample_dir = workspace / "mash_samples"
        sample_dir.mkdir()
        sequences = dict(read_fasta(pansn))
        sample_files = []
        for sample in manifest.samples:
            sample_file = sample_dir / f"{sample.sample}.fa"
            write_fasta(sample_file, [(name, sequences[name]) for name in sample.headers])
            sample_files.append(sample_file)
        mash_prefix = workspace / "all"
        # Mash's default unit is one input file. One file per biological sample
        # preserves multi-molecule membership and avoids a pooled self-distance.
        sketch_argv = [
            mash,
            "sketch",
            "-p",
            str(threads),
            "-o",
            str(mash_prefix),
            *map(str, sample_files),
        ]
        _require_success(_run_command(sketch_argv, cwd=workspace), "Mash sketch")
        sketch = mash_prefix.with_suffix(".msh")
        if not sketch.is_file():
            _fail("pangenome.mash_output_missing", "Mash sketch did not produce its database")
        mash_output = workspace / "mash-dist.tsv"
        distance_argv = [mash, "dist", str(sketch), str(sketch)]
        _require_success(
            _run_command(distance_argv, stdout_path=mash_output, cwd=workspace),
            "Mash distance",
        )
        distances = parse_mash_distances(mash_output.read_text(encoding="utf-8"))
        _require_sample_pairs(mash_output.read_text(encoding="utf-8"), sample_files)

        repeat_argv: list[str] = []
        repeat_intervals = ()
        if run_repeatmasker:
            repeat_dir = workspace / "repeatmasker"
            repeat_dir.mkdir()
            repeat_argv = [repeatmasker, "-pa", str(threads), "-dir", str(repeat_dir)]
            if species:
                repeat_argv.extend(("-species", species))
            else:
                repeat_argv.extend(("-lib", repeat_library))
            repeat_argv.append(str(pansn))
            _require_success(_run_command(repeat_argv, cwd=workspace), "RepeatMasker")
            repeat_output = repeat_dir / f"{pansn.name}.out"
            if not repeat_output.is_file():
                _fail(
                    "pangenome.repeatmasker_output_missing",
                    "RepeatMasker did not produce its .out table",
                )
            repeat_intervals = parse_repeatmasker_output(repeat_output.read_text(encoding="utf-8"))

        identity = estimate_identity(distances, margin=identity_margin)
        segment = estimate_segment_length(
            repeat_intervals,
            multiplier=repeat_multiplier,
            round_to=round_to,
            no_repeat_fallback=no_repeat_fallback,
            include_rna=include_rna,
        )
        recommendation: Recommendation = {
            "schema_version": "organelleverse.pangenome.recommendation.v1",
            **policy,
            "identity": identity,
            "segment_length": protected_segment_length(genomes, segment.segment_length),
            "max_mash_distance": max(distances),
            "longest_repeat_span": segment.longest_span,
            "repeat_fallback_used": segment.fallback_used,
            "mash_sketch_argv": sketch_argv,
            "mash_distance_argv": distance_argv,
            "repeatmasker_argv": repeat_argv,
        }
        recommendation["recommendation_digest"] = _digest(recommendation)
        artifact_dir = staging / "artifacts"
        artifact_dir.mkdir()
        sample_evidence = workspace / "sample-evidence.json"
        sample_evidence.write_text(
            _canonical_json(
                {
                    "unit": "biological_sample",
                    "mash_label_policy": "historical input filenames in raw Mash output",
                    "samples": [
                        {
                            "sample": sample.sample,
                            "molecules": list(sample.headers),
                            "mash_label": str(path),
                        }
                        for sample, path in zip(manifest.samples, sample_files, strict=True)
                    ],
                }
            )
            + "\n"
        )
        evidence_paths = []
        for source, name, kind, fmt in [
            (mash_output, "mash-distances.tsv", "pangenome_sample_distances", "tsv"),
            (sample_evidence, "samples.json", "pangenome_staging_manifest", "json"),
            *(
                [(repeat_output, "repeats.out", "pangenome_repeat_evidence", "txt")]
                if run_repeatmasker
                else []
            ),
        ]:
            output = artifact_dir / name
            shutil.copyfile(source, output)
            evidence_paths.append(
                ArtifactRef.from_path(output, kind=kind, format=fmt).model_copy(
                    update={"uri": f"artifacts/{name}"}
                )
            )
        shutil.rmtree(workspace)
        target = artifact_dir / "recommendation.json"
        target.write_text(_canonical_json(recommendation) + "\n", encoding="utf-8")
        artifact = ArtifactRef.from_path(
            target,
            kind="pangenome_parameter_recommendation",
            format="json",
            media_type="application/json",
        ).model_copy(update={"uri": "artifacts/recommendation.json"})
        base_result = _recommendation_result(genomes, recommendation, artifact)
        base_result = base_result.model_copy(
            update={
                "metrics": {
                    **base_result.metrics,
                    "segment_length_policy": {
                        "derived": segment.segment_length,
                        "recommended": recommendation["segment_length"],
                        "floor": 5000 if any(g.organelle == "plastid" for g in genomes) else None,
                        "source": SEGMENT_POLICY_SOURCE,
                    },
                }
            }
        )
        base_result = base_result.model_copy(
            update={"artifacts": (*base_result.artifacts, *evidence_paths)}
        )
        record_path = staging / "recommendation-run-record.json"
        record_path.write_text(
            _canonical_json(
                {
                    "schema_version": 1,
                    "operation_id": _OPERATION_ID,
                    "operation_version": OPERATION_VERSION,
                    "run_digest": run_digest,
                    "policy": policy,
                    "result": base_result.model_dump(mode="json"),
                }
            )
            + "\n",
            encoding="utf-8",
        )
        record_artifact = ArtifactRef.from_path(
            record_path,
            kind="pangenome_recommendation_run_record",
            format="json",
            media_type="application/json",
        ).model_copy(update={"uri": "recommendation-run-record.json"})
        result = base_result.model_copy(
            update={"artifacts": (*base_result.artifacts, record_artifact)}
        )
        published = publish_staged_result(
            result,
            staging,
            completed,
            trusted_input_hashes=frozenset(hashes),
        )
        assert isinstance(published, OrganelleResult)
        return published
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def _recommendation_result(
    genomes: list[OrganelleGenome],
    recommendation: Recommendation,
    artifact: ArtifactRef,
) -> OrganelleResult:
    backend = "mash+repeatmasker" if recommendation["run_repeatmasker"] else "mash"
    return OrganelleResult(
        operation_id=_OPERATION_ID,
        operation_version=OPERATION_VERSION,
        scope=result_scope(genomes),
        status="ok",
        summary_text="Evidence-linked PGGB parameters recommended.",
        metrics=cast(Any, {"recommendation": recommendation}),
        artifacts=(artifact,),
        provenance=make_provenance(
            operation_id=_OPERATION_ID,
            parameters={
                key: recommendation[key]
                for key in (
                    "threads",
                    "run_repeatmasker",
                    "species",
                    "identity_margin",
                    "repeat_multiplier",
                    "round_to",
                    "no_repeat_fallback",
                    "include_rna",
                )
            },
            input_object_ids=input_object_ids(genomes),
            input_artifact_hashes=sequence_input_hashes(genomes),
            actual_backend=backend,
            attempted_backends=(backend,),
            argv=tuple(cast(list[str], recommendation["mash_distance_argv"])),
        ),
    )


def _load_cached(
    completed: Path,
    genomes: list[OrganelleGenome],
    *,
    run_digest: str,
    policy: dict[str, Any],
) -> OrganelleResult:
    if completed.is_symlink() or not completed.is_dir():
        raise OrganelleInputError(
            code="pangenome.invalid_recommendation_cache",
            message="cached pangenome recommendation must be a real managed directory",
            details={"path": str(completed)},
        )
    path = completed / "artifacts" / "recommendation.json"
    record_path = completed / "recommendation-run-record.json"
    try:
        loaded_record = cast(object, json.loads(record_path.read_text(encoding="utf-8")))
        if not isinstance(loaded_record, dict):
            raise ValueError("run record must be an object")
        payload = cast(dict[str, Any], loaded_record)
        expected = {
            "schema_version": 1,
            "operation_id": _OPERATION_ID,
            "operation_version": OPERATION_VERSION,
            "run_digest": run_digest,
            "policy": policy,
        }
        if any(payload.get(key) != value for key, value in expected.items()):
            raise ValueError("run record identity does not match the request")
        result = OrganelleResult.model_validate(payload["result"])
        expected_names = {
            "artifacts/recommendation.json",
            "artifacts/mash-distances.tsv",
            "artifacts/samples.json",
        }
        if policy["run_repeatmasker"]:
            expected_names.add("artifacts/repeats.out")
        if (
            result.operation_id != _OPERATION_ID
            or {a.uri for a in result.artifacts} != expected_names
        ):
            raise ValueError("run record result shape does not match the operation")
        loaded = cast(object, json.loads(path.read_text(encoding="utf-8")))
        if not isinstance(loaded, dict):
            raise ValueError("recommendation must be an object")
        raw = cast(dict[str, object], loaded)
        recommendation = validate_recommendation(
            raw,
            input_hashes=sequence_input_hashes(genomes),
            identity=cast(float, raw.get("identity")),
            segment_length=cast(int, raw.get("segment_length")),
        )
        if result.model_dump(mode="json")["metrics"]["recommendation"] != recommendation:
            raise ValueError("run record recommendation does not match the artifact")
        if any(recommendation.get(key) != value for key, value in policy.items()):
            raise ValueError("recommendation policy does not match the request")
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise OrganelleInputError(
            code="pangenome.invalid_recommendation_cache",
            message="cached pangenome recommendation is invalid",
            details={"reason": str(error)},
        ) from error
    record_artifact = ArtifactRef.from_path(
        record_path,
        kind="pangenome_recommendation_run_record",
        format="json",
        media_type="application/json",
    ).model_copy(update={"uri": "recommendation-run-record.json"})
    staged_shape = result.model_copy(update={"artifacts": (*result.artifacts, record_artifact)})
    verify_staged_artifacts(
        staged_shape,
        completed,
        trusted_input_hashes=frozenset(sequence_input_hashes(genomes)),
    )
    return staged_shape.model_copy(
        update={
            "artifacts": tuple(
                artifact.model_copy(update={"uri": str(completed / artifact.uri)})
                for artifact in staged_shape.artifacts
            )
        }
    )


def _find_executable(name: str) -> str:
    found = shutil.which(name)
    if found is None:
        _fail("pangenome.tuning_tool_missing", f"Required tuning tool is missing: {name}")
    return str(Path(found).expanduser().resolve())


def _require_sample_pairs(output: str, sample_files: list[Path]) -> None:
    names = {str(path) for path in sample_files}
    observed = []
    for line in output.splitlines():
        if line.strip():
            fields = line.split("\t") if "\t" in line else line.split()
            observed.append((fields[0], fields[1]))
    expected = {(left, right) for left in names for right in names}
    if len(observed) != len(expected) or set(observed) != expected:
        _fail(
            "pangenome.mash_sample_pairs_missing",
            "Mash must report every biological-sample pair exactly once",
        )


def _repeatmasker_library(executable: str, species: str) -> tuple[str, str, str]:
    if species:
        return "", f"species:{species}", "installed-famdb"
    configured = os.environ.get("REPEATMASKER_LIBRARY")
    library = (
        Path(configured).expanduser()
        if configured
        else Path(executable).resolve().parent / "Libraries" / "RepeatMasker.lib"
    )
    if not library.is_file():
        _fail(
            "pangenome.repeatmasker_library_missing",
            "RepeatMasker default library is missing",
            path=str(library),
        )
    digest = hashlib.sha256(library.read_bytes()).hexdigest()
    return str(library.resolve()), f"sha256:{digest}", f"RepeatMasker.lib:{digest[:12]}"


def _tool_version(executable: str, flag: str) -> str:
    with tempfile.TemporaryDirectory(prefix="organelleverse-version-") as temporary:
        output = Path(temporary) / "version.txt"
        record = _run_command([executable, flag], stdout_path=output)
        _require_success(record, f"{Path(executable).name} version probe")
        text = output.read_text(encoding="utf-8", errors="replace").strip() or record.stderr.strip()
    if not text:
        _fail("pangenome.tuning_version_missing", "Tuning tool returned no version")
    return text.splitlines()[0].strip()


def _require_success(record: CommandRecord, label: str) -> None:
    if not record.ok:
        _fail(
            "pangenome.tuning_tool_failed",
            failure_message(label, record.returncode, record.stderr),
            argv=list(record.argv),
            returncode=record.returncode,
            stderr=record.stderr,
        )


def _canonical_json(value: object) -> str:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True
    )


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _fail(code: str, message: str, **details: object) -> NoReturn:
    raise OrganelleExecutionError(code=code, message=message, details=details)
