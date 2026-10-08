"""Result codecs: turn an implementation's direct Python return value into an
OrganelleResult, without ever falling back to ``str(value)``.

Exactly five codecs, matching ``organelleverse.operations.spec.ResultCodec``:

* ``canonical`` — the value already is an ``OrganelleResult``; revalidate it,
  preserving its own provenance unchanged (never fabricated or overwritten).
* ``canonical_json`` — reconstruct exactly the trusted contract's declared L1
  output class from finite worker JSON in the parent.
* ``legacy_result`` — accept an already-canonical result, or convert an
  archived pre-v1 legacy result field by field (suite+op -> operation_id,
  organelle -> scope, observed_metrics -> metrics, key_findings -> Finding,
  output_paths -> real-hash ArtifactRef, anomalies -> ErrorDetail on failure,
  provenance -> a real ResultProvenance); never a wholesale
  ``OrganelleResult.model_validate(dict(old))`` masquerading as conversion,
  and never a restoration of the pre-v1 legacy shim.
* ``json_metric`` — accept only recursively finite JSON and store it under
  ``metrics[result_key]``.
* ``artifact`` — accept a ``Path``, a sequence of paths, an
  :class:`~organelleverse.visualization.plot_object.OrganellePlot`, or NumPy
  data, materialize it inside an L6 managed run, and declare it as a
  content-hashed :class:`~organelleverse.core.artifacts.ArtifactRef`.
"""

from __future__ import annotations

import math
import os
import shutil
from collections.abc import Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import cast, get_args, overload

from pydantic import Field, ValidationError

from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.data import LineageRecord, OrganelleData
from organelleverse.core.errors import OrganelleContractError, OrganelleExecutionError
from organelleverse.core.frozen import FrozenJson, FrozenMap, freeze_json
from organelleverse.core.genome import OrganelleGenome
from organelleverse.core.provenance import (
    InlineMaterializationRecord,
    PathTranslationRecord,
    ResultProvenance,
)
from organelleverse.core.result import (
    ErrorDetail,
    Finding,
    OrganelleResult,
    ResultScope,
    ResultStatus,
)
from organelleverse.operations.spec import (
    CoreKind,
    PythonBindingSpec,
    ResultCodec,
    StrictSpecModel,
)
from organelleverse.plugin_protocol import PLUGIN_OPTIMIZATION_SCORE_METRIC
from organelleverse.runtime import managed_run_path


class CapabilityExecutionContext(StrictSpecModel):
    """Identity of one capability invocation, threaded into its ResultProvenance.

    ``run_id`` identifies where a managed run is materialized on disk - it is
    run bookkeeping, not scientific identity, and (like a timestamp) must
    never enter ``parameters_hash``/``input_object_ids``/``input_artifact_hashes``.
    The other fields are exactly what :class:`~organelleverse.core.provenance.ResultProvenance`
    needs to record who produced a result and from what, so every codec can
    build a real provenance record instead of leaving it ``None``.
    """

    operation_id: str = Field(pattern=r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$")
    operation_version: str = Field(pattern=r"^[0-9]+\.[0-9]+$")
    callable_locator: str = Field(pattern=r"^[a-zA-Z_][\w.]*:[a-zA-Z_]\w*$")
    run_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    input_object_ids: tuple[str, ...] = ()
    input_artifact_hashes: tuple[str, ...] = ()
    parameters_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    path_translations: tuple[PathTranslationRecord, ...] = ()
    inline_materializations: tuple[InlineMaterializationRecord, ...] = ()


def _codec_error(message: str, **details: object) -> OrganelleContractError:
    return OrganelleContractError(
        code="capability.result_codec_invalid", message=message, details=details
    )


def _package_version() -> str:
    try:
        return version("organelleverse")
    except PackageNotFoundError:  # pragma: no cover - unreachable in an editable install
        return "0.0.1"


def _build_provenance(context: CapabilityExecutionContext) -> ResultProvenance:
    """Build the shared, real provenance record every non-canonical codec attaches.

    ``context`` already carries every identity-relevant fact (see its
    docstring for why ``run_id`` itself is deliberately excluded here).
    """
    return ResultProvenance(
        run_id=context.run_id,
        operation_id=context.operation_id,
        operation_version=context.operation_version,
        package_version=_package_version(),
        git_commit=os.environ.get("ORG_VERSE_GIT_COMMIT", ""),
        input_object_ids=context.input_object_ids,
        input_artifact_hashes=context.input_artifact_hashes,
        parameters_hash=context.parameters_hash,
        callable_locator=context.callable_locator,
        path_translations=context.path_translations,
        inline_materializations=context.inline_materializations,
    )


@overload
def encode_capability_result(
    value: object,
    binding: PythonBindingSpec,
    context: CapabilityExecutionContext,
) -> OrganelleResult: ...


@overload
def encode_capability_result(
    value: object,
    binding: PythonBindingSpec,
    context: CapabilityExecutionContext,
    *,
    output_kind: CoreKind,
) -> OrganelleGenome | OrganelleData | OrganelleResult: ...


def encode_capability_result(
    value: object,
    binding: PythonBindingSpec,
    context: CapabilityExecutionContext,
    *,
    output_kind: CoreKind | None = None,
) -> OrganelleGenome | OrganelleData | OrganelleResult:
    """Encode *value* per ``binding.result_codec``. Never coerces via ``str()``."""
    if binding.result_codec is ResultCodec.CANONICAL:
        return _encode_canonical(value)
    if binding.result_codec is ResultCodec.CANONICAL_JSON:
        return _encode_canonical_json(value, output_kind, context)
    if binding.result_codec is ResultCodec.LEGACY_RESULT:
        return _encode_legacy_result(value, context)
    if binding.result_codec is ResultCodec.JSON_METRIC:
        assert binding.result_key is not None, "PythonBindingSpec guarantees this for json_metric"
        return _encode_json_metric(value, context, binding.result_key)
    if binding.result_codec is ResultCodec.ARTIFACT:
        return _encode_artifact(value, context)
    raise AssertionError(f"unhandled result codec: {binding.result_codec!r}")  # pragma: no cover


def _encode_canonical_json(
    value: object,
    output_kind: CoreKind | None,
    context: CapabilityExecutionContext,
) -> OrganelleGenome | OrganelleData | OrganelleResult:
    if output_kind is None:
        raise _codec_error(
            "canonical_json requires a trusted Genome, Data, or Result output kind",
            output_kind=None,
        )
    model_types = {
        CoreKind.GENOME: OrganelleGenome,
        CoreKind.DATA: OrganelleData,
        CoreKind.RESULT: OrganelleResult,
    }
    model_type = model_types.get(output_kind)
    if model_type is None:
        raise _codec_error(
            "canonical_json requires a trusted Genome, Data, or Result output kind",
            output_kind=output_kind.value,
        )
    if not _is_finite_json(value):
        raise _codec_error(
            "canonical_json requires recursively finite JSON from the worker",
            actual_type=type(value).__name__,
        )
    if output_kind is CoreKind.RESULT and isinstance(value, Mapping):
        worker_operation_id = cast("Mapping[str, object]", value).get("operation_id")
        if worker_operation_id != context.operation_id:
            raise _codec_error(
                "canonical_json Result operation_id does not match the bound operation",
                operation_id=context.operation_id,
                worker_operation_id=worker_operation_id,
            )
    try:
        decoded = model_type.model_validate(value)
    except ValidationError as error:
        raise _codec_error(
            f"canonical_json failed to reconstruct {output_kind.value}",
            errors=error.errors(
                include_url=False,
                include_context=False,
                include_input=False,
            ),
        ) from error
    if isinstance(decoded, OrganelleResult) and decoded.operation_id != context.operation_id:
        raise _codec_error(
            "canonical_json Result operation_id does not match the bound operation",
            operation_id=context.operation_id,
            worker_operation_id=decoded.operation_id,
        )
    return decoded


def normalize_worker_result(
    result: OrganelleResult,
    context: CapabilityExecutionContext,
) -> OrganelleResult:
    """Apply parent-owned identity to a decoded Result at the worker boundary.

    Result codecs remain usable by trusted in-process bindings with their
    established semantics.  A bundle worker, however, can supply scientific
    fields only: the parent correlates the operation and exclusively owns the
    root contract version and execution provenance.
    """
    if result.operation_id != context.operation_id:
        raise _codec_error(
            "worker Result operation_id does not match the bound operation",
            operation_id=context.operation_id,
            worker_operation_id=result.operation_id,
        )
    return result.model_copy(
        update={
            "operation_version": context.operation_version,
            "provenance": _build_provenance(context),
        }
    )


def normalize_worker_core_object(
    value: OrganelleGenome | OrganelleData | OrganelleResult,
    context: CapabilityExecutionContext,
    *,
    input: OrganelleGenome | OrganelleData | OrganelleResult | None,
) -> OrganelleGenome | OrganelleData | OrganelleResult:
    """Apply parent-owned provenance or lineage to a decoded worker value.

    Trusted in-process codecs retain their established semantics.  Only the
    controlled-worker boundary calls this helper, so a worker cannot forge
    Result provenance or Genome/Data transformation lineage.
    """

    if isinstance(value, OrganelleResult):
        return normalize_worker_result(value, context)
    if value.lineage:
        raise _codec_error(
            "worker Genome/Data output must not supply transformation lineage",
            operation_id=context.operation_id,
            output_kind=value.kind,
        )
    inherited: tuple[LineageRecord, ...] = ()
    if isinstance(input, (OrganelleGenome, OrganelleData)):
        inherited = input.lineage
    parent_record = LineageRecord(
        parent_object_ids=context.input_object_ids,
        operation_id=context.operation_id,
        operation_version=context.operation_version,
        parameters_hash=context.parameters_hash,
    )
    return value.model_copy(update={"lineage": (*inherited, parent_record)})


def _encode_canonical(value: object) -> OrganelleResult:
    if not isinstance(value, OrganelleResult):
        raise _codec_error(
            "canonical result codec requires the implementation to already return "
            "an OrganelleResult",
            actual_type=type(value).__name__,
        )
    try:
        return OrganelleResult.model_validate(value)
    except ValidationError as error:
        raise _codec_error("canonical result failed revalidation", errors=error.errors()) from error


def _encode_legacy_result(value: object, context: CapabilityExecutionContext) -> OrganelleResult:
    if isinstance(value, OrganelleResult):
        return OrganelleResult.model_validate(value)
    if isinstance(value, Mapping):
        return _convert_legacy_result_mapping(cast("Mapping[str, object]", value), context)
    raise _codec_error(
        "legacy_result codec supports only an OrganelleResult or an archived legacy "
        "result mapping; restoring the pre-v1 legacy shim to interpret an older shape "
        "is forbidden",
        actual_type=type(value).__name__,
    )


_LEGACY_RESULT_REQUIRED_KEYS = frozenset(
    {
        "suite",
        "op",
        "organelle",
        "status",
        "output_paths",
        "observed_metrics",
        "key_findings",
        "flags",
        "anomalies",
        "summary_text",
        "provenance",
    }
)
_LEGACY_SCOPES: frozenset[str] = frozenset(get_args(ResultScope))
_LEGACY_STATUSES: frozenset[str] = frozenset(get_args(ResultStatus))


def _convert_legacy_result_mapping(
    legacy: Mapping[str, object], context: CapabilityExecutionContext
) -> OrganelleResult:
    """Convert an archived pre-v1 legacy result, field by field, into an OrganelleResult.

    Every field is read and validated explicitly; an unconvertible field
    raises immediately rather than being silently dropped or coerced.
    """
    missing = _LEGACY_RESULT_REQUIRED_KEYS - legacy.keys()
    if missing:
        raise _codec_error(
            "legacy_result mapping is missing required legacy fields",
            operation_id=context.operation_id,
            missing_fields=sorted(missing),
        )
    operation_id = _legacy_operation_id(legacy, context)
    status = _legacy_status(legacy)
    try:
        return OrganelleResult(
            operation_id=operation_id,
            scope=_legacy_scope(legacy),
            status=status,
            summary_text=_legacy_summary_text(legacy),
            metrics=_legacy_metrics(legacy),
            findings=_legacy_findings(legacy),
            flags=_legacy_flags(legacy),
            artifacts=_legacy_artifacts(legacy, context),
            errors=_legacy_errors(legacy) if status == "failed" else (),
            provenance=_legacy_provenance(legacy, context),
        )
    except ValidationError as error:
        raise _codec_error(
            "legacy_result conversion produced an invalid OrganelleResult",
            operation_id=context.operation_id,
            errors=error.errors(),
        ) from error


def _legacy_operation_id(legacy: Mapping[str, object], context: CapabilityExecutionContext) -> str:
    suite = legacy.get("suite")
    op = legacy.get("op")
    if not isinstance(suite, str) or not suite or not isinstance(op, str) or not op:
        raise _codec_error(
            "legacy_result 'suite' and 'op' must be non-empty strings",
            operation_id=context.operation_id,
        )
    operation_id = f"{suite}.{op}"
    if operation_id != context.operation_id:
        raise _codec_error(
            "legacy_result suite+op does not match the bound operation_id",
            operation_id=context.operation_id,
            legacy_operation_id=operation_id,
        )
    return operation_id


def _legacy_scope(legacy: Mapping[str, object]) -> ResultScope:
    organelle = legacy.get("organelle")
    if organelle not in _LEGACY_SCOPES:
        raise _codec_error(
            f"legacy_result 'organelle' must be one of {sorted(_LEGACY_SCOPES)}",
            actual=organelle,
        )
    return cast("ResultScope", organelle)


def _legacy_status(legacy: Mapping[str, object]) -> ResultStatus:
    status = legacy.get("status")
    if status not in _LEGACY_STATUSES:
        raise _codec_error(
            f"legacy_result 'status' must be one of {sorted(_LEGACY_STATUSES)}",
            actual=status,
        )
    return cast("ResultStatus", status)


def _legacy_summary_text(legacy: Mapping[str, object]) -> str:
    summary_text = legacy.get("summary_text")
    if not isinstance(summary_text, str):
        raise _codec_error("legacy_result 'summary_text' must be a string")
    return summary_text


def _legacy_metrics(legacy: Mapping[str, object]) -> FrozenMap[FrozenJson]:
    observed_metrics = legacy.get("observed_metrics")
    if not isinstance(observed_metrics, Mapping):
        raise _codec_error(
            "legacy_result 'observed_metrics' must be a recursively finite JSON object"
        )
    typed_metrics = cast("Mapping[str, object]", observed_metrics)
    if not _is_finite_json(typed_metrics):
        raise _codec_error(
            "legacy_result 'observed_metrics' must be a recursively finite JSON object"
        )
    return cast("FrozenMap[FrozenJson]", freeze_json(typed_metrics))


def _legacy_findings(legacy: Mapping[str, object]) -> tuple[Finding, ...]:
    key_findings = legacy.get("key_findings")
    if not isinstance(key_findings, Sequence) or isinstance(key_findings, (str, bytes)):
        raise _codec_error("legacy_result 'key_findings' must be a sequence")
    findings: list[Finding] = []
    for index, item in enumerate(cast("Sequence[object]", key_findings)):
        if not isinstance(item, Mapping):
            raise _codec_error(f"legacy_result key_findings[{index}] must be a JSON object")
        try:
            findings.append(Finding.model_validate(dict(cast("Mapping[str, object]", item))))
        except ValidationError as error:
            raise _codec_error(
                f"legacy_result key_findings[{index}] did not convert to a Finding",
                errors=error.errors(),
            ) from error
    return tuple(findings)


def _legacy_flags(legacy: Mapping[str, object]) -> tuple[str, ...]:
    flags = legacy.get("flags")
    if (
        not isinstance(flags, Sequence)
        or isinstance(flags, (str, bytes))
        or not all(isinstance(flag, str) for flag in cast("Sequence[object]", flags))
    ):
        raise _codec_error("legacy_result 'flags' must be a sequence of strings")
    return tuple(cast("Sequence[str]", flags))


def _legacy_artifacts(
    legacy: Mapping[str, object], context: CapabilityExecutionContext
) -> tuple[ArtifactRef, ...]:
    output_paths = legacy.get("output_paths")
    if not isinstance(output_paths, Sequence) or isinstance(output_paths, (str, bytes)):
        raise _codec_error("legacy_result 'output_paths' must be a sequence of paths")
    paths: list[Path] = []
    for item in cast("Sequence[object]", output_paths):
        if not isinstance(item, (str, Path)):
            raise _codec_error("legacy_result 'output_paths' entries must be str or Path")
        paths.append(Path(item))
    if not paths:
        return ()
    return _encode_path_artifacts(context, tuple(paths)).artifacts


def _legacy_errors(legacy: Mapping[str, object]) -> tuple[ErrorDetail, ...]:
    anomalies = legacy.get("anomalies")
    if not isinstance(anomalies, Sequence) or isinstance(anomalies, (str, bytes)) or not anomalies:
        raise _codec_error(
            "legacy_result 'failed' status requires a non-empty 'anomalies' sequence"
        )
    errors: list[ErrorDetail] = []
    for index, item in enumerate(cast("Sequence[object]", anomalies)):
        if not isinstance(item, Mapping):
            raise _codec_error(f"legacy_result anomalies[{index}] must be a JSON object")
        try:
            errors.append(ErrorDetail.model_validate(dict(cast("Mapping[str, object]", item))))
        except ValidationError as error:
            raise _codec_error(
                f"legacy_result anomalies[{index}] did not convert to an ErrorDetail",
                errors=error.errors(),
            ) from error
    return tuple(errors)


def _legacy_provenance(
    legacy: Mapping[str, object], context: CapabilityExecutionContext
) -> ResultProvenance:
    old = legacy.get("provenance")
    if not isinstance(old, Mapping):
        raise _codec_error("legacy_result 'provenance' must be a JSON object")
    old_map = cast("Mapping[str, object]", old)
    try:
        return ResultProvenance(
            operation_id=context.operation_id,
            operation_version=_legacy_provenance_string(old_map, "operation_version"),
            package_version=_legacy_provenance_string(old_map, "package_version"),
            git_commit=_legacy_provenance_string(old_map, "git_commit"),
            parameters_hash=_legacy_provenance_string(old_map, "parameters_hash"),
            callable_locator=context.callable_locator,
        )
    except ValidationError as error:
        raise _codec_error(
            "legacy_result 'provenance' did not convert to a ResultProvenance",
            errors=error.errors(),
        ) from error


def _legacy_provenance_string(old: Mapping[str, object], field: str) -> str:
    value = old.get(field)
    if not isinstance(value, str):
        raise _codec_error(f"legacy_result 'provenance.{field}' must be a string", actual=value)
    return value


def _encode_json_metric(
    value: object, context: CapabilityExecutionContext, result_key: str
) -> OrganelleResult:
    if not _is_finite_json(value):
        raise _codec_error(
            "json_metric codec requires a recursively finite JSON value "
            "(no NaN/Inf, no arbitrary objects, no sets or generators)",
            actual_type=type(value).__name__,
        )
    frozen_metrics = cast("FrozenMap[FrozenJson]", freeze_json({result_key: value}))
    return OrganelleResult(
        operation_id=context.operation_id,
        scope="none",
        status="ok",
        metrics=frozen_metrics,
        provenance=_build_provenance(context),
    )


def _is_finite_json(value: object) -> bool:
    if value is None or isinstance(value, (bool, str)):
        return True
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, Mapping):
        mapping = cast("Mapping[object, object]", value)
        return all(isinstance(key, str) and _is_finite_json(item) for key, item in mapping.items())
    if isinstance(value, (list, tuple)):
        items = cast("Sequence[object]", value)
        return all(_is_finite_json(item) for item in items)
    return False


def _encode_artifact(value: object, context: CapabilityExecutionContext) -> OrganelleResult:
    from organelleverse.visualization.plot_object import OrganellePlot

    if isinstance(value, OrganellePlot):
        return _encode_plot_artifact(value, context)
    if isinstance(value, Path):
        return _encode_path_artifacts(context, (value,))
    path_sequence = _as_path_sequence(value)
    if path_sequence is not None:
        return _encode_path_artifacts(context, path_sequence)
    numpy_result = _try_encode_numpy_artifact(value, context)
    if numpy_result is not None:
        return numpy_result
    raise _codec_error(
        "artifact codec supports Path, a sequence of Path, OrganellePlot, or NumPy data",
        actual_type=type(value).__name__,
    )


def _as_path_sequence(value: object) -> tuple[Path, ...] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    items = cast("Sequence[object]", value)
    if not items or not all(isinstance(item, Path) for item in items):
        return None
    return tuple(cast("list[Path]", list(items)))


def _encode_plot_artifact(plot: object, context: CapabilityExecutionContext) -> OrganelleResult:
    from organelleverse.visualization.plot_object import OrganellePlot

    assert isinstance(plot, OrganellePlot)
    if plot.renderer is None:
        raise _codec_error("OrganellePlot artifact has no captured renderer to materialize")
    destination = managed_run_path(context.operation_id, context.run_id) / "artifact.png"
    materialized = plot.materialize(destination)
    if materialized.provenance is not None:
        # The plot object already tracked its own real provenance; the codec
        # attaches one only when the source had none, never overwriting it.
        return materialized
    return materialized.model_copy(update={"provenance": _build_provenance(context)})


def _unique_artifact_destination_names(paths: tuple[Path, ...]) -> tuple[str, ...]:
    """Return a stable, deterministic, collision-free destination name per path.

    A single path keeps its exact original basename - there is no possible
    collision to guard against, and this preserves prior behavior exactly.
    Two or more paths are always index-prefixed, even when their basenames
    already differ: the index is unique by construction, so this can never
    silently overwrite one artifact's declared digest with another's content
    the way a bare ``path.name`` destination could when two inputs share a
    basename from different source directories (e.g. one "summary.txt" per
    chromosome).
    """
    if len(paths) <= 1:
        return tuple(path.name for path in paths)
    digits = len(str(len(paths) - 1))
    return tuple(f"{index:0{digits}d}_{path.name}" for index, path in enumerate(paths))


def _encode_path_artifacts(
    context: CapabilityExecutionContext, paths: tuple[Path, ...]
) -> OrganelleResult:
    for path in paths:
        if not path.is_file():
            raise _codec_error(f"artifact path does not exist: {path}")
    run_root = managed_run_path(context.operation_id, context.run_id)
    run_root.mkdir(parents=True, exist_ok=True)
    destination_names = _unique_artifact_destination_names(paths)
    artifacts: list[ArtifactRef] = []
    for path, destination_name in zip(paths, destination_names, strict=True):
        destination = run_root / destination_name
        if destination.resolve() != path.resolve():
            shutil.copy2(path, destination)
        artifacts.append(
            ArtifactRef.from_path(
                destination,
                kind="artifact",
                format=destination.suffix.lstrip(".") or "bin",
            )
        )
    return OrganelleResult(
        operation_id=context.operation_id,
        scope="none",
        status="ok",
        artifacts=tuple(artifacts),
        provenance=_build_provenance(context),
    )


def _try_encode_numpy_artifact(
    value: object, context: CapabilityExecutionContext
) -> OrganelleResult | None:
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is a declared dependency
        return None
    if not isinstance(value, np.ndarray):
        return None
    array = cast("np.ndarray[tuple[int, ...], np.dtype[np.generic]]", value)
    run_root = managed_run_path(context.operation_id, context.run_id)
    run_root.mkdir(parents=True, exist_ok=True)
    destination = run_root / "artifact.npy"
    np.save(destination, array)
    artifact = ArtifactRef.from_path(destination, kind="artifact", format="npy")
    return OrganelleResult(
        operation_id=context.operation_id,
        scope="none",
        status="ok",
        artifacts=(artifact,),
        provenance=_build_provenance(context),
    )


def encode_plugin_worker_result(
    value: object,
    context: CapabilityExecutionContext,
    staging_root: Path,
) -> OrganelleResult:
    """Turn one plugin protocol response into a managed canonical result."""

    if not isinstance(value, Mapping):
        raise _codec_error("plugin worker result must be a JSON object")
    result = cast("Mapping[str, object]", value)
    summary = result.get("summary")
    outputs = result.get("outputs")
    log_paths = result.get("log_paths", [])
    if not isinstance(summary, Mapping):
        raise _codec_error("plugin worker summary must be finite JSON")
    if not isinstance(outputs, Mapping):
        raise _codec_error("plugin worker outputs must be an object")
    if not isinstance(log_paths, Sequence) or isinstance(log_paths, (str, bytes)):
        raise _codec_error("plugin worker log paths must be an array")
    summary_mapping = cast("Mapping[str, object]", summary)
    output_mapping = cast("Mapping[str, object]", outputs)
    if not _is_finite_json(summary_mapping):
        raise _codec_error("plugin worker summary must be finite JSON")
    if PLUGIN_OPTIMIZATION_SCORE_METRIC in summary_mapping:
        raise _codec_error(
            f"plugin summary key {PLUGIN_OPTIMIZATION_SCORE_METRIC!r} is reserved"
        )
    score = result.get("score")
    if score is not None and (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(score)
    ):
        raise _codec_error("plugin worker score must be a finite JSON number")

    root = staging_root.resolve(strict=True)
    artifacts: list[ArtifactRef] = []
    for name, relative_text in output_mapping.items():
        if not isinstance(relative_text, str):
            raise _codec_error("plugin worker outputs must map names to relative paths")
        relative = Path(relative_text)
        if relative.is_absolute():
            raise _codec_error(f"plugin output {name!r} must be staging-relative")
        try:
            produced = (root / relative).resolve(strict=True)
            produced.relative_to(root)
        except (OSError, ValueError) as error:
            raise _codec_error(f"plugin output {name!r} escapes the staging tree") from error
        paths = (
            tuple(sorted(path for path in produced.rglob("*") if path.is_file()))
            if produced.is_dir()
            else (produced,)
        )
        for path in paths:
            artifact = ArtifactRef.from_path(
                path,
                kind=name,
                format=path.suffix.lstrip(".") or "bin",
            )
            artifacts.append(artifact.model_copy(update={"uri": path.relative_to(root).as_posix()}))
    for relative_text in cast("Sequence[object]", log_paths):
        if not isinstance(relative_text, str):
            raise _codec_error("plugin worker log paths must be strings")
        relative = Path(relative_text)
        if relative.is_absolute():
            raise _codec_error("plugin worker log paths must be staging-relative")
        try:
            log_path = (root / relative).resolve(strict=True)
            log_path.relative_to(root)
        except (OSError, ValueError) as error:
            raise _codec_error("plugin worker log path escapes the staging tree") from error
        if not log_path.is_file():
            raise _codec_error("plugin worker log path does not exist")
        artifact = ArtifactRef.from_path(
            log_path,
            kind="log",
            format=log_path.suffix.lstrip(".") or "txt",
        )
        artifacts.append(artifact.model_copy(update={"uri": log_path.relative_to(root).as_posix()}))
    metrics: dict[str, object] = dict(summary_mapping)
    if score is not None:
        # The declared score is published under the reserved key without
        # mutating the plugin's own summary.
        metrics[PLUGIN_OPTIMIZATION_SCORE_METRIC] = float(score)
    return OrganelleResult(
        operation_id=context.operation_id,
        scope="none",
        status="ok",
        metrics=FrozenMap(metrics),
        artifacts=tuple(artifacts),
        provenance=_build_provenance(context),
    )


def encode_plugin_worker_failure(
    error: OrganelleExecutionError,
    context: CapabilityExecutionContext,
    staging_root: Path,
) -> OrganelleResult:
    """Preserve files produced before a plugin exception as failed-run evidence."""

    root = staging_root.resolve(strict=True)
    artifacts: list[ArtifactRef] = []
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        relative = path.relative_to(root)
        artifact = ArtifactRef.from_path(
            path,
            kind="log" if relative == Path("plugin.log") else "partial_output",
            format=path.suffix.lstrip(".") or "bin",
        )
        artifacts.append(artifact.model_copy(update={"uri": relative.as_posix()}))
    return OrganelleResult(
        operation_id=context.operation_id,
        scope="none",
        status="failed",
        artifacts=tuple(artifacts),
        provenance=_build_provenance(context),
        errors=(
            ErrorDetail.model_validate(
                {
                    "code": error.code,
                    "message": error.message,
                    "details": error.details,
                    "retryable": error.retryable,
                    "suggested_action": error.suggested_action,
                }
            ),
        ),
    )


__all__ = [
    "CapabilityExecutionContext",
    "encode_capability_result",
    "encode_plugin_worker_failure",
    "encode_plugin_worker_result",
    "normalize_worker_core_object",
    "normalize_worker_result",
]
