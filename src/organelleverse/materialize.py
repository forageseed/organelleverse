"""``materialize_result`` builders: what ``organelleverse.write`` calls for a suite.

``organelleverse.write`` publishes a result through ``<suite>/writer.py:materialize_result``.
Two builders cover the suites:

* :func:`make_materializer` maps each operation to the suite's own ``write_*`` function.
* ``generic=True`` writes any result of the suite as plain tables: the scalar metrics, each
  metric that is a list of records, the findings, and a copy of every artifact file.

Both return the canonical ``<suite>.write`` result with the written files as artifacts, and
save the source result as ``<name>.result.json`` beside them so ``ov.read`` can load it again.
The manifest names artifacts by their path relative to its own directory when they sit
inside it, so a moved directory stays readable.
"""

from __future__ import annotations

import csv
import json
import shutil
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .core.artifacts import ArtifactRef
from .core.errors import OrganelleInputError
from .core.frozen import thaw_json
from .core.result import OrganelleResult
from .core.serialization import save_contract

_KINDS = {".tsv": "table", ".csv": "table", ".json": "json", ".svg": "figure", ".png": "figure"}
_MEDIA_TYPES = {
    ".tsv": "text/tab-separated-values",
    ".csv": "text/csv",
    ".json": "application/json",
    ".svg": "image/svg+xml",
    ".png": "image/png",
}

Writer = Callable[[OrganelleResult, Path], "Path | Mapping[str, Path]"]


def _artifact(path: Path) -> ArtifactRef:
    suffix = path.suffix.lower()
    return ArtifactRef.from_path(
        path,
        kind=_KINDS.get(suffix, "file"),
        format=suffix.lstrip(".") or "text",
        media_type=_MEDIA_TYPES.get(suffix, "text/plain"),
    )


def _cell(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return "" if value is None else str(value)


def _write_table(path: Path, header: list[str], rows: list[list[Any]]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows([[_cell(value) for value in row] for row in rows])
    return path


def _is_records(value: Any) -> bool:
    return bool(value) and isinstance(value, list) and all(isinstance(v, dict) for v in value)


def _publish_tables(result: OrganelleResult, output: Path) -> tuple[list[Path], dict[str, str]]:
    """Write ``result`` as tables in the directory ``output``.

    Returns the written files, and the copied artifact files by their original ``uri``.
    """
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    name = result.operation_id.split(".", 1)[-1]
    metrics = thaw_json(result.metrics)
    assert isinstance(metrics, dict)
    written: list[Path] = []
    scalars = [[key, value] for key, value in metrics.items() if not _is_records(value)]
    written.append(_write_table(out / f"{name}.metrics.tsv", ["metric", "value"], scalars))
    for key, value in metrics.items():
        if _is_records(value):
            columns = list(dict.fromkeys(column for record in value for column in record))
            rows = [[record.get(column) for column in columns] for record in value]
            written.append(_write_table(out / f"{name}.{key}.tsv", columns, rows))
    if result.findings:
        rows = [[f.code, f.metric, f.value, f.unit, f.confidence] for f in result.findings]
        header = ["code", "metric", "value", "unit", "confidence"]
        written.append(_write_table(out / f"{name}.findings.tsv", header, rows))
    copies: dict[str, str] = {}
    for artifact in result.artifacts:
        source = Path(artifact.uri)
        if source.is_file():
            target = out / source.name
            if source.resolve() != target.resolve():
                shutil.copy2(source, target)
            copies[artifact.uri] = target.name
            written.append(target)
    return written, copies


def make_materializer(
    suite: str,
    writers: Callable[[], Mapping[str, Writer]] | None = None,
    *,
    generic: bool = False,
) -> Callable[[OrganelleResult, str | Path], OrganelleResult]:
    """``materialize_result`` for ``suite``.

    ``writers`` maps an operation id to its ``write_*`` function; it is a function so the
    suite's heavy modules are imported when a result is written, not when ``ov.write`` looks
    for the suite. ``generic=True`` writes every result of the suite as plain tables instead.
    """
    if (writers is None) == (not generic):
        raise ValueError("pass either writers or generic=True")

    def materialize_result(result: OrganelleResult, output: str | Path) -> OrganelleResult:
        if result.status == "failed":
            raise OrganelleInputError(
                code="output.failed_result",
                message=f"{suite}.write requires a non-failed {suite} result",
                details={"operation_id": result.operation_id},
            )
        copies: dict[str, str] = {}
        if generic:
            paths, copies = _publish_tables(result, Path(output))
        else:
            assert writers is not None
            table = writers()
            writer = table.get(result.operation_id)
            if writer is None:
                raise OrganelleInputError(
                    code="output.unsupported_operation",
                    message=f"{suite}.write accepts results of {', '.join(sorted(table))}",
                    details={"operation_id": result.operation_id},
                )
            produced = writer(result, Path(output))
            paths = (
                list(produced.values()) if isinstance(produced, Mapping) else [Path(produced)]
            )
        # The result itself, so that ``ov.read`` can load it again. Artifacts that were
        # copied beside it are named relative to this directory.
        stem = result.operation_id.split(".", 1)[-1] if generic else Path(paths[0]).stem
        manifest = Path(paths[0]).with_name(f"{stem}.result.json")
        portable = tuple(
            a.model_copy(update={"uri": copies[a.uri]}) if a.uri in copies else a
            for a in result.artifacts
        )
        save_contract(result.model_copy(update={"artifacts": portable}), manifest)
        artifacts = (*(_artifact(Path(path)) for path in paths), _artifact(manifest))
        names = ", ".join(Path(path).name for path in paths)
        return OrganelleResult(
            operation_id=f"{suite}.write",
            operation_version="1.0",
            scope=result.scope,
            status=result.status,
            summary_text=f"Materialized {result.operation_id} to {names}.",
            metrics={
                "file_count": len(artifacts),
                "source_result_object_id": result.object_id,
                "source_result_status": result.status,
            },
            flags=tuple(dict.fromkeys((f"{suite}_materialized", *result.flags))),
            artifacts=artifacts,
            provenance=result.provenance,
        )

    return materialize_result
