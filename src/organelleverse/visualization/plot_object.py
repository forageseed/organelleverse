"""Small printable/savable plot result objects.

The visualization suite speaks the canonical v1 contract: every public plot
call returns an :class:`~organelleverse.core.result.OrganelleResult` (an
:class:`OrganellePlot` for the deferred-render plots), and every file a plot
materializes is declared as an :class:`~organelleverse.core.artifacts.ArtifactRef`
with role (``kind``), path (``uri``), content hash and media type.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
import uuid
from collections.abc import Callable, Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from pydantic import PrivateAttr

from ..core.artifacts import ArtifactRef
from ..core.frozen import thaw_json
from ..core.provenance import ResultProvenance
from ..core.result import ErrorDetail, Finding, OrganelleResult, ResultScope

PlotRenderer = Callable[[str | Path], Path]

SUITE = "visualization"
OPERATION_VERSION = "1.0"

#: Legacy organelle labels accepted at the public surface mapped onto the
#: canonical :data:`~organelleverse.core.result.ResultScope` vocabulary.
_SCOPE_BY_ORGANELLE: dict[str, ResultScope] = {
    "mito": "mitochondrion",
    "mitochondrion": "mitochondrion",
    "mitochondrial": "mitochondrion",
    "chloro": "plastid",
    "chloroplast": "plastid",
    "plastid": "plastid",
    "plastome": "plastid",
    "nuclear": "cytonuclear",
    "cytonuclear": "cytonuclear",
    "mixed": "mixed",
    "none": "none",
    "": "none",
}

_MEDIA_TYPES = {
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".pdf": "application/pdf",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".eps": "application/postscript",
    ".ps": "application/postscript",
    ".html": "text/html",
    ".tsv": "text/tab-separated-values",
    ".csv": "text/csv",
    ".json": "application/json",
}
_IMAGE_SUFFIXES = frozenset(
    {".png", ".svg", ".pdf", ".tif", ".tiff", ".jpg", ".jpeg", ".eps", ".ps"}
)


def operation_id_for(op: str, suite: str = SUITE) -> str:
    """Return the canonical ``suite.operation`` identifier for a plot call."""
    return f"{suite}.{op}"


def resolve_scope(organelle: str | None) -> ResultScope:
    """Map an organelle label used by the plot API onto a canonical result scope."""
    key = str(organelle or "").strip().lower()
    return _SCOPE_BY_ORGANELLE.get(key, "none")


def jsonable(value: Any) -> Any:
    """Coerce arbitrary plot metrics into a JSON-compatible structure.

    Canonical ``metrics`` are recursively frozen JSON; the plot API accepts
    plain Python analysis data (``Path``, numpy scalars/arrays, sets, ...), so
    the boundary coerces instead of rejecting.
    """
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [jsonable(item) for item in value]
    item_method = getattr(value, "item", None)
    if callable(item_method) and getattr(value, "shape", None) == ():
        return jsonable(item_method())
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return jsonable(tolist())
    return str(value)


def _finding_value(value: Any) -> int | float | str | bool | None:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def findings_from(
    key_findings: Sequence[Mapping[str, Any]],
    *,
    suite: str = SUITE,
) -> tuple[Finding, ...]:
    """Convert ``{"metric": ..., "value": ...}`` rows into canonical findings."""
    findings: list[Finding] = []
    for row in key_findings:
        metric = str(row.get("metric", ""))
        default_code = f"{suite}.{metric}" if metric else f"{suite}.finding"
        code = str(row.get("code") or default_code)
        findings.append(
            Finding(
                code=code,
                metric=metric,
                value=_finding_value(row.get("value")),
                unit=str(row.get("unit", "")),
            )
        )
    return tuple(findings)


def parameters_hash(payload: object) -> str:
    """Return a deterministic SHA-256 over the normalized plot parameters."""
    encoded = json.dumps(
        jsonable(payload),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def package_version() -> str:
    """Return the installed OrganelleVerse version, or the source default."""
    try:
        return version("organelleverse")
    except PackageNotFoundError:
        return "0.0.1"


def build_provenance(
    op: str,
    *,
    suite: str = SUITE,
    method: str,
    payload: object,
    software_versions: Mapping[str, Any] | None = None,
) -> ResultProvenance:
    """Build the canonical provenance record for one visualization operation."""
    backend = method or op
    return ResultProvenance(
        operation_id=operation_id_for(op, suite),
        operation_version=OPERATION_VERSION,
        package_version=package_version(),
        git_commit=os.environ.get("ORG_VERSE_GIT_COMMIT", ""),
        parameters_hash=parameters_hash(payload),
        actual_backend=backend,
        attempted_backends=(backend,),
        software_versions=jsonable(dict(software_versions or {})),
    )


def artifact_role(path: Path) -> str:
    """Return the canonical artifact role for one materialized plot output."""
    return "figure" if path.suffix.lower() in _IMAGE_SUFFIXES else "figure_data"


def media_type_for(path: Path) -> str:
    """Return the IANA media type declared for one materialized plot output."""
    return _MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream")


def figure_artifact(path: str | Path) -> ArtifactRef:
    """Declare one written plot file as a content-addressed artifact."""
    candidate = Path(path)
    return ArtifactRef.from_path(
        candidate,
        kind=artifact_role(candidate),
        format=candidate.suffix.lower().lstrip(".") or "bin",
        media_type=media_type_for(candidate),
    )


def figure_artifacts(paths: Sequence[str | Path]) -> tuple[ArtifactRef, ...]:
    """Declare every written plot file, primary output first."""
    return tuple(figure_artifact(path) for path in paths if Path(path).is_file())


def plot_result(
    op: str,
    renderer: PlotRenderer,
    *,
    suite: str = SUITE,
    organelle: str = "mito",
    metrics: Mapping[str, Any] | None = None,
    key_findings: tuple[Mapping[str, Any], ...] = (),
    flags: tuple[str, ...] = (),
    summary: str | None = None,
    method: str | None = None,
    software_versions: Mapping[str, Any] | None = None,
) -> OrganellePlot:
    """Build a compute-only :class:`OrganellePlot` that defers rendering.

    The public visualization call captures its normalized scientific/style
    inputs in ``renderer`` (a closure over the private path-taking renderer)
    and returns the plot without writing any file. ``ov.write(plot, output)``
    or ``plot.save(output)`` later invokes ``renderer`` to materialize it; the
    written files are declared as artifacts only once they exist.
    """
    observed: dict[str, Any] = {"plot_kind": op}
    if metrics:
        observed.update(metrics)
    frozen_metrics = jsonable(observed)
    resolved_method = method or op
    plot = OrganellePlot(
        operation_id=operation_id_for(op, suite),
        operation_version=OPERATION_VERSION,
        scope=resolve_scope(organelle),
        status="ok",
        summary_text=(
            summary
            if summary is not None
            else f"{op} prepared; render with ov.write(plot, output)."
        ),
        metrics=frozen_metrics,
        findings=findings_from(key_findings, suite=suite),
        flags=tuple(flags),
        artifacts=(),
        provenance=build_provenance(
            op,
            suite=suite,
            method=resolved_method,
            payload={"op": op, "method": resolved_method, "metrics": frozen_metrics},
            software_versions=software_versions,
        ),
    )
    plot._renderer = renderer
    return plot


def data_plot_result(
    op: str,
    *,
    suite: str = SUITE,
    organelle: str = "mito",
    metrics: Mapping[str, Any] | None = None,
    key_findings: tuple[Mapping[str, Any], ...] = (),
    flags: tuple[str, ...] = (),
    summary: str = "",
    method: str | None = None,
) -> OrganelleResult:
    """Build a compute-only canonical Result for data-shaped plot objects.

    Used by the prepare/render split (``ideogram`` → ``write_ideogram``) where
    the figure size is chosen at write time, so no renderer can be captured.
    """
    observed: dict[str, Any] = {"plot_kind": op}
    if metrics:
        observed.update(metrics)
    frozen_metrics = jsonable(observed)
    resolved_method = method or op
    return OrganelleResult(
        operation_id=operation_id_for(op, suite),
        operation_version=OPERATION_VERSION,
        scope=resolve_scope(organelle),
        status="ok",
        summary_text=summary,
        metrics=frozen_metrics,
        findings=findings_from(key_findings, suite=suite),
        flags=tuple(flags),
        artifacts=(),
        provenance=build_provenance(
            op,
            suite=suite,
            method=resolved_method,
            payload={"op": op, "method": resolved_method, "metrics": frozen_metrics},
        ),
    )


def failed_plot_result(
    op: str,
    *,
    suite: str = SUITE,
    organelle: str = "mito",
    code: str,
    message: str,
    method: str,
    retryable: bool = False,
    suggested_action: Mapping[str, Any] | None = None,
    flags: tuple[str, ...] = (),
    software_versions: Mapping[str, Any] | None = None,
) -> OrganelleResult:
    """Build a canonical failed Result for an unavailable rendering backend."""
    return OrganelleResult(
        operation_id=operation_id_for(op, suite),
        operation_version=OPERATION_VERSION,
        scope=resolve_scope(organelle),
        status="failed",
        summary_text=message,
        metrics={"plot_kind": op, "method": method},
        flags=tuple(flags),
        errors=(
            ErrorDetail(
                code=code,
                message=message,
                retryable=retryable,
                suggested_action=dict(suggested_action or {}),
            ),
        ),
        provenance=build_provenance(
            op,
            suite=suite,
            method=method,
            payload={"op": op, "method": method, "error": code},
            software_versions=software_versions,
        ),
    )


class OrganellePlot(OrganelleResult):
    """An ``OrganelleResult`` that can be displayed and saved like a plot."""

    _renderer: PlotRenderer | None = PrivateAttr(default=None)
    _render_state: dict[str, Any] = PrivateAttr(default_factory=dict)

    @property
    def renderer(self) -> PlotRenderer | None:
        """The deferred renderer captured by the compute-only plot call."""
        return self._renderer

    @property
    def installed_paths(self) -> tuple[Path, ...]:
        """Every file materialized by the most recent :meth:`save` call."""
        return tuple(self._render_state.get("installed", ()))

    def save(self, output: str | Path | None = None) -> Path:
        """Render the plot to ``output`` and return the written path."""
        if output is None:
            if self.artifacts:
                return Path(self.artifacts[0].uri)
            return self._ensure_png_preview()
        if self._renderer is None:
            raise ValueError("this plot does not have a renderer attached")
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary_root = Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name}.",
                suffix=".tmp",
                dir=destination.parent,
            )
        )
        temporary_output = temporary_root / destination.name
        backup_root = temporary_root / ".backups"
        installed: list[Path] = []
        backups: list[tuple[Path, Path]] = []
        try:
            rendered = Path(self._renderer(temporary_output))
            try:
                relative_rendered = rendered.relative_to(temporary_root)
            except ValueError as exc:
                raise RuntimeError(
                    "plot renderer wrote outside the writer-managed temporary directory"
                ) from exc
            generated = tuple(
                child for child in temporary_root.iterdir() if child.name != ".backups"
            )
            if not generated:
                raise RuntimeError("plot renderer did not create an output")

            backup_root.mkdir()
            for source in generated:
                final = destination.parent / source.name
                if final.exists() or final.is_symlink():
                    backup = backup_root / f"{uuid.uuid4().hex}-{source.name}"
                    os.replace(final, backup)
                    backups.append((backup, final))
                os.replace(source, final)
                installed.append(final)

            path = destination.parent / relative_rendered
            if not path.exists():
                raise RuntimeError("plot renderer returned an output that was not materialized")
            shutil.rmtree(backup_root, ignore_errors=True)
        except Exception:
            for installed_path in reversed(installed):
                if installed_path.is_dir() and not installed_path.is_symlink():
                    shutil.rmtree(installed_path, ignore_errors=True)
                else:
                    installed_path.unlink(missing_ok=True)
            for backup, final in reversed(backups):
                if backup.exists() or backup.is_symlink():
                    os.replace(backup, final)
            raise
        finally:
            shutil.rmtree(temporary_root, ignore_errors=True)
        self._render_state["installed"] = _materialized_paths(path, installed)
        if path.suffix.lower() == ".png":
            self._render_state["preview_path"] = path
        return path

    def materialize(self, output: str | Path) -> OrganelleResult:
        """Render to ``output`` and return a released Result declaring artifacts.

        This is the canonical publication boundary for a plot: the returned
        Result is the same operation with every written file declared as an
        :class:`ArtifactRef` (role, path, SHA-256, media type), primary output
        first.
        """
        self.save(output)
        artifacts = figure_artifacts(self.installed_paths)
        return OrganelleResult(
            operation_id=self.operation_id,
            operation_version=self.operation_version,
            scope=self.scope,
            status=self.status,
            summary_text=self.summary_text,
            metrics=thaw_json(self.metrics),
            findings=self.findings,
            flags=self.flags,
            artifacts=artifacts,
            provenance=self.provenance,
            suggested_operations=self.suggested_operations,
        )

    def show(self) -> Path:
        """Display the plot in notebooks when possible and return the preview path."""
        path = self._ensure_png_preview()
        try:
            from IPython import get_ipython  # type: ignore
            from IPython.display import Image, display  # type: ignore

            if get_ipython() is not None:
                display(Image(filename=str(path)))
        except Exception:
            pass
        return path

    def _repr_png_(self) -> bytes:
        """Jupyter/IPython rich display hook."""
        return self._ensure_png_preview().read_bytes()

    def __str__(self) -> str:
        path = self.show()
        return (
            f"OrganellePlot(operation_id={self.operation_id!r}, "
            f"image={str(path)!r}, save=ov.write(plot, ...))"
        )

    def _ensure_png_preview(self) -> Path:
        preview = self._render_state.get("preview_path")
        if isinstance(preview, Path) and preview.exists():
            return preview
        for artifact in self.artifacts:
            candidate = Path(artifact.uri)
            if candidate.suffix.lower() == ".png" and candidate.exists():
                self._render_state["preview_path"] = candidate
                return candidate
        if self._renderer is None:
            raise ValueError("this plot does not have a renderer attached")
        with tempfile.NamedTemporaryFile(
            prefix="organelleverse_plot_",
            suffix=".png",
            delete=False,
        ) as handle:
            preview_name = handle.name
        path = self._renderer(preview_name)
        self._render_state["preview_path"] = path
        return path


def _materialized_paths(primary: Path, installed: Sequence[Path]) -> tuple[Path, ...]:
    """Return the primary output first, then every sidecar file it produced."""
    ordered: list[Path] = [primary]
    for candidate in installed:
        if candidate == primary:
            continue
        if candidate.is_dir():
            ordered.extend(sorted(child for child in candidate.rglob("*") if child.is_file()))
        else:
            ordered.append(candidate)
    return tuple(ordered)


__all__ = [
    "OPERATION_VERSION",
    "SUITE",
    "OrganellePlot",
    "PlotRenderer",
    "artifact_role",
    "build_provenance",
    "data_plot_result",
    "failed_plot_result",
    "figure_artifact",
    "figure_artifacts",
    "findings_from",
    "jsonable",
    "media_type_for",
    "operation_id_for",
    "package_version",
    "parameters_hash",
    "plot_result",
    "resolve_scope",
]
