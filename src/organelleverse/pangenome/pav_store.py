"""Columnar node PAV access with bounded rows/columns and streaming analysis.

Published tables are immutable artifacts. Ordinary page reads check size,
Arrow schema, and counts; they are not fresh whole-file digest audits. Call
verify_pav_store at publication/resume/admission to verify captured ArtifactRefs.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import closing
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..core.artifacts import ArtifactRef
from ._pav_write import MAX_BATCH_CELLS, PAVMetadata, Unit, _schema, write_node_pav

__all__ = [
    "PAVMetadata",
    "iter_pav_batches",
    "iter_pav_rows",
    "load_pav_metadata",
    "pav_summary",
    "read_pav_page",
    "verify_pav_store",
    "write_node_pav",
]


def load_pav_metadata(directory: str | Path) -> PAVMetadata:
    metadata = PAVMetadata.model_validate_json((Path(directory) / "pav-metadata.json").read_text())
    if (
        not metadata.paths
        or not metadata.samples
        or len(metadata.paths) != len(metadata.path_samples)
    ):
        raise ValueError("PAV store requires consistent nonempty path/sample labels")
    if len(set(metadata.paths)) != len(metadata.paths) or len(set(metadata.samples)) != len(
        metadata.samples
    ):
        raise ValueError("PAV store column labels must be unique")
    if (
        set(metadata.path_samples) != set(metadata.samples)
        or set(metadata.tables) != {"path", "sample"}
        or set(metadata.tsvs) != {"path", "sample"}
    ):
        raise ValueError("PAV store sample grouping or artifact units are inconsistent")
    return metadata


def _artifact_path(directory: str | Path, artifact: ArtifactRef) -> Path:
    root = Path(directory).resolve()
    path = artifact.resolve(root).resolve()
    if not path.is_relative_to(root):
        raise ValueError("PAV table reference is outside its store")
    if path.stat().st_size != artifact.size_bytes:
        raise ValueError("PAV table size changed after publication")
    return path


def verify_pav_store(directory: str | Path) -> PAVMetadata:
    """Perform full content verification of every captured Parquet and TSV file."""
    metadata = load_pav_metadata(directory)
    for artifact in [*metadata.tables.values(), *metadata.tsvs.values()]:
        path = _artifact_path(directory, artifact)
        actual = ArtifactRef.from_path(path, kind=artifact.kind, format=artifact.format)
        if (actual.sha256, actual.size_bytes) != (artifact.sha256, artifact.size_bytes):
            raise ValueError(f"PAV artifact changed after capture: {artifact.uri}")
    for unit in ("sample", "path"):
        with closing(_open(directory, metadata, unit)):
            pass
    return metadata


def _labels(
    metadata: PAVMetadata, unit: Unit, columns: Sequence[str] | None
) -> tuple[list[str], list[int]]:
    if unit not in {"sample", "path"}:
        raise ValueError("PAV unit must be sample or path")
    labels = metadata.samples if unit == "sample" else metadata.paths
    selected = list(labels if columns is None else columns)
    if len(set(selected)) != len(selected):
        raise ValueError("Selected PAV columns must be unique")
    indices = {name: index for index, name in enumerate(labels)}
    missing = set(selected) - indices.keys()
    if missing:
        raise ValueError(f"Unknown {unit} columns: {sorted(missing)}")
    return selected, [indices[name] for name in selected]


def _open(directory: str | Path, metadata: PAVMetadata, unit: Unit) -> pq.ParquetFile:
    labels, _ = _labels(metadata, unit, None)
    reference = metadata.tables[unit]
    if reference.format != "parquet" or reference.media_type != "application/vnd.apache.parquet":
        raise ValueError("PAV table must be declared Apache Parquet")
    parquet = pq.ParquetFile(_artifact_path(directory, reference))
    if parquet.metadata.num_rows != metadata.node_count or not parquet.schema_arrow.equals(
        _schema(len(labels)), check_metadata=False
    ):
        parquet.close()
        raise ValueError("PAV table schema or row count disagrees with captured metadata")
    maximum = max(1, MAX_BATCH_CELLS // len(labels))
    if any(parquet.metadata.row_group(i).num_rows > maximum for i in range(parquet.num_row_groups)):
        parquet.close()
        raise ValueError("PAV row group exceeds the store's bounded-cell encoding")
    return parquet


def _values(array: pa.FixedSizeListArray) -> np.ndarray:
    width = array.type.list_size
    if array.null_count or array.values.null_count:
        raise ValueError("PAV presence cannot contain unknown/null values")
    flat = array.values.slice(array.offset * width, len(array) * width)
    return flat.to_numpy(zero_copy_only=False).reshape(len(array), width)


def iter_pav_batches(
    directory: str | Path,
    *,
    unit: Unit = "sample",
    columns: Sequence[str] | None = None,
    batch_size: int = 4096,
) -> Iterator[np.ndarray]:
    """Yield node x column boolean arrays in source order, never the whole matrix.

    Batch rows are at most batch_size and are also bounded by the full encoded
    column width. Selected columns follow the caller's explicit order.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    metadata = load_pav_metadata(directory)
    _, indices = _labels(metadata, unit, columns)
    width = len(metadata.samples if unit == "sample" else metadata.paths)
    batch_size = min(batch_size, max(1, MAX_BATCH_CELLS // width))
    with closing(_open(directory, metadata, unit)) as parquet:
        for batch in parquet.iter_batches(batch_size=batch_size, columns=["presence"]):
            yield _values(batch.column(0))[:, indices]


def iter_pav_rows(
    directory: str | Path, *, unit: Unit = "sample", batch_size: int = 4096
) -> Iterator[dict[str, Any]]:
    """Stream scalar node/count/frequency/category rows without reading presence."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    metadata = load_pav_metadata(directory)
    with closing(_open(directory, metadata, unit)) as parquet:
        for batch in parquet.iter_batches(
            batch_size=batch_size,
            columns=["row_index", "node_id", "count", "frequency", "category"],
        ):
            yield from batch.to_pylist()


def _row_selection(
    parquet: pq.ParquetFile, offset: int, limit: int, node_ids: Sequence[str] | set[str] | None
) -> tuple[list[tuple[int, list[int]]], int]:
    selected, position = [], 0
    wanted = None if node_ids is None else set(node_ids)
    found = set()
    for group in range(parquet.num_row_groups):
        count = parquet.metadata.row_group(group).num_rows
        if wanted is None:
            left, right = max(0, offset - position), min(count, offset + limit - position)
            matches = list(range(left, max(left, right)))
            position += count
        else:
            ids = parquet.read_row_group(group, columns=["node_id"]).column(0).to_pylist()
            local = [index for index, name in enumerate(ids) if name in wanted]
            found.update(name for name in ids if name in wanted)
            left, right = max(0, offset - position), max(0, offset + limit - position)
            matches = local[left:right]
            position += len(local)
        if matches:
            selected.append((group, matches))
    if wanted is not None and wanted - found:
        raise ValueError(f"Unknown graph node IDs in PAV: {sorted(wanted - found)}")
    return selected, position


def read_pav_page(
    directory: str | Path,
    *,
    unit: Unit = "sample",
    offset: int = 0,
    limit: int = 100,
    column_offset: int = 0,
    column_limit: int = 50,
    columns: Sequence[str] | None = None,
    node_ids: Sequence[str] | set[str] | None = None,
) -> dict[str, Any]:
    """Read at most 500 x 500 cells, seeking only matching Parquet row groups.

    Node and column selections occur before pagination. Frequency/category/count
    retain the full original unit denominator, including unselected columns.
    """
    for name, value in (("offset", offset), ("column_offset", column_offset)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    for name, value in (("limit", limit), ("column_limit", column_limit)):
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 500:
            raise ValueError(f"{name} must be in [1,500]")
    metadata = load_pav_metadata(directory)
    labels, indices = _labels(metadata, unit, columns)
    all_width = len(metadata.samples if unit == "sample" else metadata.paths)
    chosen = indices[column_offset : column_offset + column_limit]
    displayed = labels[column_offset : column_offset + column_limit]
    rows = []
    with closing(_open(directory, metadata, unit)) as parquet:
        groups, total = _row_selection(parquet, offset, limit, node_ids)
        for group, matches in groups:
            table = parquet.read_row_group(group)
            batch = table.combine_chunks().to_batches()[0]
            values = _values(batch.column(batch.schema.get_field_index("presence")))
            scalars = table.select(["node_id", "count", "frequency", "category"]).to_pylist()
            for index in matches:
                rows.append(
                    {**scalars[index], "presence": values[index, chosen].astype(np.uint8).tolist()}
                )
    return {
        "samples": displayed,
        "columns": displayed,
        "unit": unit,
        "rows": rows,
        "total_rows": total,
        "total_graph_rows": metadata.node_count,
        "offset": offset,
        "limit": limit,
        "total_columns": len(labels),
        "total_graph_columns": all_width,
        "column_offset": column_offset,
        "column_limit": column_limit,
        "frequency_denominator": all_width,
    }


def pav_summary(directory: str | Path) -> dict[str, Any]:
    """Compact exact counts/frequency distributions and ArtifactRefs for reports."""
    metadata = load_pav_metadata(directory)
    return {
        "node_count": metadata.node_count,
        "path_count": len(metadata.paths),
        "sample_count": len(metadata.samples),
        "cloud_threshold": metadata.cloud_threshold,
        "core_threshold": metadata.core_threshold,
        "sample_class_counts": metadata.sample_class_counts,
        "path_class_counts": metadata.path_class_counts,
        "sample_frequency_counts": metadata.sample_frequency_counts,
        "path_frequency_counts": metadata.path_frequency_counts,
        "artifacts": {
            name: ref.model_dump(mode="json")
            for name, ref in {
                **{f"{u}_parquet": a for u, a in metadata.tables.items()},
                **{f"{u}_tsv": a for u, a in metadata.tsvs.items()},
            }.items()
        },
    }
