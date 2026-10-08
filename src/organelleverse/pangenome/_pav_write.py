"""Bounded Arrow/Parquet node PAV writer and typed store metadata."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Literal

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..core.artifacts import ArtifactRef
from .graph import _classify, load_gfa

Unit = Literal["sample", "path"]
MAX_BATCH_CELLS = 1_000_000


class PAVMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["organelleverse.pangenome.pav-store.v1"] = (
        "organelleverse.pangenome.pav-store.v1"
    )
    source_graph: ArtifactRef
    node_count: int = Field(ge=0)
    paths: tuple[str, ...]
    path_samples: tuple[str, ...]
    samples: tuple[str, ...]
    cloud_threshold: float = Field(ge=0, lt=1)
    core_threshold: float = Field(default=1.0, gt=0, le=1)
    tables: dict[Unit, ArtifactRef]
    tsvs: dict[Unit, ArtifactRef]
    sample_class_counts: dict[str, int]
    path_class_counts: dict[str, int]
    sample_frequency_counts: dict[int, int]
    path_frequency_counts: dict[int, int]
    sample_aggregation: str = "OR across all paths/intervals assigned to a biological sample"

    @model_validator(mode="after")
    def check_thresholds(self):
        if self.cloud_threshold >= self.core_threshold:
            raise ValueError("cloud_threshold must be smaller than core_threshold")
        return self

    @field_validator("source_graph", mode="before")
    @classmethod
    def parse_graph_reference(cls, value):
        return ArtifactRef.model_validate(value) if isinstance(value, dict) else value

    @field_validator("tables", "tsvs", mode="before")
    @classmethod
    def parse_table_references(cls, values):
        if isinstance(values, dict):
            return {unit: ArtifactRef.model_validate(value) for unit, value in values.items()}
        return values


def _schema(width: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("row_index", pa.uint64(), nullable=False),
            pa.field("node_id", pa.string(), nullable=False),
            pa.field("count", pa.uint64(), nullable=False),
            pa.field("frequency", pa.float64(), nullable=False),
            pa.field("category", pa.string(), nullable=False),
            pa.field("presence", pa.list_(pa.bool_(), width), nullable=False),
        ]
    )


def write_node_pav(
    gfa_path: str | Path,
    output_dir: str | Path,
    *,
    cloud_threshold: float = 0.05,
    core_threshold: float = 1.0,
    row_group_size: int = 4096,
) -> PAVMetadata:
    """Write exact node x path and node x sample PAV without a dense Python matrix.

    Presence is accumulated as packed integer bitsets, then expanded only for a
    bounded row group (at most MAX_BATCH_CELLS booleans, except one wider row).
    Parquet stores bit-packed Arrow booleans; TSV is streamed row by row. Memory
    beyond the parsed graph is O(N(P+S)/8) packed memberships plus one batch.
    Files are worker staging outputs; publication is the caller's managed Result
    boundary. The metadata contains references, not matrix values.
    """
    if not 0 <= cloud_threshold < core_threshold <= 1:
        raise ValueError("thresholds must satisfy 0 <= cloud_threshold < core_threshold <= 1")
    if (
        isinstance(row_group_size, bool)
        or not isinstance(row_group_size, int)
        or row_group_size < 1
    ):
        raise ValueError("row_group_size must be a positive integer")
    graph = load_gfa(gfa_path)
    if not graph.paths:
        raise ValueError("node PAV requires P or W records; no paths are present")
    nodes = list(graph.segments)
    paths = tuple(graph.paths)
    samples = tuple(dict.fromkeys(record.sample for record in graph.paths.values()))
    path_samples = tuple(record.sample for record in graph.paths.values())
    sample_indices = {name: index for index, name in enumerate(samples)}
    masks = {"path": dict.fromkeys(nodes, 0), "sample": dict.fromkeys(nodes, 0)}
    for index, record in enumerate(graph.paths.values()):
        path_bit, sample_bit = 1 << index, 1 << sample_indices[record.sample]
        for node, _ in record.steps:
            masks["path"][node] |= path_bit
            masks["sample"][node] |= sample_bit
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    refs, tsv_refs, class_counts, frequency_counts = {}, {}, {}, {}
    for unit, labels in (("path", paths), ("sample", samples)):
        width = len(labels)
        chunk_rows = min(row_group_size, max(1, MAX_BATCH_CELLS // width))
        parquet_path = directory / f"node_{unit}_pav.parquet"
        tsv_path = directory / f"node_{unit}_pav.tsv"
        schema = _schema(width)
        class_counts[unit], frequency_counts[unit] = Counter(), Counter()
        with (
            pq.ParquetWriter(
                parquet_path, schema, compression="zstd", use_dictionary=["category"]
            ) as writer,
            tsv_path.open("w", newline="", encoding="utf-8") as stream,
        ):
            tsv = csv.writer(stream, delimiter="\t", lineterminator="\n")
            tsv.writerow(["node_id", "count", "frequency", "category", *labels])
            for offset in range(0, len(nodes), chunk_rows):
                batch_nodes = nodes[offset : offset + chunk_rows]
                packed = [masks[unit][node] for node in batch_nodes]
                stride = (width + 7) // 8
                values = np.unpackbits(
                    np.frombuffer(
                        b"".join(mask.to_bytes(stride, "little") for mask in packed), dtype=np.uint8
                    ).reshape(len(packed), stride),
                    axis=1,
                    count=width,
                    bitorder="little",
                ).astype(np.bool_)
                counts = [mask.bit_count() for mask in packed]
                frequencies = [count / width for count in counts]
                categories = [
                    _classify(count, width, cloud_threshold, core_threshold) for count in counts
                ]
                class_counts[unit].update(categories)
                frequency_counts[unit].update(counts)
                arrays = [
                    pa.array(range(offset, offset + len(packed)), type=pa.uint64()),
                    pa.array(batch_nodes),
                    pa.array(counts, type=pa.uint64()),
                    pa.array(frequencies),
                    pa.array(categories),
                    pa.FixedSizeListArray.from_arrays(pa.array(values.ravel()), width),
                ]
                writer.write_batch(
                    pa.RecordBatch.from_arrays(arrays, schema=schema), row_group_size=chunk_rows
                )
                for node, count, frequency, category, row in zip(
                    batch_nodes, counts, frequencies, categories, values, strict=True
                ):
                    tsv.writerow([node, count, frequency, category, *row.astype(np.uint8).tolist()])
        refs[unit] = ArtifactRef.from_path(
            parquet_path,
            kind=f"node_{unit}_pav",
            format="parquet",
            media_type="application/vnd.apache.parquet",
        ).model_copy(update={"uri": parquet_path.name})
        tsv_refs[unit] = ArtifactRef.from_path(
            tsv_path, kind=f"node_{unit}_pav", format="tsv", media_type="text/tab-separated-values"
        ).model_copy(update={"uri": tsv_path.name})
    metadata = PAVMetadata(
        source_graph=ArtifactRef.from_path(gfa_path, kind="pangenome_graph", format="gfa"),
        node_count=len(nodes),
        paths=paths,
        path_samples=path_samples,
        samples=samples,
        cloud_threshold=cloud_threshold,
        core_threshold=core_threshold,
        tables=refs,
        tsvs=tsv_refs,
        sample_class_counts=dict(class_counts["sample"]),
        path_class_counts=dict(class_counts["path"]),
        sample_frequency_counts=dict(frequency_counts["sample"]),
        path_frequency_counts=dict(frequency_counts["path"]),
    )
    (directory / "pav-metadata.json").write_text(
        metadata.model_dump_json(indent=2, round_trip=True) + "\n"
    )
    return metadata
