"""Pre-v1 dataclass contracts for unregistered compatibility suites.

These objects are not the public Agent contract. Canonical code imports the
immutable Pydantic models from ``organelleverse.core``; old suites that have
not migrated must import this module explicitly so the two contracts cannot be
confused or serialized under the same schema version.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

ORGANELLE_DATA_PERSIST_VERSION = "organelleverse.data.legacy.v0"

Organelle = Literal["mito", "chloro", "plastid"]
ORGANELLE_RESULT_SCHEMA_VERSION = "organelleverse.result.legacy.v0"

__all__ = [
    "ORGANELLE_DATA_PERSIST_VERSION",
    "ORGANELLE_RESULT_SCHEMA_VERSION",
    "Organelle",
    "OrganelleData",
    "OrganelleGenome",
    "OrganelleMetadata",
    "OrganelleResult",
    "ResultProvenance",
    "organelle_result_json_schema",
]


@dataclass
class OrganelleData:
    """Project-level data container for ``ov.module.function`` workflows.

    ``OrganelleGenome`` remains the small single-genome input object. This
    container is for multi-file or multi-sample analyses such as OrthoFinder,
    GFA graph, pangenome, and editing evidence workflows.
    """

    modality: str
    X: Any = None
    obs: dict[str, Any] = field(default_factory=dict)
    var: dict[str, Any] = field(default_factory=dict)
    uns: dict[str, Any] = field(default_factory=dict)
    layers: dict[str, Any] = field(default_factory=dict)
    results: dict[str, OrganelleResult] = field(default_factory=dict)
    paths: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def copy(self) -> OrganelleData:
        """Return a deep copy of the analysis container."""
        return OrganelleData(
            modality=self.modality,
            X=deepcopy(self.X),
            obs=deepcopy(self.obs),
            var=deepcopy(self.var),
            uns=deepcopy(self.uns),
            layers=deepcopy(self.layers),
            results=deepcopy(self.results),
            paths=deepcopy(self.paths),
            metadata=deepcopy(self.metadata),
        )

    def add_result(self, key: str, result: OrganelleResult) -> OrganelleData:
        """Attach an analysis result and return ``self`` for chaining."""
        self.results[key] = result
        return self

    # -- persistence (resume / handoff) -----------------------------------

    def save(self, path: str | Path) -> Path:
        """Write the container to a human- and agent-readable JSON file.

        For cross-session resume / agent handoff. Data fields (``X`` / ``uns`` /
        ``layers`` / ...) are coerced to JSON via :func:`_jsonable` (numpy →
        list, set/tuple → list, Path → str). ``results`` are stored as their
        :meth:`OrganelleResult.summary_json` payload. Round-trips via
        :meth:`load`; ``results`` come back as plain dicts (read-only audit).
        """
        p = Path(path)
        payload = {
            "version": ORGANELLE_DATA_PERSIST_VERSION,
            "modality": self.modality,
            "X": _jsonable(self.X),
            "obs": _jsonable(self.obs),
            "var": _jsonable(self.var),
            "uns": _jsonable(self.uns),
            "layers": _jsonable(self.layers),
            "results": {k: v.summary_json() for k, v in self.results.items()},
            "paths": {k: str(v) for k, v in self.paths.items()},
            "metadata": _jsonable(self.metadata),
        }
        p.write_text(json.dumps(payload, indent=2))
        return p

    @classmethod
    def load(cls, path: str | Path) -> OrganelleData:
        """Restore a container saved by :meth:`save` (for resume / handoff).

        ``results`` are restored as plain dicts (their ``summary_json`` payload),
        not :class:`OrganelleResult` objects — sufficient for audit and to
        continue computing from the restored ``layers`` / ``uns`` data.
        """
        payload = json.loads(Path(path).read_text())
        return cls(
            modality=payload.get("modality", ""),
            X=payload.get("X"),
            obs=payload.get("obs", {}),
            var=payload.get("var", {}),
            uns=payload.get("uns", {}),
            layers=payload.get("layers", {}),
            results=payload.get("results", {}),
            paths=payload.get("paths", {}),
            metadata=payload.get("metadata", {}),
        )


def _jsonable(obj: Any) -> Any:
    """Coerce arbitrary analysis data into a JSON-serializable structure.

    Handles the common types found in ``OrganelleData`` fields: dict, list,
    tuple, set, ``Path``, numpy arrays (any object exposing ``tolist``),
    ``OrganelleResult`` (→ ``summary_json``), and primitives. Anything else
    falls back to ``str(obj)`` so a save never crashes mid-pipeline.
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, OrganelleResult):
        return obj.summary_json()
    if hasattr(obj, "tolist"):  # numpy ndarray / pandas-like
        return _jsonable(obj.tolist())
    return str(obj)


@dataclass(frozen=True)
class OrganelleMetadata:
    """Lightweight metadata attached to an :class:`OrganelleGenome`.

    All fields optional and string-typed so the object stays hashable and cheap
    to construct. Suite functions that need richer metadata (e.g. population
    group labels, sequencing depth) carry them in dedicated suite kwargs, not
    here, to keep the unified variable minimal.
    """

    species: str = ""
    accession: str = ""
    genetic_code: str = "1"  # NCBI translation table; plant mitochondria use table 1
    assembly_type: str = ""  # e.g. "chromosome", "contig", "haplotype"
    provenance: str = ""  # free-text source note (download URL, lab, ...)
    extra: tuple[tuple[str, str], ...] = ()  # sorted stable extra key/value pairs

    def as_dict(self) -> dict[str, str]:
        d = {
            "species": self.species,
            "accession": self.accession,
            "genetic_code": self.genetic_code,
            "assembly_type": self.assembly_type,
            "provenance": self.provenance,
        }
        for k, v in self.extra:
            d[k] = v
        return d


@dataclass(frozen=True)
class OrganelleGenome:
    """Input-state unified variable: one organelle genome ready for analysis.

    Mitochondrial and chloroplast/plastid genomes share this type; the
    ``organelle`` field selects which. A genome carries at most one sequence
    (FASTA) and one annotation (GenBank/GFF3) path — exactly the two input
    shapes the OrganelleVerse suites consume.
    """

    organelle: Organelle
    sequence_path: Path | None = None  # FASTA (annotate / qc / cms / repeat / ...)
    annotation_path: Path | None = None  # GenBank/GFF3 (codon / kaks / synteny / ...)
    metadata: OrganelleMetadata = field(default_factory=OrganelleMetadata)

    def __post_init__(self) -> None:
        if self.organelle not in ("mito", "chloro", "plastid"):
            raise ValueError(
                f"organelle must be 'mito' | 'chloro' | 'plastid', got {self.organelle!r}"
            )
        if self.sequence_path is None and self.annotation_path is None:
            raise ValueError(
                "OrganelleGenome needs at least one of sequence_path / annotation_path"
            )

    # -- constructors -----------------------------------------------------

    @classmethod
    def from_fasta(
        cls,
        path: str | Path,
        *,
        organelle: Organelle,
        **metadata: Any,
    ) -> OrganelleGenome:
        """Build from a FASTA sequence file."""
        return cls(
            organelle=organelle,
            sequence_path=Path(path),
            metadata=_build_metadata(metadata),
        )

    @classmethod
    def from_genbank(
        cls,
        path: str | Path,
        *,
        organelle: Organelle,
        **metadata: Any,
    ) -> OrganelleGenome:
        """Build from a GenBank annotation file."""
        return cls(
            organelle=organelle,
            annotation_path=Path(path),
            metadata=_build_metadata(metadata),
        )

    # -- accessors --------------------------------------------------------

    @property
    def primary_path(self) -> Path:
        """The path most suite functions should read first (sequence > annotation)."""
        if self.sequence_path is not None:
            return self.sequence_path
        assert self.annotation_path is not None  # post_init guarantees non-None
        return self.annotation_path


@dataclass(frozen=True)
class ResultProvenance:
    """Provenance carried by every :class:`OrganelleResult`.

    The ``run_manifest_id`` field is the bridge to the ``_c`` governance layer:
    when a result is produced via the ``BioToolExecutionAdapter`` consumer path,
    the manifest id is back-filled here so the result is traceable to a run
    manifest (see design §2.2 / §6 red line).
    """

    suite: str
    op: str
    method: str = ""
    software_version: str = ""
    argv: tuple[str, ...] = ()
    input_fingerprint: str = ""  # e.g. sha1 of primary input, for caching/dedup
    run_manifest_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "suite": self.suite,
            "op": self.op,
            "method": self.method,
            "software_version": self.software_version,
            "argv": list(self.argv),
            "input_fingerprint": self.input_fingerprint,
            "run_manifest_id": self.run_manifest_id,
        }


@dataclass(frozen=True)
class OrganelleResult:
    """Output-state unified variable: an analysis result with provenance.

    Carries the produced files plus a flat, LLM-readable summary
    (``key_findings`` / ``flags`` / ``anomalies``) so any agent can interpret
    the result without parsing binary formats (design §2.3).
    """

    suite: str
    op: str
    organelle: Organelle
    status: str  # "ok" | "warning" | "failed"
    output_paths: tuple[Path, ...] = ()
    observed_metrics: dict[str, Any] = field(default_factory=dict)
    key_findings: tuple[dict[str, Any], ...] = ()
    flags: tuple[str, ...] = ()
    anomalies: tuple[str, ...] = ()
    summary_text: str = ""
    provenance: ResultProvenance = field(default_factory=ResultProvenance)

    def __post_init__(self) -> None:
        # Coerce list-like fields to tuples so callers can pass plain lists.
        object.__setattr__(self, "output_paths", tuple(self.output_paths))
        object.__setattr__(self, "key_findings", tuple(self.key_findings))
        object.__setattr__(self, "flags", tuple(self.flags))
        object.__setattr__(self, "anomalies", tuple(self.anomalies))

    @classmethod
    def ok(
        cls,
        *,
        suite: str,
        op: str,
        organelle: Organelle,
        output_paths: tuple[Path, ...] = (),
        observed_metrics: dict[str, Any] | None = None,
        key_findings: tuple[dict[str, Any], ...] = (),
        flags: tuple[str, ...] = (),
        anomalies: tuple[str, ...] = (),
        summary_text: str = "",
        provenance: ResultProvenance | None = None,
    ) -> OrganelleResult:
        return cls(
            suite=suite,
            op=op,
            organelle=organelle,
            status="ok",
            output_paths=output_paths,
            observed_metrics=dict(observed_metrics or {}),
            key_findings=key_findings,
            flags=flags,
            anomalies=anomalies,
            summary_text=summary_text,
            provenance=provenance or ResultProvenance(suite=suite, op=op),
        )

    @classmethod
    def failed(
        cls,
        *,
        suite: str,
        op: str,
        organelle: Organelle,
        output_paths: tuple[Path, ...] = (),
        observed_metrics: dict[str, Any] | None = None,
        key_findings: tuple[dict[str, Any], ...] = (),
        flags: tuple[str, ...] = (),
        summary_text: str = "",
        anomalies: tuple[str, ...] = (),
        provenance: ResultProvenance | None = None,
    ) -> OrganelleResult:
        return cls(
            suite=suite,
            op=op,
            organelle=organelle,
            status="failed",
            output_paths=output_paths,
            observed_metrics=dict(observed_metrics or {}),
            key_findings=key_findings,
            flags=flags,
            summary_text=summary_text,
            anomalies=anomalies,
            provenance=provenance or ResultProvenance(suite=suite, op=op),
        )

    def summary_json(self) -> dict[str, Any]:
        """Return the LLM-readable summary payload (design §2.3 fixed schema)."""
        return {
            "schema_version": ORGANELLE_RESULT_SCHEMA_VERSION,
            "suite": self.suite,
            "op": self.op,
            "organelle": self.organelle,
            "status": self.status,
            "summary_text": self.summary_text,
            "key_findings": list(self.key_findings),
            "flags": list(self.flags),
            "anomalies": list(self.anomalies),
            "output_files": {str(p): str(p) for p in self.output_paths},
            "artifacts": [_artifact_record(p) for p in self.output_paths],
            "observed_metrics": dict(self.observed_metrics),
            "provenance": self.provenance.as_dict(),
        }


def organelle_result_json_schema() -> dict[str, Any]:
    """Return the versioned JSON Schema for ``OrganelleResult.summary_json()``."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": ORGANELLE_RESULT_SCHEMA_VERSION,
        "type": "object",
        "required": [
            "schema_version",
            "suite",
            "op",
            "organelle",
            "status",
            "summary_text",
            "key_findings",
            "flags",
            "anomalies",
            "output_files",
            "artifacts",
            "observed_metrics",
            "provenance",
        ],
        "properties": {
            "schema_version": {"const": ORGANELLE_RESULT_SCHEMA_VERSION},
            "suite": {"type": "string"},
            "op": {"type": "string"},
            "organelle": {"enum": ["mito", "chloro", "plastid"]},
            "status": {"type": "string"},
            "summary_text": {"type": "string"},
            "key_findings": {"type": "array", "items": {"type": "object"}},
            "flags": {"type": "array", "items": {"type": "string"}},
            "anomalies": {"type": "array", "items": {"type": "string"}},
            "output_files": {"type": "object"},
            "artifacts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["path", "kind", "exists"],
                    "properties": {
                        "path": {"type": "string"},
                        "kind": {"type": "string"},
                        "exists": {"type": "boolean"},
                        "sha256": {"type": "string"},
                        "size_bytes": {"type": "integer"},
                        "preview": {"type": "string"},
                    },
                },
            },
            "observed_metrics": {"type": "object"},
            "provenance": {"type": "object"},
        },
    }


def _build_metadata(raw: dict[str, Any]) -> OrganelleMetadata:
    """Construct OrganelleMetadata from kwargs, pulling known fields out first."""
    known = {"species", "accession", "genetic_code", "assembly_type", "provenance"}
    known_kwargs = {k: str(v) for k, v in raw.items() if k in known}
    extra = tuple(sorted((k, str(v)) for k, v in raw.items() if k not in known))
    return OrganelleMetadata(extra=extra, **known_kwargs)


def _artifact_record(path: Path) -> dict[str, Any]:
    p = Path(path)
    record: dict[str, Any] = {
        "path": str(p),
        "kind": _artifact_kind(p),
        "exists": p.exists(),
    }
    if not p.is_file():
        return record
    record["size_bytes"] = p.stat().st_size
    record["sha256"] = _sha256_file(p)
    preview = _text_preview(p)
    if preview:
        record["preview"] = preview
    return record


def _artifact_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".tsv", ".csv"}:
        return "table"
    if suffix in {".nwk", ".tree", ".treefile"}:
        return "tree"
    if suffix in {".gfa", ".graph"}:
        return "graph"
    if suffix in {".svg", ".png", ".pdf"}:
        return "figure"
    if suffix in {".fa", ".fasta", ".fna", ".faa"}:
        return "sequence"
    if suffix in {".gff", ".gff3", ".gb", ".gbk"}:
        return "annotation"
    if suffix == ".json":
        return "json"
    return "file"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_preview(path: Path, limit: int = 512) -> str:
    data = path.read_bytes()[:limit]
    if b"\x00" in data:
        return ""
    return data.decode("utf-8", errors="replace").strip()
