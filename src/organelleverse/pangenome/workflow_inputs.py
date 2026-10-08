"""Snapshot scientific inputs without joining molecules or changing coordinates."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from Bio import SeqIO
from Bio.Seq import Seq

from ..core.artifacts import ArtifactRef
from ..core.genome import OrganelleGenome, OrganelleMetadata
from ..runtime import managed_run_path
from .normalization import MoleculeTransform
from .project import normalize_identity
from .workflow_models import WorkflowRequest

_FASTA = {".fa", ".fasta", ".fna", ".fas"}
_GENBANK = {".gb", ".gbk", ".genbank"}


def prepare_inputs(request: WorkflowRequest, directory: Path) -> list[Path]:
    """Freeze every source and record source record→PanSN identity correspondence."""
    target = managed_run_path("pangenome.input_snapshot", directory.name.removeprefix(".staging-"))
    target.mkdir(parents=True, exist_ok=True)
    manifest: dict = {
        "samples": [],
        "sources": [],
        "coordinate_policy": "preserve",
        "snapshot_directory": str(target),
        "snapshots": [],
    }
    files: list[Path] = []
    if request.backend == "existing":
        source = Path(request.gfa_path).expanduser().resolve()
        graph = target / "graph.gfa"
        shutil.copyfile(source, graph)
        manifest["sources"].append(_source_record(source, "pangenome_graph", "gfa"))
        files.append(graph)
        manifest["graph"] = _source_record(graph, "pangenome_graph", "gfa")
    else:
        folder = Path(request.dataset_path).expanduser().resolve()
        if not folder.is_dir():
            raise ValueError("dataset_path must be a directory containing one file per sample")
        sources = sorted(p for p in folder.iterdir() if p.suffix.lower() in _FASTA | _GENBANK)
        if len(sources) < 2:
            raise ValueError("pangenome construction requires at least two sample files")
        names: set[str] = set()
        known_paths: set[str] = set()
        coordinate_mappings = []
        for source in sources:
            name = normalize_identity(source.stem)
            if not name or name in names:
                raise ValueError(f"empty or duplicate normalized sample identity: {source.name}")
            names.add(name)
            fmt = "genbank" if source.suffix.lower() in _GENBANK else "fasta"
            records = list(SeqIO.parse(source, fmt))
            if not records or any(not len(r.seq) for r in records):
                raise ValueError(f"empty sequence input: {source.name}")
            ids = [r.id for r in records]
            if len(set(ids)) != len(ids):
                raise ValueError(f"duplicate molecule identifiers in {source.name}")
            if any(set(str(r.seq).upper()) - set("ACGTRYSWKMBDHVN") for r in records):
                raise ValueError(f"non-IUPAC nucleotide sequence in {source.name}")
            molecules = []
            for i, record in enumerate(records, 1):
                path = f"{name}#1#{i}"
                known_paths.add(path)
                source_topology = record.annotations.get("topology", "unknown")
                topology = request.molecule_topologies.get(path, source_topology)
                declared = request.normalization.get(path)
                transform = MoleculeTransform(
                    path,
                    path,
                    len(record.seq),
                    topology,
                    orientation=declared.orientation if declared else "+",
                    origin=declared.origin if declared else 0,
                )
                record.seq = Seq(transform.sequence(str(record.seq)))
                molecules.append(
                    {
                        "source_id": record.id,
                        "path": path,
                        "length": len(record.seq),
                        "topology": topology,
                        "source_topology": source_topology,
                        "normalization": {
                            "orientation": transform.orientation,
                            "origin": transform.origin,
                        },
                    }
                )
                coordinate_mappings.extend(transform.coordinate_rows())
            fasta = target / f"{name}.fa"
            SeqIO.write(records, fasta, "fasta")
            files.append(fasta)
            annotation = None
            if fmt == "genbank":
                annotation = target / f"{name}.gb"
                shutil.copyfile(source, annotation)
                files.append(annotation)
            manifest["sources"].append(_source_record(source, "sequence", fmt))
            manifest["samples"].append(
                {
                    "name": name,
                    "fasta": str(fasta),
                    "annotation": str(annotation) if annotation else None,
                    "molecules": molecules,
                }
            )
        unknown = (set(request.normalization) | set(request.molecule_topologies)) - known_paths
        if unknown:
            raise ValueError(f"Normalization refers to unknown molecule paths: {sorted(unknown)}")
        manifest["coordinate_mappings"] = coordinate_mappings
        if request.normalization or request.molecule_topologies:
            manifest["coordinate_policy"] = (
                "explicit reversible transforms; other molecules preserved"
            )
    if request.msa_path:
        source = Path(request.msa_path).expanduser().resolve()
        alignment = target / ("alignment.maf" if request.msa_format == "maf" else "alignment.fa")
        shutil.copyfile(source, alignment)
        files.append(alignment)
        manifest["alignment"] = str(alignment)
        manifest["sources"].append(_source_record(source, "alignment", request.msa_format))
    if request.annotations_path:
        source = Path(request.annotations_path).expanduser().resolve()
        annotation = target / ("annotations" + source.suffix.lower())
        shutil.copyfile(source, annotation)
        files.append(annotation)
        manifest["annotations"] = str(annotation)
        manifest["sources"].append(_source_record(source, "annotation", source.suffix.lstrip(".")))
    manifest["snapshots"] = [
        _source_record(path, "input_snapshot", path.suffix.lstrip(".")) for path in files
    ]
    manifest_path = directory / "inputs.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    files.append(manifest_path)
    return files


def load_genomes(directory: Path, organelle: str) -> list[OrganelleGenome]:
    manifest = json.loads((directory / "inputs.json").read_text())
    return [
        OrganelleGenome(
            organelle=organelle,
            sequence=ArtifactRef.from_path(
                directory / sample["fasta"], kind="sequence", format="fasta"
            ),
            annotation=(
                ArtifactRef.from_path(
                    directory / sample["annotation"], kind="annotation", format="genbank"
                )
                if sample["annotation"]
                else None
            ),
            metadata=OrganelleMetadata(accession=sample["name"], source="pangenome-workflow"),
        )
        for sample in manifest["samples"]
    ]


def _source_record(path: Path, kind: str, fmt: str) -> dict:
    return ArtifactRef.from_path(path, kind=kind, format=fmt).model_dump(mode="json")


def validate_project_paths(graph_path: Path, manifest: dict) -> dict:
    """Require every input molecule to be exactly reconstructible with its identity."""
    from .graph import path_sequences

    expected = {}
    for sample in manifest["samples"]:
        records = list(SeqIO.parse(sample["fasta"], "fasta"))
        for molecule, record in zip(sample["molecules"], records, strict=True):
            expected[molecule["path"]] = str(record.seq).upper()
    actual = path_sequences(graph_path)
    if set(actual) != set(expected):
        raise ValueError(
            f"Built graph path identities differ from input molecules: missing={sorted(set(expected) - set(actual))}, unexpected={sorted(set(actual) - set(expected))}"
        )
    changed = [name for name in expected if actual[name].upper() != expected[name]]
    if changed:
        raise ValueError(f"Built graph paths do not exactly reconstruct input molecules: {changed}")
    return {
        "status": "passed",
        "molecules": len(expected),
        "bases": sum(len(sequence) for sequence in expected.values()),
        "comparison": "exact full-molecule sequence and PanSN identity; case-insensitive DNA",
    }
