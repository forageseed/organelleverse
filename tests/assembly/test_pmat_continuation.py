from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from organelleverse.assembly.continuation import (
    DirectoryManifest,
    DirectoryManifestFile,
    materialize_directory_manifest,
)
from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.errors import OrganelleInputError


def _artifact(tmp_path: Path, role: str, content: bytes) -> ArtifactRef:
    path = tmp_path / "artifacts" / role
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return ArtifactRef.from_path(
        path,
        kind="pmat_continuation",
        format="fasta" if role.endswith("fasta") else "txt",
        media_type="application/octet-stream",
    )


def test_directory_manifest_rejects_traversal_and_duplicate_roles() -> None:
    with pytest.raises(ValidationError):
        DirectoryManifest(
            role="pmat_subsample",
            files=(
                DirectoryManifestFile(
                    relative_path="../escape",
                    artifact_role="one",
                    sha256="a" * 64,
                ),
            ),
        )
    with pytest.raises(ValidationError):
        DirectoryManifest(
            role="pmat_assembly_result",
            files=(
                DirectoryManifestFile(relative_path="a", artifact_role="same", sha256="a" * 64),
                DirectoryManifestFile(relative_path="b", artifact_role="same", sha256="b" * 64),
            ),
        )


def test_materialize_directory_manifest_verifies_and_restores_names(tmp_path: Path) -> None:
    fasta = _artifact(tmp_path, "pmat_all_contigs_fasta", b">ctg\nACGT\n")
    graph = _artifact(tmp_path, "pmat_contig_graph", b"1\t2\n")
    manifest = DirectoryManifest(
        role="pmat_assembly_result",
        files=(
            DirectoryManifestFile(
                relative_path="PMATAllContigs.fna",
                artifact_role="pmat_all_contigs",
                sha256=fasta.sha256,
            ),
            DirectoryManifestFile(
                relative_path="PMATContigGraph.txt",
                artifact_role="pmat_contig_graph",
                sha256=graph.sha256,
            ),
        ),
    )
    destination = tmp_path / "restored"

    materialize_directory_manifest(
        manifest,
        {"pmat_all_contigs": fasta, "pmat_contig_graph": graph},
        destination,
    )

    assert (destination / "PMATAllContigs.fna").read_bytes() == b">ctg\nACGT\n"
    assert (destination / "PMATContigGraph.txt").read_bytes() == b"1\t2\n"


def test_materialize_rejects_mutation_before_creating_destination(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path, "pmat_subsample_fasta", b">read\nACGT\n")
    manifest = DirectoryManifest(
        role="pmat_subsample",
        files=(
            DirectoryManifestFile(
                relative_path="PMAT_cut_seq.fa",
                artifact_role="pmat_subsample",
                sha256=artifact.sha256,
            ),
        ),
    )
    Path(artifact.uri).write_bytes(b"tampered")
    destination = tmp_path / "not-created"

    with pytest.raises(OrganelleInputError, match="hash"):
        materialize_directory_manifest(
            manifest,
            {"pmat_subsample": artifact},
            destination,
        )
    assert not destination.exists()


def test_manifest_canonical_bytes_are_stable() -> None:
    manifest = DirectoryManifest(
        role="pmat_subsample",
        files=(
            DirectoryManifestFile(
                relative_path="PMAT_cut_seq.fa",
                artifact_role="pmat_subsample",
                sha256="a" * 64,
            ),
        ),
    )
    assert json.loads(manifest.canonical_bytes())["schema_version"] == (
        "organelleverse.directory-manifest.v1"
    )
