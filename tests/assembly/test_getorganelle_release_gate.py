"""Non-skippable real GetOrganelle release gate using the official fixture."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Literal

import pytest

from organelleverse.assembly.contracts import AssemblyRequest
from organelleverse.assembly.environment_specs import AssemblyEnvironmentSpec
from organelleverse.assembly.environments import (
    EnvironmentManager,
    PreparedEnvironment,
    PreparedExecutable,
)
from organelleverse.assembly.execution import managed_prefix_environment
from organelleverse.assembly.manifests import AssemblyRunManifest
from organelleverse.assembly.service import execute_assembly
from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.data import OrganelleData
from organelleverse.runtime import cache_root

_R1_URL = (
    "https://github.com/Kinggerm/GetOrganelleGallery/raw/master/"
    "Test/reads/Arabidopsis_simulated.1.fq.gz"
)
_R2_URL = (
    "https://github.com/Kinggerm/GetOrganelleGallery/raw/master/"
    "Test/reads/Arabidopsis_simulated.2.fq.gz"
)
_R1_MD5 = "935589bc609397f1bfc9c40f571f0f19"
_R2_MD5 = "d0f62eed78d2d2c6bed5f5aeaf4a2c11"
_EXECUTABLES = (
    "get_organelle_from_reads.py",
    "get_organelle_config.py",
    "blastn",
    "bowtie2",
    "spades.py",
)


def _existing_provider() -> Path:
    configured = os.environ.get("ORGANELLEVERSE_GETORGANELLE_PROVIDER")
    if configured:
        return Path(configured).expanduser().resolve()
    executable = shutil.which("get_organelle_from_reads.py")
    if executable:
        return Path(executable).resolve().parent.parent
    raise AssertionError(
        "GetOrganelle provider not found; set ORGANELLEVERSE_GETORGANELLE_PROVIDER "
        "or add get_organelle_from_reads.py to PATH"
    )


def _download(url: str, destination: Path, expected_md5: str) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and hashlib.md5(destination.read_bytes()).hexdigest() == expected_md5:
        return destination
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with urllib.request.urlopen(url) as response:
            data = response.read()
        actual_md5 = hashlib.md5(data).hexdigest()
        if actual_md5 != expected_md5:
            raise RuntimeError(
                f"fixture MD5 mismatch: expected {expected_md5}, observed {actual_md5}"
            )
        temporary.write_bytes(data)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _existing_environment() -> PreparedEnvironment:
    provider = _existing_provider()
    executables: list[PreparedExecutable] = []
    for name in _EXECUTABLES:
        path = provider / "bin" / name
        if not path.is_file():
            raise AssertionError(f"required existing GetOrganelle executable is missing: {path}")
        executables.append(PreparedExecutable(name=name, path=path))
    probe = subprocess.run(
        [str(provider / "bin" / "get_organelle_from_reads.py"), "--version"],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
        shell=False,
        env=managed_prefix_environment(provider),
    )
    version = probe.stdout.strip()
    assert version == "GetOrganelle v1.7.7.1"
    return PreparedEnvironment(
        backend_id="getorganelle",
        carrier="conda",
        platform="linux-64",
        digest="sha256:" + "e" * 64,
        prefix=provider,
        executables=tuple(executables),
        version=version,
    )


class _ExistingProviderManager(EnvironmentManager):
    def __init__(self, cache_root: Path, environment: PreparedEnvironment) -> None:
        super().__init__(cache_root=cache_root)
        self.environment = environment

    def prepare(
        self,
        spec: AssemblyEnvironmentSpec,
        *,
        policy: Literal["ensure", "require"],
        platform: str | None = None,
    ) -> PreparedEnvironment:
        del spec, policy, platform
        return self.environment

    def expected_environment_digest(
        self, spec: AssemblyEnvironmentSpec, *, platform: str | None = None
    ) -> str:
        del spec, platform
        return self.environment.digest


def _data(read1: Path, read2: Path) -> OrganelleData:
    return OrganelleData.model_validate(
        {
            "modality": "sequencing_reads",
            "artifacts": {
                "read1": ArtifactRef.from_path(
                    read1,
                    kind="short_read",
                    format="fastq",
                    media_type="application/x-fastq",
                ),
                "read2": ArtifactRef.from_path(
                    read2,
                    kind="short_read",
                    format="fastq",
                    media_type="application/x-fastq",
                ),
            },
            "payload": {
                "contract_version": "organelleverse.assembly-input.v1",
                "short_libraries": [
                    {
                        "technology": "illumina",
                        "layout": "paired_end",
                        "read1_artifact": "read1",
                        "read2_artifact": "read2",
                        "read_length": 150,
                    }
                ],
            },
        }
    )


@pytest.mark.release_assembly_getorganelle
def test_official_arabidopsis_assembly_and_verified_reuse() -> None:
    fixture_root = cache_root() / "fixtures" / "getorganelle"
    r1_gz = _download(_R1_URL, fixture_root / "reads.1.fq.gz", _R1_MD5)
    r2_gz = _download(_R2_URL, fixture_root / "reads.2.fq.gz", _R2_MD5)
    assert hashlib.md5(r1_gz.read_bytes()).hexdigest() == _R1_MD5
    assert hashlib.md5(r2_gz.read_bytes()).hexdigest() == _R2_MD5

    manager = _ExistingProviderManager(cache_root(), _existing_environment())
    request = AssemblyRequest(
        data=_data(r1_gz, r2_gz),
        organelle="plastid",
        method="getorganelle",
        threads=1,
        environment_source="managed",
    )
    first = execute_assembly(request, environment_manager=manager)
    assert first.status == "ok"
    assert first.provenance is not None
    assert first.provenance.actual_backend == "getorganelle"
    manifest_path = next(
        artifact.resolve() for artifact in first.artifacts if artifact.kind == "assembly_run_record"
    )
    run_dir = manifest_path.parent
    manifest = AssemblyRunManifest.model_validate_json(manifest_path.read_bytes())
    assert manifest.selected_backend == "getorganelle"
    assert manifest.organelle == "plastid"
    assert manifest.environment.digest == manager.environment.digest
    fasta = run_dir / "normalized" / "assembly.fasta"
    graph = run_dir / "normalized" / "assembly.gfa"
    assert fasta.is_file() and fasta.stat().st_size > 100
    assert graph.is_file() and "S\t" in graph.read_text()

    started = time.monotonic()
    second = execute_assembly(request, environment_manager=manager)
    elapsed = time.monotonic() - started
    assert second.object_id == first.object_id
    assert second.status == "ok"
    assert elapsed < 10
