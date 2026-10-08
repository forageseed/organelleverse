"""Immutable platform-capability environment and database specifications.

This module holds only static specifications: which exact Conda platform builds
carriers Oatk, and which exact OatkDB profile files are managed. It imports
contracts and static specifications only and performs no Conda probing, network
access, cache creation, or subprocess execution.
"""

from __future__ import annotations

import platform
from importlib import resources as importlib_resources
from typing import Literal

from pydantic import Field, model_validator

from organelleverse.core.errors import OrganelleDependencyError
from organelleverse.operations.spec import StrictSpecModel

OATKDB_SOURCE_COMMIT = "75e8db0ac4a7d508a9a518d900876003ceb70737"
OATKDB_VERSION = "v20230921"
PMAT_ORIENTATION_BANNER = "2.1.5 + OrganelleVerse orientation patch"
PMAT_ORIENTATION_VERSION = f"PMAT2 {PMAT_ORIENTATION_BANNER}"
OATKDB_RESOURCE = "organelleverse.assembly.resources.databases.oatkdb-v20230921.json"
OATKDB_RAW_BASE = (
    f"https://raw.githubusercontent.com/c-zhou/OatkDB/{OATKDB_SOURCE_COMMIT}/{OATKDB_VERSION}/"
)


class ManagedFileSpec(StrictSpecModel):
    name: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    url: str = Field(pattern=r"^https://")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(gt=0)


class ManagedDatabaseSpec(StrictSpecModel):
    database_id: str
    version: str
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    license: str
    files: tuple[ManagedFileSpec, ...]


class ManagedBuildOutput(StrictSpecModel):
    role: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    relative_path: str
    kind: Literal["executable", "resource"]
    install_name: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_.+-]+$")

    @model_validator(mode="after")
    def validate_path_and_install_name(self) -> ManagedBuildOutput:
        _require_safe_relative_path(self.relative_path)
        if (self.kind == "executable") != (self.install_name is not None):
            raise ValueError("only executable source outputs require install_name")
        return self


class ManagedExecutableAlias(StrictSpecModel):
    name: str = Field(pattern=r"^[A-Za-z0-9_.+-]+$")
    target: str = Field(pattern=r"^[A-Za-z0-9_.+-]+$")


class ManagedSourcePatch(StrictSpecModel):
    resource: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ManagedSourceBuildSpec(StrictSpecModel):
    source_uri: str = Field(pattern=r"^https://")
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    archive_root: str = Field(pattern=r"^[A-Za-z0-9_.+-]+$")
    build_argv: tuple[str, ...] = Field(min_length=1)
    outputs: tuple[ManagedBuildOutput, ...] = Field(min_length=1)
    executable_aliases: tuple[ManagedExecutableAlias, ...] = ()
    required_host_commands: tuple[tuple[str, ...], ...] = ()
    patches: tuple[ManagedSourcePatch, ...] = ()

    @model_validator(mode="after")
    def validate_closed_build(self) -> ManagedSourceBuildSpec:
        if any(not value or "\x00" in value for value in self.build_argv):
            raise ValueError("build argv must contain non-empty safe process arguments")
        roles = tuple(item.role for item in self.outputs)
        paths = tuple(item.relative_path for item in self.outputs)
        install_names = tuple(
            item.install_name for item in self.outputs if item.install_name is not None
        )
        aliases = tuple(item.name for item in self.executable_aliases)
        if len(set(roles)) != len(roles) or len(set(paths)) != len(paths):
            raise ValueError("managed build outputs must have unique roles and paths")
        if len(set(install_names + aliases)) != len(install_names + aliases):
            raise ValueError("managed executable names and aliases must be unique")
        if any(
            not alternatives or len(set(alternatives)) != len(alternatives)
            for alternatives in self.required_host_commands
        ):
            raise ValueError("host command alternatives must be non-empty and unique")
        return self


class CondaPlatformSpec(StrictSpecModel):
    platform: str
    package: str
    lock_resource: str
    executable_names: tuple[str, ...]
    version_argv: tuple[str, ...]
    version_exit_codes: tuple[int, ...] = (0,)
    source_build: ManagedSourceBuildSpec | None = None


class AssemblyEnvironmentSpec(StrictSpecModel):
    backend_id: str
    contract_version: str
    # "native": a binary shipped with OrganelleVerse; no prefix is created or resolved for it
    carrier: Literal["conda", "native"] = "conda"
    platforms: tuple[CondaPlatformSpec, ...]
    database: ManagedDatabaseSpec | None = None

    def require_platform(self, platform: str) -> CondaPlatformSpec:
        match = next((item for item in self.platforms if item.platform == platform), None)
        if match is None:
            raise OrganelleDependencyError(
                code="assembly.environment_unavailable",
                message=f"no managed {self.backend_id} environment for {platform}",
                details={
                    "platform": platform,
                    "available_platforms": [item.platform for item in self.platforms],
                },
                suggested_action={"select_compatible_compute": True},
            )
        return match

    def require_database(self) -> ManagedDatabaseSpec:
        if self.database is None:
            raise OrganelleDependencyError(
                code="assembly.environment_unavailable",
                message=f"{self.backend_id} has no managed database contract",
            )
        return self.database


def normalize_platform(
    *,
    system: str | None = None,
    machine: str | None = None,
) -> str:
    """Normalize host OS/architecture into a Conda platform string."""
    resolved_system = (system if system is not None else platform.system()).strip().lower()
    resolved_machine = (machine if machine is not None else platform.machine()).strip().lower()
    # Conda uses "osx" for macOS despite the kernel reporting "Darwin".
    conda_system = "osx" if resolved_system == "darwin" else resolved_system
    if conda_system in {"linux", "osx"}:
        if resolved_machine in {"x86_64", "amd64"}:
            arch = "64"
        elif resolved_machine in {"aarch64", "arm64"}:
            arch = "aarch64" if conda_system == "linux" else "arm64"
        else:
            arch = resolved_machine
        return f"{conda_system}-{arch}"
    return f"{conda_system}-{resolved_machine}"


def _require_safe_relative_path(value: str) -> None:
    from pathlib import PurePosixPath

    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("managed source output must be a safe relative path")


def _load_managed_database() -> ManagedDatabaseSpec:
    raw = importlib_resources.files("organelleverse.assembly.resources.databases").joinpath(
        "oatkdb-v20230921.json"
    )
    payload = raw.read_text(encoding="utf-8")
    import json

    data = json.loads(payload)
    files = tuple(
        ManagedFileSpec(
            name=item["name"],
            url=OATKDB_RAW_BASE + item["name"],
            sha256=item["sha256"],
            size_bytes=item["size_bytes"],
        )
        for item in data["files"]
    )
    names = [item.name for item in files]
    hashes = [item.sha256 for item in files]
    if len(set(names)) != len(names):
        raise OrganelleDependencyError(
            code="assembly.environment_unavailable",
            message="managed OatkDB manifest contains duplicate file names",
        )
    if len(set(hashes)) != len(hashes):
        raise OrganelleDependencyError(
            code="assembly.environment_unavailable",
            message="managed OatkDB manifest contains duplicate file hashes",
        )
    return ManagedDatabaseSpec(
        database_id=data["database_id"],
        version=data["version"],
        source_commit=data["source_commit"],
        license=data["license"],
        files=files,
    )


_OATK_PLATFORMS = (
    CondaPlatformSpec(
        platform="linux-64",
        package="oatk=1.0=h577a1d6_1",
        lock_resource="organelleverse.assembly.resources.environments.oatk/linux-64.explicit.txt",
        executable_names=("oatk", "nhmmscan", "hmmpress"),
        version_argv=("--version",),
    ),
    CondaPlatformSpec(
        platform="osx-64",
        package="oatk=1.0=h7f84b70_1",
        lock_resource="organelleverse.assembly.resources.environments.oatk/osx-64.explicit.txt",
        executable_names=("oatk", "nhmmscan", "hmmpress"),
        version_argv=("--version",),
    ),
)


OATK_ENVIRONMENT = AssemblyEnvironmentSpec(
    backend_id="oatk",
    contract_version="organelleverse.assembly-environment.v1",
    platforms=_OATK_PLATFORMS,
    database=_load_managed_database(),
)

_HIMT_PLATFORMS = (
    CondaPlatformSpec(
        platform="linux-64",
        package="himt=1.1.3=0",
        lock_resource="organelleverse.assembly.resources.environments.himt/linux-64.explicit.txt",
        executable_names=("himt", "flye", "makeblastdb", "tblastn", "minimap2", "miniprot"),
        version_argv=("--version",),
    ),
    CondaPlatformSpec(
        platform="linux-aarch64",
        package="himt=1.1.3=0",
        lock_resource="organelleverse.assembly.resources.environments.himt/linux-aarch64.explicit.txt",
        executable_names=("himt", "flye", "makeblastdb", "tblastn", "minimap2", "miniprot"),
        version_argv=("--version",),
    ),
    CondaPlatformSpec(
        platform="osx-64",
        package="himt=1.1.3=0",
        lock_resource="organelleverse.assembly.resources.environments.himt/osx-64.explicit.txt",
        executable_names=("himt", "flye", "makeblastdb", "tblastn", "minimap2", "miniprot"),
        version_argv=("--version",),
    ),
    CondaPlatformSpec(
        platform="osx-arm64",
        package="himt=1.1.3=0",
        lock_resource="organelleverse.assembly.resources.environments.himt/osx-arm64.explicit.txt",
        executable_names=("himt", "flye", "makeblastdb", "tblastn", "minimap2", "miniprot"),
        version_argv=("--version",),
    ),
)


HIMT_ENVIRONMENT = AssemblyEnvironmentSpec(
    backend_id="himt",
    contract_version="organelleverse.assembly-environment.v1",
    platforms=_HIMT_PLATFORMS,
    database=None,
)


# GetOrganelle 1.7.7.1 is a noarch Bioconda package; the strict conda-forge +
# bioconda solve surfaces build number 0 as the ``pyhdfd78af_0`` build string.
# The upstream version is the four-part ``1.7.7.1`` (``get_organelle_from_reads.py
# --version`` prints ``GetOrganelle v1.7.7.1``). It is both the pinned package
# identity here and a first-class ``ResolvedBackendVersion`` in the tested
# resolver registry (the shared identity accepts strict 3- or 4-component numeric
# releases); the capability probe parses it with the strict four-part style.
GETORGANELLE_VERSION = "1.7.7.1"
GETORGANELLE_PACKAGE = f"getorganelle={GETORGANELLE_VERSION}=pyhdfd78af_0"
GETORGANELLE_PLATFORM = "linux-64"

# Use the official commit archive, selecting its full 0.0.1 subtree (never
# 0.0.1.minima). All 14 FASTA files match the published 0.0.1 release byte for
# byte. The archive bytes, size and digest were verified by a real download.
GETORGANELLEDB_VERSION = "0.0.1"
GETORGANELLEDB_SOURCE_COMMIT = "8610b6e67d4d9269c85de91e0f58266c31f72388"
GETORGANELLEDB_LICENSE = "GPL-3.0"
GETORGANELLEDB_ARCHIVE_NAME = f"GetOrganelleDB-{GETORGANELLEDB_SOURCE_COMMIT}.tar.gz"
GETORGANELLEDB_ARCHIVE_URL = (
    "https://github.com/Kinggerm/GetOrganelleDB/archive/"
    f"{GETORGANELLEDB_SOURCE_COMMIT}.tar.gz"
)
GETORGANELLEDB_ARCHIVE_SHA256 = "a522e42f7127d38bb2ce3eb9224213f5997fdd12aae1fa782c57b29ed3754315"
GETORGANELLEDB_ARCHIVE_SIZE_BYTES = 43367830

_GETORGANELLE_PLATFORMS = (
    CondaPlatformSpec(
        platform=GETORGANELLE_PLATFORM,
        package=GETORGANELLE_PACKAGE,
        lock_resource=(
            "organelleverse.assembly.resources.environments.getorganelle/linux-64.explicit.txt"
        ),
        executable_names=(
            "get_organelle_from_reads.py",
            "get_organelle_config.py",
            "blastn",
            "bowtie2",
            "spades.py",
        ),
        version_argv=("--version",),
    ),
)


GETORGANELLE_ENVIRONMENT = AssemblyEnvironmentSpec(
    backend_id="getorganelle",
    contract_version="organelleverse.assembly-environment.v1",
    platforms=_GETORGANELLE_PLATFORMS,
    database=ManagedDatabaseSpec(
        database_id="getorganelledb",
        version=GETORGANELLEDB_VERSION,
        source_commit=GETORGANELLEDB_SOURCE_COMMIT,
        license=GETORGANELLEDB_LICENSE,
        files=(
            ManagedFileSpec(
                name=GETORGANELLEDB_ARCHIVE_NAME,
                url=GETORGANELLEDB_ARCHIVE_URL,
                sha256=GETORGANELLEDB_ARCHIVE_SHA256,
                size_bytes=GETORGANELLEDB_ARCHIVE_SIZE_BYTES,
            ),
        ),
    ),
)


PTGAUL_ENVIRONMENT = AssemblyEnvironmentSpec(
    backend_id="ptgaul",
    contract_version="organelleverse.assembly-environment.v1",
    platforms=(
        CondaPlatformSpec(
            platform="linux-64",
            package="ptgaul=1.0.5",
            lock_resource="existing-prefix-required",
            executable_names=(
                "ptGAUL.sh",
                "combine_gfa.py",
                "python3",
                "minimap2",
                "seqkit",
                "assembly-stats",
                "seqtk",
                "flye",
            ),
            version_argv=(),
        ),
    ),
    database=None,
)


TIPPO_ENVIRONMENT = AssemblyEnvironmentSpec(
    backend_id="tippo",
    contract_version="organelleverse.assembly-environment.v1",
    platforms=(
        CondaPlatformSpec(
            platform="linux-64",
            package="tipp=2.4",
            lock_resource="existing-prefix-required",
            executable_names=("TIPPo.v2.4.pl",),
            version_argv=("-v",),
        ),
    ),
    database=None,
)


PMAT_ENVIRONMENT = AssemblyEnvironmentSpec(
    backend_id="pmat",
    contract_version="organelleverse.assembly-environment.v1",
    platforms=(
        CondaPlatformSpec(
            platform="linux-64",
            package="pmat-source=2.1.5",
            lock_resource=(
                "organelleverse.assembly.resources.environments.pmat/linux-64.explicit.txt"
            ),
            executable_names=("pmat", "blastn", "canu", "nextdenovo", "Rscript"),
            version_argv=("--version",),
            source_build=ManagedSourceBuildSpec(
                source_uri=(
                    "https://github.com/aiPGAB/PMAT2/archive/04534a2adf0c5309cb2e7478fd2bcc98ced3818c.tar.gz"
                ),
                source_sha256=("5564becd242ab254e379758133b514ce0c38431df55c77bee74836f1914bf872"),
                archive_root="PMAT2-04534a2adf0c5309cb2e7478fd2bcc98ced3818c",
                build_argv=("make", "CC=x86_64-conda-linux-gnu-cc"),
                patches=(
                    ManagedSourcePatch(
                        resource="organelleverse.assembly.resources.environments.pmat/orientation.patch",
                        sha256="e156e7a2a8428694448f38d754def1cf194425477658a4468ecc02bceee0fcee",
                    ),
                ),
                outputs=(
                    ManagedBuildOutput(
                        role="pmat",
                        relative_path="PMAT",
                        kind="executable",
                        install_name="pmat",
                    ),
                    ManagedBuildOutput(
                        role="pmat_container",
                        relative_path="container/runAssembly.sif",
                        kind="resource",
                    ),
                    ManagedBuildOutput(
                        role="pmat_genomescope",
                        relative_path="lib/genomescope.R",
                        kind="resource",
                    ),
                ),
                executable_aliases=(
                    ManagedExecutableAlias(name="nextdenovo", target="nextDenovo"),
                ),
                required_host_commands=(("apptainer", "singularity"), ("patch",)),
            ),
        ),
    ),
)


# ovasm is one self-contained binary (github.com/forageseed/ovasm), found by ``_ovasm.resolve_ovasm``: there is
# no lock and no prefix. ``CondaPlatformSpec`` only names the executable and its version probe.
OVASM_ENVIRONMENT = AssemblyEnvironmentSpec(
    backend_id="ovasm",
    contract_version="organelleverse.assembly-environment.v1",
    carrier="native",
    platforms=tuple(
        CondaPlatformSpec(
            platform=name,
            package="ovasm",
            lock_resource="native-binary",
            executable_names=("ovasm",),
            version_argv=("--version",),
        )
        for name in ("linux-64", "linux-aarch64", "osx-64", "osx-arm64", "windows-amd64")
    ),
    database=None,
)


NOVOPLASTY_ENVIRONMENT = AssemblyEnvironmentSpec(
    backend_id="novoplasty",
    contract_version="organelleverse.assembly-environment.v1",
    platforms=(
        CondaPlatformSpec(
            platform="linux-64",
            package="novoplasty=4.3.5",
            lock_resource="organelleverse.assembly.resources.environments.novoplasty/linux-64.explicit.txt",
            executable_names=("NOVOPlasty4.3.5.pl", "perl"),
            version_argv=("-c", ""),
            version_exit_codes=(2,),
        ),
    ),
)
