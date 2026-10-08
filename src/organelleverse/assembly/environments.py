"""Managed Oatk environment and profile materialization.

The manager resolves content-addressed cache identities from committed lock and
database manifest bytes without touching the cache, materializes verified Conda
prefixes and OatkDB profiles under ``ensure``, and verifies existing caches under
``require``. ``require`` is strictly offline: it never creates a directory, invokes a
downloader, or falls back to ``ensure``.

Environment identity and database/profile identity are independent: changing a profile
does not duplicate an otherwise identical Conda prefix.
"""

from __future__ import annotations

import fcntl
import hashlib
import io
import json
import logging
import os
import shutil
import tarfile
import tempfile
import time
from collections.abc import Callable, Mapping
from http.client import IncompleteRead
from importlib import resources as importlib_resources
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Any, Literal, Protocol, TextIO, cast

from pydantic import ConfigDict, Field

from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.errors import OrganelleDependencyError
from organelleverse.core.external import MESSAGE_TAIL_LINES, run_external, tail_lines
from organelleverse.operations.spec import StrictSpecModel

from .environment_contracts import (
    BackendCapabilityContract,
    EnvironmentSource,
    ProviderComponentIdentity,
    ResolvedProvider,
    canonical_json_bytes,
)
from .environment_registry import InstallationRegistry, default_tool_root
from .environment_specs import (
    PMAT_ORIENTATION_BANNER,
    PMAT_ORIENTATION_VERSION,
    AssemblyEnvironmentSpec,
    CondaPlatformSpec,
    ManagedDatabaseSpec,
    ManagedFileSpec,
    ManagedSourceBuildSpec,
    normalize_platform,
)
from .execution import managed_prefix_environment
from .manifests import AssemblyComponentIdentity

__all__ = [
    "EnvironmentManager",
    "PreparedEnvironment",
    "PreparedExecutable",
    "PreparedProfile",
]

_DEFAULT_CACHE_ROOT = Path.home() / ".cache" / "organelleverse"


class Carrier(Protocol):
    def create_prefix(self, explicit_lock: str, destination: Path) -> None: ...
    def probe_version(self, executable: Path, argv: tuple[str, ...]) -> str: ...
    def probe_package_identity(self, prefix: Path) -> tuple[tuple[str, str, str], ...]: ...


class Downloader(Protocol):
    def download(self, url: str, destination: Path, expected_sha256: str) -> bytes: ...


class ProfileCompiler(Protocol):
    def press(self, executable: Path, profile: Path) -> None: ...


class _DefaultProfileCompiler:
    def press(self, executable: Path, profile: Path) -> None:
        try:
            run_external(
                [str(executable), "-f", str(profile)],
                cwd=profile.parent,
                tool="hmmpress",
                code="assembly.environment_unavailable",
                error_class=OrganelleDependencyError,
                extra_details={"profile": str(profile)},
            )
        except OrganelleDependencyError as error:
            details = cast(Mapping[str, Any], error.details)
            raise OrganelleDependencyError(
                code=error.code,
                message=_message_with_stderr_tail(
                    "hmmpress failed for the user-supplied profile", details
                ),
                details=details,
            ) from error


class _DefaultCarrier:
    """Carrier that locates micromamba/conda and creates a locked prefix.

    Only used when no carrier is injected; tests inject fakes.
    """

    def create_prefix(self, explicit_lock: str, destination: Path) -> None:
        executable = _locate_carrier()
        destination.parent.mkdir(parents=True, exist_ok=True)
        lock_path = _materialize_lock(explicit_lock, destination.parent)
        argv = [
            executable,
            "create",
            "-y",
            "-p",
            str(destination),
            "-f",
            str(lock_path),
        ]
        run_external(
            argv,
            code="assembly.environment_unavailable",
            error_class=OrganelleDependencyError,
        )
        lock_path.unlink(missing_ok=True)

    def probe_version(self, executable: Path, argv: tuple[str, ...]) -> str:
        completed = run_external(
            [str(executable), *argv],
            code="assembly.environment_unavailable",
            error_class=OrganelleDependencyError,
            env=managed_prefix_environment(executable.parent.parent),
        )
        return completed.stdout.strip()

    def probe_package_identity(self, prefix: Path) -> tuple[tuple[str, str, str], ...]:
        meta = prefix / "conda-meta"
        if not meta.exists():
            return ()
        records: list[tuple[str, str, str]] = []
        for entry in sorted(meta.glob("*.json")):
            data = json.loads(entry.read_text())
            records.append((data["name"], data["version"], data["build"]))
        return tuple(records)


class _DefaultDownloader:
    def download(self, url: str, destination: Path, expected_sha256: str) -> bytes:
        import urllib.error
        import urllib.request

        destination.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(1, 4):
            try:
                with urllib.request.urlopen(url, timeout=180) as response:
                    data = response.read()
                break
            except (urllib.error.URLError, TimeoutError, ConnectionError, IncompleteRead) as error:
                retryable = not isinstance(error, urllib.error.HTTPError) or (
                    error.code in (408, 429) or 500 <= error.code < 600
                )
                if attempt == 3 or not retryable:
                    raise OrganelleDependencyError(
                        code="assembly.environment_unavailable",
                        message="managed download failed; retry the managed installation",
                        details={"url": url, "attempts": attempt, "error": str(error)},
                    ) from error
                logging.getLogger(__name__).warning(
                    "Managed download attempt %s failed for %s: %s; retrying",
                    attempt,
                    url,
                    error,
                )
                time.sleep(2)
        actual = hashlib.sha256(data).hexdigest()
        if actual != expected_sha256:
            raise OrganelleDependencyError(
                code="assembly.environment_unavailable",
                message="downloaded database file hash does not match the manifest",
                details={"url": url, "expected_sha256": expected_sha256, "actual_sha256": actual},
            )
        destination.write_bytes(data)
        return data


class PreparedExecutable(StrictSpecModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        revalidate_instances="always",
        arbitrary_types_allowed=True,
    )
    name: str = Field(pattern=r"^[A-Za-z0-9_.+-]+$")
    path: Path


class PreparedEnvironment(StrictSpecModel):
    model_config = ConfigDict(
        frozen=True, extra="forbid", revalidate_instances="always", arbitrary_types_allowed=True
    )

    backend_id: str
    carrier: Literal["conda", "native"]
    platform: str
    digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    prefix: Path
    executables: tuple[PreparedExecutable, ...]
    version: str
    software_version: str | None = None

    def require_executable(self, name: str) -> Path:
        matches = tuple(item.path for item in self.executables if item.name == name)
        if len(matches) != 1:
            raise OrganelleDependencyError(
                code="assembly.environment_unavailable",
                message=f"managed {self.backend_id} environment has no unique {name!r}",
                details={"executable": name, "prefix": str(self.prefix)},
            )
        return matches[0]


class PreparedProfile(StrictSpecModel):
    model_config = ConfigDict(
        frozen=True, extra="forbid", revalidate_instances="always", arbitrary_types_allowed=True
    )

    source: Literal["managed", "user_supplied"]
    target: Literal["mitochondrion", "plastid"]
    path: Path
    artifact: ArtifactRef
    component: AssemblyComponentIdentity


class EnvironmentManager:
    """Content-addressed cache manager for managed Oatk environments and profiles."""

    def __init__(
        self,
        *,
        cache_root: Path | None = None,
        carrier: Carrier | None = None,
        downloader: Downloader | None = None,
        profile_compiler: ProfileCompiler | None = None,
        find_executable: Callable[[tuple[str, ...]], Path | None] | None = None,
        tool_root: Path | None = None,
    ) -> None:
        env_root = os.environ.get("ORGANELLEVERSE_CACHE_ROOT")
        self._cache_root_explicit = cache_root is not None or bool(env_root)
        if cache_root is not None:
            self.cache_root = Path(cache_root)
        elif env_root:
            self.cache_root = Path(env_root)
        else:
            self.cache_root = _DEFAULT_CACHE_ROOT
        self._carrier = carrier if carrier is not None else _DefaultCarrier()
        self._downloader = downloader if downloader is not None else _DefaultDownloader()
        self._profile_compiler = (
            profile_compiler if profile_compiler is not None else _DefaultProfileCompiler()
        )
        self._find_executable = find_executable or _find_first_executable
        self.tool_root = tool_root if tool_root is not None else default_tool_root()

    # ------------------------------------------------------------------
    # Static identities (no cache, no carrier, no download).
    # ------------------------------------------------------------------

    def expected_environment_digest(
        self,
        spec: AssemblyEnvironmentSpec,
        *,
        platform: str | None = None,
    ) -> str:
        resolved = platform if platform is not None else normalize_platform()
        platform_spec = spec.require_platform(resolved)
        lock_bytes = _read_resource(platform_spec.lock_resource)
        payload: dict[str, object] = {
            "contract_version": spec.contract_version,
            "backend_id": spec.backend_id,
            "carrier": spec.carrier,
            "platform": resolved,
            "package": platform_spec.package,
            "lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
            "executable_names": list(platform_spec.executable_names),
            "version_argv": list(platform_spec.version_argv),
        }
        if platform_spec.version_exit_codes != (0,):
            payload["version_exit_codes"] = list(platform_spec.version_exit_codes)
        if platform_spec.source_build is not None:
            payload["source_build"] = platform_spec.source_build.model_dump(mode="json")
        return (
            "sha256:"
            + hashlib.sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
        )

    def expected_profile_identity(
        self,
        spec: AssemblyEnvironmentSpec,
        *,
        organelle: Literal["mitochondrion", "plastid"],
        override: ArtifactRef | None = None,
    ) -> AssemblyComponentIdentity:
        if override is not None:
            actual = _hash_artifact_ref(override)
            if actual != override.sha256:
                raise OrganelleDependencyError(
                    code="assembly.environment_unavailable",
                    message="user-supplied profile ArtifactRef hash no longer matches its content",
                    details={
                        "declared_sha256": override.sha256,
                        "actual_sha256": actual,
                    },
                )
            return AssemblyComponentIdentity(
                category="profile",
                name=f"user-{organelle}-hmm-profile",
                sha256=_user_profile_identity_digest(
                    raw_sha256=actual,
                    environment_digest=self.expected_environment_digest(spec),
                ),
                locator="user-supplied",
            )
        database = spec.require_database()
        name = _profile_name(organelle)
        return AssemblyComponentIdentity(
            category="profile",
            name=name,
            sha256=_profile_bundle_digest(database, organelle),
            locator=(f"{database.database_id}:{database.version}#{name}-hmm-bundle"),
        )

    # ------------------------------------------------------------------
    # Environment materialization.
    # ------------------------------------------------------------------

    def prepare(
        self,
        spec: AssemblyEnvironmentSpec,
        *,
        policy: Literal["ensure", "require"],
        platform: str | None = None,
    ) -> PreparedEnvironment:
        digest = self.expected_environment_digest(spec, platform=platform)
        cache_key = _digest_cache_key(digest)
        resolved_platform = platform if platform is not None else normalize_platform()
        platform_spec = spec.require_platform(resolved_platform)
        environment_root = self.cache_root / "environments"
        if platform_spec.source_build is not None and not self._cache_root_explicit:
            environment_root = self.tool_root / "providers"
        prefix_root = environment_root / spec.backend_id / cache_key

        if platform_spec.source_build is not None:
            self._require_host_commands(platform_spec.source_build)

        if policy == "require":
            return self._verify_existing_environment(spec, platform_spec, prefix_root, digest)

        # ensure: lazily create cache root, lock, recheck, create sibling temp, verify, publish.
        environment_root.parent.mkdir(parents=True, exist_ok=True)
        locks_root = environment_root.parent / "locks"
        locks_root.mkdir(parents=True, exist_ok=True)
        lock_path = locks_root / f"{cache_key}.lock"
        with _advisory_lock(lock_path):
            if prefix_root.exists():
                return self._verify_prefix(spec, platform_spec, prefix_root, digest)
            return self._create_and_publish(spec, platform_spec, prefix_root, digest)

    def prepare_provider(self, provider: ResolvedProvider) -> PreparedEnvironment:
        """Revalidate and adapt an existing resolved provider without installing it."""
        if provider.prefix is None:
            raise OrganelleDependencyError(
                code="assembly.environment_unavailable",
                message="resolved provider has no executable prefix",
            )
        executables: list[PreparedExecutable] = []
        version: str | None = None
        for component in provider.components:
            if _hash_component(component.path) != component.sha256:
                raise OrganelleDependencyError(
                    code="assembly.environment_unavailable",
                    message=f"resolved provider component {component.role!r} changed after resolution",
                )
            if component.kind in ("executable", "host_provider"):
                executables.append(PreparedExecutable(name=component.role, path=component.path))
                if component.role == _provider_backend_id(provider):
                    version = component.version
        # A provider registers the backend's own executable under its role, the backend id;
        # adapters written for the managed path ask for it by file name
        # (get_organelle_from_reads.py, TIPPo.v2.4.pl, ptGAUL.sh), so it answers to that
        # name too unless another component already does.
        backend_id = _provider_backend_id(provider)
        names = {item.name for item in executables}
        for item in tuple(executables):
            if item.name == backend_id and item.path.name not in names:
                executables.append(PreparedExecutable(name=item.path.name, path=item.path))
                names.add(item.path.name)
        if version is None:
            version = next(
                (item.version for item in provider.components if item.version is not None), None
            )
        if not executables or version is None:
            raise OrganelleDependencyError(
                code="assembly.environment_unavailable",
                message="resolved provider has no executable with a verified version",
            )
        software_version = None
        if _provider_backend_id(provider) == "pmat":
            primary = next(item for item in executables if item.name == "pmat")
            banner = self._carrier.probe_version(primary.path, ("--version",))
            if PMAT_ORIENTATION_BANNER in banner:
                software_version = PMAT_ORIENTATION_VERSION
        return PreparedEnvironment(
            backend_id=_provider_backend_id(provider),
            carrier=provider.carrier,
            platform=provider.platform,
            digest=provider.provider_digest,
            prefix=provider.prefix,
            executables=tuple(executables),
            version=version,
            software_version=software_version,
        )

    def register_prepared_provider(
        self,
        environment: PreparedEnvironment,
        contract: BackendCapabilityContract,
        *,
        requested_source: EnvironmentSource,
        resolved_version: str,
        registry: InstallationRegistry | None = None,
    ) -> PreparedEnvironment:
        """Freeze an installed environment's actual component bytes and register it."""
        components: list[ProviderComponentIdentity] = []
        executable_paths = {item.name: item.path for item in environment.executables}
        for item in contract.items:
            path = next(
                (
                    executable_paths[name]
                    for name in (item.role, *item.safe_names)
                    if name in executable_paths
                ),
                None,
            )
            if path is None and item.kind == "host_provider":
                path = self._find_executable(item.safe_names)
            if path is None and item.kind == "resource":
                matches = {
                    candidate.resolve()
                    for name in item.safe_names
                    for candidate in (
                        environment.prefix / name,
                        environment.prefix / "share" / name,
                    )
                    if candidate.exists()
                }
                if len(matches) == 1:
                    path = matches.pop()
            if path is None:
                raise OrganelleDependencyError(
                    code="assembly.environment_unavailable",
                    message=f"installed provider is missing capability {item.role!r}",
                )
            _harden_managed_component(path, environment.prefix)
            components.append(
                ProviderComponentIdentity(
                    role=item.role,
                    kind=item.kind,
                    path=path.resolve(),
                    sha256=_hash_component(path),
                    version=resolved_version if item.role == contract.backend_id else None,
                )
            )
        contract_digest = (
            "sha256:"
            + hashlib.sha256(canonical_json_bytes(contract.model_dump(mode="json"))).hexdigest()
        )
        provider = ResolvedProvider(
            requested_source=requested_source,
            discovery_source="managed",
            carrier=environment.carrier,
            platform=environment.platform,
            prefix=environment.prefix.resolve(),
            capability_contract_digest=contract_digest,
            components=tuple(components),
        )
        (registry or InstallationRegistry(tool_root=self.tool_root)).register_verified(provider)
        return environment.model_copy(
            update={"digest": provider.provider_digest, "version": resolved_version}
        )

    def _verify_existing_environment(
        self,
        spec: AssemblyEnvironmentSpec,
        platform_spec: CondaPlatformSpec,
        prefix_root: Path,
        digest: str,
    ) -> PreparedEnvironment:
        if not prefix_root.exists():
            raise OrganelleDependencyError(
                code="assembly.environment_unavailable",
                message="required managed environment is not present in the cache",
                details={"digest": digest, "prefix": str(prefix_root)},
            )
        return self._verify_prefix(spec, platform_spec, prefix_root, digest)

    def _create_and_publish(
        self,
        spec: AssemblyEnvironmentSpec,
        platform_spec: CondaPlatformSpec,
        prefix_root: Path,
        digest: str,
    ) -> PreparedEnvironment:
        prefix_root.parent.mkdir(parents=True, exist_ok=True)
        temp_prefix = Path(
            tempfile.mkdtemp(
                prefix=f".{_digest_cache_key(digest)}.",
                suffix=".tmp",
                dir=str(prefix_root.parent),
            )
        )
        try:
            shutil.rmtree(temp_prefix)
            self._carrier.create_prefix(platform_spec.lock_resource, temp_prefix)
            if platform_spec.source_build is not None:
                self._materialize_source_build(platform_spec.source_build, temp_prefix)
            self._verify_prefix(spec, platform_spec, temp_prefix, digest)
            os.replace(temp_prefix, prefix_root)
            return self._verify_prefix(spec, platform_spec, prefix_root, digest)
        except BaseException:
            shutil.rmtree(temp_prefix, ignore_errors=True)
            raise

    def _verify_prefix(
        self,
        spec: AssemblyEnvironmentSpec,
        platform_spec: CondaPlatformSpec,
        prefix: Path,
        digest: str,
    ) -> PreparedEnvironment:
        executables: list[PreparedExecutable] = []
        missing: list[str] = []
        for name in platform_spec.executable_names:
            path = prefix / "bin" / name
            if not path.exists():
                missing.append(name)
            executables.append(PreparedExecutable(name=name, path=path))
        if missing:
            raise OrganelleDependencyError(
                code="assembly.environment_unavailable",
                message="managed environment probe failed: expected executables missing",
                details={"prefix": str(prefix), "digest": digest, "missing": missing},
            )
        if platform_spec.source_build is not None:
            self._verify_source_outputs(platform_spec.source_build, prefix)
        try:
            version = self._carrier.probe_version(executables[0].path, platform_spec.version_argv)
        except Exception as error:
            # NOVOPlasty prints its version banner and exits 2 when invoked
            # with an empty config filename. This is its declared probe,
            # not an assembly failure and not a replacement for a version.
            if (
                isinstance(error, OrganelleDependencyError)
                and error.details.get("returncode") in platform_spec.version_exit_codes
            ):
                version = str(error.details.get("stdout_tail", "")).strip()
            else:
                raise OrganelleDependencyError(
                    code="assembly.environment_unavailable",
                    message="managed environment probe failed: version probe error",
                    details={"prefix": str(prefix), "error": str(error)},
                ) from error
        if not version:
            raise OrganelleDependencyError(
                code="assembly.environment_unavailable",
                message="managed environment probe failed: empty version",
                details={"prefix": str(prefix)},
            )
        try:
            packages = self._carrier.probe_package_identity(prefix)
        except Exception as error:
            raise OrganelleDependencyError(
                code="assembly.environment_unavailable",
                message="managed environment probe failed: package identity error",
                details={"prefix": str(prefix), "error": str(error)},
            ) from error
        if platform_spec.source_build is None and not _package_record_present(
            packages, spec.backend_id
        ):
            raise OrganelleDependencyError(
                code="assembly.environment_unavailable",
                message="managed environment probe failed: backend package missing",
                details={"prefix": str(prefix), "backend_id": spec.backend_id},
            )
        return PreparedEnvironment(
            backend_id=spec.backend_id,
            carrier=spec.carrier,
            platform=platform_spec.platform,
            digest=digest,
            prefix=prefix,
            executables=tuple(executables),
            version=version,
            software_version=PMAT_ORIENTATION_VERSION
            if spec.backend_id == "pmat" and PMAT_ORIENTATION_BANNER in version
            else None,
        )

    def _require_host_commands(self, source: ManagedSourceBuildSpec) -> None:
        missing = [
            list(alternatives)
            for alternatives in source.required_host_commands
            if self._find_executable(alternatives) is None
        ]
        if missing:
            raise OrganelleDependencyError(
                code="assembly.environment_unavailable",
                message="managed source build is missing a required host capability",
                details={"required_alternatives": missing},
            )

    def _materialize_source_build(self, source: ManagedSourceBuildSpec, prefix: Path) -> None:
        archive = prefix / ".organelleverse-source.tar.gz"
        extracted = prefix / ".organelleverse-source"
        try:
            data = self._downloader.download(source.source_uri, archive, source.source_sha256)
            if hashlib.sha256(data).hexdigest() != source.source_sha256:
                raise OrganelleDependencyError(
                    code="assembly.environment_unavailable",
                    message="managed source archive hash does not match its frozen identity",
                )
            _safe_extract_tar(data, extracted)
            archive_root = extracted / source.archive_root
            if not archive_root.is_dir():
                raise OrganelleDependencyError(
                    code="assembly.environment_unavailable",
                    message="managed source archive has an unexpected root directory",
                    details={"archive_root": source.archive_root},
                )
            source_root = prefix / "share" / "pmat"
            source_root.parent.mkdir(parents=True, exist_ok=True)
            os.replace(archive_root, source_root)
            for patch in source.patches:
                patch_data = _read_resource(patch.resource)
                if hashlib.sha256(patch_data).hexdigest() != patch.sha256:
                    raise OrganelleDependencyError(
                        code="assembly.environment_unavailable",
                        message="managed source patch hash does not match its frozen identity",
                        details={"resource": patch.resource},
                    )
                patch_path = prefix / ".organelleverse-source.patch"
                patch_path.write_bytes(patch_data)
                try:
                    run_external(
                        ["patch", "--batch", "--forward", "--fuzz=0", "-p1", "-i", str(patch_path)],
                        cwd=source_root,
                        tool="patch",
                        code="assembly.environment_unavailable",
                        error_class=OrganelleDependencyError,
                    )
                finally:
                    patch_path.unlink(missing_ok=True)
            build_executable = shutil.which(
                source.build_argv[0],
                path=str(prefix / "bin") + os.pathsep + os.environ.get("PATH", ""),
            )
            if build_executable is None:
                raise OrganelleDependencyError(
                    code="assembly.environment_unavailable",
                    message=f"managed source build command {source.build_argv[0]!r} is unavailable",
                )
            env = dict(os.environ)
            env["PATH"] = str(prefix / "bin") + os.pathsep + env.get("PATH", "")
            env["CPATH"] = _prepend_path(prefix / "include", env.get("CPATH"))
            env["LIBRARY_PATH"] = _prepend_path(prefix / "lib", env.get("LIBRARY_PATH"))
            try:
                run_external(
                    [build_executable, *source.build_argv[1:]],
                    cwd=source_root,
                    env=env,
                    tool=source.build_argv[0],
                    code="assembly.environment_unavailable",
                    error_class=OrganelleDependencyError,
                )
            except OrganelleDependencyError as error:
                details = cast(Mapping[str, Any], error.details)
                raise OrganelleDependencyError(
                    code=error.code,
                    message=_message_with_stderr_tail("managed source build failed", details),
                    details=details,
                ) from error
            self._install_source_outputs(source, prefix)
        except OrganelleDependencyError:
            raise
        except (OSError, tarfile.TarError) as error:
            raise OrganelleDependencyError(
                code="assembly.environment_unavailable",
                message="managed source build failed",
                details=_source_build_error_details(error),
            ) from error
        finally:
            archive.unlink(missing_ok=True)
            shutil.rmtree(extracted, ignore_errors=True)

    @staticmethod
    def _install_source_outputs(source: ManagedSourceBuildSpec, prefix: Path) -> None:
        source_root = prefix / "share" / "pmat"
        bindir = prefix / "bin"
        bindir.mkdir(parents=True, exist_ok=True)
        for output in source.outputs:
            path = source_root / output.relative_path
            if not path.is_file() or path.stat().st_size == 0:
                raise OrganelleDependencyError(
                    code="assembly.environment_unavailable",
                    message=f"managed source build did not produce {output.relative_path}",
                )
            if output.kind == "executable":
                if not os.access(path, os.X_OK):
                    path.chmod(path.stat().st_mode | 0o500)
                assert output.install_name is not None
                relative_target = os.path.relpath(path, bindir)
                (bindir / output.install_name).symlink_to(relative_target)
        for alias in source.executable_aliases:
            target = bindir / alias.target
            if not target.is_file():
                raise OrganelleDependencyError(
                    code="assembly.environment_unavailable",
                    message=f"managed executable alias target {alias.target!r} is missing",
                )
            (bindir / alias.name).symlink_to(target.name)

    @staticmethod
    def _verify_source_outputs(source: ManagedSourceBuildSpec, prefix: Path) -> None:
        source_root = prefix / "share" / "pmat"
        for output in source.outputs:
            path = source_root / output.relative_path
            if not path.is_file() or path.stat().st_size == 0:
                raise OrganelleDependencyError(
                    code="assembly.environment_unavailable",
                    message=f"managed source output {output.relative_path!r} is missing",
                )
            if output.kind == "executable" and not os.access(path, os.X_OK):
                raise OrganelleDependencyError(
                    code="assembly.environment_unavailable",
                    message=f"managed source output {output.relative_path!r} is not executable",
                )

    # ------------------------------------------------------------------
    # Profile materialization.
    # ------------------------------------------------------------------

    def prepare_profile(
        self,
        spec: AssemblyEnvironmentSpec,
        *,
        organelle: Literal["mitochondrion", "plastid"],
        policy: Literal["ensure", "require"],
        override: ArtifactRef | None = None,
        environment: PreparedEnvironment | None = None,
    ) -> PreparedProfile:
        database = spec.require_database()
        target_name = _profile_name(organelle)
        if override is not None:
            actual = _hash_artifact_ref(override)
            if actual != override.sha256:
                raise OrganelleDependencyError(
                    code="assembly.environment_unavailable",
                    message="user-supplied profile ArtifactRef hash no longer matches its content",
                    details={"declared_sha256": override.sha256, "actual_sha256": actual},
                )
            resolved_environment = environment or self.prepare(spec, policy=policy)
            expected_environment_digest = self.expected_environment_digest(
                spec, platform=resolved_environment.platform
            )
            if resolved_environment.digest != expected_environment_digest:
                raise OrganelleDependencyError(
                    code="assembly.environment_unavailable",
                    message="profile compiler environment does not match the managed lock",
                )
            component = self.expected_profile_identity(spec, organelle=organelle, override=override)
            profile_path = self._prepare_user_profile_bundle(
                override,
                component,
                resolved_environment,
                policy=policy,
            )
            artifact = _verified_artifact(profile_path, "hmm_profile", "fam", "text/plain")
            return PreparedProfile(
                source="user_supplied",
                target=organelle,
                path=profile_path,
                artifact=artifact,
                component=component,
            )

        component = self.expected_profile_identity(spec, organelle=organelle)
        db_digest = self._database_digest(database)
        db_root = (
            self.cache_root / "databases" / f"{database.database_id}-{database.version}" / db_digest
        )

        if policy == "require":
            target_path, artifact = _verify_managed_profile_bundle(db_root, database, organelle)
            return PreparedProfile(
                source="managed",
                target=organelle,
                path=target_path,
                artifact=artifact,
                component=component,
            )

        self.cache_root.mkdir(parents=True, exist_ok=True)
        locks_root = self.cache_root / "locks"
        locks_root.mkdir(parents=True, exist_ok=True)
        lock_path = locks_root / f"{db_digest}.lock"
        with _advisory_lock(lock_path):
            target_path = db_root / target_name
            if not db_root.exists():
                self._materialize_database(database, db_root)
            target_path, artifact = _verify_managed_profile_bundle(db_root, database, organelle)
            return PreparedProfile(
                source="managed",
                target=organelle,
                path=target_path,
                artifact=artifact,
                component=component,
            )

    def materialize_database_files(self, database: ManagedDatabaseSpec) -> Path:
        """Download and verify every managed database file into a content-addressed dir.

        Returns the directory holding the exact declared files. The directory is
        content-addressed by the database identity digest, so repeated calls reuse
        one verified cache. Reuse re-verifies every file name, hash, and size; a
        missing, extra, or tampered file re-materializes atomically. This is the
        shared primitive used by backends that initialize a working database from a
        managed archive (e.g. GetOrganelle); it owns no backend-specific layout.
        """
        digest = self._database_digest(database)
        cache_key = _digest_cache_key("sha256:" + digest)
        db_root = (
            self.cache_root / "databases" / f"{database.database_id}-{database.version}" / cache_key
        )
        self.cache_root.mkdir(parents=True, exist_ok=True)
        locks_root = self.cache_root / "locks"
        locks_root.mkdir(parents=True, exist_ok=True)
        with _advisory_lock(locks_root / f"db-{cache_key}.lock"):
            if db_root.exists() and self._verify_database_files(database, db_root):
                return db_root
            if db_root.exists():
                shutil.rmtree(db_root, ignore_errors=True)
            self._materialize_database(database, db_root)
            if not self._verify_database_files(database, db_root):
                shutil.rmtree(db_root, ignore_errors=True)
                raise OrganelleDependencyError(
                    code="assembly.environment_unavailable",
                    message="managed database files failed post-materialization verification",
                    details={"database_id": database.database_id, "version": database.version},
                )
        return db_root

    def safe_extract_tar(self, data: bytes, destination: Path) -> None:
        """Extract ``data`` (a tar archive) into a fresh ``destination``, fail-closed.

        Rejects symlinks, hardlinks, devices, absolute paths, and any member that
        escapes the destination root. ``destination`` must not exist beforehand.
        """
        _safe_extract_tar(data, destination)

    def _verify_database_files(self, database: ManagedDatabaseSpec, db_root: Path) -> bool:
        expected_names = {file_spec.name for file_spec in database.files}
        actual_names: set[str] = (
            {entry.name for entry in db_root.iterdir()} if db_root.is_dir() else set()
        )
        if actual_names != expected_names:
            return False
        for file_spec in database.files:
            candidate = db_root / file_spec.name
            if not candidate.is_file():
                return False
            digest = hashlib.sha256()
            with candidate.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != file_spec.sha256:
                return False
            if candidate.stat().st_size != file_spec.size_bytes:
                return False
        return True

    def _database_digest(self, database: ManagedDatabaseSpec) -> str:
        payload = {
            "database_id": database.database_id,
            "version": database.version,
            "source_commit": database.source_commit,
            "files": [
                {"name": item.name, "sha256": item.sha256}
                for item in sorted(database.files, key=lambda x: x.name)
            ],
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def _prepare_user_profile_bundle(
        self,
        override: ArtifactRef,
        component: AssemblyComponentIdentity,
        environment: PreparedEnvironment,
        *,
        policy: Literal["ensure", "require"],
    ) -> Path:
        assert component.sha256 is not None
        bundle_root = self.cache_root / "profiles" / "user" / component.sha256
        if policy == "require":
            return _verify_user_profile_bundle(
                bundle_root,
                raw_sha256=override.sha256,
                environment_digest=environment.digest,
            )

        self.cache_root.mkdir(parents=True, exist_ok=True)
        locks_root = self.cache_root / "locks"
        locks_root.mkdir(parents=True, exist_ok=True)
        with _advisory_lock(locks_root / f"user-profile-{component.sha256}.lock"):
            if bundle_root.exists():
                return _verify_user_profile_bundle(
                    bundle_root,
                    raw_sha256=override.sha256,
                    environment_digest=environment.digest,
                )
            bundle_root.parent.mkdir(parents=True, exist_ok=True)
            temporary = Path(
                tempfile.mkdtemp(
                    prefix=f".{component.sha256}.",
                    suffix=".tmp",
                    dir=str(bundle_root.parent),
                )
            )
            try:
                profile_path = temporary / "profile.fam"
                shutil.copyfile(Path(override.uri), profile_path)
                self._profile_compiler.press(
                    environment.require_executable("hmmpress"), profile_path
                )
                _write_user_profile_metadata(
                    temporary,
                    raw_sha256=override.sha256,
                    environment_digest=environment.digest,
                )
                os.replace(temporary, bundle_root)
            except BaseException:
                shutil.rmtree(temporary, ignore_errors=True)
                raise
            return _verify_user_profile_bundle(
                bundle_root,
                raw_sha256=override.sha256,
                environment_digest=environment.digest,
            )

    def _materialize_database(self, database: ManagedDatabaseSpec, db_root: Path) -> None:
        db_root.parent.mkdir(parents=True, exist_ok=True)
        temp_root = Path(
            tempfile.mkdtemp(prefix=".oatkdb.", suffix=".tmp", dir=str(db_root.parent))
        )
        try:
            for file_spec in database.files:
                destination = temp_root / file_spec.name
                data = self._downloader.download(file_spec.url, destination, file_spec.sha256)
                actual = hashlib.sha256(data).hexdigest()
                if actual != file_spec.sha256:
                    raise OrganelleDependencyError(
                        code="assembly.environment_unavailable",
                        message="managed database file hash does not match the manifest",
                        details={
                            "file": file_spec.name,
                            "expected_sha256": file_spec.sha256,
                            "actual_sha256": actual,
                        },
                    )
                if destination.stat().st_size != file_spec.size_bytes:
                    raise OrganelleDependencyError(
                        code="assembly.environment_unavailable",
                        message="managed database file size does not match the manifest",
                        details={"file": file_spec.name},
                    )
            os.replace(temp_root, db_root)
        except BaseException:
            shutil.rmtree(temp_root, ignore_errors=True)
            raise


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _profile_name(organelle: Literal["mitochondrion", "plastid"]) -> str:
    return "embryophyta_mito.fam" if organelle == "mitochondrion" else "embryophyta_pltd.fam"


def _profile_bundle_specs(
    database: ManagedDatabaseSpec,
    organelle: Literal["mitochondrion", "plastid"],
) -> tuple[ManagedFileSpec, ...]:
    name = _profile_name(organelle)
    return tuple(
        _database_file(database, candidate)
        for candidate in (name, *(f"{name}.{suffix}" for suffix in ("h3f", "h3i", "h3m", "h3p")))
    )


def _profile_bundle_digest(
    database: ManagedDatabaseSpec,
    organelle: Literal["mitochondrion", "plastid"],
) -> str:
    payload = {
        "database_id": database.database_id,
        "version": database.version,
        "source_commit": database.source_commit,
        "target": organelle,
        "files": [
            {"name": item.name, "sha256": item.sha256, "size_bytes": item.size_bytes}
            for item in _profile_bundle_specs(database, organelle)
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _user_profile_identity_digest(*, raw_sha256: str, environment_digest: str) -> str:
    payload = {
        "raw_profile_sha256": raw_sha256,
        "compiler_environment_digest": environment_digest,
        "compiler": "hmmpress",
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _user_profile_metadata_payload(
    root: Path,
    *,
    raw_sha256: str,
    environment_digest: str,
) -> dict[str, object]:
    files: list[dict[str, object]] = []
    for name in (
        "profile.fam",
        *(f"profile.fam.{suffix}" for suffix in ("h3f", "h3i", "h3m", "h3p")),
    ):
        path = root / name
        if not path.is_file():
            raise OrganelleDependencyError(
                code="assembly.environment_unavailable",
                message="compiled user HMM profile bundle is incomplete",
                details={"profile_file": name, "bundle_root": str(root)},
            )
        artifact = _verified_artifact(path, "hmm_profile", "fam", "application/octet-stream")
        files.append({"name": name, "sha256": artifact.sha256, "size_bytes": artifact.size_bytes})
    if files[0]["sha256"] != raw_sha256:
        raise OrganelleDependencyError(
            code="assembly.environment_unavailable",
            message="cached user HMM profile differs from the declared input",
            details={"declared_sha256": raw_sha256, "actual_sha256": files[0]["sha256"]},
        )
    return {
        "schema_version": "organelleverse.user-hmm-profile.v1",
        "raw_profile_sha256": raw_sha256,
        "compiler_environment_digest": environment_digest,
        "files": files,
    }


def _write_user_profile_metadata(
    root: Path,
    *,
    raw_sha256: str,
    environment_digest: str,
) -> None:
    payload = _user_profile_metadata_payload(
        root,
        raw_sha256=raw_sha256,
        environment_digest=environment_digest,
    )
    (root / "profile_bundle.json").write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    )


def _verify_user_profile_bundle(
    root: Path,
    *,
    raw_sha256: str,
    environment_digest: str,
) -> Path:
    metadata_path = root / "profile_bundle.json"
    if not metadata_path.is_file():
        raise OrganelleDependencyError(
            code="assembly.environment_unavailable",
            message="compiled user HMM profile is unavailable",
            details={"bundle_root": str(root)},
        )
    expected = _user_profile_metadata_payload(
        root,
        raw_sha256=raw_sha256,
        environment_digest=environment_digest,
    )
    try:
        recorded = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise OrganelleDependencyError(
            code="assembly.environment_unavailable",
            message="compiled user HMM profile metadata is invalid",
            details={"metadata": str(metadata_path)},
        ) from error
    if recorded != expected:
        raise OrganelleDependencyError(
            code="assembly.environment_unavailable",
            message="compiled user HMM profile metadata does not match its files",
            details={"metadata": str(metadata_path)},
        )
    return root / "profile.fam"


def _verify_managed_profile_bundle(
    root: Path,
    database: ManagedDatabaseSpec,
    organelle: Literal["mitochondrion", "plastid"],
) -> tuple[Path, ArtifactRef]:
    specs = _profile_bundle_specs(database, organelle)
    primary: ArtifactRef | None = None
    for item in specs:
        path = root / item.name
        if not path.is_file():
            raise OrganelleDependencyError(
                code="assembly.environment_unavailable",
                message="managed HMM profile bundle is incomplete",
                details={"profile_file": item.name, "database_root": str(root)},
            )
        artifact = _verified_artifact(path, "hmm_profile", "fam", "text/plain")
        if artifact.sha256 != item.sha256 or artifact.size_bytes != item.size_bytes:
            raise OrganelleDependencyError(
                code="assembly.environment_unavailable",
                message="managed HMM profile bundle does not match its manifest",
                details={
                    "profile_file": item.name,
                    "declared_sha256": item.sha256,
                    "actual_sha256": artifact.sha256,
                },
            )
        if item is specs[0]:
            primary = artifact
    assert primary is not None
    return root / specs[0].name, primary


def _hash_artifact_ref(artifact: ArtifactRef) -> str:
    path = Path(artifact.uri)
    if not path.is_file():
        raise OrganelleDependencyError(
            code="assembly.environment_unavailable",
            message="user-supplied profile ArtifactRef does not resolve to an existing file",
            details={"uri": artifact.uri},
        )
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_artifact(path: Path, kind: str, format: str, media_type: str) -> ArtifactRef:
    return ArtifactRef.from_path(path, kind=kind, format=format, media_type=media_type)


def _read_resource(resource_name: str) -> bytes:
    resource = _resolve_resource(resource_name)
    return resource.read_bytes()


def _digest_cache_key(digest: str) -> str:
    """Return a PATH-safe directory name while preserving the semantic digest."""
    algorithm, separator, value = digest.partition(":")
    if algorithm != "sha256" or separator != ":" or len(value) != 64:
        raise ValueError("managed environment digest must be a SHA256 content ID")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError("managed environment digest must use lowercase hexadecimal")
    return value


def _resolve_resource(resource_name: str) -> Traversable:
    # resource_name is a dotted package plus a slash-separated relative path, e.g.
    # "organelleverse.assembly.resources.environments.oatk/linux-64.explicit.txt".
    if "/" in resource_name:
        package, relative = resource_name.split("/", 1)
    else:
        package, relative = resource_name.rsplit(".", 1)[0], resource_name.rsplit(".", 1)[1]
    resource = importlib_resources.files(package)
    for segment in relative.split("/"):
        resource = resource.joinpath(segment)
    return resource


def _safe_extract_tar(data: bytes, destination: Path) -> None:
    """Extract a source archive after rejecting links, devices, and path escapes."""
    destination.mkdir(parents=True, exist_ok=False)
    root = destination.resolve()
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as archive:
            members = archive.getmembers()
            for member in members:
                member_path = Path(member.name)
                if (
                    member_path.is_absolute()
                    or any(part in {"", ".", ".."} for part in member_path.parts)
                    or member.issym()
                    or member.islnk()
                    or member.isdev()
                ):
                    raise OrganelleDependencyError(
                        code="assembly.environment_unavailable",
                        message=f"unsafe member in managed source archive: {member.name!r}",
                    )
                target = (root / member_path).resolve(strict=False)
                try:
                    target.relative_to(root)
                except ValueError:
                    raise OrganelleDependencyError(
                        code="assembly.environment_unavailable",
                        message=f"managed source archive path escapes its root: {member.name!r}",
                    ) from None
            archive.extractall(root, members=members, filter="data")
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise


def _find_first_executable(names: tuple[str, ...]) -> Path | None:
    for name in names:
        found = shutil.which(name)
        if found is not None:
            return Path(found)
    return None


def _prepend_path(path: Path, current: str | None) -> str:
    return str(path) if not current else str(path) + os.pathsep + current


def _source_build_error_details(error: BaseException) -> dict[str, str]:
    return {"error": str(error)}


def _message_with_stderr_tail(prefix: str, details: Mapping[str, object]) -> str:
    """Restore a call site's message prefix, appending the captured stderr tail."""
    tail = tail_lines(str(details.get("stderr_tail", "")), MESSAGE_TAIL_LINES)
    return f"{prefix}: {tail}" if tail else prefix


def _provider_backend_id(provider: ResolvedProvider) -> str:
    for component in provider.components:
        if component.kind == "executable":
            return component.role
    raise OrganelleDependencyError(
        code="assembly.environment_unavailable",
        message="resolved provider has no backend executable",
    )


def _harden_managed_component(path: Path, prefix: Path) -> None:
    """Remove group/other write access from a managed capability path."""
    resolved_path = path.resolve()
    resolved_prefix = prefix.resolve()
    try:
        resolved_path.relative_to(resolved_prefix)
    except ValueError:
        return
    candidates = [resolved_path]
    candidate = resolved_path.parent
    while True:
        candidates.append(candidate)
        if candidate == resolved_prefix:
            break
        candidate = candidate.parent
    for candidate in candidates:
        metadata = candidate.stat()
        if metadata.st_uid != os.getuid():
            raise OrganelleDependencyError(
                code="assembly.environment_unavailable",
                message=f"managed provider path is not owned by the current user: {candidate}",
            )
        mode = metadata.st_mode & 0o7777
        hardened = mode & ~0o022
        if hardened != mode:
            candidate.chmod(hardened)


def _hash_component(path: Path) -> str:
    if path.is_file():
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    if path.is_dir():
        digest = hashlib.sha256()
        for child in sorted(item for item in path.rglob("*") if item.is_file()):
            digest.update(child.relative_to(path).as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(_hash_component(child).encode("ascii"))
            digest.update(b"\0")
        return digest.hexdigest()
    raise OrganelleDependencyError(
        code="assembly.environment_unavailable",
        message=f"provider component is missing: {path}",
    )


def _materialize_lock(lock_resource: str, parent: Path) -> Path:
    resource = _resolve_resource(lock_resource)
    data = resource.read_bytes()
    name = lock_resource.rsplit("/", 1)[-1]
    lock_path = parent / f".{name}"
    lock_path.write_bytes(data)
    return lock_path


def _locate_carrier() -> str:
    for candidate in ("micromamba", "conda", "mamba"):
        path = shutil.which(candidate)
        if path is not None:
            return path
    raise OrganelleDependencyError(
        code="assembly.environment_unavailable",
        message="no supported Conda carrier (micromamba/conda/mamba) is available",
    )


class _advisory_lock:
    def __init__(self, lock_path: Path) -> None:
        self._lock_path = lock_path
        self._handle: TextIO | None = None

    def __enter__(self) -> _advisory_lock:
        lock_path = self._lock_path
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = lock_path.open("a+")
        fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc: object) -> None:
        if self._handle is not None:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            self._handle.close()


# ---------------------------------------------------------------------------
# Managed database file lookup helpers.
# ---------------------------------------------------------------------------


def _database_file(database: ManagedDatabaseSpec, name: str) -> ManagedFileSpec:
    for item in database.files:
        if item.name == name:
            return item
    raise OrganelleDependencyError(
        code="assembly.environment_unavailable",
        message=f"managed database has no file named {name!r}",
    )


def _package_record_present(packages: tuple[tuple[str, str, str], ...], backend_id: str) -> bool:
    return any(record[0] == backend_id for record in packages)
