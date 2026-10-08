"""Read-only deterministic environment provider resolver.

Discovers and verifies environment providers in a fixed priority order
(agent_hint → registry → active_conda → path → managed) without side-effects.
Discovery itself performs zero subprocess calls and never writes a trust record.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import stat as stat_module
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, Protocol, cast

from organelleverse.assembly.environment_contracts import (
    BackendCapabilityContract,
    CapabilityItem,
    EnvironmentHint,
    EnvironmentResolution,
    EnvironmentSource,
    ManagedProviderPlan,
    ProviderComponentIdentity,
    ProviderRejection,
    ResolvedBackendVersion,
    ResolvedProvider,
    _DiscoverySource,  # type: ignore[reportPrivateUsage]
    canonical_json_bytes,
)
from organelleverse.core.data import OrganelleData
from organelleverse.core.errors import OrganelleDependencyError


class EnvironmentRequest(Protocol):
    """Minimum request surface needed for provider discovery."""

    data: OrganelleData
    environment_source: EnvironmentSource
    environment_hint: EnvironmentHint | None


class _RegistryComponentPaths(dict[str, Path]):
    """Explicit hash-bound registry paths plus their recorded versions."""

    def __init__(
        self,
        paths: Mapping[str, Path],
        versions: Mapping[str, str | None],
    ) -> None:
        super().__init__(paths)
        self.versions = dict(versions)


# ---------------------------------------------------------------------------
# Default injectable implementations
# ---------------------------------------------------------------------------


def _default_hash_path(path: Path) -> str:
    """Compute SHA256 hex digest of a file or directory.

    Directories are hashed recursively with stable ordering: each entry
    contributes its relative POSIX path, type marker (F/D), and content
    hash (file) or recursive hash (subdirectory). Symlinks that escape
    the directory tree are rejected.
    """
    if path.is_file():
        h = hashlib.sha256()
        with open(str(path), "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    if not path.is_dir():
        raise ValueError(f"cannot hash {path}: not a file or directory")

    resolved_root = Path(os.path.realpath(str(path)))
    entries: list[tuple[str, str, str]] = []  # (relpath, type, content_hash)
    for entry_path in sorted(resolved_root.rglob("*"), key=lambda p: str(p)):
        rel = str(entry_path.relative_to(resolved_root))
        # Normalize to POSIX separators for stability
        rel_posix = rel.replace(os.sep, "/")
        if entry_path.is_symlink():
            real = Path(os.path.realpath(str(entry_path)))
            try:
                real.relative_to(resolved_root)
            except ValueError:
                raise ValueError(
                    f"symlink escape in directory resource: {entry_path} -> {real}"
                ) from None
        if entry_path.is_file():
            content_hash = _default_hash_path(entry_path)
            entries.append((rel_posix, "F", content_hash))
        elif entry_path.is_dir():
            content_hash = _default_hash_path(entry_path)
            entries.append((rel_posix, "D", content_hash))
        else:
            raise ValueError(f"unsupported file type in directory resource: {entry_path}")

    h = hashlib.sha256()
    for rel_posix, type_marker, content_hash in entries:
        h.update(rel_posix.encode("utf-8"))
        h.update(b"\x00")
        h.update(type_marker.encode("utf-8"))
        h.update(b"\x00")
        h.update(content_hash.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _default_trusted_runner(
    executable: Path,
    argv: tuple[str, ...],
    *,
    timeout: float = 5.0,
    max_output: int = 4096,
) -> tuple[int, str, str]:
    """Run fixed probe argv while retaining at most ``max_output + 1`` bytes.

    The child is killed and reaped on timeout or output overflow. Pipes are
    consumed incrementally; ``communicate()`` is intentionally not used because
    it buffers an untrusted amount of output before a caller can truncate it.
    """
    import selectors
    import subprocess
    import time as _time

    if max_output < 1:
        return (-1, "", "invalid output limit")
    if timeout <= 0:
        return (-1, "", "timeout")
    try:
        proc: subprocess.Popen[bytes] = subprocess.Popen(
            [str(executable), *argv],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            env={"HOME": "/nonexistent", "PATH": "/nonexistent"},
        )
    except OSError:
        return (-1, "", "trusted runner launch failed")

    deadline = _time.monotonic() + timeout
    stdout_buf = bytearray()
    stderr_buf = bytearray()
    sel: selectors.BaseSelector | None = None

    def _close_pipe(selector: selectors.BaseSelector, pipe: object) -> None:
        with contextlib.suppress(KeyError, ValueError):
            selector.unregister(cast(Any, pipe))
        close = getattr(pipe, "close", None)
        if callable(close):
            close()

    def _kill_close_reap(
        process: subprocess.Popen[bytes],
        selector: selectors.BaseSelector | None,
    ) -> None:
        if process.poll() is None:
            process.kill()
        if selector is not None:
            for key in tuple(selector.get_map().values()):
                _close_pipe(selector, key.fileobj)
        for pipe in (process.stdout, process.stderr):
            if pipe is not None and not pipe.closed:
                pipe.close()
        process.wait(timeout=5.0)

    def _failure(reason: str) -> tuple[int, str, str]:
        nonlocal sel
        _kill_close_reap(proc, sel)
        if sel is not None:
            sel.close()
            sel = None
        return (-1, "", reason)

    try:
        sel = selectors.DefaultSelector()
        for pipe, buf_name in [
            (proc.stdout, "stdout"),
            (proc.stderr, "stderr"),
        ]:
            if pipe is not None:
                os.set_blocking(pipe.fileno(), False)
                sel.register(pipe, selectors.EVENT_READ, data=buf_name)

        while sel.get_map():
            remaining = max(0.0, deadline - _time.monotonic())
            if remaining <= 0:
                return _failure("timeout")

            events = sel.select(timeout=min(remaining, 0.1))
            for key, _mask in events:
                fileobj = key.fileobj
                target = stdout_buf if key.data == "stdout" else stderr_buf
                read_limit = max_output + 1 - len(target)
                if read_limit <= 0:
                    return _failure("output limit exceeded")
                try:
                    raw = os.read(
                        fileobj.fileno(),  # type: ignore[union-attr]
                        min(64 * 1024, read_limit),
                    )
                except BlockingIOError:
                    continue
                except (OSError, ValueError):
                    _close_pipe(sel, fileobj)
                    continue

                if not raw:  # empty bytes → EOF
                    _close_pipe(sel, fileobj)
                    continue

                target.extend(raw)
                if len(target) > max_output:
                    return _failure("output limit exceeded")

        remaining = max(0.0, deadline - _time.monotonic())
        try:
            proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            return _failure("timeout")
        return (
            proc.returncode if proc.returncode is not None else -1,
            stdout_buf.decode("utf-8", errors="replace"),
            stderr_buf.decode("utf-8", errors="replace"),
        )
    except BaseException:
        if proc.poll() is None:
            _kill_close_reap(proc, sel)
        raise
    finally:
        if sel is not None:
            with contextlib.suppress(Exception):
                sel.close()


def _default_find_in_path(executable_name: str) -> list[Path]:
    """Find all occurrences of *executable_name* on ``$PATH``.

    Zero subprocess — only reads ``os.environ`` and calls ``os.access``.
    Empty PATH elements and current directory ``.`` are ignored.
    """
    path_var: str = os.environ.get("PATH", "")
    found: list[Path] = []
    for dir_entry in path_var.split(os.pathsep):
        stripped = dir_entry.strip()
        if not stripped or stripped == ".":
            continue
        dir_path = Path(stripped)
        if not dir_path.is_dir():
            continue
        candidate = dir_path / executable_name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            found.append(candidate)
    return found


def _default_find_conda() -> list[Path]:
    """Read the active conda environment from ``CONDA_PREFIX``.

    Zero subprocess — only reads ``os.environ``.  Does NOT scan home directories.
    """
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return []
    bin_dir = Path(conda_prefix) / "bin"
    if bin_dir.is_dir():
        return [bin_dir]
    return []


class _DefaultProbeSandbox:
    """Run fixed version probes in a read-only, networkless bubblewrap sandbox."""

    read_only = True
    network_disabled = True

    def __init__(self) -> None:
        self._bwrap = shutil.which("bwrap")

    @property
    def available(self) -> bool:
        return self._bwrap is not None

    def __call__(
        self,
        executable: Path,
        argv: tuple[str, ...],
        *,
        timeout: float,
        max_output: int,
    ) -> tuple[int, str, str]:
        if self._bwrap is None:
            return (-1, "", "bubblewrap is unavailable")
        command = (
            "--ro-bind",
            "/",
            "/",
            "--dev",
            "/dev",
            "--proc",
            "/proc",
            "--tmpfs",
            "/tmp",
            "--unshare-net",
            "--die-with-parent",
            "--new-session",
            "--chdir",
            "/tmp",
            "--setenv",
            "HOME",
            "/tmp",
            "--setenv",
            "PATH",
            "/usr/bin:/bin",
            str(executable),
            *argv,
        )
        return _default_trusted_runner(
            Path(self._bwrap), command, timeout=timeout, max_output=max_output
        )


def _default_registry_lookup(
    backend_id: str,
) -> dict[str, object] | None:
    """Return verified registry providers in the resolver's closed format."""
    from organelleverse.assembly.environment_registry import InstallationRegistry

    providers = InstallationRegistry().locate(backend_id)
    if not providers:
        return None
    return {
        "providers": [
            {
                "prefix": str(provider.prefix) if provider.prefix is not None else None,
                "components": {
                    component.role: {
                        "role": component.role,
                        "kind": component.kind,
                        "path": str(component.path),
                        "registered_sha256": component.sha256,
                        "version": component.version,
                    }
                    for component in provider.components
                },
            }
            for provider in providers
        ]
    }


def _default_managed_spec(backend_id: str) -> object | None:
    """Return the managed environment spec for *backend_id*, or ``None``."""
    try:
        from organelleverse.assembly.environment_specs import (
            GETORGANELLE_ENVIRONMENT,
            HIMT_ENVIRONMENT,
            OATK_ENVIRONMENT,
            PMAT_ENVIRONMENT,
        )
    except ImportError:
        return None
    _SPECS: dict[str, object] = {
        "oatk": OATK_ENVIRONMENT,
        "himt": HIMT_ENVIRONMENT,
        "getorganelle": GETORGANELLE_ENVIRONMENT,
        "pmat": PMAT_ENVIRONMENT,
    }
    return _SPECS.get(backend_id)


def _default_platform() -> str:
    """Normalize host platform to a Conda-style string."""
    import platform as _plat

    system = _plat.system().strip().lower()
    machine = _plat.machine().strip().lower()
    conda_system = "osx" if system == "darwin" else system
    if conda_system in {"linux", "osx"}:
        if machine in {"x86_64", "amd64"}:
            arch = "64"
        elif machine in {"aarch64", "arm64"}:
            arch = "aarch64" if conda_system == "linux" else "arm64"
        else:
            arch = machine
        return f"{conda_system}-{arch}"
    return f"{conda_system}-{machine}"


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------


class EnvironmentResolver:
    """Deterministic, read-only resolver for verified environment providers.

    Discovery proceeds in a fixed priority order: *agent_hint*, *registry*,
    *active_conda*, *path*, *managed*. The resolver never writes trust records
    and discovery itself performs zero subprocess calls. Verification of
    unknown executables is delegated to an injectable sandbox probe.
    """

    def __init__(self, **overrides: object) -> None:
        self._stat: Callable[..., os.stat_result] = overrides.get("_stat", os.stat)  # type: ignore[assignment]
        self._realpath: Callable[..., Path] = overrides.get("_realpath", _fs_realpath)  # type: ignore[assignment]
        self._access: Callable[..., bool] = overrides.get("_access", _fs_access)  # type: ignore[assignment]
        self._is_file: Callable[[Path], bool] = overrides.get("_is_file", _fs_is_file)  # type: ignore[assignment]
        self._is_dir: Callable[[Path], bool] = overrides.get("_is_dir", _fs_is_dir)  # type: ignore[assignment]
        self._exists: Callable[[Path], bool] = overrides.get("_exists", _fs_exists)  # type: ignore[assignment]
        self._getuid: Callable[[], int] = overrides.get("_getuid", os.getuid)  # type: ignore[assignment]
        self._hash_path: Callable[[Path], str] = overrides.get("_hash_path", _default_hash_path)  # type: ignore[assignment]
        self._trusted_runner: Callable[..., tuple[int, str, str]] = overrides.get(
            "_trusted_runner", _default_trusted_runner
        )  # type: ignore[assignment]
        if "_probe_sandbox" in overrides:
            self._probe_sandbox: object | None = overrides["_probe_sandbox"]
        else:
            default_sandbox = _DefaultProbeSandbox()
            self._probe_sandbox = default_sandbox if default_sandbox.available else None
        self._find_in_path: Callable[[str], list[Path]] = overrides.get(
            "_find_in_path", _default_find_in_path
        )  # type: ignore[assignment]
        self._find_conda: Callable[[], list[Path]] = overrides.get(
            "_find_conda", _default_find_conda
        )  # type: ignore[assignment]
        self._registry_lookup: Callable[[str], dict[str, object] | None] = overrides.get(
            "_registry_lookup", _default_registry_lookup
        )  # type: ignore[assignment]
        self._managed_spec: Callable[[str], object | None] = overrides.get(
            "_managed_spec", _default_managed_spec
        )  # type: ignore[assignment]
        self._default_platform: Callable[[], str] = overrides.get(
            "_default_platform", _default_platform
        )  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(
        self,
        *,
        request: EnvironmentRequest,
        capability_contract: BackendCapabilityContract,
        resolved_version: ResolvedBackendVersion,
        effective_parameters: Mapping[str, object],
        platform: str | None = None,
    ) -> EnvironmentResolution:
        """Resolve an environment provider for the given assembly request.

        Returns an :class:`EnvironmentResolution` with either a
        ``selected_provider`` or a ``managed_plan``, plus structured
        :class:`ProviderRejection` records for every rejected candidate.
        """
        resolved_platform = platform if platform is not None else self._default_platform()
        backend_id = capability_contract.backend_id

        # Validate resolved_version.backend_id matches capability contract.
        if resolved_version.backend_id != backend_id:
            raise ValueError(
                f"resolved_version.backend_id {resolved_version.backend_id!r} "
                f"does not match capability_contract.backend_id {backend_id!r}"
            )

        # Filter capability items by profile / effective parameters.
        effective_items = self._filter_items(
            capability_contract.items, request, effective_parameters
        )

        # managed source: skip all arbitrary discovery.
        if request.environment_source == "managed":
            return EnvironmentResolution(
                managed_plan=ManagedProviderPlan(
                    backend_id=backend_id,
                    carrier="conda",
                    platform=resolved_platform,
                ),
            )

        # Discover and verify candidates.
        rejections: list[ProviderRejection] = []
        for discovery_source, candidates in self._discover(
            request, backend_id, effective_items, resolved_platform
        ):
            # Tie-break: sort candidates by canonical prefix path lexicographically.
            sorted_candidates = sorted(candidates, key=lambda c: str(self._realpath(c[0])))
            for prefix, executable_map in sorted_candidates:
                result = self._verify_candidate(
                    prefix=prefix,
                    executable_map=executable_map,
                    effective_items=effective_items,
                    backend_id=backend_id,
                    request_source=request.environment_source,
                    discovery_source=discovery_source,
                    platform=resolved_platform,
                    request=request,
                    resolved_version=resolved_version,
                    capability_contract=capability_contract,
                )
                if isinstance(result, ResolvedProvider):
                    return EnvironmentResolution(
                        selected_provider=result,
                        rejections=tuple(rejections),
                    )
                rejections.append(result)

        # No verified candidate.
        if request.environment_source == "existing":
            raise OrganelleDependencyError(
                code="assembly.environment_unavailable",
                message=f"no verified {backend_id} environment found",
                details={
                    "backend_id": backend_id,
                    "rejections": [
                        {"reason_code": r.reason_code, "detail": r.detail} for r in rejections
                    ],
                },
            )

        # auto: fall back to managed plan.
        return EnvironmentResolution(
            managed_plan=ManagedProviderPlan(
                backend_id=backend_id,
                carrier="conda",
                platform=resolved_platform,
            ),
            rejections=tuple(rejections),
        )

    # ------------------------------------------------------------------
    # Capability filtering
    # ------------------------------------------------------------------

    def _filter_items(
        self,
        items: tuple[CapabilityItem, ...],
        request: EnvironmentRequest,
        effective_parameters: Mapping[str, object],
    ) -> list[CapabilityItem]:
        """Return capability items that are active for the current request.

        An item is active when:
        - It has no required_profiles, OR at least one profile matches.
        - It has no required_parameter, OR that parameter is present with
          a non-None value.
        """
        active_profiles = _active_profiles_for_request(request)
        result: list[CapabilityItem] = []
        for item in items:
            if not self._item_matches_profiles(item, active_profiles):
                continue
            if not self._item_matches_parameters(item, effective_parameters):
                continue
            result.append(item)
        return result

    @staticmethod
    def _item_matches_profiles(
        item: CapabilityItem,
        active_profiles: set[str],
    ) -> bool:
        if not item.required_profiles:
            return True
        item_profile_values = {p.value for p in item.required_profiles}
        return bool(active_profiles & item_profile_values)

    @staticmethod
    def _item_matches_parameters(
        item: CapabilityItem,
        effective_parameters: Mapping[str, object],
    ) -> bool:
        if item.required_parameter is None:
            return True
        value = effective_parameters.get(item.required_parameter)
        return value is not None

    # ------------------------------------------------------------------
    # Discovery sequence
    # ------------------------------------------------------------------

    def _discover(
        self,
        request: EnvironmentRequest,
        backend_id: str,
        effective_items: list[CapabilityItem],
        platform: str,
    ) -> Iterable[tuple[_DiscoverySource, list[tuple[Path, dict[str, Path]]]]]:
        """Yield (discovery_source, candidates) pairs in fixed priority order.

        Each candidate is a ``(prefix, executable_map)`` pair where
        *executable_map* maps capability role -> resolved executable path.
        """

        # 1. agent_hint
        if request.environment_hint is not None:
            hint = request.environment_hint
            hint_candidates = self._candidates_from_hint(hint, effective_items)
            if hint_candidates:
                yield ("agent_hint", hint_candidates)

        # 2. registry
        reg = self._registry_lookup(backend_id)
        if reg is not None:
            reg_candidates = self._candidates_from_registry(reg, effective_items)
            if reg_candidates:
                yield ("registry", reg_candidates)

        # 3. active_conda
        conda_candidates = self._candidates_from_conda(effective_items)
        if conda_candidates:
            yield ("active_conda", conda_candidates)

        # 4. path
        path_candidates = self._candidates_from_path(effective_items)
        if path_candidates:
            yield ("path", path_candidates)

        # 5. managed (cached)
        managed_candidates = self._candidates_from_managed(backend_id, effective_items, platform)
        if managed_candidates:
            yield ("managed", managed_candidates)

    # ------------------------------------------------------------------
    # Per-source candidate gathering
    # ------------------------------------------------------------------

    def _candidates_from_hint(
        self,
        hint: object,
        effective_items: list[CapabilityItem],
    ) -> list[tuple[Path, dict[str, Path]]]:
        """Build candidates from an environment hint.

        When *executable* is provided, the hint binds only the primary
        executable role whose safe_name matches the basename.  If zero
        or more than one executable role matches, the hint is rejected.
        """
        executable = getattr(hint, "executable", None)
        prefix = getattr(hint, "prefix", None)

        if executable is not None:
            exec_path = Path(str(executable))
            if not self._is_file(exec_path):
                return []
            basename = exec_path.name
            # Find executable roles whose safe_names include the basename.
            matching_roles: list[CapabilityItem] = [
                item
                for item in effective_items
                if item.kind in ("executable", "host_provider") and basename in item.safe_names
            ]
            # Must match exactly one primary executable role.
            if len(matching_roles) != 1:
                return []
            prefix_dir = exec_path.parent.parent
            exec_map: dict[str, Path] = {}
            # Map the matching role; resolve all other items from prefix.
            for item in effective_items:
                if item == matching_roles[0]:
                    exec_map[item.role] = exec_path
            # Resolve remaining items in prefix.
            for item in effective_items:
                if item.role not in exec_map:
                    if item.kind in ("executable", "host_provider"):
                        found = self._find_executable_in_prefix(prefix_dir, item.safe_names)
                        if found is not None:
                            exec_map[item.role] = found
                    elif item.kind == "resource":
                        found = self._find_resource_in_prefix(prefix_dir, item.safe_names)
                        if found is not None:
                            exec_map[item.role] = found
            if len(exec_map) == len(effective_items):
                return [(prefix_dir, exec_map)]
            return []

        if prefix is not None:
            prefix_dir = Path(str(prefix))
            exec_map = self._resolve_items_in_prefix(prefix_dir, effective_items)
            if exec_map:
                return [(prefix_dir, exec_map)]

        return []

    def _candidates_from_registry(
        self,
        reg_info: dict[str, object] | None,
        effective_items: list[CapabilityItem],
    ) -> list[tuple[Path, dict[str, Path]]]:
        """Build candidates from static registry information.

        Registry must return explicit registered provider/component paths
        and registered hashes.  Does NOT search PATH.  Each provider's
        declared components must match the **complete** expected role and
        kind set exactly — missing, extra, duplicate, or hash-drift entries
        cause rejection.
        """
        if reg_info is None:
            return []
        # Registry structure: {"providers": [{"prefix": str, "components": {...}}]}
        providers = reg_info.get("providers")
        if providers is None or not isinstance(providers, list):
            return []

        # Build the expected (role, kind) set from effective items.
        expected_set: set[tuple[str, str]] = {(item.role, item.kind) for item in effective_items}
        items_by_role = {item.role: item for item in effective_items}

        candidates: list[tuple[Path, dict[str, Path]]] = []
        for prov in cast(list[object], providers):
            if not isinstance(prov, dict):
                continue
            prov_d: dict[str, object] = cast(dict[str, object], prov)
            prefix_raw = prov_d.get("prefix")
            if prefix_raw is None:
                continue
            prefix = Path(str(prefix_raw))
            if not self._is_dir(prefix):
                continue

            # Verify registered components exactly match the expected set.
            components_reg = prov_d.get("components")
            if not isinstance(components_reg, dict):
                # No registered component info → cannot verify → skip.
                continue

            components_d: dict[str, object] = cast(dict[str, object], components_reg)

            declared_set: set[tuple[str, str]] = set()
            declared_roles: set[str] = set()
            exec_map: dict[str, Path] = {}
            registered_versions: dict[str, str | None] = {}
            valid = True
            for comp_info_obj in components_d.values():
                if not isinstance(comp_info_obj, dict):
                    valid = False
                    break
                comp_info = cast(dict[str, object], comp_info_obj)
                role = comp_info.get("role")
                kind = comp_info.get("kind")
                path_value = comp_info.get("path")
                registered_sha = comp_info.get("registered_sha256")
                if not all(
                    isinstance(value, str) for value in (role, kind, path_value, registered_sha)
                ):
                    valid = False
                    break
                role = cast(str, role)
                kind = cast(str, kind)
                path_value = cast(str, path_value)
                registered_sha = cast(str, registered_sha)
                if "version" in comp_info:
                    version = comp_info["version"]
                    if version is not None and not isinstance(version, str):
                        valid = False
                        break
                    registered_versions[role] = version
                item = items_by_role.get(role)
                if role in declared_roles or item is None or item.kind != kind:
                    valid = False
                    break
                declared_roles.add(role)
                declared_set.add((role, kind))
                registered_path = Path(path_value)
                canonical_path = self._realpath(registered_path)
                if not self._exists(registered_path) or not any(
                    tuple(canonical_path.parts[-len(Path(name).parts) :]) == Path(name).parts
                    for name in item.safe_names
                ):
                    valid = False
                    break
                trust_prefix = prefix
                if item.kind == "host_provider" and item.host_scope == "path":
                    try:
                        canonical_path.relative_to(self._realpath(prefix))
                    except ValueError:
                        trust_prefix = canonical_path.parent.parent
                if self._verify_trust(registered_path, trust_prefix, item) is not None:
                    valid = False
                    break
                try:
                    actual_hash = self._hash_path(registered_path)
                except (OSError, ValueError):
                    valid = False
                    break
                if actual_hash != registered_sha:
                    valid = False
                    break
                exec_map[role] = canonical_path
            if valid and declared_set == expected_set:
                candidates.append((prefix, _RegistryComponentPaths(exec_map, registered_versions)))

        seen: set[str] = set()
        unique: list[tuple[Path, dict[str, Path]]] = []
        for pref, emap in candidates:
            key = str(self._realpath(pref))
            if key not in seen:
                seen.add(key)
                unique.append((pref, emap))
        return unique

    def _candidates_from_conda(
        self,
        effective_items: list[CapabilityItem],
    ) -> list[tuple[Path, dict[str, Path]]]:
        """Build candidates from conda environment bin/ directories."""
        candidates: list[tuple[Path, dict[str, Path]]] = []
        for bin_dir in self._find_conda():
            prefix_dir = bin_dir.parent
            exec_map = self._resolve_items_in_prefix(prefix_dir, effective_items)
            if exec_map:
                candidates.append((prefix_dir, exec_map))
        return candidates

    def _candidates_from_path(
        self,
        effective_items: list[CapabilityItem],
    ) -> list[tuple[Path, dict[str, Path]]]:
        """Build candidates from ``$PATH`` executables."""
        candidates: list[tuple[Path, dict[str, Path]]] = []
        exec_items = [item for item in effective_items if item.kind == "executable"]

        if not exec_items:
            return []
        primary = exec_items[0]
        for name in primary.safe_names:
            for exec_path in self._find_in_path(name):
                prefix_dir = exec_path.parent.parent
                exec_map = self._resolve_items_in_prefix(prefix_dir, effective_items)
                if exec_map:
                    candidates.append((prefix_dir, exec_map))

        seen: set[str] = set()
        unique: list[tuple[Path, dict[str, Path]]] = []
        for prefix, exec_map in candidates:
            key = str(self._realpath(prefix))
            if key not in seen:
                seen.add(key)
                unique.append((prefix, exec_map))
        return unique

    def _candidates_from_managed(
        self,
        backend_id: str,
        effective_items: list[CapabilityItem],
        platform: str,
    ) -> list[tuple[Path, dict[str, Path]]]:
        """Check if a managed environment is already cached."""
        spec_raw = self._managed_spec(backend_id)
        if spec_raw is None:
            return []

        spec: Any = spec_raw  # AssemblyEnvironmentSpec at runtime
        try:
            platform_spec: Any = spec.require_platform(platform)  # type: ignore[attr-defined]
        except Exception:
            return []

        lock_resource = getattr(platform_spec, "lock_resource", None)
        if lock_resource is None:
            return []

        try:
            from importlib import resources as importlib_resources

            resource_name = str(lock_resource)
            package, relative = resource_name.split("/", 1)
            resource = importlib_resources.files(package)
            for segment in relative.split("/"):
                resource = resource.joinpath(segment)
            lock_bytes = resource.read_bytes()
        except Exception:
            return []

        lock_sha256 = hashlib.sha256(lock_bytes).hexdigest()
        payload: dict[str, object] = {
            "contract_version": str(getattr(spec, "contract_version", "")),
            "backend_id": backend_id,
            "carrier": "conda",
            "platform": platform,
            "package": str(getattr(platform_spec, "package", "")),
            "lock_sha256": lock_sha256,
            "executable_names": list(getattr(platform_spec, "executable_names", ())),
            "version_argv": list(getattr(platform_spec, "version_argv", ())),
        }
        if getattr(platform_spec, "source_build", None) is not None:
            payload["source_build"] = platform_spec.source_build.model_dump(mode="json")
        digest = (
            "sha256:"
            + hashlib.sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
        )
        cache_key = digest.replace("sha256:", "sha256-")
        if getattr(platform_spec, "source_build", None) is not None:
            from organelleverse.assembly.environment_registry import default_tool_root

            prefix_root = default_tool_root() / "providers" / backend_id / cache_key
        else:
            cache_root = Path(
                os.environ.get(
                    "ORGANELLEVERSE_CACHE_ROOT",
                    str(Path.home() / ".cache" / "organelleverse"),
                )
            )
            prefix_root = cache_root / "environments" / backend_id / cache_key
        if not self._is_dir(prefix_root):
            return []
        exec_map = self._resolve_items_in_prefix(prefix_root, effective_items)
        if exec_map:
            return [(prefix_root, exec_map)]
        return []

    # ------------------------------------------------------------------
    # Item resolution within a prefix
    # ------------------------------------------------------------------

    def _resolve_items_in_prefix(
        self,
        prefix: Path,
        effective_items: list[CapabilityItem],
    ) -> dict[str, Path]:
        """Map each capability role to a resolved path within *prefix*.

        Returns an empty dict if any required item cannot be resolved.
        host_provider items are resolved as executables (bin/ prefix, X_OK required).
        """
        result: dict[str, Path] = {}
        for item in effective_items:
            if item.kind == "executable":
                found = self._find_executable_in_prefix(prefix, item.safe_names)
                if found is None:
                    return {}
                result[item.role] = found
            elif item.kind == "host_provider":
                found = self._find_executable_in_prefix(prefix, item.safe_names)
                if found is None and item.host_scope == "path":
                    matches = {
                        self._realpath(path)
                        for name in item.safe_names
                        for path in self._find_in_path(name)
                    }
                    if len(matches) != 1:
                        return {}
                    found = matches.pop()
                if found is None:
                    return {}
                result[item.role] = found
            elif item.kind == "resource":
                found = self._find_resource_in_prefix(prefix, item.safe_names)
                if found is None:
                    return {}
                result[item.role] = found
        return result

    def _find_executable_in_prefix(
        self,
        prefix: Path,
        safe_names: tuple[str, ...],
    ) -> Path | None:
        """Find the executable from *safe_names* in *prefix*.

        Gathers every safe-name match, resolves each to its canonical
        real path, deduplicates, and returns the single result.  Returns
        ``None`` when zero files are found **or** when more than one
        *distinct* canonical path emerges (ambiguous).
        """
        candidates: set[Path] = set()
        for name in safe_names:
            for candidate in (prefix / "bin" / name, prefix / name):
                if self._is_file(candidate) and self._access(candidate, os.X_OK):
                    candidates.add(self._realpath(candidate))
        if len(candidates) != 1:
            return None
        return candidates.pop()

    def _find_resource_in_prefix(
        self,
        prefix: Path,
        safe_names: tuple[str, ...],
    ) -> Path | None:
        """Return the one canonical matching resource, rejecting ambiguity."""
        candidates: set[Path] = set()
        for name in safe_names:
            for candidate in (prefix / name, prefix / "share" / name):
                if self._exists(candidate):
                    candidates.add(self._realpath(candidate))
        if len(candidates) != 1:
            return None
        return candidates.pop()

    # ------------------------------------------------------------------
    # Candidate verification
    # ------------------------------------------------------------------

    def _verify_candidate(
        self,
        *,
        prefix: Path,
        executable_map: dict[str, Path],
        effective_items: list[CapabilityItem],
        backend_id: str,
        request_source: EnvironmentSource,
        discovery_source: _DiscoverySource,
        platform: str,
        request: EnvironmentRequest,
        resolved_version: ResolvedBackendVersion,
        capability_contract: BackendCapabilityContract | None = None,
    ) -> ResolvedProvider | ProviderRejection:
        """Verify all capability items in a candidate prefix.

        Returns a :class:`ResolvedProvider` if all items pass verification,
        or a :class:`ProviderRejection` with the first failure reason.

        Trust checks (ownership, permissions, symlink containment) run BEFORE
        any hash computation for security — we must never hash a file we
        haven't verified ownership of.
        """
        hint = request.environment_hint
        components: list[ProviderComponentIdentity] = []
        canonical_prefix = self._realpath(prefix)

        # Identify the primary backend item: the first executable whose
        # role matches backend_id, or fall back to the first executable.
        primary_item: CapabilityItem | None = None
        for item in effective_items:
            if item.kind in ("executable", "host_provider") and item.role == backend_id:
                primary_item = item
                break
        if primary_item is None:
            for item in effective_items:
                if item.kind == "executable":
                    primary_item = item
                    break

        # Require primary backend to declare version_argv.
        if primary_item is not None and not primary_item.version_argv:
            return ProviderRejection(
                backend_id=backend_id,
                reason_code="primary_version_argv_required",
                detail=(
                    f"primary backend {primary_item.role!r} must declare "
                    f"version_argv to enable version verification"
                ),
            )

        # ---- Phase 1: trust verification for all items (BEFORE any hashing) ----
        for item in effective_items:
            exec_path = executable_map.get(item.role)
            if exec_path is None:
                if item.kind in ("executable", "host_provider"):
                    found = self._find_executable_in_prefix(prefix, item.safe_names)
                elif item.kind == "resource":
                    found = self._find_resource_in_prefix(prefix, item.safe_names)
                else:
                    continue
                if found is None:
                    return ProviderRejection(
                        backend_id=backend_id,
                        reason_code="missing_item",
                        detail=f"cannot resolve {item.role} ({item.kind}) in {prefix}",
                    )
                exec_path = found

            # Verify trust (type, ownership, permissions, symlink containment).
            trust_prefix = prefix
            if item.kind == "host_provider" and item.host_scope == "path":
                try:
                    self._realpath(exec_path).relative_to(self._realpath(prefix))
                except ValueError:
                    trust_prefix = self._realpath(exec_path).parent.parent
            trust_result = self._verify_trust(exec_path, trust_prefix, item)
            if trust_result is not None:
                fixed = ProviderRejection(
                    backend_id=backend_id,
                    reason_code=trust_result.reason_code,
                    detail=trust_result.detail,
                )
                return fixed

        # ---- Phase 2: prefix-level expected_sha256 (after trust verified) ----
        # C10: expected_sha256 check for prefix-based hints (once, after trust)
        if (
            hint is not None
            and hint.expected_sha256 is not None
            and hint.executable is None
            and hint.prefix is not None
        ):
            hint_prefix_real = self._realpath(hint.prefix)
            if canonical_prefix == hint_prefix_real:
                prefix_hash = self._hash_path(canonical_prefix)
                if prefix_hash != hint.expected_sha256:
                    return ProviderRejection(
                        backend_id=backend_id,
                        reason_code="expected_hash_mismatch",
                        detail=(
                            f"prefix: expected sha256 {hint.expected_sha256}, got {prefix_hash}"
                        ),
                    )

        # ---- Phase 3: per-item hash, version, and executable-level checks ----
        for item in effective_items:
            exec_path = executable_map.get(item.role)
            if exec_path is None:
                # Must exist — we resolved it in Phase 1
                if item.kind in ("executable", "host_provider"):
                    found = self._find_executable_in_prefix(prefix, item.safe_names)
                elif item.kind == "resource":
                    found = self._find_resource_in_prefix(prefix, item.safe_names)
                else:
                    continue
                if found is None:
                    return ProviderRejection(
                        backend_id=backend_id,
                        reason_code="missing_item",
                        detail=f"cannot resolve {item.role} ({item.kind}) in {prefix}",
                    )
                exec_path = found

            canonical_exec_path = self._realpath(exec_path)

            # C10: expected_sha256 only checked on the hinted primary component
            if (
                hint is not None
                and hint.expected_sha256 is not None
                and hint.executable is not None
            ):
                hint_exec_real = self._realpath(hint.executable)
                if canonical_exec_path == hint_exec_real:
                    hint_file_hash = self._hash_path(exec_path)
                    if hint_file_hash != hint.expected_sha256:
                        return ProviderRejection(
                            backend_id=backend_id,
                            reason_code="expected_hash_mismatch",
                            detail=(
                                f"{item.role}: expected sha256 {hint.expected_sha256}, "
                                f"got {hint_file_hash}"
                            ),
                        )

            # Compute hash.
            file_hash = self._hash_path(exec_path)

            # Check against trusted hashes.
            if item.trusted_sha256 and file_hash not in item.trusted_sha256:
                return ProviderRejection(
                    backend_id=backend_id,
                    reason_code="hash_mismatch",
                    detail=f"{item.role} hash {file_hash} not in trusted set",
                )

            is_trusted = discovery_source == "registry" or bool(
                item.trusted_sha256 and file_hash in item.trusted_sha256
            )

            # Version detection: trusted items use trusted runner;
            # unknown items require ProbeSandbox.
            version: str | None = None
            stdout: str = ""
            stderr: str = ""
            registered_versions = (
                executable_map.versions
                if isinstance(executable_map, _RegistryComponentPaths)
                else {}
            )
            has_registered_version = (
                discovery_source == "registry" and item.role in registered_versions
            )
            if has_registered_version and item.required_version_text is None:
                version = registered_versions[item.role]
            elif item.version_argv:
                if is_trusted:
                    returncode, stdout, stderr = self._trusted_runner(exec_path, item.version_argv)
                else:
                    sandbox = self._probe_sandbox
                    if sandbox is None:
                        return ProviderRejection(
                            backend_id=backend_id,
                            reason_code="sandbox_required",
                            detail=f"{item.role}: sandbox required for unknown candidate",
                        )
                    if not (
                        getattr(sandbox, "read_only", False)
                        and getattr(sandbox, "network_disabled", False)
                    ):
                        return ProviderRejection(
                            backend_id=backend_id,
                            reason_code="sandbox_required",
                            detail=f"{item.role}: sandbox capabilities insufficient",
                        )
                    else:
                        try:
                            returncode, stdout, stderr = cast(
                                "tuple[int, str, str]",
                                sandbox(exec_path, item.version_argv, timeout=5.0, max_output=4096),  # type: ignore[operator]
                            )
                        except Exception:
                            returncode, stdout, stderr = (-1, "", "sandbox probe failed")

                if returncode not in item.version_exit_codes:
                    return ProviderRejection(
                        backend_id=backend_id,
                        reason_code="probe_failed",
                        detail=f"{item.role} probe exit {returncode}: {stderr.strip()[:200]}",
                    )
                combined_output = "\n".join((stdout, stderr))
                if (
                    item.required_version_text is not None
                    and item.required_version_text not in combined_output
                ):
                    return ProviderRejection(
                        backend_id=backend_id,
                        reason_code="version_mismatch",
                        detail=f"{item.role} requires {item.required_version_text!r}",
                    )
                if item.version_style == "strict_four_part":
                    version = _extract_strict_four_part(combined_output)
                else:
                    version = _extract_canonical_semver(combined_output)
                    if version is None and item.version_style == "major_minor":
                        version = _extract_major_minor(combined_output)
                if version is None:
                    return ProviderRejection(
                        backend_id=backend_id,
                        reason_code="version_unparseable",
                        detail=(
                            f"{item.role}: could not extract one stable canonical version "
                            f"from probe output {stdout[:200]!r} {stderr[:200]!r}"
                        ),
                    )

            if item is primary_item and item.version_argv and version is None:
                return ProviderRejection(
                    backend_id=backend_id,
                    reason_code="version_unparseable",
                    detail=f"{item.role}: registered primary version is missing",
                )

            if (
                version is not None
                and item is primary_item
                and backend_id == "pmat"
                and not version.startswith("2.")
            ):
                return ProviderRejection(
                    backend_id=backend_id,
                    reason_code="unsupported_version",
                    detail=f"PMAT requires a stable 2.x.y release, got {version!r}",
                )

            # Compare the resolved backend version only with the primary item.
            if version is not None and item is primary_item and version != resolved_version.version:
                return ProviderRejection(
                    backend_id=backend_id,
                    reason_code="version_mismatch",
                    detail=(
                        f"{item.role} version {version!r} != "
                        f"resolved_version {resolved_version.version!r}"
                    ),
                )

            # Check version specifier if present (strict, no substring fallback).
            if item.version_specifier is not None and version is not None:
                try:
                    if not self._version_matches(version, item.version_specifier):
                        return ProviderRejection(
                            backend_id=backend_id,
                            reason_code="version_mismatch",
                            detail=(
                                f"{item.role} version {version!r} "
                                f"does not match {item.version_specifier!r}"
                            ),
                        )
                except (ValueError, ImportError):
                    return ProviderRejection(
                        backend_id=backend_id,
                        reason_code="version_unparseable",
                        detail=(
                            f"{item.role} version {version!r} "
                            f"could not be parsed against {item.version_specifier!r}"
                        ),
                    )

            components.append(
                ProviderComponentIdentity(
                    role=item.role,
                    kind=item.kind,
                    path=canonical_exec_path,
                    sha256=file_hash,
                    version=version,
                )
            )

        # Compute capability contract digest from the canonical BackendCapabilityContract
        # declaration only (schema_version, backend_id, items).  Provider digest already
        # owns actual component identity; the contract digest must be stable regardless
        # of which specific binary resolved.
        if capability_contract is None:
            capability_contract = BackendCapabilityContract(
                schema_version="organelleverse.backend-capabilities.v1",
                backend_id=backend_id,
                items=tuple(effective_items),
            )
        contract_digest = self._capability_contract_digest(capability_contract)

        return ResolvedProvider(
            requested_source=request_source,
            discovery_source=discovery_source,
            carrier="conda",
            platform=platform,
            prefix=canonical_prefix,
            capability_contract_digest=contract_digest,
            components=tuple(components),
        )

    # ------------------------------------------------------------------
    # Trust verification
    # ------------------------------------------------------------------

    def _verify_trust(
        self,
        exec_path: Path,
        prefix: Path,
        item: CapabilityItem,
    ) -> ProviderRejection | None:
        """Verify file type, ownership, permissions, and symlink containment.

        Returns ``None`` on success, or a :class:`ProviderRejection`.
        host_provider items receive the same executable-level trust checks
        as regular executables.
        """
        is_exec_like = item.kind in ("executable", "host_provider")
        is_dir_resource = item.kind == "resource" and exec_path.is_dir()

        # Type check: executable / host_provider items must be regular files.
        if is_exec_like and not exec_path.is_file():
            return ProviderRejection(
                backend_id="",
                reason_code="not_a_file",
                detail=f"{item.role}: {exec_path} is not a regular file",
            )

        # Symlink containment: realpath must be inside prefix.
        real = self._realpath(exec_path)
        real_prefix = self._realpath(prefix)
        try:
            real.relative_to(real_prefix)
        except ValueError:
            return ProviderRejection(
                backend_id="",
                reason_code="symlink_escape",
                detail=f"{item.role}: realpath {real} is outside prefix {real_prefix}",
            )

        current_uid = self._getuid()

        # ---- Ancestor chain trust verification ----
        # Walk every ancestor directory from the candidate path up to and
        # including ``real_prefix``.  An intermediate symlink or writable
        # parent directory is a security risk equal to a writable binary.
        ancestor = real.parent
        while True:
            try:
                ancestor.relative_to(real_prefix)
            except ValueError:
                break  # walked past prefix boundary
            # Symlink containment for the ancestor itself.
            ancestor_real = self._realpath(ancestor)
            try:
                ancestor_real.relative_to(real_prefix)
            except ValueError:
                return ProviderRejection(
                    backend_id="",
                    reason_code="symlink_escape",
                    detail=(
                        f"{item.role}: ancestor {ancestor} realpath "
                        f"{ancestor_real} is outside prefix {real_prefix}"
                    ),
                )
            try:
                ancestor_st = self._stat(str(ancestor))
            except OSError:
                return ProviderRejection(
                    backend_id="",
                    reason_code="stat_failed",
                    detail=f"{item.role}: cannot stat ancestor {ancestor}",
                )
            if ancestor_st.st_uid != current_uid and ancestor_st.st_uid != 0:
                return ProviderRejection(
                    backend_id="",
                    reason_code="owner_mismatch",
                    detail=(
                        f"{item.role}: ancestor {ancestor} owner uid "
                        f"{ancestor_st.st_uid} is not current uid "
                        f"{current_uid} or root (0)"
                    ),
                )
            ancestor_mode = ancestor_st.st_mode
            if ancestor_mode & stat_module.S_IWOTH:
                return ProviderRejection(
                    backend_id="",
                    reason_code="unsafe_permissions",
                    detail=f"{item.role}: ancestor {ancestor} is world-writable",
                )
            if ancestor_mode & stat_module.S_IWGRP:
                return ProviderRejection(
                    backend_id="",
                    reason_code="unsafe_permissions",
                    detail=f"{item.role}: ancestor {ancestor} is group-writable",
                )
            if ancestor == real_prefix:
                break
            ancestor = ancestor.parent

        # Ownership, permissions, and type checks via stat for the target.
        try:
            st = self._stat(str(exec_path))
        except OSError:
            return ProviderRejection(
                backend_id="",
                reason_code="stat_failed",
                detail=f"{item.role}: cannot stat {exec_path}",
            )

        # C11: ownership — must be owned by current uid or root (uid 0).
        if st.st_uid != current_uid and st.st_uid != 0:
            return ProviderRejection(
                backend_id="",
                reason_code="owner_mismatch",
                detail=(
                    f"{item.role}: owner uid {st.st_uid} is not "
                    f"current uid {current_uid} or root (0)"
                ),
            )

        # Recursive trust for directories (resource items).
        if is_dir_resource:
            rec_result = self._verify_directory_trust(exec_path, prefix, item)
            if rec_result is not None:
                return rec_result

        mode = st.st_mode
        if mode & stat_module.S_IWOTH:
            return ProviderRejection(
                backend_id="",
                reason_code="unsafe_permissions",
                detail=f"{item.role}: {exec_path} is world-writable",
            )
        if is_exec_like and (mode & stat_module.S_IWGRP):
            return ProviderRejection(
                backend_id="",
                reason_code="unsafe_permissions",
                detail=f"{item.role}: {exec_path} is group-writable",
            )

        return None

    def _verify_directory_trust(
        self,
        dir_path: Path,
        prefix: Path,
        item: CapabilityItem,
    ) -> ProviderRejection | None:
        """Recursively verify all entries in a resource directory.

        Checks symlink containment, ownership, and unsafe permissions
        for every file and subdirectory.  Rejects on first violation.
        """
        resolved_root = self._realpath(dir_path)
        current_uid = self._getuid()
        for entry_path in sorted(resolved_root.rglob("*"), key=lambda p: str(p)):
            # Symlink containment check.
            if entry_path.is_symlink():
                real = self._realpath(entry_path)
                try:
                    real.relative_to(resolved_root)
                except ValueError:
                    return ProviderRejection(
                        backend_id="",
                        reason_code="symlink_escape",
                        detail=(
                            f"{item.role}: symlink {entry_path} -> {real} "
                            f"escapes resource directory {resolved_root}"
                        ),
                    )

            try:
                st = self._stat(str(entry_path))
            except OSError:
                return ProviderRejection(
                    backend_id="",
                    reason_code="stat_failed",
                    detail=f"{item.role}: cannot stat {entry_path}",
                )

            # Ownership.
            if st.st_uid != current_uid and st.st_uid != 0:
                return ProviderRejection(
                    backend_id="",
                    reason_code="owner_mismatch",
                    detail=(
                        f"{item.role}: {entry_path} owner uid {st.st_uid} "
                        f"is not current uid {current_uid} or root (0)"
                    ),
                )

            # Permissions: world-writable or group-writable rejected.
            mode = st.st_mode
            if mode & stat_module.S_IWOTH:
                return ProviderRejection(
                    backend_id="",
                    reason_code="unsafe_permissions",
                    detail=f"{item.role}: {entry_path} is world-writable",
                )
            if entry_path.is_file() and (mode & stat_module.S_IWGRP):
                return ProviderRejection(
                    backend_id="",
                    reason_code="unsafe_permissions",
                    detail=f"{item.role}: {entry_path} is group-writable",
                )

        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_contract_digest(
        items: list[CapabilityItem],
        *,
        schema_version: str,
        backend_id: str,
    ) -> str:
        """Hash the declaration exactly as its canonical model JSON."""
        payload = {
            "schema_version": schema_version,
            "backend_id": backend_id,
            "items": [item.model_dump(mode="json") for item in items],
        }
        return "sha256:" + hashlib.sha256(canonical_json_bytes(payload)).hexdigest()

    @staticmethod
    def _capability_contract_digest(contract: BackendCapabilityContract) -> str:
        return (
            "sha256:"
            + hashlib.sha256(canonical_json_bytes(contract.model_dump(mode="json"))).hexdigest()
        )

    @staticmethod
    def _version_matches(version: str, specifier: str) -> bool:
        """Check if *version* matches a PEP 440-style *specifier*.

        Uses packaging library for strict parsing.  No substring/string fallback.
        If packaging is unavailable or parsing fails, raises the error
        (fail closed) — the caller must handle this at a higher level.
        """
        from packaging.specifiers import SpecifierSet
        from packaging.version import Version

        parsed = Version(version)
        return SpecifierSet(specifier).contains(parsed)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _fs_realpath(path: Path) -> Path:
    return Path(os.path.realpath(str(path)))


def _fs_access(path: Path, mode: int) -> bool:
    return os.access(str(path), mode)


def _fs_is_file(path: Path) -> bool:
    return Path(str(path)).is_file()


def _fs_is_dir(path: Path) -> bool:
    return Path(str(path)).is_dir()


def _fs_exists(path: Path) -> bool:
    return Path(str(path)).exists()


# ---------------------------------------------------------------------------
# Canonical semver extraction
# ---------------------------------------------------------------------------

_SEMVER_RE = re.compile(
    r"(?<![0-9A-Za-z])v?"
    r"(?P<major>0|[1-9][0-9]*)\."
    r"(?P<minor>0|[1-9][0-9]*)\."
    r"(?P<patch>0|[1-9][0-9]*)"
    r"(?![0-9A-Za-z.+-])"
)
_MAJOR_MINOR_RE = re.compile(
    r"(?<![0-9A-Za-z.])v?(?P<major>0|[1-9][0-9]*)\."
    r"(?P<minor>0|[1-9][0-9]*)(?![0-9A-Za-z.+-])"
)
# Strict canonical four-part numeric release, e.g. GetOrganelle 1.7.7.1. The
# leading lookbehind also rejects a preceding dot so a five-component token like
# ``1.7.7.1.0`` cannot satisfy the probe via a shifted four-part window.
_STRICT_FOUR_PART_RE = re.compile(
    r"(?<![0-9A-Za-z.])v?"
    r"(?P<a>0|[1-9][0-9]*)\."
    r"(?P<b>0|[1-9][0-9]*)\."
    r"(?P<c>0|[1-9][0-9]*)\."
    r"(?P<d>0|[1-9][0-9]*)"
    r"(?![0-9A-Za-z.+-])"
)


def _extract_canonical_semver(stdout: str) -> str | None:
    """Extract exactly one unique stable canonical semver token from *stdout*.

    A display-only ``v`` prefix is accepted, as in ``PMAT v2.1.5``. Prerelease
    suffixes, build metadata, moving tags, leading-zero components, and
    ambiguous outputs fail closed.
    """
    if not stdout.strip():
        return None
    matches = _SEMVER_RE.findall(stdout)
    if not matches:
        return None
    versions = {
        f"{match.group('major')}.{match.group('minor')}.{match.group('patch')}"
        for match in _SEMVER_RE.finditer(stdout)
    }
    if len(versions) != 1:
        return None
    return versions.pop()


def _extract_strict_four_part(stdout: str) -> str | None:
    """Extract exactly one strict canonical four-part numeric release from *stdout*.

    Used by backends whose upstream version is a real four-component release, as
    in ``GetOrganelle v1.7.7.1`` -> ``1.7.7.1``. A display-only ``v`` prefix is
    accepted; three-part tokens, prerelease suffixes, build metadata, leading
    zeros, extra components, and ambiguous outputs fail closed (return ``None``).
    There is deliberately no fallback to three-part semver.
    """
    if not stdout.strip():
        return None
    versions = {
        f"{match.group('a')}.{match.group('b')}.{match.group('c')}.{match.group('d')}"
        for match in _STRICT_FOUR_PART_RE.finditer(stdout)
    }
    if len(versions) != 1:
        return None
    return versions.pop()


def _extract_major_minor(stdout: str) -> str | None:
    matches = tuple(_MAJOR_MINOR_RE.finditer(stdout))
    if len(matches) != 1:
        return None
    match = matches[0]
    return f"{match.group('major')}.{match.group('minor')}.0"


def _active_profiles_for_request(request: EnvironmentRequest) -> set[str]:
    """Determine which AssemblyProfile values are active for *request*.

    Reuses routing.classify_profile for canonical classification.
    Failures propagate — no silent fallback to empty set.
    """
    from organelleverse.assembly.routing import classify_profile

    if request.data.modality != "sequencing_reads":
        return set()
    profile = classify_profile(request.data)
    return {profile.value}
