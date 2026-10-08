"""Resolve deterministic backend versions; network is opt-in via ``latest`` or a new exact pin."""

from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from typing import Protocol, cast

from organelleverse.assembly.environment_contracts import ResolvedBackendVersion
from organelleverse.assembly.environment_specs import (
    AssemblyEnvironmentSpec,
    CondaPlatformSpec,
)
from organelleverse.core.errors import OrganelleDependencyError

__all__ = ["ReleaseClient", "environment_spec_for_version", "resolve_backend_version"]

_PMAT_REPOSITORY = "https://github.com/aiPGAB/PMAT2"
_PMAT_API = "https://api.github.com/repos/aiPGAB/PMAT2"
_STABLE_PMAT = re.compile(r"^v?(2\.[0-9]+\.[0-9]+)$")


class ReleaseClient(Protocol):
    def release(self, selector: str) -> dict[str, object]: ...


_TESTED: dict[str, ResolvedBackendVersion] = {
    "novoplasty": ResolvedBackendVersion(
        backend_id="novoplasty", version="4.3.5", requested_selector="tested"
    ),
    "oatk": ResolvedBackendVersion(
        backend_id="oatk", version="1.0.0", requested_selector="tested", tag="v1.0"
    ),
    "himt": ResolvedBackendVersion(
        backend_id="himt", version="1.1.3", requested_selector="tested", tag="v1.1.3"
    ),
    # GetOrganelle's upstream version is the four-part 1.7.7.1
    # (``get_organelle_from_reads.py --version`` prints ``GetOrganelle v1.7.7.1``).
    # ``tested`` and the exact ``1.7.7.1`` pin resolve offline; every other
    # selector fails closed without network access.
    "getorganelle": ResolvedBackendVersion(
        backend_id="getorganelle", version="1.7.7.1", requested_selector="tested"
    ),
    "tippo": ResolvedBackendVersion(
        backend_id="tippo", version="2.4.0", requested_selector="tested"
    ),
    "ptgaul": ResolvedBackendVersion(
        backend_id="ptgaul", version="1.0.5", requested_selector="tested"
    ),
    # the crate version of ovasm; the run records the version the binary itself reports
    "ovasm": ResolvedBackendVersion(
        backend_id="ovasm", version="0.1.1", requested_selector="tested"
    ),
    "pmat": ResolvedBackendVersion(
        backend_id="pmat",
        version="2.1.5",
        requested_selector="tested",
        tag="v2.1.5",
        commit="04534a2adf0c5309cb2e7478fd2bcc98ced3818c",
        release_metadata_uri=f"{_PMAT_REPOSITORY}/releases/tag/v2.1.5",
        source_uri=f"{_PMAT_REPOSITORY}/archive/04534a2adf0c5309cb2e7478fd2bcc98ced3818c.tar.gz",
        source_sha256="5564becd242ab254e379758133b514ce0c38431df55c77bee74836f1914bf872",
    ),
}


def environment_spec_for_version(
    spec: AssemblyEnvironmentSpec,
    resolved: ResolvedBackendVersion,
) -> AssemblyEnvironmentSpec:
    """Bind a PMAT managed-source spec to one immutable resolved 2.x release."""
    if spec.backend_id != resolved.backend_id:
        raise ValueError("environment spec and resolved backend version do not match")
    if spec.backend_id != "pmat":
        return spec
    if resolved.source_uri is None or resolved.source_sha256 is None:
        raise OrganelleDependencyError(
            code="assembly.environment_unavailable",
            message="resolved PMAT version has no immutable source archive identity",
        )
    platforms: list[CondaPlatformSpec] = []
    for platform_spec in spec.platforms:
        source = platform_spec.source_build
        if source is None:
            raise OrganelleDependencyError(
                code="assembly.environment_unavailable",
                message="PMAT managed environment has no source-build contract",
            )
        platforms.append(
            platform_spec.model_copy(
                update={
                    "package": f"pmat-source={resolved.version}",
                    "source_build": source.model_copy(
                        update={
                            "source_uri": resolved.source_uri,
                            "source_sha256": resolved.source_sha256,
                            "archive_root": f"PMAT2-{resolved.commit}",
                            # This patch is reviewed only against the frozen 2.1.5 source.
                            "patches": source.patches
                            if resolved.commit == _TESTED["pmat"].commit
                            else (),
                        }
                    ),
                }
            )
        )
    return spec.model_copy(update={"platforms": tuple(platforms)})


def resolve_backend_version(
    backend_id: str,
    selector: str,
    *,
    release_client: ReleaseClient | None = None,
) -> ResolvedBackendVersion:
    tested = _TESTED.get(backend_id)
    if tested is None:
        raise OrganelleDependencyError(
            code="assembly.environment_unavailable",
            message=f"no version resolver is registered for {backend_id}",
        )
    if selector == "tested" or selector == tested.version:
        return tested.model_copy(update={"requested_selector": selector})
    if backend_id != "pmat":
        raise OrganelleDependencyError(
            code="assembly.environment_unavailable",
            message=f"{backend_id} currently supports backend_version='tested' only",
        )
    if selector != "latest" and _STABLE_PMAT.fullmatch(selector) is None:
        raise OrganelleDependencyError(
            code="assembly.environment_unavailable",
            message="PMAT backend_version must be latest or a stable 2.x.y release",
        )
    client = release_client if release_client is not None else _GitHubPmatReleaseClient()
    data = client.release(selector)
    tag = _required_string(data, "tag")
    match = _STABLE_PMAT.fullmatch(tag)
    if match is None:
        raise OrganelleDependencyError(
            code="assembly.environment_unavailable",
            message=f"PMAT release {tag!r} is not a stable 2.x.y tag",
        )
    version = match.group(1)
    if selector != "latest" and version != selector:
        raise OrganelleDependencyError(
            code="assembly.environment_unavailable",
            message=f"PMAT resolved {version}, not requested {selector}",
        )
    return ResolvedBackendVersion(
        backend_id="pmat",
        version=version,
        requested_selector=selector,
        tag=tag,
        commit=_required_string(data, "commit"),
        release_metadata_uri=_required_string(data, "release_metadata_uri"),
        source_uri=_required_string(data, "source_uri"),
        source_sha256=_required_string(data, "source_sha256"),
    )


class _GitHubPmatReleaseClient:
    def release(self, selector: str) -> dict[str, object]:
        if selector == "latest":
            metadata_uri = f"{_PMAT_API}/releases/latest"
            release = _read_json(metadata_uri)
            tag = _required_string(release, "tag_name")
        else:
            tag = f"v{selector}"
            metadata_uri = f"{_PMAT_REPOSITORY}/releases/tag/{tag}"
        ref = _read_json(f"{_PMAT_API}/git/ref/tags/{tag}")
        object_value = ref.get("object")
        if not isinstance(object_value, dict):
            raise _release_error("official PMAT tag response has no object")
        object_data = cast(dict[str, object], object_value)
        if object_data.get("type") != "commit":
            raise _release_error("official PMAT tag is not a direct immutable commit")
        commit = _required_string(object_data, "sha")
        source_uri = f"{_PMAT_REPOSITORY}/archive/{commit}.tar.gz"
        source = _read_bytes(source_uri)
        return {
            "tag": tag,
            "commit": commit,
            "release_metadata_uri": metadata_uri,
            "source_uri": source_uri,
            "source_sha256": hashlib.sha256(source).hexdigest(),
        }


def _read_json(uri: str) -> dict[str, object]:
    try:
        raw = json.loads(_read_bytes(uri))
    except (OSError, ValueError) as error:
        raise _release_error(f"could not read official PMAT release metadata: {error}") from error
    if not isinstance(raw, dict):
        raise _release_error("official PMAT release metadata is not an object")
    return cast(dict[str, object], raw)


def _read_bytes(uri: str) -> bytes:
    request = urllib.request.Request(uri, headers={"Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def _required_string(data: dict[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise _release_error(f"official PMAT release field {key!r} is missing")
    return value


def _release_error(message: str) -> OrganelleDependencyError:
    return OrganelleDependencyError(code="assembly.environment_unavailable", message=message)
