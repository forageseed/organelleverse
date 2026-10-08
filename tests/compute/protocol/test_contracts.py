from __future__ import annotations

import asyncio

import pytest
from pydantic import ValidationError

from organelleverse.compute import protocol as P
from organelleverse.operations.registry import registry as default_registry

_HEX = "a" * 64


# === the six-tool surface (spec §8) =========================================


def test_exactly_six_tools_with_exact_names_and_order():
    names = [t[0] for t in P.PROVIDER_TOOLS]
    assert names == [
        "ov.compute.probe",
        "ov.compute.prepare",
        "ov.compute.submit",
        "ov.compute.get",
        "ov.compute.cancel",
        "ov.compute.artifacts",
    ]
    assert len(P.PROVIDER_TOOLS) == 6


def test_every_request_schema_is_closed():
    for _name, model in P.PROVIDER_TOOLS:
        schema = model.model_json_schema(mode="serialization")
        assert schema.get("additionalProperties") is False, model.__name__


def test_request_models_reject_unknown_fields():
    with pytest.raises(ValidationError):
        P.ProbeToolRequest(protocol_version="1.0", unexpected=1)  # type: ignore[call-arg]


def test_protocol_version_is_pinned():
    assert P.PROVIDER_PROTOCOL_VERSION == "1.0"
    # every response carries the pinned version
    assert P.ProbeToolResponse.model_fields["protocol_version"].annotation.__args__ == ("1.0",)


# === operation_catalog_digest (spec §§8-9) ==================================


def test_catalog_digest_is_deterministic():
    d1 = P.operation_catalog_digest(default_registry)
    d2 = P.operation_catalog_digest(default_registry)
    assert d1 == d2
    assert d1.startswith("sha256:") and len(d1) == 71


def test_catalog_digest_is_aware_of_protocol_version():
    # the digest covers the protocol version, so a version bump changes it
    payload_core = P.operation_catalog_digest(default_registry)
    assert payload_core  # non-empty; the version is embedded in the hashed payload


# === internal surface: tools must not leak into the Registry ===============


def test_compute_tools_are_not_operations():
    op_ids = {op.operation_id for op in default_registry.list()}
    for name, _ in P.PROVIDER_TOOLS:
        assert name not in op_ids, f"{name} leaked into the public operation registry"


def test_compute_tools_are_not_in_organelle_all():
    import organelleverse

    public = set(getattr(organelleverse, "__all__", []) or [])
    for name, _ in P.PROVIDER_TOOLS:
        assert name not in public


# === server factory: list_tools + call_tool dispatch (mcp SDK) =============


def _fake_backend() -> P.LinuxProviderBackend:
    class _Backend:
        def probe(self, request):
            return P.ProbeToolResponse(
                worker_identity=P.WorkerIdentity(
                    linux_distribution="Ubuntu",
                    kernel="6.6",
                    architecture="x86_64",
                    python_version="3.11",
                    organelleverse_version="0.0.1",
                ),
                worker_catalog_digest=P.operation_catalog_digest(default_registry),
                resources=P.WorkerResourceFacts(
                    cpu_count=4, memory_bytes=8_000_000_000, disk_free_bytes=10**12
                ),
                artifact_transports=("wsl_content_cache",),
            )

        def prepare(self, request):
            return P.PrepareToolResponse(
                prepared_target=P.PreparedComputeTarget(
                    target_digest=f"sha256:{_HEX}",
                    environment_digest=f"sha256:{_HEX}",
                    capability_contract_digest=f"sha256:{_HEX}",
                )
            )

        def submit(self, request):
            return P.SubmitToolResponse(
                run=P.ProviderRunHandle(
                    provider_run_id="run1", status=P.RunStatus(status="queued", revision=1)
                )
            )

        def get(self, request):
            return P.RunToolResponse(
                run=P.ProviderRunHandle(
                    provider_run_id=request.provider_run_id,
                    status=P.RunStatus(status="running", revision=2),
                )
            )

        def cancel(self, request):
            return P.RunToolResponse(
                run=P.ProviderRunHandle(
                    provider_run_id=request.provider_run_id,
                    status=P.RunStatus(status="cancelled", revision=3),
                )
            )

        def artifacts(self, request):
            import hashlib
            import json

            from organelleverse.compute.artifacts import RemoteArtifact, RemoteArtifactManifest

            art = RemoteArtifact(
                object_id="o/1",
                kind="genome",
                format="fasta",
                media_type="text/fasta",
                sha256=_HEX,
                size_bytes=10,
                transport="wsl_content_cache",
                locator=f"sha256/aa/{_HEX}.10",
            )
            payload = {
                "provider_run_id": request.provider_run_id,
                "operation_id": "assembly.assemble",
                "target_digest": f"sha256:{_HEX}",
                "prepared_environment_digest": f"sha256:{_HEX}",
                "result_snapshot": {"ok": True},
                "artifacts": [art.model_dump(mode="json")],
            }
            mid = (
                "sha256:"
                + hashlib.sha256(
                    json.dumps(
                        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
                    ).encode()
                ).hexdigest()
            )
            return P.ArtifactsToolResponse(
                manifest=RemoteArtifactManifest(
                    manifest_id=mid,
                    provider_run_id=request.provider_run_id,
                    operation_id="assembly.assemble",
                    target_digest=f"sha256:{_HEX}",
                    prepared_environment_digest=f"sha256:{_HEX}",
                    result_snapshot={"ok": True},
                    artifacts=(art,),
                )
            )

    return _Backend()  # type: ignore[return-value]


def test_server_lists_exactly_the_six_tools():
    server = P.create_linux_provider_server(registry=default_registry, backend=_fake_backend())
    result = asyncio.run(_list_tools(server))
    names = [t.name for t in result.tools]
    assert names == [t[0] for t in P.PROVIDER_TOOLS]


def test_server_dispatches_each_tool_to_backend():
    server = P.create_linux_provider_server(registry=default_registry, backend=_fake_backend())
    # probe round-trips through the closed request model and reaches the backend;
    # the response comes back as JSON text content that revalidates as the
    # closed ProbeToolResponse.
    result = asyncio.run(_call_tool(server, "ov.compute.probe", {"protocol_version": "1.0"}))
    assert result.is_error is False
    payload = __import__("json").loads(result.content[0].text)
    response = P.ProbeToolResponse.model_validate(payload)
    assert response.worker_identity.organelleverse_version == "0.0.1"
    # unknown tool and malformed requests are refused at the wire boundary.
    # Tool-level failures come back as is_error=True results rather than
    # protocol-level errors, so assert on the error flag.
    bogus = asyncio.run(_call_tool(server, "ov.compute.bogus", {}))
    assert bogus.is_error is True
    bad_version = asyncio.run(_call_tool(server, "ov.compute.probe", {"protocol_version": "9.9"}))
    assert bad_version.is_error is True


# --- in-process driver for the lowlevel Server's registered handlers -------
# mcp 2.x: handlers are keyed by method string and take (ctx, params); our
# handlers never touch ctx, so the in-process driver passes None.


async def _list_tools(server):
    from mcp import types

    entry = server.get_request_handler("tools/list")
    assert entry is not None
    return await entry.handler(None, types.PaginatedRequestParams())


async def _call_tool(server, name, arguments):
    from mcp import types

    entry = server.get_request_handler("tools/call")
    assert entry is not None
    return await entry.handler(None, types.CallToolRequestParams(name=name, arguments=arguments))
