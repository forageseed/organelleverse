"""The fixed first-party Linux Provider MCP protocol (spec §8, §9).

Six internal tools, exactly:

    ov.compute.probe      ov.compute.prepare     ov.compute.submit
    ov.compute.get        ov.compute.cancel      ov.compute.artifacts

They are **internal**: they never enter ``OperationRegistry.list()``,
``organelleverse.__all__``, an ``ov.Agent`` projection, or the public L7 MCP
catalog (spec §14). A generic shell MCP server cannot substitute for them
because generic shell cannot guarantee Registry admission, exact version
identity, durable run lifecycle, cooperative cancellation, content identities,
or L6 manifest equivalence (spec §3/§9/§14).

The transport is the open-source ``mcp`` SDK (decision C): this module uses
``mcp.server.lowlevel.Server`` and registers list/call handlers via the SDK's
decorator API — it does not hand-roll a wire protocol.

This slice delivers the wire contracts, ``operation_catalog_digest``, and the
server factory. The durable detached-worker process manager (real spawn /
fsync state / PID-tracked cancellation) is a substantial process subsystem that
arrives with the worker execution path; here an injectable
:class:`LinuxProviderBackend` seam lets tests drive the full six-tool surface
in-memory without a real worker.
"""

from __future__ import annotations

import hashlib
import json
from typing import Literal, Protocol

from pydantic import Field, JsonValue

from organelleverse.operations.spec import SideEffect, StrictSpecModel

__all__ = [
    "PROVIDER_PROTOCOL_VERSION",
    "PROVIDER_TOOLS",
    "ArtifactsToolResponse",
    "InvocationPolicyProjection",
    "LinuxProviderBackend",
    "PreparePolicyProjection",
    "PrepareToolRequest",
    "PrepareToolResponse",
    "PreparedComputeTarget",
    "ProbeToolRequest",
    "ProbeToolResponse",
    "ProviderRunHandle",
    "RequiredAssetIdentity",
    "RunStatus",
    "RunToolRequest",
    "RunToolResponse",
    "SubmitToolRequest",
    "SubmitToolResponse",
    "WorkerIdentity",
    "WorkerResourceFacts",
    "create_linux_provider_server",
    "main",
    "operation_catalog_digest",
]

PROVIDER_PROTOCOL_VERSION = "1.0"

_OP_ID = r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$"
_RUN_ID = r"^[a-z0-9][a-z0-9._-]{0,127}$"
_SHA256 = r"^sha256:[0-9a-f]{64}$"


# ---------------------------------------------------------------------------
# Worker identity + policy projections (spec §7.2, §8.1, §14)
# ---------------------------------------------------------------------------


class WorkerIdentity(StrictSpecModel):
    """Identity of the Linux worker inside the target (spec §7.2/§8.1).

    ``probe`` populates this with real distro/kernel/architecture/Python/
    OrganelleVerse facts. No environment variables, usernames, home/cache paths,
    SSH configuration, or secrets (spec §8.1).
    """

    linux_distribution: str
    kernel: str
    architecture: str
    python_version: str
    organelleverse_version: str
    provider_protocol_version: Literal["1.0"] = PROVIDER_PROTOCOL_VERSION


class WorkerResourceFacts(StrictSpecModel):
    """Bounded resource facts ``probe`` may return (spec §8.1)."""

    cpu_count: int = Field(ge=1)
    memory_bytes: int = Field(ge=0)
    disk_free_bytes: int = Field(ge=0)
    gpu_count: int = Field(default=0, ge=0)


class RequiredAssetIdentity(StrictSpecModel):
    """An exact asset the prepared environment must provide (spec §8.2)."""

    asset_id: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)


class PreparePolicyProjection(StrictSpecModel):
    """Immutable host grants projected to prepare (spec §8.2, §14).

    Package/network activity without the corresponding immutable grant fails
    before any activity.
    """

    install_granted: bool = False
    network_granted: bool = False


class InvocationPolicyProjection(StrictSpecModel):
    """Immutable side-effect grants projected to invocation (spec §14)."""

    side_effect_grants: frozenset[SideEffect] = frozenset()


# ---------------------------------------------------------------------------
# Run lifecycle (spec §8.3, §8.4)
# ---------------------------------------------------------------------------


class RunStatus(StrictSpecModel):
    """The stable provider run lifecycle (spec §8.4).

    ``cancel`` reports ``cancelled`` only after the worker acknowledges; a run
    stays nonterminal until then. ``get`` returns this after server/Windows
    restart.
    """

    status: Literal["queued", "running", "completed", "failed", "cancelled"]
    revision: int = Field(ge=1)
    result: dict[str, JsonValue] | None = None
    error: dict[str, JsonValue] | None = None


class ProviderRunHandle(StrictSpecModel):
    """A stable handle to a provider run (spec §8.3/§8.4)."""

    provider_run_id: str = Field(pattern=_RUN_ID)
    status: RunStatus


class PreparedComputeTarget(StrictSpecModel):
    """A target environment verified by ``prepare`` (spec §8.2)."""

    target_digest: str = Field(pattern=_SHA256)
    environment_digest: str = Field(pattern=_SHA256)
    capability_contract_digest: str = Field(pattern=_SHA256)


# ---------------------------------------------------------------------------
# The six tool request/response contracts (spec §8)
# ---------------------------------------------------------------------------


class ProbeToolRequest(StrictSpecModel):
    protocol_version: Literal["1.0"] = PROVIDER_PROTOCOL_VERSION


class ProbeToolResponse(StrictSpecModel):
    protocol_version: Literal["1.0"] = PROVIDER_PROTOCOL_VERSION
    worker_identity: WorkerIdentity
    worker_catalog_digest: str = Field(pattern=_SHA256)
    resources: WorkerResourceFacts
    artifact_transports: tuple[str, ...]


class PrepareToolRequest(StrictSpecModel):
    protocol_version: Literal["1.0"] = PROVIDER_PROTOCOL_VERSION
    operation_id: str = Field(pattern=_OP_ID)
    capability_contract_digest: str = Field(pattern=_SHA256)
    software_selector: str = Field(min_length=1)
    required_assets: tuple[RequiredAssetIdentity, ...] = ()
    policy: PreparePolicyProjection


class PrepareToolResponse(StrictSpecModel):
    protocol_version: Literal["1.0"] = PROVIDER_PROTOCOL_VERSION
    prepared_target: PreparedComputeTarget


class SubmitToolRequest(StrictSpecModel):
    protocol_version: Literal["1.0"] = PROVIDER_PROTOCOL_VERSION
    operation_id: str = Field(pattern=_OP_ID)
    catalog_digest: str = Field(pattern=_SHA256)
    input_snapshot: dict[str, JsonValue] | None
    parameters_snapshot: dict[str, JsonValue]
    staged_artifacts: tuple[
        object, ...
    ] = ()  # StagedArtifact (artifacts.py); typed loosely to avoid a cycle
    policy: InvocationPolicyProjection
    target_digest: str = Field(pattern=_SHA256)
    prepared_environment_digest: str = Field(pattern=_SHA256)


class SubmitToolResponse(StrictSpecModel):
    protocol_version: Literal["1.0"] = PROVIDER_PROTOCOL_VERSION
    run: ProviderRunHandle


class RunToolRequest(StrictSpecModel):
    protocol_version: Literal["1.0"] = PROVIDER_PROTOCOL_VERSION
    provider_run_id: str = Field(pattern=_RUN_ID)


class RunToolResponse(StrictSpecModel):
    protocol_version: Literal["1.0"] = PROVIDER_PROTOCOL_VERSION
    run: ProviderRunHandle


class ArtifactsToolResponse(StrictSpecModel):
    protocol_version: Literal["1.0"] = PROVIDER_PROTOCOL_VERSION
    manifest: object  # RemoteArtifactManifest (artifacts.py); typed loosely to avoid a cycle


# ---------------------------------------------------------------------------
# Backend seam (spec §8) — reused by WSL now, SSH/Slurm later
# ---------------------------------------------------------------------------


class LinuxProviderBackend(Protocol):
    """The six-tool execution seam. The server dispatches each tool here.

    A durable detached-worker implementation lives behind this seam; tests
    inject an in-memory backend to exercise the full six-tool surface and wire
    contracts without a real worker process.
    """

    def probe(self, request: ProbeToolRequest) -> ProbeToolResponse: ...
    def prepare(self, request: PrepareToolRequest) -> PrepareToolResponse: ...
    def submit(self, request: SubmitToolRequest) -> SubmitToolResponse: ...
    def get(self, request: RunToolRequest) -> RunToolResponse: ...
    def cancel(self, request: RunToolRequest) -> RunToolResponse: ...
    def artifacts(self, request: RunToolRequest) -> ArtifactsToolResponse: ...


# ---------------------------------------------------------------------------
# Tool table + catalog digest
# ---------------------------------------------------------------------------

#: The exact six tool names and their request model. Order is stable.
PROVIDER_TOOLS: tuple[tuple[str, type[StrictSpecModel]], ...] = (
    ("ov.compute.probe", ProbeToolRequest),
    ("ov.compute.prepare", PrepareToolRequest),
    ("ov.compute.submit", SubmitToolRequest),
    ("ov.compute.get", RunToolRequest),
    ("ov.compute.cancel", RunToolRequest),
    ("ov.compute.artifacts", RunToolRequest),
)


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def operation_catalog_digest(registry: object) -> str:
    """Canonical SHA256 of the protocol version + every admitted operation's
    full spec and invocation schema, sorted by operation id (spec §§8-9).

    This is the worker catalog identity the host compares at submit. It is not a
    substitute for provider trust; it detects catalog drift between the host
    that admitted the operation and the worker that runs it.
    """
    payload: dict[str, object] = {"protocol_version": PROVIDER_PROTOCOL_VERSION, "operations": {}}
    operations: dict[str, object] = {}
    for spec in registry.list():  # type: ignore[attr-defined]
        op_id = spec.operation_id  # type: ignore[attr-defined]
        operations[op_id] = {
            "spec": spec.model_dump(mode="json"),  # type: ignore[attr-defined]
            "invocation_schema": registry.invocation_schema(op_id),  # type: ignore[attr-defined]
        }
    payload["operations"] = operations
    return "sha256:" + hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


# ---------------------------------------------------------------------------
# MCP server factory (uses the open-source mcp SDK — decision C)
# ---------------------------------------------------------------------------


def create_linux_provider_server(*, registry: object, backend: LinuxProviderBackend) -> object:
    """Build the fixed six-tool MCP server over ``backend``.

    Uses ``mcp.server.lowlevel.Server`` (lazy import, so bare
    ``import organelleverse`` loads neither mcp nor this server) with the
    mcp 2.x constructor-callback API. Each tool revalidates its closed
    request model and dispatches to the backend; per the MCP tool contract,
    dispatch/validation failures come back as ``CallToolResult(is_error=...)`
    results rather than protocol-level errors.
    """
    from mcp.server.lowlevel.server import Server, ServerRequestContext
    from mcp.types import (
        CallToolRequestParams,
        CallToolResult,
        ListToolsResult,
        PaginatedRequestParams,
        TextContent,
        Tool,
    )

    async def _list_tools(
        ctx: ServerRequestContext[object], params: PaginatedRequestParams | None
    ) -> ListToolsResult:
        tools: list[Tool] = []
        for name, request_model in PROVIDER_TOOLS:
            schema = request_model.model_json_schema(mode="serialization")
            schema.setdefault("additionalProperties", False)
            tools.append(
                Tool(
                    name=name,
                    description=f"Internal OrganelleVerse compute tool {name}",
                    input_schema=schema,
                )
            )
        return ListToolsResult(tools=tools)

    async def _call_tool(ctx: object, params: CallToolRequestParams) -> CallToolResult:
        name = params.name
        arguments = params.arguments or {}
        try:
            dispatch = {
                "ov.compute.probe": (backend.probe, ProbeToolRequest, ProbeToolResponse),
                "ov.compute.prepare": (backend.prepare, PrepareToolRequest, PrepareToolResponse),
                "ov.compute.submit": (backend.submit, SubmitToolRequest, SubmitToolResponse),
                "ov.compute.get": (backend.get, RunToolRequest, RunToolResponse),
                "ov.compute.cancel": (backend.cancel, RunToolRequest, RunToolResponse),
                "ov.compute.artifacts": (backend.artifacts, RunToolRequest, ArtifactsToolResponse),
            }
            if name not in dispatch:
                raise ValueError(f"unknown compute tool: {name}")
            method, request_model, response_model = dispatch[name]
            # validate the closed request model at the wire boundary
            request = request_model.model_validate(dict(arguments))
            # the backend returns a StrictSpecModel; revalidate through the response
            # model so the wire boundary stays closed, then serialize as JSON content.
            # MCP tools carry structured results as JSON text content; the client
            # revalidates it through the same response model on receipt.
            response = response_model.model_validate(
                method(request).model_dump(mode="json", by_alias=True)
            )
            return CallToolResult(
                content=[TextContent(type="text", text=response.model_dump_json(by_alias=True))]
            )
        except Exception as exc:
            # Tool-level failures are is_error results, never protocol errors.
            return CallToolResult(content=[TextContent(type="text", text=str(exc))], is_error=True)

    return Server(
        "organelleverse-linux-provider",
        version=PROVIDER_PROTOCOL_VERSION,
        on_list_tools=_list_tools,
        on_call_tool=_call_tool,
    )


def main(argv: list[str] | None = None) -> int:
    """Console entry point: ``organelleverse-linux-provider mcp stdio``.

    Creates the default admitted registry and durable Linux backend, then serves
    the MCP protocol on stdio. Writes protocol bytes only to stdout and bounded
    redacted diagnostics only to stderr; exits nonzero on protocol setup failure.
    """
    import os
    import sys
    from pathlib import Path
    from typing import Any, cast

    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments != ["mcp", "stdio"]:
        print("usage: organelleverse-linux-provider mcp stdio", file=sys.stderr)
        return 2

    try:
        import anyio
        from mcp.server.stdio import stdio_server

        from organelleverse.compute.worker import DurableWorkerBackend
        from organelleverse.operations.registry import registry

        configured_root = os.environ.get("ORGANELLEVERSE_WORKER_RUN_ROOT")
        run_root = (
            Path(configured_root).expanduser()
            if configured_root
            else Path.home() / ".cache" / "organelleverse" / "compute-worker"
        )
        backend = DurableWorkerBackend(registry=registry, run_root=run_root)
        server = cast("Any", create_linux_provider_server(registry=registry, backend=backend))

        async def _serve() -> None:
            async with stdio_server() as (read_stream, write_stream):
                await server.run(
                    read_stream,
                    write_stream,
                    server.create_initialization_options(),
                )

        anyio.run(_serve)
    except KeyboardInterrupt:
        return 130
    except Exception:
        print("organelleverse Linux provider failed to serve stdio", file=sys.stderr)
        return 1
    return 0
