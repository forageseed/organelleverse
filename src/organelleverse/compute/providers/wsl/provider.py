"""Host-side MCP client for the first-party WSL compute provider."""

from __future__ import annotations

import asyncio
import json
import tempfile
from collections.abc import Mapping
from typing import TypeVar

from pydantic import ValidationError

from organelleverse.compute.artifacts import RemoteArtifactManifest
from organelleverse.compute.protocol import (
    ArtifactsToolResponse,
    PrepareToolRequest,
    PrepareToolResponse,
    ProbeToolRequest,
    ProbeToolResponse,
    RunToolRequest,
    RunToolResponse,
    SubmitToolRequest,
    SubmitToolResponse,
)
from organelleverse.core.errors import OrganelleContractError, OrganelleExecutionError
from organelleverse.operations.spec import StrictSpecModel

from .config import WslTargetConfig
from .launcher import WslProviderLauncher

__all__ = ["WslComputeProvider"]

_ResponseT = TypeVar("_ResponseT", bound=StrictSpecModel)


class WslComputeProvider:
    """Call the fixed six Linux Provider tools over an SDK stdio session.

    Each method opens a new control session.  Provider run identity lives on
    the Linux side, so reconnecting for ``get``, ``cancel``, or ``artifacts``
    never resubmits scientific work.
    """

    def __init__(
        self,
        config: WslTargetConfig,
        *,
        launcher: WslProviderLauncher | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self.config = WslTargetConfig.model_validate(config)
        self.launcher = launcher or WslProviderLauncher()
        self.env = dict(env) if env is not None else None

    def probe(self, request: ProbeToolRequest | None = None) -> ProbeToolResponse:
        return self._call(
            "ov.compute.probe",
            request or ProbeToolRequest(),
            ProbeToolResponse,
        )

    def prepare(self, request: PrepareToolRequest) -> PrepareToolResponse:
        return self._call("ov.compute.prepare", request, PrepareToolResponse)

    def submit(self, request: SubmitToolRequest) -> SubmitToolResponse:
        return self._call("ov.compute.submit", request, SubmitToolResponse)

    def get(self, request: RunToolRequest | str) -> RunToolResponse:
        return self._call("ov.compute.get", self._run_request(request), RunToolResponse)

    def cancel(self, request: RunToolRequest | str) -> RunToolResponse:
        return self._call("ov.compute.cancel", self._run_request(request), RunToolResponse)

    def artifacts(self, request: RunToolRequest | str) -> ArtifactsToolResponse:
        response = self._call(
            "ov.compute.artifacts", self._run_request(request), ArtifactsToolResponse
        )
        try:
            manifest = RemoteArtifactManifest.model_validate(response.manifest)
        except ValidationError as error:
            raise OrganelleContractError(
                code="compute.provider_response_invalid",
                message="the Linux Provider artifact manifest does not match its fixed contract",
                details={"tool": "ov.compute.artifacts"},
            ) from error
        return response.model_copy(update={"manifest": manifest})

    @staticmethod
    def _run_request(request: RunToolRequest | str) -> RunToolRequest:
        return (
            request
            if isinstance(request, RunToolRequest)
            else RunToolRequest(provider_run_id=request)
        )

    def _call(
        self,
        tool_name: str,
        request: StrictSpecModel,
        response_model: type[_ResponseT],
    ) -> _ResponseT:
        return asyncio.run(self._call_async(tool_name, request, response_model))

    async def _call_async(
        self,
        tool_name: str,
        request: StrictSpecModel,
        response_model: type[_ResponseT],
    ) -> _ResponseT:
        # Lazy imports preserve the provider package's no-MCP bare-import boundary.
        from mcp.client.session import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client
        from mcp.types import TextContent

        argv = self.launcher.mcp_argv(self.config)
        parameters = StdioServerParameters(
            command=argv[0],
            args=list(argv[1:]),
            env=self.env,
        )
        with tempfile.TemporaryFile(mode="w+") as diagnostics:
            try:
                async with (
                    stdio_client(parameters, errlog=diagnostics) as streams,
                    ClientSession(*streams) as session,
                ):
                    await session.initialize()
                    result = await session.call_tool(
                        tool_name,
                        request.model_dump(mode="json", by_alias=True),
                    )
            except OSError as error:
                raise OrganelleExecutionError(
                    code="compute.wsl_provider_unavailable",
                    message="the WSL Linux Provider control process could not be started",
                    details={"errno": error.errno},
                    retryable=True,
                ) from error

        if result.is_error:
            raise OrganelleExecutionError(
                code="compute.provider_tool_failed",
                message="the Linux Provider refused the compute control request",
                details={"tool": tool_name},
                retryable=True,
            )
        if len(result.content) != 1 or not isinstance(result.content[0], TextContent):
            raise OrganelleContractError(
                code="compute.provider_response_invalid",
                message="the Linux Provider returned an invalid control response",
                details={"tool": tool_name},
            )
        try:
            payload = json.loads(result.content[0].text)
            return response_model.model_validate(payload)
        except (AttributeError, TypeError, ValueError, ValidationError) as error:
            raise OrganelleContractError(
                code="compute.provider_response_invalid",
                message="the Linux Provider response does not match its fixed contract",
                details={"tool": tool_name},
            ) from error
