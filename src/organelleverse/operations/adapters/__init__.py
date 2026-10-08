"""Operation adapters for external invocation transports."""

from .json import AgentInvocationRequest, invoke_json

__all__ = ["AgentInvocationRequest", "invoke_json"]
