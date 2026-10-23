"""Operation adapters for external invocation transports."""

from .json import AgentInvocationRequest, NextInvocationRequest, invoke_json, invoke_next_json

__all__ = ["AgentInvocationRequest", "NextInvocationRequest", "invoke_json", "invoke_next_json"]
