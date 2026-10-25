"""The JSON adapter preserves the request, grants, advice and errors across a caller hook."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from organelleverse.core.errors import OrganelleExecutionError
from organelleverse.operations.adapters.json import invoke_json
from organelleverse.operations.execution_hooks import ExecutionDelegated, execution_scope
from organelleverse.operations.registry import registry
from organelleverse.operations.spec import SideEffect

TARGET = {"caller_selection": "gpu1"}
INVOCATION = {"operation_id": "io.read_long_reads", "input": None,
              "parameters": {"reads": "C:/data/r.fastq", "technology": "ont", "quality_state": "raw"}}
GRANTS = [SideEffect.READ_FILES]


class _Hooks:
    def __init__(self, *, error=None):
        self.calls = []
        self.requirements = []
        self.error = error

    def select(self, requirement):
        self.requirements.append(requirement)
        return TARGET

    def execute(self, request, *, registry, granted, selection):
        if self.error is not None:
            raise self.error
        self.calls.append({"invocation": dict(request), "granted": set(granted), "target": selection})
        return {"ok": True, "operation_id": request["operation_id"], "result": {"kind": "result"},
                "execution": {"caller_selection": "gpu1"}}


@pytest.fixture
def hooks():
    caller = _Hooks()
    with execution_scope(hooks=caller):
        yield caller


def _delegate(monkeypatch):
    def routed(*_a, **_k):
        raise ExecutionDelegated(TARGET, reason="oatk environment")
    monkeypatch.setattr(registry, "require", lambda _op: SimpleNamespace(invoke_with_parameter_decoder=routed))


def test_delegation_preserves_the_request_and_granted_side_effects(monkeypatch, hooks):
    _delegate(monkeypatch)
    response = invoke_json(INVOCATION, registry=registry, granted_side_effects=GRANTS)
    assert response["ok"] is True and response["execution"] == {"caller_selection": "gpu1"}
    assert hooks.calls == [{"invocation": INVOCATION, "granted": {SideEffect.READ_FILES}, "target": TARGET}]


def test_missing_dependencies_ask_the_installed_caller(monkeypatch, hooks):
    monkeypatch.setattr(registry, "check_dependencies", lambda _op: SimpleNamespace(ready=False, model_dump=lambda **_: {}))
    response = invoke_json(INVOCATION, registry=registry, granted_side_effects=GRANTS)
    assert response["ok"] is True and len(hooks.calls) == 1
    assert hooks.requirements[0].reason == "missing_dependencies"


def test_delegation_preserves_advice_metadata(monkeypatch, hooks):
    _delegate(monkeypatch)
    invocation = {**INVOCATION, "advice_context": {"species": "Arabidopsis thaliana"}}
    response = invoke_json(invocation, registry=registry, granted_side_effects=GRANTS)
    assert response["ok"] and hooks.calls[0]["invocation"] == invocation


def test_no_caller_or_local_only_keeps_missing_dependencies_an_error(monkeypatch, hooks):
    monkeypatch.setattr(registry, "check_dependencies", lambda _op: SimpleNamespace(ready=False, model_dump=lambda **_: {}))
    with execution_scope(hooks=None):
        assert invoke_json(INVOCATION, registry=registry, granted_side_effects=GRANTS)["error"]["error_code"] == "dependency.missing"
    with execution_scope(local_only=True):
        assert invoke_json(INVOCATION, registry=registry, granted_side_effects=GRANTS)["error"]["error_code"] == "dependency.missing"
    assert hooks.calls == [] and hooks.requirements == []


def test_caller_failure_returns_an_ordinary_structured_error(monkeypatch):
    _delegate(monkeypatch)
    caller = _Hooks(error=OrganelleExecutionError(code="compute.catalog_mismatch", message="redeploy", details={}))
    with execution_scope(hooks=caller):
        response = invoke_json(INVOCATION, registry=registry, granted_side_effects=GRANTS)
    assert response["ok"] is False and response["error"]["error_code"] == "compute.catalog_mismatch"


def test_a_selection_without_its_caller_is_an_explicit_error(monkeypatch):
    _delegate(monkeypatch)
    with execution_scope(hooks=None):
        response = invoke_json(INVOCATION, registry=registry, granted_side_effects=GRANTS)
    assert response["ok"] is False and response["error"]["error_code"] == "execution.host_required"
