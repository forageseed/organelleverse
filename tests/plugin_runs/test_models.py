"""Closed shared plugin-run contracts."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from organelleverse.capabilities.plugin_descriptor import PluginDescriptor
from organelleverse.plugin_runs import PluginRunRecord, PluginRunRequest


def _descriptor() -> PluginDescriptor:
    return PluginDescriptor(
        capability_id="demo.experiment",
        title="Demo",
        summary="",
        inputs=(),
        outputs=(),
        parameters=(),
        agent_task_description="",
        agent_examples=(),
        gui_page=None,
        optimization=None,
    )


def test_request_and_record_are_closed_models() -> None:
    with pytest.raises(ValidationError):
        PluginRunRequest.model_validate(
            {"capability_id": "demo.experiment", "inputs": {}, "unexpected": True}
        )
    with pytest.raises(ValidationError):
        PluginRunRecord.model_validate(
            {
                "run_id": "run-1",
                "capability_id": "demo.experiment",
                "contract_identity": "contract-v1",
                "status": "queued",
                "submitted_at": datetime.now(UTC),
                "request": {"capability_id": "demo.experiment", "inputs": {}},
                "descriptor": _descriptor().model_dump(mode="json"),
                "unexpected": True,
            }
        )


def test_legacy_record_allows_only_the_two_new_identity_fields_to_be_absent() -> None:
    record = PluginRunRecord(
        run_id="legacy-run",
        capability_id="demo.experiment",
        status="failed",
        submitted_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
        request=PluginRunRequest(capability_id="demo.experiment", inputs={}),
        descriptor=_descriptor(),
    )

    assert record.contract_identity is None
    assert record.l6_run_id is None


def test_record_rejects_a_run_id_that_cannot_be_a_store_filename() -> None:
    with pytest.raises(ValidationError):
        PluginRunRecord(
            run_id="../outside",
            capability_id="demo.experiment",
            contract_identity="contract-v1",
            status="queued",
            submitted_at=datetime.now(UTC),
            request=PluginRunRequest(capability_id="demo.experiment", inputs={}),
            descriptor=_descriptor(),
        )
