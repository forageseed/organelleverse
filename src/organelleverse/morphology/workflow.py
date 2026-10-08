"""Declarative image-analysis workflows: the agent-facing task schema.

A workflow is an ordered chain of operations over one EM image, declared in
JSON (``organelleverse.image-workflow.v1``) so an agent can (a) discover what
steps exist and in which order, (b) execute the chain per image, and (c)
read the per-image operation record back from the revision lineage — every
step registers its output as a child revision, so the chain IS the history.

The canonical example (``mito-count``): auto-segment -> vision adjudication
(LLM or human looks at the overlay; unadjudicated objects stay
``not_verified`` and never silently pass) -> scale calibration (µm/px) ->
measurement + per-class summary.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from ..operations.spec import StrictSpecModel

WORKFLOW_SCHEMA_VERSION = "organelleverse.image-workflow.v1"

__all__ = ["WORKFLOWS", "ImageWorkflow", "WorkflowStep", "describe_workflows"]


class WorkflowStep(StrictSpecModel):
    """One link of the chain. `op` is a capability id or builtin op name."""

    id: str = Field(min_length=1)
    op: str = Field(min_length=1)
    title: str = ""
    description: str = ""
    inputs: tuple[str, ...] = ()  # carried forward from earlier steps
    parameters: dict[str, Any] = Field(default_factory=dict)
    #: "capability" steps run a registered operation; "adjudication" steps
    #: pause for a vision verdict (LLM or human) and record it; they never
    #: fabricate an "agree".
    kind: Literal["capability", "adjudication"] = "capability"


class ImageWorkflow(StrictSpecModel):
    schema_version: str = WORKFLOW_SCHEMA_VERSION
    id: str = Field(min_length=1)
    title: str
    description: str
    steps: tuple[WorkflowStep, ...] = Field(min_length=1)


_MITO_COUNT = ImageWorkflow(
    id="mito-count",
    title="Per-image mitochondria statistics",
    description=(
        "Auto-segment every image, adjudicate the masks with vision (an LLM "
        "or a human looks at the overlay; objects stay not_verified until "
        "then), calibrate the pixel scale, then measure and summarize — "
        "per-image mitochondrion counts, areas, and densities."
    ),
    steps=(
        WorkflowStep(
            id="segment",
            op="morphology.segment",
            title="Automatic segmentation",
            description="micro-SAM / OrgSegNet prediction; registers a mask layer revision.",
        ),
        WorkflowStep(
            id="verify",
            op="vision.adjudicate",
            title="Vision adjudication",
            description=(
                "A multimodal LLM or a human inspects the mask overlay and "
                "records agree/disagree/uncertain per object. Until then "
                "objects remain not_verified — never silently accepted."
            ),
            kind="adjudication",
        ),
        WorkflowStep(
            id="calibrate",
            op="scale.set",
            title="Scale calibration",
            description="Pixel size in µm/px (from the image scale bar); uncalibrated rows stay in px.",
        ),
        WorkflowStep(
            id="measure",
            op="morphology.measure",
            title="Measure + summarize",
            description="Per-object morphometrics and per-class summary (count, area fraction, density).",
        ),
    ),
)

WORKFLOWS: dict[str, ImageWorkflow] = {_MITO_COUNT.id: _MITO_COUNT}


def describe_workflows() -> list[dict[str, Any]]:
    """Agent-readable workflow catalog (the task schema surface)."""
    return [
        {
            "schema_version": WORKFLOW_SCHEMA_VERSION,
            "id": wf.id,
            "title": wf.title,
            "description": wf.description,
            "steps": [
                {
                    "id": step.id,
                    "op": step.op,
                    "kind": step.kind,
                    "title": step.title,
                    "description": step.description,
                    "parameters": step.parameters,
                }
                for step in wf.steps
            ],
        }
        for wf in WORKFLOWS.values()
    ]
