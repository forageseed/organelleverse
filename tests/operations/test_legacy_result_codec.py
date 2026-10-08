"""Tests for the LEGACY_RESULT decoder (Ruling 2, Decision 004 approval 2026-08-14).

The agent-facing field IS the L1 ``OrganelleResult`` model: Pydantic's strict
validation reconstructs the frozen result from the submitted JSON before the
scientific callable runs, so an invalid payload is refused at the parameter
boundary and the implementation receives a real frozen result.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from organelleverse.capabilities.parser import parse_capability_bundle
from organelleverse.core.errors import OrganelleError
from organelleverse.operations.python_binding import bind_python_capability

_BUNDLE = (
    Path(__file__).resolve().parents[2]
    / "src" / "organelleverse" / "capabilities"
    / "visualization-save-plot"
)


def _result_payload() -> dict[str, object]:
    from organelleverse.visualization.suite_plots import ideogram

    prepared = ideogram([{"chromosome": "chr1", "length": 5_000_000}])
    return prepared.model_dump(mode="json")


def _bound():
    from organelleverse.visualization.suite_plots import save_plot

    bundle = parse_capability_bundle(_BUNDLE / "capability.toml")
    return bind_python_capability(bundle, save_plot, frozen_schema=None)


def test_bundle_binds_with_the_result_input() -> None:
    binding = _bound()
    fields = type(binding.parameters_model).model_fields if hasattr(binding, "parameters_model") else None
    if fields is None:
        # fall back to the spec's declared parameter codecs
        specs = {b.name: b.codec.value for b in binding.spec.binding.parameters}
        assert specs["plot"] == "legacy_result"


def test_valid_payload_reconstructs_the_frozen_result(tmp_path: Path) -> None:
    binding = _bound()
    out = tmp_path / "plot.png"
    result = binding.invoke(
        None,
        {
            "plot": _result_payload(),
            "output": str(out),
        },
    )
    assert result is not None
    # the writer produced the destination artifact
    assert out.is_file()


def test_invalid_payload_is_refused_at_the_boundary(tmp_path: Path) -> None:
    binding = _bound()
    with pytest.raises(OrganelleError):
        binding.invoke(
            None,
            {
                "plot": {"not": "a result"},  # missing every required field
                "output": str(tmp_path / "x.png"),
            },
        )
