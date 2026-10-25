"""The registered schema of ``rna_editing.predict_edits`` lists every backend it runs."""

from __future__ import annotations

import organelleverse as ov


def test_backend_schema_is_the_closed_set_predict_edits_dispatches() -> None:
    backend = ov.operations.parameter_schema("rna_editing.predict_edits")["properties"]["backend"]
    assert backend["enum"] == ["auto", "deepred", "plantc2u", "heuristic"]
    assert backend["default"] == "auto"
