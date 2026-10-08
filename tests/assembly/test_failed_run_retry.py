"""A failed assembly run must not permanently wedge its destination.

Found by dual-path real-data testing (2026-08-15): the reuse check demanded
the full publication set even for exit != 0 runs, then atomic publication
refused the existing directory — one bad read set blocked that request hash
forever.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from organelleverse.assembly.service import _load_reusable_result


def _write_failed_run(destination: Path) -> None:
    destination.mkdir(parents=True)
    (destination / "assembly_run_record.json").write_text(
        json.dumps({"process_exit_code": 1, "selected_backend": "oatk"}), encoding="utf-8"
    )
    (destination / "stdout.log").write_text("syncasm failed", encoding="utf-8")
    # deliberately no result.json / primary_genome.json — a failed run


def test_failed_run_allows_retry(tmp_path: Path) -> None:
    destination = tmp_path / "sha256-deadbeef"
    _write_failed_run(destination)
    # the call needs the full kwargs; only the failed-run branch matters here
    result = _load_reusable_result(
        destination,
        request=None,  # type: ignore[arg-type] - not reached for failed runs
        resolved_request_hash="sha256-deadbeef",
        selected_backend="oatk",  # type: ignore[arg-type]
        expected_route_reason_code="rule",  # type: ignore[arg-type]
        expected_environment_digest="digest",  # type: ignore[arg-type]
        expected_resources=None,  # type: ignore[arg-type]
    )
    assert result is None  # retry proceeds
    # the failed tree was archived aside, freeing the destination
    assert not destination.exists()
    archived = list(tmp_path.glob("sha256-deadbeef.failed-*"))
    assert len(archived) == 1
    assert (archived[0] / "assembly_run_record.json").is_file()
