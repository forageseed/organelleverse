from __future__ import annotations

import json
import os
import runpy
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest

_GATE = runpy.run_path(
    str(
        Path(__file__).parents[2]
        / "scripts"
        / "annotation_benchmarks"
        / "run_annotation_release_gate.py"
    )
)
run_scientific_gate = cast(Callable[..., dict[str, Any]], _GATE["run_scientific_gate"])


def _required_path(variable: str) -> Path:
    value = os.environ.get(variable, "")
    assert value, f"{variable} is required for the non-skippable annotation release gate"
    path = Path(value)
    assert path.exists(), f"{variable} does not exist: {path}"
    return path


@pytest.mark.release_annotation
def test_released_annotation_real_data_gate(tmp_path: Path) -> None:
    fixture_root = _required_path("ORG_VERSE_ANNOTATION_FIXTURE_ROOT")
    arabidopsis_genbank = _required_path("ORG_VERSE_ARABIDOPSIS_GENBANK")
    software_dir_value = os.environ.get("ORG_VERSE_SOFTWARE_DIR")

    work_dir_value = os.environ.get("ORG_VERSE_ANNOTATION_RELEASE_WORK_DIR")
    threads = int(os.environ.get("ORG_VERSE_ANNOTATION_RELEASE_THREADS", "4"))
    report = run_scientific_gate(
        fixture_root=fixture_root,
        arabidopsis_genbank=arabidopsis_genbank,
        work_dir=Path(work_dir_value) if work_dir_value else tmp_path,
        software_dir=Path(software_dir_value) if software_dir_value else None,
        threads=threads,
    )
    report_path_value = os.environ.get("ORG_VERSE_ANNOTATION_RELEASE_REPORT")
    if report_path_value:
        Path(report_path_value).write_text(
            json.dumps(report, sort_keys=True, allow_nan=False, ensure_ascii=False, indent=2) + "\n"
        )
    metrics = report["metrics"]

    assert metrics["validated_unique_pcg_recall"] >= 0.85
    assert metrics["pcg_name_precision"] >= 0.95
    assert metrics["rrna_recovered"] == 3
    assert metrics["trna_locus_recall"] >= 0.85
    assert metrics["trna_match_precision"] >= 0.80
    assert metrics["trna_hit_ratio"] <= 1.25
    assert metrics["invalid_trna_lengths"] == 0
    assert metrics["invalid_structures"] == 0
    assert metrics["nad5_spliced_length"] == 2010
    assert report["checks"]["direct_agent_artifact_identity"] is True
    assert report["checks"]["deterministic_rerun"] is True
    assert report["checks"]["annotation_qc_python_agent_identity"] is True
    assert report["checks"]["annotation_qc_deterministic_reuse"] is True
    assert report["checks"]["annotation_qc_bundle_roundtrip"] is True
    assert report["checks"]["annotation_qc_zero_contract_failures"] is True
    assert {"qc.annotation", "qc.write"} <= set(report["operations"])
    annotation_qc = report["annotation_qc"]
    assert annotation_qc["profile_id"] == "organelleverse.plant-mito-pcg.v1"
    assert annotation_qc["decision"] in {"ready", "needs_review"}
    assert annotation_qc["expected_profile_gene_count"] == 24
    assert annotation_qc["run_manifest_id"].startswith("annotation-qc-run:sha256:")
