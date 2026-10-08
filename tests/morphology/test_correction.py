"""T4: human correction as a first-class provenance object (task brief 2026-08-17).

The correction record is contract-level evidence: both endpoint hashes, the
affected region, the correction type, a free-text reason, the operator, and
a timestamp. The reason is evidence, never mechanism — this module performs
no LLM or network calls (pinned by a static source check below).
"""

from __future__ import annotations

import hashlib
import importlib.metadata
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from organelleverse.morphology.correction import LABEL_SOURCE, correct
from organelleverse.morphology.measure import measure, write_csv


def _write_label(path: Path, values: np.ndarray) -> Path:
    Image.fromarray(values).save(path)
    return path


def _base_map() -> np.ndarray:
    label = np.zeros((40, 40), dtype=np.uint8)
    label[5:15, 5:15] = 1  # Chloroplast
    label[25:30, 25:30] = 2  # Mitochondria
    return label


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture()
def maps(tmp_path: Path) -> tuple[Path, Path]:
    original = _write_label(tmp_path / "original.png", _base_map())
    corrected_values = _base_map()
    # A split correction: one chloroplast becomes two, shifting some pixels
    # out of class 1, plus a small relabel of mitochondria to vacuole.
    corrected_values[10:15, 5:15] = 0
    corrected_values[25:27, 25:30] = 3
    corrected = _write_label(tmp_path / "corrected.png", corrected_values)
    return original, corrected


def test_correction_record_has_every_required_field(maps: tuple[Path, Path]) -> None:
    original, corrected = maps
    result = correct(
        original,
        corrected,
        correction_type="split",
        reason="two touching chloroplasts merged by the model",
        operator="jane.doe@example.org",
    )
    assert result.status == "ok"
    record = result.metrics["correction"]
    assert record["original_sha256"] == _sha256(original)
    assert record["corrected_sha256"] == _sha256(corrected)
    assert record["correction_type"] == "split"
    assert record["reason"] == "two touching chloroplasts merged by the model"
    assert record["operator"] == "jane.doe@example.org"
    assert record["corrected_at"]
    assert record["label_source"] == LABEL_SOURCE == "human_corrected"
    assert result.artifacts[0].kind == "corrected_label_map"
    assert result.artifacts[0].sha256 == record["corrected_sha256"]


def test_region_summary_matches_the_constructed_diff(maps: tuple[Path, Path]) -> None:
    original, corrected = maps
    result = correct(original, corrected, correction_type="split", reason="r", operator="op")
    region = result.metrics["correction"]["region"]
    # Constructed diff: rows 10-14 (5x10 removed from class 1) + rows 25-26
    # (2x5 relabelled 2->3). Union bbox spans rows 10..27, cols 5..30.
    assert region["bbox"] == (10, 5, 27, 30)
    assert region["n_changed_pixels"] == 5 * 10 + 2 * 5
    assert region["n_pixels"] == 40 * 40
    assert region["transitions"] == {"1->0": 50, "2->3": 10}


def test_pixel_identical_maps_fail_closed(tmp_path: Path) -> None:
    original = _write_label(tmp_path / "a.png", _base_map())
    corrected = _write_label(tmp_path / "b.png", _base_map())
    result = correct(original, corrected, correction_type="relabel", reason="r", operator="op")
    assert result.status == "failed"
    assert result.errors[0].code == "morphology.correct_no_op"


def test_shape_mismatch_fails_closed(tmp_path: Path) -> None:
    original = _write_label(tmp_path / "a.png", _base_map())
    corrected = _write_label(tmp_path / "b.png", np.zeros((20, 20), dtype=np.uint8))
    result = correct(original, corrected, correction_type="add", reason="r", operator="op")
    assert result.status == "failed"
    assert result.errors[0].code == "morphology.correct_shape_mismatch"


def test_invalid_type_and_blank_metadata_fail_closed(maps: tuple[Path, Path]) -> None:
    original, corrected = maps
    bad_type = correct(original, corrected, correction_type="improve", reason="r", operator="op")
    assert bad_type.status == "failed"
    assert bad_type.errors[0].code == "morphology.correct_invalid_type"

    blank_reason = correct(original, corrected, correction_type="add", reason="  ", operator="op")
    assert blank_reason.status == "failed"
    assert blank_reason.errors[0].code == "morphology.correct_missing_metadata"

    blank_operator = correct(original, corrected, correction_type="add", reason="r", operator="")
    assert blank_operator.status == "failed"
    assert blank_operator.errors[0].code == "morphology.correct_missing_metadata"


def test_measure_distinguishes_model_and_corrected_sources(tmp_path: Path) -> None:
    label_path = _write_label(tmp_path / "label.png", _base_map())
    model_rows = measure(label_path, label_source="model")
    corrected_rows = measure(label_path, label_source=LABEL_SOURCE)
    assert model_rows["label_source"] == "model"
    assert corrected_rows["label_source"] == "human_corrected"
    assert {row["label_source"] for row in model_rows["per_object"]} == {"model"}
    assert {row["label_source"] for row in corrected_rows["per_object"]} == {"human_corrected"}
    # The CSV projection carries the same column.
    csv_path = write_csv(corrected_rows, tmp_path / "out.csv")
    header = csv_path.read_text().splitlines()[0]
    assert "label_source" in header.split(",")


def test_correction_module_makes_no_llm_or_network_calls() -> None:
    """AST-level: no LLM SDK or network import/call anywhere in the module."""
    import ast

    import organelleverse.morphology.correction as correction_module

    source = Path(correction_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden_roots = {
        "openai",
        "anthropic",
        "langchain",
        "llm",
        "requests",
        "urllib",
        "httpx",
        "socket",
        "aiohttp",
    }
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not (imported & forbidden_roots), f"forbidden imports: {imported & forbidden_roots}"


def test_correct_bundle_discovers_and_invokes_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, maps: tuple[Path, Path]
) -> None:
    """The hand-authored morphology.correct bundle admits through the real pipeline."""
    from organelleverse.capabilities.discovery import discover_capabilities
    from organelleverse.capabilities.verification import (
        LocalVerificationEnvironment,
        VerificationStore,
        verify_capability,
    )

    home = tmp_path / "home"
    project = tmp_path / "project"
    (project / ".organelleverse" / "capabilities").mkdir(parents=True)
    home.mkdir()
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda **kw: ())
    monkeypatch.setenv("ORGANELLEVERSE_HOME", str(home))
    monkeypatch.delenv("ORGANELLEVERSE_CAPABILITY_PATH", raising=False)
    monkeypatch.chdir(project)

    discovered = discover_capabilities()
    entry = discovered.describe("morphology.correct")
    assert entry is not None
    store = VerificationStore(home / "verifications")
    verify_capability(
        "morphology.correct",
        store=store,
        environment=LocalVerificationEnvironment(discovered),
    )
    admitted = discover_capabilities()
    binding = admitted.binding_source().resolve("morphology.correct")
    assert binding is not None

    original, corrected = maps
    result = binding.invoke(
        None,
        {
            "original_label_map": str(original),
            "corrected_label_map": str(corrected),
            "correction_type": "relabel",
            "reason": "mitochondria misclassified",
            "operator": "jane.doe@example.org",
        },
    )
    assert result.status == "ok"
    assert result.metrics["correction"]["label_source"] == "human_corrected"
