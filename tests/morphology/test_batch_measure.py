"""T-C2 batch driver: state file, resume, failure isolation, gate respect."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from organelleverse.morphology.batch import BatchItem, batch_measure, batch_measure_dir


def _label_map(path: Path, n_objects: int = 2) -> Path:
    lab = np.zeros((40, 40), dtype=np.uint8)
    for i in range(n_objects):
        r = 5 + i * 12
        lab[r : r + 8, 5:13] = i + 1
    Image.fromarray(lab).save(path)
    return path


def test_resume_skips_completed_items(tmp_path: Path) -> None:
    a = _label_map(tmp_path / "a.png")
    b = _label_map(tmp_path / "b.png")
    items = [BatchItem("a", a), BatchItem("b", b)]
    first = batch_measure(items, state_dir=tmp_path / "state")
    assert first["done"] == 2 and first["failed"] == 0
    assert (tmp_path / "state" / "progress_state.json").is_file()
    second = batch_measure(items, state_dir=tmp_path / "state")
    assert second == {
        "done": 0,
        "failed": 0,
        "skipped": 2,
        "failed_items": [],
        "excluded_blocked": 0,
        "long_table": None,
        "summary_table": None,
        "excluded_table": None,
    }


def test_single_failure_isolated_and_listed(tmp_path: Path) -> None:
    good = _label_map(tmp_path / "good.png")
    missing = tmp_path / "missing.png"
    items = [BatchItem("good", good), BatchItem("bad", missing)]
    summary = batch_measure(items, state_dir=tmp_path / "state")
    assert summary["done"] == 1
    assert summary["failed"] == 1
    assert summary["failed_items"][0]["image_id"] == "bad"


def test_all_failed_batch_is_an_explicit_failure(tmp_path: Path) -> None:
    items = [BatchItem("bad", tmp_path / "nope.png")]
    with pytest.raises(RuntimeError, match="every item"):
        batch_measure(items, state_dir=tmp_path / "state")


def test_preflight_processes_only_the_first_five(tmp_path: Path) -> None:
    items = [BatchItem(f"img{i}", _label_map(tmp_path / f"{i}.png")) for i in range(7)]
    summary = batch_measure(items, state_dir=tmp_path / "state", preflight=True)
    assert summary["done"] == 5


def test_blocked_objects_excluded_from_aggregates_with_count(tmp_path: Path) -> None:
    label = _label_map(tmp_path / "a.png", n_objects=2)
    items = [
        BatchItem(
            "a",
            label,
            verdicts={1001: "agree", 2001: "disagree"},  # mito (2001) blocked by the gate
        )
    ]
    summary = batch_measure(items, state_dir=tmp_path / "state")
    assert summary["excluded_blocked"] == 1

    # the main table carries only unblocked rows; the blocked one is listed apart
    long_rows = (tmp_path / "state" / "objects_long.csv").read_text().splitlines()
    assert len(long_rows) == 2  # header + the one unblocked object
    excluded = (tmp_path / "state" / "excluded_blocked.csv").read_text()
    assert "disagree" in excluded

    summary_rows = (tmp_path / "state" / "summary_by_image_class.csv").read_text().splitlines()
    # summary counts only the unblocked object
    assert " n," not in summary_rows[0]  # sanity on header position
    assert ",1," in summary_rows[1].replace('"', "")


def test_verification_states_stay_distinct(tmp_path: Path) -> None:
    label = _label_map(tmp_path / "a.png", n_objects=2)
    items = [BatchItem("a", label, verdicts={1001: "agree"})]  # mito (2001) unverified
    batch_measure(items, state_dir=tmp_path / "state")
    text = (tmp_path / "state" / "objects_long.csv").read_text()
    assert "agree" in text and "not_verified" in text


def test_label_map_hash_is_recorded_per_row(tmp_path: Path) -> None:
    label = _label_map(tmp_path / "a.png")
    batch_measure([BatchItem("a", label)], state_dir=tmp_path / "state")
    text = (tmp_path / "state" / "objects_long.csv").read_text()
    state = json.loads((tmp_path / "state" / "progress_state.json").read_text())
    sha = next(iter(state["done"].values()))["label_sha256"]
    assert sha in text


def test_calibrated_and_uncalibrated_never_pool(tmp_path: Path) -> None:
    label = _label_map(tmp_path / "a.png")
    items = [
        BatchItem("cal", label, pixel_size_um=0.1),
        BatchItem("raw", label),  # uncalibrated
    ]
    batch_measure(items, state_dir=tmp_path / "state")
    rows = (tmp_path / "state" / "summary_by_image_class.csv").read_text().splitlines()
    # two classes x two calibration identities = four separate group rows;
    # calibrated and uncalibrated are never pooled into one n
    assert len(rows) == 5
    calibrated = [r for r in rows[1:] if ",0.1," in r]
    uncalibrated = [r for r in rows[1:] if "uncalibrated" in r]
    assert len(calibrated) == 2 and len(uncalibrated) == 2
    assert all(",1," in r for r in calibrated + uncalibrated)  # each group counts only itself


def test_rerun_with_one_new_broken_item_is_not_an_all_fail(tmp_path):
    """Regression: skipped items are prior successes — a rerun whose only new
    item fails must report the isolated failure, not raise all-failed."""
    import numpy as np
    from PIL import Image

    good = tmp_path / "good.png"
    Image.fromarray(np.eye(40, dtype=np.uint8) * 0 + np.pad(np.ones((20, 20)), 10).astype(np.uint8), mode="L").save(good)
    state = tmp_path / "state"
    first = batch_measure([BatchItem(image_id="a", label_map=good)], state_dir=state)
    assert first["done"] == 1
    broken = tmp_path / "missing.png"
    second = batch_measure(
        [
            BatchItem(image_id="a", label_map=good),
            BatchItem(image_id="b", label_map=broken),
        ],
        state_dir=state,
    )
    assert second["skipped"] == 1
    assert second["failed"] == 1
    assert second["done"] == 0  # and crucially: no raise


def test_batch_measure_dir_agent_entry(tmp_path):
    """The JSON-bindable wrapper: one item per png, stem as image_id."""
    labels = tmp_path / "labels"
    labels.mkdir()
    for name in ("a", "b"):
        _label_map(labels / f"{name}.png")
    state = tmp_path / "state"
    summary = batch_measure_dir(
        str(labels), str(state), pixel_size_um=0.1, label_source="model"
    )
    assert summary["done"] == 2
    rows = (state / "objects_long.csv").read_text().splitlines()
    assert rows[0].startswith("image_id,object_id")
    assert any(row.startswith("a,") for row in rows[1:])
    assert any(row.startswith("b,") for row in rows[1:])


def test_batch_measure_dir_rejects_missing_or_empty(tmp_path):
    import pytest as _pytest

    with _pytest.raises(FileNotFoundError):
        batch_measure_dir(str(tmp_path / "nope"), str(tmp_path / "state"))
    empty = tmp_path / "empty"
    empty.mkdir()
    with _pytest.raises(FileNotFoundError):
        batch_measure_dir(str(empty), str(tmp_path / "state"))
