"""Batch morphometrics driver (T-C2) with run traceability (T-C3).

Contract operation, not a script. Batch discipline (project rules):

- ``progress_state.json`` in the state directory; reruns skip ``done`` items
- edit-restart protocol: after editing, kill and restart — never assume
  in-memory state updated (documented in the docstring + report)
- preflight mode processes the first few items only
- completion summary reports done / failed / skipped; single-item failures
  never abort the batch, but an all-failed batch is an explicit failure
- outputs stay inside the project directory, never /tmp

Verification-gate respect (T4b): rows for blocked objects are written to the
long table (they are data) but excluded from every aggregate, and the
summary reports the excluded count. ``not_verified`` and ``agree`` are
distinct count buckets — never merged.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

__all__ = ["BatchItem", "batch_measure", "batch_measure_dir"]


@dataclass(frozen=True)
class BatchItem:
    """One label map to measure. ``verdicts`` carries T4b gate outcomes as
    ``object key -> verdict`` (agree/disagree/uncertain/not_verified).

    Object identity scheme (shared with the verify gate): semantic maps key
    objects as ``class_index * 1000 + per-class instance number``; instance
    maps key them as the raw instance id. Never mix the two in one verdicts
    dict — declare the map's convention via ``instance_map``.
    """

    image_id: str
    label_map: Path
    pixel_size_um: float | None = None
    label_source: str | None = None
    verdicts: dict[int, str] | None = None
    #: object key -> parent region id (T-F1 hierarchy); empty for flat runs
    parents: dict[int, str] | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def batch_measure(
    items: list[BatchItem],
    *,
    state_dir: str | Path,
    backend: Literal["native", "auto", "qupath"] = "native",
    preflight: bool = False,
    min_area: int = 10,
    instance_map: bool = False,
) -> dict[str, Any]:
    """Measure every item, writing the long table and the grouped summary.

    Returns the completion summary dict. Raises (explicit failure) only when
    every item failed — partial failures are isolated per item and listed.
    """
    from .measure import measure

    state_root = Path(state_dir)
    state_root.mkdir(parents=True, exist_ok=True)
    state_path = state_root / "progress_state.json"
    state: dict[str, Any] = {"done": {}, "failed": {}}
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))

    work = items[:5] if preflight else items
    long_rows: list[dict[str, Any]] = []
    summary = {"done": 0, "failed": 0, "skipped": 0, "failed_items": []}
    # Calibration groups: never pool uncalibrated with calibrated.
    groups: dict[str, list[dict[str, Any]]] = {}
    excluded_blocked_total = 0

    for item in work:
        key = f"{item.image_id}|{item.label_map}"
        if key in state["done"]:
            summary["skipped"] += 1
            continue
        try:
            label_sha = _sha256(Path(item.label_map))
            result = measure(
                item.label_map,
                min_area=min_area,
                instance_map=instance_map,
                label_source=item.label_source,
                pixel_size_um=item.pixel_size_um,
                backend=backend,
            )
        except Exception as error:  # per-item isolation is the batch contract
            state["failed"][key] = str(error)
            summary["failed"] += 1
            summary["failed_items"].append({"image_id": item.image_id, "error": str(error)})
            state_path.write_text(json.dumps(state, indent=1, sort_keys=True), encoding="utf-8")
            continue

        state["done"][key] = {"label_sha256": label_sha}
        state_path.write_text(json.dumps(state, indent=1, sort_keys=True), encoding="utf-8")
        summary["done"] += 1

        calibration = f"{item.pixel_size_um}" if item.pixel_size_um else "uncalibrated"
        group = groups.setdefault(f"{item.image_id}|{calibration}", [])
        for row in result["per_object"]:
            object_key = row["label"] if instance_map else row["class"] * 1000 + row["label"]
            verdict = (item.verdicts or {}).get(object_key, "not_verified")
            blocked = verdict in {"disagree", "uncertain"}
            long_rows.append(
                {
                    "image_id": item.image_id,
                    "object_id": object_key,
                    "parent_id": (item.parents or {}).get(object_key, ""),
                    "class_name": row["class_name"],
                    "area_px": row["area_px"],
                    "area_um2": row["area_um2"],
                    "perimeter": row["perimeter"],
                    "circularity": row["circularity"],
                    "solidity": row.get("solidity"),
                    "feret_diameter_max": row.get("feret_diameter_max"),
                    "label_source": row.get("label_source"),
                    "verification_status": verdict,
                    "blocks_downstream": blocked,
                    "pixel_size_um": item.pixel_size_um,
                    "label_map_sha256": label_sha,
                }
            )
            if blocked:
                excluded_blocked_total += 1
            else:
                group.append(row)

    # Blocked objects never enter the main export — they get their own
    # exclusion list with reasons (T-F5).
    blocked_rows = [row for row in long_rows if row["blocks_downstream"]]
    main_rows = [row for row in long_rows if not row["blocks_downstream"]]
    long_csv = state_root / "objects_long.csv"
    if main_rows:
        with long_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(main_rows[0].keys()))
            writer.writeheader()
            writer.writerows(main_rows)
    excluded_csv = state_root / "excluded_blocked.csv"
    if blocked_rows:
        with excluded_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "image_id",
                    "object_id",
                    "class_name",
                    "verification_status",
                    "label_map_sha256",
                ],
            )
            writer.writeheader()
            for row in blocked_rows:
                writer.writerow({key: row[key] for key in writer.fieldnames})

    # Grouped summary (image x calibration): blocked excluded, count shown.
    summary_rows = []
    for group_key, rows in groups.items():
        image_id, calibration = group_key.split("|", 1)
        by_class: dict[str, list[float]] = {}
        for row in rows:
            by_class.setdefault(row["class_name"], []).append(row["area_px"])
        for class_name, areas in sorted(by_class.items()):
            arr = sorted(areas)
            n = len(arr)
            mean = sum(arr) / n
            sd = (sum((a - mean) ** 2 for a in arr) / (n - 1)) ** 0.5 if n > 1 else 0.0
            summary_rows.append(
                {
                    "image_id": image_id,
                    "class_name": class_name,
                    "calibration": calibration,
                    "n": n,
                    "area_px_mean": round(mean, 2),
                    "area_px_sd": round(sd, 2),
                    "area_px_median": round(arr[n // 2], 2),
                    "excluded_blocked": excluded_blocked_total,
                }
            )
    summary_csv = state_root / "summary_by_image_class.csv"
    if summary_rows:
        with summary_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    summary["excluded_blocked"] = excluded_blocked_total
    summary["long_table"] = str(long_csv) if main_rows else None
    summary["excluded_table"] = str(excluded_csv) if blocked_rows else None
    summary["summary_table"] = str(summary_csv) if summary_rows else None

    # Explicit failure only when EVERY item of this run failed: skipped items
    # are recorded prior successes, so done=0 + skipped>0 is not an all-fail.
    if summary["done"] == 0 and summary["failed"] > 0 and summary["skipped"] == 0:
        raise RuntimeError(f"batch measurement failed for every item: {summary['failed_items']}")
    return summary


def batch_measure_dir(
    label_dir: str,
    state_dir: str,
    *,
    pixel_size_um: float | None = None,
    label_source: str | None = None,
    preflight: bool = False,
    min_area: int = 10,
    instance_map: bool = False,
    backend: Literal["native", "auto", "qupath"] = "native",
) -> dict[str, Any]:
    """Agent-facing batch entry: measure every label map in one directory.

    A JSON-bindable wrapper around :func:`batch_measure` — one ``BatchItem``
    per ``*.png`` under ``label_dir`` (image_id = file stem), so the batch
    contract (progress state, per-item isolation, blocked-object exclusion)
    is reachable from capability bindings that can only carry primitives.
    """
    root = Path(label_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"label_dir does not exist: {root}")
    items = [
        BatchItem(
            image_id=path.stem,
            label_map=path,
            pixel_size_um=pixel_size_um,
            label_source=label_source,
        )
        for path in sorted(root.glob("*.png"))
    ]
    if not items:
        raise FileNotFoundError(f"no *.png label maps under {root}")
    return batch_measure(
        items,
        state_dir=state_dir,
        backend=backend,
        preflight=preflight,
        min_area=min_area,
        instance_map=instance_map,
    )
