"""Closed browser/Agent projection regressions."""

from organelleverse.plugin_runs.projection import browser_safe_json_object


def test_projection_removes_private_uri_leaves_and_structural_worker_fields() -> None:
    projected = browser_safe_json_object(
        {
            "scientific_value": "chloroplast",
            "managed_reference": "managed://runs/demo/output",
            "file_reference": "file:///tmp/worker-payload.json",
            "nested": {
                "argv": ["python", "/tmp/plugin.py"],
                "worker_payload": {"request": "private"},
                "staging_root": "relative-but-private",
                "plugin_root": "relative-but-private",
                "metric": 0.25,
            },
        }
    )

    assert projected == {
        "scientific_value": "chloroplast",
        "nested": {"metric": 0.25},
    }
