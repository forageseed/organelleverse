from __future__ import annotations

import runpy
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

_GATE = runpy.run_path(
    str(
        Path(__file__).parents[2]
        / "scripts"
        / "annotation_benchmarks"
        / "run_annotation_release_gate.py"
    )
)
_find_tool = cast(Callable[[str, Path | None], Path], _GATE["_find_tool"])
_tool_version = cast(Callable[[str, Path], str], _GATE["_tool_version"])
ReleaseGateError = cast(type[Exception], _GATE["ReleaseGateError"])
_match_trna_loci = _GATE.get("_match_trna_loci")
_is_plausible_trna_locus = _GATE.get("_is_plausible_trna_locus")
_normalized_result_payload = _GATE.get("_normalized_result_payload")
_run_extraction_gate = _GATE.get("_run_extraction_gate")
_assert_thresholds = _GATE.get("_assert_thresholds")
_run_checked = _GATE.get("_run_checked")
_markdown_report = _GATE.get("_markdown_report")
_clear_report_outputs = _GATE.get("_clear_report_outputs")
_clean_wheel_probe = _GATE.get("_clean_wheel_probe")
_write_reports = _GATE.get("_write_reports")
_run_annotation_qc_gate = _GATE.get("_run_annotation_qc_gate")
_dependency_preflight = _GATE.get("_dependency_preflight")
_thresholds = cast(dict[str, tuple[str, float]], _GATE["THRESHOLDS"])


def test_release_preflight_excludes_external_trna_programs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert callable(_dependency_preflight)
    observed: list[str] = []
    globals_ = _dependency_preflight.__globals__

    def find_tool(name: str, _software_dir: Path | None) -> Path:
        observed.append(name)
        return Path(f"/tools/{name}")

    monkeypatch.setitem(globals_, "_find_tool", find_tool)
    monkeypatch.setitem(globals_, "_tool_version", lambda name, _path: f"{name} 1")

    tools, _versions = _dependency_preflight(None)

    assert observed == ["blastn", "makeblastdb", "tblastn"]
    assert set(tools) == {"blastn", "makeblastdb", "tblastn"}


def test_trna_matching_is_one_to_one() -> None:
    assert callable(_match_trna_loci)
    baseline = [
        (100, 170, "+", "trna(aaa)"),
        (120, 190, "+", "trna(aaa)"),
    ]
    observed = [(90, 200, "+", "trna(aaa)")]

    assert len(_match_trna_loci(baseline, observed)) == 1


def test_trna_matching_requires_compatible_strand_and_anticodon() -> None:
    assert callable(_match_trna_loci)
    baseline = [(100, 170, "+", "trnk(ttt)")]

    assert _match_trna_loci(baseline, [(100, 170, "-", "trnk(ttt)")]) == ()
    assert _match_trna_loci(baseline, [(100, 170, "+", "trnt(cgt)")]) == ()


def test_trna_locus_length_filter_rejects_oversized_features() -> None:
    assert callable(_is_plausible_trna_locus)

    assert _is_plausible_trna_locus((100, 170, "+", "trna(aaa)")) is True
    assert _is_plausible_trna_locus((100, 700, "+", "trna(aaa)")) is False


def test_result_normalization_ignores_environment_evidence() -> None:
    assert callable(_normalized_result_payload)
    first = {
        "status": "ok",
        "object_id": "result:sha256:" + "e" * 64,
        "artifacts": [
            {
                "kind": "annotation",
                "format": "json",
                "sha256": "a" * 64,
                "uri": "/workspace-a/annotation.json",
            },
            {
                "kind": "annotation_log",
                "format": "text",
                "sha256": "b" * 64,
                "uri": "/workspace-a/commands.jsonl",
            },
        ],
        "provenance": {
            "run_manifest_id": "annotation-run:sha256:" + "c" * 64,
            "argv": ["blastn", "-in", "<path>/input.fasta"],
            "started_at": "2026-07-13T00:00:00Z",
            "finished_at": "2026-07-13T00:00:01Z",
            "duration_seconds": 1.0,
        },
    }
    second = {
        **first,
        "object_id": "result:sha256:" + "f" * 64,
        "artifacts": [
            {**first["artifacts"][0], "uri": "/workspace-b/annotation.json"},
            {
                **first["artifacts"][1],
                "sha256": "d" * 64,
                "uri": "/workspace-b/commands.jsonl",
            },
        ],
        "provenance": {
            **first["provenance"],
            "started_at": "2026-07-13T01:00:00Z",
            "finished_at": "2026-07-13T01:00:02Z",
            "duration_seconds": 2.0,
        },
    }

    assert _normalized_result_payload(first) == _normalized_result_payload(second)


def test_result_normalization_keeps_content_hashes_and_manifest_identity() -> None:
    assert callable(_normalized_result_payload)
    result = {
        "status": "ok",
        "artifacts": [{"kind": "annotation", "format": "json", "sha256": "a" * 64, "uri": "/a"}],
        "provenance": {"run_manifest_id": "annotation-run:sha256:" + "b" * 64},
    }
    changed_artifact = {
        **result,
        "artifacts": [{**result["artifacts"][0], "sha256": "c" * 64}],
    }
    changed_manifest = {
        **result,
        "provenance": {"run_manifest_id": "annotation-run:sha256:" + "d" * 64},
    }

    assert _normalized_result_payload(result) != _normalized_result_payload(changed_artifact)
    assert _normalized_result_payload(result) != _normalized_result_payload(changed_manifest)


def test_release_extraction_uses_public_python_and_agent_paths(tmp_path: Path) -> None:
    assert callable(_run_extraction_gate)

    result, nad5_length, published_digest = _run_extraction_gate(
        Path(__file__).parents[1] / "data" / "Arabidopsis_thaliana.gb",
        tmp_path,
    )

    assert result.status == "ok"
    assert nad5_length == 2010
    assert len(published_digest) == 64


def test_release_threshold_requires_validated_pcg_recall_and_precision() -> None:
    assert callable(_assert_thresholds)
    metrics = {
        "validated_unique_pcg_recall": 0.92,
        "pcg_name_precision": 1.0,
        "rrna_recovered": 3,
        "trna_locus_recall": 0.90,
        "trna_match_precision": 0.90,
        "trna_hit_ratio": 1.0,
        "invalid_trna_lengths": 0,
        "invalid_structures": 0,
        "nad5_spliced_length": 2010,
    }

    _assert_thresholds(metrics)
    metrics["validated_unique_pcg_recall"] = 0.84
    with pytest.raises(ReleaseGateError, match="validated_unique_pcg_recall"):
        _assert_thresholds(metrics)


def test_failed_command_preserves_head_and_tail_diagnostics(tmp_path: Path) -> None:
    import os
    import sys

    assert callable(_run_checked)
    script = tmp_path / "fail.py"
    script.write_text(
        "import sys\n"
        "print('TRACEBACK-HEAD')\n"
        "print('x' * 12000)\n"
        "print('FAILURE-TAIL', file=sys.stderr)\n"
        "raise SystemExit(2)\n"
    )
    command = [sys.executable, str(script)]

    with pytest.raises(ReleaseGateError) as caught:
        _run_checked(command, cwd=tmp_path, env=os.environ)

    message = str(caught.value)
    assert "TRACEBACK-HEAD" in message
    assert "FAILURE-TAIL" in message


def test_markdown_report_discloses_rejected_and_excluded_candidates() -> None:
    assert callable(_markdown_report)
    metrics = {name: expected for name, (_operator, expected) in _thresholds.items()}
    metrics.update(
        {
            "shared_unique_pcg": 35,
            "baseline_unique_pcg": 39,
            "observed_unique_pcg": 36,
            "rejected_cds_count": 1,
            "rejected_cds_candidates": [
                {
                    "gene_name": "cox2",
                    "start": 10,
                    "end": 20,
                    "parts": [{"start": 10, "end": 20, "strand": 1}],
                    "issue_codes": ["missing_stop"],
                    "issue_messages": ["CDS has no terminal stop codon"],
                }
            ],
            "baseline_trna_loci": 134,
            "evaluable_baseline_trna_loci": 111,
            "excluded_intron_spanning_baseline_trna_loci": 23,
        }
    )
    report = {
        "metrics": metrics,
        "dependencies": {},
        "thresholds": {
            name: {"operator": operator, "value": expected}
            for name, (operator, expected) in _thresholds.items()
        },
        "clean_wheel": {"wheel_sha256": "a" * 64},
    }

    markdown = _markdown_report(report)

    assert "Rejected CDS candidates: 1" in markdown
    assert "Excluded intron-spanning baseline tRNA features: 23" in markdown


def test_release_report_cleanup_removes_stale_passed_outputs(tmp_path: Path) -> None:
    assert callable(_clear_report_outputs)
    json_report = tmp_path / "annotation-release-report.json"
    markdown_report = tmp_path / "annotation-release-report.md"
    json_report.write_text('{"status":"passed"}\n')
    markdown_report.write_text("Status: **passed**\n")
    bundle = tmp_path / "annotation-release-report"
    bundle.mkdir()
    (bundle / "annotation-release-report.json").write_text('{"status":"passed"}\n')

    _clear_report_outputs(tmp_path)

    assert not json_report.exists()
    assert not markdown_report.exists()
    assert not bundle.exists()


def test_release_reports_publish_as_one_atomic_directory(tmp_path: Path) -> None:
    assert callable(_write_reports)
    metrics = {name: expected for name, (_operator, expected) in _thresholds.items()}
    metrics.update(
        {
            "shared_unique_pcg": 1,
            "baseline_unique_pcg": 1,
            "observed_unique_pcg": 1,
            "rejected_cds_count": 0,
            "rejected_cds_candidates": [],
            "baseline_trna_loci": 1,
            "evaluable_baseline_trna_loci": 1,
            "excluded_intron_spanning_baseline_trna_loci": 0,
        }
    )
    report = {
        "metrics": metrics,
        "dependencies": {},
        "thresholds": {
            name: {"operator": operator, "value": expected}
            for name, (operator, expected) in _thresholds.items()
        },
        "clean_wheel": {"wheel_sha256": "a" * 64},
    }

    json_path, markdown_path = _write_reports(report, tmp_path)

    assert json_path.parent == markdown_path.parent == tmp_path / "annotation-release-report"
    assert json_path.is_file()
    assert markdown_path.is_file()


def test_release_report_failed_publish_preserves_prior_report_backup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert callable(_write_reports)
    metrics = {name: expected for name, (_operator, expected) in _thresholds.items()}
    metrics.update(
        {
            "shared_unique_pcg": 1,
            "baseline_unique_pcg": 1,
            "observed_unique_pcg": 1,
            "rejected_cds_count": 0,
            "rejected_cds_candidates": [],
            "baseline_trna_loci": 1,
            "evaluable_baseline_trna_loci": 1,
            "excluded_intron_spanning_baseline_trna_loci": 0,
        }
    )
    report = {
        "metrics": metrics,
        "dependencies": {},
        "thresholds": {
            name: {"operator": operator, "value": expected}
            for name, (operator, expected) in _thresholds.items()
        },
        "clean_wheel": {"wheel_sha256": "a" * 64},
    }
    _write_reports(report, tmp_path)

    os_module = _write_reports.__globals__["os"]
    real_replace = os_module.replace
    report_dir = tmp_path / "annotation-release-report"

    def fail_publish_and_rollback(source: Path, destination: Path) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        if destination_path == report_dir and (
            source_path.name.startswith(".annotation-release-report.tmp-")
            or source_path.name.startswith(".annotation-release-report.backup-")
        ):
            raise OSError("injected report switch failure")
        real_replace(source, destination)

    monkeypatch.setattr(os_module, "replace", fail_publish_and_rollback)

    with pytest.raises(ReleaseGateError, match="prior report is preserved"):
        _write_reports(report, tmp_path)

    backups = list(tmp_path.glob(".annotation-release-report.backup-*"))
    assert len(backups) == 1
    preserved = backups[0] / "annotation-release-report.json"
    assert preserved.is_file()
    assert '"wheel_sha256": "aaaaaaaa' in preserved.read_text()


def test_clean_wheel_probe_exercises_agent_json_equivalence() -> None:
    assert callable(_clean_wheel_probe)

    probe = _clean_wheel_probe(Path("/source/checkout"))

    assert "invoke_json" in probe
    assert "normalized_result" in probe
    assert "agent_annotation_status" in probe
    assert '"qc.annotation"' in probe
    assert '"qc.write"' in probe
    assert "agent_annotation_qc_status" in probe


def test_annotation_qc_gate_python_agent_reuse_and_bundle_roundtrip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import sys

    from tests.quality_control.annotation_helpers import annotation_result_fixture

    assert callable(_run_annotation_qc_gate)
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "source-cache"))
    # The synthetic annotation fixture is produced under the source checkout's
    # tests module; ensure the worktree src is importable for the in-process gate.
    source_root = Path(__file__).parents[2] / "src"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    direct = annotation_result_fixture(tmp_path / "source")

    gate = _run_annotation_qc_gate(direct, tmp_path / "work")

    assert gate["checks"]["annotation_qc_python_agent_identity"] is True
    assert gate["checks"]["annotation_qc_deterministic_reuse"] is True
    assert gate["checks"]["annotation_qc_bundle_roundtrip"] is True
    assert gate["checks"]["annotation_qc_zero_contract_failures"] is True
    annotation_qc = gate["annotation_qc"]
    assert annotation_qc["decision"] in {"ready", "needs_review"}
    assert annotation_qc["profile_id"] == "organelleverse.plant-mito-pcg.v1"
    assert annotation_qc["rejected_cds_candidate_count"] == 0
    assert annotation_qc["expected_profile_gene_count"] == 24
    assert annotation_qc["run_manifest_id"].startswith("annotation-qc-run:sha256:")
    bundle = tmp_path / "work" / "annotation-qc-bundle-parent" / "bundle"
    assert (bundle / "annotation_qc_report.json").is_file()
    assert (bundle / "gene_profile.csv").is_file()


def test_markdown_report_discloses_annotation_qc_section() -> None:
    assert callable(_markdown_report)
    metrics = {name: expected for name, (_operator, expected) in _thresholds.items()}
    metrics.update(
        {
            "shared_unique_pcg": 36,
            "baseline_unique_pcg": 39,
            "observed_unique_pcg": 36,
            "rejected_cds_count": 9,
            "rejected_cds_candidates": [],
            "baseline_trna_loci": 134,
            "evaluable_baseline_trna_loci": 111,
            "excluded_intron_spanning_baseline_trna_loci": 23,
        }
    )
    report = {
        "metrics": metrics,
        "dependencies": {},
        "thresholds": {
            name: {"operator": operator, "value": expected}
            for name, (operator, expected) in _thresholds.items()
        },
        "clean_wheel": {"wheel_sha256": "a" * 64},
        "annotation_qc": {
            "decision": "needs_review",
            "profile_id": "organelleverse.plant-mito-pcg.v1",
            "profile_recovery_fraction": 0.9167,
            "recovered_profile_gene_count": 22,
            "expected_profile_gene_count": 24,
            "missing_profile_genes": ["rps19", "sdh3"],
            "duplicated_profile_genes": [],
            "rejected_cds_candidate_count": 9,
            "report_object_id": "annotation_qc_report:sha256:" + "a" * 64,
            "run_manifest_id": "annotation-qc-run:sha256:" + "b" * 64,
            "bundle_tree_sha256": "c" * 64,
        },
    }

    markdown = _markdown_report(report)

    assert "## Annotation quality control" in markdown
    assert "`needs_review`" in markdown
    assert "`organelleverse.plant-mito-pcg.v1`" in markdown
    assert "22/24" in markdown
    assert "rps19, sdh3" in markdown
    assert "not a universal accuracy score" in markdown
