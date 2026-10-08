from __future__ import annotations

from pathlib import Path

import pytest

from organelleverse.assembly import gold_mode as gm
from organelleverse.assembly.graph_compare import Link


def _evidence(links, branches, phased=0, ambiguous=()):
    return {
        "links": [
            {"from": a, "to": b, "reads": n, "ambiguous_reads": 7 if (a, b) in ambiguous else 0}
            for a, b, n in links
        ],
        "links_supported": sum(1 for *_, n in links if n > 0),
        "branches": [
            {
                "from": f,
                "alternatives": [{"to": t, "reads": n} for t, n in alts],
                "minority_fraction": mf,
                "minority_ci95": [mf - 0.05, mf + 0.05],
            }
            for f, alts, mf in branches
        ],
        "phased_paths": [{"path": "x", "reads": 1}] * phased,
        "repeat_segments": [],
    }


def test_multi_platform_tells_ambiguous_and_silent_from_support():
    links = [("u0+", "u1+", 30), ("u1+", "u2+", 12), ("u2+", "u3+", 9)]
    hifi = _evidence(links, [("u0+", [("u1+", 30), ("u2+", 20)], 0.4)], 3)
    # short reads: one link supported, one crossed but not told apart, one not reached
    short = _evidence(
        [("u0+", "u1+", 8), ("u1+", "u2+", 0), ("u2+", "u3+", 0)],
        [],
        0,
        ambiguous={("u1+", "u2+")},
    )
    mp = gm._multi_platform({"hifi": (Path("a"), hifi), "short": (Path("b"), short)})
    assert mp["links_total"] == 3
    assert mp["links_confirmed_by_two_platforms"] == 1
    assert mp["per_platform"]["short"] == {"supported": 1, "ambiguous_only": 1, "silent": 1}
    assert mp["per_platform"]["hifi"] == {"supported": 3, "ambiguous_only": 0, "silent": 0}
    assert mp["links"]["u0+ -> u1+"] == {"hifi": 30, "short": 8}
    assert mp["branches"]["u0+"]["hifi"]["minority_fraction"] == 0.4
    assert mp["phased_paths"] == {"hifi": 3, "short": 0}


def test_checks_fail_on_unsupported_link_and_residual_calls():
    cert = {
        "evidence_primary": {"links": 4, "links_supported": 3},
        "multi_platform": {
            "platforms": ["hifi"],
            "links_total": 4,
            "links_confirmed_by_two_platforms": 0,
            "per_platform": {"hifi": {"supported": 3, "ambiguous_only": 0, "silent": 1}},
        },
        "unringing": {"molecules": 1, "unsolved_components": 0, "decisive": False},
        "polishing": {"residual_calls_after_polishing": {"hifi": 0, "short": 2}},
    }
    checks = {c["name"]: c["passed"] for c in gm._checks(cert)}
    assert checks["every link supported by the primary platform's reads"] is False
    assert checks["the graph unrings into molecules"] is True
    assert checks["no confident change left after polishing"] is False
    # a single platform gives no cross-platform check
    assert "every link confirmed by a second platform" not in checks


def test_apply_edits_right_to_left_and_skips_mismatched_ref(tmp_path):
    fa = tmp_path / "m.fa"
    fa.write_text(">m1 circular\nACGTACGTAC\nGT\n")
    out = tmp_path / "p.fa"
    # SNV at 2, deletion at 5 (ACG -> A), insertion at 10, and a stale edit that no longer matches
    gm._apply(
        fa,
        [("m1", 2, "C", "G"), ("m1", 5, "ACG", "A"), ("m1", 10, "C", "CTT"), ("m1", 1, "T", "G")],
        out,
    )
    lines = out.read_text().splitlines()
    assert lines[0] == ">m1 circular"
    assert "".join(lines[1:]) == "AGGTATACTTGT"


def test_run_gold_mode_needs_reads(tmp_path):
    with pytest.raises(ValueError):
        gm.run_gold_mode(gm.GoldInputs(sample="s", out_dir=tmp_path, seeds={}))


def test_markdown_renders_every_section():
    cert = {
        "sample": "NIP",
        "organelle": "mitochondrion",
        "checks": [{"name": "c", "passed": True, "detail": "1/1"}],
        "unringing": {
            "molecule_list": [{"length": 376023, "circular": False, "path": "u0+"}],
            "decisive": False,
            "margin_ln": 0.35,
        },
        "multi_platform": {
            "platforms": ["hifi", "noisy"],
            "branches": {"u0-": {"hifi": {"minority_fraction": 0.406}}},
            "phased_paths": {"hifi": 9, "noisy": 7},
        },
        "cross_validation": {
            "tools": {
                "oatk": {
                    "ovasm_junctions_found": 16,
                    "ovasm_junctions": 20,
                    "its_junctions_found_in_ovasm": 12,
                    "its_junctions": 12,
                },
                "pmat": {"note": "graph has no links"},
            }
        },
        "polishing": {
            "polished_with": "hifi",
            "edits": 2,
            "edit_kinds": {"snv": 2},
            "residual_calls_after_polishing": {"hifi": 0},
        },
        "reference_comparison": {
            "reference": "NC_011033",
            "snvs": 10,
            "insertions": 1,
            "insertion_bp": 3,
            "deletions": 2,
            "deletion_bp": 4,
            "indels_50bp_or_more": 0,
            "primary_alignment_blocks": 5,
            "reference_aligned_fraction": 0.95,
        },
        "steps": [{"name": "unring", "status": "completed", "seconds": 0.1}],
    }
    md = gm._markdown(cert)
    for text in (
        "| c | pass | 1/1 |",
        "376,023",
        "40.6%",
        "| u0- | 40.6% | - |",
        "16/20",
        "graph has no links",
        "10 SNVs",
        "95.00%",
        "| unring | completed |",
    ):
        assert text in md


def test_failed_optional_step_is_recorded_not_raised(tmp_path):
    run = gm._Run(tmp_path)

    def boom():
        raise RuntimeError("tool missing")

    assert run.step("external_pmat", boom, fatal=False) is None
    step = run.record["steps"][0]
    assert step["status"] == "failed" and "tool missing" in step["error"]
    assert (tmp_path / "logs" / "external_pmat.traceback.txt").is_file()
    with pytest.raises(RuntimeError):
        run.step("assemble", boom)


def test_second_platform_check_passes_when_every_link_has_two():
    mp = {
        "platforms": ["hifi", "noisy", "short"],
        "links_total": 2,
        "links_confirmed_by_two_platforms": 2,
        "per_platform": {
            "hifi": {"supported": 2, "ambiguous_only": 0, "silent": 0},
            "noisy": {"supported": 2, "ambiguous_only": 0, "silent": 0},
            "short": {"supported": 0, "ambiguous_only": 2, "silent": 0},
        },
    }
    (check,) = gm._checks({"multi_platform": mp})
    assert check["passed"] is True
    assert "short: 0 supported, 2 ambiguous only, 0 silent" in check["detail"]


def test_same_assembler_short_read_graph_is_not_independent_confirmation(monkeypatch):
    links = [Link("a", "+", "b", "+", 0), Link("a", "+", "c", "+", 0)]
    monkeypatch.setattr(gm, "_tool", lambda _: "minimap2")
    monkeypatch.setattr(gm, "parse_gfa", lambda _: ({}, links))

    def score(gold, candidate, *args):
        missing = ["J02"] if candidate.name == "oatk.gfa" else []
        return {
            "missing_junctions_tolerant": missing,
            "junctions_recovered_tolerant": 2 - len(missing),
            "gold_junctions": 2,
        }

    monkeypatch.setattr(gm, "score_graphs", score)
    report = gm._concordance(
        Path("primary.gfa"), {"oatk": Path("oatk.gfa"), "ovasm-short": Path("short.gfa")}
    )
    assert report["junctions_confirmed_by_another_graph"] == 2
    assert report["junctions_confirmed_by_independent_assembler"] == 1
    assert report["per_junction_independent"]["a+ -> c+"] == []
    (check,) = gm._checks({"cross_validation": report})
    assert check["passed"] is False
    assert "1/2" in check["detail"]
    short_only = gm._concordance(Path("primary.gfa"), {"ovasm-short": Path("short.gfa")})
    assert short_only["independent_assemblers"] == []
    assert gm._checks({"cross_validation": short_only})[0]["passed"] is False


def test_failed_polish_or_skipped_external_tool_keeps_certificate_incomplete():
    cert = {
        "steps": [
            {"name": "assemble", "status": "completed"},
            {"name": "polish", "status": "failed"},
        ]
    }
    (check,) = gm._checks(cert)
    assert check["passed"] is False
    assert "polish" in check["detail"]
    cert["steps"] = [{"name": "assemble", "status": "completed"}]
    cert["external_skipped"] = {"getorganelle": "needs short reads"}
    assert gm._checks(cert)[0]["passed"] is False
    cert["external_skipped"] = {}
    assert gm._checks(cert)[0]["passed"] is True
