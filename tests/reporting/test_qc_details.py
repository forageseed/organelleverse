"""Coverage curve and tables of the QC detail fragments, on hand-built report data."""

from __future__ import annotations

from organelleverse.reporting.qc_details import _checks, _coverage


def _window(sequence, start, end, any_depth, confident_depth):
    return {
        "sequence_id": sequence,
        "start": start,
        "end": end,
        "any_alignment_mean_depth": any_depth,
        "confident_mean_depth": confident_depth,
    }


def test_a_single_window_is_still_drawn():
    html = _coverage([_window("ctg1", 0, 1000, 30.0, 25.0)])
    assert html.count("<polyline") == 2
    assert "1 sequence(s), 1,000 bp" in html


def test_missing_depth_breaks_the_line_and_zero_depth_is_marked():
    windows = [
        _window("ctg1", 0, 1000, 30.0, 30.0),
        _window("ctg1", 1000, 2000, 0.0, None),
        _window("ctg1", 2000, 3000, 28.0, 28.0),
    ]
    html = _coverage(windows)
    # confident track: the window without a value splits it into two runs
    assert html.count("class='cov-conf'") == 2
    assert html.count("class='cov-gap'") == 1
    assert "no alignment" in html


def test_sequences_are_laid_end_to_end_with_a_seam():
    windows = [_window("a", 0, 1000, 10.0, 10.0), _window("b", 0, 3000, 20.0, 20.0)]
    html = _coverage(windows)
    assert "2 sequence(s), 4,000 bp" in html
    assert html.count("class='cov-seam'") == 1
    assert "x1='250.0'" in html  # sequence b starts a quarter of the way along


def test_no_windows_draws_nothing():
    assert _coverage([]) == ""


def test_checks_list_attention_first_and_open_the_table():
    checks = [
        {"check_id": "qc.a", "status": "pass", "message": "fine"},
        {"check_id": "qc.b", "status": "warn", "message": "look <here>"},
        {"check_id": "qc.c", "status": "not_assessed", "message": "no data"},
        {"check_id": "qc.d", "status": "fail", "message": "broken"},
    ]
    html = _checks(checks)
    order = [html.index(f">qc.{name}<") for name in "dbca"]
    assert order == sorted(order)
    assert "<details class='ov-group' open>" in html
    assert "2 need attention" in html
    assert "look &lt;here&gt;" in html


def test_all_passing_checks_start_collapsed():
    html = _checks([{"check_id": "qc.a", "status": "pass", "message": "fine"}])
    assert "<details class='ov-group'>" in html
    assert "0 need attention" in html
