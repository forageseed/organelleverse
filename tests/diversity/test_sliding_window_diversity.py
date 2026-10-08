"""Sliding-window π + hotspot calling (DnaSP-style) — hand-computed and cross-checked."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from organelleverse._bio import write_fasta, write_genbank
from organelleverse.diversity.diversity import sliding_window_diversity

_NO_GAP = ">s0\nACGTACGTACGT\n>s1\nACGTACGTACGA\n>s2\nACGTACGTTCGT\n"  # segregating columns 9 (A/T) and 12 (T/A), both 2:1 splits


def _window(result, index: int) -> dict:
    return result.metrics["windows"][index]


def test_hand_computed_pi_seg_sites_and_coordinates(tmp_path: Path) -> None:
    fasta = tmp_path / "aln.fa"
    fasta.write_text(_NO_GAP)
    none = sliding_window_diversity(
        fasta, window_size=12, step=12, pi_threshold=None, top_fraction=0
    )
    window = _window(none, 0)
    assert (window["start"], window["end"]) == (1, 12)
    assert (window["ref_start"], window["ref_end"]) == (1, 12)
    assert window["pi"] == pytest.approx(4 / 36, abs=1e-6)  # (2/3 + 2/3) / 12
    assert window["segregating_sites"] == 2
    assert window["comparable_sites"] == 12
    assert list(none.metrics["hotspots"]) == []  # both rules disabled
    # Default rules: the single window has π = 0.111 ≥ 0.03 (absolute rule)
    # and ranks in the top 5% — either way it is one hotspot region.
    default = sliding_window_diversity(fasta, window_size=12, step=12)
    assert len(default.metrics["hotspots"]) == 1


def test_gap_modes_exclude_vs_pairwise(tmp_path: Path) -> None:
    fasta = tmp_path / "aln.fa"
    fasta.write_text(">s0\nACGTACGTACGT\n>s1\nACGTACGTACGA\n>s2\nACGT-CGTTCGT\n")
    excluded = sliding_window_diversity(
        fasta, window_size=12, step=12, gap_mode="exclude", pi_threshold=None, top_fraction=0
    )
    window = _window(excluded, 0)
    assert window["comparable_sites"] == 11  # gapped column dropped entirely
    assert window["pi"] == pytest.approx(4 / 33, abs=1e-6)
    pairwise = sliding_window_diversity(
        fasta, window_size=12, step=12, gap_mode="pairwise", pi_threshold=None, top_fraction=0
    )
    window = _window(pairwise, 0)
    assert window["comparable_sites"] == 12  # gapped column still compares s0/s1
    assert window["pi"] == pytest.approx(4 / 36, abs=1e-6)


def test_independent_pairwise_bruteforce_crosscheck(tmp_path: Path) -> None:
    """Naive all-pairs recomputation of every window's π (no shared code)."""
    import random

    rng = random.Random(42)
    n, length = 6, 240
    seqs = [list("".join(rng.choice("ACGT") for _ in range(length))) for _ in range(n)]
    for i in range(100, 160):
        for s in seqs[1:]:
            s[i] = rng.choice("ACGT")
    fasta = tmp_path / "aln.fa"
    write_fasta(fasta, [(f"s{i}", "".join(s)) for i, s in enumerate(seqs)])
    result = sliding_window_diversity(
        fasta, window_size=60, step=30, pi_threshold=None, top_fraction=0
    )
    upper = ["".join(s) for s in seqs]
    for window in result.metrics["windows"]:
        start, end = window["start"] - 1, window["end"]
        per_pair = []
        for a in range(n):
            for b in range(a + 1, n):
                diffs = sum(
                    upper[a][i] in "ACGT" and upper[b][i] in "ACGT" and upper[a][i] != upper[b][i]
                    for i in range(start, end)
                )
                per_pair.append(diffs / (end - start))
        expected = sum(per_pair) / len(per_pair)
        assert window["pi"] == pytest.approx(expected, abs=1e-6)


def test_hotspot_thresholds_and_zero_pi_guard(tmp_path: Path) -> None:
    # 20 columns; last three columns differentiate d/c/b from a progressively
    seq = (
        ">a\nAAAAAAAAAAAAAAAAAAAA\n>b\nAAAAAAAAAAAAAAAAAAAT\n"
        ">c\nAAAAAAAAAAAAAAAAAATT\n>d\nAAAAAAAAAAAAAAAAATTT\n"
    )
    fasta = tmp_path / "aln.fa"
    fasta.write_text(seq)
    result = sliding_window_diversity(
        fasta, window_size=5, step=5, pi_threshold=0.1, top_fraction=0.25
    )
    pis = [w["pi"] for w in result.metrics["windows"]]
    assert len(pis) == 4
    assert pis[:3] == [0.0, 0.0, 0.0]
    assert pis[3] == pytest.approx((0.5 + 4 / 6 + 0.5) / 5, abs=1e-6)
    hotspots = result.metrics["hotspots"]
    # absolute rule (π ≥ 0.1) and top-25% rule both select only the last
    # window; the π = 0 windows are never selected.
    assert [h["start"] for h in hotspots] == [16]
    assert hotspots[0]["pi"] == pytest.approx(pis[3], abs=1e-6)


def test_hotspot_merging_recomputes_span_stats(tmp_path: Path) -> None:
    # one variable block (columns 121-180) flanked by conserved sequence
    import random

    rng = random.Random(7)
    ref = "".join(rng.choice("ACGT") for _ in range(300))
    seqs = [list(ref) for _ in range(4)]
    for i in range(120, 180):
        for s in seqs[1:]:
            s[i] = rng.choice("ACGT")
    fasta = tmp_path / "aln.fa"
    write_fasta(fasta, [(f"s{i}", "".join(s)) for i, s in enumerate(seqs)])
    result = sliding_window_diversity(fasta, window_size=100, step=50)
    hotspots = result.metrics["hotspots"]
    assert len(hotspots) == 1
    hotspot = hotspots[0]
    assert (hotspot["start"], hotspot["end"]) == (51, 250)
    assert hotspot["n_windows"] == 3
    assert hotspot["max_window_pi"] == max(w["pi"] for w in result.metrics["windows"])
    # span π recomputed over 51-250: variable block is 60 of 200 columns
    assert 0.1 < hotspot["pi"] < 0.4
    assert hotspot["segregating_sites"] == sum(
        1 for i in range(50, 250) if len({s[i] for s in seqs}) > 1
    )


def test_genbank_labelling_genes_and_spacers(tmp_path: Path) -> None:
    fasta = tmp_path / "aln.fa"
    fasta.write_text(">s0\n" + "A" * 12 + "\n>s1\n" + "A" * 12 + "\n")
    genbank = tmp_path / "ref.gb"
    write_genbank(
        genbank,
        [
            (
                "ref",
                "A" * 12,
                [
                    {
                        "type": "gene",
                        "start": 0,
                        "end": 4,
                        "strand": 1,
                        "qualifiers": {"gene": ["trnH"]},
                    },
                    {
                        "type": "gene",
                        "start": 8,
                        "end": 12,
                        "strand": 1,
                        "qualifiers": {"gene": ["psbA"]},
                    },
                ],
            )
        ],
    )
    result = sliding_window_diversity(
        fasta, window_size=4, step=4, genbank=genbank, pi_threshold=None, top_fraction=0
    )
    labels = [w.get("label") for w in result.metrics["windows"]]
    assert labels[0] == "trnH"  # 1-4 fully inside trnH
    assert labels[1] == "trnH-psbA"  # 5-8: intergenic spacer naming
    assert labels[2] == "psbA"  # 9-12 fully inside psbA


def test_reference_coordinate_mapping_skips_ref_gaps(tmp_path: Path) -> None:
    fasta = tmp_path / "aln.fa"
    fasta.write_text(">ref\nACG-TACG-TAC\n>other\nACGATACGATAC\n")
    result = sliding_window_diversity(
        fasta, window_size=12, step=12, pi_threshold=None, top_fraction=0
    )
    window = _window(result, 0)
    assert (window["ref_start"], window["ref_end"]) == (1, 10)
    assert (window["start"], window["end"]) == (1, 12)


def test_error_paths(tmp_path: Path) -> None:
    one = tmp_path / "one.fa"
    one.write_text(">only\nACGT\n")
    result = sliding_window_diversity(one)
    assert result.status == "failed"
    assert result.errors[0].code == "diversity.sliding_window_diversity.too_few_sequences"

    two = tmp_path / "two.fa"
    two.write_text(">a\nACGT\n>b\nACGT\n")
    result = sliding_window_diversity(two, gap_mode="bogus")
    assert result.status == "failed"
    assert result.errors[0].code == "diversity.sliding_window_diversity.unknown_gap_mode"
    result = sliding_window_diversity(two, window_size=1)
    assert result.status == "failed"
    assert result.errors[0].code == "diversity.sliding_window_diversity.bad_window"
    result = sliding_window_diversity(two, reference_index=5)
    assert result.status == "failed"
    assert result.errors[0].code == "diversity.sliding_window_diversity.bad_reference"


def test_top_fraction_selects_ceil_k_windows() -> None:
    """top_fraction must select exactly ceil(f·n) windows (regression: k stayed
    an int only when it was 1; ceil>1 crashed with a float index)."""
    seq = ">a\n" + "A" * 200 + "\n>b\n" + "A" * 155 + "TT" + "A" * 3 + "TTTT" + "A" * 36 + "\n"
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        fasta = Path(tmp) / "aln.fa"
        fasta.write_text(seq)
        # 40 windows of 5 bp; variation in windows 32 (cols 156-160) and
        # 33 (cols 161-165); top 5% of 40 = 2 → both variable windows qualify.
        result = sliding_window_diversity(
            fasta, window_size=5, step=5, pi_threshold=None, top_fraction=0.05
        )
    windows = list(result.metrics["windows"])
    assert len(windows) == 40
    assert [w["pi"] for w in windows] == [0.0] * 31 + [0.4, 0.8] + [0.0] * 7
    hotspots = list(result.metrics["hotspots"])
    assert len(hotspots) == 1  # adjacent qualifying windows merge
    assert hotspots[0]["n_windows"] == 2


def test_result_is_finite_json(tmp_path: Path) -> None:
    fasta = tmp_path / "aln.fa"
    fasta.write_text(_NO_GAP)
    result = sliding_window_diversity(fasta)
    json.dumps(result.model_dump(mode="json"), allow_nan=False)


def test_single_sample_driven_hotspot_and_two_sample_unavailable(tmp_path):
    fasta = tmp_path / "outlier.fa"
    write_fasta(fasta, [("a", "A" * 20), ("b", "A" * 20), ("c", "A" * 20), ("outlier", "T" * 20)])
    result = sliding_window_diversity(fasta, window_size=20)
    for row in [result.metrics["windows"][0], result.metrics["hotspots"][0]]:
        assert row["single_sample_driven"] is True
        assert row["driving_sample"] == "outlier"
        assert row["max_pi_drop_fraction"] == 1
        assert row["leave_one_out"][3]["pi"] == 0
    assert "single_sample_driven_hotspots" in result.flags
    write_fasta(fasta, [("a", "A" * 20), ("b", "T" * 20)])
    assert (
        sliding_window_diversity(fasta, window_size=20).metrics["windows"][0][
            "single_sample_driven"
        ]
        is None
    )


@pytest.mark.parametrize("gap_mode", ["exclude", "pairwise"])
def test_leave_one_out_against_brute_force_with_missing_bases(tmp_path, gap_mode):
    import random

    rng = random.Random(85)
    seqs = ["".join(rng.choices("ACGT-N", k=80)) for _ in range(5)]
    path = write_fasta(tmp_path / "gaps.fa", [(str(i), seq) for i, seq in enumerate(seqs)])
    result = sliding_window_diversity(path, window_size=40, step=20, gap_mode=gap_mode)
    for window in result.metrics["windows"]:
        for omitted, row in enumerate(window["leave_one_out"]):
            values = []
            for j in range(window["start"] - 1, window["end"]):
                if gap_mode == "exclude" and not all(s[j] in "ACGT" for s in seqs):
                    continue
                col = [s[j] for i, s in enumerate(seqs) if i != omitted and s[j] in "ACGT"]
                if len(col) >= 2:
                    values.append(
                        sum(col[a] != col[b] for a in range(len(col)) for b in range(a))
                        / (len(col) * (len(col) - 1) / 2)
                    )
            assert row["comparable_sites"] == len(values)
            if values:
                assert row["pi"] == pytest.approx(sum(values) / len(values), abs=1e-6)
            else:
                assert row["pi"] is None


def test_orientation_pipeline_uses_normalized_sequences_and_retains_report(tmp_path, monkeypatch):
    from Bio.Seq import reverse_complement as rc

    import organelleverse.phylogeny as phylogeny
    from organelleverse.core.result import OrganelleResult
    from tests.comparative.test_orientation import plastome

    lsc, ir, ssc = plastome()
    original = lsc + ir + ssc + rc(ir)
    path = write_fasta(
        tmp_path / "plastomes.fa",
        [("a", original), ("b", original), ("c", lsc + ir + rc(ssc) + rc(ir))],
    )

    def align(path, method):
        from organelleverse._bio import read_fasta

        records = read_fasta(path)
        assert {seq for _, seq in records} == {original}
        return OrganelleResult(
            operation_id="phylogeny.align",
            scope="plastid",
            status="ok",
            metrics={"alignment": records, "method": "test_aligner"},
        )

    monkeypatch.setattr(phylogeny, "align", align)
    result = sliding_window_diversity(path, normalize_orientation=True)
    assert result.status == "ok"
    assert result.metrics["mean_pi"] == 0
    assert result.metrics["orientation"]["samples"][2]["ssc_reverse_complemented"]
    write_fasta(path, [("a", original), ("irless", lsc)])
    failed = sliding_window_diversity(path, normalize_orientation=True)
    assert failed.status == "failed"
    assert failed.metrics["orientation"]["unresolved"][0]["sample"] == "irless"


def test_orientation_pipeline_rejects_placeholder(tmp_path, monkeypatch):
    import organelleverse.phylogeny as phylogeny
    from organelleverse.core.errors import OrganelleDependencyError
    from organelleverse.core.result import OrganelleResult
    from tests.comparative.test_orientation import plastome, rc

    lsc, ir, ssc = plastome()
    seq = lsc + ir + ssc + rc(ir)
    path = write_fasta(tmp_path / "plastomes.fa", [("a", seq), ("b", seq)])
    monkeypatch.setattr(
        phylogeny,
        "align",
        lambda *args, **kwargs: OrganelleResult(
            operation_id="phylogeny.align",
            scope="plastid",
            status="ok",
            flags=("placeholder_alignment",),
        ),
    )
    with pytest.raises(OrganelleDependencyError, match="MAFFT"):
        sliding_window_diversity(path, normalize_orientation=True)


@pytest.mark.parametrize("gap_mode", ["exclude", "pairwise"])
def test_leave_one_out_exact_half_threshold_and_ties(tmp_path, gap_mode):
    path = write_fasta(
        tmp_path / "half.fa",
        [
            ("a", "AAAAA" * 41),
            ("b", "AAAAT" * 41),
            ("c", "AAATT" * 41),
            ("d", "TTTAA" * 41),
        ],
    )
    result = sliding_window_diversity(path, window_size=5, step=5, gap_mode=gap_mode)
    for row in result.metrics["windows"]:
        assert row["single_sample_driven"] is True
        assert row["max_pi_drop_fraction"] == 0.5
        assert row["driving_sample"] == "d"
