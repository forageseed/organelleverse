"""Real-data validation regressions for the ``variation`` domain.

Defects found by validating ``variation.snp`` / ``snp_density`` /
``compute_snp`` against an independent numpy implementation and against
minimap2 -cx asm10 --cs + paftools.js call on 10 real Triticeae plastomes
(141,770 alignment columns, 11,838 SNPs).
"""

from __future__ import annotations

from pathlib import Path

from organelleverse.variation import compute_snp, snp_density

_ALN = """>ref
ACGTACGTAC
>s1
ACGTACGAAC
>s2
ACGTTCGTAC
"""


def _write(path: Path) -> Path:
    path.write_text(_ALN, encoding="utf-8")
    return path


def test_snp_density_emits_the_final_partial_window(tmp_path: Path) -> None:
    """``range(0, L - window + 1, step)`` dropped the trailing partial window:
    on the real Triticeae alignment 1,770 of 141,770 columns (1.25%, holding 20
    SNPs) were inside no window at all."""
    aln = _write(tmp_path / "aln.fa")
    result = snp_density(aln, window_size=5, step=5)
    windows = list(result.metrics["windowed_density"])
    assert windows[-1]["end"] == 10  # L=10 with window 5, step 5
    assert sum(w["end"] - w["start"] + 1 for w in windows) == 10


def test_snp_density_partial_window_is_smaller_and_still_reported(
    tmp_path: Path,
) -> None:
    aln = _write(tmp_path / "aln.fa")
    result = snp_density(aln, window_size=4, step=4)
    windows = list(result.metrics["windowed_density"])
    # windows: 1-4, 5-8, 9-10 (partial, 2 columns)
    assert [(w["start"], w["end"]) for w in windows] == [(1, 4), (5, 8), (9, 10)]
    assert windows[-1]["snp_count"] == sum(
        1
        for _, seq in [("ref", "ACGTACGTAC"), ("s1", "ACGTACGAAC"), ("s2", "ACGTTCGTAC")]
        if _ != "ref"
        for i in range(8, 10)
        if seq[i] != "ACGTACGTAC"[i] and seq[i] in "ACGT"
    )
    assert result.metrics["columns_covered"] == 10


def test_snp_density_reports_uncovered_columns_when_windows_are_gapped(
    tmp_path: Path,
) -> None:
    aln = _write(tmp_path / "aln.fa")
    result = snp_density(aln, window_size=3, step=6)
    # L=10, windows 1-3 and 7-9 -> columns 4-6 and 10 sit in no window
    assert result.metrics["columns_covered"] == 6
    assert result.metrics["columns_not_covered"] == 4


def test_snp_density_tiles_every_column_when_step_equals_window(
    tmp_path: Path,
) -> None:
    aln = _write(tmp_path / "aln.fa")
    result = snp_density(aln, window_size=3, step=3)
    assert result.metrics["columns_covered"] == 10
    assert result.metrics["columns_not_covered"] == 0


def test_snp_counts_are_unchanged_by_the_window_fix(tmp_path: Path) -> None:
    """The per-sequence SNP totals must stay exactly as validated."""
    aln = _write(tmp_path / "aln.fa")
    core = compute_snp(aln)
    assert core["per_sequence"] == [{"name": "ref", "snp_count": 0},
                                    {"name": "s1", "snp_count": 1},
                                    {"name": "s2", "snp_count": 1}]
    assert core["total_snps"] == 2
