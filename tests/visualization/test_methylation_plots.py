"""Tests for plant organelle methylation profile figures (ov.viz.methylation_profile)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import organelleverse as ov
from organelleverse.core.errors import OrganelleInputError, OrganelleParameterError
from organelleverse.visualization import compose_figure, methylation_profile, plot_methylation_profile


def _make_mock_ovasm_methylation_run(root: Path, dual: bool = True) -> Path:
    organelles = [("mitochondrion", 20000), ("plastid", 10000)] if dual else [("mitochondrion", 20000)]
    for org, seq_len in organelles:
        mdir = root / org / "methylation"
        mdir.mkdir(parents=True, exist_ok=True)
        report = {
            "schema": "ovasm.methylation.v1",
            "parameters": {
                "prob_threshold": 0.5,
                "cg_threshold_policy": "auto",
                "effective_max_cg_methylation": 0.38,
                "min_read_cpgs": 5,
                "max_overhang_bp": 300,
                "kmer": 21,
            },
            "reads": {
                "total_bam_primary_reads": 60,
                "reads_with_mm_ml_tags": 60,
                "organelle_placed_reads": 50,
                "clean_organelle_reads": 40,
                "numt_nupt_hypermethylated_reads": 8,
                "numt_nupt_chimeric_reads": 2,
                "organelle_cg_median": 0.018,
                "numt_nupt_cg_median": 0.81,
                "roc_auc_organelle_vs_numt": 0.992,
            },
            "contigs": [
                {
                    "name": "c1",
                    "length": seq_len,
                    "circular": True,
                }
            ],
        }
        (mdir / "methylation.json").write_text(json.dumps(report), encoding="utf-8")

        clean_lines = ["#chrom\tstart\tend\tcontext\tstrand\tmod_rate\tcov\tmod_count\tunmod_count"]
        unfilt_lines = ["#chrom\tstart\tend\tcontext\tstrand\tmod_rate\tcov\tmod_count\tunmod_count"]
        for pos in range(100, seq_len, 500):
            clean_lines.append(f"c1\t{pos}\t{pos + 2}\tCG\t.\t0.020000\t50\t1\t49")
            spike = 20 if pos >= seq_len // 2 else 1
            unfilt_lines.append(
                f"c1\t{pos}\t{pos + 2}\tCG\t.\t{spike / 50:.6f}\t50\t{spike}\t{50 - spike}"
            )
        (mdir / "methylation.bed").write_text("\n".join(clean_lines) + "\n", encoding="utf-8")
        (mdir / "methylation.unfiltered.bed").write_text("\n".join(unfilt_lines) + "\n", encoding="utf-8")

        tsv_lines = [
            "read_name\tprimary_target\tread_len\torganelle_span\tnuclear_overhang\tclass\t"
            "cg_calls\tcg_frac\tchg_calls\tchg_frac\tchh_calls\tchh_frac\tm6a_calls\tm6a_frac"
        ]
        for i in range(40):
            tsv_lines.append(f"org_{i}\tc1\t8000\t8000\t0\torganelle\t40\t0.0200\t60\t0.0250\t200\t0.0150\t300\t0.0300")
        for i in range(10):
            tsv_lines.append(
                f"numt_{i}\tc1\t8000\t7000\t1000\tnumt_hypermethylated\t40\t0.8200\t60\t0.5100\t200\t0.1100\t300\t0.0300"
            )
        (mdir / "reads.methylation.tsv").write_text("\n".join(tsv_lines) + "\n", encoding="utf-8")

    return root


def test_methylation_profile_renders_dual_organelle_run_and_sidecar_tsv(tmp_path: Path) -> None:
    run_dir = _make_mock_ovasm_methylation_run(tmp_path / "run", dual=True)
    paf = tmp_path / "transfers.paf"
    paf.write_text("c1\t20000\t10000\t19500\t+\tchr4\t80000000\t100000\t109500\t9450\t9500\t60\n", encoding="utf-8")

    plot = methylation_profile(run_dir, transfer_paf=paf, window_bp=2000, width="double", dpi=200)
    assert plot.status == "ok"
    assert plot.scope == "mixed"
    assert plot.metrics["organelle_count"] == 2
    assert plot.metrics["clean_reads_plotted"] == 80
    assert plot.metrics["numt_nupt_reads_plotted"] == 20

    out_png = tmp_path / "out" / "meth.png"
    res = ov.write(plot, out_png)
    assert out_png.is_file() and out_png.stat().st_size > 1000
    sidecar = out_png.with_suffix(".windows.tsv")
    assert sidecar.is_file()
    lines = sidecar.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("organelle\tstart\tend\tmid_kb")
    assert len(lines) > 10
    assert len(res.artifacts) >= 2


def test_methylation_profile_composes_into_multi_panel_figure(tmp_path: Path) -> None:
    run_dir = _make_mock_ovasm_methylation_run(tmp_path / "single", dual=False)
    p1 = plot_methylation_profile(run_dir / "mitochondrion" / "methylation", width="single", dpi=150)
    p2 = plot_methylation_profile(run_dir / "mitochondrion", width="single", dpi=150)
    comp = compose_figure([p1, p2], titles=["Track A", "Track B"], width="double", dpi=150)
    out_svg = tmp_path / "composed.svg"
    ov.write(comp, out_svg)
    assert out_svg.is_file() and "<svg" in out_svg.read_text(encoding="utf-8")


def test_methylation_profile_rejects_invalid_inputs_and_parameters(tmp_path: Path) -> None:
    with pytest.raises(OrganelleInputError):
        plot_methylation_profile(tmp_path / "does_not_exist")

    run_dir = _make_mock_ovasm_methylation_run(tmp_path / "valid", dual=False)
    with pytest.raises(OrganelleParameterError):
        plot_methylation_profile(run_dir, context="INVALID")
    with pytest.raises(OrganelleParameterError):
        plot_methylation_profile(run_dir, window_bp=50)
