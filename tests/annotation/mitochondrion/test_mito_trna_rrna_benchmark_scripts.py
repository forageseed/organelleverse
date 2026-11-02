from __future__ import annotations

import csv
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqFeature import FeatureLocation, SeqFeature
from Bio.SeqRecord import SeqRecord

from tests._paths import PROJECT_ROOT

ROOT = PROJECT_ROOT
SCRIPTS = ROOT / "scripts" / "annotation_benchmarks"


def _load_script(name: str):
    path = SCRIPTS / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_benchmark_scripts_exist_outside_package_runtime():
    expected = [
        SCRIPTS / "fetch_trna_rrna_tools.sh",
        SCRIPTS / "run_mito_trna_rrna_benchmark.py",
        SCRIPTS / "run_multispecies_trna_rrna_validation.py",
        SCRIPTS / "compare_trna_rrna_annotations.py",
    ]

    for path in expected:
        assert path.exists()
        assert "/src/_vendor" not in path.read_text()


def test_pmga_gff_parser_extracts_trna_and_rrna_rows(tmp_path: Path):
    runner = _load_script("run_mito_trna_rrna_benchmark.py")
    gff = tmp_path / "mini.gff"
    gff.write_text(
        "\n".join(
            [
                "##gff-version 3",
                "seq\tmaker\ttRNA\t10\t80\t.\t+\t.\tID=t1;Name=trnK-UUU",
                "seq\tmaker\trRNA\t100\t250\t.\t-\t.\tID=r1;Name=rrn18",
                "seq\tmaker\tCDS\t300\t900\t.\t+\t.\tID=c1;Name=cox1",
            ]
        )
    )

    rows = runner.rows_from_pmga_gff(gff, "mini")

    assert [row["feature_type"] for row in rows] == ["tRNA", "rRNA"]
    assert rows[0]["gene"] == "trnK(ttt)"
    assert rows[1]["gene"] == "rrn18"


def test_genbank_parser_extracts_trna_and_rrna_baseline_rows(tmp_path: Path):
    runner = _load_script("run_mito_trna_rrna_benchmark.py")
    gb = tmp_path / "mini.gb"
    rec = SeqRecord(Seq("A" * 300), id="mini", name="mini")
    rec.annotations["molecule_type"] = "DNA"
    rec.features = [
        SeqFeature(
            FeatureLocation(9, 80, strand=1),
            type="tRNA",
            qualifiers={"product": ["tRNA-Lys"], "anticodon": ["(pos:20..22,aa:Lys,seq:ttt)"]},
        ),
        SeqFeature(
            FeatureLocation(99, 250, strand=-1),
            type="rRNA",
            qualifiers={"gene": ["rrn18"]},
        ),
    ]
    SeqIO.write(rec, gb, "genbank")

    rows = runner.rows_from_genbank(gb, "mini")

    assert [(row["feature_type"], row["gene"]) for row in rows] == [
        ("tRNA", "trnK(ttt)"),
        ("rRNA", "rrn18"),
    ]
    assert rows[0]["start"] == "10"
    assert rows[0]["end"] == "80"
    assert rows[0]["method"] == "genbank"


def test_multispecies_runner_uses_manifest_and_genbank_baseline(tmp_path: Path, monkeypatch):
    multi = _load_script("run_multispecies_trna_rrna_validation.py")
    gb = tmp_path / "mini.gb"
    rec = SeqRecord(Seq("A" * 120), id="mini", name="mini")
    rec.annotations["molecule_type"] = "DNA"
    rec.features = [
        SeqFeature(
            FeatureLocation(9, 80, strand=1),
            type="tRNA",
            qualifiers={"product": ["tRNA-Phe"], "anticodon": ["(pos:20..22,aa:Phe,seq:gaa)"]},
        )
    ]
    SeqIO.write(rec, gb, "genbank")
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text(
        "\t".join(["sample", "fasta", "baseline_gff", "baseline_genbank", "baseline_method"])
        + f"\nmini\t\t\t{gb}\tgenbank\n"
    )
    output = tmp_path / "combined.tsv"

    monkeypatch.setattr(multi.runner, "find_tool", lambda software_dir, name: None)
    monkeypatch.setattr(
        multi.runner,
        "run_trnascan",
        lambda fasta, out_dir, sample, tool: [
            multi.runner.benchmark_row(
                sample=sample,
                method="tRNAscan-SE",
                feature_type="tRNA",
                gene="trnF(gaa)",
                start="10",
                end="80",
                strand="+",
                score="99",
                status="ok",
                source_file=str(fasta),
            )
        ],
    )
    monkeypatch.setattr(multi.runner, "run_aragorn", lambda *args, **kwargs: [])
    monkeypatch.setattr(multi.runner, "run_barrnap", lambda *args, **kwargs: [])

    rows = multi.run_manifest(
        manifest,
        output=output,
        work_dir=tmp_path / "work",
        software_dir=None,
        include_native_trna=False,
        include_cleanroom_trna=False,
        native_db=None,
    )

    assert output.exists()
    assert (tmp_path / "work" / "mini" / "mini.fasta").exists()
    assert [(row["method"], row["gene"]) for row in rows] == [
        ("genbank", "trnF(gaa)"),
        ("tRNAscan-SE", "trnF(gaa)"),
    ]


def test_compare_script_summarizes_unique_genes(tmp_path: Path):
    compare = _load_script("compare_trna_rrna_annotations.py")
    table = tmp_path / "bench.tsv"
    with table.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample",
                "method",
                "feature_type",
                "gene",
                "start",
                "end",
                "strand",
                "score",
                "status",
                "source_file",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerow(
            {
                "sample": "s",
                "method": "pmga",
                "feature_type": "tRNA",
                "gene": "trnK(ttt)",
                "status": "ok",
            }
        )
        writer.writerow(
            {
                "sample": "s",
                "method": "pmga",
                "feature_type": "tRNA",
                "gene": "trnK(ttt)",
                "status": "ok",
            }
        )
        writer.writerow(
            {
                "sample": "s",
                "method": "tool",
                "feature_type": "rRNA",
                "gene": "",
                "status": "skipped:missing",
            }
        )

    summary = compare.summarize(compare.read_rows(table))

    assert summary[0]["unique_genes"] == "1"
    assert summary[0]["genes"] == "trnK(ttt)"
    assert summary[1]["skipped_rows"] == "1"


def test_compare_script_computes_locus_metrics_and_release_gate():
    compare = _load_script("compare_trna_rrna_annotations.py")
    rows = [
        {
            "sample": "s",
            "method": "pmga_gff",
            "feature_type": "tRNA",
            "gene": "trnA(tgc)",
            "start": "10",
            "end": "80",
            "status": "ok",
        },
        {
            "sample": "s",
            "method": "pmga_gff",
            "feature_type": "tRNA",
            "gene": "trnB(gtt)",
            "start": "100",
            "end": "170",
            "status": "ok",
        },
        {
            "sample": "s",
            "method": "candidate",
            "feature_type": "tRNA",
            "gene": "trnA(tgc)",
            "start": "10",
            "end": "80",
            "status": "ok",
        },
        {
            "sample": "s",
            "method": "candidate",
            "feature_type": "tRNA",
            "gene": "trnB(gtt)",
            "start": "100",
            "end": "170",
            "status": "ok",
        },
        {
            "sample": "s",
            "method": "candidate",
            "feature_type": "tRNA",
            "gene": "trnX(nnn)",
            "start": "300",
            "end": "370",
            "status": "ok",
        },
        {
            "sample": "s",
            "method": "overcall",
            "feature_type": "tRNA",
            "gene": "trnA(tgc)",
            "start": "10",
            "end": "80",
            "status": "ok",
        },
        {
            "sample": "s",
            "method": "overcall",
            "feature_type": "tRNA",
            "gene": "trnB(gtt)",
            "start": "100",
            "end": "170",
            "status": "ok",
        },
        {
            "sample": "s",
            "method": "overcall",
            "feature_type": "tRNA",
            "gene": "trnX(nnn)",
            "start": "300",
            "end": "370",
            "status": "ok",
        },
        {
            "sample": "s",
            "method": "overcall",
            "feature_type": "tRNA",
            "gene": "trnY(nnn)",
            "start": "400",
            "end": "470",
            "status": "ok",
        },
    ]

    metrics = compare.compute_locus_metrics(rows, max_hit_ratio=1.5, min_locus_recall=0.9)
    by_method = {row["method"]: row for row in metrics}

    assert by_method["candidate"]["baseline_loci_supported"] == "2"
    assert by_method["candidate"]["hit_ratio"] == "1.50"
    assert by_method["candidate"]["release_gate"] == "pass"
    assert by_method["overcall"]["hit_ratio"] == "2.00"
    assert by_method["overcall"]["release_gate"] == "fail"


def test_compare_script_locus_metrics_are_sample_scoped():
    compare = _load_script("compare_trna_rrna_annotations.py")
    rows = [
        {
            "sample": "s1",
            "method": "pmga_gff",
            "feature_type": "tRNA",
            "gene": "trnA(tgc)",
            "start": "10",
            "end": "80",
            "status": "ok",
        },
        {
            "sample": "s2",
            "method": "pmga_gff",
            "feature_type": "tRNA",
            "gene": "trnA(tgc)",
            "start": "10",
            "end": "80",
            "status": "ok",
        },
        {
            "sample": "s1",
            "method": "candidate",
            "feature_type": "tRNA",
            "gene": "trnA(tgc)",
            "start": "10",
            "end": "80",
            "status": "ok",
        },
    ]

    metrics = compare.compute_locus_metrics(rows, max_hit_ratio=1.25, min_locus_recall=0.9)

    assert len(metrics) == 1
    assert metrics[0]["sample"] == "s1"
    assert metrics[0]["baseline_loci_supported"] == "1"


def test_compare_cli_uses_gate_threshold_arguments(tmp_path: Path, monkeypatch):
    compare = _load_script("compare_trna_rrna_annotations.py")
    table = tmp_path / "bench.tsv"
    summary = tmp_path / "summary.tsv"
    strict_loci = tmp_path / "strict_loci.tsv"
    relaxed_loci = tmp_path / "relaxed_loci.tsv"
    fields = [
        "sample",
        "method",
        "feature_type",
        "gene",
        "start",
        "end",
        "strand",
        "score",
        "status",
        "source_file",
    ]
    rows = [
        {
            "sample": "s",
            "method": "pmga_gff",
            "feature_type": "tRNA",
            "gene": "trnA(tgc)",
            "start": "10",
            "end": "80",
            "status": "ok",
        },
        {
            "sample": "s",
            "method": "pmga_gff",
            "feature_type": "tRNA",
            "gene": "trnB(gtt)",
            "start": "100",
            "end": "170",
            "status": "ok",
        },
        {
            "sample": "s",
            "method": "native",
            "feature_type": "tRNA",
            "gene": "trnA(tgc)",
            "start": "10",
            "end": "80",
            "status": "ok",
        },
        {
            "sample": "s",
            "method": "native",
            "feature_type": "tRNA",
            "gene": "trnB(gtt)",
            "start": "100",
            "end": "170",
            "status": "ok",
        },
        {
            "sample": "s",
            "method": "native",
            "feature_type": "tRNA",
            "gene": "trnX(nnn)",
            "start": "300",
            "end": "370",
            "status": "ok",
        },
    ]
    with table.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_trna_rrna_annotations.py",
            "--input",
            str(table),
            "--output",
            str(summary),
            "--locus-output",
            str(strict_loci),
            "--max-hit-ratio",
            "1.25",
            "--min-locus-recall",
            "0.90",
        ],
    )
    assert compare.main() == 0
    strict = list(csv.DictReader(strict_loci.open(), delimiter="\t"))
    assert strict[0]["hit_ratio"] == "1.50"
    assert strict[0]["locus_recall"] == "1.00"
    assert strict[0]["release_gate"] == "fail"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_trna_rrna_annotations.py",
            "--input",
            str(table),
            "--output",
            str(summary),
            "--locus-output",
            str(relaxed_loci),
            "--max-hit-ratio",
            "1.50",
            "--min-locus-recall",
            "0.90",
        ],
    )
    assert compare.main() == 0
    relaxed = list(csv.DictReader(relaxed_loci.open(), delimiter="\t"))
    assert relaxed[0]["release_gate"] == "pass"


def test_aragorn_runner_uses_trna_mode_and_parses_batch_output(tmp_path: Path, monkeypatch):
    runner = _load_script("run_mito_trna_rrna_benchmark.py")
    fasta = tmp_path / "seq.fasta"
    fasta.write_text(">s\nACGT\n")
    tool = tmp_path / "aragorn"
    tool.write_text("#!/bin/sh\n")

    def fake_run_command(cmd, output_file, timeout=600, env=None):
        assert "-t" in cmd
        assert "-m" not in cmd
        assert "-gcstd" in cmd
        output_file.write_text(
            "\n".join(
                [
                    ">mini",
                    "2   tRNA-Lys                     [249231,249303]\t34  \t(ttt)",
                    "5   tRNA-Asp                    c[45367,45440]\t35  \t(gtc)",
                ]
            )
        )
        return True

    monkeypatch.setattr(runner, "run_command", fake_run_command)

    rows = runner.run_aragorn(fasta, tmp_path, "mini", tool)

    assert [row["gene"] for row in rows] == ["trnK(ttt)", "trnD(gtc)"]
    assert rows[0]["start"] == "249231"
    assert rows[0]["end"] == "249303"
    assert rows[0]["strand"] == "+"
    assert rows[1]["start"] == "45367"
    assert rows[1]["end"] == "45440"
    assert rows[1]["strand"] == "-"


def test_find_tool_accepts_installed_trnascan_layout(tmp_path: Path, monkeypatch):
    runner = _load_script("run_mito_trna_rrna_benchmark.py")
    monkeypatch.setenv("PATH", "")
    tool = tmp_path / "software" / "tRNAscan-SE" / "install" / "bin" / "tRNAscan-SE"
    tool.parent.mkdir(parents=True)
    tool.write_text("#!/bin/sh\n")
    tool.chmod(0o755)

    assert runner.find_tool(tmp_path / "software", "tRNAscan-SE") == tool


def test_trnascan_parser_reads_standard_2_0_output(tmp_path: Path):
    runner = _load_script("run_mito_trna_rrna_benchmark.py")
    output = tmp_path / "trnascan.out"
    output.write_text(
        "\n".join(
            [
                "Sequence \t\ttRNA \tBounds\ttRNA\tAnti\tIntron Bounds\tInf\t",
                "Name     \ttRNA #\tBegin\tEnd  \tType\tCodon\tBegin\tEnd\tScore\tNote",
                "-------- \t------\t-----\t------\t----\t-----\t-----\t----\t------\t------",
                "mini     \t1\t12619\t12738\tLeu\tCAA\t12657\t12692\t72.9\t",
                "mini     \t2\t26992\t26920\tPhe\tGAA\t0\t0\t70.8\t",
                "mini     \t3\t30000\t30075\tUndet\tNNN\t0\t0\t17.7\t",
            ]
        )
    )

    rows = runner.rows_from_trnascan_output(output, "mini")

    assert [row["gene"] for row in rows] == ["trnL(caa)", "trnF(gaa)", "trn?(nnn)"]
    assert rows[0]["start"] == "12619"
    assert rows[0]["end"] == "12738"
    assert rows[0]["strand"] == "+"
    assert rows[0]["score"] == "72.9"
    assert rows[1]["start"] == "26920"
    assert rows[1]["end"] == "26992"
    assert rows[1]["strand"] == "-"


def test_trnascan_runner_uses_organelle_mode_and_configured_infernal(tmp_path: Path, monkeypatch):
    runner = _load_script("run_mito_trna_rrna_benchmark.py")
    fasta = tmp_path / "seq.fasta"
    fasta.write_text(">s\nACGT\n")
    tool = tmp_path / "software" / "tRNAscan-SE" / "install" / "bin" / "tRNAscan-SE"
    tool.parent.mkdir(parents=True)
    tool.write_text("#!/bin/sh\n")
    tool.chmod(0o755)
    conf = tool.parent / "tRNAscan-SE.conf"
    conf.write_text("bin_dir: /tool/bin\ninfernal_dir: {bin_dir}\n")
    captured = {}

    monkeypatch.setattr(
        runner.shutil, "which", lambda name: "/usr/bin/cmsearch" if name == "cmsearch" else None
    )

    def fake_run(cmd, capture_output=True, text=True, timeout=600, env=None):
        captured["cmd"] = cmd
        output = Path(cmd[cmd.index("-o") + 1])
        output.write_text(
            "\n".join(
                [
                    "Sequence \t\ttRNA \tBounds\ttRNA\tAnti\tIntron Bounds\tInf\t",
                    "Name     \ttRNA #\tBegin\tEnd  \tType\tCodon\tBegin\tEnd\tScore\tNote",
                    "-------- \t------\t-----\t------\t----\t-----\t-----\t----\t------\t------",
                    "mini     \t1\t10\t80\tPhe\tGAA\t0\t0\t70.8\t",
                ]
            )
        )

        class Result:
            returncode = 0
            stderr = ""

        return Result()

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    rows = runner.run_trnascan(fasta, tmp_path, "mini", tool)

    assert rows[0]["gene"] == "trnF(gaa)"
    assert "-O" in captured["cmd"]
    assert "-q" in captured["cmd"]
    assert "-Q" in captured["cmd"]
    assert "--format" not in captured["cmd"]
    conf_path = Path(captured["cmd"][captured["cmd"].index("-c") + 1])
    assert f"infernal_dir: {Path('/usr/bin/cmsearch').parent}" in conf_path.read_text()


def test_barrnap_runner_adds_aragorn_neighbor_to_path(tmp_path: Path, monkeypatch):
    runner = _load_script("run_mito_trna_rrna_benchmark.py")
    software = tmp_path / "software"
    tool = software / "barrnap" / "bin" / "barrnap"
    tool.parent.mkdir(parents=True)
    tool.write_text("#!/bin/sh\n")
    (software / "aragorn").mkdir()
    fasta = tmp_path / "seq.fasta"
    fasta.write_text(">s\nACGT\n")
    seen = {}

    def fake_run_command(cmd, output_file, timeout=600, env=None):
        seen["env"] = env
        output_file.write_text("seq\tbarrnap\trRNA\t10\t120\t42\t+\t.\tName=18S_rRNA\n")
        return True

    monkeypatch.setattr(runner, "run_command", fake_run_command)

    rows = runner.run_barrnap(fasta, tmp_path, "mini", tool)

    path_entries = seen["env"]["PATH"].split(os.pathsep)
    assert str(software / "aragorn") == path_entries[0]
    assert rows[0]["gene"] == "18S_rRNA"


def test_native_trna_rows_from_raw_hits(tmp_path: Path):
    runner = _load_script("run_mito_trna_rrna_benchmark.py")
    hits = [
        SimpleNamespace(
            gene_name="trnF(gaa)",
            start=10,
            end=80,
            strand=1,
            score=99.5,
            source="Native",
        ),
        SimpleNamespace(
            gene_name="trnD(gtc)",
            start=120,
            end=190,
            strand=-1,
            score=88.0,
            source="Native",
        ),
    ]

    rows = runner.rows_from_raw_trna_hits(hits, "mini", "OrganelleVerse-native", tmp_path)

    assert [
        (row["gene"], row["start"], row["end"], row["strand"], row["score"]) for row in rows
    ] == [
        ("trnF(gaa)", "10", "80", "+", "99.5"),
        ("trnD(gtc)", "120", "190", "-", "88.0"),
    ]
    assert rows[0]["source_file"] == str(tmp_path)


def test_cleanroom_trna_runner_labels_experimental_method(tmp_path: Path, monkeypatch):
    from organelleverse.annotation.mitochondrion import trna_native

    runner = _load_script("run_mito_trna_rrna_benchmark.py")
    fasta = tmp_path / "seq.fasta"
    fasta.write_text(">s\nACGT\n")

    def fake_cleanroom(fasta_path, output_dir, db_manager, debug=False):
        assert fasta_path == fasta
        assert debug is True
        assert output_dir.name == "organelleverse_cleanroom"
        assert db_manager is not None
        return [
            SimpleNamespace(
                gene_name="trnF(gaa)",
                start=10,
                end=80,
                strand=1,
                score=101.0,
                source="NativeCleanRoom",
            )
        ]

    monkeypatch.setattr(trna_native, "annotate_trna_cleanroom_experimental", fake_cleanroom)

    rows = runner.run_organelleverse_cleanroom_trna(fasta, tmp_path, "mini")

    assert rows[0]["method"] == "NativeCleanRoom"
    assert rows[0]["gene"] == "trnF(gaa)"


def test_run_command_saves_stderr_on_failure(tmp_path: Path):
    runner = _load_script("run_mito_trna_rrna_benchmark.py")
    output = tmp_path / "tool.out"

    ok = runner.run_command(
        [
            sys.executable,
            "-c",
            "import sys; print('stdout'); print('missing dep', file=sys.stderr); raise SystemExit(2)",
        ],
        output,
    )

    assert ok is False
    assert output.read_text().strip() == "stdout"
    assert output.with_suffix(".out.stderr.txt").read_text().strip() == "missing dep"
