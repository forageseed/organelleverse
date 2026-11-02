from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

from tests._paths import PROJECT_ROOT

ROOT = PROJECT_ROOT
SCRIPTS = ROOT / "scripts" / "annotation_benchmarks"
FIXTURES = ROOT / "tests" / "fixtures" / "trna_core"
REQUIRED_FIELDS = [
    "sample",
    "gene",
    "start",
    "end",
    "strand",
    "anticodon",
    "source",
    "sequence",
    "label",
]


def _load_script(name: str):
    path = SCRIPTS / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def test_committed_trna_core_fixtures_have_required_schema():
    for name, label in [("positives.tsv", "positive"), ("hard_negatives.tsv", "hard_negative")]:
        rows = _read_tsv(FIXTURES / name)
        assert rows, f"{name} must contain at least one row"
        assert list(rows[0]) == REQUIRED_FIELDS
        assert {row["label"] for row in rows} == {label}
        assert all(row["sequence"] for row in rows)
        assert all(row["source"] for row in rows)


def test_fixture_builder_has_no_local_default_paths():
    script = SCRIPTS / "build_trna_core_fixtures.py"
    text = script.read_text()
    forbidden = ["/" + "home/", "data" + "16t", "ji" + "azc"]
    assert [pattern for pattern in forbidden if pattern in text] == []


def test_fixture_builder_creates_positive_and_hard_negative_sets(tmp_path: Path):
    builder = _load_script("build_trna_core_fixtures.py")
    fasta = tmp_path / "mini.fasta"
    pmga = tmp_path / "mini.gff"
    benchmark = tmp_path / "benchmark.tsv"
    positives = tmp_path / "positives.tsv"
    negatives = tmp_path / "hard_negatives.tsv"
    sequence = "A" * 9 + "GACTGACTGACTGACTGACTGACTGACTG" + "C" * 49 + "TTTTTTTTTTTTTTTTTTTTTTTTT"
    fasta.write_text(f">mini\n{sequence}\n")
    pmga.write_text(
        "\n".join(
            [
                "##gff-version 3",
                "mini\tPMGA\ttRNA\t10\t40\t.\t+\t.\tID=t1;Name=trnF-GAA",
            ]
        )
    )
    with benchmark.open("w", newline="") as handle:
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
                "sample": "mini",
                "method": "OrganelleVerse-native",
                "feature_type": "tRNA",
                "gene": "trnX(nnn)",
                "start": "80",
                "end": "104",
                "strand": "+",
                "score": "10",
                "status": "ok",
                "source_file": "native",
            }
        )

    assert (
        builder.main(
            [
                "--fasta",
                str(fasta),
                "--pmga-gff",
                str(pmga),
                "--benchmark",
                str(benchmark),
                "--positive-output",
                str(positives),
                "--negative-output",
                str(negatives),
                "--sample",
                "mini",
            ]
        )
        == 0
    )

    positive_rows = _read_tsv(positives)
    negative_rows = _read_tsv(negatives)
    assert positive_rows[0]["source"] == "PMGA"
    assert positive_rows[0]["label"] == "positive"
    assert positive_rows[0]["anticodon"] == "gaa"
    assert negative_rows[0]["source"] == "OrganelleVerse-native"
    assert negative_rows[0]["label"] == "hard_negative"
