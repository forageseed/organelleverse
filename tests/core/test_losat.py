from types import SimpleNamespace

import pytest

from organelleverse import _losat
from organelleverse.core.errors import OrganelleDependencyError


def test_coverage_merges_unsorted_overlapping_hsps():
    rows = [
        ["ref", "genome", "100", "20", "0", "0", str(a), str(b), "1", "60", "0", "50"]
        for a, b in [(61, 80), (1, 20), (15, 40)]
    ]
    assert _losat.union_query_coverage(rows, {"ref": 100}) == {"ref": 60.0}


def test_tblastn_uses_standard_genetic_code(tmp_path):
    calls = []

    class Runner:
        def run(self, argv, **kwargs):
            calls.append(argv)
            return SimpleNamespace(stdout="")

    _losat.run_losat(
        "tblastn", tmp_path / "q", tmp_path / "s", executable="LOSAT", command_runner=Runner()
    )
    assert calls[0][calls[0].index("--db-gencode") + 1] == "1"
    assert "--outfmt" not in calls[0]


def test_tblastn_seg_flag_is_forwarded_only_when_set(tmp_path):
    calls = []

    class Runner:
        def run(self, argv, **kwargs):
            calls.append(argv)
            return SimpleNamespace(stdout="")

    for seg in (None, False):
        _losat.run_losat(
            "tblastn",
            tmp_path / "q",
            tmp_path / "s",
            executable="LOSAT",
            seg=seg,
            command_runner=Runner(),
        )
    assert not any(arg.startswith("--seg") for arg in calls[0])
    assert "--seg=false" in calls[1]


@pytest.mark.parametrize(
    "option",
    ["task", "word_size", "reward", "penalty", "gap_open", "gap_extend", "percent_identity"],
)
def test_tblastn_rejects_unsupported_options(option):
    with pytest.raises(ValueError, match="does not support"):
        _losat.run_losat("tblastn", "q", "s", executable="LOSAT", **{option: 1})


def test_ncbi_environment_disables_losat(monkeypatch):
    monkeypatch.setenv("ORG_VERSE_LOSAT_BIN", "ncbi")
    assert _losat.resolve_losat() is None
    with pytest.raises(OrganelleDependencyError, match="Install LOSAT"):
        _losat.run_losat("blastn", "q", "s")


def test_negative_penalty_is_a_single_argument():
    calls = []

    class Runner:
        def run(self, argv, **kwargs):
            calls.append(argv)
            return SimpleNamespace(stdout="")

    _losat.run_losat(
        "blastn", "q", "s", task="blastn", penalty=-3, executable="LOSAT", command_runner=Runner()
    )
    assert "--penalty=-3" in calls[0]
    assert "-3" not in calls[0]


def test_query_coverage_uses_each_record_length_with_duplicate_ids(tmp_path):
    from Bio import SeqIO

    query = tmp_path / "query.fasta"
    query.write_text(
        ">duplicate\n" + "M" * 100 + "\n>other\n" + "M" * 33 + "\n>duplicate\n" + "M" * 200 + "\n"
    )
    calls = []

    class Runner:
        def run(self, argv, **kwargs):
            records = list(SeqIO.parse(argv[argv.index("-q") + 1], "fasta"))
            calls.append([r.id for r in records])
            assert len({r.id for r in records}) == len(records)
            rows = []
            for record in records:
                end = 60 if record.id == "duplicate" else 20
                rows.append(
                    [
                        record.id,
                        "genome",
                        "100",
                        str(end),
                        "0",
                        "0",
                        "1",
                        str(end),
                        "1",
                        str(end * 3),
                        "1e-30",
                        "100",
                    ]
                )
            return SimpleNamespace(stdout="\n".join("\t".join(r) for r in rows))

    rows = _losat.run_losat_with_query_coverage(
        "tblastn",
        query,
        tmp_path / "genome.fasta",
        executable="LOSAT",
        command_runner=Runner(),
    )
    assert calls == [["duplicate", "other"], ["duplicate"]]
    assert [(r[0], r[12]) for r in rows] == [
        ("duplicate", "60"),
        ("other", "61"),
        ("duplicate", "30"),
    ]
