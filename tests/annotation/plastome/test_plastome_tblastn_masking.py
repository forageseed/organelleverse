"""tblastn must not SEG-mask small hydrophobic plastid proteins.

With SEG on, psbJ (40 aa, mostly one transmembrane helix) keeps an 11-residue
alignment (query coverage 0.28), falls below the 0.5 transfer threshold, and
was missed in every benchmark plastome. The search command is also part of the
result cache key, so an output directory searched with SEG on is re-searched.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from Bio.SeqFeature import FeatureLocation, SeqFeature

from organelleverse.annotation.plastome import blast
from organelleverse.annotation.plastome.models import (
    BlastTools,
    ReferenceFeature,
    ReferenceQuery,
)

PSBJ = "MADTTGRIPLWIIGTVTGIPVIGLIGIFFYGSYSGLGSSL"


def _protein_query() -> ReferenceQuery:
    feature = SeqFeature(FeatureLocation(0, len(PSBJ) * 3, strand=-1), type="CDS")
    ref = ReferenceFeature(
        feature_id="Ref:1",
        feature=feature,
        sequence=PSBJ,
        gene="psbJ",
        feature_type="CDS",
        reference_name="Ref",
    )
    return ReferenceQuery(query_id="q1", group="reference3", sequence=PSBJ, reference_feature=ref)


def test_losat_tblastn_disables_seg(tmp_path: Path) -> None:
    kwargs = dict(
        query_path=tmp_path / "q", target_fasta=tmp_path / "t", out_path=tmp_path / "o", threads=1
    )

    protein = blast._losat_group_command("losat", protein=True, **kwargs)
    nucleotide = blast._losat_group_command("losat", protein=False, **kwargs)

    assert "--seg=false" in protein
    assert "--seg=false" not in nucleotide


@pytest.fixture()
def recorded(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    calls: list[list[str]] = []

    def fake_run(cmd, timeout):
        calls.append([str(part) for part in cmd])
        out = cmd[cmd.index("-out") + 1] if "-out" in cmd else cmd[cmd.index("-o") + 1]
        Path(out).write_text("")

    monkeypatch.setattr(blast, "run_command", fake_run)
    return calls


def _search(tmp_path: Path, tools: BlastTools) -> None:
    blast.run_plastome_blasts(
        "ACGT" * 50,
        (_protein_query(),),
        work_dir=tmp_path,
        tools=tools,
        min_identity=0.4,
        qcoverage_range=(0.5, 2.0),
    )


def test_ncbi_tblastn_disables_seg(tmp_path: Path, recorded: list[list[str]]) -> None:
    _search(tmp_path, BlastTools(blastn="blastn", makeblastdb="makeblastdb", tblastn="tblastn"))

    tblastn = [cmd for cmd in recorded if cmd[0] == "tblastn"]
    assert len(tblastn) == 1
    assert tblastn[0][tblastn[0].index("-seg") + 1] == "no"


def test_changed_search_command_invalidates_cached_hits(
    tmp_path: Path, recorded: list[list[str]]
) -> None:
    tools = BlastTools(blastn=None, makeblastdb=None, tblastn=None, losat="losat")
    _search(tmp_path, tools)
    _search(tmp_path, tools)
    assert len(recorded) == 1  # identical inputs and command: cached

    # An output directory left by the old SEG-on command must be re-searched.
    cmd_file = tmp_path / "reference3.cmd"
    cmd_file.write_text(cmd_file.read_text().replace(" --seg=false", ""))
    _search(tmp_path, tools)
    assert len(recorded) == 2
