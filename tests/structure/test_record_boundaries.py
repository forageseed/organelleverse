"""FASTA records are independent molecules, with local repeat coordinates."""

import random

import pytest

from organelleverse.core.errors import OrganelleInputError
from organelleverse.structure.structure import multiconf, resolve_configs, write_multiconf
from organelleverse.structure.structure_core import compute_multiconf


def test_record_boundary_cannot_create_repeat(tmp_path):
    rng = random.Random(771)

    def dna(n):
        return "".join(rng.choice("ACGT") for _ in range(n))

    repeat = dna(24)
    records = [dna(80) + repeat[:12], repeat[12:] + dna(80), dna(80) + repeat + dna(80)]
    p = tmp_path / "boundary.fa"
    p.write_text("".join(f">r{i}\n{s}\n" for i, s in enumerate(records)))
    core = compute_multiconf(p, 24)
    result = multiconf(p, min_repeat_len=24)
    resolved = resolve_configs(p, min_repeat_len=24)
    assert core["repeats"] == []
    assert result.metrics["direct_repeat_pairs"] == 0
    assert result.metrics["predicted_configs"] == 0
    assert resolved.metrics["configs"] == ()


def test_repeats_retain_sequence_identity_and_local_coordinates(tmp_path):
    block = "ACGTTGCACTAGGATCCGTACGATCGTTACGGCATC"
    p = tmp_path / "records.fa"
    p.write_text(f">short\nACGT\n>repeat\n{block}{'N' * 20}{block}\n")
    result = multiconf(p, min_repeat_len=len(block))
    pair = result.metrics["repeats"][0]
    assert pair["sequence_id"] == "repeat"
    assert tuple(pair["positions"]) == (1, len(block) + 21)
    assert "repeat_pair_candidates_only" in result.flags
    assert compute_multiconf(p, len(block))["repeats"][0]["sequence_id"] == "repeat"
    cfg = resolve_configs(p, min_repeat_len=len(block)).metrics["configs"][0]
    assert cfg["sequence_id"] == "repeat"
    out = write_multiconf(result, tmp_path / "pairs.tsv")
    assert out.read_text().splitlines()[0].endswith("\tsequence_id")
    assert out.read_text().splitlines()[1].endswith("\trepeat")


def test_requested_missing_graph_does_not_change_inference_method(tmp_path):
    p = tmp_path / "seq.fa"
    p.write_text(">one\nACGT\n")
    with pytest.raises(OrganelleInputError, match="Requested GFA"):
        resolve_configs(p, gfa_path=tmp_path / "missing.gfa")
