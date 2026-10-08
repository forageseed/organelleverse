from __future__ import annotations

from pathlib import Path

import pytest

from organelleverse.annotation.mitochondrion.trna import RawTRNA, _select_trna_engine


def test_trna_engine_defaults_to_native(monkeypatch):
    monkeypatch.delenv("ORG_VERSE_TRNA_ENGINE", raising=False)
    monkeypatch.delenv("ORG_VERSE_FORCE_EXTERNAL_TRNA", raising=False)
    monkeypatch.delenv("ORG_VERSE_ENABLE_CLEANROOM_TRNA", raising=False)
    assert _select_trna_engine() == "native"


def test_trna_engine_external_opt_in(monkeypatch):
    monkeypatch.setenv("ORG_VERSE_TRNA_ENGINE", "external")
    assert _select_trna_engine() == "external"


class MiniDB:
    def __init__(self, trna_ref_dir: Path):
        self.trna_ref_dir = trna_ref_dir


def _reverse_complement(seq: str) -> str:
    return seq.translate(str.maketrans("ACGTacgt", "TGCAtgca"))[::-1]


def _write_fasta(path: Path, name: str, seq: str) -> None:
    path.write_text(f">{name}\n{seq}\n")


def test_native_trna_finds_reference_hits_on_both_strands(tmp_path: Path):
    from organelleverse.annotation.mitochondrion.trna_native import annotate_trna_native

    ref = "G" * 7 + "T" * 5 + "ACGATCGATCGATCGATCGATCGATCG" + "TTC" + "A" * 32
    genome = "N" * 30 + ref + "N" * 45 + _reverse_complement(ref) + "N" * 20
    fasta = tmp_path / "genome.fasta"
    _write_fasta(fasta, "mini", genome)
    ref_dir = tmp_path / "refs"
    ref_dir.mkdir()
    _write_fasta(ref_dir / "trna_refs.fasta", "Ref_tRNA_trnF-GAA_1", ref)

    hits = annotate_trna_native(fasta, tmp_path / "out", MiniDB(ref_dir))

    assert [(hit.gene_name, hit.start, hit.end, hit.strand, hit.source) for hit in hits] == [
        ("trnF(gaa)", 31, 30 + len(ref), 1, "Native"),
        ("trnF(gaa)", 31 + len(ref) + 45, 30 + len(ref) + 45 + len(ref), -1, "Native"),
    ]


def test_native_trna_rejects_high_identity_partial_reference_fragment(tmp_path: Path):
    from organelleverse.annotation.mitochondrion.trna_native import annotate_trna_native

    ref = "G" * 7 + "T" * 5 + "ACGATCGATCGATCGATCGATCGATCG" + "TTC" + "A" * 32
    partial = ref[:50]
    fasta = tmp_path / "genome.fasta"
    _write_fasta(fasta, "mini", "N" * 30 + partial + "N" * 30)
    ref_dir = tmp_path / "refs"
    ref_dir.mkdir()
    _write_fasta(ref_dir / "trna_refs.fasta", "Ref_tRNA_trnF-GAA_1", ref)

    hits = annotate_trna_native(fasta, tmp_path / "out", MiniDB(ref_dir))

    assert hits == []


def test_scan_reference_respects_minimum_seed_support():
    from organelleverse.annotation.mitochondrion.trna_native import TRNAReference, _scan_reference

    sequence = (
        "ACGTACGTACGT" + "TTGCAAGCTTAG" + "GATCGATCGATC" + "CGTACGTACGTA" + "TTC" + "AACCGGTTAACC"
    )
    subject = "N" * 20 + sequence + "N" * 20
    reference = TRNAReference("trnF(gaa)", "GAA", "F", sequence)
    subject_index = {
        sequence[0:12]: [20],
        sequence[2:14]: [22],
    }

    low_support = _scan_reference(
        subject,
        subject_index,
        reference,
        strand=1,
        min_identity=70.0,
        min_coverage=85.0,
        min_seed_hits=3,
    )
    supported = _scan_reference(
        subject,
        subject_index,
        reference,
        strand=1,
        min_identity=70.0,
        min_coverage=85.0,
        min_seed_hits=2,
    )

    assert low_support == []
    assert len(supported) == 1


def test_annotate_trna_uses_native_before_external_tools(tmp_path: Path, monkeypatch):
    from organelleverse.annotation.mitochondrion import trna, trna_native

    fasta = tmp_path / "genome.fasta"
    _write_fasta(fasta, "mini", "ACGT" * 100)

    def fail_external(*args, **kwargs):
        raise AssertionError("external tRNA tools should not run before native tRNA")

    monkeypatch.setattr(trna, "_run_trnascan_se", fail_external)
    monkeypatch.setattr(trna, "_run_aragorn", fail_external)
    monkeypatch.setattr(trna, "_trna_by_blastn", fail_external)
    monkeypatch.delenv("ORG_VERSE_TRNA_ENGINE", raising=False)
    monkeypatch.setattr(
        trna_native,
        "annotate_trna_cm",
        lambda *args, **kwargs: [RawTRNA("trnF(gaa)", 10, 80, 1, "GAA", "F", 95.0, "NativeCM")],
    )

    annotations = trna.annotate_trna(fasta, tmp_path / "out", db_manager=MiniDB(tmp_path))

    assert len(annotations) == 1
    assert annotations[0].gene_name == "trnF(gaa)"
    assert annotations[0].source_tool == "NativeCM"


def test_cleanroom_experimental_path_converts_high_confidence_candidates(
    tmp_path: Path, monkeypatch
):
    from organelleverse.annotation.mitochondrion import trna_native
    from organelleverse.annotation.trna_core.models import (
        TRNACandidate,
        TRNAFeatureSet,
        TRNAFilterDecision,
        TRNAScore,
    )

    fasta = tmp_path / "genome.fasta"
    _write_fasta(fasta, "mini", "A" * 200)
    candidate = TRNACandidate("A" * 73, 10, 82, 1, "gaa", "F")
    features = TRNAFeatureSet(anticodon_offset=30, structure_score=34)
    score = TRNAScore(110, 4, 34, 60, 6, 0, 0, 0)

    monkeypatch.setattr(
        trna_native, "scan_candidate_windows", lambda seq: [candidate], raising=False
    )
    monkeypatch.setattr(trna_native, "score_candidate", lambda item: score, raising=False)
    monkeypatch.setattr(
        trna_native, "best_candidate_features", lambda item: features, raising=False
    )
    monkeypatch.setattr(
        trna_native,
        "apply_overcall_density_filter",
        lambda calls: [(calls[0][0], calls[0][1], TRNAFilterDecision(True, "pass", "high"))],
        raising=False,
    )

    hits = trna_native.annotate_trna_cleanroom_experimental(
        fasta, tmp_path / "out", MiniDB(tmp_path)
    )

    assert len(hits) == 1
    assert hits[0].gene_name == "trnF(gaa)"
    assert hits[0].source == "NativeCleanRoom"


def test_native_engine_is_default_annotate_trna(tmp_path: Path, monkeypatch):
    from organelleverse.annotation.mitochondrion import trna, trna_native

    fasta = tmp_path / "genome.fasta"
    _write_fasta(fasta, "mini", "A" * 200)
    monkeypatch.delenv("ORG_VERSE_TRNA_ENGINE", raising=False)
    monkeypatch.delenv("ORG_VERSE_FORCE_EXTERNAL_TRNA", raising=False)
    monkeypatch.setattr(
        trna_native,
        "annotate_trna_cm",
        lambda *args, **kwargs: [RawTRNA("trnF(gaa)", 10, 82, 1, "GAA", "F", 101.0, "NativeCM")],
    )
    monkeypatch.setattr(
        trna,
        "_run_trnascan_se",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("external tools must not run by default")
        ),
        raising=False,
    )
    monkeypatch.setattr(trna, "_run_aragorn", lambda *args, **kwargs: [], raising=False)
    monkeypatch.setattr(trna, "_trna_by_blastn", lambda *args, **kwargs: [], raising=False)

    annotations = trna.annotate_trna(fasta, tmp_path / "out", db_manager=MiniDB(tmp_path))
    assert len(annotations) == 1
    assert annotations[0].source_tool == "NativeCM"


def test_native_engine_does_not_fall_back_to_external_programs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from organelleverse.annotation.mitochondrion import trna, trna_native

    fasta = tmp_path / "genome.fasta"
    _write_fasta(fasta, "mini", "A" * 200)
    monkeypatch.setattr(trna_native, "annotate_trna_cm", lambda *args, **kwargs: [])
    experimental_calls: list[bool] = []

    def experimental_fallback(*args, **kwargs):
        experimental_calls.append(True)
        return []

    monkeypatch.setattr(
        trna_native,
        "annotate_trna_cleanroom_experimental",
        experimental_fallback,
    )

    def fail_external(*args, **kwargs):
        raise AssertionError("native tRNA mode must not invoke external fallback programs")

    monkeypatch.setattr(trna, "_run_trnascan_se", fail_external)
    monkeypatch.setattr(trna, "_run_aragorn", fail_external)
    monkeypatch.setattr(trna, "_trna_by_blastn", fail_external)

    annotations = trna.annotate_trna(
        fasta,
        tmp_path / "out",
        db_manager=MiniDB(tmp_path),
        engine="native",
    )

    assert annotations == []
    assert experimental_calls == []


def test_native_engine_propagates_cm_failures_without_algorithm_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from organelleverse.annotation.mitochondrion import trna, trna_native

    fasta = tmp_path / "genome.fasta"
    _write_fasta(fasta, "mini", "A" * 200)

    def fail_cm(*args, **kwargs):
        raise RuntimeError("native CM failed")

    def fail_experimental(*args, **kwargs):
        raise AssertionError("native mode must not switch algorithms")

    monkeypatch.setattr(trna_native, "annotate_trna_cm", fail_cm)
    monkeypatch.setattr(
        trna_native,
        "annotate_trna_cleanroom_experimental",
        fail_experimental,
    )

    with pytest.raises(RuntimeError, match="native CM failed"):
        trna.annotate_trna(
            fasta,
            tmp_path / "out",
            db_manager=MiniDB(tmp_path),
            engine="native",
        )


def test_annotate_trna_external_opt_in_runs_external_tools(tmp_path: Path, monkeypatch):
    from organelleverse.annotation.mitochondrion import trna, trna_native

    fasta = tmp_path / "genome.fasta"
    _write_fasta(fasta, "mini", "A" * 200)

    monkeypatch.setenv("ORG_VERSE_TRNA_ENGINE", "external")
    monkeypatch.setattr(
        trna_native,
        "annotate_trna_cleanroom_experimental",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("native engine must not run when external is selected")
        ),
    )
    monkeypatch.setattr(
        trna,
        "_run_trnascan_se",
        lambda *args, **kwargs: [RawTRNA("trnF(gaa)", 10, 82, 1, "GAA", "F", 88.0, "tRNAscan-SE")],
    )
    monkeypatch.setattr(trna, "_run_aragorn", lambda *args, **kwargs: [], raising=False)

    annotations = trna.annotate_trna(fasta, tmp_path / "out", db_manager=MiniDB(tmp_path))

    assert len(annotations) == 1
    assert annotations[0].gene_name == "trnF(gaa)"
    assert annotations[0].source_tool == "tRNAscan-SE"
