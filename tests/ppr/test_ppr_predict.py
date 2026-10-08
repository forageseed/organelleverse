from __future__ import annotations

from pathlib import Path

from organelleverse.ppr.predict import SUPPORTED_PPR_CODES, predict_binding_sites


def _repeat(code: str) -> str:
    repeat = list("A" * 35)
    repeat[4] = code[0]
    repeat[34] = code[1]
    return "".join(repeat)


def _write_inputs(
    directory: Path,
    codes: list[str],
    sequence: str,
) -> tuple[Path, Path]:
    repeat_file = directory / "repeats.tsv"
    rows = ["protein_id\trepeat_index\trepeat_sequence"]
    for index, code in enumerate(codes, start=1):
        rows.append(f"ppr1\t{index}\t{_repeat(code)}")
    repeat_file.write_text("\n".join(rows) + "\n", encoding="utf-8")
    transcript_file = directory / "transcripts.fa"
    transcript_file.write_text(f">tx1\n{sequence}\n", encoding="utf-8")
    return repeat_file, transcript_file


def test_supported_default_codes_are_the_experimentally_supported_core() -> None:
    assert SUPPORTED_PPR_CODES == {"TN": "A", "TD": "G", "NS": "C", "ND": "U"}


def test_exact_match_maps_repeat_order_to_transcript_5prime_to_3prime(tmp_path: Path) -> None:
    repeat_file, transcript_file = _write_inputs(tmp_path, ["TN", "ND", "TD", "NS"], "TTAUGCTTA")
    result = predict_binding_sites(
        repeat_file,
        transcript_file,
        "plastid",
    )

    assert result["status"] == "ok"
    assert result["metrics"]["candidate_sites"] == 1
    site = result["metrics"]["sites"][0]
    assert site == {
        "protein_id": "ppr1",
        "transcript_id": "tx1",
        "start_1based": 3,
        "end_1based": 6,
        "target_sequence": "AUGC",
        "repeat_count": 4,
    }


def test_overlapping_exact_matches_are_all_reported(tmp_path: Path) -> None:
    repeat_file, transcript_file = _write_inputs(tmp_path, ["TN", "ND"], "AUAUA")
    result = predict_binding_sites(repeat_file, transcript_file, "mitochondrion")

    assert result["status"] == "ok"
    assert [site["start_1based"] for site in result["metrics"]["sites"]] == [1, 3]


def test_unknown_code_requires_explicit_mapping_extension(tmp_path: Path) -> None:
    repeat_file, transcript_file = _write_inputs(tmp_path, ["KR"], "G")
    failed = predict_binding_sites(repeat_file, transcript_file, "plastid")
    assert failed["status"] == "failed"
    assert "KR" in failed["errors"][0]["message"]

    accepted = predict_binding_sites(
        repeat_file,
        transcript_file,
        "plastid",
        '{"kr":"G"}',
    )
    assert accepted["status"] == "ok"
    assert accepted["metrics"]["sites"][0]["target_sequence"] == "G"


def test_default_code_mapping_cannot_be_overridden(tmp_path: Path) -> None:
    repeat_file, transcript_file = _write_inputs(tmp_path, ["TN"], "G")
    result = predict_binding_sites(
        repeat_file,
        transcript_file,
        "plastid",
        '{"TN":"G"}',
    )
    assert result["status"] == "failed"
    assert "cannot override" in result["errors"][0]["message"]


def test_repeats_must_be_exactly_35_aa_and_contiguous(tmp_path: Path) -> None:
    repeat_file, transcript_file = _write_inputs(tmp_path, ["TN"], "A")
    text = repeat_file.read_text(encoding="utf-8").replace(_repeat("TN"), "A" * 34)
    repeat_file.write_text(text, encoding="utf-8")
    result = predict_binding_sites(repeat_file, transcript_file, "plastid")
    assert result["status"] == "failed"
    assert "exactly 35 aa" in result["errors"][0]["message"]


def test_repeat_order_is_explicitly_by_one_based_index(tmp_path: Path) -> None:
    repeat_file = tmp_path / "repeats.tsv"
    repeat_file.write_text(
        "protein_id\trepeat_index\trepeat_sequence\n"
        f"ppr1\t2\t{_repeat('ND')}\n"
        f"ppr1\t1\t{_repeat('TN')}\n",
        encoding="utf-8",
    )
    transcript_file = tmp_path / "transcripts.fa"
    transcript_file.write_text(">tx\nAU\n", encoding="utf-8")
    result = predict_binding_sites(repeat_file, transcript_file, "plastid")
    assert result["metrics"]["protein_targets"][0]["target_sequence"] == "AU"
    assert result["metrics"]["candidate_sites"] == 1
