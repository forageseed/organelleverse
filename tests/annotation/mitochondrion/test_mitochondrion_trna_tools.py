from __future__ import annotations

from pathlib import Path


def test_parse_trnascan_output_accepts_standard_2_0_table(tmp_path: Path):
    from organelleverse.annotation.mitochondrion.trna import _parse_trnascan_output

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

    hits = _parse_trnascan_output(output)

    assert [(hit.gene_name, hit.start, hit.end, hit.strand, hit.score) for hit in hits] == [
        ("trnL(caa)", 12619, 12738, 1, 72.9),
        ("trnF(gaa)", 26920, 26992, -1, 70.8),
        ("trn?(nnn)", 30000, 30075, 1, 17.7),
    ]
    assert hits[0].intron_start == 12657
    assert hits[0].intron_end == 12692
    assert hits[1].intron_start is None


def test_run_trnascan_se_uses_organelle_mode_and_patched_infernal(tmp_path: Path, monkeypatch):
    from organelleverse.annotation.mitochondrion import trna

    fasta = tmp_path / "seq.fasta"
    fasta.write_text(">s\nACGT\n")
    tool = tmp_path / "tools" / "tRNAscan-SE"
    tool.parent.mkdir()
    tool.write_text("#!/bin/sh\n")
    tool.chmod(0o755)
    conf = tool.parent / "tRNAscan-SE.conf"
    conf.write_text("bin_dir: /tool/bin\ninfernal_dir: {bin_dir}\n")
    captured = {}

    def fake_which(name: str):
        if name == "tRNAscan-SE":
            return str(tool)
        if name == "cmsearch":
            return "/usr/bin/cmsearch"
        return None

    def fake_run(cmd, capture_output=True, text=True, timeout=300):
        captured["cmd"] = cmd
        captured["timeout"] = timeout
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

    monkeypatch.setattr(trna.shutil, "which", fake_which)
    monkeypatch.setattr(trna.subprocess, "run", fake_run)

    hits = trna._run_trnascan_se(fasta, tmp_path / "out", threads=2)

    assert hits[0].gene_name == "trnF(gaa)"
    assert "--thread" in captured["cmd"]
    assert "2" in captured["cmd"]
    assert "-O" in captured["cmd"]
    assert "-q" in captured["cmd"]
    assert "-Q" in captured["cmd"]
    assert "--format" not in captured["cmd"]
    assert captured["timeout"] == 900
    conf_path = Path(captured["cmd"][captured["cmd"].index("-c") + 1])
    assert "infernal_dir: /usr/bin" in conf_path.read_text()
