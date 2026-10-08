"""Managed CodeML execution keeps its inputs together without user writes."""

from __future__ import annotations

from pathlib import Path

from organelleverse.selection import run_codeml


def test_codeml_copies_relative_inputs_to_the_managed_run(tmp_path, monkeypatch) -> None:
    import organelleverse.selection.models as models

    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    source = tmp_path / "source"
    source.mkdir()
    (source / "aligned.phy").write_text("2 3\na ACG\nb ACG\n")
    (source / "tree.nwk").write_text("(a,b);\n")
    control = source / "run.ctl"
    control.write_text("seqfile = aligned.phy\ntreefile = tree.nwk\noutfile = run.out\n")

    def fake_codeml(local_ctl: Path, work_dir: Path, *, timeout: float | None) -> dict:
        assert local_ctl.parent == work_dir
        assert (work_dir / "aligned.phy").read_text() == (source / "aligned.phy").read_text()
        assert (work_dir / "tree.nwk").read_text() == (source / "tree.nwk").read_text()
        assert work_dir != source
        assert timeout == 12.0
        return {"lnL": -3.0}

    monkeypatch.setattr(models, "run_codeml", fake_codeml)
    assert run_codeml(control, timeout=12.0) == {"lnL": -3.0}
    assert not (source / "run.out").exists()
