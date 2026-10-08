"""References whose junction their own domain VI contradicts are set aside as splice evidence."""

from __future__ import annotations

from organelleverse.annotation.plastome import group_ii


def _offsets(monkeypatch, table):
    group_ii._offsets.cache_clear()
    monkeypatch.setattr(group_ii, "_offsets", lambda: table)


def test_deviant_reference_at_reliable_intron_is_suspect(monkeypatch):
    table = {f"ref{i}.gb": {"petB": {"1": 0}} for i in range(8)}
    table["zea.gb"] = {"petB": {"1": 28}, "atpF": {"1": 0}}
    table["near.gb"] = {"petB": {"1": 2}}  # within tolerance: an equivalent-slide or alignment wobble
    _offsets(monkeypatch, table)
    files = set(table)
    assert group_ii.suspect_references(files) == {("zea.gb", "petB")}
    # a reference not loaded is neither judged nor counted
    assert group_ii.suspect_references(files - {"zea.gb"}) == set()


def test_unreliable_intron_flags_nothing(monkeypatch):
    table = {f"ref{i}.gb": {"ycf3": {"1": 0 if i < 4 else -2}} for i in range(8)}
    table["odd.gb"] = {"ycf3": {"1": 40}}
    _offsets(monkeypatch, table)
    assert group_ii.suspect_references(set(table)) == set()
