from __future__ import annotations

from pathlib import PurePosixPath

import pytest

from organelleverse.compute.discovery import (
    compute_provider_record_digest,
    discover_compute_providers,
)
from organelleverse.core.errors import OrganelleContractError


class _FakeDist:
    def __init__(self, name, version, files, record=None):
        self.name = name
        self.version = version
        self.files = [PurePosixPath(f) for f in files]
        self._record = record

    def read_text(self, filename):
        return self._record if filename == "RECORD" else None


class _FakeEntryPoint:
    def __init__(self, name, value, dist):
        self.name = name
        self.value = value
        self.dist = dist

    def load(self):  # pragma: no cover - discovery must never call this
        raise AssertionError("discovery must never call EntryPoint.load()")


def test_discover_returns_metadata_only_candidates_in_deterministic_order():
    eps = [
        _FakeEntryPoint("slurm", "acme_slurm:factory", _FakeDist("acme-slurm", "2.0.0", ["acme_slurm/__init__.py"])),
        _FakeEntryPoint("wsl", "acme_wsl:factory", _FakeDist("acme-wsl", "1.0.0", ["acme_wsl/__init__.py"])),
        _FakeEntryPoint("ssh", "acme_ssh:factory", _FakeDist("acme-ssh", "1.1.0", ["acme_ssh/__init__.py"])),
    ]
    candidates = discover_compute_providers(entry_points=eps)
    assert [c.provider_id for c in candidates] == ["slurm", "ssh", "wsl"]
    assert candidates[0].distribution_name == "acme-slurm"
    assert candidates[0].distribution_version == "2.0.0"
    assert candidates[0].entry_point_locator == "acme_slurm:factory"
    assert candidates[0].distribution_record_digest.startswith("sha256:")


def test_discover_never_loads_the_entry_point():
    # load() raises AssertionError; if discovery called it, this test fails.
    eps = [_FakeEntryPoint("wsl", "acme_wsl:factory", _FakeDist("acme-wsl", "1.0.0", ["a.py"]))]
    discover_compute_providers(entry_points=eps)  # must not raise


def test_record_digest_is_deterministic_and_path_order_independent():
    d1 = _FakeDist("p", "1.0.0", ["b.py", "a.py"])
    d2 = _FakeDist("p", "1.0.0", ["a.py", "b.py"])
    assert compute_provider_record_digest(d1) == compute_provider_record_digest(d2)


def test_record_digest_changes_when_installed_files_change():
    d1 = _FakeDist("p", "1.0.0", ["a.py"])
    d2 = _FakeDist("p", "1.0.0", ["a.py", "b.py"])
    assert compute_provider_record_digest(d1) != compute_provider_record_digest(d2)


def test_record_digest_binds_recorded_file_hashes_not_just_paths():
    # Regression for the review finding: a paths-only digest cannot detect a
    # reinstall whose bytes changed under the same file list.  Binding the
    # full RECORD entry (path + recorded archive hash + size) can.
    record_a = "a.py,sha256=AAAA,10\nb.py,sha256=BBBB,20\n"
    record_b = "a.py,sha256=CCCC,10\nb.py,sha256=BBBB,20\n"
    d1 = _FakeDist("p", "1.0.0", ["a.py", "b.py"], record=record_a)
    d2 = _FakeDist("p", "1.0.0", ["a.py", "b.py"], record=record_b)
    assert compute_provider_record_digest(d1) != compute_provider_record_digest(d2)


def test_record_digest_is_record_line_order_independent():
    record_1 = "b.py,sha256=BBBB,20\na.py,sha256=AAAA,10\n"
    record_2 = "a.py,sha256=AAAA,10\nb.py,sha256=BBBB,20\n"
    d1 = _FakeDist("p", "1.0.0", ["b.py", "a.py"], record=record_1)
    d2 = _FakeDist("p", "1.0.0", ["a.py", "b.py"], record=record_2)
    assert compute_provider_record_digest(d1) == compute_provider_record_digest(d2)


def test_entry_point_without_distribution_is_rejected():
    class _OrphanEP:
        name = "wsl"
        value = "x:factory"
        dist = None

    with pytest.raises(OrganelleContractError) as exc:
        discover_compute_providers(entry_points=[_OrphanEP()])
    assert exc.value.code == "compute.provider_entry_point_orphan"


def test_empty_sequence_selects_nothing():
    assert discover_compute_providers(entry_points=[]) == ()
