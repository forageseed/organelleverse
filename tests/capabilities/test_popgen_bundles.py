"""The five popgen-figure capabilities are admitted by real immutable fixtures."""

import sys
from pathlib import Path

import pytest

import organelleverse as ov
from organelleverse.capabilities.discovery import discover_capabilities
from organelleverse.capabilities.index import CapabilityStatus
from organelleverse.capabilities.verification import (
    LocalVerificationEnvironment,
    VerificationStore,
    verify_capability,
)
from organelleverse.core.result import OrganelleResult

sys.path.insert(0, str(Path(__file__).parents[1] / "visualization"))
import popgen_fixture as toy

CAPABILITIES = {
    "visualization.plot_fst_scan": lambda f: {
        "vcf": f["toy.vcf"], "groups": toy.GROUPS, "pair": ["wild", "crop"], "window_size": 500, "step": 250,
    },
    "visualization.plot_pi_ratio": lambda f: {
        "groups": toy.GROUPS, "vcf": f["toy.vcf"], "reference_fasta": f["ref.fasta"], "genbank": f["ref.gb"],
        "pair": ["wild", "crop"], "window_size": 500, "step": 250, "min_sites": 100,
    },
    "visualization.plot_sv_schematic": lambda f: {
        "gfa": f["toy.gfa"], "reference": "REF", "start": 1, "end": 2000, "samples": ["W4", "C1"],
    },
    "visualization.plot_sv_frequency": lambda f: {
        "groups": toy.GROUPS, "gfa": f["toy.gfa"], "reference": "REF", "min_length": 100,
    },
    "visualization.plot_presence_tracks": lambda f: {
        "gfa": f["toy.gfa"], "reference": "REF", "start": 1, "end": 2000, "groups": toy.GROUPS,
        "genbank": f["ref.gb"],
    },
}


@pytest.mark.parametrize("capability", sorted(CAPABILITIES))
def test_popgen_bundle_is_admitted_and_renders(capability, tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("ORGANELLEVERSE_HOME", str(home))
    monkeypatch.delenv("ORGANELLEVERSE_CAPABILITY_PATH", raising=False)
    monkeypatch.chdir(tmp_path)
    index = discover_capabilities()
    verify_capability(
        capability,
        store=VerificationStore(home / "verifications"),
        environment=LocalVerificationEnvironment(index),
    )
    index = discover_capabilities()
    assert index.describe(capability).status is CapabilityStatus.ADMITTED

    files = toy.write_all(tmp_path / "input")
    parameters = {k: (str(v.resolve()) if isinstance(v, Path) else v) for k, v in CAPABILITIES[capability](files).items()}
    result = index.binding_source().resolve(capability).invoke(None, parameters)
    assert isinstance(result, OrganelleResult) and result.status == "ok"
    assert not result.artifacts

    # the canonical Result renders after its input files are gone
    for path in files.values():
        path.unlink()
    published = ov.write(result, tmp_path / "capability.svg")
    assert {a.kind for a in published.artifacts} == {"figure", "figure_data"}
