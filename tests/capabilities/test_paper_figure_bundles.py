"""The four paper-figure capabilities are admitted by real fixtures and render via ov.write."""

import shutil
from pathlib import Path

import pytest

import organelleverse as ov
import organelleverse.capabilities as capabilities_package
from organelleverse.capabilities.discovery import discover_capabilities
from organelleverse.capabilities.index import CapabilityStatus
from organelleverse.capabilities.verification import (
    LocalVerificationEnvironment,
    VerificationStore,
    verify_capability,
)
from organelleverse.core.result import OrganelleResult

CASES = {
    "visualization.plot_ir_boundaries": (
        {"genbank_path": ["a.gb", "b.gb"]},
        lambda m: m["genomes"][0]["ir_detected"] and not m["genomes"][1]["ir_detected"],
    ),
    "visualization.plot_nucleotide_diversity": (
        {"alignment_fasta": "aln.fasta", "genbank_path": "ref.gb", "show_ir": False},
        lambda m: m["coordinate"] == "reference" and m["peaks"][0]["label"] == "matK",
    ),
    "visualization.plot_repeat_summary": (
        {"fasta_path": ["g.fasta"], "min_repeat_len": 30},
        lambda m: m["samples"][0]["ssr_total"] >= 2 and m["samples"][0]["repeat_total"] >= 1,
    ),
    "visualization.plot_gene_presence": (
        {"genbank_path": ["A.gb", "B.gb", "C.gb"], "tree_path": "t.nwk"},
        lambda m: m["order"] == "tree" and len(m["species"]) == 3,
    ),
}


@pytest.mark.parametrize("capability", sorted(CASES))
def test_paper_figure_bundle(capability, tmp_path, monkeypatch):
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

    slug = capability.replace(".", "-").replace("_", "-")
    source = Path(capabilities_package.__file__).parent / slug / "fixtures" / "basic" / "input"
    work = tmp_path / "input"
    shutil.copytree(source, work)
    parameters, check = CASES[capability]
    resolved = {
        key: ([str(work / v) for v in value] if isinstance(value, list) else str(work / value))
        if key.endswith(("_path", "_fasta")) else value
        for key, value in parameters.items()
    }
    result = index.binding_source().resolve(capability).invoke(None, resolved)
    assert isinstance(result, OrganelleResult) and result.status == "ok"
    assert check(result.metrics)
    assert not result.artifacts

    shutil.rmtree(work)  # rendering needs only the serialized metrics
    published = ov.write(result, tmp_path / "figure.svg")
    assert {a.kind for a in published.artifacts} == {"figure", "figure_data"}
