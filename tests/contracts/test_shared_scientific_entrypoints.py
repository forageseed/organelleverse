"""Registered scientific operations share one Python and agent callable."""

from __future__ import annotations

import importlib
import tomllib
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "docs" / "operations" / "restored-capabilities.toml"
IDS = (
    "coevolution.run_coevolution",
    "coevolution.run_orthofinder",
    "comparative.compute_genome_identity",
    "localization.predict",
    "morphology.segment",
    "morphology.train",
    "morphology.training_code_snippet",
    "pangenome.build_graph",
    "phylogeny.render_network",
    "phylogeny.run_build_tree",
    "phylogeny.run_trim_alignment",
    "population.call_variants",
    "population.cytonuclear_gwas",
    "population.prepare_gemma_input",
    "selection.fasta_to_axt",
    "selection.run_codeml",
    "selection.run_kaks_calculator_workflow",
    "visualization.plot_genome_identity",
    "visualization.plot_synteny_matrix",
)


@pytest.mark.parametrize("capability_id", IDS)
def test_public_facade_is_the_agent_callable(capability_id: str) -> None:
    records = tomllib.loads(LEDGER.read_text())["capability"]
    record = next(record for record in records if record["id"] == capability_id)
    module_name, function_name = record["python_locator"].split(":")
    implementation = getattr(importlib.import_module(module_name), function_name)
    facade = importlib.import_module(f"organelleverse.{record['domain']}")
    assert getattr(facade, record["public_name"]) is implementation


def test_public_aliases_reuse_the_same_callable() -> None:
    import organelleverse.localization as localization
    import organelleverse.visualization as visualization

    assert localization.localize is localization.predict
    assert visualization.genome_identity is visualization.plot_genome_identity
    assert visualization.synteny_matrix is visualization.plot_synteny_matrix
