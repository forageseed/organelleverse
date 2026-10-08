"""The complete lazy Python surface: every one of the 253 ledger locators reachable.

Task 1 froze the 253-item restoration ledger; this file is Task 2's own
verification that every one of those locators is actually importable and
callable, that the 15 historically-exported-but-currently-missing package
attributes (across selection, phylogeny, morphology, localization, pangenome,
and population) are restored, and that none of this touched the
already-correct root-level mechanisms: the 22 canonical domains, the 20
short-name lazy aliases, the Registry (still exactly 20 operations), and
``CONTRACTED_SUITES`` (still empty). See ``tests/contracts/test_legacy_removed.py``
for the pre-existing, unmodified assertions this file deliberately does not
duplicate (short-name identity for every attribute, the Registry catalog
guard, ``CONTRACTED_SUITES``).
"""

from __future__ import annotations

import importlib
import inspect
import subprocess
import sys
import tomllib
from typing import Any

import pytest

import organelleverse as ov
from tests._paths import PROJECT_ROOT, child_env

ROOT = PROJECT_ROOT
INVENTORY = ROOT / "docs" / "operations" / "restored-capabilities.toml"

CANONICAL_DOMAINS = {
    "barcode",
    "codon_composition",
    "coevolution",
    "comparative",
    "composition",
    "diversity",
    "format_conversion",
    "hgt",
    "ir_boundary",
    "localization",
    "morphology",
    "pangenome",
    "phenotype",
    "phylogeny",
    "population",
    "rna_editing",
    "selection",
    "structure",
    "trans_splicing",
    "transfer",
    "variation",
    "visualization",
}

ALIASES = {
    "anno": "annotation",
    "asm": "assembly",
    "cmp": "comparative",
    "codon": "codon_composition",
    "div": "diversity",
    "editing": "rna_editing",
    "erc": "coevolution",
    "fmt": "format_conversion",
    "gc": "composition",
    "graph": "structure",
    "ir": "ir_boundary",
    "loc": "localization",
    "morph": "morphology",
    "pan": "pangenome",
    "pheno": "phenotype",
    "phylo": "phylogeny",
    "pop": "population",
    "splicing": "trans_splicing",
    "var": "variation",
    "viz": "visualization",
}

# The 15 historically-exported package-level names this task restores, plus
# `localize` (the archive's `localize = predict` alias, restored alongside
# `predict` since both are the *same* object bound under two package names).
RESTORED_PACKAGE_EXPORTS = {
    "selection": ("fasta_to_axt", "run_codeml", "run_kaks_calculator_workflow"),
    "phylogeny": ("run_trim_alignment", "run_build_tree", "render_network"),
    "morphology": ("segment", "train", "training_code_snippet"),
    "localization": ("predict", "localize"),
    "pangenome": ("build_graph",),
    "population": ("call_variants", "cytonuclear_gwas", "prepare_gemma_input"),
}


def _load_ledger_records() -> list[dict[str, Any]]:
    payload = tomllib.loads(INVENTORY.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = payload["capability"]
    return records


# --- root domains and aliases stay lazy and identity-routed -----------------


def test_root_exposes_all_domains_and_aliases_lazily() -> None:
    script = """
import sys
import organelleverse as ov
assert 'organelleverse.visualization' not in sys.modules
assert 'matplotlib.pyplot' not in sys.modules
assert ov.viz.plot_genome_map is ov.visualization.plot_genome_map
assert 'organelleverse.visualization' in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True, env=child_env())


def test_canonical_domains_and_aliases_match_the_established_mechanism() -> None:
    public_modules: frozenset[str] = getattr(ov, "_PUBLIC_MODULES", frozenset())
    non_domain_facades = {
        "annotation",
        # Infrastructure, not a scientific domain: the bundle/discovery envelope.
        # Reachable as an attribute so an Agent can drive the authoring loop from
        # one import, and lazy so the bare import stays cold — both pinned by
        # tests/test_capabilities_reachability.py.
        "capabilities",
        "assembly",
        "assets",
        "fetch",
        "operations",
        "environments",
        "io",
        "qc",
        "report",
    }
    assert set(public_modules) - non_domain_facades == CANONICAL_DOMAINS
    assert len(CANONICAL_DOMAINS) == 22

    short_aliases: dict[str, str] = getattr(ov, "_SHORT_MODULE_ALIASES", {})
    assert dict(short_aliases) == ALIASES


@pytest.mark.parametrize(("short", "long"), sorted(ALIASES.items()))
def test_short_and_long_domain_are_the_same_module_object(short: str, long: str) -> None:
    assert getattr(ov, short) is getattr(ov, long)


# --- every one of the 253 ledger locators is importable and callable -------


@pytest.mark.parametrize(
    "record",
    _load_ledger_records(),
    ids=lambda record: record["id"],
)
def test_every_ledger_locator_is_importable_and_callable(record: dict[str, Any]) -> None:
    module_path, _, function_name = record["python_locator"].partition(":")
    module = importlib.import_module(module_path)
    callable_obj = getattr(module, function_name)
    assert callable(callable_obj)
    assert inspect.isfunction(callable_obj) or inspect.iscoroutinefunction(callable_obj)


def test_ledger_covers_exactly_255_locators_across_22_domains() -> None:
    records = _load_ledger_records()
    assert len(records) == 255
    assert {record["domain"] for record in records} == CANONICAL_DOMAINS


# --- the 15 restored historical package-level exports -----------------------


@pytest.mark.parametrize(
    ("domain", "name"),
    [(domain, name) for domain, names in RESTORED_PACKAGE_EXPORTS.items() for name in names],
)
def test_restored_export_is_reachable_from_its_domain_package(domain: str, name: str) -> None:
    domain_module = importlib.import_module(f"organelleverse.{domain}")
    assert hasattr(domain_module, name), f"organelleverse.{domain}.{name} is still missing"
    assert callable(getattr(domain_module, name))


def test_restored_export_is_the_same_object_as_its_canonical_implementation() -> None:
    """Re-export only: the package attribute must be the implementation itself.

    Loaded via ``importlib.import_module`` rather than ``import a.b.c as x``:
    several of these submodules (``kaks_calculator``, ``train``, ``localize``)
    share their name with a function the domain package re-exports under the
    same spelling, which shadows the submodule as a *package attribute* —
    exactly like the archive's own historical ``__init__.py`` did. Only a
    ``sys.modules``-backed lookup reaches the real submodule regardless.
    """
    import organelleverse.localization as localization
    import organelleverse.morphology as morphology
    import organelleverse.pangenome as pangenome
    import organelleverse.phylogeny as phylogeny
    import organelleverse.population as population
    import organelleverse.selection as selection

    localize_impl = importlib.import_module("organelleverse.localization.localize")
    morphology_service = importlib.import_module("organelleverse.morphology.service")
    train_impl = importlib.import_module("organelleverse.morphology.train")
    pangenome_service = importlib.import_module("organelleverse.pangenome.service")
    hapnet_impl = importlib.import_module("organelleverse.phylogeny.hapnet")
    phylogeny_service = importlib.import_module("organelleverse.phylogeny.service")
    population_service = importlib.import_module("organelleverse.population.service")
    selection_service = importlib.import_module("organelleverse.selection.service")

    assert selection.fasta_to_axt is selection_service.fasta_to_axt
    assert (
        selection.run_kaks_calculator_workflow is selection_service.run_kaks_calculator_workflow
    )
    assert selection.run_codeml is selection_service.run_codeml
    assert phylogeny.run_trim_alignment is phylogeny_service.run_trim_alignment
    assert phylogeny.run_build_tree is phylogeny_service.run_build_tree
    assert phylogeny.render_network is hapnet_impl.render_network
    assert morphology.segment is morphology_service.segment
    assert morphology.train is morphology_service.train
    assert morphology.training_code_snippet is train_impl.training_code_snippet
    assert localization.predict is localize_impl.predict
    assert localization.localize is localize_impl.localize
    assert localization.localize is localization.predict
    assert pangenome.build_graph is pangenome_service.build_graph
    assert population.call_variants is population_service.call_variants
    assert population.cytonuclear_gwas is population_service.cytonuclear_gwas
    assert population.prepare_gemma_input is population_service.prepare_gemma_input


def test_restored_exports_keep_their_original_signature() -> None:
    """Public facades expose the selected scientific implementation."""
    import organelleverse.localization as localization
    import organelleverse.morphology as morphology
    import organelleverse.pangenome as pangenome
    import organelleverse.phylogeny as phylogeny
    import organelleverse.population as population
    import organelleverse.selection as selection

    localize_impl = importlib.import_module("organelleverse.localization.localize")
    morphology_service = importlib.import_module("organelleverse.morphology.service")
    train_impl = importlib.import_module("organelleverse.morphology.train")
    pangenome_service = importlib.import_module("organelleverse.pangenome.service")
    hapnet_impl = importlib.import_module("organelleverse.phylogeny.hapnet")
    phylogeny_service = importlib.import_module("organelleverse.phylogeny.service")
    population_service = importlib.import_module("organelleverse.population.service")
    selection_service = importlib.import_module("organelleverse.selection.service")

    # Individual assertions, not a loop over a combined list: unifying all 14
    # callables' types into one collection forces pyright to widen them into a
    # single Union, which goes fully unknown the moment any one signature (here,
    # hapnet.render_network's pre-existing bare `haplotypes: list[dict]`) is
    # only partially typed — a restoration task must not "fix" that annotation
    # just to appease the checker.
    assert inspect.signature(selection.fasta_to_axt) == inspect.signature(
        selection_service.fasta_to_axt
    )
    assert inspect.signature(selection.run_kaks_calculator_workflow) == inspect.signature(
        selection_service.run_kaks_calculator_workflow
    )
    assert inspect.signature(selection.run_codeml) == inspect.signature(selection_service.run_codeml)
    assert inspect.signature(phylogeny.run_trim_alignment) == inspect.signature(
        phylogeny_service.run_trim_alignment
    )
    assert inspect.signature(phylogeny.run_build_tree) == inspect.signature(
        phylogeny_service.run_build_tree
    )
    assert inspect.signature(phylogeny.render_network) == inspect.signature(
        hapnet_impl.render_network
    )
    assert inspect.signature(morphology.segment) == inspect.signature(morphology_service.segment)
    assert inspect.signature(morphology.train) == inspect.signature(morphology_service.train)
    assert inspect.signature(morphology.training_code_snippet) == inspect.signature(
        train_impl.training_code_snippet
    )
    assert inspect.signature(localization.predict) == inspect.signature(localize_impl.predict)
    assert inspect.signature(pangenome.build_graph) == inspect.signature(pangenome_service.build_graph)
    assert inspect.signature(population.call_variants) == inspect.signature(
        population_service.call_variants
    )
    assert inspect.signature(population.cytonuclear_gwas) == inspect.signature(
        population_service.cytonuclear_gwas
    )
    assert inspect.signature(population.prepare_gemma_input) == inspect.signature(
        population_service.prepare_gemma_input
    )


@pytest.mark.parametrize("domain", sorted(RESTORED_PACKAGE_EXPORTS))
def test_restored_domain_declares_the_export_in_all(domain: str) -> None:
    module = importlib.import_module(f"organelleverse.{domain}")
    declared = set(module.__all__)
    for name in RESTORED_PACKAGE_EXPORTS[domain]:
        assert name in declared, f"organelleverse.{domain}.__all__ omits restored export {name!r}"


# --- bare import stays light; nothing here changed that ---------------------


def test_bare_import_does_not_load_scientific_or_heavy_backends() -> None:
    script = f"""
import sys
import organelleverse  # noqa: F401
heavy = ("matplotlib", "scipy", "skimage", "tensorflow", "torch")
loaded_heavy = [name for name in heavy if name in sys.modules]
domains = {sorted(CANONICAL_DOMAINS)!r}
loaded_domains = [name for name in domains if f"organelleverse.{{name}}" in sys.modules]
assert loaded_heavy == [], loaded_heavy
assert loaded_domains == [], loaded_domains
"""
    subprocess.run([sys.executable, "-c", script], check=True, env=child_env())


def test_accessing_a_domain_twice_returns_the_cached_module() -> None:
    first = ov.selection
    second = ov.selection
    assert first is second
    assert ov.selection is ov.selection


# --- unchanged surrounding contracts -----------------------------------------


def test_registry_still_has_exactly_the_twenty_core_operations() -> None:
    assert len({spec.operation_id for spec in ov.operations.list()}) == 20


def test_contracted_suites_still_empty() -> None:
    from tests.contracts.test_legacy_removed import CONTRACTED_SUITES

    assert CONTRACTED_SUITES == []


def test_save_root_alias_still_forbidden() -> None:
    assert not hasattr(ov, "save")


def test_inventory_is_still_not_a_runtime_contract_after_restoration() -> None:
    """Restoring package exports must not have accidentally registered them."""
    registered_ids = {spec.operation_id for spec in ov.operations.list()}
    ledger_locators = {record["python_locator"] for record in _load_ledger_records()}
    assert registered_ids.isdisjoint(ledger_locators)
