"""Architectural guard: what came back, what must stay gone, and what is contracted.

This file used to assert that 22 scientific suites and the pre-v1 dataclass shim
were "gone for good". They are back — deleted on 2026-07-27 and restored on
2026-07-29 — so the guard is inverted rather than dropped. The discipline it
encoded is still the one worth keeping; only its polarity changed:

- the suites must be **importable**, not absent;
- a suite becomes part of the released catalog by **registering a contract**,
  and ``CONTRACTED_SUITES`` is the ledger of which ones have;
- the 20 short human-facing names (``ov.viz``, ``ov.erc``, ``ov.phylo`` …) are
  approved and restored: each is a lazy route to the *same* module object as
  its full name (``ov.viz is ov.visualization``), so there is one
  implementation, one contract, and one Registry ID — see
  ``SHORT_MODULE_ALIASES`` below;
- ``ov.save`` stays forbidden: unlike a short module name, it is a second
  spelling of ``write`` inside one module — ambiguity with no shortening
  benefit;
- no production source may import the legacy shim, which is what finally lets
  ``core/legacy.py`` be removed.

``CONTRACTED_SUITES`` grows one entry at a time as contracts land. It is a
ledger, not a target: nothing here is expected to fail while it is incomplete.
"""

from __future__ import annotations

import importlib.util
from importlib.machinery import ModuleSpec
from pathlib import Path

import pytest

import organelleverse as ov

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src" / "organelleverse"

# Scientific suites restored on 2026-07-29. Each must import.
RESTORED_SUITES = [
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
]

# Suites that publish `materialize_result`, so `ov.write` can route them.
# Grows as suites gain a publication boundary.
MATERIALIZING_SUITES = [
    "coevolution",
    "ir_boundary",
    "phylogeny",
    "selection",
    "structure",
    "visualization",
]

# Suites whose operations are registered in the released catalog.
# EMPTY BY DESIGN: restoration and contracting are separate steps, and a suite
# is only listed here once its OperationSpec is registered and verified.
CONTRACTED_SUITES: list[str] = []

# The released catalog before any restored suite is contracted.
CORE_OPERATIONS = {
    "annotation.annotate",
    "annotation.extract",
    "annotation.write",
    "assembly.assemble",
    "assembly.pmat_graph_build",
    "assembly.write",
    "fetch.accession_list",
    "fetch.dedup_genomes",
    "fetch.entrez_query",
    "fetch.gene_records",
    "fetch.genome_size_candidates",
    "fetch.ngdc_gwh",
    "fetch.nuclear_assembly",
    "fetch.refseq_snapshot",
    "io.read_fasta_genome",
    "io.read_genbank_genome",
    "io.read_long_reads",
    "qc.annotation",
    "qc.assembly",
    "qc.write",
}

# Short human-facing names, the spelling people actually type — OmicVerse's
# ``pp`` / ``pl`` convention. Each resolves to the *same module object* as its
# full name, so there is one implementation, one contract, and one registry id.
SHORT_MODULE_ALIASES = {
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

# ``save`` stays gone: unlike a short module name, it is a second spelling of
# ``write`` inside one module — ambiguity with no shortening benefit.
FORBIDDEN_ROOT_ATTRS = ["save"]

CANONICAL_ROOT_ATTRS = [
    "annotation",
    "assembly",
    "assets",
    "fetch",
    "operations",
    "environments",
    "io",
    "qc",
    "report",
    "read",
    "write",
    "OrganelleGenome",
    "OrganelleData",
    "OrganelleResult",
]

FORBIDDEN_MODULE_ATTRS = {
    "assembly": {
        "BACKEND_ORGANELLES",
        "SUPPORTED_METHODS",
        "classify_reads",
        "detect_data_type",
        "list_supported_methods",
        "recommend_assembly_method",
        "recommend_method",
    },
    "fetch": {"accessions", "entrez", "refseq", "save", "write"},
    "report": {"report", "save"},
    "qc": {"run", "save"},
    "operations": {"json_schema"},
}


def _spec(name: str) -> ModuleSpec | None:
    """find_spec with a clean view of the organelleverse package only."""
    return importlib.util.find_spec(f"organelleverse.{name}")


# --- what came back --------------------------------------------------------


@pytest.mark.parametrize("name", RESTORED_SUITES)
def test_restored_suites_are_importable(name: str) -> None:
    """Every restored suite must be reachable; none may silently drop out."""
    assert _spec(name) is not None, f"restored suite is missing: organelleverse.{name}"
    importlib.import_module(f"organelleverse.{name}")


@pytest.mark.parametrize("name", MATERIALIZING_SUITES)
def test_materializing_suites_publish_a_materializer(name: str) -> None:
    """A suite claiming a publication boundary must actually provide one."""
    from organelleverse.writer import materializing_suites

    assert name in materializing_suites()


def test_restored_suites_do_not_regress_out_of_the_writer() -> None:
    """Once a suite can be written, it stays writable."""
    from organelleverse.writer import materializing_suites

    published = set(materializing_suites())
    assert set(MATERIALIZING_SUITES) <= published


# --- what stays gone -------------------------------------------------------


@pytest.mark.parametrize(("short", "full"), sorted(SHORT_MODULE_ALIASES.items()))
def test_short_names_resolve_to_the_same_module(short: str, full: str) -> None:
    """A short name is a route, not a copy: same object, so no second contract."""
    assert getattr(ov, short) is getattr(ov, full)


@pytest.mark.parametrize(("short", "full"), sorted(SHORT_MODULE_ALIASES.items()))
def test_short_names_share_every_callable(short: str, full: str) -> None:
    """Identity at the module level means identity at the function level."""
    module = getattr(ov, full)
    for attribute in dir(module):
        if attribute.startswith("_"):
            continue
        assert getattr(getattr(ov, short), attribute) is getattr(module, attribute)


@pytest.mark.parametrize("name", FORBIDDEN_ROOT_ATTRS)
def test_root_does_not_expose_forbidden_names(name: str) -> None:
    assert not hasattr(ov, name), f"root still exposes compatibility name: ov.{name}"


@pytest.mark.parametrize(
    ("module_name", "attribute"),
    [
        (module_name, attribute)
        for module_name, attributes in FORBIDDEN_MODULE_ATTRS.items()
        for attribute in sorted(attributes)
    ],
)
def test_modules_do_not_expose_compatibility_names(module_name: str, attribute: str) -> None:
    module = getattr(ov, module_name)
    assert not hasattr(module, attribute), (
        f"compatibility name remains: ov.{module_name}.{attribute}"
    )


def test_removed_external_installer_is_gone() -> None:
    assert _spec("tools.install_external_backends") is None


# --- the port-completion criterion -----------------------------------------


def test_no_source_references_core_legacy() -> None:
    """No production source may import the pre-v1 shim.

    This is the definition of "the port is finished": when it passes,
    ``core/legacy.py`` has no consumers left and can be removed.
    """
    offenders: list[str] = []
    for path in SRC_ROOT.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if path.name == "legacy.py" and path.parent.name == "core":
            continue
        if "core.legacy" in text or "core import legacy" in text or ".legacy import" in text:
            offenders.append(str(path.relative_to(SRC_ROOT)))
    assert not offenders, f"source files still reference core.legacy: {offenders}"


# --- the released catalog --------------------------------------------------


def test_canonical_root_surface_is_present() -> None:
    for name in CANONICAL_ROOT_ATTRS:
        assert hasattr(ov, name), f"root missing canonical surface: ov.{name}"


def test_operations_catalog_matches_the_contracted_inventory() -> None:
    """The catalog is exactly the core operations plus the contracted suites.

    This is the migration's scoreboard. Registering a restored suite's
    operations requires adding its name to ``CONTRACTED_SUITES`` in the same
    change, so the catalog can never grow by accident.
    """
    registered = {spec.operation_id for spec in ov.operations.list()}
    contracted_prefixes = tuple(f"{suite}." for suite in CONTRACTED_SUITES)
    from_contracted = {op for op in registered if op.startswith(contracted_prefixes)}
    assert registered - from_contracted == CORE_OPERATIONS
    for suite in CONTRACTED_SUITES:
        assert any(op.startswith(f"{suite}.") for op in registered), (
            f"{suite} is listed as contracted but registers no operation"
        )


def test_every_operation_invocation_schema_is_generatable() -> None:
    for spec in ov.operations.list():
        schema = ov.operations.invocation_schema(spec.operation_id)
        assert isinstance(schema, dict)
        assert schema.get("type") == "object"


# --- native acceleration ---------------------------------------------------


def test_rust_source_declares_every_kernel_the_suites_call() -> None:
    """The Rust source must provide every kernel a suite guards on.

    Four kernels were deleted from the crate while callers kept referencing
    them; because ``HAS_RUST`` was still true, those calls raised
    ``AttributeError`` instead of taking their pure-Python fallback.
    """
    rust_source = (REPO_ROOT / "rust" / "src" / "lib.rs").read_text(encoding="utf-8")
    for symbol in (
        "cm_cyk",
        "cm_cyk_trace",
        "erc_correlation",
        "kmer_jaccard",
        "kmer_overlap",
        "mafft_align",
    ):
        assert symbol in rust_source, f"rust crate no longer provides {symbol}"


@pytest.mark.parametrize(
    "name",
    ["cm_cyk", "cm_cyk_trace", "erc_correlation", "kmer_jaccard", "kmer_overlap", "mafft_align"],
)
def test_accel_exposes_every_kernel_name(name: str) -> None:
    """Absent kernels must read as None, never raise AttributeError.

    Callers guard with ``accel.<kernel> is not None``; an attribute that does
    not exist turns a graceful fallback into a crash.
    """
    from organelleverse import accel

    assert hasattr(accel, name)


def test_unverified_kernels_are_withheld_from_the_default_backend() -> None:
    """A native kernel is not a default backend until it is pinned to a reference.

    ``kmer_jaccard`` is why: it shares ``encode_kmers`` with ``kmer_overlap``,
    which indexes reverse complements on purpose, so it computes a
    strand-agnostic Jaccard where the Python reference is strand-specific.
    """
    from organelleverse import accel

    assert set(accel.available_kernels()) <= set(accel.VERIFIED_KERNELS)
    for name in accel.UNVERIFIED_KERNELS:
        assert getattr(accel, name) is None, (
            f"{name} has no equivalence fixture and must not be reachable by default"
        )


def test_withheld_kernels_are_reported_rather_than_hidden() -> None:
    """A crate ahead of its fixtures must say so, not look like a build without them."""
    from organelleverse import accel

    if accel.HAS_RUST:
        assert set(accel.withheld_kernels()) == set(accel.UNVERIFIED_KERNELS)
