"""Gene-name aliases of the mitochondrial scoring scripts."""

from __future__ import annotations

import importlib.util
import sys
from types import SimpleNamespace

from tests._paths import PROJECT_ROOT

SCRIPTS = PROJECT_ROOT / "scripts" / "annotation_benchmarks"


def _load(name: str):
    path = SCRIPTS / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


aliases = _load("gene_name_aliases.py")
KNOWN = frozenset({"atp1", "atp6", "nad4l", "nad6", "cob", "rpl5", "ccmfc", "rps12"})


def test_legacy_mitochondrial_symbols_fold_to_the_gene():
    for raw, gene in {"ND1": "nad1", "nd4L": "nad4l", "ND9": "nad9", "CYTB": "cob"}.items():
        assert aliases.fold_alias(raw, KNOWN) == gene


def test_typos_and_product_names_fold_to_the_gene():
    assert aliases.fold_alias("rp15", KNOWN) == "rpl5"
    assert aliases.fold_alias("mat-R", KNOWN) == "matr"
    product = "Sec-independent protein translocase protein"
    assert aliases.fold_alias(product, KNOWN) == "mttb"
    # the same name once a normaliser has removed its hyphens
    assert aliases.fold_alias("secindependentproteintranslocaseprotein", KNOWN) == "mttb"


def test_ccmfc_fragments_fold_onto_ccmfc():
    assert aliases.fold_alias("ccmFc1", KNOWN) == "ccmfc"
    assert aliases.fold_alias("ccmFc2", KNOWN) == "ccmfc"


def test_numbered_copies_fold_when_the_stem_is_a_known_gene():
    assert aliases.fold_alias("atp6-1", KNOWN) == "atp6"
    assert aliases.fold_alias("atp1-3", KNOWN) == "atp1"
    assert aliases.fold_alias("nad4L-2", KNOWN) == "nad4l"
    assert aliases.fold_alias("nad6-2", KNOWN) == "nad6"


def test_a_name_that_only_looks_like_a_copy_is_left_alone():
    assert aliases.fold_alias("orf25-1", KNOWN) == "orf25-1"
    assert aliases.fold_alias("trnK-UUU", KNOWN) == "trnK-UUU"
    assert aliases.fold_alias("cox1", KNOWN) == "cox1"


def test_neighbouring_genes_are_not_aliases():
    # matR sits next to rps10 in some genomes; a prediction on the neighbour is a different gene
    assert "rps10" not in aliases.ALIASES
    assert aliases.fold_alias("rps10", KNOWN) == "rps10"
    assert aliases.fold_alias("sdh4", KNOWN) == "sdh4"


def test_known_genes_come_from_the_packaged_references():
    known = aliases.known_cds_genes()
    assert {"atp6", "cox1", "nad4l", "rps12", "ccmfc"} <= known
    # every alias resolves to a gene the references or the extras know
    assert set(aliases.ALIASES.values()) <= known


def test_install_folds_only_cds_and_reports_changes():
    seen = []
    module = SimpleNamespace(canon=lambda name, kind: name.lower())
    aliases.install(module, on_change=lambda raw, before, after: seen.append((raw, before, after)))
    assert module.canon("ND5", "CDS") == "nad5"
    assert module.canon("atp6-2", "CDS") == "atp6"
    assert module.canon("ND5", "tRNA") == "nd5"
    assert module.canon("cox1", "CDS") == "cox1"
    assert seen == [("ND5", "nd5", "nad5"), ("atp6-2", "atp6-2", "atp6")]


def test_the_core_benchmark_script_uses_the_aliases():
    script = _load("run_mito_core_benchmark.py")
    assert script.ref_canon("ND1", False) == "nad1"
    assert script.ref_canon("CYTB", False) == "cob"
    assert script.ref_canon("rp15", False) == "rpl5"
    assert script.ref_canon("mat-R", False) == "matr"
    assert script.ref_canon("atp6-1", True) == "atp6"
    assert script.ref_canon("trnC-GCA", False) == "trncgca"
