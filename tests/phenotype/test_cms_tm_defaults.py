"""The CMS hydrophobicity hint uses the Kyte-Doolittle transmembrane rule (19 residues, mean >= 1.6)."""

from __future__ import annotations

import inspect
import random

from organelleverse._bio import read_fasta
from organelleverse.phenotype.cms.core import compute_cms_candidates
from organelleverse.phenotype.cms.evidence import assess_cms_candidates
from organelleverse.phenotype.cms.pipeline import _has_tm, cms
from organelleverse.phenotype.cms.resources import cms_protein_fasta


def _default(fn, name):
    return inspect.signature(inspect.unwrap(fn)).parameters[name].default


def test_defaults_are_the_kyte_doolittle_rule_everywhere() -> None:
    assert (_default(cms, "tm_window"), _default(cms, "tm_threshold")) == (19, 1.6)
    assert (_default(assess_cms_candidates, "tm_window"), _default(assess_cms_candidates, "tm_threshold")) == (19, 1.6)
    assert _default(compute_cms_candidates, "tm_threshold") == 1.6


def test_checked_cms_seeds_pass_and_random_orfs_mostly_do_not() -> None:
    seeds = dict(read_fasta(cms_protein_fasta()))
    assert seeds, "packaged accession-checked CMS seed is empty"
    assert all(_has_tm(seq, 19, 1.6) for seq in seeds.values())  # orf79, orf138
    rng = random.Random(11)
    orfs = ["M" + "".join(rng.choice("ACDEFGHIKLMNPQRSTVWY") for _ in range(199)) for _ in range(1000)]
    new_rate = sum(_has_tm(s, 19, 1.6) for s in orfs) / len(orfs)
    old_rate = sum(_has_tm(s, 21, 0.5) for s in orfs) / len(orfs)
    assert new_rate < 0.15 < old_rate  # about 6% vs 84% for 200-aa random ORFs
