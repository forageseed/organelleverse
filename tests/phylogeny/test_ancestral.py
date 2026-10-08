"""Independent enumeration checks pruning, root prior and outside messages."""

import itertools
import math

import numpy as np
import pytest
from Bio import Phylo, SeqIO
from Bio.Seq import Seq
from Bio.SeqFeature import SeqFeature, SimpleLocation
from Bio.SeqRecord import SeqRecord

from organelleverse.core.errors import OrganelleInputError
from organelleverse.phylogeny import gene_presence_traits, reconstruct_ancestral_states


def enumerate_posterior(tree, observed, k, rate):
    nodes = list(tree.find_clades(order="preorder"))
    ix = {node: i for i, node in enumerate(nodes)}
    joint = np.zeros((len(nodes), k))
    total = 0
    for assignment in itertools.product(range(k), repeat=len(nodes)):
        if any(
            observed.get(n.name) is not None and assignment[ix[n]] != observed[n.name]
            for n in tree.get_terminals()
        ):
            continue
        weight = 1 / k
        for parent in nodes:
            for child in parent.clades:
                e = math.exp(-rate * child.branch_length * k / (k - 1))
                weight *= (1 - e) / k + (
                    e if assignment[ix[parent]] == assignment[ix[child]] else 0
                )
        total += weight
        for i, state in enumerate(assignment):
            joint[i, state] += weight
    return joint / total, math.log(total)


@pytest.mark.parametrize("k", [2, 3])
@pytest.mark.parametrize("newick", ["((A:0.1,B:0.2):0.3,C:0.4);", "(A:0.1,B:0.2,C:0.4);"])
def test_marginals_match_exhaustive_joint_sum(tmp_path, k, newick):
    path = tmp_path / "tree.nwk"
    path.write_text(newick)
    observed = {"A": 0, "B": k - 1, "C": None}
    result = reconstruct_ancestral_states(
        path,
        {n: {"trait": v} for n, v in observed.items()},
        state_space={"trait": list(map(str, range(k)))},
        rate=0.7,
    )
    expected, loglike = enumerate_posterior(Phylo.read(path, "newick"), observed, k, 0.7)
    actual = np.array([list(n["posterior"].values()) for n in result.metrics["traits"]["trait"]])
    np.testing.assert_allclose(actual, expected, atol=1e-13)
    assert result.metrics["models"]["trait"]["log_likelihood"] == pytest.approx(loglike, abs=1e-13)
    np.testing.assert_allclose(actual.sum(axis=1), 1)


def test_fitted_likelihood_at_least_dense_independent_grid(tmp_path):
    path = tmp_path / "tree.nwk"
    path.write_text("((A:0.01,B:0.01):1,C:0.02);")
    obs = {"A": 0, "B": 0, "C": 1}
    r = reconstruct_ancestral_states(
        path, {n: {"g": v} for n, v in obs.items()}, rate_bounds=[0.001, 1000]
    )
    tree = Phylo.read(path, "newick")
    grid_max = max(
        enumerate_posterior(tree, obs, 2, rate)[1] for rate in np.geomspace(0.001, 1000, 501)
    )
    assert r.metrics["models"]["g"]["log_likelihood"] >= grid_max - 1e-8


def test_ties_and_zero_branches_are_not_fabricated_changes(tmp_path):
    path = tmp_path / "tree.nwk"
    path.write_text("(A:1,B:1);")
    r = reconstruct_ancestral_states(path, {"A": {"g": 0}, "B": {"g": 1}}, rate=1)
    assert r.metrics["traits"]["g"][0]["most_likely_state"] is None
    assert r.metrics["changes"] == ()
    path.write_text("(A:0,B:0);")
    with pytest.raises(OrganelleInputError, match="impossible"):
        reconstruct_ancestral_states(path, {"A": {"g": 0}, "B": {"g": 1}}, rate=1)
    r = reconstruct_ancestral_states(
        path, {"A": {"g": 0}, "B": {"g": 0}}, rate=0, state_space={"g": ["0", "1"]}
    )
    assert r.metrics["traits"]["g"][0]["posterior"]["0"] == 1


@pytest.mark.parametrize(
    "tree,traits,error",
    [
        ("(A:1,B:1);", {"A": {"g": 0}}, "exactly match"),
        ("(A:1,B:1);", {"A": {"g": 0}, "B": {"x": 1}}, "same nonempty"),
        ("(A,B:1);", {"A": {"g": 0}, "B": {"g": 1}}, "explicit finite"),
        ("(A:-1,B:1);", {"A": {"g": 0}, "B": {"g": 1}}, "explicit finite"),
        ("(A:1,B:1);", {"A": {"g": None}, "B": {"g": None}}, "observations"),
    ],
)
def test_input_boundary(tmp_path, tree, traits, error):
    path = tmp_path / "tree.nwk"
    path.write_text(tree)
    with pytest.raises(OrganelleInputError, match=error):
        reconstruct_ancestral_states(path, traits)


def test_genbank_presence_pseudogenes_and_absence_semantics(tmp_path):
    record = SeqRecord(Seq("ATG" * 100), id="sample", annotations={"molecule_type": "DNA"})
    record.features = [
        SeqFeature(SimpleLocation(0, 30), type="CDS", qualifiers={"gene": ["accD"]}),
        SeqFeature(
            SimpleLocation(30, 60), type="CDS", qualifiers={"gene": ["ycf1"], "pseudo": [""]}
        ),
    ]
    path = tmp_path / "sample.gb"
    SeqIO.write(record, path, "genbank")
    result = gene_presence_traits([path], ["accD", "ycf1", "ycf2"])
    assert result.metrics["traits"]["sample"] == {"accD": 1, "ycf1": 0, "ycf2": 0}
    assert result.metrics["absence_definition"] == "not annotated as CDS"
    assert (
        gene_presence_traits([path], ["ycf1"], include_pseudogenes=True).metrics["traits"][
            "sample"
        ]["ycf1"]
        == 1
    )


def test_constant_trait_declared_alphabet_and_rate_boundary(tmp_path):
    path = tmp_path / "tree.nwk"
    path.write_text("(A:0.1,B:0.1);")
    r = reconstruct_ancestral_states(
        path, {"A": {"g": 0}, "B": {"g": 0}}, state_space={"g": ["0", "1"]}
    )
    assert r.status == "warning"
    assert r.metrics["models"]["g"]["rate"] == 1e-6
    assert r.metrics["models"]["g"]["rate_at_boundary"]


@pytest.mark.parametrize(
    "newick,observations",
    [
        ("(A:0,B:0);", {"A": {"g": 0}, "B": {"g": 0}}),
        ("(A:1,B:1);", {"A": {"g": 0}, "B": {"g": None}}),
        ("((A:0,B:0):1,C:1);", {"A": {"g": 0}, "B": {"g": 0}, "C": {"g": None}}),
    ],
)
def test_unidentifiable_rate_requires_explicit_parameter(tmp_path, newick, observations):
    path = tmp_path / "tree.nwk"
    path.write_text(newick)
    with pytest.raises(OrganelleInputError, match="not identifiable"):
        reconstruct_ancestral_states(path, observations, state_space={"g": ["0", "1"]})
    assert (
        reconstruct_ancestral_states(
            path,
            observations,
            state_space={"g": ["0", "1"]},
            rate=1,
        ).status
        == "ok"
    )
