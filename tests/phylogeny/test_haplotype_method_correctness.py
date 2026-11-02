"""Scientific counterexamples for haplotype-network algorithms."""
from itertools import permutations

import pytest

from organelleverse.phylogeny import hapnet


def haplotypes(*sequences):
    return [{"id": f"H{i}", "sequence": sequence} for i, sequence in enumerate(sequences)]


def test_minimum_spanning_network_retains_all_equal_optimum_links():
    edges, components = hapnet.build_msn(haplotypes("AA", "AC", "CC", "CA"))
    assert len(edges) == 4
    assert all(distance == 1 for _, _, distance in edges)
    assert len(components) == 1


@pytest.mark.parametrize("sequences", list(permutations(("CAA", "ACA", "AAC"))))
def test_median_joining_recovers_the_unsampled_star_center(sequences):
    edges, components, medians = hapnet.build_mjn(haplotypes(*sequences))
    assert {median["sequence"] for median in medians} == {"AAA"}
    assert sum(distance for _, _, distance in edges) == 3
    assert len(edges) == 3 and len(components) == 1


def test_quasi_medians_retain_all_three_way_character_ties():
    assert hapnet._quasi_medians("AA", "CC", "GG") == {
        a + b for a in "ACG" for b in "ACG"
    }


def test_tcs_collapsed_nodes_are_absent_from_the_result_and_components():
    edges, components, intermediates = hapnet.build_tcs_network(
        haplotypes("AAA", "CCC"), {"H0": ["a"], "H1": ["b"]}, connection_limit=3
    )
    assert edges == [("H0", "H1", 3)]
    assert intermediates == []
    assert list(components.values()) == [["H0", "H1"]]


def test_tcs_retained_intermediate_nodes_match_the_graph():
    edges, components, intermediates = hapnet.build_tcs_network(
        haplotypes("AAA", "CCC"), {"H0": ["a"], "H1": ["b"]},
        connection_limit=3, collapse_degree2=False,
    )
    assert len(edges) == 3 and len(intermediates) == 2
    assert len(components) == 1
    assert set(next(iter(components.values()))) == {"H0", "H1"} | {
        node["id"] for node in intermediates
    }


def test_tcs_reuses_a_shared_branching_intermediate():
    edges, components, intermediates = hapnet.build_tcs_network(
        haplotypes("AAA", "ACC", "CCA"),
        {"H0": ["a"], "H1": ["b"], "H2": ["c"]}, connection_limit=2,
    )
    assert len(intermediates) == 1 and len(edges) == 3 and len(components) == 1
    center = intermediates[0]["id"]
    assert all(center in (a, b) and distance == 1 for a, b, distance in edges)


def test_tcs_does_not_collapse_an_observed_node_named_like_an_intermediate():
    haps = [{"id": "mv0", "sequence": "AAA"}, {"id": "H1", "sequence": "CCC"}]
    edges, components, intermediates = hapnet.build_tcs_network(
        haps, {"mv0": ["a"], "H1": ["b"]}, connection_limit=3
    )
    assert {node for a, b, _ in edges for node in (a, b)} == {"mv0", "H1"}
    assert intermediates == [] and len(components) == 1


def test_mjn_iteration_limit_is_not_reported_as_a_converged_network():
    with pytest.raises(RuntimeError, match="converge"):
        hapnet.build_mjn(haplotypes("CAA", "ACA", "AAC"), max_iterations=1)


def test_tcs_uses_sequence_length_and_not_number_of_sampled_haplotypes():
    # Published TCS analysis of the 521-base HTLV-1 alignment: nine steps.
    assert hapnet.tcs_connection_limit(168, 521, .95) == 9
    assert hapnet.tcs_connection_limit(10, 521, .95) == 9
    assert hapnet.tcs_connection_limit(10, 100, .95) < hapnet.tcs_connection_limit(10, 1000, .95)


def test_tcs_does_not_fabricate_a_limit_without_comparable_length():
    with pytest.raises(ValueError, match="sequence_length"):
        hapnet.tcs_connection_limit(10)


def test_tcs_analytic_moments_agree_with_independent_probability_integrals():
    from scipy.integrate import quad

    steps, m = 5, 95
    expected = 1.0
    for i in range(steps):
        def density(q):
            return q**i * (1-q)**(2*m+1) * (1-q/3) * (2-4*q/3)**i * (1-2*q+2*q*q/3)
        expected *= 1 - quad(lambda q: q * density(q), 0, 1, epsabs=1e-20)[0] / quad(density, 0, 1, epsabs=1e-20)[0]
    assert hapnet._tcs_parsimony_probability(steps, m) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize(("steps", "shared", "reference"), [
    (50, 0, 2.778094897668209e-30),
    (80, 0, 1.6647772964024045e-49),
    (50, 10, 9.13173228230067e-11),
])
def test_tcs_probability_remains_accurate_without_alternating_sum_cancellation(steps, shared, reference):
    # Independently log-scaled original integrand + scipy.quad, not a
    # recurrence or polynomial expansion sharing the implementation.
    actual = hapnet._tcs_parsimony_probability(steps, shared)
    assert 0 <= actual <= 1
    assert actual == pytest.approx(reference, rel=1e-8, abs=0)
