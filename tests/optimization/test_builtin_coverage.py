from organelleverse.capabilities.discovery import discover_capabilities
from organelleverse.capabilities.index import CapabilityStatus
from organelleverse.optimization.coverage import (
    builtin_classifications,
    builtin_profiles,
    coverage_matrix,
)


def test_every_core_manifest_has_an_explicit_reviewed_optimization_classification():
    index = discover_capabilities()
    expected = {
        entry.capability_id
        for entry in index.entries
        if all(origin.channel == "core" for origin in entry.origins)
    }
    assert set(builtin_classifications()) == expected
    matrix = coverage_matrix(index)
    assert matrix["total_suites"] == len({name.split(".")[0] for name in expected})
    assert matrix["total_suites"] >= 26
    assert matrix["total_capabilities"] == len(expected)
    assert matrix["counts"]["enabled"] == 0
    assert sum(matrix["counts"].values()) == len(expected)
    assert all(row["reason_code"] for row in matrix["rows"])


def test_eligibility_never_bypasses_admission_or_claims_a_scientific_score():
    index = discover_capabilities()
    profiles = builtin_profiles(index)
    for entry in index.entries:
        if entry.status is not CapabilityStatus.ADMITTED:
            assert entry.capability_id not in profiles
    rows = {row["capability_id"]: row for row in coverage_matrix(index)["rows"]}
    assert rows["pangenome.build_graph"]["status"] == "eligible"
    assert rows["pangenome.convert_graph_result"]["status"] == "not_applicable"
    assert all(row["contract"] is None for row in rows.values())
