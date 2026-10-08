"""Routing evidence for an explicit backend with a genome size (PMAT -g)."""

from organelleverse.assembly.profiles import AssemblyProfile
from organelleverse.assembly.routing import AssemblyRoute, RoutingContext
from organelleverse.assembly.service import _routing_evidence


def _route(rule_id: str) -> AssemblyRoute:
    return AssemblyRoute(
        requested_method="pmat",
        selected_backend="pmat",
        profile=AssemblyProfile.PACBIO_HIFI,
        rule_id=rule_id,
        compatible_candidates=("pmat",),
    )


def test_explicit_backend_with_genome_size_carries_the_coverage_display():
    context = RoutingContext(
        profile=AssemblyProfile.PACBIO_HIFI,
        taxon_group="plant",
        total_bases=27_000_000,
        genome_size_bp=135_000_000,
    )
    evidence = _routing_evidence(_route("explicit.backend"), context)
    assert evidence.coverage_display == "0.20x"
    assert evidence.threshold_multiplier is None
    assert evidence.threshold_numerator is None
    assert evidence.threshold_denominator is None


def test_coverage_routing_keeps_its_threshold_evidence():
    context = RoutingContext(
        profile=AssemblyProfile.PACBIO_HIFI,
        taxon_group="plant",
        total_bases=3_000,
        genome_size_bp=1_000,
        threshold_multiplier=3,
    )
    evidence = _routing_evidence(_route("auto.mito_hifi_lte_3x_pmat"), context)
    assert evidence.coverage_display == "3.00x"
    assert (evidence.threshold_numerator, evidence.threshold_denominator) == (3_000, 3_000)


def test_without_genome_size_there_is_no_display():
    context = RoutingContext(profile=AssemblyProfile.PACBIO_HIFI, taxon_group="plant")
    assert _routing_evidence(_route("explicit.backend"), context).coverage_display is None
