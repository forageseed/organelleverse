import organelleverse.core as core
from organelleverse.core.data import OrganelleData
from organelleverse.core.genome import OrganelleGenome
from organelleverse.core.provenance import ResultProvenance
from organelleverse.core.result import OrganelleResult


def test_core_exports_canonical_contracts_without_migration_aliases() -> None:
    assert core.OrganelleGenome is OrganelleGenome
    assert core.OrganelleData is OrganelleData
    assert core.OrganelleResult is OrganelleResult
    assert core.ResultProvenance is ResultProvenance
    assert not any(name.startswith("_V1") for name in vars(core))
