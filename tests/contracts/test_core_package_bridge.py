from organelleverse.core import OrganelleData, OrganelleGenome, OrganelleResult
from organelleverse.core.data import OrganelleData as V1OrganelleData
from organelleverse.core.genome import OrganelleGenome as V1OrganelleGenome
from organelleverse.core.result import OrganelleResult as V1OrganelleResult


def test_core_package_exports_only_canonical_contracts() -> None:
    assert OrganelleData is V1OrganelleData
    assert OrganelleGenome is V1OrganelleGenome
    assert OrganelleResult is V1OrganelleResult
