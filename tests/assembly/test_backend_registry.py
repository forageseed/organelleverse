from typing import get_args

import pytest
from pydantic import ValidationError

from organelleverse.assembly.backends import (
    BACKENDS,
    RUNTIMES,
    AssemblyBackendRegistry,
    AssemblyBackendSpec,
    AssemblyEnvironmentRef,
    AssemblyProfile,
    AssemblyRuntimeRegistry,
    AuxiliaryRole,
    EnvironmentCarrier,
    backend_ids,
    get_backend,
)
from organelleverse.assembly.contracts import AssemblyMethod
from organelleverse.core.errors import OrganelleContractError

EXPECTED_IDS = (
    "oatk", "himt", "getorganelle", "pmat", "tippo", "ptgaul", "novoplasty", "ovasm"
)

EXPECTED_PROFILES = {
    "oatk": ("pacbio_hifi",),
    "himt": (
        "pacbio_hifi",
        "pacbio_clr_raw",
        "pacbio_clr_corrected",
        "ont_raw",
        "ont_corrected",
        "ont_hq",
        "ont_duplex",
    ),
    "getorganelle": ("illumina_pe", "illumina_se"),
    "tippo": ("pacbio_hifi",),
    "ptgaul": ("ont_raw",),
    "novoplasty": ("illumina_pe",),
    "ovasm": (
        "pacbio_hifi",
        "pacbio_clr_raw",
        "ont_raw",
        "illumina_se",
        "illumina_pe",
        "ont_raw_illumina",
        "pacbio_clr_raw_illumina",
    ),
    "pmat": (
        "pacbio_hifi",
        "pacbio_clr_raw",
        "pacbio_clr_corrected",
        "ont_raw",
        "ont_corrected",
        "ont_hq",
        "ont_duplex",
    ),
}


def test_all_default_backend_surfaces_share_the_released_vocabulary() -> None:
    assert get_args(AssemblyMethod) == EXPECTED_IDS
    assert backend_ids() == EXPECTED_IDS
    assert tuple(BACKENDS) == EXPECTED_IDS
    assert tuple(RUNTIMES) == EXPECTED_IDS


def test_registry_rejects_duplicate_backend_id() -> None:
    spec = get_backend("oatk")
    with pytest.raises(OrganelleContractError) as raised:
        AssemblyBackendRegistry((spec, spec))
    assert raised.value.code == "contract.duplicate_assembly_backend"


def test_backends_cannot_be_mutated() -> None:
    with pytest.raises(TypeError, match="does not support item assignment"):
        BACKENDS["oatk"] = get_backend("himt")  # type: ignore[index]

    assert BACKENDS["oatk"] is get_backend("oatk")


def test_backend_spec_rejects_unknown_fields() -> None:
    payload = get_backend("oatk").model_dump(mode="python", round_trip=True)
    payload["unexpected"] = "value"

    with pytest.raises(ValidationError) as raised:
        AssemblyBackendSpec.model_validate(payload)

    assert raised.value.errors()[0]["type"] == "extra_forbidden"
    assert raised.value.errors()[0]["loc"] == ("unexpected",)


def test_environment_spec_rejects_unknown_fields() -> None:
    payload = get_backend("oatk").environment.model_dump(mode="python", round_trip=True)
    payload["unexpected"] = "value"

    with pytest.raises(ValidationError) as raised:
        AssemblyEnvironmentRef.model_validate(payload)

    assert raised.value.errors()[0]["type"] == "extra_forbidden"
    assert raised.value.errors()[0]["loc"] == ("unexpected",)


def test_backend_spec_rejects_attribute_reassignment() -> None:
    spec = get_backend("oatk")

    with pytest.raises(ValidationError) as raised:
        spec.cli = "other"

    assert raised.value.errors()[0]["type"] == "frozen_instance"
    assert raised.value.errors()[0]["loc"] == ("cli",)


def test_environment_spec_rejects_attribute_reassignment() -> None:
    environment = get_backend("oatk").environment

    with pytest.raises(ValidationError) as raised:
        environment.contract_locator = "organelleverse.assembly.environment_specs:other"

    assert raised.value.errors()[0]["type"] == "frozen_instance"
    assert raised.value.errors()[0]["loc"] == ("contract_locator",)


def test_oatk_is_hifi_only_and_uses_official_source() -> None:
    oatk = get_backend("oatk")
    assert oatk.profiles == (AssemblyProfile.PACBIO_HIFI,)
    assert oatk.official_url == "https://github.com/c-zhou/oatk"
    assert oatk.cli == "oatk"
    # hmm_profiles is not a hard routing requirement: the Oatk HiFi execution
    # slice provisions the managed OatkDB profile automatically. Presence of
    # AssemblyAuxiliary.hmm_profiles overrides the managed default instead of
    # gating whether the route is even reachable.
    assert oatk.required_auxiliary == ()


def test_pmat_does_not_unconditionally_require_genome_size() -> None:
    """PMAT genome size is carried as traceable evidence (GenomeSizeEvidence),
    not gated as an unconditional routing requirement. Only the raw CLR/ONT
    correction route remains a conditional auxiliary, enforced in routing rather
    than in the static backend spec."""
    pmat = get_backend("pmat")
    assert AuxiliaryRole.GENOME_SIZE not in pmat.required_auxiliary
    assert pmat.required_auxiliary == ()
    assert RUNTIMES.require("pmat").backend_id == "pmat"


def test_pmat_static_facts_match_pmat2_v215_contract() -> None:
    """PMAT2 v2.1.5: corrected repository, Conda carrier plus managed source,
    and the full long-read profile matrix (HiFi + raw/corrected CLR +
    raw/corrected/HQ/duplex ONT)."""
    pmat = get_backend("pmat")
    assert pmat.official_url == "https://github.com/aiPGAB/PMAT2"
    assert pmat.environment.carrier == EnvironmentCarrier.CONDA
    assert pmat.install.tier == "source"
    assert pmat.install.source == "https://github.com/aiPGAB/PMAT2"
    assert AssemblyProfile.ONT_HQ in pmat.profiles
    assert AssemblyProfile.ONT_DUPLEX in pmat.profiles


@pytest.mark.parametrize(
    ("backend", "organelles"),
    [
        ("oatk", ("mitochondrion", "plastid")),
        ("himt", ("mitochondrion", "plastid")),
        ("getorganelle", ("mitochondrion", "plastid")),
        ("pmat", ("mitochondrion", "plastid")),
    ],
)
def test_backend_organelle_matrix(backend: str, organelles: tuple[str, ...]) -> None:
    assert get_backend(backend).organelles == organelles


def test_himt_cli_and_source_facts() -> None:
    assert get_backend("himt").cli == "himt"
    assert get_backend("himt").official_url == "https://github.com/tang-shuyuan/HiMT"


def test_himt_has_corrected_install_locator_and_duplex_profile() -> None:
    himt = BACKENDS["himt"]
    assert himt.install.conda == "shuyuan_tang::himt=1.1.3=0"
    assert AssemblyProfile.ONT_DUPLEX in himt.profiles


@pytest.mark.parametrize("backend", EXPECTED_PROFILES)
def test_backend_profiles_are_exact(backend: str) -> None:
    assert (
        tuple(profile.value for profile in get_backend(backend).profiles)
        == EXPECTED_PROFILES[backend]
    )


def test_get_backend_unknown_backend_raises_exact_contract_code() -> None:
    with pytest.raises(OrganelleContractError) as raised:
        get_backend("unknown")

    assert raised.value.code == "contract.unknown_assembly_backend"


def test_release_runtime_registry_contains_oatk() -> None:
    runtime = RUNTIMES.require("oatk")
    assert runtime.backend_id == "oatk"
    assert runtime.environment_spec.backend_id == "oatk"
    assert runtime.adapter_factory().backend_id == "oatk"


def test_runtime_registry_rejects_duplicate_backend_ids() -> None:
    runtime = RUNTIMES.require("oatk")
    with pytest.raises(OrganelleContractError) as raised:
        AssemblyRuntimeRegistry((runtime, runtime))
    assert raised.value.code == "contract.duplicate_assembly_backend_runtime"


def test_release_runtime_registry_contains_himt() -> None:
    runtime = RUNTIMES.require("himt")
    assert runtime.backend_id == "himt"
    assert runtime.environment_spec.backend_id == "himt"
    assert runtime.adapter_factory().backend_id == "himt"


def test_release_runtime_registry_contains_pmat() -> None:
    runtime = RUNTIMES.require("pmat")
    assert runtime.backend_id == "pmat"
    assert runtime.environment_spec.backend_id == "pmat"
    assert runtime.adapter_factory().backend_id == "pmat"
