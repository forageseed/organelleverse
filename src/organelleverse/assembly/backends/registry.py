from collections.abc import Iterator, Mapping, Sequence
from types import MappingProxyType
from typing import Literal, TypeAlias, cast

from organelleverse.assembly.contracts import assembly_method_ids
from organelleverse.core.errors import OrganelleContractError

from .spec import (
    AssemblyBackendSpec,
    AssemblyEnvironmentRef,
    AssemblyProfile,
    AuxiliaryRole,
    BackendInstallSpec,
    EnvironmentCarrier,
    OrganelleType,
)


class AssemblyBackendRegistry(Mapping[str, AssemblyBackendSpec]):
    def __init__(self, specs: Sequence[AssemblyBackendSpec]) -> None:
        validated: dict[str, AssemblyBackendSpec] = {}
        for candidate in specs:
            spec = AssemblyBackendSpec.model_validate(candidate)
            if spec.backend_id in validated:
                raise OrganelleContractError(
                    code="contract.duplicate_assembly_backend",
                    message=f"duplicate assembly backend: {spec.backend_id}",
                )
            validated[spec.backend_id] = spec
        self._specs: Mapping[str, AssemblyBackendSpec] = MappingProxyType(validated)

    def __getitem__(self, backend_id: str) -> AssemblyBackendSpec:
        try:
            return self._specs[backend_id]
        except KeyError as error:
            raise OrganelleContractError(
                code="contract.unknown_assembly_backend",
                message=f"unknown assembly backend: {backend_id}",
            ) from error

    def __iter__(self) -> Iterator[str]:
        return iter(self._specs)

    def __len__(self) -> int:
        return len(self._specs)

    def ids(self) -> tuple[str, ...]:
        return tuple(self._specs)


_INSTALL = {
    "novoplasty": BackendInstallSpec(
        tier="conda",
        conda="bioconda::novoplasty=4.3.5",
        source="https://github.com/ndierckx/NOVOPlasty",
        note="Pinned NOVOPlasty 4.3.5 managed Conda environment; Illumina paired reads",
    ),
    "oatk": BackendInstallSpec(
        tier="conda",
        conda="bioconda::oatk",
        source="https://github.com/c-zhou/oatk",
        note="Requires pinned OatkDB HMM profiles",
    ),
    "himt": BackendInstallSpec(
        tier="conda",
        conda="shuyuan_tang::himt=1.1.3=0",
        source="https://github.com/tang-shuyuan/HiMT",
        note="Raw and corrected long reads use different declared modes",
    ),
    "getorganelle": BackendInstallSpec(
        tier="pip",
        pip="getorganelle",
        conda="bioconda::getorganelle",
        note="Requires the pinned seed/label database in released execution",
    ),
    "pmat": BackendInstallSpec(
        tier="source",
        source="https://github.com/aiPGAB/PMAT2",
        note="Raw CLR/ONT also requires a pinned correction route",
    ),
    "tippo": BackendInstallSpec(
        tier="source",
        source="https://github.com/Wenfei-Xian/TIPP",
        note="Requires an existing verified TIPPo v2.4 environment; no managed lock is bundled",
    ),
    "ptgaul": BackendInstallSpec(
        tier="source",
        source="https://github.com/Bean061/ptgaul",
        note="Requires an existing ptGAUL environment and its CLI dependencies; no tools are installed automatically",
    ),
    "ovasm": BackendInstallSpec(
        tier="source",
        source="https://github.com/forageseed/ovasm",
        note="Native binary: build it from https://github.com/forageseed/ovasm (`cargo build --release`), then put it on PATH or set ORG_VERSE_OVASM_BIN; no managed environment",
    ),
}

_BACKEND_FACTS = (
    (
        "oatk",
        "Oatk",
        ("mitochondrion", "plastid"),
        ("pacbio_hifi",),
        (),
        (),
        (),
        "oatk",
        "conda",
        "https://github.com/c-zhou/oatk",
        ("plant-hifi",),
    ),
    (
        "himt",
        "HiMT",
        ("mitochondrion", "plastid"),
        (
            "pacbio_hifi",
            "pacbio_clr_raw",
            "pacbio_clr_corrected",
            "ont_raw",
            "ont_corrected",
            "ont_hq",
            "ont_duplex",
        ),
        (),
        (),
        (),
        "himt",
        "conda",
        "https://github.com/tang-shuyuan/HiMT",
        (
            "plant-hifi",
            "plant-clr-raw",
            "plant-clr-corrected",
            "plant-ont-raw",
            "plant-ont-hq",
            "plant-ont-duplex",
        ),
    ),
    (
        "getorganelle",
        "GetOrganelle",
        ("mitochondrion", "plastid"),
        ("illumina_pe", "illumina_se"),
        (),
        (),
        (),
        "get_organelle_from_reads.py",
        "conda",
        "https://github.com/Kinggerm/GetOrganelle",
        ("plastid-illumina-pe", "plastid-illumina-se", "mito-illumina-pe"),
    ),
    (
        "pmat",
        "PMAT",
        ("mitochondrion", "plastid"),
        (
            "pacbio_hifi",
            "pacbio_clr_raw",
            "pacbio_clr_corrected",
            "ont_raw",
            "ont_corrected",
            "ont_hq",
            "ont_duplex",
        ),
        (),
        (),
        (),
        "PMAT",
        "conda",
        "https://github.com/aiPGAB/PMAT2",
        ("plant-hifi", "plant-clr-raw", "plant-clr-corrected", "plant-ont-raw"),
    ),
    (
        "tippo",
        "TIPPo",
        ("plastid",),
        ("pacbio_hifi",),
        (),
        (),
        (),
        "TIPPo.v2.4.pl",
        "conda",
        "https://github.com/Wenfei-Xian/TIPP",
        ("tippo-plastid-hifi",),
    ),
    (
        "ptgaul",
        "ptGAUL",
        ("plastid",),
        ("ont_raw",),
        (),
        ("reference_fasta",),
        (),
        "ptGAUL.sh",
        "conda",
        "https://github.com/Bean061/ptgaul",
        ("ptgaul-plastid-ont-raw",),
    ),
    (
        "novoplasty",
        "NOVOPlasty",
        ("mitochondrion", "plastid"),
        ("illumina_pe",),
        (),
        (),
        ("insert_size",),
        "NOVOPlasty4.3.5.pl",
        "conda",
        "https://github.com/ndierckx/NOVOPlasty",
        ("novoplasty-plastid-pe",),
    ),
    (
        "ovasm",
        "ovasm",
        ("mitochondrion", "plastid"),
        (
            "pacbio_hifi",
            "pacbio_clr_raw",
            "ont_raw",
            "illumina_se",
            "illumina_pe",
            "ont_raw_illumina",
            "pacbio_clr_raw_illumina",
        ),
        (),
        (),
        (),
        "ovasm",
        "native",
        "https://github.com/forageseed/ovasm",
        (
            "ovasm-hifi",
            "ovasm-ont-raw",
            "ovasm-clr-raw",
            "ovasm-illumina",
            "ovasm-illumina-pe",
            "ovasm-ont-illumina",
            "ovasm-clr-illumina",
        ),
    ),
)

_ADAPTER_CLASSES = {
    "novoplasty": "NovoplastyAdapter",
    "oatk": "OatkAdapter",
    "himt": "HiMTAdapter",
    "getorganelle": "GetOrganelleAdapter",
    "pmat": "PmatAdapter",
    "tippo": "TippoAdapter",
    "ptgaul": "PtgaulAdapter",
    "ovasm": "OvasmAdapter",
}

BackendFact: TypeAlias = tuple[
    str,
    str,
    tuple[OrganelleType, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[Literal["read_length", "insert_size"], ...],
    str,
    str,
    str,
    tuple[str, ...],
]

specs: list[AssemblyBackendSpec] = []
for fact in cast(tuple[BackendFact, ...], _BACKEND_FACTS):
    (
        backend_id,
        display_name,
        organelles,
        profile_values,
        required_values,
        mitochondrial_values,
        required_short_fields,
        cli,
        carrier_value,
        official_url,
        fixture_ids,
    ) = fact
    adapter_locator = (
        f"organelleverse.assembly.backends.{backend_id}:{_ADAPTER_CLASSES[backend_id]}"
    )
    output_contract = f"organelleverse.assembly-output.{backend_id}.v1"
    environment = AssemblyEnvironmentRef(
        carrier=EnvironmentCarrier(carrier_value),
        contract_locator=f"organelleverse.assembly.environment_specs:{backend_id}",
    )
    specs.append(
        AssemblyBackendSpec(
            backend_id=backend_id,
            display_name=display_name,
            organelles=organelles,
            profiles=tuple(AssemblyProfile(value) for value in profile_values),
            required_auxiliary=tuple(AuxiliaryRole(value) for value in required_values),
            mitochondrial_auxiliary=tuple(AuxiliaryRole(value) for value in mitochondrial_values),
            required_short_fields=required_short_fields,
            cli=cli,
            adapter_locator=adapter_locator,
            output_contract=output_contract,
            environment=environment,
            official_url=official_url,
            fixture_ids=fixture_ids,
            install=_INSTALL[backend_id],
        )
    )

BACKENDS = AssemblyBackendRegistry(tuple(specs))

if BACKENDS.ids() != assembly_method_ids():
    raise OrganelleContractError(
        code="contract.assembly_backend_registry_mismatch",
        message="default assembly backend registry does not match AssemblyMethod",
        details={
            "expected": list(assembly_method_ids()),
            "actual": list(BACKENDS.ids()),
        },
    )


def backend_ids() -> tuple[str, ...]:
    return BACKENDS.ids()


def get_backend(backend_id: str) -> AssemblyBackendSpec:
    return BACKENDS[backend_id]


def install_info() -> dict[str, dict[str, str]]:
    return {
        spec.backend_id: {
            "url": spec.official_url,
            "cli": spec.cli,
            "tier": spec.install.tier,
            "note": spec.install.note,
            **({"pip": spec.install.pip} if spec.install.pip is not None else {}),
            **({"conda": spec.install.conda} if spec.install.conda is not None else {}),
            **({"source": spec.install.source} if spec.install.source is not None else {}),
        }
        for spec in BACKENDS.values()
    }
