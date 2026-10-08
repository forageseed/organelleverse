"""Immutable released backend runtime registry.

A runtime binds together the static environment contract, input validation,
resource preparation, and adapter factory for one released assembly backend.
The service resolves a runtime from the selected route and executes it without
any backend-name conditionals.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from organelleverse.assembly.contracts import assembly_method_ids
from organelleverse.assembly.environment_specs import AssemblyEnvironmentSpec
from organelleverse.core.data import OrganelleData
from organelleverse.core.errors import OrganelleContractError

if TYPE_CHECKING:
    from organelleverse.assembly.backends.base import (
        AssemblyAdapter,
        ExpectedBackendResources,
        PreparedBackendResources,
    )
    from organelleverse.assembly.contracts import (
        AssemblyInputPayload,
        AssemblyMethod,
        AssemblyRequest,
    )
    from organelleverse.assembly.environments import (
        EnvironmentManager,
        PreparedEnvironment,
    )


class AssemblyResourceProvider(Protocol):
    """Prepare backend-specific resources and expose their expected identity."""

    def expected(
        self,
        manager: EnvironmentManager,
        environment_spec: AssemblyEnvironmentSpec,
        request: AssemblyRequest,
        payload: AssemblyInputPayload,
    ) -> ExpectedBackendResources: ...

    def prepare(
        self,
        manager: EnvironmentManager,
        environment_spec: AssemblyEnvironmentSpec,
        request: AssemblyRequest,
        payload: AssemblyInputPayload,
        environment: PreparedEnvironment,
    ) -> PreparedBackendResources: ...


@dataclass(frozen=True)
class BackendRuntime:
    """One released backend binding."""

    backend_id: AssemblyMethod
    environment_spec: AssemblyEnvironmentSpec
    data_validator: Callable[[OrganelleData], AssemblyInputPayload]
    resource_provider: AssemblyResourceProvider
    adapter_factory: Callable[[], AssemblyAdapter]
    #: Native backends bring their own environment manager (the binary and its identity);
    #: the service then neither resolves providers nor prepares a conda prefix.
    environment_manager_factory: Callable[[], EnvironmentManager] | None = None


class AssemblyRuntimeRegistry(Mapping[str, BackendRuntime]):
    """Immutable registry of released backend runtimes."""

    def __init__(self, runtimes: Iterable[BackendRuntime]) -> None:
        seen: set[str] = set()
        by_id: dict[str, BackendRuntime] = {}
        for runtime in runtimes:
            backend_id = runtime.backend_id
            if backend_id in seen:
                raise OrganelleContractError(
                    code="contract.duplicate_assembly_backend_runtime",
                    message=f"duplicate runtime for backend {backend_id!r}",
                    details={"backend_id": backend_id},
                )
            self._validate_runtime(runtime)
            seen.add(backend_id)
            by_id[backend_id] = runtime
        self._runtimes = by_id

    @staticmethod
    def _validate_runtime(runtime: BackendRuntime) -> None:
        backend_id = runtime.backend_id
        if runtime.environment_spec.backend_id != backend_id:
            raise OrganelleContractError(
                code="contract.assembly_backend_runtime_mismatch",
                message="runtime environment backend does not match runtime backend",
                details={
                    "backend_id": backend_id,
                    "environment_backend": runtime.environment_spec.backend_id,
                },
            )
        adapter = runtime.adapter_factory()
        if adapter.backend_id != backend_id:
            raise OrganelleContractError(
                code="contract.assembly_backend_runtime_mismatch",
                message="runtime adapter backend does not match runtime backend",
                details={
                    "backend_id": backend_id,
                    "adapter_backend": adapter.backend_id,
                },
            )

    def __getitem__(self, backend_id: str) -> BackendRuntime:
        return self._runtimes[backend_id]

    def __iter__(self):
        return iter(self._runtimes)

    def __len__(self) -> int:
        return len(self._runtimes)

    def require(self, backend_id: str) -> BackendRuntime:
        if backend_id not in self._runtimes:
            raise OrganelleContractError(
                code="contract.assembly_backend_not_released",
                message=f"assembly backend {backend_id!r} is not released",
                details={"backend_id": backend_id},
            )
        return self._runtimes[backend_id]


# Build the released runtime registry.  Keep this at the bottom of the module so
# the classes above are defined before any construction-time adapter_factory()
# calls are evaluated.
def _build_runtimes() -> AssemblyRuntimeRegistry:
    from organelleverse.assembly.backends.getorganelle import (
        GetOrganelleAdapter,
        GetOrganelleResourceProvider,
    )
    from organelleverse.assembly.backends.himt import (
        EmptyAssemblyResourceProvider,
        HimtAdapter,
    )
    from organelleverse.assembly.backends.novoplasty import NovoplastyAdapter
    from organelleverse.assembly.backends.oatk import OatkAdapter, OatkResourceProvider
    from organelleverse.assembly.backends.ovasm import NativeOvasmEnvironment, OvasmAdapter
    from organelleverse.assembly.backends.pmat import PmatAdapter
    from organelleverse.assembly.backends.ptgaul import PtgaulAdapter
    from organelleverse.assembly.backends.tippo import TippoAdapter
    from organelleverse.assembly.data_contract import (
        validate_getorganelle_assembly_data,
        validate_himt_assembly_data,
        validate_novoplasty_assembly_data,
        validate_oatk_assembly_data,
        validate_ovasm_assembly_data,
        validate_pmat_assembly_data,
        validate_ptgaul_assembly_data,
        validate_tippo_assembly_data,
    )
    from organelleverse.assembly.environment_specs import (
        GETORGANELLE_ENVIRONMENT,
        HIMT_ENVIRONMENT,
        NOVOPLASTY_ENVIRONMENT,
        OATK_ENVIRONMENT,
        OVASM_ENVIRONMENT,
        PMAT_ENVIRONMENT,
        PTGAUL_ENVIRONMENT,
        TIPPO_ENVIRONMENT,
    )

    return AssemblyRuntimeRegistry(
        (
            BackendRuntime(
                backend_id="oatk",
                environment_spec=OATK_ENVIRONMENT,
                data_validator=validate_oatk_assembly_data,
                resource_provider=OatkResourceProvider(),
                adapter_factory=OatkAdapter,
            ),
            BackendRuntime(
                backend_id="himt",
                environment_spec=HIMT_ENVIRONMENT,
                data_validator=validate_himt_assembly_data,
                resource_provider=EmptyAssemblyResourceProvider(),
                adapter_factory=HimtAdapter,
            ),
            BackendRuntime(
                backend_id="getorganelle",
                environment_spec=GETORGANELLE_ENVIRONMENT,
                data_validator=validate_getorganelle_assembly_data,
                resource_provider=GetOrganelleResourceProvider(),
                adapter_factory=GetOrganelleAdapter,
            ),
            BackendRuntime(
                backend_id="pmat",
                environment_spec=PMAT_ENVIRONMENT,
                data_validator=validate_pmat_assembly_data,
                resource_provider=EmptyAssemblyResourceProvider(),
                adapter_factory=PmatAdapter,
            ),
            BackendRuntime(
                backend_id="tippo",
                environment_spec=TIPPO_ENVIRONMENT,
                data_validator=validate_tippo_assembly_data,
                resource_provider=EmptyAssemblyResourceProvider(),
                adapter_factory=TippoAdapter,
            ),
            BackendRuntime(
                backend_id="ptgaul",
                environment_spec=PTGAUL_ENVIRONMENT,
                data_validator=validate_ptgaul_assembly_data,
                resource_provider=EmptyAssemblyResourceProvider(),
                adapter_factory=PtgaulAdapter,
            ),
            BackendRuntime(
                backend_id="novoplasty",
                environment_spec=NOVOPLASTY_ENVIRONMENT,
                data_validator=validate_novoplasty_assembly_data,
                resource_provider=EmptyAssemblyResourceProvider(),
                adapter_factory=NovoplastyAdapter,
            ),
            BackendRuntime(
                backend_id="ovasm",
                environment_spec=OVASM_ENVIRONMENT,
                data_validator=validate_ovasm_assembly_data,
                resource_provider=EmptyAssemblyResourceProvider(),
                adapter_factory=OvasmAdapter,
                environment_manager_factory=NativeOvasmEnvironment,
            ),
        )
    )


RUNTIMES = _build_runtimes()

if tuple(RUNTIMES) != assembly_method_ids():
    raise OrganelleContractError(
        code="contract.assembly_backend_runtime_registry_mismatch",
        message="default assembly backend runtime registry does not match AssemblyMethod",
        details={
            "expected": list(assembly_method_ids()),
            "actual": list(RUNTIMES),
        },
    )


def pmat_runtime() -> BackendRuntime:
    """Return the released PMAT2 runtime."""
    return RUNTIMES.require("pmat")
