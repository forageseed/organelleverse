"""Exact admitted search strategy binding registry."""

from __future__ import annotations

from organelleverse.capabilities.index import CapabilityEntry, CapabilityStatus
from organelleverse.core.errors import OrganelleContractError

from .profiles_v3 import identity_from_capability_entry
from .strategy_models import SearchStrategy, StrategyBinding


class StrategyRegistry:
    def __init__(self) -> None:
        self._strategies: dict[str, tuple[StrategyBinding, SearchStrategy]] = {}

    def register(
        self,
        binding: StrategyBinding,
        strategy: SearchStrategy,
        *,
        entry: CapabilityEntry,
    ) -> None:
        if entry.status is not CapabilityStatus.ADMITTED:
            raise _error(
                "optimization.strategy_not_admitted",
                "search strategy registration requires admission",
                binding,
            )
        if identity_from_capability_entry(entry) != binding.identity:
            raise _error(
                "optimization.strategy_binding_mismatch",
                "strategy registration identity does not match its admitted capability",
                binding,
            )
        existing = self._strategies.get(binding.kind)
        if existing is not None and existing[0] != binding:
            raise _error(
                "optimization.strategy_binding_mismatch",
                "strategy kind is already registered to another admitted identity",
                binding,
            )
        self._strategies[binding.kind] = (binding, strategy)

    def resolve(self, binding: StrategyBinding) -> SearchStrategy:
        registered = self._strategies.get(binding.kind)
        if registered is None or registered[0] != binding:
            raise _error(
                "optimization.strategy_binding_mismatch",
                "search strategy binding does not exactly match an admitted registration",
                binding,
            )
        return registered[1]


def _error(code: str, message: str, binding: StrategyBinding) -> OrganelleContractError:
    return OrganelleContractError(
        code=code,
        message=message,
        details={
            "strategy_kind": binding.kind,
            "strategy_capability_id": binding.identity.capability_id,
        },
    )


__all__ = ["StrategyRegistry"]
