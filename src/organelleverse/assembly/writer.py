"""Released assembly writer: promote a managed assembly Result to a destination.

``assembly.write`` is the explicit typed materializer for a completed assembly
Result. It performs no scientific work. It moves one managed run tree to a
caller-selected destination through :func:`organelleverse.runtime.promote_result`,
which verifies every declared artifact hash, leaves a single complete artifact
tree, and replaces the former managed run path with a safe symbolic link. The
returned Result is the preferred published handle: its artifact references point
directly at the destination while the pre-write references keep resolving
through the link.
"""

from __future__ import annotations

from pathlib import Path

from organelleverse.core.errors import OrganelleInputError
from organelleverse.core.result import OrganelleResult
from organelleverse.runtime import promote_result

# The released artifact-producing assembly operations whose managed run trees
# ``assembly.write`` may promote. Failed runs are valid inputs too: their
# retained logs and manifests are the evidence a user needs to diagnose them.
_ASSEMBLY_SOURCE_OPERATIONS: frozenset[str] = frozenset(
    {"assembly.assemble", "assembly.pmat_graph_build"}
)


def materialize_result(result: OrganelleResult, output: str | Path) -> OrganelleResult:
    """Promote one managed assembly Result to a user-selected destination."""

    if result.operation_id not in _ASSEMBLY_SOURCE_OPERATIONS:
        raise OrganelleInputError(
            code="assembly.write_input_contract",
            message="assembly.write accepts only a released assembly Result",
            details={"operation_id": result.operation_id},
        )
    return promote_result(result, output)


__all__ = ["materialize_result"]
