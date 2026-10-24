"""Which public Python function runs a registered operation.

For almost every operation the answer is its id: ``ov.<suite>.<name>`` runs
``<suite>.<name>`` and takes the same parameters. A few registered callables are
JSON-flat wrappers (``fetch._agent_ops``, ``morphology.batch_measure_dir``)
while the Python API keeps the function people already call under its own name.
Those are listed in :data:`PUBLIC_FUNCTIONS`; they are not second spellings —
OrganelleVerse keeps one name per function (see ``ov.save``) — so a caller holding
an operation id looks the function up here instead of guessing.

The rule is pinned by ``tests/operations/test_public_functions.py``: a resolved
function accepts every parameter of its operation's schema. Operations that
resolve to nothing are reachable only through the registry; the reasons are in
``docs/operations/public-python-functions.md``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any

__all__ = ["PUBLIC_FUNCTIONS", "public_function", "resolve_public_function"]

#: Operation id -> ``"<suite>.<function>"`` where the public name differs from the id.
PUBLIC_FUNCTIONS: Mapping[str, str] = MappingProxyType(
    {
        "fetch.accession_list": "fetch.fetch_accessions",
        "fetch.gene_records": "fetch.fetch_genes",
        "fetch.ngdc_gwh": "fetch.fetch_ngdc",
        "fetch.nuclear_assembly": "fetch.fetch_nuclear_genome",
        "morphology.batch_measure": "morphology.batch_measure_dir",
        "rna_editing.quantify_efficiency": "rna_editing.quantify_known_site_efficiency",
    }
)


def resolve_public_function(operation_id: str) -> Callable[..., Any] | None:
    """The public ``ov.<suite>.<function>`` that runs ``operation_id``, or ``None``."""
    import organelleverse

    suite, _, name = PUBLIC_FUNCTIONS.get(operation_id, operation_id).partition(".")
    if suite not in organelleverse._PUBLIC_MODULES:
        return None
    function = getattr(getattr(organelleverse, suite, None), name, None)
    return function if callable(function) else None


def public_function(operation_id: str) -> str | None:
    """``"<suite>.<function>"`` (under ``ov.``) that runs ``operation_id``, or ``None``."""
    if resolve_public_function(operation_id) is None:
        return None
    return PUBLIC_FUNCTIONS.get(operation_id, operation_id)
