"""I2: the field split must not change any input-side behavior.

The expected values below were recorded from the pre-split catalog. They are
literals on purpose: reading them from the specs under test would make the
test vacuous.
"""

from typing import cast

import pytest

from organelleverse import operations as op
from organelleverse.core.data import OrganelleData
from organelleverse.core.errors import OrganelleInputError
from organelleverse.core.frozen import FrozenJson, FrozenMap


def _mapping(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return cast(dict[str, object], value)


EXPECTED_INPUT_MODALITIES = {
    "annotation.annotate": (),
    "annotation.extract": (),
    "annotation.write": (),
    "assembly.assemble": ("sequencing_reads",),
    "assembly.pmat_graph_build": ("pmat_graph_input",),
    "assembly.write": (),
    "fetch.accession_list": (),
    "fetch.dedup_genomes": ("organelle_records",),
    "fetch.entrez_query": (),
    "fetch.gene_records": (),
    "fetch.genome_size_candidates": (),
    "fetch.ngdc_gwh": (),
    "fetch.nuclear_assembly": (),
    "fetch.refseq_snapshot": (),
    "qc.annotation": (),
    "qc.assembly": (),
    "qc.write": (),
}


@pytest.mark.parametrize("operation_id", sorted(EXPECTED_INPUT_MODALITIES))
def test_accepted_input_modalities_are_unchanged(operation_id: str) -> None:
    spec = op.describe(operation_id)
    assert spec.input_modalities == EXPECTED_INPUT_MODALITIES[operation_id]


def test_unsupported_modality_error_shape_is_unchanged() -> None:
    wrong = OrganelleData(
        modality="organelle_records",
        payload=cast("FrozenMap[FrozenJson]", {"records": []}),
    )
    with pytest.raises(OrganelleInputError) as excinfo:
        op.invoke("assembly.assemble", input=wrong, parameters={})
    error = excinfo.value
    assert error.code == "input.unsupported_operation_modality"

    # Assert through as_dict(), which is the Agent-facing serialization: it thaws
    # the frozen details back into builtin JSON containers, so accepted_modalities
    # is a list here while error.details holds the frozen tuple. Both shapes
    # predate this change; as_dict() is the one an agent actually observes.
    details = _mapping(error.as_dict()["details"])
    assert details["operation_id"] == "assembly.assemble"
    assert details["provided_modality"] == "organelle_records"
    assert details["accepted_modalities"] == ["sequencing_reads"]
