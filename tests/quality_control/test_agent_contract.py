from __future__ import annotations

import organelleverse as ov


def test_qc_operations_are_result_to_result_consume_specs() -> None:
    from organelleverse.quality_control.operations import (
        ANNOTATION_QC_SPEC,
        ASSEMBLY_QC_SPEC,
        QC_WRITE_SPEC,
    )

    assert ANNOTATION_QC_SPEC.stage.value == "consume"
    assert ANNOTATION_QC_SPEC.input_kind.value == "result"
    assert ASSEMBLY_QC_SPEC.stage.value == "consume"
    assert ASSEMBLY_QC_SPEC.input_kind.value == "result"
    assert QC_WRITE_SPEC.input_kind.value == "result"
    assert ANNOTATION_QC_SPEC.fallback.allowed is False
    assert ASSEMBLY_QC_SPEC.fallback.allowed is False


def test_qc_assembly_parameter_schema_is_closed() -> None:
    from organelleverse.operations import OperationRegistry
    from organelleverse.quality_control.api import assembly
    from organelleverse.quality_control.operations import ASSEMBLY_QC_SPEC

    registry = OperationRegistry()
    registry.register(ASSEMBLY_QC_SPEC, assembly)
    schema = registry.parameter_schema("qc.assembly")
    assert schema["additionalProperties"] is False
    assert set(schema["properties"]) == {"threads", "timeout_seconds"}


def test_qc_operations_are_in_release_catalog_once() -> None:
    operation_ids = {spec.operation_id for spec in ov.operations.list()}
    assert "qc.annotation" in operation_ids
    assert "qc.assembly" in operation_ids
    assert "qc.write" in operation_ids
    assert len(operation_ids) == len(ov.operations.list())


def test_qc_write_parameter_schema_is_closed() -> None:
    schema = ov.operations.parameter_schema("qc.write")
    assert schema["additionalProperties"] is False
    assert set(schema["properties"]) == {"output"}
    assert schema["required"] == ["output"]
