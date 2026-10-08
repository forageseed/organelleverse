from organelleverse.operations import CoreKind, ExecutionMode, OperationSpec, OperationStage

from .reachability import compute_reachability


def _spec(
    operation_id: str,
    stage: OperationStage,
    input_kind: CoreKind,
    output_kind: CoreKind,
    *,
    input_modalities: tuple[str, ...] = (),
    output_modalities: tuple[str, ...] = (),
) -> OperationSpec:
    return OperationSpec(
        operation_id=operation_id,
        contract_version="1.0",
        title="Demo reachability probe",
        description="Test fixture spec used only to probe reachability computation.",
        keywords=("demo", "fixture", "test"),
        execution_mode=ExecutionMode.INLINE,
        stage=stage,
        input_kind=input_kind,
        output_kind=output_kind,
        input_modalities=input_modalities,
        output_modalities=output_modalities,
        callable_locator="demo.api:op",
    )


def test_mutually_feeding_transforms_without_a_root_are_unreachable() -> None:
    """Set membership over declared outputs would call both reachable."""
    t1 = _spec(
        "demo.t1",
        OperationStage.TRANSFORM,
        CoreKind.DATA,
        CoreKind.DATA,
        input_modalities=("m2",),
        output_modalities=("m1",),
    )
    t2 = _spec(
        "demo.t2",
        OperationStage.TRANSFORM,
        CoreKind.DATA,
        CoreKind.DATA,
        input_modalities=("m1",),
        output_modalities=("m2",),
    )
    report = compute_reachability((t1, t2), contract_modalities=frozenset({"m1", "m2"}))
    assert report.invocable == ()
    assert {entry.operation_id for entry in report.unreachable} == {"demo.t1", "demo.t2"}


def test_a_genome_to_genome_transform_does_not_bootstrap_genome() -> None:
    transform = _spec("demo.g", OperationStage.TRANSFORM, CoreKind.GENOME, CoreKind.GENOME)
    analyze = _spec("demo.a", OperationStage.ANALYZE, CoreKind.GENOME, CoreKind.RESULT)
    report = compute_reachability((transform, analyze), contract_modalities=frozenset())
    assert report.invocable == ()
    assert "genome" not in report.producible_kinds


def test_adding_a_root_makes_the_whole_chain_invocable() -> None:
    root = _spec(
        "demo.root",
        OperationStage.READ,
        CoreKind.NONE,
        CoreKind.DATA,
        output_modalities=("m1",),
    )
    analyze = _spec(
        "demo.a",
        OperationStage.ANALYZE,
        CoreKind.DATA,
        CoreKind.RESULT,
        input_modalities=("m1",),
    )
    consume = _spec("demo.c", OperationStage.CONSUME, CoreKind.RESULT, CoreKind.RESULT)
    report = compute_reachability((root, analyze, consume), contract_modalities=frozenset({"m1"}))
    assert set(report.invocable) == {"demo.root", "demo.a", "demo.c"}
    assert report.unreachable == ()


def test_partial_satisfaction_is_not_reported_as_blocked() -> None:
    root = _spec(
        "demo.root",
        OperationStage.READ,
        CoreKind.NONE,
        CoreKind.DATA,
        output_modalities=("m1",),
    )
    analyze = _spec(
        "demo.a",
        OperationStage.ANALYZE,
        CoreKind.DATA,
        CoreKind.RESULT,
        input_modalities=("m1", "m_missing"),
    )
    report = compute_reachability((root, analyze), contract_modalities=frozenset({"m1"}))
    assert "demo.a" in report.invocable
    assert ("demo.a", "m_missing") in report.unsatisfiable_declarations


def test_producer_without_a_contract_yields_reachable_unvalidated() -> None:
    root = _spec(
        "demo.root",
        OperationStage.READ,
        CoreKind.NONE,
        CoreKind.DATA,
        output_modalities=("m_nocontract",),
    )
    analyze = _spec(
        "demo.a",
        OperationStage.ANALYZE,
        CoreKind.DATA,
        CoreKind.RESULT,
        input_modalities=("m_nocontract",),
    )
    report = compute_reachability((root, analyze), contract_modalities=frozenset())
    assert "m_nocontract" in report.declared_only_modalities
    assert "demo.a" in report.reachable_unvalidated


def test_released_catalog_matches_the_recorded_state() -> None:
    from organelleverse import operations as op
    from organelleverse.operations.data_contracts import BUILTIN_DATA_CONTRACTS

    specs = op.list()
    assert len(specs) == 20
    contract_modalities = frozenset(
        {contract.modality for contract in BUILTIN_DATA_CONTRACTS}
        | {"sequencing_reads", "pmat_graph_input"}
    )
    report = compute_reachability(specs, contract_modalities=contract_modalities)
    assert len(report.invocable) == 19
    assert len(report.unreachable) == 1
    # io.read_long_reads produces sequencing_reads, so assembly.assemble becomes
    # invocable and result becomes producible - the whole consume tier follows.
    assert report.producible_kinds == ("data", "genome", "result")
    assert set(report.declared_only_modalities) == {"gene_records", "genome_size_candidates"}


def test_scan_accepts_literals_and_module_constants() -> None:
    from pathlib import Path

    from .reachability import scan_suggestions

    fixture = Path(__file__).parent / "fixtures" / "suggestion_samples.py"
    findings = scan_suggestions(fixture)
    ids = {finding.operation_id for finding in findings if finding.operation_id}
    assert "annotation.write" in ids
    assert "io.does_not_exist" in ids
    assert any(finding.non_literal for finding in findings)


def test_released_source_has_no_non_literal_suggestion_ids() -> None:
    from pathlib import Path

    from .reachability import scan_suggestions

    src = Path(__file__).resolve().parents[2] / "src" / "organelleverse"
    assert [finding for finding in scan_suggestions(src) if finding.non_literal] == []


def test_baseline_matches_the_computation_in_both_directions() -> None:
    from .reachability import load_baseline, released_inventory

    computed = released_inventory()
    baseline = load_baseline()
    assert computed["A"] == baseline["A"], "class A holes drifted from the baseline"
    assert computed["B"] == baseline["B"], "class B holes drifted from the baseline"
    assert computed["C"] == baseline["C"], "class C holes drifted from the baseline"
    assert computed["D"] == baseline["D"], "class D holes drifted from the baseline"


def test_class_c_is_empty_today() -> None:
    from .reachability import released_inventory

    assert released_inventory()["C"] == []


def test_every_baseline_entry_carries_intent() -> None:
    from .reachability import load_baseline_entries

    for entry in load_baseline_entries():
        assert entry["first_recorded"], f"{entry} lacks first_recorded"
        if entry.get("derived"):
            assert "tracked_by" not in entry, f"derived entry {entry} must not carry tracked_by"
        else:
            assert entry["tracked_by"], f"{entry} lacks tracked_by"


def test_rendered_report_matches_the_committed_file() -> None:
    from pathlib import Path

    from .reachability import released_inventory, render_report

    committed = (
        Path(__file__).resolve().parents[2] / "docs" / "operations" / "reachability.md"
    ).read_text(encoding="utf-8")
    assert render_report(released_inventory()) == committed


def test_regeneration_cannot_invent_tracked_by() -> None:
    """The --write refusal is the mechanism that stops reflexive regeneration.

    A non-derived entry with no ``tracked_by`` is refused and named; the derived
    result hole is the only explicit exemption and is accepted without
    ``tracked_by``.
    """
    import pytest

    from .reachability import UntrackedHoleError, assert_baseline_writable

    derived_only: dict[str, list[dict[str, object]]] = {
        "A": [{"missing": "result", "blocks": ["annotation.write"], "derived": True}],
        "B": [],
        "C": [],
        "D": [],
    }
    assert_baseline_writable(derived_only)

    untracked: dict[str, list[dict[str, object]]] = {
        "A": [{"missing": "genome", "blocks": ["annotation.annotate"]}],
        "B": [{"missing": "new_modality", "blocks": ["demo.op"]}],
        "C": [],
        "D": [{"modality": "no_contract"}],
    }
    with pytest.raises(UntrackedHoleError) as excinfo:
        assert_baseline_writable(untracked)
    message = str(excinfo.value)
    assert "genome" in message
    assert "new_modality" in message
    assert "no_contract" in message
