"""Regeneration must preserve the released scientific parameter declarations.

Compare the independent adapter plans with the formal registered specs without
running optional aligners, model backends or fixture computations.
"""

import importlib.util
import sys
import tomllib

import organelleverse as ov
from tests._paths import PROJECT_ROOT


def test_adapter_plans_and_registered_bindings_expose_the_same_parameters():
    path = PROJECT_ROOT / "scripts/capabilities/build_restored_bundles.py"
    module_spec = importlib.util.spec_from_file_location("_scientific_parameter_generator", path)
    assert module_spec is not None and module_spec.loader is not None
    generator = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = generator
    module_spec.loader.exec_module(generator)
    ledger = PROJECT_ROOT / "docs/operations/restored-capabilities.toml"
    payload = tomllib.loads(ledger.read_text(encoding="utf-8"))
    domains = {record["domain"] for record in payload["capability"]}
    registered = {spec.operation_id: spec for spec in ov.operations.list()}
    checked = 0
    for domain in sorted(domains):
        overrides = generator._load_overrides(domain)
        for record in generator._load_domain_records(ledger, domain):
            operation_id = record["id"]
            if operation_id not in registered or operation_id not in overrides:
                continue
            plan = generator._plan_bundle(record, PROJECT_ROOT / "src" / "organelleverse", overrides)
            assert isinstance(plan, generator.PlannedBundle), (operation_id, plan)
            actual = registered[operation_id].binding.parameters
            assert [(p.name, p.codec, p.path_role) for p in plan.parameters] == [
                (p.name, p.codec, p.path_role) for p in actual
            ], operation_id
            checked += 1
    assert checked >= 130
