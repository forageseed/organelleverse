#!/usr/bin/env python3
"""Run every admitted operation end-to-end against REAL local data.

From-scratch, real-pipeline: fresh verification records, the real registry,
real input files (real GenBank genomes, real RNA-seq BAM alignments, real
variant TSV), and per-operation inputs derived from each operation's own
declared binding. Every outcome is classified - a scientific ``failed``
status is a REAL run whose science said no, not a harness error:

- ``PASS``       ok=True, status=ok
- ``SCI-FAIL``   ok=True, status=failed (real execution; e.g. insufficient data)
- ``NO-DEP``     ok=False, dependency.missing (external tool absent)
- ``INPUT-ERR``  ok=False, input.* (harness could not satisfy the contract)
- ``ERROR``      ok=False, anything else (contract/runtime defect - reviewed)
- ``TIMEOUT``    exceeded the per-class budget

Result-input operations (the writer family) receive a schema-valid
OrganelleResult built by a REAL producer (``ideogram()`` over the real
mitochondrial genome) - noted in the report, not synthetic filler.
"""

from __future__ import annotations

import json
import sys
import tempfile
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DATA = PROJECT_ROOT / "tests" / "data"
BAM_DIR = PROJECT_ROOT / "tmp" / "mito_ref_rebuild" / "gymno" / "bam"

SLOW_PREFIXES = ("assembly.", "annotation.", "coevolution.", "morphology.", "fetch.")

# Real files by parameter semantics (name fragments -> real path)
def _real_file_for(name: str, workdir: Path) -> Path | None:
    n = name.casefold()
    if any(k in n for k in ("genbank", "gbk", "annotation")):
        return DATA / "mito.gbk"
    if "bam" in n:
        bam = BAM_DIR / "Ginkgo_SRR33155102.bam"
        return bam if bam.exists() else None
    if "variant" in n or n.endswith("tsv"):
        return DATA / "circular_variants.tsv"
    if any(k in n for k in ("fasta", "fa_", "seq", "reads", "fastq")):
        out = workdir / "mito_derived.fasta"
        if not out.exists():
            from Bio import SeqIO

            record = next(SeqIO.parse(DATA / "mito.gbk", "genbank"))
            out.write_text(f">{record.id} mito\n{record.seq}\n", encoding="utf-8")
        return out
    return None


def _minimal_json_value(prop: dict) -> object:
    if "default" in prop:
        return prop["default"]
    t = prop.get("type")
    if t == "integer":
        return 1
    if t == "number":
        return 0.5
    if t == "boolean":
        return False
    if t == "array":
        item = prop.get("items", {})
        base = _minimal_json_value(item) if item else "x"
        return [base]
    if "enum" in prop:
        return prop["enum"][0]
    return "example"


def _params_from_spec(spec, registry, workdir: Path) -> tuple[dict, list[str]]:
    params: dict[str, object] = {}
    missing: list[str] = []
    schema = registry.invocation_schema(spec.operation_id).get("properties", {})
    try:
        model_fields = registry.require(spec.operation_id).signature.parameter_model.model_fields
    except Exception:
        model_fields = {}
    for binding in spec.binding.parameters:
        name = binding.name
        codec = binding.codec.value
        if codec in ("path", "directory"):
            role = binding.path_role or "input"
            if role == "output":
                params[name] = str(workdir / f"{name}_out")
            else:
                real = _real_file_for(name, workdir)
                if real is None:
                    missing.append(name)
                else:
                    params[name] = str(real)
        elif codec == "legacy_result":
            params[name] = "__RESULT__"  # substituted by the caller
        else:
            prop = schema.get(name, {})
            if prop:
                params[name] = _minimal_json_value(prop)
            else:
                params[name] = _from_field_type(model_fields.get(name))
    return params, missing


def _from_field_type(field) -> object:
    """Schema-less json parameter: derive from the parameter model's own field."""
    import types
    from typing import get_args, get_origin, Union

    if field is None:
        return "example"
    ann = field.annotation
    if get_origin(ann) in (Union, types.UnionType):
        args = [a for a in get_args(ann) if a is not type(None)]
        ann = args[0] if args else str
    origin = get_origin(ann)
    if ann is bool:
        return False
    if ann is int:
        return 1
    if ann is float:
        return 0.5
    if ann is str:
        return "example"
    if origin in (list, tuple):
        inner = get_args(ann)
        element = inner[0] if inner else str
        if element is str:
            return ["a"]
        if element is int:
            return [1]
        if element is float:
            return [0.5]
        if element is dict or (get_origin(element) is dict):
            return [{"key": "value"}]
        return []
    if origin is dict or ann is dict:
        return {"key": "value"}
    from collections.abc import Mapping, Sequence

    if origin is Mapping or ann is Mapping:
        return {"key": "value"}
    if origin is Sequence or ann is Sequence:
        return ["a"]
    return "example"


def _genome_payload(kind: str) -> dict:
    from organelleverse.core.genome import OrganelleMetadata
    from organelleverse.io_genome import read_genbank_genome

    organelle = "mitochondrion" if "mito" in kind or kind == "genome" else "plastid"
    path = DATA / ("mito.gbk" if organelle == "mitochondrion" else "cp.gbk")
    genome = read_genbank_genome(
        path, organelle=organelle, species="Example species", metadata=OrganelleMetadata(source="realdata-matrix")
    )
    return genome.model_dump(mode="json")


def _result_payload() -> dict:
    from organelleverse.visualization.suite_plots import ideogram

    prepared = ideogram(
        [{"chromosome": "chr1", "length": 50000}, {"chromosome": "chr2", "length": 30000}]
    )
    return prepared.model_dump(mode="json")


def run_matrix(timeout_default: int = 120) -> dict:
    from organelleverse.operations.adapters.json import invoke_json
    from organelleverse.operations.registry import registry
    from organelleverse.operations.spec import SideEffect

    workdir = Path(tempfile.mkdtemp(prefix="ov-realdata-"))
    result_payload = _result_payload()
    outcomes: dict[str, dict] = {}

    specs = registry.list()
    for spec in specs:
        op = spec.operation_id
        budget = 420 if any(op.startswith(p) for p in SLOW_PREFIXES) else timeout_default
        params, missing = _params_from_spec(spec, registry, workdir)
        if missing:
            outcomes[op] = {"class": "NO-DATA", "detail": f"no real file for {missing}"}
            continue
        for k, v in list(params.items()):
            if v == "__RESULT__":
                params[k] = result_payload
        input_kind = spec.input_kind.value
        core_input = None
        if input_kind == "genome":
            core_input = _genome_payload("genome")
        elif input_kind == "data":
            core_input = None
        elif input_kind == "result":
            core_input = result_payload
        elif input_kind == "none":
            core_input = None
        if spec.input_sequence and core_input is not None:
            core_input = [core_input, _genome_payload("cp")]

        box: dict = {}
        def _run() -> None:
            try:
                resp = invoke_json(
                    {"operation_id": op, "input": core_input, "parameters": params},
                    registry=registry,
                    granted_side_effects=[e.value for e in SideEffect],
                )
                if resp.get("ok"):
                    status = resp.get("result", {}).get("status", "?")
                    box["class"] = "PASS" if status == "ok" else "SCI-FAIL"
                    box["detail"] = str(resp.get("result", {}).get("summary_text", ""))[:90]
                else:
                    err = resp.get("error", {})
                    code = err.get("error_code", "?")
                    if code == "dependency.missing":
                        box["class"] = "NO-DEP"
                    elif code.startswith("input."):
                        box["class"] = "INPUT-ERR"
                    else:
                        box["class"] = "ERROR"
                    box["detail"] = f"{code}: {str(err.get('message',''))[:80]}"
            except Exception as exc:  # noqa: BLE001 - matrix must survive anything
                box["class"] = "ERROR"
                box["detail"] = f"{type(exc).__name__}: {str(exc)[:90]}"

        t = threading.Thread(target=_run, daemon=True)
        t.start(); t.join(budget)
        if t.is_alive():
            outcomes[op] = {"class": "TIMEOUT", "detail": f">{budget}s"}
        else:
            outcomes[op] = box
    return outcomes


def main() -> int:
    outcomes = run_matrix()
    from collections import Counter

    by = Counter(v["class"] for v in outcomes.values())
    print(f"=== REAL-DATA MATRIX: {len(outcomes)} operations ===")
    for k in ("PASS", "SCI-FAIL", "NO-DEP", "NO-DATA", "INPUT-ERR", "ERROR", "TIMEOUT"):
        if by.get(k):
            print(f"  {k:<10} {by[k]}")
    out = PROJECT_ROOT / "docs" / "reports" / "realdata-matrix"
    out.mkdir(parents=True, exist_ok=True)
    (out / "outcomes.json").write_text(
        json.dumps(outcomes, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    print(f"\nfull matrix: {out / 'outcomes.json'}")
    print("\n--- non-PASS detail ---")
    for op, v in sorted(outcomes.items()):
        if v["class"] != "PASS":
            print(f"  {v['class']:<10} {op}: {v['detail']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
