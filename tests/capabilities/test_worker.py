# pyright: basic
from __future__ import annotations

import ast
import hashlib
import importlib
import json
import os
import py_compile
import signal
import struct
import subprocess
import sys
import threading
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest
from pydantic import ValidationError

import organelleverse.capabilities.worker as worker_module
from organelleverse.capabilities._worker_main import (
    _loads_json,
    _require_request,
    _WorkerFailure,
)
from organelleverse.capabilities.code_identity import inspect_bundle_code
from organelleverse.capabilities.hashing import hash_bundle
from organelleverse.capabilities.index import (
    CapabilityEntry,
    CapabilityIndex,
    CapabilityOrigin,
)
from organelleverse.capabilities.models import CapabilityBundle
from organelleverse.capabilities.trust import TrustDocument, TrustStore, trust
from organelleverse.capabilities.worker import (
    MAX_FRAME_BYTES,
    MAX_STDERR_BYTES,
    OneShotBundleWorkerExecutor,
    WorkerInvocationStrategy,
    _decode_frame,
    _encode_frame,
    _validate_response,
)
from organelleverse.capabilities.worker_contracts import (
    WORKER_PROTOCOL,
    WorkerParameter,
    WorkerRequest,
    WorkerResponse,
    WorkerResult,
)
from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.errors import (
    OrganelleContractError,
    OrganelleExecutionError,
    OrganelleInputError,
    OrganelleParameterError,
    OrganellePermissionError,
)
from organelleverse.core.provenance import ResultProvenance
from organelleverse.core.result import OrganelleResult
from organelleverse.operations.python_binding import worker_parameter_schema
from organelleverse.operations.registry import CoreObject, OperationRegistry
from organelleverse.operations.spec import (
    ArgumentMode,
    CoreKind,
    ExecutionContext,
    ParameterBindingSpec,
    ParameterCodec,
    PythonBindingSpec,
    ResultCodec,
)


def _bundle(
    *,
    locator: str = "private_worker_pkg.impl:run",
    result_codec: ResultCodec = ResultCodec.CANONICAL_JSON,
    parameter_annotation: str = "int",
) -> CapabilityBundle:
    return CapabilityBundle.model_validate(
        {
            "schema": "organelleverse.capability.v1",
            "capability": {
                "id": "demo.worker",
                "bundle_version": "1.0.0",
                "implementation": "native",
            },
            "contract": {
                "operation_id": "demo.worker",
                "contract_version": "1.0",
                "title": "Controlled worker fixture",
                "description": "A bundle-local controlled worker fixture for identity tests.",
                "keywords": ("controlled", "fixture", "worker"),
                "execution_mode": "inline",
                "stage": "analyze",
                "input_kind": "none",
                "output_kind": "result",
                "callable_locator": locator,
                "binding": PythonBindingSpec(
                    argument_mode=ArgumentMode.NAMED_PARAMETERS,
                    parameters=(
                        ParameterBindingSpec(
                            name="value",
                            codec=ParameterCodec.JSON,
                        ),
                    ),
                    result_codec=result_codec,
                ),
            },
        }
    )


def _result_source(
    *,
    value_expression: str = "helper_value + value",
    annotation: str = "int",
    before_return: str = "",
) -> str:
    indented = "".join(f"    {line}\n" for line in before_return.splitlines())
    return (
        "from .helper import helper_value\n"
        f"def run(*, value: {annotation} = 1):\n"
        f"{indented}"
        "    return {\n"
        "        'schema_version': 'organelleverse.result.v1',\n"
        "        'kind': 'result',\n"
        "        'operation_id': 'demo.worker',\n"
        "        'scope': 'none',\n"
        "        'status': 'ok',\n"
        f"        'metrics': {{'value': {value_expression}}},\n"
        "    }\n"
    )


def _write_bundle(
    root: Path,
    *,
    source: str | None = None,
    helper_value: int = 10,
) -> tuple[CapabilityBundle, Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    (root / "capability.toml").write_text("schema='fixture'\n", encoding="utf-8")
    package = root / "code" / "private_worker_pkg"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    implementation = package / "impl.py"
    implementation.write_text(source or _result_source(), encoding="utf-8")
    helper = package / "helper.py"
    helper.write_text(f"helper_value = {helper_value}\n", encoding="utf-8")
    return _bundle(), implementation, helper


def _entry(root: Path, bundle: CapabilityBundle | None = None) -> CapabilityEntry:
    selected = bundle or _bundle()
    content_hash = hash_bundle(root)
    identity = inspect_bundle_code(
        root,
        capability_id=selected.capability.id,
        bundle_content_hash=content_hash,
        callable_locator=cast(str, selected.contract.callable_locator),
    )
    return CapabilityEntry(
        capability_id=selected.capability.id,
        content_hash=content_hash,
        bundle_root=root,
        bundle=selected,
        origins=(
            CapabilityOrigin(
                channel="local",
                source_path=str(root / "capability.toml"),
                search_root=str(root.parent),
            ),
        ),
        execution_identity=identity,
    )


def _invoke(
    executor: OneShotBundleWorkerExecutor,
    entry: CapabilityEntry,
    tmp_path: Path,
    *,
    value: object = 2,
) -> WorkerResult:
    staging = tmp_path / "staging"
    staging.mkdir(exist_ok=True)
    if entry.worker_parameters is None:
        inspection = executor.inspect(entry)
        entry = entry.model_copy(update={"worker_parameters": inspection.parameters})
    return executor.invoke(
        entry,
        input=None,
        parameters={"value": value},
        run_id="worker-test-run",
        staging_root=staging,
    )


def _metric(result: WorkerResult) -> int:
    assert isinstance(result.value, dict)
    metrics = result.value["metrics"]
    assert isinstance(metrics, dict)
    value = metrics["value"]
    assert isinstance(value, int)
    return value


def _remove_private_modules() -> None:
    for name in tuple(sys.modules):
        if name == "private_worker_pkg" or name.startswith("private_worker_pkg."):
            del sys.modules[name]


def test_worker_main_is_stdlib_only_by_ast() -> None:
    worker_main = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "organelleverse"
        / "capabilities"
        / "_worker_main.py"
    )
    tree = ast.parse(worker_main.read_text(encoding="utf-8"), filename=str(worker_main))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_roots.add(node.module.split(".", 1)[0])

    assert "organelleverse" not in imported_roots
    assert "pydantic" not in imported_roots
    assert imported_roots <= sys.stdlib_module_names


def test_worker_inspection_returns_closed_deterministic_signature_tokens(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bundle"
    bundle, _, _ = _write_bundle(root)
    entry = _entry(root, bundle)

    inspection = OneShotBundleWorkerExecutor().inspect(entry)

    assert inspection.execution_identity == entry.execution_identity
    assert inspection.parameters == (
        WorkerParameter(
            name="value",
            kind="keyword_only",
            required=False,
            default=1,
            annotation="int",
        ),
    )


def test_inspection_rejects_any_worker_scratch_side_effect(tmp_path: Path) -> None:
    root = tmp_path / "bundle-inspect-side-effect"
    source = (
        "from pathlib import Path\n"
        "Path('inspection-residue.txt').write_text('forbidden', encoding='utf-8')\n"
        "def run(*, value: int = 1):\n"
        "    return {'value': value}\n"
    )
    bundle, _, _ = _write_bundle(root, source=source)

    with pytest.raises(OrganelleContractError) as captured:
        OneShotBundleWorkerExecutor().inspect(_entry(root, bundle))

    assert captured.value.code == "capability.inspect_side_effect"


def test_real_worker_rechecks_full_bundle_before_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "bundle-child-manifest-drift"
    marker = tmp_path / "import-marker.txt"
    source = (
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('imported', encoding='utf-8')\n" + _result_source()
    )
    bundle, _, _ = _write_bundle(root, source=source)
    parameters = (
        WorkerParameter(
            name="value",
            kind="keyword_only",
            required=False,
            default=1,
            annotation="int",
        ),
    )
    entry = _admitted_custom_entry(root, bundle, parameters)
    store = TrustStore(tmp_path / "trust-child-drift.json")
    assert entry.execution_identity is not None
    trust(entry.capability_id, entry.execution_identity, store=store)
    strategy = WorkerInvocationStrategy(entry, OneShotBundleWorkerExecutor(), store)
    real_popen = worker_module.subprocess.Popen

    def drift_at_spawn(*args: Any, **kwargs: Any) -> subprocess.Popen[bytes]:
        (root / "capability.toml").write_text("schema='spawn-window-drift'\n", encoding="utf-8")
        return cast("subprocess.Popen[bytes]", real_popen(*args, **kwargs))

    monkeypatch.setattr(worker_module.subprocess, "Popen", drift_at_spawn)
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    with pytest.raises(OrganelleExecutionError) as captured:
        strategy.invoke(None, {"value": 1})

    assert captured.value.code == "capability.worker_identity_mismatch"
    assert not marker.exists()


@pytest.mark.parametrize("mutation_phase", ["import", "invoke"])
def test_real_worker_manifest_mutation_never_reaches_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation_phase: str,
) -> None:
    root = tmp_path / f"bundle-manifest-{mutation_phase}"
    manifest = root / "capability.toml"
    mutation = (
        f"Path({str(manifest)!r}).write_text("
        f"\"schema='worker-{mutation_phase}-drift'\\n\", encoding='utf-8')"
    )
    if mutation_phase == "import":
        source = "from pathlib import Path\n" + mutation + "\n" + _result_source()
    else:
        source = "from pathlib import Path\n" + _result_source(before_return=mutation)
    bundle, _, _ = _write_bundle(root, source=source)
    parameters = (
        WorkerParameter(
            name="value",
            kind="keyword_only",
            required=False,
            default=1,
            annotation="int",
        ),
    )
    entry = _admitted_custom_entry(root, bundle, parameters)
    store = TrustStore(tmp_path / f"trust-{mutation_phase}.json")
    assert entry.execution_identity is not None
    trust(entry.capability_id, entry.execution_identity, store=store)
    strategy = WorkerInvocationStrategy(entry, OneShotBundleWorkerExecutor(), store)
    cache = tmp_path / "cache"
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(cache))

    with pytest.raises(OrganelleContractError) as captured:
        strategy.invoke(None, {"value": 1})

    assert captured.value.code == "capability.execution_identity_drift"
    assert _managed_operation_entries(cache) == ()


def test_verified_zero_parameter_callable_invokes_with_empty_frozen_tokens(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bundle"
    source = _result_source(
        value_expression="helper_value",
    ).replace("def run(*, value: int = 1):", "def run():")
    bundle, _, _ = _write_bundle(root, source=source)
    bundle = bundle.model_copy(
        update={
            "contract": bundle.contract.model_copy(
                update={"binding": bundle.contract.binding.model_copy(update={"parameters": ()})}
            )
        }
    )
    entry = _entry(root, bundle)
    executor = OneShotBundleWorkerExecutor()
    inspection = executor.inspect(entry)
    staging = tmp_path / "staging"
    staging.mkdir()

    result = executor.invoke(
        entry.model_copy(update={"worker_parameters": inspection.parameters}),
        input=None,
        parameters={},
        run_id="zero-parameter-test",
        staging_root=staging,
    )

    assert inspection.parameters == ()
    assert _metric(result) == 10


def test_parent_preloaded_same_named_module_cannot_substitute_for_verified_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "bundle"
    bundle, implementation, _ = _write_bundle(root)
    entry = _entry(root, bundle)
    parent_module = ModuleType("private_worker_pkg.impl")
    parent_module.__file__ = str(tmp_path / "ambient" / "impl.py")
    parent_module.run = lambda **_: {  # pyright: ignore[reportAttributeAccessIssue]
        "metrics": {"value": 999}
    }
    monkeypatch.setitem(sys.modules, "private_worker_pkg.impl", parent_module)
    before = sys.modules["private_worker_pkg.impl"]

    result = _invoke(OneShotBundleWorkerExecutor(), entry, tmp_path)

    assert _metric(result) == 12
    assert sys.modules["private_worker_pkg.impl"] is before
    assert sys.modules["private_worker_pkg.impl"].__file__ != str(implementation)


def test_parent_old_module_at_same_path_remains_old_while_worker_executes_new_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "bundle"
    bundle, _, helper = _write_bundle(root, helper_value=1)
    monkeypatch.syspath_prepend(str(root / "code"))
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    _remove_private_modules()
    try:
        old_helper = importlib.import_module("private_worker_pkg.helper")
        assert old_helper.helper_value == 1
        helper.write_text("helper_value = 20\n", encoding="utf-8")
        entry = _entry(root, bundle)

        result = _invoke(OneShotBundleWorkerExecutor(), entry, tmp_path)

        assert _metric(result) == 22
        assert old_helper.helper_value == 1
        assert sys.modules["private_worker_pkg.helper"] is old_helper
    finally:
        _remove_private_modules()


def test_timestamp_valid_ambient_pyc_cannot_substitute_for_verified_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "bundle"
    bundle, _, _ = _write_bundle(root)
    entry = _entry(root, bundle)
    ambient_package = tmp_path / "ambient" / "private_worker_pkg"
    ambient_package.mkdir(parents=True)
    (ambient_package / "__init__.py").write_text("", encoding="utf-8")
    malicious = ambient_package / "impl.py"
    malicious.write_text(
        "def run(*, value=1):\n    return {'metrics': {'value': 999}}\n",
        encoding="utf-8",
    )
    py_compile.compile(str(malicious), doraise=True)
    monkeypatch.syspath_prepend(str(tmp_path / "ambient"))

    result = _invoke(OneShotBundleWorkerExecutor(), entry, tmp_path)

    assert _metric(result) == 12
    assert "private_worker_pkg.impl" not in sys.modules


def test_changed_sibling_helper_fails_identity_before_any_bundle_code_runs(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "scientific-marker.txt"
    root = tmp_path / "bundle"
    source = (
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('imported', encoding='utf-8')\n" + _result_source()
    )
    bundle, _, helper = _write_bundle(root, source=source)
    entry = _entry(root, bundle)
    helper.write_text("helper_value = 999\n", encoding="utf-8")

    with pytest.raises(OrganelleExecutionError) as captured:
        _invoke(OneShotBundleWorkerExecutor(), entry, tmp_path)

    assert captured.value.code == "capability.worker_identity_mismatch"
    assert not marker.exists()


def test_worker_hashes_and_compiles_one_snapshot_when_code_rewrites_a_helper(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bundle"
    source = (
        "from pathlib import Path\n"
        "def run(*, value: int = 1):\n"
        "    Path(__file__).with_name('helper.py').write_text(\n"
        "        'helper_value = 999\\n', encoding='utf-8'\n"
        "    )\n"
        "    from .helper import helper_value\n"
        "    return {\n"
        "        'schema_version': 'organelleverse.result.v1',\n"
        "        'kind': 'result',\n"
        "        'operation_id': 'demo.worker',\n"
        "        'scope': 'none',\n"
        "        'status': 'ok',\n"
        "        'metrics': {'value': helper_value + value},\n"
        "    }\n"
    )
    bundle, _, helper = _write_bundle(root, source=source, helper_value=10)
    entry = _entry(root, bundle)
    executor = OneShotBundleWorkerExecutor()

    first = _invoke(executor, entry, tmp_path)

    assert _metric(first) == 12
    assert helper.read_text(encoding="utf-8") == "helper_value = 999\n"
    with pytest.raises(OrganelleExecutionError) as captured:
        _invoke(executor, entry, tmp_path)
    assert captured.value.code == "capability.worker_identity_mismatch"


def test_missing_private_helper_never_falls_back_to_an_ambient_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "scientific-marker.txt"
    root = tmp_path / "bundle"
    source = (
        "from pathlib import Path\n"
        "from .missing_helper import helper_value\n"
        f"Path({str(marker)!r}).write_text('imported', encoding='utf-8')\n"
        "def run(*, value: int = 1):\n"
        "    return {'metrics': {'value': helper_value + value}}\n"
    )
    bundle, _, helper = _write_bundle(root, source=source)
    helper.unlink()
    entry = _entry(root, bundle)
    ambient = ModuleType("private_worker_pkg.missing_helper")
    ambient.helper_value = 999  # pyright: ignore[reportAttributeAccessIssue]
    monkeypatch.setitem(sys.modules, "private_worker_pkg.missing_helper", ambient)

    with pytest.raises(OrganelleExecutionError) as captured:
        OneShotBundleWorkerExecutor().inspect(entry)

    assert captured.value.code == "capability.worker_execution_failed"
    assert not marker.exists()
    assert sys.modules["private_worker_pkg.missing_helper"] is ambient


@pytest.mark.parametrize(
    ("payload", "code"),
    [
        (struct.pack(">Q", 1) + b"{", "capability.worker_frame_invalid"),
        (struct.pack(">Q", MAX_FRAME_BYTES + 1), "capability.worker_frame_too_large"),
        (struct.pack(">Q", 2) + b"{}trailing", "capability.worker_frame_invalid"),
    ],
)
def test_invalid_or_oversized_response_frames_fail_closed(payload: bytes, code: str) -> None:
    with pytest.raises(OrganelleExecutionError) as captured:
        _decode_frame(payload)

    assert captured.value.code == code


@pytest.mark.parametrize(
    "body",
    [
        b'{"duplicate":1,"duplicate":2}',
        b'{"nonfinite":NaN}',
        (b"[" * 80) + b"0" + (b"]" * 80),
        b"[" + b",".join(b"0" for _ in range(20_000)) + b"]",
    ],
)
def test_response_json_rejects_duplicate_nonfinite_or_excessive_structure(
    body: bytes,
) -> None:
    frame = struct.pack(">Q", len(body)) + body

    with pytest.raises(OrganelleExecutionError) as captured:
        _decode_frame(frame)

    assert captured.value.code == "capability.worker_frame_invalid"


def test_response_json_rejects_noncanonical_wire_encoding() -> None:
    body = b'{"z": 1, "a": 2}'
    frame = struct.pack(">Q", len(body)) + body

    with pytest.raises(OrganelleExecutionError) as captured:
        _decode_frame(frame)

    assert captured.value.code == "capability.worker_frame_invalid"


def test_parent_decode_wraps_json_parser_recursion_as_a_protocol_error() -> None:
    body = (b"[" * 10_000) + b"0" + (b"]" * 10_000)
    frame = struct.pack(">Q", len(body)) + body

    with pytest.raises(OrganelleExecutionError) as captured:
        _decode_frame(frame)

    assert captured.value.code == "capability.worker_frame_invalid"


def test_child_decode_wraps_json_parser_recursion_without_a_traceback() -> None:
    body = (b"[" * 10_000) + b"0" + (b"]" * 10_000)
    worker_main = Path(worker_module.__file__).with_name("_worker_main.py")

    process = subprocess.run(
        [sys.executable, "-I", "-S", "-E", "-B", str(worker_main)],
        input=struct.pack(">Q", len(body)) + body,
        capture_output=True,
        check=False,
    )

    assert process.returncode == 2
    assert b"capability.worker_frame_invalid" in process.stderr
    assert b"Traceback" not in process.stderr
    with pytest.raises(_WorkerFailure) as captured:
        _loads_json(body)
    assert captured.value.code == "capability.worker_frame_invalid"


def _protocol_fixture_script(path: Path, *, wrong_protocol: bool, trailing: bool) -> None:
    path.write_text(
        "import json, struct, sys\n"
        "header = sys.stdin.buffer.read(8)\n"
        "size = struct.unpack('>Q', header)[0]\n"
        "request = json.loads(sys.stdin.buffer.read(size))\n"
        "response = {\n"
        f"    'protocol': {'wrong.protocol'!r} if {wrong_protocol!r} else request['protocol'],\n"
        "    'request_id': request['request_id'],\n"
        "    'operation_id': request['operation_id'],\n"
        "    'mode': request['mode'],\n"
        "    'execution_identity': request['execution_identity'],\n"
        "    'status': 'ok',\n"
        "    'parameters': [],\n"
        "    'value': None,\n"
        "    'artifact_paths': [],\n"
        "    'error': None,\n"
        "}\n"
        "body = json.dumps(response, sort_keys=True, separators=(',', ':'), "
        "ensure_ascii=True, allow_nan=False).encode('utf-8')\n"
        "sys.stdout.buffer.write(struct.pack('>Q', len(body)) + body)\n"
        f"sys.stdout.buffer.write(b'x' if {trailing!r} else b'')\n"
        "sys.stdout.buffer.flush()\n",
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("wrong_protocol", "trailing", "expected_code"),
    [
        (True, False, "capability.worker_protocol_invalid"),
        (False, True, "capability.worker_frame_invalid"),
    ],
)
def test_real_exchange_decode_path_rejects_wrong_protocol_or_trailing_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    wrong_protocol: bool,
    trailing: bool,
    expected_code: str,
) -> None:
    root = tmp_path / "bundle"
    bundle, _, _ = _write_bundle(root)
    responder = tmp_path / "protocol_fixture.py"
    _protocol_fixture_script(
        responder,
        wrong_protocol=wrong_protocol,
        trailing=trailing,
    )
    monkeypatch.setattr(worker_module, "_WORKER_MAIN", responder.resolve())

    with pytest.raises(OrganelleExecutionError) as captured:
        OneShotBundleWorkerExecutor().inspect(_entry(root, bundle))

    assert captured.value.code == expected_code


def test_oversized_request_frame_is_rejected_before_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "bundle"
    bundle, _, _ = _write_bundle(root)
    entry = _entry(root, bundle)
    inspection = OneShotBundleWorkerExecutor().inspect(entry)
    entry = entry.model_copy(update={"worker_parameters": inspection.parameters})

    def forbidden_spawn(*args: object, **kwargs: object) -> None:
        raise AssertionError("oversized request must fail before Popen")

    monkeypatch.setattr(subprocess, "Popen", forbidden_spawn)
    with pytest.raises(OrganelleContractError) as captured:
        _invoke(OneShotBundleWorkerExecutor(), entry, tmp_path, value="x" * MAX_FRAME_BYTES)

    assert captured.value.code == "capability.worker_frame_too_large"


def test_worker_spawn_failure_is_structured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "bundle"
    bundle, _, _ = _write_bundle(root)
    entry = _entry(root, bundle)

    def fail_spawn(*args: object, **kwargs: object) -> None:
        raise OSError("injected spawn failure")

    monkeypatch.setattr(subprocess, "Popen", fail_spawn)

    with pytest.raises(OrganelleExecutionError) as captured:
        OneShotBundleWorkerExecutor().inspect(entry)

    assert captured.value.code == "capability.worker_spawn_failed"


def test_timeout_covers_a_blocked_request_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "bundle"
    bundle, _, _ = _write_bundle(root)
    request = _request(_entry(root, bundle), tmp_path).model_copy(
        update={"contract": {"padding": "x" * (1024 * 1024)}}
    )
    sleeper = tmp_path / "never_reads_stdin.py"
    sleeper.write_text("import time\ntime.sleep(60)\n", encoding="utf-8")
    monkeypatch.setattr(worker_module, "_WORKER_MAIN", sleeper.resolve())
    executor = OneShotBundleWorkerExecutor(timeout_seconds=0.1)
    started = time.monotonic()

    with pytest.raises(OrganelleExecutionError) as captured:
        executor._exchange(request)

    assert captured.value.code == "capability.worker_timeout"
    assert time.monotonic() - started < 5


def test_partial_thread_start_failure_terminates_reaps_and_closes_worker_pipes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "bundle"
    bundle, _, _ = _write_bundle(root)
    real_popen = subprocess.Popen
    processes: list[subprocess.Popen[bytes]] = []
    real_start = threading.Thread.start
    starts = 0

    def tracking_popen(*args: Any, **kwargs: Any) -> subprocess.Popen[bytes]:
        process = cast("subprocess.Popen[bytes]", real_popen(*args, **kwargs))
        processes.append(process)
        return process

    def fail_second_start(thread: threading.Thread) -> None:
        nonlocal starts
        starts += 1
        if starts == 2:
            raise RuntimeError("injected thread startup failure")
        real_start(thread)

    monkeypatch.setattr(subprocess, "Popen", tracking_popen)
    monkeypatch.setattr(threading.Thread, "start", fail_second_start)
    try:
        with pytest.raises(OrganelleExecutionError) as captured:
            OneShotBundleWorkerExecutor().inspect(_entry(root, bundle))

        assert captured.value.code == "capability.worker_thread_start_failed"
        assert len(processes) == 1
        process = processes[0]
        assert process.poll() is not None
        assert process.stdin is not None and process.stdin.closed
        assert process.stdout is not None and process.stdout.closed
        assert process.stderr is not None and process.stderr.closed
    finally:
        for process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)


def _request(entry: CapabilityEntry, tmp_path: Path) -> WorkerRequest:
    assert entry.execution_identity is not None
    return WorkerRequest(
        request_id="request-1",
        operation_id=entry.capability_id,
        mode="inspect",
        execution_identity=entry.execution_identity,
        bundle_root=str(entry.bundle_root.resolve()),
        contract=entry.bundle.contract.model_dump(mode="json"),
        worker_parameters=(),
        input=None,
        parameters={},
        run_id="worker-test-run",
        staging_root=str(tmp_path.resolve()),
    )


@pytest.mark.parametrize(
    ("update", "expected_code"),
    [
        ({"protocol": "wrong.protocol"}, "capability.worker_protocol_mismatch"),
        ({"request_id": "wrong-request"}, "capability.worker_request_mismatch"),
        ({"operation_id": "other.worker"}, "capability.worker_operation_mismatch"),
    ],
)
def test_wrong_response_protocol_request_or_operation_is_rejected(
    tmp_path: Path, update: dict[str, object], expected_code: str
) -> None:
    root = tmp_path / "bundle"
    bundle, _, _ = _write_bundle(root)
    entry = _entry(root, bundle)
    request = _request(entry, tmp_path)
    response_values: dict[str, Any] = {
        "protocol": WORKER_PROTOCOL,
        "request_id": request.request_id,
        "operation_id": request.operation_id,
        "mode": request.mode,
        "execution_identity": request.execution_identity,
        "status": "ok",
        "parameters": (),
        "value": None,
        "artifact_paths": (),
        "error": None,
    }
    response_values.update(update)
    response = WorkerResponse.model_construct(**response_values)

    with pytest.raises(OrganelleExecutionError) as captured:
        _validate_response(request, response)

    assert captured.value.code == expected_code


def test_wrong_response_execution_identity_is_rejected(tmp_path: Path) -> None:
    first_root = tmp_path / "one"
    second_root = tmp_path / "two"
    first_bundle, _, _ = _write_bundle(first_root, helper_value=1)
    second_bundle, _, _ = _write_bundle(second_root, helper_value=2)
    first = _entry(first_root, first_bundle)
    second = _entry(second_root, second_bundle)
    request = _request(first, tmp_path)
    assert second.execution_identity is not None
    response = WorkerResponse(
        request_id=request.request_id,
        operation_id=request.operation_id,
        mode=request.mode,
        execution_identity=second.execution_identity,
        status="ok",
    )

    with pytest.raises(OrganelleExecutionError) as captured:
        _validate_response(request, response)

    assert captured.value.code == "capability.worker_identity_mismatch"


def test_worker_timeout_kills_the_one_shot_process_group(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    bundle, _, _ = _write_bundle(
        root,
        source=_result_source(before_return="import time\ntime.sleep(5)"),
    )
    entry = _entry(root, bundle)

    with pytest.raises(OrganelleExecutionError) as captured:
        _invoke(OneShotBundleWorkerExecutor(timeout_seconds=0.1), entry, tmp_path)

    assert captured.value.code == "capability.worker_timeout"


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group semantics")
def test_worker_timeout_kills_a_spawned_descendant_process(tmp_path: Path) -> None:
    descendant_pid = tmp_path / "descendant.pid"
    descendant_code = (
        "import os, pathlib, time; "
        f"pathlib.Path({str(descendant_pid)!r}).write_text(str(os.getpid()), encoding='utf-8'); "
        "time.sleep(60)"
    )
    root = tmp_path / "bundle"
    bundle, _, _ = _write_bundle(
        root,
        source=_result_source(
            before_return=(
                "import subprocess\n"
                "import sys\n"
                "import time\n"
                f"subprocess.Popen([sys.executable, '-c', {descendant_code!r}])\n"
                "time.sleep(60)"
            )
        ),
    )

    # The timeout has to outlast interpreter startup, or the worker is killed
    # before it ever reaches run() and spawns the descendant this test is about.
    with pytest.raises(OrganelleExecutionError) as captured:
        _invoke(OneShotBundleWorkerExecutor(timeout_seconds=2.0), _entry(root, bundle), tmp_path)

    assert captured.value.code == "capability.worker_timeout"
    assert descendant_pid.is_file()
    pid = int(descendant_pid.read_text(encoding="utf-8"))
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        proc_stat = Path(f"/proc/{pid}/stat")
        if not proc_stat.exists():
            break
        if proc_stat.read_text(encoding="utf-8").split()[2] == "Z":
            break
        time.sleep(0.05)
    else:
        os.kill(pid, signal.SIGKILL)
        pytest.fail("worker descendant survived process-group cleanup")


def test_scientific_function_body_type_error_is_not_misreported_as_binding_invalid(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bundle"
    bundle, _, _ = _write_bundle(
        root,
        source=_result_source(
            before_return="raise TypeError('scientific body failure' + 'x' * 20_000)"
        ),
    )

    with pytest.raises(OrganelleExecutionError) as captured:
        _invoke(OneShotBundleWorkerExecutor(), _entry(root, bundle), tmp_path)

    assert captured.value.code == "capability.worker_execution_failed"
    details = captured.value.as_dict()["details"]
    assert details["exception_type"] == "TypeError"
    assert str(details["reason"]).startswith("scientific body failure")
    assert len(str(details["reason"])) <= 4_096


@pytest.mark.skipif(os.name == "nt", reason="POSIX signal exit semantics")
def test_worker_signal_exit_is_structured(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    bundle, _, _ = _write_bundle(
        root,
        source=_result_source(
            before_return=("import os\nimport signal\nos.kill(os.getpid(), signal.SIGTERM)")
        ),
    )
    entry = _entry(root, bundle)

    with pytest.raises(OrganelleExecutionError) as captured:
        _invoke(OneShotBundleWorkerExecutor(), entry, tmp_path)

    assert captured.value.code == "capability.worker_signal"
    assert captured.value.as_dict()["details"]["signal"] == signal.SIGTERM


def test_worker_nonzero_exit_and_stderr_capture_are_bounded(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    bundle, _, _ = _write_bundle(
        root,
        source=_result_source(
            before_return=(
                "import os\n"
                "import sys\n"
                f"sys.stderr.write('x' * {MAX_STDERR_BYTES * 4})\n"
                "sys.stderr.flush()\n"
                "os._exit(7)"
            )
        ),
    )
    entry = _entry(root, bundle)

    with pytest.raises(OrganelleExecutionError) as captured:
        _invoke(OneShotBundleWorkerExecutor(), entry, tmp_path)

    assert captured.value.code == "capability.worker_nonzero_exit"
    details = captured.value.as_dict()["details"]
    assert details["returncode"] == 7
    assert details["stderr_truncated"] is True
    assert len(details["stderr"].encode("utf-8")) <= MAX_STDERR_BYTES


def test_bundle_stdout_is_redirected_away_from_the_protocol_frame(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    bundle, _, _ = _write_bundle(
        root,
        source=_result_source(before_return="print('bundle chatter')"),
    )
    entry = _entry(root, bundle)

    result = _invoke(OneShotBundleWorkerExecutor(), entry, tmp_path)

    assert _metric(result) == 12


def test_launcher_uses_exact_isolated_flags_minimal_environment_and_private_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "bundle"
    bundle, _, _ = _write_bundle(root)
    entry = _entry(root, bundle)
    real_popen = subprocess.Popen
    observed: list[tuple[object, dict[str, object]]] = []

    def tracking_popen(*args: Any, **kwargs: Any) -> subprocess.Popen[bytes]:
        assert list(Path(cast(str, kwargs["cwd"])).iterdir()) == []
        observed.append((args[0], dict(kwargs)))
        return cast("subprocess.Popen[bytes]", real_popen(*args, **kwargs))

    monkeypatch.setattr(subprocess, "Popen", tracking_popen)
    OneShotBundleWorkerExecutor().inspect(entry)

    assert len(observed) == 1
    command, options = observed[0]
    assert isinstance(command, list)
    assert command[:5] == [sys.executable, "-I", "-S", "-E", "-B"]
    # ``-B`` only stops the worker writing bytecode; ``pycache_prefix`` is what
    # stops it *reading* a ``__pycache__`` that ships inside a bundle. The
    # directory must be private to this run and outside the bundle, and the
    # launcher must remove it again.
    assert command[5] == "-X"
    assert isinstance(command[6], str)
    cache_root = Path(command[6].removeprefix("pycache_prefix="))
    assert command[6] != str(cache_root)
    assert cache_root.is_absolute()
    assert not cache_root.is_relative_to(root)
    assert not cache_root.exists()
    assert Path(command[7]).is_absolute()
    assert Path(command[7]).name == "_worker_main.py"
    assert len(command) == 8
    assert options["shell"] is False
    assert options["cwd"] != str(root)
    environment = cast(dict[str, str], options["env"])
    assert "PYTHONPATH" not in environment
    assert "PYTHONHOME" not in environment
    assert set(environment) <= {"LANG", "LC_ALL", "PATH", "SYSTEMROOT", "TZ"}


def test_windows_fails_provider_required_without_claiming_process_tree_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import organelleverse.capabilities.worker as worker

    root = tmp_path / "bundle"
    bundle, _, _ = _write_bundle(root)
    entry = _entry(root, bundle)
    monkeypatch.setattr(worker, "_WINDOWS", True)

    with pytest.raises(OrganelleContractError) as captured:
        OneShotBundleWorkerExecutor().inspect(entry)

    assert captured.value.code == "capability.execution_provider_required"


class _FakeExecutor:
    def __init__(self, entry: CapabilityEntry, parameters: tuple[WorkerParameter, ...]) -> None:
        self.entry = entry
        self.parameters = parameters
        self.inspect_calls = 0
        self.invoke_calls = 0

    def inspect(self, entry: CapabilityEntry):
        from organelleverse.capabilities.worker_contracts import WorkerInspection

        self.inspect_calls += 1
        assert entry == self.entry
        assert entry.execution_identity is not None
        return WorkerInspection(
            request_id="fake-inspect",
            execution_identity=entry.execution_identity,
            parameters=self.parameters,
        )

    def invoke(
        self,
        entry: CapabilityEntry,
        *,
        input: CoreObject | None,
        parameters: Mapping[str, object],
        run_id: str,
        staging_root: Path,
    ) -> WorkerResult:
        self.invoke_calls += 1
        assert entry == self.entry
        assert entry.execution_identity is not None
        value = parameters["value"]
        assert isinstance(value, int)
        return WorkerResult(
            request_id="fake-invoke",
            execution_identity=entry.execution_identity,
            value={
                "schema_version": "organelleverse.result.v1",
                "kind": "result",
                "operation_id": "demo.worker",
                "scope": "none",
                "status": "ok",
                "metrics": {"value": value},
            },
        )


class _PayloadExecutor:
    def __init__(self, entry: CapabilityEntry, value: object) -> None:
        self.entry = entry
        self.value = value
        self.invoke_calls = 0
        self.parameters: Mapping[str, object] | None = None

    def inspect(self, entry: CapabilityEntry):
        raise AssertionError("admitted proxy tests must not inspect again")

    def invoke(
        self,
        entry: CapabilityEntry,
        *,
        input: CoreObject | None,
        parameters: Mapping[str, object],
        run_id: str,
        staging_root: Path,
    ) -> WorkerResult:
        self.invoke_calls += 1
        self.parameters = parameters
        assert entry == self.entry
        assert entry.execution_identity is not None
        return WorkerResult(
            request_id="payload-invoke",
            execution_identity=entry.execution_identity,
            value=cast(Any, self.value),
        )


def _admitted_entry(
    root: Path,
) -> tuple[CapabilityEntry, tuple[WorkerParameter, ...]]:
    bundle, _, _ = _write_bundle(root)
    entry = _entry(root, bundle)
    inspection = OneShotBundleWorkerExecutor().inspect(entry)
    schema = worker_parameter_schema(bundle, inspection.parameters)
    return (
        entry.model_copy(
            update={
                "status": "admitted",
                "parameter_schema": schema,
                "worker_parameters": inspection.parameters,
            }
        ),
        inspection.parameters,
    )


def _admitted_custom_entry(
    root: Path,
    bundle: CapabilityBundle,
    parameters: tuple[WorkerParameter, ...],
) -> CapabilityEntry:
    entry = _entry(root, bundle)
    return entry.model_copy(
        update={
            "status": "admitted",
            "parameter_schema": worker_parameter_schema(bundle, parameters),
            "worker_parameters": parameters,
        }
    )


def test_non_host_execution_context_fails_before_worker_dispatch(tmp_path: Path) -> None:
    entry, _ = _admitted_entry(tmp_path / "bundle")
    non_host_entry = cast(
        CapabilityEntry,
        SimpleNamespace(
            capability_id=entry.capability_id,
            execution_identity=entry.execution_identity,
            bundle=SimpleNamespace(
                contract=SimpleNamespace(
                    callable_locator=entry.bundle.contract.callable_locator,
                    execution_context=ExecutionContext.CONTAINER,
                )
            ),
        ),
    )
    executor = _PayloadExecutor(non_host_entry, {"unused": True})
    store = TrustStore(tmp_path / "trust.json")
    assert entry.execution_identity is not None
    trust(entry.capability_id, entry.execution_identity, store=store)

    strategy = WorkerInvocationStrategy(non_host_entry, executor, store)
    with pytest.raises(OrganelleExecutionError) as captured:
        strategy.invoke(None, {"value": 1})

    assert captured.value.code == "capability.execution_context_unavailable"
    details = cast(dict[str, object], captured.value.details)
    assert details["requested_context"] == "container"
    assert executor.invoke_calls == 0


@pytest.mark.parametrize(
    ("annotation", "expected_runtime_type"),
    [("str", "str"), ("Path", type(Path()).__name__)],
)
def test_relative_path_parameter_is_resolved_once_and_sent_with_matching_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    annotation: str,
    expected_runtime_type: str,
) -> None:
    relative = Path("inputs") / "source.txt"
    source_path = tmp_path / relative
    source_path.parent.mkdir()
    source_path.write_text("verified path content", encoding="utf-8")
    root = tmp_path / "bundle"
    source = (
        "from pathlib import Path\n"
        f"def run(*, value: {annotation}):\n"
        "    path = Path(value)\n"
        "    return {\n"
        "        'schema_version': 'organelleverse.result.v1',\n"
        "        'kind': 'result',\n"
        "        'operation_id': 'demo.worker',\n"
        "        'scope': 'none',\n"
        "        'status': 'ok',\n"
        "        'metrics': {\n"
        "            'absolute': path.is_absolute(),\n"
        "            'content': path.read_text(encoding='utf-8'),\n"
        "            'runtime_type': type(value).__name__,\n"
        "        },\n"
        "    }\n"
    )
    base, _, _ = _write_bundle(root, source=source)
    bundle = base.model_copy(
        update={
            "contract": base.contract.model_copy(
                update={
                    "binding": base.contract.binding.model_copy(
                        update={
                            "parameters": (
                                ParameterBindingSpec(
                                    name="value",
                                    codec=ParameterCodec.PATH,
                                ),
                            )
                        }
                    )
                }
            )
        }
    )
    entry = _entry(root, bundle)
    executor = OneShotBundleWorkerExecutor()
    inspection = executor.inspect(entry)
    entry = _admitted_custom_entry(root, bundle, inspection.parameters)
    trust_store = TrustStore(tmp_path / "trust.json")
    assert entry.execution_identity is not None
    trust(entry.capability_id, entry.execution_identity, store=trust_store)
    registry = OperationRegistry(
        capability_source=CapabilityIndex(entries=(entry,)).binding_source(
            trust_store=trust_store,
            executor=executor,
        )
    )
    monkeypatch.chdir(tmp_path)

    result = registry.invoke(
        entry.capability_id,
        input=None,
        parameters={"value": relative.as_posix()},
    )

    assert isinstance(result, OrganelleResult)
    assert result.metrics["absolute"] is True
    assert result.metrics["content"] == "verified path content"
    assert result.metrics["runtime_type"] == expected_runtime_type
    assert result.provenance is not None
    expected_artifact_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    assert result.provenance.input_artifact_hashes == (expected_artifact_hash,)
    expected_parameters_hash = hashlib.sha256(b'{"value":"inputs/source.txt"}').hexdigest()
    assert result.provenance.parameters_hash == expected_parameters_hash


def _trusted_registry_with_payload(
    tmp_path: Path,
    entry: CapabilityEntry,
    payload: object,
) -> tuple[OperationRegistry, _PayloadExecutor]:
    executor = _PayloadExecutor(entry, payload)
    store = TrustStore(tmp_path / "trust.json")
    assert entry.execution_identity is not None
    trust(entry.capability_id, entry.execution_identity, store=store)
    registry = OperationRegistry(
        capability_source=CapabilityIndex(entries=(entry,)).binding_source(
            trust_store=store,
            executor=executor,
        )
    )
    return registry, executor


def _artifact_payload(*, kind: str = "artifact") -> dict[str, object]:
    return ArtifactRef(
        kind=kind,
        uri="worker-controlled-path.bin",
        format="bin",
        sha256="1" * 64,
        size_bytes=1,
    ).model_dump(mode="json")


def _managed_operation_entries(cache_root: Path) -> tuple[Path, ...]:
    operation_root = cache_root / "runs" / "demo.worker"
    return tuple(operation_root.iterdir()) if operation_root.exists() else ()


def test_legacy_worker_output_paths_are_rejected_before_managed_run_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "bundle"
    base, _, _ = _write_bundle(root)
    bundle = _worker_result_bundle(
        base,
        result_codec=ResultCodec.LEGACY_RESULT,
        output_kind=CoreKind.RESULT,
    )
    parameters = (
        WorkerParameter(
            name="value",
            kind="keyword_only",
            required=False,
            default=1,
            annotation="int",
        ),
    )
    entry = _admitted_custom_entry(root, bundle, parameters)
    output = tmp_path / "worker-output.txt"
    output.write_text("must not be published", encoding="utf-8")
    payload = {
        "suite": "demo",
        "op": "worker",
        "organelle": "none",
        "status": "ok",
        "output_paths": [str(output)],
        "observed_metrics": {},
        "key_findings": [],
        "flags": [],
        "anomalies": [],
        "summary_text": "worker legacy output",
        "provenance": {
            "operation_version": "1.0",
            "package_version": "1.0.0",
            "git_commit": "worker-spoof",
            "parameters_hash": "f" * 64,
        },
    }
    cache_root = tmp_path / "cache"
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(cache_root))
    registry, _ = _trusted_registry_with_payload(tmp_path, entry, payload)

    with pytest.raises(OrganelleContractError) as captured:
        registry.invoke(entry.capability_id, input=None, parameters={"value": 1})

    assert captured.value.code == "capability.execution_provider_required"
    assert _managed_operation_entries(cache_root) == ()


@pytest.mark.parametrize(
    "result_codec",
    [
        ResultCodec.CANONICAL_JSON,
        ResultCodec.LEGACY_RESULT,
        ResultCodec.JSON_METRIC,
    ],
)
def test_every_accepted_worker_result_codec_uses_one_parent_owned_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    result_codec: ResultCodec,
) -> None:
    root = tmp_path / "bundle"
    base, _, _ = _write_bundle(root)
    result_key = "worker_metric" if result_codec is ResultCodec.JSON_METRIC else None
    binding = base.contract.binding.model_copy(
        update={
            "parameters": (ParameterBindingSpec(name="value", codec=ParameterCodec.PATH),),
            "result_codec": result_codec,
            "result_key": result_key,
        }
    )
    bundle = base.model_copy(
        update={
            "contract": base.contract.model_copy(
                update={
                    "contract_version": "2.0",
                    "binding": binding,
                }
            )
        }
    )
    entry = _entry(root, bundle).model_copy(
        update={
            "status": "admitted",
            "worker_parameters": (
                WorkerParameter(
                    name="value",
                    kind="keyword_only",
                    required=True,
                    default=None,
                    annotation="str",
                ),
            ),
        }
    )
    worker_provenance = ResultProvenance(
        operation_id="demo.worker",
        operation_version="9.9",
        package_version="worker-package",
        git_commit="worker-commit",
        parameters_hash="f" * 64,
        callable_locator="worker.spoof:run",
    )
    worker_object_id: str | None = None
    if result_codec is ResultCodec.CANONICAL_JSON:
        worker_value: object = OrganelleResult(
            operation_id="demo.worker",
            operation_version="9.9",
            scope="none",
            status="ok",
            provenance=worker_provenance,
        ).model_dump(mode="json")
        worker_object_id = cast(dict[str, object], worker_value)["object_id"]  # type: ignore[assignment]
        assert isinstance(worker_object_id, str)
    elif result_codec is ResultCodec.LEGACY_RESULT:
        worker_value = {
            "suite": "demo",
            "op": "worker",
            "organelle": "none",
            "status": "ok",
            "output_paths": [],
            "observed_metrics": {"worker": 1},
            "key_findings": [],
            "flags": [],
            "anomalies": [],
            "summary_text": "legacy worker",
            "provenance": {
                "operation_version": "9.9",
                "package_version": "worker-package",
                "git_commit": "worker-commit",
                "parameters_hash": "f" * 64,
            },
        }
    else:
        worker_value = {"worker": 1}
    executor = _PayloadExecutor(entry, worker_value)
    store = TrustStore(tmp_path / "trust.json")
    assert entry.execution_identity is not None
    trust(entry.capability_id, entry.execution_identity, store=store)
    strategy = WorkerInvocationStrategy(entry, executor, store)
    input_result = OrganelleResult(
        operation_id="input.source",
        scope="none",
        status="ok",
    )
    input_path = tmp_path / "input.txt"
    input_path.write_text("parent-hashed input", encoding="utf-8")
    submitted_parameters = {"value": str(input_path)}
    monkeypatch.setenv("ORG_VERSE_GIT_COMMIT", "parent-commit")

    normalized = strategy.invoke(input_result, submitted_parameters)

    assert isinstance(normalized, OrganelleResult)
    assert normalized.operation_id == "demo.worker"
    assert normalized.operation_version == "2.0"
    assert normalized.provenance is not None
    assert normalized.provenance.operation_id == "demo.worker"
    assert normalized.provenance.operation_version == "2.0"
    assert normalized.provenance.package_version != "worker-package"
    assert normalized.provenance.git_commit == "parent-commit"
    assert normalized.provenance.input_object_ids == (input_result.object_id,)
    assert normalized.provenance.input_artifact_hashes == (
        hashlib.sha256(input_path.read_bytes()).hexdigest(),
    )
    expected_parameter_hash = hashlib.sha256(
        json.dumps(
            submitted_parameters,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    assert normalized.provenance.parameters_hash == expected_parameter_hash
    assert normalized.provenance.callable_locator == "private_worker_pkg.impl:run"
    if worker_object_id is not None:
        assert normalized.object_id != worker_object_id


@pytest.mark.parametrize(
    ("output_kind", "payload"),
    [
        (
            CoreKind.GENOME,
            {
                "schema_version": "organelleverse.genome.v1",
                "kind": "genome",
                "organelle": "mitochondrion",
                "sequence": _artifact_payload(kind="sequence"),
                "source_manifests": [_artifact_payload(kind="manifest")],
            },
        ),
        (
            CoreKind.GENOME,
            {
                "schema_version": "organelleverse.genome.v1",
                "kind": "genome",
                "organelle": "mitochondrion",
                "annotation": _artifact_payload(kind="annotation"),
            },
        ),
        (
            CoreKind.DATA,
            {
                "schema_version": "organelleverse.data.v1",
                "kind": "data",
                "modality": "worker_data",
                "artifacts": {"payload": _artifact_payload()},
            },
        ),
        (
            CoreKind.RESULT,
            {
                "schema_version": "organelleverse.result.v1",
                "kind": "result",
                "operation_id": "demo.worker",
                "scope": "none",
                "status": "ok",
                "artifacts": [_artifact_payload()],
            },
        ),
    ],
)
def test_canonical_worker_core_artifacts_without_staged_bytes_fail_without_residue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    output_kind: CoreKind,
    payload: dict[str, object],
) -> None:
    root = tmp_path / "bundle"
    base, _, _ = _write_bundle(root)
    bundle = _worker_result_bundle(
        base,
        result_codec=ResultCodec.CANONICAL_JSON,
        output_kind=output_kind,
    )
    parameters = (
        WorkerParameter(
            name="value",
            kind="keyword_only",
            required=False,
            default=1,
            annotation="int",
        ),
    )
    entry = _admitted_custom_entry(root, bundle, parameters)
    cache_root = tmp_path / "cache"
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(cache_root))
    registry, _ = _trusted_registry_with_payload(tmp_path, entry, payload)

    with pytest.raises(OrganelleInputError) as captured:
        registry.invoke(entry.capability_id, input=None, parameters={"value": 1})

    assert captured.value.code in {
        "runtime.artifact_tree_mismatch",
        "runtime.duplicate_artifact",
    }
    assert _managed_operation_entries(cache_root) == ()


def test_nested_invalid_worker_parameter_preflight_never_leaks_bare_exceptions(
    tmp_path: Path,
) -> None:
    entry, parameters = _admitted_entry(tmp_path / "bundle")
    valid_payload = {
        "schema_version": "organelleverse.result.v1",
        "kind": "result",
        "operation_id": "demo.worker",
        "scope": "none",
        "status": "ok",
    }
    executor = _PayloadExecutor(entry, valid_payload)
    store = TrustStore(tmp_path / "trust.json")
    assert entry.execution_identity is not None
    trust(entry.capability_id, entry.execution_identity, store=store)
    strategy = WorkerInvocationStrategy(entry, executor, store)
    deeply_nested: object = 0
    for _ in range(2_000):
        deeply_nested = [deeply_nested]
    invalid_values: tuple[object, ...] = (
        {1: "non-string-key"},
        [float("nan")],
        [object()],
        deeply_nested,
    )

    for value in invalid_values:
        with pytest.raises(OrganelleContractError) as captured:
            strategy.invoke(None, cast("Mapping[str, object]", {"value": value}))
        assert captured.value.code == "capability.worker_frame_invalid"

    assert parameters == entry.worker_parameters
    assert executor.invoke_calls == 0


def test_list_describe_and_schema_do_not_spawn_a_worker(tmp_path: Path) -> None:
    entry, parameters = _admitted_entry(tmp_path / "bundle")
    fake = _FakeExecutor(entry, parameters)
    registry = OperationRegistry(
        capability_source=CapabilityIndex(entries=(entry,)).binding_source(executor=fake)
    )

    assert registry.list()[0].operation_id == entry.capability_id
    assert registry.describe(entry.capability_id) == entry.bundle.contract
    assert registry.parameter_schema(entry.capability_id) == entry.parameter_schema
    assert registry.invocation_schema(entry.capability_id)["type"] == "object"
    assert fake.inspect_calls == 0
    assert fake.invoke_calls == 0


def test_removing_trust_after_proxy_caching_blocks_next_dispatch_without_spawn(
    tmp_path: Path,
) -> None:
    entry, parameters = _admitted_entry(tmp_path / "bundle")
    fake = _FakeExecutor(entry, parameters)
    store = TrustStore(tmp_path / "trust.json")
    assert entry.execution_identity is not None
    trust(entry.capability_id, entry.execution_identity, store=store)
    registry = OperationRegistry(
        capability_source=CapabilityIndex(entries=(entry,)).binding_source(
            trust_store=store,
            executor=fake,
        )
    )

    first = registry.invoke(entry.capability_id, input=None, parameters={"value": 3})
    assert isinstance(first, OrganelleResult)
    assert first.metrics["value"] == 3
    cached = registry.require(entry.capability_id)
    assert fake.invoke_calls == 1

    store.path.write_text(
        json.dumps(TrustDocument().model_dump(mode="json", by_alias=True)),
        encoding="utf-8",
    )
    if os.name != "nt":
        store.path.chmod(0o600)
    with pytest.raises(OrganellePermissionError) as captured:
        registry.invoke(entry.capability_id, input=None, parameters={"value": 4})

    assert captured.value.code == "capability.untrusted"
    assert registry.require(entry.capability_id) is cached
    assert fake.invoke_calls == 1


def test_manifest_only_drift_after_proxy_cache_fails_before_spawn_or_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry, parameters = _admitted_entry(tmp_path / "bundle")
    fake = _FakeExecutor(entry, parameters)
    store = TrustStore(tmp_path / "trust.json")
    assert entry.execution_identity is not None
    trust(entry.capability_id, entry.execution_identity, store=store)
    registry = OperationRegistry(
        capability_source=CapabilityIndex(entries=(entry,)).binding_source(
            trust_store=store,
            executor=fake,
        )
    )
    cache = tmp_path / "cache"
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(cache))

    first = registry.invoke(entry.capability_id, input=None, parameters={"value": 3})
    assert isinstance(first, OrganelleResult)
    assert fake.invoke_calls == 1
    cached = registry.require(entry.capability_id)
    before_drift = _managed_operation_entries(cache)
    assert len(before_drift) == 1
    assert not any(path.name.startswith(".staging-") for path in before_drift)
    (entry.bundle_root / "capability.toml").write_text(
        "schema='manifest-only-drift'\n",
        encoding="utf-8",
    )

    with pytest.raises(OrganelleContractError) as captured:
        registry.invoke(entry.capability_id, input=None, parameters={"value": 4})

    assert captured.value.code == "capability.execution_identity_drift"
    assert registry.require(entry.capability_id) is cached
    assert fake.invoke_calls == 1
    assert _managed_operation_entries(cache) == before_drift


def test_malformed_parameters_fail_before_worker_dispatch(tmp_path: Path) -> None:
    entry, parameters = _admitted_entry(tmp_path / "bundle")
    fake = _FakeExecutor(entry, parameters)
    store = TrustStore(tmp_path / "trust.json")
    assert entry.execution_identity is not None
    trust(entry.capability_id, entry.execution_identity, store=store)
    registry = OperationRegistry(
        capability_source=CapabilityIndex(entries=(entry,)).binding_source(
            trust_store=store,
            executor=fake,
        )
    )

    with pytest.raises(OrganelleParameterError):
        registry.invoke(entry.capability_id, input=None, parameters={"value": "not-an-int"})

    assert fake.invoke_calls == 0


def test_concurrent_first_invocation_has_no_parent_import_race(tmp_path: Path) -> None:
    entry, _ = _admitted_entry(tmp_path / "bundle")
    store = TrustStore(tmp_path / "trust.json")
    assert entry.execution_identity is not None
    trust(entry.capability_id, entry.execution_identity, store=store)
    registry = OperationRegistry(
        capability_source=CapabilityIndex(entries=(entry,)).binding_source(trust_store=store)
    )
    _remove_private_modules()

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = tuple(
            pool.map(
                lambda value: registry.invoke(
                    entry.capability_id,
                    input=None,
                    parameters={"value": value},
                ),
                range(4),
            )
        )

    assert [cast(OrganelleResult, result).metrics["value"] for result in results] == [
        10,
        11,
        12,
        13,
    ]
    assert not any(
        name == "private_worker_pkg" or name.startswith("private_worker_pkg.")
        for name in sys.modules
    )


def test_artifact_result_codec_remains_provider_required(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    bundle, _, _ = _write_bundle(root)
    artifact_bundle = bundle.model_copy(
        update={
            "contract": bundle.contract.model_copy(
                update={
                    "binding": bundle.contract.binding.model_copy(
                        update={"result_codec": ResultCodec.ARTIFACT}
                    )
                }
            )
        }
    )
    entry = _entry(root, artifact_bundle)
    inspection = OneShotBundleWorkerExecutor().inspect(entry)

    with pytest.raises(OrganelleContractError) as captured:
        worker_parameter_schema(artifact_bundle, inspection.parameters)

    assert captured.value.code == "capability.execution_provider_required"


def _worker_result_bundle(
    bundle: CapabilityBundle,
    *,
    result_codec: ResultCodec,
    output_kind: CoreKind,
) -> CapabilityBundle:
    stage = "read" if output_kind in {CoreKind.GENOME, CoreKind.DATA} else "analyze"
    result_key = "value" if result_codec is ResultCodec.JSON_METRIC else None
    return bundle.model_copy(
        update={
            "contract": bundle.contract.model_copy(
                update={
                    "stage": stage,
                    "output_kind": output_kind,
                    "output_modalities": ("worker_data",) if output_kind is CoreKind.DATA else (),
                    "binding": bundle.contract.binding.model_copy(
                        update={
                            "result_codec": result_codec,
                            "result_key": result_key,
                        }
                    ),
                }
            )
        }
    )


@pytest.mark.parametrize(
    ("result_codec", "output_kind", "accepted"),
    [
        (ResultCodec.JSON_METRIC, CoreKind.RESULT, True),
        (ResultCodec.LEGACY_RESULT, CoreKind.RESULT, True),
        (ResultCodec.CANONICAL_JSON, CoreKind.GENOME, True),
        (ResultCodec.CANONICAL_JSON, CoreKind.DATA, True),
        (ResultCodec.CANONICAL_JSON, CoreKind.RESULT, True),
        (ResultCodec.JSON_METRIC, CoreKind.GENOME, False),
        (ResultCodec.LEGACY_RESULT, CoreKind.DATA, False),
        (ResultCodec.CANONICAL, CoreKind.RESULT, False),
        (ResultCodec.ARTIFACT, CoreKind.RESULT, False),
    ],
)
def test_worker_result_codec_output_kind_matrix_fails_closed_before_fixture_binding(
    tmp_path: Path,
    result_codec: ResultCodec,
    output_kind: CoreKind,
    accepted: bool,
) -> None:
    root = tmp_path / "bundle"
    base, _, _ = _write_bundle(root)
    bundle = _worker_result_bundle(
        base,
        result_codec=result_codec,
        output_kind=output_kind,
    )
    parameters = (
        WorkerParameter(
            name="value",
            kind="keyword_only",
            required=False,
            default=1,
            annotation="int",
        ),
    )

    if accepted:
        assert worker_parameter_schema(bundle, parameters)["type"] == "object"
    else:
        with pytest.raises(OrganelleContractError) as captured:
            worker_parameter_schema(bundle, parameters)
        assert captured.value.code == "capability.execution_provider_required"


@pytest.mark.parametrize(
    "parameters",
    [
        (
            WorkerParameter(
                name="prefix",
                kind="positional_only",
                required=False,
                default=1,
                annotation="int",
            ),
            WorkerParameter(
                name="value",
                kind="positional_only",
                required=False,
                default=2,
                annotation="int",
            ),
        ),
        (
            WorkerParameter(
                name="prefix",
                kind="positional_only",
                required=False,
                default=1,
                annotation="int",
            ),
            WorkerParameter(
                name="middle",
                kind="positional_only",
                required=False,
                default=2,
                annotation="int",
            ),
            WorkerParameter(
                name="value",
                kind="positional_only",
                required=True,
                default=None,
                annotation="int",
            ),
        ),
    ],
)
def test_targeted_positional_only_parameter_is_rejected_before_verification(
    tmp_path: Path,
    parameters: tuple[WorkerParameter, ...],
) -> None:
    root = tmp_path / "bundle"
    bundle, _, _ = _write_bundle(root)

    with pytest.raises(OrganelleContractError) as captured:
        worker_parameter_schema(bundle, parameters)

    assert captured.value.code == "capability.execution_provider_required"


def test_untargeted_required_positional_only_parameter_remains_binding_invalid(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bundle"
    bundle, _, _ = _write_bundle(root)
    parameters = (
        WorkerParameter(
            name="prefix",
            kind="positional_only",
            required=True,
            default=None,
            annotation="int",
        ),
        WorkerParameter(
            name="value",
            kind="keyword_only",
            required=False,
            default=1,
            annotation="int",
        ),
    )

    with pytest.raises(OrganelleContractError) as captured:
        worker_parameter_schema(bundle, parameters)

    assert captured.value.code == "capability.binding_invalid"


def test_parameterized_container_annotation_is_rejected_instead_of_broadening_schema(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bundle"
    bundle, _, _ = _write_bundle(
        root,
        source=_result_source(annotation="list[int]", value_expression="len(value)"),
    )
    entry = _entry(root, bundle)

    inspection = OneShotBundleWorkerExecutor().inspect(entry)

    assert inspection.parameters[0].annotation == "unsupported"
    with pytest.raises(OrganelleContractError) as captured:
        worker_parameter_schema(bundle, inspection.parameters)
    assert captured.value.code == "capability.execution_provider_required"


def test_worker_contract_envelopes_are_frozen_and_forbid_extra_fields(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bundle"
    bundle, _, _ = _write_bundle(root)
    request = _request(_entry(root, bundle), tmp_path)

    with pytest.raises(ValidationError):
        request.operation_id = "other.worker"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        WorkerRequest.model_validate({**request.model_dump(mode="python"), "unexpected": True})


def test_stdlib_child_rejects_extra_request_envelope_fields(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    bundle, _, _ = _write_bundle(root)
    request = _request(_entry(root, bundle), tmp_path)
    payload = request.model_dump(mode="json")
    payload["unexpected"] = True

    with pytest.raises(_WorkerFailure, match="closed protocol envelope"):
        _require_request(payload)


def test_frame_round_trip_is_canonical_and_finite() -> None:
    first = _encode_frame({"z": 1, "a": [True, None, 2.5]})
    second = _encode_frame({"a": [True, None, 2.5], "z": 1})

    assert first == second
    assert _decode_frame(first) == {"a": [True, None, 2.5], "z": 1}


def _plugin_bundle() -> CapabilityBundle:
    """A v1-envelope bundle whose binding uses the plugin protocol."""

    return CapabilityBundle.model_validate(
        {
            "schema": "organelleverse.capability.v1",
            "capability": {
                "id": "demo.worker",
                "bundle_version": "1.0.0",
                "implementation": "native",
            },
            "contract": {
                "operation_id": "demo.worker",
                "contract_version": "1.0",
                "title": "Controlled plugin worker fixture",
                "description": "A bundle-local plugin-protocol fixture for inspection tests.",
                "keywords": ("controlled", "fixture", "plugin"),
                "execution_mode": "inline",
                "stage": "analyze",
                "input_kind": "none",
                "output_kind": "result",
                "callable_locator": "private_worker_pkg.impl:run",
                "binding": {"argument_mode": "plugin_protocol"},
            },
        }
    )


def test_worker_inspection_accepts_the_standard_plugin_signature(tmp_path: Path) -> None:
    source = (
        "from organelleverse.plugin_protocol import PluginContext, PluginResult\n"
        "def run(inputs: dict, outputs: dict, parameters: dict,\n"
        "        context: PluginContext) -> PluginResult:\n"
        "    return PluginResult()\n"
    )
    root = tmp_path / "bundle-plugin"
    _write_bundle(root, source=source)

    inspection = OneShotBundleWorkerExecutor().inspect(_entry(root, _plugin_bundle()))

    assert [(parameter.name, parameter.annotation) for parameter in inspection.parameters] == [
        ("inputs", "dict"),
        ("outputs", "dict"),
        ("parameters", "dict"),
        ("context", "plugin_context"),
    ]


def test_worker_inspection_rejects_a_drifting_plugin_signature(tmp_path: Path) -> None:
    source = "def run(inputs: dict, parameters: dict):\n    return {}\n"
    root = tmp_path / "bundle-plugin-drift"
    _write_bundle(root, source=source)

    with pytest.raises(OrganelleContractError) as captured:
        OneShotBundleWorkerExecutor().inspect(_entry(root, _plugin_bundle()))

    assert captured.value.code == "capability.plugin_signature_invalid"
