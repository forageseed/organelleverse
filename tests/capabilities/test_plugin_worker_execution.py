"""Worker execution contract for v2 scientific plugins."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
from pydantic import ValidationError

from organelleverse.capabilities.code_identity import inspect_bundle_code
from organelleverse.capabilities.hashing import hash_bundle
from organelleverse.capabilities.index import CapabilityEntry, CapabilityOrigin
from organelleverse.capabilities.models import PluginCapabilityBundle
from organelleverse.capabilities.worker import OneShotBundleWorkerExecutor
from organelleverse.core.errors import OrganelleContractError, OrganelleInputError
from organelleverse.core.result import OrganelleResult
from organelleverse.operations.python_binding import bind_worker_capability, worker_parameter_schema
from organelleverse.operations.spec import (
    ExecutionContext,
    InlineParameterValue,
    PathParameterValue,
)
from tests.capabilities.test_plugin_v2_models import v2_payload

if TYPE_CHECKING:
    from organelleverse.capabilities.worker import WorkerInvocationStrategy

_PLUGIN = """\
from pathlib import Path

from organelleverse.plugin_protocol import PluginContext, PluginResult


def run(inputs: dict, outputs: dict, parameters: dict, context: PluginContext) -> PluginResult:
    image_count = len(list(Path(inputs[\"images\"]).glob(\"*.tif\")))
    masks = Path(outputs[\"masks\"])
    (masks / \"mask.txt\").write_text(\"mask\", encoding=\"utf-8\")
    report = Path(outputs[\"report\"])
    report.write_text(str(image_count), encoding=\"utf-8\")
    return PluginResult(
        summary={\"image_count\": image_count, \"threshold\": parameters[\"threshold\"]},
        outputs={\"masks\": str(masks), \"report\": str(report)},
    )
"""

_MISSING_OUTPUT_PLUGIN = """\
from organelleverse.plugin_protocol import PluginContext, PluginResult


def run(inputs: dict, outputs: dict, parameters: dict, context: PluginContext) -> PluginResult:
    return PluginResult(summary={}, outputs={})
"""

_EMPTY_DIRECTORY_PLUGIN = """\
from pathlib import Path

from organelleverse.plugin_protocol import PluginContext, PluginResult


def run(inputs: dict, outputs: dict, parameters: dict, context: PluginContext) -> PluginResult:
    report = Path(outputs[\"report\"])
    report.write_text(\"done\", encoding=\"utf-8\")
    return PluginResult(
        summary={},
        outputs={\"masks\": outputs[\"masks\"], \"report\": str(report)},
    )
"""

_MUTATING_INPUT_PLUGIN = """\
from pathlib import Path

from organelleverse.plugin_protocol import PluginContext, PluginResult


def run(inputs: dict, outputs: dict, parameters: dict, context: PluginContext) -> PluginResult:
    Path(inputs[\"images\"]).joinpath(\"worker-only.txt\").write_text(\"changed\", encoding=\"utf-8\")
    masks = Path(outputs[\"masks\"])
    (masks / \"mask.txt\").write_text(\"mask\", encoding=\"utf-8\")
    report = Path(outputs[\"report\"])
    report.write_text(\"done\", encoding=\"utf-8\")
    return PluginResult(summary={}, outputs={\"masks\": str(masks), \"report\": str(report)})
"""

_LOGGING_PLUGIN = """\
from pathlib import Path

from organelleverse.plugin_protocol import PluginContext, PluginResult


def run(inputs: dict, outputs: dict, parameters: dict, context: PluginContext) -> PluginResult:
    masks = Path(outputs[\"masks\"])
    (masks / \"mask.txt\").write_text(\"mask\", encoding=\"utf-8\")
    report = Path(outputs[\"report\"])
    report.write_text(\"done\", encoding=\"utf-8\")
    log = context.work_dir / \"plugin.log\"
    log.write_text(\"completed\", encoding=\"utf-8\")
    return PluginResult(
        summary={},
        outputs={\"masks\": str(masks), \"report\": str(report)},
        log_paths=(str(log),),
    )
"""

_FAILING_WITH_LOG_PLUGIN = """\
from pathlib import Path

from organelleverse.plugin_protocol import PluginContext, PluginResult


def run(inputs: dict, outputs: dict, parameters: dict, context: PluginContext) -> PluginResult:
    assert context.log_path is not None
    context.log_path.write_text("segmentation failed", encoding="utf-8")
    raise RuntimeError("synthetic plugin failure")
"""

_MUTATED_RESULT_PLUGIN = """\
from pathlib import Path

from organelleverse.plugin_protocol import PluginContext, PluginResult


def run(inputs: dict, outputs: dict, parameters: dict, context: PluginContext) -> PluginResult:
    masks = Path(outputs["masks"])
    (masks / "mask.txt").write_text("mask", encoding="utf-8")
    report = Path(outputs["report"])
    report.write_text("done", encoding="utf-8")
    result = PluginResult(
        summary={},
        outputs={"masks": str(masks), "report": str(report)},
    )
    result.summary["invalid"] = object()
    return result
"""

_COPY_DIRECTORY_INPUT_PLUGIN = """\
from pathlib import Path
from shutil import copyfile

from organelleverse.plugin_protocol import PluginContext, PluginResult


def run(inputs: dict, outputs: dict, parameters: dict, context: PluginContext) -> PluginResult:
    masks = Path(outputs["masks"])
    (masks / "mask.txt").write_text("mask", encoding="utf-8")
    report = Path(outputs["report"])
    copyfile(Path(inputs["images"]) / "one.tif", report)
    return PluginResult(summary={}, outputs={"masks": str(masks), "report": str(report)})
"""


def _scored_plugin_source(score_body: str, *, summary: str = '{"quality": 0.875}') -> str:
    """A v2 plugin whose declared score callable has the given body."""

    return f"""\
from pathlib import Path

from organelleverse.plugin_protocol import PluginContext, PluginResult


def run(inputs: dict, outputs: dict, parameters: dict, context: PluginContext) -> PluginResult:
    masks = Path(outputs["masks"])
    (masks / "mask.txt").write_text("mask", encoding="utf-8")
    report = Path(outputs["report"])
    report.write_text("done", encoding="utf-8")
    return PluginResult(
        summary={summary},
        outputs={{"masks": str(masks), "report": str(report)}},
    )


def score(result: PluginResult) -> float:
{score_body}
"""


_SCORED_PLUGIN = _scored_plugin_source('    return float(result.summary["quality"])')
_NAN_SCORE_PLUGIN = _scored_plugin_source("    return float('nan')")
_STRING_SCORE_PLUGIN = _scored_plugin_source("    return 'high'")
_RESERVED_SUMMARY_PLUGIN = _scored_plugin_source(
    '    return float(result.summary["quality"])',
    summary='{"quality": 0.5, "plugin_optimization_score": 0.1}',
)
_UNDECLARED_RESERVED_SUMMARY_PLUGIN = _scored_plugin_source(
    '    return float(result.summary["quality"])',
    summary='{"quality": 0.5, "plugin_optimization_score": 0.1}',
)
_HUGE_INTEGER_SCORE_PLUGIN = _scored_plugin_source("    return 10**10000")
_RAISING_SCORE_PLUGIN = _scored_plugin_source("    raise RuntimeError('scorer boom')")

_MISSING_SCORE_PLUGIN = """\
from pathlib import Path

from organelleverse.plugin_protocol import PluginContext, PluginResult


def run(inputs: dict, outputs: dict, parameters: dict, context: PluginContext) -> PluginResult:
    masks = Path(outputs["masks"])
    (masks / "mask.txt").write_text("mask", encoding="utf-8")
    report = Path(outputs["report"])
    report.write_text("done", encoding="utf-8")
    return PluginResult(summary={}, outputs={"masks": str(masks), "report": str(report)})
"""

_NON_CALLABLE_SCORE_PLUGIN = _MISSING_SCORE_PLUGIN + "\nscore = 42\n"

_TWO_PARAMETER_SCORE_PLUGIN = _MISSING_SCORE_PLUGIN + """

def score(result: PluginResult, extra: int) -> float:
    return 1.0
"""

_WRONG_ANNOTATION_SCORE_PLUGIN = _MISSING_SCORE_PLUGIN + """

def score(result: dict) -> float:
    return 1.0
"""

_POSITIONAL_ONLY_SCORE_PLUGIN = _MISSING_SCORE_PLUGIN + """

def score(result: PluginResult, /) -> float:
    return 1.0
"""

_STRING_RETURN_SCORE_PLUGIN = _MISSING_SCORE_PLUGIN + """

def score(result: PluginResult) -> str:
    return "1.0"
"""

_MISSING_RETURN_SCORE_PLUGIN = _MISSING_SCORE_PLUGIN + """

def score(result: PluginResult):
    return 1.0
"""

_POSTPONED_FLOAT_SCORE_PLUGIN = "from __future__ import annotations\n" + _MISSING_SCORE_PLUGIN + """

def score(result: PluginResult) -> float:
    return 1.0
"""

_EXPLODING_SCORE_PLUGIN = _MISSING_SCORE_PLUGIN + """

def score(result: PluginResult) -> float:
    raise AssertionError("a plugin without optimization must never be scored")
"""


def _plugin_entry(
    root: Path,
    source: str = _PLUGIN,
    *,
    scored: bool = False,
    inline_source: bool = False,
    inline_threshold: bool = False,
    source_accepts: tuple[str, ...] = ("path", "inline"),
    threshold_schema: dict[str, object] | None = None,
) -> CapabilityEntry:
    payload = v2_payload()
    capability = cast(dict[str, object], payload["capability"])
    capability["id"] = "demo.plugin"
    contract = cast(dict[str, object], payload["contract"])
    contract["operation_id"] = "demo.plugin"
    contract["callable_locator"] = "demo_plugin.implementation:run"
    contract.pop("optimization")
    if scored:
        contract["optimization"] = {
            "score_locator": "demo_plugin.implementation:score",
            "parameters": ["threshold"],
            "max_trials": 3,
            "parallelism": 1,
        }
    binding = cast(dict[str, object], contract["binding"])
    binding["parameters"] = [
        {
            "name": "images",
            "codec": "directory",
            "path_role": "input",
            "json_schema": {"type": "string"},
        },
        {
            "name": "masks_dir",
            "codec": "directory",
            "path_role": "output",
            "json_schema": {"type": "string"},
        },
        {
            "name": "report_path",
            "codec": "path",
            "path_role": "output",
            "json_schema": {"type": "string"},
        },
        {
            "name": "threshold",
            "codec": "json",
            "json_schema": threshold_schema or {"type": "number", "default": 0.5},
            **({"accepts": ["inline"]} if inline_threshold else {}),
        },
    ]
    if inline_source:
        cast(list[dict[str, object]], binding["parameters"]).append(
            {
                "name": "source_file",
                "codec": "path",
                "path_role": "input",
                "accepts": list(source_accepts),
                "inline_max_bytes": 1024,
                "json_schema": {"type": "string"},
            }
        )
    contract["outputs"] = [
        {"name": "masks", "kind": "directory", "parameter": "masks_dir", "description": "Masks."},
        {"name": "report", "kind": "file", "parameter": "report_path", "description": "Report."},
    ]
    bundle = PluginCapabilityBundle.model_validate(payload)
    package = root / "code" / "demo_plugin"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "implementation.py").write_text(source, encoding="utf-8")
    (root / "capability.toml").write_text("schema = 'fixture'\n", encoding="utf-8")
    content_hash = hash_bundle(root)
    identity = inspect_bundle_code(
        root,
        capability_id=bundle.capability.id,
        bundle_content_hash=content_hash,
        callable_locator=cast(str, bundle.contract.callable_locator),
    )
    entry = CapabilityEntry(
        capability_id=bundle.capability.id,
        content_hash=content_hash,
        bundle_root=root,
        bundle=bundle,
        origins=(
            CapabilityOrigin(channel="local", source_path=str(root / "capability.toml"), search_root=str(root)),
        ),
        execution_identity=identity,
    )
    inspection = OneShotBundleWorkerExecutor().inspect(entry)
    return entry.model_copy(update={"worker_parameters": inspection.parameters})


def test_worker_runs_plugin_protocol_with_managed_named_outputs(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    (images / "one.tif").write_bytes(b"image")
    staging = tmp_path / "staging"
    staging.mkdir()

    result = OneShotBundleWorkerExecutor().invoke(
        _plugin_entry(tmp_path / "bundle"),
        input=None,
        parameters={"images": str(images), "threshold": 0.75},
        run_id="plugin-run",
        staging_root=staging,
    )

    value = cast(dict[str, object], json.loads(json.dumps(result.value)))
    assert value["summary"] == {"image_count": 1, "threshold": 0.75}
    outputs = cast(dict[str, str], value["outputs"])
    assert (staging / outputs["masks"] / "mask.txt").is_file()
    assert (staging / outputs["report"]).is_file()


def test_plugin_worker_schema_comes_from_declared_ports(tmp_path: Path) -> None:
    entry = _plugin_entry(tmp_path / "bundle")
    assert entry.worker_parameters is not None

    schema = worker_parameter_schema(entry.bundle, entry.worker_parameters)

    assert set(cast(dict[str, object], schema["properties"])) == {"images", "threshold"}
    assert schema["required"] == ["images"]


def test_plugin_parameter_model_enforces_declared_json_schemas(tmp_path: Path) -> None:
    from organelleverse.capabilities.trust import TrustStore
    from organelleverse.capabilities.worker import WorkerInvocationStrategy

    entry = _plugin_entry(tmp_path / "bundle")
    assert entry.worker_parameters is not None
    schema = worker_parameter_schema(entry.bundle, entry.worker_parameters)
    from jsonschema import Draft202012Validator

    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(  # pyright: ignore[reportUnknownMemberType]
        {"images": str(tmp_path), "threshold": 0.75}
    )
    bound = bind_worker_capability(
        entry.bundle,
        entry.worker_parameters,
        schema,
        invocation_strategy=WorkerInvocationStrategy(
            entry,
            OneShotBundleWorkerExecutor(),
            TrustStore(tmp_path / "trust.json"),
        ),
    )

    with pytest.raises(ValidationError):
        bound.signature.parameter_model.model_validate({"images": 7, "threshold": "wrong"})


def test_worker_rejects_a_plugin_missing_declared_output(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()

    with pytest.raises(OrganelleContractError) as captured:
        OneShotBundleWorkerExecutor().invoke(
            _plugin_entry(tmp_path / "bundle", _MISSING_OUTPUT_PLUGIN),
            input=None,
            parameters={"images": str(tmp_path), "threshold": 0.75},
            run_id="plugin-run",
            staging_root=staging,
        )

    assert captured.value.code == "capability.plugin_result_invalid"


def test_worker_rejects_an_empty_declared_directory_output(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()

    with pytest.raises(OrganelleContractError) as captured:
        OneShotBundleWorkerExecutor().invoke(
            _plugin_entry(tmp_path / "bundle", _EMPTY_DIRECTORY_PLUGIN),
            input=None,
            parameters={"images": str(tmp_path), "threshold": 0.75},
            run_id="plugin-run",
            staging_root=staging,
        )

    assert captured.value.code == "capability.plugin_result_invalid"


def test_plugin_strategy_publishes_declared_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from organelleverse.capabilities.trust import TrustStore, trust
    from organelleverse.capabilities.worker import WorkerInvocationStrategy

    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    images = tmp_path / "images"
    images.mkdir()
    (images / "one.tif").write_bytes(b"image")
    entry = _plugin_entry(tmp_path / "bundle")
    assert entry.execution_identity is not None
    trust(entry.capability_id, entry.execution_identity, store=TrustStore(tmp_path / "trust.json"))

    result = WorkerInvocationStrategy(
        entry,
        OneShotBundleWorkerExecutor(),
        TrustStore(tmp_path / "trust.json"),
    ).invoke(None, {"images": str(images), "threshold": 0.75})

    assert isinstance(result, OrganelleResult)
    assert result.metrics["image_count"] == 1
    assert {artifact.kind for artifact in result.artifacts} == {"masks", "report"}
    assert result.provenance is not None
    assert len(result.provenance.input_artifact_hashes) == 1


def test_inline_path_enters_provenance_and_publication_trust_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from organelleverse.capabilities.trust import TrustStore, trust
    from organelleverse.capabilities.worker import WorkerInvocationStrategy

    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    images = tmp_path / "images"
    images.mkdir()
    (images / "one.tif").write_bytes(b"image")
    content = ">sample\nACGT\n"
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    inline = InlineParameterValue(
        value=content,
        sha256=content_hash,
        format="fasta",
        encoding="text",
    )
    entry = _plugin_entry(tmp_path / "bundle", inline_source=True)
    store = TrustStore(tmp_path / "trust.json")
    assert entry.execution_identity is not None
    trust(entry.capability_id, entry.execution_identity, store=store)

    assert entry.worker_parameters is not None
    schema = worker_parameter_schema(entry.bundle, entry.worker_parameters)
    bound = bind_worker_capability(
        entry.bundle,
        entry.worker_parameters,
        schema,
        invocation_strategy=WorkerInvocationStrategy(
            entry, OneShotBundleWorkerExecutor(), store
        ),
    )
    result = bound.invoke(
        None,
        {
            "images": str(images),
            "source_file": inline.model_dump(mode="json"),
            "threshold": 0.75,
        },
    )

    assert isinstance(result, OrganelleResult)
    assert result.provenance is not None
    assert content_hash in result.provenance.input_artifact_hashes
    assert result.provenance.inline_materializations[0].sha256 == content_hash


def test_inline_only_parameter_rejects_path_wrapper(tmp_path: Path) -> None:
    from organelleverse.capabilities.trust import TrustStore, trust
    from organelleverse.capabilities.worker import WorkerInvocationStrategy

    source_file = tmp_path / "source.fasta"
    source_file.write_text(">sample\nACGT\n", encoding="utf-8")
    entry = _plugin_entry(
        tmp_path / "bundle",
        inline_source=True,
        source_accepts=("inline",),
    )
    store = TrustStore(tmp_path / "trust.json")
    assert entry.execution_identity is not None
    trust(entry.capability_id, entry.execution_identity, store=store)

    with pytest.raises(OrganelleContractError) as captured:
        WorkerInvocationStrategy(entry, OneShotBundleWorkerExecutor(), store).invoke(
            None,
            {
                "images": str(tmp_path),
                "source_file": PathParameterValue(
                    value=str(source_file), context=ExecutionContext.HOST
                ),
                "threshold": 0.75,
            },
        )

    assert captured.value.code == "capability.path_not_accepted"


@pytest.mark.parametrize(
    ("payload", "expected_error"),
    [("0.75", None), ('"not-a-number"', "capability.parameter_schema_invalid")],
)
def test_bound_plugin_decodes_inline_json_before_schema_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload: str,
    expected_error: str | None,
) -> None:
    from organelleverse.capabilities.trust import TrustStore, trust
    from organelleverse.capabilities.worker import WorkerInvocationStrategy

    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    images = tmp_path / "images"
    images.mkdir()
    (images / "one.tif").write_bytes(b"image")
    inline = InlineParameterValue(
        value=payload,
        sha256=hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        format="json",
        encoding="text",
    )
    entry = _plugin_entry(tmp_path / "bundle", inline_threshold=True)
    store = TrustStore(tmp_path / "trust.json")
    assert entry.execution_identity is not None
    assert entry.worker_parameters is not None
    trust(entry.capability_id, entry.execution_identity, store=store)
    schema = worker_parameter_schema(entry.bundle, entry.worker_parameters)
    bound = bind_worker_capability(
        entry.bundle,
        entry.worker_parameters,
        schema,
        invocation_strategy=WorkerInvocationStrategy(
            entry, OneShotBundleWorkerExecutor(), store
        ),
    )

    if expected_error is not None:
        with pytest.raises(OrganelleContractError) as captured:
            bound.invoke(
                None,
                {"images": str(images), "threshold": inline.model_dump(mode="json")},
            )
        assert captured.value.code == expected_error
    else:
        result = bound.invoke(
            None,
            {"images": str(images), "threshold": inline.model_dump(mode="json")},
        )
        assert isinstance(result, OrganelleResult)
        assert result.metrics["threshold"] == 0.75
        assert result.provenance is not None
        assert inline.sha256 in result.provenance.input_artifact_hashes


def test_json_parameter_rejects_path_wrapper(tmp_path: Path) -> None:
    from organelleverse.capabilities.trust import TrustStore, trust
    from organelleverse.capabilities.worker import WorkerInvocationStrategy

    source_file = tmp_path / "threshold.txt"
    source_file.write_text("0.75", encoding="utf-8")
    entry = _plugin_entry(tmp_path / "bundle")
    store = TrustStore(tmp_path / "trust.json")
    assert entry.execution_identity is not None
    trust(entry.capability_id, entry.execution_identity, store=store)

    with pytest.raises(OrganelleContractError) as captured:
        WorkerInvocationStrategy(entry, OneShotBundleWorkerExecutor(), store).invoke(
            None,
            {
                "images": str(tmp_path),
                "threshold": PathParameterValue(
                    value=str(source_file), context=ExecutionContext.HOST
                ),
            },
        )

    assert captured.value.code == "capability.path_not_accepted"


def test_scientific_json_kind_field_is_not_hijacked_as_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from organelleverse.capabilities.trust import TrustStore, trust
    from organelleverse.capabilities.worker import WorkerInvocationStrategy

    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    images = _images_dir(tmp_path)
    schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "kind": {"const": "inline"},
            "score": {"type": "number"},
        },
        "required": ["kind", "score"],
        "additionalProperties": False,
    }
    entry = _plugin_entry(tmp_path / "bundle", threshold_schema=schema)
    store = TrustStore(tmp_path / "trust.json")
    assert entry.execution_identity is not None
    assert entry.worker_parameters is not None
    trust(entry.capability_id, entry.execution_identity, store=store)
    bound = bind_worker_capability(
        entry.bundle,
        entry.worker_parameters,
        worker_parameter_schema(entry.bundle, entry.worker_parameters),
        invocation_strategy=WorkerInvocationStrategy(
            entry, OneShotBundleWorkerExecutor(), store
        ),
    )
    payload = {"kind": "inline", "score": 0.75}

    result = bound.invoke(None, {"images": str(images), "threshold": payload})

    assert isinstance(result, OrganelleResult)
    assert result.metrics["threshold"] == payload


def test_plugin_directory_inputs_are_snapshotted_before_worker_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from organelleverse.capabilities.trust import TrustStore, trust
    from organelleverse.capabilities.worker import WorkerInvocationStrategy

    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    images = tmp_path / "images"
    images.mkdir()
    (images / "one.tif").write_bytes(b"image")
    entry = _plugin_entry(tmp_path / "bundle", _MUTATING_INPUT_PLUGIN)
    assert entry.execution_identity is not None
    trust(entry.capability_id, entry.execution_identity, store=TrustStore(tmp_path / "trust.json"))

    WorkerInvocationStrategy(
        entry,
        OneShotBundleWorkerExecutor(),
        TrustStore(tmp_path / "trust.json"),
    ).invoke(None, {"images": str(images), "threshold": 0.75})

    assert not (images / "worker-only.txt").exists()


def test_plugin_log_paths_are_published_as_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from organelleverse.capabilities.trust import TrustStore, trust
    from organelleverse.capabilities.worker import WorkerInvocationStrategy

    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    images = tmp_path / "images"
    images.mkdir()
    (images / "one.tif").write_bytes(b"image")
    entry = _plugin_entry(tmp_path / "bundle", _LOGGING_PLUGIN)
    assert entry.execution_identity is not None
    trust(entry.capability_id, entry.execution_identity, store=TrustStore(tmp_path / "trust.json"))

    result = WorkerInvocationStrategy(
        entry,
        OneShotBundleWorkerExecutor(),
        TrustStore(tmp_path / "trust.json"),
    ).invoke(None, {"images": str(images), "threshold": 0.75})

    assert isinstance(result, OrganelleResult)
    assert {artifact.kind for artifact in result.artifacts} == {"masks", "report", "log"}


def test_plugin_exception_publishes_its_managed_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from organelleverse.capabilities.trust import TrustStore, trust
    from organelleverse.capabilities.worker import WorkerInvocationStrategy

    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    images = tmp_path / "images"
    images.mkdir()
    (images / "one.tif").write_bytes(b"image")
    entry = _plugin_entry(tmp_path / "bundle", _FAILING_WITH_LOG_PLUGIN)
    assert entry.execution_identity is not None
    trust(entry.capability_id, entry.execution_identity, store=TrustStore(tmp_path / "trust.json"))

    result = WorkerInvocationStrategy(
        entry,
        OneShotBundleWorkerExecutor(),
        TrustStore(tmp_path / "trust.json"),
    ).invoke(None, {"images": str(images), "threshold": 0.75})

    assert isinstance(result, OrganelleResult)
    assert result.status == "failed"
    assert result.errors[0].code == "capability.plugin_execution_failed"
    log = next(artifact for artifact in result.artifacts if artifact.kind == "log")
    assert Path(log.uri).read_text(encoding="utf-8") == "segmentation failed"


def test_mutated_plugin_result_remains_a_contract_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from organelleverse.capabilities.trust import TrustStore, trust
    from organelleverse.capabilities.worker import WorkerInvocationStrategy

    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    images = tmp_path / "images"
    images.mkdir()
    (images / "one.tif").write_bytes(b"image")
    entry = _plugin_entry(tmp_path / "bundle", _MUTATED_RESULT_PLUGIN)
    assert entry.execution_identity is not None
    trust(entry.capability_id, entry.execution_identity, store=TrustStore(tmp_path / "trust.json"))

    with pytest.raises(OrganelleContractError) as captured:
        WorkerInvocationStrategy(
            entry,
            OneShotBundleWorkerExecutor(),
            TrustStore(tmp_path / "trust.json"),
        ).invoke(None, {"images": str(images), "threshold": 0.75})

    assert captured.value.code == "capability.plugin_result_invalid"


def test_plugin_cannot_reemit_a_directory_input_member(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from organelleverse.capabilities.trust import TrustStore, trust
    from organelleverse.capabilities.worker import WorkerInvocationStrategy

    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    images = _images_dir(tmp_path)
    entry = _plugin_entry(tmp_path / "bundle", _COPY_DIRECTORY_INPUT_PLUGIN)
    store = TrustStore(tmp_path / "trust.json")
    assert entry.execution_identity is not None
    trust(entry.capability_id, entry.execution_identity, store=store)

    with pytest.raises(OrganelleInputError) as captured:
        WorkerInvocationStrategy(entry, OneShotBundleWorkerExecutor(), store).invoke(
            None, {"images": str(images), "threshold": 0.75}
        )

    assert captured.value.code == "runtime.input_artifact_reemission"


# === declared optimization score path (Plugin-04, Task 1) ===================

_IMAGES_PARAMETERS = {"images": None, "threshold": 0.75}  # images filled per-test


def _score_strategy(entry: CapabilityEntry, tmp_path: Path) -> WorkerInvocationStrategy:
    from organelleverse.capabilities.trust import TrustStore, trust
    from organelleverse.capabilities.worker import WorkerInvocationStrategy

    assert entry.execution_identity is not None
    trust(entry.capability_id, entry.execution_identity, store=TrustStore(tmp_path / "trust.json"))
    return WorkerInvocationStrategy(
        entry,
        OneShotBundleWorkerExecutor(),
        TrustStore(tmp_path / "trust.json"),
    )


def _images_dir(tmp_path: Path) -> Path:
    images = tmp_path / "images"
    images.mkdir()
    (images / "one.tif").write_bytes(b"image")
    return images


def test_declared_score_reaches_l6_metrics_through_the_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from organelleverse.capabilities.index import CapabilityIndex, CapabilityStatus
    from organelleverse.capabilities.trust import TrustStore, trust
    from organelleverse.operations.registry import OperationRegistry

    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    images = _images_dir(tmp_path)
    entry = _plugin_entry(tmp_path / "bundle", _SCORED_PLUGIN, scored=True)
    assert entry.execution_identity is not None
    assert entry.worker_parameters is not None
    admitted = entry.model_copy(
        update={
            "status": CapabilityStatus.ADMITTED,
            "parameter_schema": worker_parameter_schema(entry.bundle, entry.worker_parameters),
        }
    )
    store = TrustStore(tmp_path / "trust.json")
    trust(entry.capability_id, entry.execution_identity, store=store)
    registry = OperationRegistry(
        capability_source=CapabilityIndex(entries=(admitted,)).binding_source(
            trust_store=store,
            executor=OneShotBundleWorkerExecutor(),
        )
    )

    result = registry.require("demo.plugin").invoke(
        None, {"images": str(images), "threshold": 0.75}
    )

    assert isinstance(result, OrganelleResult)
    assert result.status == "ok"
    assert result.metrics["plugin_optimization_score"] == 0.875
    assert result.metrics["quality"] == 0.875


def test_plugin_without_optimization_never_touches_a_score_callable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    images = _images_dir(tmp_path)
    # The module defines a score callable that explodes if imported for scoring;
    # with no optimization declaration the run must succeed untouched.
    entry = _plugin_entry(tmp_path / "bundle", _EXPLODING_SCORE_PLUGIN)
    strategy = _score_strategy(entry, tmp_path)

    result = strategy.invoke(None, {"images": str(images), "threshold": 0.75})

    assert isinstance(result, OrganelleResult)
    assert result.status == "ok"
    assert "plugin_optimization_score" not in result.metrics


def test_worker_json_carries_null_score_without_optimization(tmp_path: Path) -> None:
    images = _images_dir(tmp_path)
    staging = tmp_path / "staging"
    staging.mkdir()

    result = OneShotBundleWorkerExecutor().invoke(
        _plugin_entry(tmp_path / "bundle", _EXPLODING_SCORE_PLUGIN),
        input=None,
        parameters={"images": str(images), "threshold": 0.75},
        run_id="plugin-run",
        staging_root=staging,
    )

    value = cast(dict[str, object], result.value)
    assert "score" in value
    assert value["score"] is None


def test_non_finite_score_is_a_contract_error(tmp_path: Path) -> None:
    images = _images_dir(tmp_path)
    staging = tmp_path / "staging"
    staging.mkdir()

    with pytest.raises(OrganelleContractError) as captured:
        OneShotBundleWorkerExecutor().invoke(
            _plugin_entry(tmp_path / "bundle", _NAN_SCORE_PLUGIN, scored=True),
            input=None,
            parameters={"images": str(images), "threshold": 0.75},
            run_id="plugin-run",
            staging_root=staging,
        )

    assert captured.value.code == "capability.plugin_score_invalid"


def test_non_numeric_score_is_a_contract_error(tmp_path: Path) -> None:
    images = _images_dir(tmp_path)
    staging = tmp_path / "staging"
    staging.mkdir()

    with pytest.raises(OrganelleContractError) as captured:
        OneShotBundleWorkerExecutor().invoke(
            _plugin_entry(tmp_path / "bundle", _STRING_SCORE_PLUGIN, scored=True),
            input=None,
            parameters={"images": str(images), "threshold": 0.75},
            run_id="plugin-run",
            staging_root=staging,
        )

    assert captured.value.code == "capability.plugin_score_invalid"


def test_plugin_summary_may_not_shadow_the_reserved_score_metric(tmp_path: Path) -> None:
    images = _images_dir(tmp_path)
    staging = tmp_path / "staging"
    staging.mkdir()

    with pytest.raises(OrganelleContractError) as captured:
        OneShotBundleWorkerExecutor().invoke(
            _plugin_entry(tmp_path / "bundle", _RESERVED_SUMMARY_PLUGIN, scored=True),
            input=None,
            parameters={"images": str(images), "threshold": 0.75},
            run_id="plugin-run",
            staging_root=staging,
        )

    assert captured.value.code == "capability.plugin_score_reserved"


def test_plugin_without_optimization_may_not_publish_the_reserved_score_metric(
    tmp_path: Path
) -> None:
    images = _images_dir(tmp_path)
    staging = tmp_path / "staging"
    staging.mkdir()

    with pytest.raises(OrganelleContractError) as captured:
        OneShotBundleWorkerExecutor().invoke(
            _plugin_entry(tmp_path / "bundle", _UNDECLARED_RESERVED_SUMMARY_PLUGIN),
            input=None,
            parameters={"images": str(images), "threshold": 0.75},
            run_id="plugin-run",
            staging_root=staging,
        )

    assert captured.value.code == "capability.plugin_score_reserved"


def test_huge_integer_score_has_the_declared_contract_error(tmp_path: Path) -> None:
    images = _images_dir(tmp_path)
    staging = tmp_path / "staging"
    staging.mkdir()

    with pytest.raises(OrganelleContractError) as captured:
        OneShotBundleWorkerExecutor().invoke(
            _plugin_entry(tmp_path / "bundle", _HUGE_INTEGER_SCORE_PLUGIN, scored=True),
            input=None,
            parameters={"images": str(images), "threshold": 0.75},
            run_id="plugin-run",
            staging_root=staging,
        )

    assert captured.value.code == "capability.plugin_score_invalid"


def test_scorer_exception_is_an_execution_error(tmp_path: Path) -> None:
    from organelleverse.core.errors import OrganelleExecutionError

    images = _images_dir(tmp_path)
    staging = tmp_path / "staging"
    staging.mkdir()

    with pytest.raises(OrganelleExecutionError) as captured:
        OneShotBundleWorkerExecutor().invoke(
            _plugin_entry(tmp_path / "bundle", _RAISING_SCORE_PLUGIN, scored=True),
            input=None,
            parameters={"images": str(images), "threshold": 0.75},
            run_id="plugin-run",
            staging_root=staging,
        )

    assert captured.value.code == "capability.plugin_score_failed"


def test_run_failure_still_publishes_failed_result_without_scoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    images = _images_dir(tmp_path)
    failing_scored = _FAILING_WITH_LOG_PLUGIN + """

def score(result: PluginResult) -> float:
    return 1.0
"""
    entry = _plugin_entry(tmp_path / "bundle", failing_scored, scored=True)
    strategy = _score_strategy(entry, tmp_path)

    result = strategy.invoke(None, {"images": str(images), "threshold": 0.75})

    assert isinstance(result, OrganelleResult)
    assert result.status == "failed"
    assert result.errors[0].code == "capability.plugin_execution_failed"
    assert "plugin_optimization_score" not in result.metrics


@pytest.mark.parametrize(
    "source",
    (
        _MISSING_SCORE_PLUGIN,
        _NON_CALLABLE_SCORE_PLUGIN,
        _TWO_PARAMETER_SCORE_PLUGIN,
        _WRONG_ANNOTATION_SCORE_PLUGIN,
        _POSITIONAL_ONLY_SCORE_PLUGIN,
        _STRING_RETURN_SCORE_PLUGIN,
        _MISSING_RETURN_SCORE_PLUGIN,
    ),
    ids=(
        "missing",
        "non_callable",
        "two_parameters",
        "wrong_parameter_annotation",
        "positional_only",
        "string_return_annotation",
        "missing_return_annotation",
    ),
)
def test_worker_inspection_rejects_an_invalid_declared_score_callable(
    tmp_path: Path, source: str
) -> None:
    # The score hook is pinned at worker inspection, i.e. before verification
    # completes and long before trust: an undeclared, non-callable, or wrongly
    # shaped score callable must fail with capability.plugin_score_invalid.
    with pytest.raises(OrganelleContractError) as captured:
        _plugin_entry(tmp_path / "bundle", source, scored=True)

    assert captured.value.code == "capability.plugin_score_invalid"


def test_worker_inspection_accepts_a_postponed_float_score_annotation(tmp_path: Path) -> None:
    entry = _plugin_entry(tmp_path / "bundle", _POSTPONED_FLOAT_SCORE_PLUGIN, scored=True)

    assert entry.worker_parameters is not None
