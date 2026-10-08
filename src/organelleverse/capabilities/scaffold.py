"""Generate a complete, self-verifying capability bundle on disk.

``scaffold(...)`` is the one path from "I have an idea for a capability" to a
bundle directory that passes the whole authoring loop -
``discover_capabilities -> verify_capability(LocalVerificationEnvironment) ->
trust -> admit_capabilities -> invoke`` - with **no manual edit**. See
``tests/capabilities/test_authoring_loop.py`` for the loop this bundle must
survive unmodified, and ``.superpowers/sdd/charters/
capability-02-task-3-authoring.md`` (Capability Plan 02 Task 3, deliverable
4) for why that matters: before this module existed, no test drove a
*generated* bundle through the seam at all.

A scaffolded bundle is deliberately narrow: one ``capability.toml`` matching
the proven ``native`` / ``named_parameters`` / ``canonical_json`` shape from
the authoring-loop test, one bundle-local ``code/<package>/`` tree (the only
layout ``code_identity.py``'s one-root rule accepts), and a README that
states the verification security property out loud - see the module
docstring in :mod:`organelleverse.capabilities.verification` for why that
statement is not optional: ``verify_capability`` imports and executes the
bundle's code inside the controlled worker *before* any trust record exists.

``scaffold`` never checks the declared parameter names against
``FORBIDDEN_FINAL_WRITE_PARAMS`` (:mod:`organelleverse.operations.
output_boundary`) itself. That gate already exists, fires from real bundle
data at ``verify_capability`` time (before the worker's real ``inspect()``
ever runs - see ``verification.py:298-311``), and duplicating it here would
just be a second, driftable copy of the same rule enforced somewhere a
scaffold-time check could never see updates to.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from organelleverse.core.errors import OrganelleContractError

from .parser import parse_capability_bundle

# The bundle-local worker's closed token language for the JSON codec (see
# ``organelleverse.capabilities._worker_main._annotation_token``). ``path``
# tokens are deliberately excluded: they require the ``path`` parameter
# codec, not ``json``, and a scaffold generating a mismatched codec/
# annotation pair would fail verification for a reason a reader could not
# see from the parameter declaration alone.
_SUPPORTED_ANNOTATIONS: frozenset[str] = frozenset({"str", "int", "float", "bool", "list", "dict"})
_NUMERIC_ANNOTATIONS: frozenset[str] = frozenset({"int", "float"})


@dataclass(frozen=True)
class ScaffoldParameter:
    """One named, JSON-codec parameter for a scaffolded capability's callable.

    ``annotation`` must be one of ``str``, ``int``, ``float``, ``bool``,
    ``list``, or ``dict`` - the bundle-local worker's closed token language
    for the ``json`` parameter codec. ``default`` must be a JSON-safe Python
    value; it becomes both the generated function's keyword default and the
    binding's declared default.
    """

    name: str
    annotation: str = "int"
    default: object = 1


_DEFAULT_PARAMETERS: tuple[ScaffoldParameter, ...] = (
    ScaffoldParameter(name="value", annotation="int", default=1),
)

# ``OperationSpec.keywords`` requires 3-8 sorted, unique, lowercase-pattern
# entries (spec.py's own ``validate_keywords``); this is the smallest set
# that satisfies it without asking every scaffold caller to supply keywords
# just to get past that rule.
_DEFAULT_KEYWORDS: tuple[str, ...] = ("capability", "generated", "scaffold")


def _default_package_name(capability_id: str) -> str:
    return f"{capability_id.replace('.', '_')}_capability"


def _validate_package_name(package_name: str) -> None:
    if not package_name.isidentifier():
        raise ValueError(f"package_name must be a valid Python identifier: {package_name!r}")
    if package_name in sys.stdlib_module_names or package_name in sys.builtin_module_names:
        raise ValueError(f"package_name collides with the Python runtime: {package_name!r}")


def _validate_parameters(parameters: Sequence[ScaffoldParameter]) -> None:
    names = [parameter.name for parameter in parameters]
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate scaffold parameter names: {sorted(names)}")
    for parameter in parameters:
        if not parameter.name.isidentifier() or parameter.name.startswith("_"):
            raise ValueError(
                f"scaffold parameter name must be a public Python identifier: {parameter.name!r}"
            )
        if parameter.annotation not in _SUPPORTED_ANNOTATIONS:
            raise ValueError(
                f"unsupported scaffold parameter annotation {parameter.annotation!r} for "
                f"{parameter.name!r}; supported: {sorted(_SUPPORTED_ANNOTATIONS)}"
            )


def _render_toml(
    *,
    capability_id: str,
    bundle_version: str,
    title: str,
    description: str,
    keywords: Sequence[str],
    stage: str,
    package_name: str,
    parameters: Sequence[ScaffoldParameter],
) -> str:
    keywords_toml = ", ".join(json.dumps(keyword) for keyword in keywords)
    lines = [
        'schema = "organelleverse.capability.v1"',
        "",
        "[capability]",
        f"id = {json.dumps(capability_id)}",
        f"bundle_version = {json.dumps(bundle_version)}",
        'implementation = "native"',
        "",
        "[contract]",
        'contract_version = "1.0"',
        f"title = {json.dumps(title)}",
        f"description = {json.dumps(description)}",
        f"keywords = [{keywords_toml}]",
        'execution_mode = "inline"',
        f"stage = {json.dumps(stage)}",
        'input_kind = "none"',
        'output_kind = "result"',
        f"callable_locator = {json.dumps(f'{package_name}.impl:run')}",
        "",
        "[contract.binding]",
        'argument_mode = "named_parameters"',
        'result_codec = "canonical_json"',
    ]
    for parameter in parameters:
        lines.extend(
            [
                "",
                "[[contract.binding.parameters]]",
                f"name = {json.dumps(parameter.name)}",
                'codec = "json"',
            ]
        )
    return "\n".join(lines) + "\n"


def _render_implementation(
    *,
    capability_id: str,
    parameters: Sequence[ScaffoldParameter],
) -> str:
    signature = ", ".join(
        f"{parameter.name}: {parameter.annotation} = {parameter.default!r}"
        for parameter in parameters
    )
    metric_lines = [f"        {parameter.name!r}: {parameter.name}," for parameter in parameters]
    numeric = next(
        (parameter for parameter in parameters if parameter.annotation in _NUMERIC_ANNOTATIONS),
        None,
    )
    if numeric is not None:
        metric_lines.append(f'        "doubled": {numeric.name} * 2,')
    metrics_body = "\n".join(metric_lines)
    return (
        '"""Generated by organelleverse.capabilities.scaffold. Edit freely."""\n'
        "\n"
        "from __future__ import annotations\n"
        "\n"
        "\n"
        f"def run(*, {signature}):\n"
        "    return {\n"
        '        "schema_version": "organelleverse.result.v1",\n'
        '        "kind": "result",\n'
        f'        "operation_id": {capability_id!r},\n'
        '        "scope": "none",\n'
        '        "status": "ok",\n'
        '        "metrics": {\n'
        f"{metrics_body}\n"
        "        },\n"
        "    }\n"
    )


def _render_readme(*, capability_id: str, title: str, description: str) -> str:
    return (
        f"# {title}\n"
        "\n"
        f"Capability id: `{capability_id}`\n"
        "\n"
        f"{description}\n"
        "\n"
        "Generated by `organelleverse.capabilities.scaffold.scaffold(...)`. Edit\n"
        "the files under `code/` freely; re-run `verify_capability` after any\n"
        "change so its recorded bundle content hash stays current.\n"
        "\n"
        "## Security notice\n"
        "\n"
        "`verify_capability` EXECUTES this bundle's code inside the controlled\n"
        "worker subprocess BEFORE any trust record exists. Verification imports\n"
        "and runs the module named by `callable_locator` in order to inspect its\n"
        "real signature and (for fixture-bearing bundles) evaluate it - so it\n"
        "already ran the code by the time it returns. Only call\n"
        "`verify_capability` on a bundle whose code you have read and are\n"
        "willing to run. `trust(...)` is a separate, later step: trust is\n"
        "granted only after verification succeeds, never before it.\n"
    )


def scaffold(
    target_dir: Path,
    *,
    capability_id: str,
    title: str,
    description: str,
    bundle_version: str = "1.0.0",
    keywords: Sequence[str] = _DEFAULT_KEYWORDS,
    stage: str = "analyze",
    parameters: Sequence[ScaffoldParameter] = _DEFAULT_PARAMETERS,
    package_name: str | None = None,
) -> Path:
    """Write one complete, verifiable capability bundle under *target_dir*.

    Returns *target_dir* (resolved), which is both the bundle root
    ``discover_capabilities(paths=[...])`` expects and the value it is safe
    to pass straight on to :func:`organelleverse.capabilities.verification.
    verify_capability` via discovery. *target_dir* must not already exist as
    a non-empty directory - ``scaffold`` never overwrites another bundle's
    files.

    The generated bundle declares zero fixtures and zero probes, matching the
    exact zero-fixture, `native`/`named_parameters`/`canonical_json` shape
    proven end-to-end by ``tests/capabilities/test_authoring_loop.py``. It
    is parsed with the real :func:`~organelleverse.capabilities.parser.
    parse_capability_bundle` before this function returns, so a scaffold that
    cannot pass its own bundle contract fails loudly here, not on the
    caller's first `discover_capabilities`.
    """
    resolved = Path(target_dir)
    if resolved.exists():
        if not resolved.is_dir():
            raise OrganelleContractError(
                code="capability.scaffold_target_not_empty",
                message="scaffold target exists and is not a directory",
                details={"target_dir": str(resolved)},
            )
        if any(resolved.iterdir()):
            raise OrganelleContractError(
                code="capability.scaffold_target_not_empty",
                message="scaffold target directory already has content",
                details={"target_dir": str(resolved)},
            )
    else:
        resolved.mkdir(parents=True)

    resolved_package_name = package_name or _default_package_name(capability_id)
    _validate_package_name(resolved_package_name)
    _validate_parameters(parameters)

    package_dir = resolved / "code" / resolved_package_name
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "impl.py").write_text(
        _render_implementation(capability_id=capability_id, parameters=parameters),
        encoding="utf-8",
    )

    (resolved / "capability.toml").write_text(
        _render_toml(
            capability_id=capability_id,
            bundle_version=bundle_version,
            title=title,
            description=description,
            keywords=keywords,
            stage=stage,
            package_name=resolved_package_name,
            parameters=parameters,
        ),
        encoding="utf-8",
    )
    (resolved / "README.md").write_text(
        _render_readme(capability_id=capability_id, title=title, description=description),
        encoding="utf-8",
    )

    # Dogfood the real, side-effect-free parser immediately: this only reads
    # capability.toml and never imports code/, so it stays within the
    # "discovery never executes anything" boundary while still catching a
    # malformed bundle at generation time.
    parse_capability_bundle(resolved / "capability.toml")

    return resolved


def _render_plugin_toml(
    *,
    capability_id: str,
    plugin_name: str,
    author: str,
    summary: str,
    package_name: str,
) -> str:
    """Render the complete v2 manifest for the standard local plugin protocol."""

    return f'''\
schema = "organelleverse.capability.v2"

[capability]
id = {json.dumps(capability_id)}
bundle_version = "0.1.0"
implementation = "native"

[contract]
contract_version = "1.0"
title = {json.dumps(plugin_name)}
description = {json.dumps(summary)}
keywords = ["generated", "plugin", "scientific"]
execution_mode = "durable"
stage = "analyze"
input_kind = "none"
output_kind = "result"
callable_locator = {json.dumps(f"{package_name}.plugin:run")}

[contract.binding]
argument_mode = "plugin_protocol"

[[contract.binding.parameters]]
name = "input_dir"
codec = "directory"
path_role = "input"
json_schema = {{ type = "string", description = "Directory containing the scientific inputs." }}

[[contract.binding.parameters]]
name = "results_dir"
codec = "directory"
path_role = "output"
json_schema = {{ type = "string", description = "Managed directory for this plugin's results." }}

[[contract.binding.parameters]]
name = "threshold"
codec = "json"
json_schema = {{ type = "number", minimum = 0.0, maximum = 1.0, default = 0.5, description = "Plugin decision threshold." }}

[[contract.outputs]]
name = "results"
kind = "directory"
parameter = "results_dir"
description = "Generated result files."

[contract.optimization]
score_locator = {json.dumps(f"{package_name}.plugin:score")}
parameters = ["threshold"]
max_trials = 1
parallelism = 1

[plugin]
name = {json.dumps(plugin_name)}
version = "0.1.0"
author = {json.dumps(author)}
summary = {json.dumps(summary)}
tags = ["generated", "plugin", "scientific"]

[agent]
task_description = {json.dumps(summary)}
examples = [{json.dumps(f"Run {plugin_name} on this input directory.")}]
'''


def _render_plugin_implementation() -> str:
    """Render the minimal standard plugin protocol implementation."""

    return '''\
"""Generated plugin entry point. Replace the analysis while retaining run's signature."""

from __future__ import annotations

import json
from pathlib import Path

from organelleverse.plugin_protocol import PluginContext, PluginResult


def run(inputs: dict, outputs: dict, parameters: dict, context: PluginContext) -> PluginResult:
    input_dir = Path(inputs["input_dir"])
    results_dir = Path(outputs["results"])
    input_file_count = sum(1 for path in input_dir.rglob("*") if path.is_file())
    summary = {
        "input_file_count": input_file_count,
        "threshold": parameters["threshold"],
    }
    (results_dir / "summary.json").write_text(
        json.dumps(summary, sort_keys=True) + "\\n", encoding="utf-8"
    )
    return PluginResult(summary=summary, outputs={"results": str(results_dir)})


def score(result: PluginResult) -> float:
    """Score one completed run for the declared threshold experiment.

    Only the finite float returned here becomes the run's score; it is
    published under the reserved ``plugin_optimization_score`` L6 metric.
    """

    return float(result.summary["input_file_count"])
'''


def _render_plugin_smoke_test(package_name: str) -> str:
    return f'''\
"""Plugin-owned smoke test for the generated entry point."""

from pathlib import Path


def test_generated_plugin_source_exists() -> None:
    source = Path(__file__).parents[1] / "code" / {package_name!r} / "plugin.py"
    assert source.is_file()
'''


def _render_plugin_readme(*, capability_id: str, plugin_name: str, summary: str) -> str:
    return f'''\
# {plugin_name}

Capability id: `{capability_id}`

{summary}

This is a v2 OrganelleVerse scientific plugin. Edit `code/` to implement the
analysis, retain the standard `run(inputs, outputs, parameters, context)`
signature, then run `verify_capability` before trusting or sharing it.

The manifest already declares a generated input form, managed output directory,
and an explicit threshold optimization experiment whose declared `score(...)`
return value is published as the reserved `plugin_optimization_score` metric.
The desktop UI and Agent
tool schema are derived directly from `capability.toml`.
'''


def scaffold_plugin(
    target_dir: Path,
    *,
    capability_id: str,
    plugin_name: str,
    author: str,
    summary: str,
    package: str | None = None,
) -> Path:
    """Write a portable v2 plugin bundle that is ready for the authoring lifecycle.

    The generated plugin uses the fixed local protocol and declares one input
    directory, one managed result directory, and one explicit numeric threshold.
    It is parsed with the production parser before return; callers can proceed
    directly to discovery, verification, trust, admission, and invocation.
    """

    if not plugin_name.strip() or not author.strip() or not summary.strip():
        raise ValueError("plugin_name, author, and summary must not be blank")
    resolved = Path(target_dir)
    if resolved.exists():
        if not resolved.is_dir() or any(resolved.iterdir()):
            raise OrganelleContractError(
                code="capability.scaffold_target_not_empty",
                message="plugin scaffold target must be a new or empty directory",
                details={"target_dir": str(resolved)},
            )
    else:
        resolved.mkdir(parents=True)

    package_name = package or f"{capability_id.replace('.', '_')}_plugin"
    _validate_package_name(package_name)
    code_dir = resolved / "code" / package_name
    code_dir.mkdir(parents=True)
    (code_dir / "__init__.py").write_text("", encoding="utf-8")
    (code_dir / "plugin.py").write_text(_render_plugin_implementation(), encoding="utf-8")
    tests_dir = resolved / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_plugin.py").write_text(
        _render_plugin_smoke_test(package_name), encoding="utf-8"
    )
    (resolved / "capability.toml").write_text(
        _render_plugin_toml(
            capability_id=capability_id,
            plugin_name=plugin_name,
            author=author,
            summary=summary,
            package_name=package_name,
        ),
        encoding="utf-8",
    )
    (resolved / "README.md").write_text(
        _render_plugin_readme(
            capability_id=capability_id,
            plugin_name=plugin_name,
            summary=summary,
        ),
        encoding="utf-8",
    )
    parse_capability_bundle(resolved / "capability.toml")
    return resolved


__all__ = ["ScaffoldParameter", "scaffold", "scaffold_plugin"]
