"""Capability Plan 05, directory-codec regeneration: the generator learns DIRECTORY.

``scripts/capabilities/build_restored_bundles.py`` predates the ``directory``
parameter codec that ``operations/spec.py``/``operations/python_binding.py``
just gained (mandatory ``path_role``, designed containment). Before this
change the generator had no way to honestly emit a ``codec = "directory"``
parameter at all: ``PlannedParameter`` had no ``path_role`` field,
``render_capability_toml`` never wrote one, and
``_validate_named_parameter_override`` never checked a DIRECTORY-codec
override's annotation shape or its declared role - meaning a malformed
override (wrong annotation, missing ``path_role``) would have silently
"passed" generator-level validation and only failed later, at
``ParameterBindingSpec`` construction inside ``parse_capability_bundle``,
which is fatal to the whole domain's build (see
``organelleverse.capabilities.adapters.rna_editing``'s module docstring for
the same shape of failure with a malformed ``operation_id``).

This test module is unit-level and adapter-agnostic: it exercises the
generator's own DIRECTORY-codec machinery directly (annotation recognition,
override validation, TOML rendering) with synthetic functions and overrides,
never touching a real restored-capability source file. The per-domain
``tests/capabilities/test_restored_<domain>_bundles.py`` files are where the
newly-admitted real capabilities are proven through the actual
discover -> verify -> admit -> invoke pipeline.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from organelleverse.capabilities.parser import parse_capability_bundle
from organelleverse.operations.spec import CoreKind, OperationStage, ParameterCodec, SideEffect
from tests._paths import PROJECT_ROOT

_GENERATOR_PATH = PROJECT_ROOT / "scripts" / "capabilities" / "build_restored_bundles.py"


def _load_generator() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_restored_bundles", _GENERATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _annotation_expr(source: str) -> ast.expr:
    module = ast.parse(source, mode="eval")
    assert isinstance(module, ast.Expression)
    return module.body


def _function_node(source: str) -> ast.FunctionDef:
    tree = ast.parse(source)
    (node,) = tree.body
    assert isinstance(node, ast.FunctionDef)
    return node


# --- _looks_like_directory_annotation --------------------------------------

_DIRECTORY_ANNOTATION_CASES: tuple[tuple[str, bool], ...] = (
    ("str", True),
    ("Path", True),
    ("str | Path", True),
    ("str | None", True),
    ("Path | None", True),
    ("int", False),
    ("dict[str, Any]", False),
    # DIRECTORY never accepts a list/Sequence - one directory names exactly
    # one tree or destination, unlike PATH's list form.
    ("list[str]", False),
    ("list[Path]", False),
    ("list[str | Path]", False),
    ("Sequence[str]", False),
)


@pytest.mark.parametrize("source,expected", _DIRECTORY_ANNOTATION_CASES)
def test_looks_like_directory_annotation(source: str, expected: bool) -> None:
    generator = _load_generator()
    assert generator._looks_like_directory_annotation(_annotation_expr(source)) is expected


def test_looks_like_directory_annotation_rejects_the_path_codec_list_form() -> None:
    """The one shape PATH accepts that DIRECTORY must not: a list of path-likes."""
    generator = _load_generator()
    path_list = _annotation_expr("list[str | Path]")
    assert generator._looks_like_path_annotation(path_list) is True
    assert generator._looks_like_directory_annotation(path_list) is False


# --- _validate_named_parameter_override: the DIRECTORY branch --------------

_PROBE_SOURCE = "def probe(data_root: str | Path, count: int = 1) -> dict[str, Any]:\n    ...\n"


def test_validate_named_parameter_override_requires_a_declared_path_role() -> None:
    generator = _load_generator()
    node = _function_node(_PROBE_SOURCE)
    override = SimpleNamespace(
        parameters=(
            SimpleNamespace(name="data_root", codec=ParameterCodec.DIRECTORY, path_role=None),
        )
    )

    error = generator._validate_named_parameter_override(node, override)

    assert error is not None
    assert "data_root" in error
    assert "path_role" in error


@pytest.mark.parametrize("bad_role", ["", "INPUT", "both", "outputs"])
def test_validate_named_parameter_override_rejects_an_unrecognized_path_role(bad_role: str) -> None:
    generator = _load_generator()
    node = _function_node(_PROBE_SOURCE)
    override = SimpleNamespace(
        parameters=(
            SimpleNamespace(name="data_root", codec=ParameterCodec.DIRECTORY, path_role=bad_role),
        )
    )

    error = generator._validate_named_parameter_override(node, override)

    assert error is not None
    assert "path_role" in error


@pytest.mark.parametrize("role", ["input", "output"])
def test_validate_named_parameter_override_accepts_a_declared_role(role: str) -> None:
    generator = _load_generator()
    node = _function_node(_PROBE_SOURCE)
    override = SimpleNamespace(
        parameters=(
            SimpleNamespace(name="data_root", codec=ParameterCodec.DIRECTORY, path_role=role),
        )
    )

    assert generator._validate_named_parameter_override(node, override) is None


def test_validate_named_parameter_override_rejects_directory_codec_on_a_non_path_annotation() -> (
    None
):
    """``count`` is an ``int`` - never a directory, role declared or not."""
    generator = _load_generator()
    node = _function_node("def probe(count: int) -> dict[str, Any]:\n    ...\n")
    override = SimpleNamespace(
        parameters=(
            SimpleNamespace(name="count", codec=ParameterCodec.DIRECTORY, path_role="input"),
        )
    )

    error = generator._validate_named_parameter_override(node, override)

    assert error is not None
    assert "count" in error
    assert "str/Path" in error


def test_validate_named_parameter_override_rejects_directory_codec_on_a_list_annotation() -> None:
    """The PATH codec's list form is not available to DIRECTORY."""
    generator = _load_generator()
    node = _function_node(
        "def probe(genbank_paths: list[str | Path]) -> dict[str, Any]:\n    ...\n"
    )
    override = SimpleNamespace(
        parameters=(
            SimpleNamespace(
                name="genbank_paths", codec=ParameterCodec.DIRECTORY, path_role="input"
            ),
        )
    )

    error = generator._validate_named_parameter_override(node, override)

    assert error is not None
    assert "genbank_paths" in error


# --- render_capability_toml: path_role round-trips through real TOML -------


def _planned_bundle(generator: ModuleType, parameters: tuple[object, ...]) -> object:
    return generator.PlannedBundle(
        capability_id="demo.directory_probe",
        title="Directory Probe",
        description="A synthetic bundle used only to test the codec/role rendering.",
        keywords=("demo", "directory", "probe"),
        stage=OperationStage.ANALYZE,
        input_kind=CoreKind.NONE,
        output_kind=CoreKind.RESULT,
        callable_locator="demo.module:probe",
        argument_mode="named_parameters",
        parameters=parameters,
        result_codec="json_metric",
        result_key="probe",
        side_effects=(SideEffect.READ_FILES,),
        dependencies=(),
    )


def test_planned_parameter_defaults_to_no_path_role() -> None:
    """Regression: a PATH/JSON PlannedParameter must not gain an unwanted role."""
    generator = _load_generator()
    parameter = generator.PlannedParameter(name="alignment_fasta", codec=ParameterCodec.PATH)
    assert parameter.path_role is None


def test_render_capability_toml_emits_path_role_for_a_directory_parameter() -> None:
    generator = _load_generator()
    parameters = (
        generator.PlannedParameter(
            name="data_root", codec=ParameterCodec.DIRECTORY, path_role="input"
        ),
    )
    rendered = generator.render_capability_toml(_planned_bundle(generator, parameters))

    assert 'codec = "directory"' in rendered
    assert 'path_role = "input"' in rendered


def test_render_capability_toml_omits_path_role_for_non_directory_parameters() -> None:
    generator = _load_generator()
    parameters = (generator.PlannedParameter(name="alignment_fasta", codec=ParameterCodec.PATH),)
    rendered = generator.render_capability_toml(_planned_bundle(generator, parameters))

    assert 'codec = "path"' in rendered
    assert "path_role" not in rendered


def test_render_capability_toml_directory_output_parses_as_a_real_bundle(tmp_path: Path) -> None:
    """The generator's own output must survive the real, strict bundle parser."""
    generator = _load_generator()
    parameters = (
        generator.PlannedParameter(
            name="work_dir", codec=ParameterCodec.DIRECTORY, path_role="output"
        ),
    )
    rendered = generator.render_capability_toml(_planned_bundle(generator, parameters))
    toml_path = tmp_path / "capability.toml"
    toml_path.write_text(rendered, encoding="utf-8")

    bundle = parse_capability_bundle(toml_path)

    (parameter,) = bundle.contract.binding.parameters
    assert parameter.name == "work_dir"
    assert parameter.codec is ParameterCodec.DIRECTORY
    assert parameter.path_role == "output"
