"""Capability Plan 03/05: the restored ``morphology`` bundles, for real.

``scripts/capabilities/build_restored_bundles.py`` generated the ten bundles
under ``src/organelleverse/capabilities/morphology-*`` from
``docs/operations/restored-capabilities.toml`` - see the generator's and
``organelleverse.capabilities.adapters.morphology``'s module docstrings for
the full per-capability reasoning.

This is Task 3's **first tool/model domain**: ``morphology.check_backend``
is ledger ``execution_class = "tool"`` and ``morphology.measure`` is
``"model"``. Both are proven below to reach ``ADMITTED`` and to actually
invoke their real implementation - ``check_backend`` genuinely spawns a
``python -c "import mmseg"`` subprocess per candidate environment
(``morphology/install.py``'s ``_env_has_mmseg``), and ``measure`` genuinely
runs numpy/scikit-image morphometrics on a real label-map image written to
``tmp_path``. Neither needs (and, per the adapter module's docstring,
neither *can* honestly get) a declared ``executable`` dependency: see that
docstring for why the ledger's ``tool``/``model`` label never actually
reaches ``LocalAdmissionEnvironment.observe_environment``'s
``DependencyKind.EXECUTABLE`` probe for any of the 13 such ledger records.

**Four more, Capability Plan 05**: ``default_executor``, ``train``,
``training_code_snippet``, and ``validate_dataset`` were blocked solely on a
required directory parameter until ``ParameterCodec.DIRECTORY`` shipped.
``validate_dataset``/``train``/``training_code_snippet`` all declare
``data_root`` with ``path_role = "input"`` (a pre-existing fine-tune dataset
tree); ``default_executor`` declares ``out_dir`` with ``path_role =
"output"`` (an Agent-chosen, unconditionally ``mkdir``-created destination).
``default_executor`` reaches ``ADMITTED`` like every other bundle here but is
**deliberately never invoked for real**: it locally imports ``mmseg``, whose
availability is environment-dependent (a restoration-ledger dependency, not
something admission enforces) - see the adapter module's own docstring.

Mirrors ``tests/capabilities/test_restored_format_conversion_bundles.py``:
discovery is left pointed at the real, installed ``src/organelleverse/``
tree for the ``"core"`` channel (only the other standard roots are
isolated), and no test constructs ``status="admitted"``, a
``VerificationRecord``, or an ``execution_identity`` by hand.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from PIL import Image

from organelleverse.capabilities.discovery import discover_capabilities
from organelleverse.capabilities.index import CapabilityStatus
from organelleverse.capabilities.verification import (
    LocalVerificationEnvironment,
    VerificationStore,
    verify_capability,
)
from organelleverse.core.result import OrganelleResult
from organelleverse.morphology import device as device_module
from organelleverse.morphology.device import DeviceInfo
from tests._paths import PROJECT_ROOT
from tests.capabilities._bundle_regeneration_check import (
    assert_regenerated_domain_matches_committed,
)

_CHECK_BACKEND_ID = "morphology.check_backend"
_CHECK_ALL_BACKENDS_ID = "morphology.check_all_backends"
_INSTALL_HINT_ID = "morphology.install_hint"
_MEASURE_ID = "morphology.measure"
_OVERLAY_ID = "morphology.overlay"
_SEGMENT_ID = "morphology.segment"
_DEFAULT_EXECUTOR_ID = "morphology.default_executor"
_TRAIN_ID = "morphology.train"
_TRAINING_CODE_SNIPPET_ID = "morphology.training_code_snippet"
_VALIDATE_DATASET_ID = "morphology.validate_dataset"
_RESTORED_IDS = (
    _CHECK_BACKEND_ID,
    _CHECK_ALL_BACKENDS_ID,
    _INSTALL_HINT_ID,
    _MEASURE_ID,
    _OVERLAY_ID,
    _SEGMENT_ID,
    _DEFAULT_EXECUTOR_ID,
    _TRAIN_ID,
    _TRAINING_CODE_SNIPPET_ID,
    _VALIDATE_DATASET_ID,
    "morphology.summarize",
)
_EXCLUDED_IDS = (
    "morphology.build_app",
    "morphology.serve",
    "morphology.write_csv",
)

_CPU_DEVICE = DeviceInfo(
    kind="cpu",
    name="test-cpu",
    total_vram_gb=None,
    runtime_version="test",
)

_GENERATOR_PATH = PROJECT_ROOT / "scripts" / "capabilities" / "build_restored_bundles.py"
_LEDGER_PATH = PROJECT_ROOT / "docs" / "operations" / "restored-capabilities.toml"
_CURRENT_ROOT = PROJECT_ROOT / "src" / "organelleverse"
_BUNDLE_ROOT = _CURRENT_ROOT / "capabilities"
_EXPECTED_BUNDLE_DIRS = {
    "morphology-check-all-backends",
    "morphology-check-backend",
    "morphology-install-hint",
    "morphology-measure",
    "morphology-overlay",
    "morphology-segment",
    "morphology-default-executor",
    "morphology-train",
    "morphology-training-code-snippet",
    "morphology-validate-dataset",
    "morphology-summarize"
}


def _load_generator() -> ModuleType:
    """Load ``build_restored_bundles.py`` standalone, mirroring the format_conversion test."""
    spec = importlib.util.spec_from_file_location("build_restored_bundles", _GENERATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _isolate_non_core_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolate every standard root except ``"core"``, which stays the real package."""
    home = tmp_path / "home"
    project = tmp_path / "project"
    (project / ".organelleverse" / "capabilities").mkdir(parents=True)
    home.mkdir()

    def no_entry_points(**kwargs: object) -> tuple[()]:
        return ()

    monkeypatch.setattr(importlib.metadata, "entry_points", no_entry_points)
    monkeypatch.setenv("ORGANELLEVERSE_HOME", str(home))
    monkeypatch.delenv("ORGANELLEVERSE_CAPABILITY_PATH", raising=False)
    monkeypatch.chdir(project)
    return home


def _write_label_map(path: Path) -> None:
    """Write a tiny synthetic OrgSegNet-style label map (class indices 0/1/2)."""
    label = np.zeros((24, 24), dtype=np.uint8)
    label[2:10, 2:10] = 1  # one Chloroplast blob
    label[14:18, 14:18] = 2  # one Mitochondria blob
    Image.fromarray(label).save(path)


def test_the_ten_bindable_capabilities_are_discovered_from_the_real_core_channel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate_non_core_roots(tmp_path, monkeypatch)

    index = discover_capabilities()

    discovered_ids = {entry.capability_id for entry in index.entries}
    for capability_id in _RESTORED_IDS:
        entry = index.describe(capability_id)
        assert entry.origins[0].channel == "core"
        assert entry.execution_identity is None
    for excluded_id in _EXCLUDED_IDS:
        assert excluded_id not in discovered_ids


def test_the_generator_produces_exactly_these_ids_and_skips_the_rest(tmp_path: Path) -> None:
    """Pin the generator's build report by id, not just by count.

    See ``test_restored_format_conversion_bundles.py``'s sibling test for why
    this asserts exact ids on both sides of the fail-closed boundary rather
    than a count.
    """
    generator = _load_generator()
    isolated_output_root = tmp_path / "capabilities"

    report = generator.build_domain(
        domain="morphology",
        ledger_path=_LEDGER_PATH,
        current_root=_CURRENT_ROOT,
        output_root=isolated_output_root,
    )

    generated_ids = {item.capability_id for item in report.generated}
    assert generated_ids == set(_RESTORED_IDS)

    skip_codes = {item.capability_id: item.code for item in report.skipped}
    assert skip_codes == {
        excluded_id: "capability.no_adapter_override" for excluded_id in _EXCLUDED_IDS
    }

    generated_dirs = {item.path.parent.name for item in report.generated}
    assert generated_dirs == _EXPECTED_BUNDLE_DIRS

    assert_regenerated_domain_matches_committed(
        isolated_output_root, _BUNDLE_ROOT, _EXPECTED_BUNDLE_DIRS
    )


def test_every_generated_bundle_reaches_admitted_through_the_real_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """discover -> verify -> re-admit, for all six real restored bundles."""
    home = _isolate_non_core_roots(tmp_path, monkeypatch)

    discovered = discover_capabilities()
    for capability_id in _RESTORED_IDS:
        assert discovered.describe(capability_id).status is CapabilityStatus.REJECTED

    store = VerificationStore(home / "verifications")
    for capability_id in _RESTORED_IDS:
        record = verify_capability(
            capability_id,
            store=store,
            environment=LocalVerificationEnvironment(discovered),
        )
        # A pure-core native capability has no bundle-local code/ tree to
        # content-hash: its integrity is the installed distribution itself.
        assert record.execution_identity is None
        assert record.worker_parameters == ()

    admitted = discover_capabilities()
    for capability_id in _RESTORED_IDS:
        entry = admitted.describe(capability_id)
        assert entry.status is CapabilityStatus.ADMITTED, entry.diagnostic
    admitted_ids = {item.capability_id for item in admitted.list()}
    assert admitted_ids == set(_RESTORED_IDS)


def _admit_all(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = _isolate_non_core_roots(tmp_path, monkeypatch)
    discovered = discover_capabilities()
    store = VerificationStore(home / "verifications")
    for capability_id in _RESTORED_IDS:
        verify_capability(
            capability_id, store=store, environment=LocalVerificationEnvironment(discovered)
        )
    return discover_capabilities()


def test_check_backend_resolves_and_really_spawns_a_probe_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ledger's one ``tool`` record: a genuine, ungated subprocess call.

    ``check_backend`` is never gated behind an injectable ``executor`` and
    reports the real environment. The assertion deliberately accepts either
    availability state: developer machines may or may not have OrgSegNet.
    """
    admitted = _admit_all(tmp_path, monkeypatch)
    binding = admitted.binding_source().resolve(_CHECK_BACKEND_ID)
    assert binding is not None

    result = binding.invoke(None, {"name": "orgseg"})

    assert isinstance(result, OrganelleResult)
    assert result.operation_id == _CHECK_BACKEND_ID
    assert result.status == "ok"
    status = result.metrics["backend_status"]
    assert status["name"] == "orgseg"
    assert isinstance(status["installed"], bool)
    if status["installed"]:
        assert status["path"]


def test_check_all_backends_resolves_and_invokes_with_zero_agent_parameters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    admitted = _admit_all(tmp_path, monkeypatch)
    binding = admitted.binding_source().resolve(_CHECK_ALL_BACKENDS_ID)
    assert binding is not None

    result = binding.invoke(None, {})

    assert isinstance(result, OrganelleResult)
    assert result.status == "ok"
    statuses = result.metrics["backend_status"]
    assert "orgseg" in statuses
    assert isinstance(statuses["orgseg"]["installed"], bool)
    if statuses["orgseg"]["installed"]:
        assert statuses["orgseg"]["path"]


def test_install_hint_resolves_and_invokes_returning_a_real_string(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    admitted = _admit_all(tmp_path, monkeypatch)
    binding = admitted.binding_source().resolve(_INSTALL_HINT_ID)
    assert binding is not None

    result = binding.invoke(None, {"name": "orgseg"})

    assert isinstance(result, OrganelleResult)
    assert result.status == "ok"
    hint_text = result.metrics["hint_text"]
    assert "OrgSegNet" in hint_text
    assert "not found" in hint_text


def test_measure_resolves_and_invokes_real_morphometrics_on_a_real_label_map(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ledger's one ``model`` record: real numpy/scikit-image compute."""
    admitted = _admit_all(tmp_path, monkeypatch)
    binding = admitted.binding_source().resolve(_MEASURE_ID)
    assert binding is not None

    label_path = tmp_path / "label.png"
    _write_label_map(label_path)

    result = binding.invoke(None, {"label_map": str(label_path)})

    assert isinstance(result, OrganelleResult)
    assert result.status == "ok"
    morphometrics = result.metrics["morphometrics"]
    assert morphometrics["n_total"] == 2
    class_names = {item["class_name"] for item in morphometrics["per_object"]}
    assert class_names == {"Chloroplast", "Mitochondria"}


def test_overlay_resolves_and_invokes_producing_a_real_numpy_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    admitted = _admit_all(tmp_path, monkeypatch)
    binding = admitted.binding_source().resolve(_OVERLAY_ID)
    assert binding is not None

    label_path = tmp_path / "label.png"
    _write_label_map(label_path)
    image_path = tmp_path / "source.png"
    Image.fromarray(np.full((24, 24), 128, dtype=np.uint8)).save(image_path)

    result = binding.invoke(None, {"image": str(image_path), "label_map": str(label_path)})

    assert isinstance(result, OrganelleResult)
    assert result.status == "ok"
    assert len(result.artifacts) == 1
    artifact = result.artifacts[0]
    assert artifact.format == "npy"
    loaded = np.load(artifact.resolve())
    assert loaded.shape == (24, 24, 3)


def test_segment_resolves_and_invokes_the_plan_only_path_on_a_real_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``executor`` is never Agent-exposed, so every real call takes the plan branch.

    That plan branch is still a real ``OrganelleResult`` the implementation
    itself constructed - the ``canonical`` result codec revalidates it
    unchanged, nothing is invented.
    """
    admitted = _admit_all(tmp_path, monkeypatch)
    binding = admitted.binding_source().resolve(_SEGMENT_ID)
    assert binding is not None
    monkeypatch.setattr(device_module, "detect_devices", lambda: (_CPU_DEVICE,))

    image_path = tmp_path / "em.png"
    Image.fromarray(np.full((16, 16), 64, dtype=np.uint8)).save(image_path)

    result = binding.invoke(None, {"images": str(image_path)})

    assert isinstance(result, OrganelleResult)
    assert result.operation_id == _SEGMENT_ID
    assert result.status == "warning"
    assert "morph_planned" in result.flags
    assert result.metrics["ran"] is False


def _write_dataset(root: Path, *, n_train: int = 2, n_val: int = 1, n_test: int = 1) -> None:
    """Build a real, well-formed OrgSegNet fine-tune dataset under ``root``."""
    image_dir = root / "image"
    label_dir = root / "label"
    splits_dir = root / "splits"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    splits_dir.mkdir(parents=True)

    counts = {"train": n_train, "val": n_val, "test": n_test}
    stems: dict[str, list[str]] = {}
    index = 0
    for split, count in counts.items():
        split_stems: list[str] = []
        for _ in range(count):
            stem = f"img{index:03d}"
            index += 1
            split_stems.append(stem)
            _write_label_map(image_dir / f"{stem}.tif")
            _write_label_map(label_dir / f"{stem}.png")
        stems[split] = split_stems
        (splits_dir / f"{split}.txt").write_text("\n".join(split_stems) + "\n")


def test_validate_dataset_resolves_and_invokes_a_real_directory_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The domain's first ``directory`` (``path_role="input"``) capability, proven for real.

    ``data_root`` genuinely walks a real directory tree - the DIRECTORY
    codec's containment (resolve, ``is_dir()``, whole-tree manifest hash)
    runs before the implementation is ever called, and the implementation
    itself then genuinely reads ``image/``, ``label/``, ``splits/*.txt``.
    """
    admitted = _admit_all(tmp_path, monkeypatch)
    binding = admitted.binding_source().resolve(_VALIDATE_DATASET_ID)
    assert binding is not None

    dataset_root = tmp_path / "dataset"
    _write_dataset(dataset_root)

    result = binding.invoke(None, {"data_root": str(dataset_root)})

    assert isinstance(result, OrganelleResult)
    assert result.status == "ok"
    validation = result.metrics["dataset_validation"]
    assert validation["valid"] is True
    assert validation["n_train"] == 2
    assert validation["n_val"] == 1
    assert validation["n_test"] == 1


def test_validate_dataset_rejects_a_missing_directory_before_invoking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Directory-input containment fails closed before the implementation runs."""
    from organelleverse.core.errors import OrganelleInputError

    admitted = _admit_all(tmp_path, monkeypatch)
    binding = admitted.binding_source().resolve(_VALIDATE_DATASET_ID)
    assert binding is not None

    with pytest.raises(OrganelleInputError) as captured:
        binding.invoke(None, {"data_root": str(tmp_path / "does-not-exist")})
    assert captured.value.code == "input.missing_artifact"


def test_training_code_snippet_resolves_and_invokes_a_real_directory_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``data_root`` is only ``str()``-embedded, never opened - containment still applies."""
    admitted = _admit_all(tmp_path, monkeypatch)
    binding = admitted.binding_source().resolve(_TRAINING_CODE_SNIPPET_ID)
    assert binding is not None

    dataset_root = tmp_path / "dataset"
    _write_dataset(dataset_root)

    result = binding.invoke(None, {"data_root": str(dataset_root)})

    assert isinstance(result, OrganelleResult)
    assert result.status == "ok"
    snippet = result.metrics["training_snippet"]
    assert str(dataset_root) in snippet
    assert "Runner.from_cfg" in snippet


def test_train_resolves_and_invokes_the_plan_only_path_on_a_real_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``executor`` is never Agent-exposed, so every real call plans rather than trains."""
    admitted = _admit_all(tmp_path, monkeypatch)
    binding = admitted.binding_source().resolve(_TRAIN_ID)
    assert binding is not None

    dataset_root = tmp_path / "dataset"
    _write_dataset(dataset_root)

    result = binding.invoke(None, {"data_root": str(dataset_root)})

    assert isinstance(result, OrganelleResult)
    assert result.operation_id == _TRAIN_ID
    assert result.status == "warning"
    assert "morph_train_planned" in result.flags
    assert result.metrics["ran"] is False
    assert result.metrics["n_train"] == 2


def test_default_executor_resolves_but_is_not_invoked_without_mmseg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Admitted through the real pipeline - not invoked, ``mmseg`` is not installed here.

    Resolving the binding proves the ``directory`` (``path_role="output"``)
    parameter and the ``artifact`` result codec (a ``list[Path]`` return)
    both bind honestly; a real call would need ``mmseg``, which this
    environment does not have, so that call is not made (see the adapter
    module's own docstring for why that is an environment gap, not a
    binding defect).
    """
    admitted = _admit_all(tmp_path, monkeypatch)
    binding = admitted.binding_source().resolve(_DEFAULT_EXECUTOR_ID)
    assert binding is not None


def test_a_capability_the_generator_refused_to_produce_is_simply_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A never-generated id is unknown, not rejected - see the format_conversion sibling test."""
    _isolate_non_core_roots(tmp_path, monkeypatch)
    index = discover_capabilities()

    for excluded_id in _EXCLUDED_IDS:
        with pytest.raises(Exception) as excinfo:
            index.describe(excluded_id)
        assert "unknown_capability" in str(excinfo.value) or "unknown capability" in str(
            excinfo.value
        )
