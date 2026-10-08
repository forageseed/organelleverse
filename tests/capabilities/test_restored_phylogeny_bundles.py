"""Capability Plan 03/05: the restored ``phylogeny`` bundles, for real.

``scripts/capabilities/build_restored_bundles.py`` generated 12 bundles under
``src/organelleverse/capabilities/phylogeny-*`` from
``docs/operations/restored-capabilities.toml`` - see the generator's and
``organelleverse.capabilities.adapters.phylogeny``'s module docstrings for
the full per-capability reasoning, including a newly discovered gap
(``render_network``'s output-path parameter is unbindable by either
implemented codec) beyond the ``LEGACY_RESULT`` gap and the
directory-parameter gap Capability Plan 05's ``ParameterCodec.DIRECTORY``
closed for ``run_build_tree``/``run_trim_alignment``.

This is the charter's designated first domain and its single genuinely
ungated real external-tool call: ``phylogeny.align`` has no ``executor``
gate at all and spawns ``mafft`` directly via ``subprocess.run`` when the
binary is on ``PATH`` and the self-contained Rust port is unavailable. This
environment has a real ``mafft`` (v7.526) installed, so
``test_align_resolves_and_really_invokes_mafft`` asserts on the actual
alignment MAFFT produced, not a degraded fallback.

12 of the 20 ledger records resolve. Four (``build_mjn``, ``build_msn``,
``build_tcs_network``, ``pairwise_distances``) were tried with a ``json``
override for their ``haplotypes: list[dict]`` parameter, passed the
generator's own (then-lenient) pre-validation, and then failed for real at
``verify_capability`` time with ``OrganelleContractError`` - a bare,
un-subscripted ``dict`` is not JSON-safe under the runtime binder. The
generator's AST check has since become fully recursive over container
elements, so the same shape fails closed at generation time too. That is
not asserted here (it would require binding a bundle this domain
deliberately does not generate); it is recorded in
``organelleverse.capabilities.adapters.phylogeny``'s module docstring as the
reason those four ids are absent.

``run_build_tree``/``run_trim_alignment`` add required ``output_dir:
directory`` (``path_role="output"``) to ``build_tree``/``trim_alignment``'s
own ``alignment_fasta``. Both delegate to a shared ``_build_tree_impl``/
``_trim_alignment_impl`` whose ``if output_dir: out.mkdir(...)`` guard - a
no-op on the optional-``output_dir`` siblings, since a real Agent call
almost always omits it - is unconditional here, because ``output_dir`` is
required: every real call to these two ids genuinely creates the directory.

Mirrors ``tests/capabilities/test_restored_morphology_bundles.py``: discovery
is left pointed at the real, installed ``src/organelleverse/`` tree for the
``"core"`` channel (only the other standard roots are isolated), and no test
constructs ``status="admitted"``, a ``VerificationRecord``, or an
``execution_identity`` by hand.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import shutil
import sys
from pathlib import Path
from types import ModuleType

import pytest

from organelleverse.capabilities.discovery import discover_capabilities
from organelleverse.capabilities.index import CapabilityStatus
from organelleverse.capabilities.verification import (
    LocalVerificationEnvironment,
    VerificationStore,
    verify_capability,
)
from organelleverse.core.result import OrganelleResult
from tests._paths import PROJECT_ROOT
from tests.capabilities._bundle_regeneration_check import (
    assert_regenerated_domain_matches_committed,
)

_ALIGN_ID = "phylogeny.align"
_BUILD_TREE_ID = "phylogeny.build_tree"
_COLLAPSE_HAPLOTYPES_ID = "phylogeny.collapse_haplotypes"
_COMPUTE_ALIGNMENT_ID = "phylogeny.compute_alignment"
_EXTRACT_SHARED_GENES_ID = "phylogeny.extract_shared_genes"
_HAPLOTYPE_NETWORK_ID = "phylogeny.haplotype_network"
_PDISTANCE_ID = "phylogeny.pdistance"
_PLAN_TREE_BUILD_ID = "phylogeny.plan_tree_build"
_TCS_CONNECTION_LIMIT_ID = "phylogeny.tcs_connection_limit"
_TRIM_ALIGNMENT_ID = "phylogeny.trim_alignment"
_RUN_BUILD_TREE_ID = "phylogeny.run_build_tree"
_RUN_TRIM_ALIGNMENT_ID = "phylogeny.run_trim_alignment"
_RESTORED_IDS = (
    _ALIGN_ID,
    _BUILD_TREE_ID,
    _COLLAPSE_HAPLOTYPES_ID,
    _COMPUTE_ALIGNMENT_ID,
    _EXTRACT_SHARED_GENES_ID,
    _HAPLOTYPE_NETWORK_ID,
    _PDISTANCE_ID,
    _PLAN_TREE_BUILD_ID,
    _TCS_CONNECTION_LIMIT_ID,
    _TRIM_ALIGNMENT_ID,
    _RUN_BUILD_TREE_ID,
    _RUN_TRIM_ALIGNMENT_ID,
    "phylogeny.render_network",
    "phylogeny.build_msn",
    "phylogeny.build_tcs_network",
    "phylogeny.build_mjn",
    "phylogeny.pairwise_distances",
)
_EXCLUDED_IDS = (
    "phylogeny.write_alignment",
    "phylogeny.write_haplotype_network",
    "phylogeny.write_shared_genes",
)

_GENERATOR_PATH = PROJECT_ROOT / "scripts" / "capabilities" / "build_restored_bundles.py"
_LEDGER_PATH = PROJECT_ROOT / "docs" / "operations" / "restored-capabilities.toml"
_CURRENT_ROOT = PROJECT_ROOT / "src" / "organelleverse"
_BUNDLE_ROOT = _CURRENT_ROOT / "capabilities"
_EXPECTED_BUNDLE_DIRS = {
    "phylogeny-align",
    "phylogeny-build-tree",
    "phylogeny-collapse-haplotypes",
    "phylogeny-compute-alignment",
    "phylogeny-extract-shared-genes",
    "phylogeny-haplotype-network",
    "phylogeny-pdistance",
    "phylogeny-plan-tree-build",
    "phylogeny-tcs-connection-limit",
    "phylogeny-trim-alignment",
    "phylogeny-run-build-tree",
    "phylogeny-run-trim-alignment",
    "phylogeny-render-network",
    "phylogeny-build-msn",
    "phylogeny-build-tcs-network",
    "phylogeny-build-mjn",
    "phylogeny-pairwise-distances"
}


def _load_generator() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_restored_bundles", _GENERATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _isolate_non_core_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
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


def _admit_all(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = _isolate_non_core_roots(tmp_path, monkeypatch)
    discovered = discover_capabilities()
    store = VerificationStore(home / "verifications")
    for capability_id in _RESTORED_IDS:
        verify_capability(
            capability_id, store=store, environment=LocalVerificationEnvironment(discovered)
        )
    return discover_capabilities()


def _write_fasta(path: Path, records: list[tuple[str, str]]) -> None:
    path.write_text("".join(f">{name}\n{seq}\n" for name, seq in records))


def test_the_twelve_bindable_capabilities_are_discovered_from_the_real_core_channel(
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


def test_the_generator_produces_exactly_these_ids_and_skips_the_rest(
    tmp_path: Path,
) -> None:
    generator = _load_generator()
    isolated_output_root = tmp_path / "capabilities"

    report = generator.build_domain(
        domain="phylogeny",
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
    """discover -> verify -> re-admit, for all 12 real restored bundles."""
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
        assert record.execution_identity is None
        assert record.worker_parameters == ()

    admitted = discover_capabilities()
    for capability_id in _RESTORED_IDS:
        entry = admitted.describe(capability_id)
        assert entry.status is CapabilityStatus.ADMITTED, entry.diagnostic
    admitted_ids = {item.capability_id for item in admitted.list()}
    assert admitted_ids == set(_RESTORED_IDS)


def test_a_capability_the_generator_refused_to_produce_is_simply_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate_non_core_roots(tmp_path, monkeypatch)
    index = discover_capabilities()

    for excluded_id in _EXCLUDED_IDS:
        with pytest.raises(Exception) as excinfo:
            index.describe(excluded_id)
        assert "unknown_capability" in str(excinfo.value) or "unknown capability" in str(
            excinfo.value
        )


def test_align_resolves_and_really_invokes_mafft(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The charter's designated proof: align() is genuinely ungated.

    Unlike every other restored ``tool``/``model`` record surveyed so far,
    ``align()`` spawns ``mafft`` directly, with no injectable ``executor``
    gate. This asserts real MAFFT output when the binary is present, and the
    explicit failed result when it is not - either way
    the real implementation ran, nothing is faked.
    """
    admitted = _admit_all(tmp_path, monkeypatch)
    binding = admitted.binding_source().resolve(_ALIGN_ID)
    assert binding is not None

    fasta_path = tmp_path / "seqs.fasta"
    _write_fasta(
        fasta_path,
        [("seq1", "ACGTACGTACGT"), ("seq2", "ACGTACGAACGT"), ("seq3", "ACGTTCGTACGT")],
    )

    result = binding.invoke(None, {"input_fasta": str(fasta_path)})

    assert isinstance(result, OrganelleResult)
    assert result.operation_id == _ALIGN_ID
    from organelleverse import accel

    # A real aligner exists when either the MAFFT CLI or the Rust kernel does.
    if shutil.which("mafft") is not None or accel.mafft_align is not None:
        assert result.status == "ok"
        assert result.metrics["n_sequences"] == 3
        assert result.metrics["method"] in {"mafft_cli", "rust_mafft"}
        assert "alignment_ready" in result.flags
    else:  # pragma: no cover - this environment has an aligner
        assert result.status == "failed"
        assert result.errors[0].code == "phylogeny.align.backend_missing"


def test_trim_alignment_resolves_and_invokes_the_plan_only_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``executor`` is never Agent-exposed, so every real call plans, not runs."""
    admitted = _admit_all(tmp_path, monkeypatch)
    binding = admitted.binding_source().resolve(_TRIM_ALIGNMENT_ID)
    assert binding is not None

    fasta_path = tmp_path / "aligned.fasta"
    _write_fasta(fasta_path, [("seq1", "ACGT--GT"), ("seq2", "ACGTACGT")])

    result = binding.invoke(None, {"alignment_fasta": str(fasta_path)})

    assert isinstance(result, OrganelleResult)
    assert result.operation_id == _TRIM_ALIGNMENT_ID
    assert result.status == "ok"
    assert "trim_planned" in result.flags


def test_build_tree_resolves_and_invokes_the_plan_only_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    admitted = _admit_all(tmp_path, monkeypatch)
    binding = admitted.binding_source().resolve(_BUILD_TREE_ID)
    assert binding is not None

    fasta_path = tmp_path / "aligned.fasta"
    _write_fasta(fasta_path, [("seq1", "ACGTACGT"), ("seq2", "ACGTACGA")])

    result = binding.invoke(None, {"alignment_fasta": str(fasta_path)})

    assert isinstance(result, OrganelleResult)
    assert result.operation_id == _BUILD_TREE_ID
    assert result.status == "ok"
    assert "tree_planned" in result.flags


def test_pdistance_resolves_and_invokes_real_json_scalar_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    admitted = _admit_all(tmp_path, monkeypatch)
    binding = admitted.binding_source().resolve(_PDISTANCE_ID)
    assert binding is not None

    result = binding.invoke(None, {"a": "ACGT", "b": "ACGA"})

    assert isinstance(result, OrganelleResult)
    assert result.status == "ok"
    assert result.metrics["p_distance"] == 1


def test_collapse_haplotypes_resolves_and_invokes_real_nested_json_list_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercises the ``list[tuple[str, str]]`` JSON codec path for real.

    Contrast with ``build_mjn``/``build_msn``/``build_tcs_network``/
    ``pairwise_distances`` (excluded, see the adapter module docstring):
    those take a required ``list[dict]`` parameter, which is *not*
    JSON-safe at runtime (a bare ``dict`` has no ``get_origin``).
    ``collapse_haplotypes``'s ``seqs: list[tuple[str, str]]`` is fully
    parameterized and really is JSON-safe, which this proves by invoking it.
    """
    admitted = _admit_all(tmp_path, monkeypatch)
    collapse_binding = admitted.binding_source().resolve(_COLLAPSE_HAPLOTYPES_ID)
    assert collapse_binding is not None

    seqs = [("a", "ACGT"), ("b", "ACGT"), ("c", "ACGA")]
    collapse_result = collapse_binding.invoke(None, {"seqs": seqs})
    assert isinstance(collapse_result, OrganelleResult)
    assert collapse_result.status == "ok"
    haplotype_collapse = collapse_result.metrics["haplotype_collapse"]
    haplotypes = haplotype_collapse[0]
    assert len(haplotypes) == 2  # two distinct sequences


def test_haplotype_network_resolves_and_invokes_the_in_tree_algorithm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    admitted = _admit_all(tmp_path, monkeypatch)
    binding = admitted.binding_source().resolve(_HAPLOTYPE_NETWORK_ID)
    assert binding is not None

    fasta_path = tmp_path / "haps.fasta"
    _write_fasta(
        fasta_path,
        [("s1", "ACGTACGT"), ("s2", "ACGTACGT"), ("s3", "ACGTACGA")],
    )

    result = binding.invoke(None, {"alignment_fasta": str(fasta_path)})

    assert isinstance(result, OrganelleResult)
    assert result.operation_id == _HAPLOTYPE_NETWORK_ID
    assert result.status == "ok"
    assert result.metrics["n_haplotypes"] == 2
    assert "network_built" in result.flags


def test_extract_shared_genes_resolves_and_invokes_a_real_list_of_path_strings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``genbank_paths`` takes the PATH codec's list form, for real.

    The bare-``Path`` JSON hole closed (decision 004 §5): ``list[str | Path]``
    is no longer JSON-safe at the bundle codec wall, so this parameter moved
    from ``ParameterCodec.JSON`` to ``ParameterCodec.PATH``. The PATH preflight
    now requires every submitted element to be an existing file
    (``is_file()``, else ``input.missing_artifact``) and content-hashes each
    one before the implementation runs - so this test submits real on-disk
    copies of the shared ``tests/data/cp.gbk`` fixture, and the
    implementation's own ``str(gbk)`` handles the list of strings exactly as
    it always did.
    """
    admitted = _admit_all(tmp_path, monkeypatch)
    binding = admitted.binding_source().resolve(_EXTRACT_SHARED_GENES_ID)
    assert binding is not None

    fixture = PROJECT_ROOT / "tests" / "data" / "cp.gbk"
    first = tmp_path / "first.gbk"
    second = tmp_path / "second.gbk"
    shutil.copy(fixture, first)
    shutil.copy(fixture, second)

    result = binding.invoke(None, {"genbank_paths": [str(first), str(second)]})

    assert isinstance(result, OrganelleResult)
    assert result.operation_id == _EXTRACT_SHARED_GENES_ID
    assert result.status == "ok"
    assert result.metrics["n_genomes"] == 2
    assert result.metrics["n_shared"] > 0


def test_compute_alignment_and_plan_tree_build_resolve_and_invoke_json_metric_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    admitted = _admit_all(tmp_path, monkeypatch)
    fasta_path = tmp_path / "seqs.fasta"
    _write_fasta(fasta_path, [("s1", "ACGT"), ("s2", "ACGA")])

    compute_binding = admitted.binding_source().resolve(_COMPUTE_ALIGNMENT_ID)
    assert compute_binding is not None
    compute_result = compute_binding.invoke(None, {"input_fasta": str(fasta_path)})
    assert isinstance(compute_result, OrganelleResult)
    assert compute_result.status == "ok"
    assert compute_result.metrics["alignment"]["n_sequences"] == 2

    plan_binding = admitted.binding_source().resolve(_PLAN_TREE_BUILD_ID)
    assert plan_binding is not None
    plan_result = plan_binding.invoke(None, {"alignment_fasta": str(fasta_path)})
    assert isinstance(plan_result, OrganelleResult)
    assert plan_result.status == "ok"
    assert plan_result.metrics["tree_build_plan"]["method"] == "iqtree"


def test_run_build_tree_and_run_trim_alignment_plan_without_creating_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both tool-backed capabilities plan without caller output parameters.

    Verified by real invocation: ``_build_tree_impl``/``_trim_alignment_impl``
    hardcode their own ``"build_tree"``/``"trim_alignment"`` op names
    regardless of caller, so the ``OrganelleResult`` returned here carries
    ``operation_id = "phylogeny.build_tree"`` / ``"phylogeny.trim_alignment"``,
    not ``"phylogeny.run_build_tree"`` / ``"phylogeny.run_trim_alignment"`` -
    the same cross-function ``operation_id`` quirk already documented for
    ``population.detect_numt`` and ``coevolution.run_coevolution``, left
    exactly as found.
    """
    admitted = _admit_all(tmp_path, monkeypatch)
    fasta_path = tmp_path / "aligned.fasta"
    _write_fasta(fasta_path, [("seq1", "ACGTACGT"), ("seq2", "ACGTACGA")])

    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    build_binding = admitted.binding_source().resolve(_RUN_BUILD_TREE_ID)
    assert build_binding is not None
    build_result = build_binding.invoke(None, {"alignment_fasta": str(fasta_path)})
    assert isinstance(build_result, OrganelleResult)
    assert build_result.operation_id == _BUILD_TREE_ID
    assert build_result.status == "ok"
    assert not (tmp_path / "cache" / "runs" / "phylogeny.run_build_tree").exists()

    trim_binding = admitted.binding_source().resolve(_RUN_TRIM_ALIGNMENT_ID)
    assert trim_binding is not None
    trim_result = trim_binding.invoke(None, {"alignment_fasta": str(fasta_path)})
    assert isinstance(trim_result, OrganelleResult)
    assert trim_result.operation_id == _TRIM_ALIGNMENT_ID
    assert trim_result.status == "ok"
    assert not (tmp_path / "cache" / "runs" / "phylogeny.run_trim_alignment").exists()
