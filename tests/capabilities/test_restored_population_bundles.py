"""Capability Plan 03/05: the restored ``population`` bundles, for real.

``scripts/capabilities/build_restored_bundles.py`` generated all 9 bundles
under ``src/organelleverse/capabilities/population-*`` from
``docs/operations/restored-capabilities.toml`` - see
``organelleverse.capabilities.adapters.population``'s module docstring for
the full per-capability reasoning. ``call_variants``/``prepare_gemma_input``
were excluded until Capability Plan 05's ``ParameterCodec.DIRECTORY``
shipped (``bam_dir`` on the input side, ``out_dir`` on the output side) -
this domain now resolves every ledger record, so ``_EXCLUDED_IDS`` below is
empty and there is no "refused to produce" demonstration test.

Mirrors ``tests/capabilities/test_restored_format_conversion_bundles.py``.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
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

_CHECK_ALL_BACKENDS_ID = "population.check_all_backends"
_CHECK_BACKEND_ID = "population.check_backend"
_COMPUTE_FST_ID = "population.compute_fst"
_CYTONUCLEAR_GWAS_ID = "population.cytonuclear_gwas"
_DETECT_NUMT_ID = "population.detect_numt"
_FST_SCAN_ID = "population.fst_scan"
_INSTALL_HINT_ID = "population.install_hint"
_CALL_VARIANTS_ID = "population.call_variants"
_PREPARE_GEMMA_INPUT_ID = "population.prepare_gemma_input"
_RESTORED_IDS = (
    _CHECK_ALL_BACKENDS_ID,
    _CHECK_BACKEND_ID,
    _COMPUTE_FST_ID,
    _CYTONUCLEAR_GWAS_ID,
    _DETECT_NUMT_ID,
    _FST_SCAN_ID,
    _INSTALL_HINT_ID,
    _CALL_VARIANTS_ID,
    _PREPARE_GEMMA_INPUT_ID,
)
_EXCLUDED_IDS: tuple[str, ...] = ()

_GENERATOR_PATH = PROJECT_ROOT / "scripts" / "capabilities" / "build_restored_bundles.py"
_LEDGER_PATH = PROJECT_ROOT / "docs" / "operations" / "restored-capabilities.toml"
_CURRENT_ROOT = PROJECT_ROOT / "src" / "organelleverse"
_BUNDLE_ROOT = _CURRENT_ROOT / "capabilities"
_EXPECTED_BUNDLE_DIRS = {
    "population-check-all-backends",
    "population-check-backend",
    "population-compute-fst",
    "population-cytonuclear-gwas",
    "population-detect-numt",
    "population-fst-scan",
    "population-install-hint",
    "population-call-variants",
    "population-prepare-gemma-input",
}

_VCF_TEXT = """\
##fileformat=VCFv4.2
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tsampleA\tsampleB\tsampleC\tsampleD
chr1\t100\t.\tA\tT\t.\tPASS\t.\tGT\t0/0\t0/1\t1/1\t0/1
chr1\t200\t.\tG\tC\t.\tPASS\t.\tGT\t0/1\t1/1\t0/0\t1/1
chr1\t300\t.\tA\tG\t.\tPASS\t.\tGT\t1/1\t0/0\t0/1\t0/1
"""
_POP_ASSIGNMENTS = {"sampleA": "pop1", "sampleB": "pop1", "sampleC": "pop2", "sampleD": "pop2"}


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


def test_the_nine_bindable_capabilities_are_discovered_from_the_real_core_channel(
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
        domain="population",
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
    home = _isolate_non_core_roots(tmp_path, monkeypatch)

    discovered = discover_capabilities()
    for capability_id in _RESTORED_IDS:
        assert discovered.describe(capability_id).status is CapabilityStatus.REJECTED

    store = VerificationStore(home / "verifications")
    for capability_id in _RESTORED_IDS:
        record = verify_capability(
            capability_id, store=store, environment=LocalVerificationEnvironment(discovered)
        )
        assert record.execution_identity is None
        assert record.worker_parameters == ()

    admitted = discover_capabilities()
    for capability_id in _RESTORED_IDS:
        entry = admitted.describe(capability_id)
        assert entry.status is CapabilityStatus.ADMITTED, entry.diagnostic
    admitted_ids = {item.capability_id for item in admitted.list()}
    assert admitted_ids == set(_RESTORED_IDS)


def test_fst_scan_and_compute_fst_resolve_and_invoke_on_a_real_vcf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    admitted = _admit_all(tmp_path, monkeypatch)
    vcf_path = tmp_path / "pop.vcf"
    vcf_path.write_text(_VCF_TEXT)

    fst_scan_binding = admitted.binding_source().resolve(_FST_SCAN_ID)
    assert fst_scan_binding is not None
    fst_scan_result = fst_scan_binding.invoke(
        None, {"vcf_path": str(vcf_path), "pop_assignments": _POP_ASSIGNMENTS}
    )
    assert isinstance(fst_scan_result, OrganelleResult)
    assert fst_scan_result.operation_id == _FST_SCAN_ID
    assert fst_scan_result.status == "ok"

    compute_fst_binding = admitted.binding_source().resolve(_COMPUTE_FST_ID)
    assert compute_fst_binding is not None
    compute_fst_result = compute_fst_binding.invoke(
        None, {"vcf_path": str(vcf_path), "pop_assignments": _POP_ASSIGNMENTS}
    )
    assert isinstance(compute_fst_result, OrganelleResult)
    assert compute_fst_result.status == "ok"
    assert compute_fst_result.metrics["fst"]["n_snps"] == 3


def test_detect_numt_resolves_and_invokes_a_real_kmer_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``detect_numt`` delegates to ``transfer._detect_transfer_alignment``.

    A pre-existing quirk of the restored source, discovered by actually
    invoking it rather than assumed: ``population.detect_numt`` (``population
    /population.py``) is a thin wrapper around
    ``transfer.transfer._detect_transfer``, which hardcodes its own
    ``operation_id`` as ``f"transfer.{op}"`` regardless of caller - so the
    real, successfully-admitted ``population.detect_numt`` binding returns
    an ``OrganelleResult`` whose own ``operation_id`` field reads
    ``"transfer.detect_numt"``, not ``"population.detect_numt"``. The
    ``canonical`` result codec only revalidates the value unchanged
    (``codecs._encode_canonical`` never checks it against the bound
    capability id), so this is not a binding failure - it is left exactly
    as the restored implementation behaves, not "fixed" here.
    """
    admitted = _admit_all(tmp_path, monkeypatch)
    nuclear_path = tmp_path / "nuclear.fasta"
    mito_path = tmp_path / "mito.fasta"
    shared = "ACGTACGTACGTACGTACGTACGTACGTACGTACGT"
    nuclear_path.write_text(f">nuc1\nGGGG{shared}TTTT\n")
    mito_path.write_text(f">mito1\n{shared}\n")

    binding = admitted.binding_source().resolve(_DETECT_NUMT_ID)
    assert binding is not None

    result = binding.invoke(
        None, {"nuclear_fasta": str(nuclear_path), "mito_fasta": str(mito_path)}
    )
    assert isinstance(result, OrganelleResult)
    assert result.operation_id == "transfer.detect_numt"
    assert result.status == "ok"


def test_check_backend_and_install_hint_resolve_and_invoke_real_json_scalars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    admitted = _admit_all(tmp_path, monkeypatch)

    check_binding = admitted.binding_source().resolve(_CHECK_BACKEND_ID)
    assert check_binding is not None
    check_result = check_binding.invoke(None, {"name": "gatk"})
    assert isinstance(check_result, OrganelleResult)
    assert check_result.status == "ok"
    assert check_result.metrics["backend_status"]["name"] == "gatk"

    hint_binding = admitted.binding_source().resolve(_INSTALL_HINT_ID)
    assert hint_binding is not None
    hint_result = hint_binding.invoke(None, {"name": "gatk"})
    assert isinstance(hint_result, OrganelleResult)
    assert hint_result.status == "ok"
    assert isinstance(hint_result.metrics["hint_text"], str)


def test_check_all_backends_resolves_and_invokes_with_zero_agent_parameters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    admitted = _admit_all(tmp_path, monkeypatch)
    binding = admitted.binding_source().resolve(_CHECK_ALL_BACKENDS_ID)
    assert binding is not None

    result = binding.invoke(None, {})

    assert isinstance(result, OrganelleResult)
    assert result.status == "ok"
    assert "bwa" in result.metrics["backend_status"]


def test_cytonuclear_gwas_resolves_and_invokes_the_plan_only_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``executor`` is never Agent-exposed, so every real call plans, not runs."""
    admitted = _admit_all(tmp_path, monkeypatch)
    binding = admitted.binding_source().resolve(_CYTONUCLEAR_GWAS_ID)
    assert binding is not None

    vcf_path = tmp_path / "pop.vcf"
    vcf_path.write_text(_VCF_TEXT)
    phenotype_path = tmp_path / "pheno.txt"
    phenotype_path.write_text("1.0\n2.0\n3.0\n4.0\n")

    result = binding.invoke(None, {"vcf_path": str(vcf_path), "phenotype": str(phenotype_path)})

    assert isinstance(result, OrganelleResult)
    assert result.operation_id == _CYTONUCLEAR_GWAS_ID
    assert result.status == "ok"
    assert "gwas_planned" in result.flags


def test_call_variants_resolves_and_invokes_a_real_bam_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The domain's ``directory`` (``path_role="input"``) capability, proven for real.

    ``bam_dir`` is only ``str()``-embedded into the planned argv (never
    opened), but DIRECTORY-input containment still runs before the
    implementation is called - the directory must genuinely exist.
    """
    admitted = _admit_all(tmp_path, monkeypatch)
    binding = admitted.binding_source().resolve(_CALL_VARIANTS_ID)
    assert binding is not None

    bam_dir = tmp_path / "bams"
    bam_dir.mkdir()
    (bam_dir / "sample1.bam").write_bytes(b"not a real bam, just a real file")

    result = binding.invoke(None, {"bam_dir": str(bam_dir)})

    assert isinstance(result, OrganelleResult)
    assert result.operation_id == _CALL_VARIANTS_ID
    assert result.status == "ok"
    assert "call_planned" in result.flags


def test_prepare_gemma_input_resolves_without_creating_a_plan_only_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The agent plans the workflow without creating an output directory."""
    admitted = _admit_all(tmp_path, monkeypatch)
    binding = admitted.binding_source().resolve(_PREPARE_GEMMA_INPUT_ID)
    assert binding is not None

    vcf_path = tmp_path / "pop.vcf"
    vcf_path.write_text(_VCF_TEXT)
    phenotype_path = tmp_path / "pheno.txt"
    phenotype_path.write_text("1.0\n2.0\n3.0\n4.0\n")
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))

    result = binding.invoke(
        None,
        {
            "vcf_path": str(vcf_path),
            "phenotype": str(phenotype_path),
        },
    )

    assert isinstance(result, OrganelleResult)
    assert result.operation_id == _PREPARE_GEMMA_INPUT_ID
    assert result.status == "ok"
    assert not (tmp_path / "cache" / "runs" / "population.prepare_gemma_input").exists()
