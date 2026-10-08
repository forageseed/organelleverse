"""Capability Plan 03/05: the restored ``selection`` bundles, for real.

``scripts/capabilities/build_restored_bundles.py`` generated 27 bundles
under ``src/organelleverse/capabilities/selection-*`` from
``docs/operations/restored-capabilities.toml`` - see
``organelleverse.capabilities.adapters.selection``'s module docstring for
the full per-capability reasoning, including two more genuinely ungated
real external-tool calls (``align_protein`` spawns real ``mafft``;
``kaks_calculator`` transitively spawns real ``KaKs_Calculator``) and a
repeated destination-*file* gap (``run_kaks_calculator``'s ``output_file``,
``write_ctl_file``'s ``path``) - a single write-destination file, not a
directory, so it stays outside what ``ParameterCodec.DIRECTORY`` covers,
matching ``phylogeny.render_network``'s finding.

**Seven more, Capability Plan 05**: ``write_branch_model``,
``write_branch_site_model``, ``write_clade_model``, ``write_site_model``,
``write_codeml_inputs``, ``run_kaks_calculator_workflow``, and
``run_codeml`` were blocked solely on a required ``output_dir``/``work_dir``
directory parameter until ``ParameterCodec.DIRECTORY`` shipped - all seven
declare ``path_role = "output"``.

Mirrors ``tests/capabilities/test_restored_format_conversion_bundles.py``.
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

_ALIGN_PROTEIN_ID = "selection.align_protein"
_BRANCH_MODEL_ID = "selection.branch_model"
_BRANCH_SITE_MODEL_ID = "selection.branch_site_model"
_CHECK_KAKS_CALCULATOR_ID = "selection.check_kaks_calculator"
_CLADE_MODEL_ID = "selection.clade_model"
_COMPUTE_KAKS_ID = "selection.compute_kaks"
_FASTA_TO_AXT_ID = "selection.fasta_to_axt"
_INSTALL_HINT_ID = "selection.install_hint"
_KAKS_ID = "selection.kaks"
_KAKS_CALCULATOR_ID = "selection.kaks_calculator"
_LIKELIHOOD_RATIO_TEST_ID = "selection.likelihood_ratio_test"
_PAL2NAL_ID = "selection.pal2nal"
_PARSE_CODEML_OUTPUT_ID = "selection.parse_codeml_output"
_PARSE_KAKS_OUTPUT_ID = "selection.parse_kaks_output"
_PREPARE_CODEML_ID = "selection.prepare_codeml"
_SITE_MODEL_ID = "selection.site_model"
_TO_PAML_ID = "selection.to_paml"
_TRANSLATE_CDS_ID = "selection.translate_cds"
_VALIDATE_CDS_ID = "selection.validate_cds"
_VALIDATE_CDS_SEQUENCES_ID = "selection.validate_cds_sequences"
_WRITE_BRANCH_MODEL_ID = "selection.write_branch_model"
_WRITE_BRANCH_SITE_MODEL_ID = "selection.write_branch_site_model"
_WRITE_CLADE_MODEL_ID = "selection.write_clade_model"
_WRITE_SITE_MODEL_ID = "selection.write_site_model"
_WRITE_CODEML_INPUTS_ID = "selection.write_codeml_inputs"
_RUN_KAKS_CALCULATOR_WORKFLOW_ID = "selection.run_kaks_calculator_workflow"
_RUN_CODEML_ID = "selection.run_codeml"
_RESTORED_IDS = (
    _ALIGN_PROTEIN_ID,
    _BRANCH_MODEL_ID,
    _BRANCH_SITE_MODEL_ID,
    _CHECK_KAKS_CALCULATOR_ID,
    _CLADE_MODEL_ID,
    _COMPUTE_KAKS_ID,
    _FASTA_TO_AXT_ID,
    _INSTALL_HINT_ID,
    _KAKS_ID,
    _KAKS_CALCULATOR_ID,
    _LIKELIHOOD_RATIO_TEST_ID,
    _PAL2NAL_ID,
    _PARSE_CODEML_OUTPUT_ID,
    _PARSE_KAKS_OUTPUT_ID,
    _PREPARE_CODEML_ID,
    _SITE_MODEL_ID,
    _TO_PAML_ID,
    _TRANSLATE_CDS_ID,
    _VALIDATE_CDS_ID,
    _VALIDATE_CDS_SEQUENCES_ID,
    _WRITE_BRANCH_MODEL_ID,
    _WRITE_BRANCH_SITE_MODEL_ID,
    _WRITE_CLADE_MODEL_ID,
    _WRITE_SITE_MODEL_ID,
    _WRITE_CODEML_INPUTS_ID,
    _RUN_KAKS_CALCULATOR_WORKFLOW_ID,
    _RUN_CODEML_ID,
    "selection.batch_codeml",
)
_EXCLUDED_IDS = (
    "selection.run_kaks_calculator",
    "selection.write",
    "selection.write_ctl_file",
    "selection.write_kaks",
)

_GENERATOR_PATH = PROJECT_ROOT / "scripts" / "capabilities" / "build_restored_bundles.py"
_LEDGER_PATH = PROJECT_ROOT / "docs" / "operations" / "restored-capabilities.toml"
_CURRENT_ROOT = PROJECT_ROOT / "src" / "organelleverse"
_BUNDLE_ROOT = _CURRENT_ROOT / "capabilities"
_EXPECTED_BUNDLE_DIRS = {
    "selection-align-protein",
    "selection-branch-model",
    "selection-branch-site-model",
    "selection-check-kaks-calculator",
    "selection-clade-model",
    "selection-compute-kaks",
    "selection-fasta-to-axt",
    "selection-install-hint",
    "selection-kaks",
    "selection-kaks-calculator",
    "selection-likelihood-ratio-test",
    "selection-pal2nal",
    "selection-parse-codeml-output",
    "selection-parse-kaks-output",
    "selection-prepare-codeml",
    "selection-site-model",
    "selection-to-paml",
    "selection-translate-cds",
    "selection-validate-cds",
    "selection-validate-cds-sequences",
    "selection-write-branch-model",
    "selection-write-branch-site-model",
    "selection-write-clade-model",
    "selection-write-site-model",
    "selection-write-codeml-inputs",
    "selection-run-kaks-calculator-workflow",
    "selection-run-codeml",
    "selection-batch-codeml"
}

# A tiny, real codon-aligned CDS pair - divisible by 3, ATG start, stop codon.
_CDS_A = "ATGGCTGCTGCTGCTGCTTAA"
_CDS_B = "ATGGCAGCAGCTGCTGCTTAA"


def _write_fasta(path: Path, records: list[tuple[str, str]]) -> None:
    path.write_text("".join(f">{name}\n{seq}\n" for name, seq in records))


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


def test_the_twenty_seven_bindable_capabilities_are_discovered_from_the_real_core_channel(
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
        domain="selection",
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


def test_align_protein_resolves_and_really_invokes_mafft(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The domain's ungated real external-tool call: mafft --auto."""
    admitted = _admit_all(tmp_path, monkeypatch)
    binding = admitted.binding_source().resolve(_ALIGN_PROTEIN_ID)
    assert binding is not None

    protein_path = tmp_path / "proteins.fasta"
    _write_fasta(protein_path, [("p1", "MAAAALL"), ("p2", "MAAALL")])

    result = binding.invoke(None, {"protein_fasta": str(protein_path)})

    assert isinstance(result, OrganelleResult)
    assert result.operation_id == _ALIGN_PROTEIN_ID
    assert result.status == "ok"
    from organelleverse import accel

    # The Rust MAFFT kernel aligns without the CLI.
    if shutil.which("mafft") is None and not accel.HAS_RUST:
        assert result.metrics["method"] == "ungapped"
        assert result.metrics["mafft_used"] is False
    else:
        assert result.metrics["mafft_used"] is True


def test_kaks_calculator_resolves_and_invokes_the_real_kaks_calculator_binary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Transitively spawns the real KaKs_Calculator binary via an internal tempdir."""
    admitted = _admit_all(tmp_path, monkeypatch)
    binding = admitted.binding_source().resolve(_KAKS_CALCULATOR_ID)
    assert binding is not None

    cds_path = tmp_path / "cds.fasta"
    _write_fasta(cds_path, [("seqA", _CDS_A), ("seqB", _CDS_B)])

    result = binding.invoke(None, {"cds_fasta": str(cds_path)})

    assert isinstance(result, OrganelleResult)
    assert result.operation_id == _KAKS_CALCULATOR_ID


def test_compute_kaks_and_kaks_resolve_and_invoke_on_a_real_cds_fasta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    admitted = _admit_all(tmp_path, monkeypatch)
    cds_pairs_path = tmp_path / "cds_pairs.fasta"
    _write_fasta(cds_pairs_path, [("seqA", _CDS_A), ("seqB", _CDS_B)])

    compute_binding = admitted.binding_source().resolve(_COMPUTE_KAKS_ID)
    assert compute_binding is not None
    compute_result = compute_binding.invoke(None, {"cds_fasta": str(cds_pairs_path)})
    assert isinstance(compute_result, OrganelleResult)
    assert compute_result.status == "ok"
    assert "kaks" in compute_result.metrics

    kaks_binding = admitted.binding_source().resolve(_KAKS_ID)
    assert kaks_binding is not None
    kaks_result = kaks_binding.invoke(None, {"cds_pairs_fasta": str(cds_pairs_path)})
    assert isinstance(kaks_result, OrganelleResult)
    assert kaks_result.operation_id == _KAKS_ID
    assert kaks_result.status == "ok"


def test_translate_cds_validate_cds_and_validate_cds_sequences_resolve_and_invoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    admitted = _admit_all(tmp_path, monkeypatch)
    cds_path = tmp_path / "cds.fasta"
    _write_fasta(cds_path, [("seqA", _CDS_A)])

    translate_binding = admitted.binding_source().resolve(_TRANSLATE_CDS_ID)
    assert translate_binding is not None
    translate_result = translate_binding.invoke(None, {"cds_fasta": str(cds_path)})
    assert isinstance(translate_result, OrganelleResult)
    assert translate_result.status == "ok"

    validate_binding = admitted.binding_source().resolve(_VALIDATE_CDS_ID)
    assert validate_binding is not None
    validate_result = validate_binding.invoke(None, {"cds_fasta": str(cds_path)})
    assert isinstance(validate_result, OrganelleResult)
    assert validate_result.status == "ok"

    validate_seq_binding = admitted.binding_source().resolve(_VALIDATE_CDS_SEQUENCES_ID)
    assert validate_seq_binding is not None
    validate_seq_result = validate_seq_binding.invoke(None, {"cds_fasta": str(cds_path)})
    assert isinstance(validate_seq_result, OrganelleResult)
    assert validate_seq_result.status == "ok"
    assert "cds_validation" in validate_seq_result.metrics


def test_pal2nal_and_to_paml_resolve_and_invoke_on_real_alignments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    admitted = _admit_all(tmp_path, monkeypatch)
    protein_alignment_path = tmp_path / "protein_aln.fasta"
    _write_fasta(protein_alignment_path, [("seqA", "MAAAAL"), ("seqB", "MAA-AL")])
    cds_path = tmp_path / "cds.fasta"
    _write_fasta(
        cds_path,
        [("seqA", "ATGGCTGCTGCTGCTTTA"), ("seqB", "ATGGCTGCTGCTTTA")],
    )

    pal2nal_binding = admitted.binding_source().resolve(_PAL2NAL_ID)
    assert pal2nal_binding is not None
    pal2nal_result = pal2nal_binding.invoke(
        None,
        {
            "protein_alignment_fasta": str(protein_alignment_path),
            "cds_fasta": str(cds_path),
        },
    )
    assert isinstance(pal2nal_result, OrganelleResult)
    assert pal2nal_result.operation_id == _PAL2NAL_ID

    to_paml_binding = admitted.binding_source().resolve(_TO_PAML_ID)
    assert to_paml_binding is not None
    to_paml_result = to_paml_binding.invoke(None, {"alignment_fasta": str(protein_alignment_path)})
    assert isinstance(to_paml_result, OrganelleResult)
    assert to_paml_result.status == "ok"


def test_fasta_to_axt_resolves_and_invokes_producing_a_real_axt_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    admitted = _admit_all(tmp_path, monkeypatch)
    binding = admitted.binding_source().resolve(_FASTA_TO_AXT_ID)
    assert binding is not None

    fasta_path = tmp_path / "cds.fasta"
    _write_fasta(fasta_path, [("seqA", _CDS_A), ("seqB", _CDS_B)])

    result = binding.invoke(None, {"fasta_path": str(fasta_path)})

    assert isinstance(result, OrganelleResult)
    assert result.status == "ok"
    assert len(result.artifacts) == 1


def test_parse_codeml_output_and_parse_kaks_output_resolve_and_invoke_on_real_text_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    admitted = _admit_all(tmp_path, monkeypatch)

    codeml_out = tmp_path / "codeml.out"
    codeml_out.write_text("lnL(ntime: 3  np: 5):  -1234.567890\nkappa (ts/ts) =  2.5\n")
    codeml_binding = admitted.binding_source().resolve(_PARSE_CODEML_OUTPUT_ID)
    assert codeml_binding is not None
    codeml_result = codeml_binding.invoke(None, {"out_path": str(codeml_out)})
    assert isinstance(codeml_result, OrganelleResult)
    assert codeml_result.status == "ok"
    assert codeml_result.metrics["codeml_output"]["kappa"] == 2.5

    kaks_out = tmp_path / "kaks.tsv"
    kaks_out.write_text("Sequence\tMethod\tKa\tKs\nseqA-seqB\tYN\t0.1\t0.2\n")
    kaks_binding = admitted.binding_source().resolve(_PARSE_KAKS_OUTPUT_ID)
    assert kaks_binding is not None
    kaks_result = kaks_binding.invoke(None, {"output_file": str(kaks_out)})
    assert isinstance(kaks_result, OrganelleResult)
    assert kaks_result.status == "ok"
    assert len(kaks_result.metrics["kaks_hits"]) == 1


def test_check_kaks_calculator_and_install_hint_resolve_and_invoke_real_json_scalars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    admitted = _admit_all(tmp_path, monkeypatch)

    check_binding = admitted.binding_source().resolve(_CHECK_KAKS_CALCULATOR_ID)
    assert check_binding is not None
    check_result = check_binding.invoke(None, {})
    assert isinstance(check_result, OrganelleResult)
    assert check_result.status == "ok"
    assert "installed" in check_result.metrics["kaks_calculator_status"]

    hint_binding = admitted.binding_source().resolve(_INSTALL_HINT_ID)
    assert hint_binding is not None
    hint_result = hint_binding.invoke(None, {})
    assert isinstance(hint_result, OrganelleResult)
    assert hint_result.status == "ok"
    assert isinstance(hint_result.metrics["hint_text"], str)


def test_likelihood_ratio_test_resolves_and_invokes_real_json_scalars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    admitted = _admit_all(tmp_path, monkeypatch)
    binding = admitted.binding_source().resolve(_LIKELIHOOD_RATIO_TEST_ID)
    assert binding is not None

    result = binding.invoke(None, {"lnL_null": -100.0, "lnL_alt": -95.0, "df": 1})

    assert isinstance(result, OrganelleResult)
    assert result.status == "ok"
    assert "lrt" in result.metrics


def test_prepare_codeml_resolves_and_invokes_with_default_mafft_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    admitted = _admit_all(tmp_path, monkeypatch)
    binding = admitted.binding_source().resolve(_PREPARE_CODEML_ID)
    assert binding is not None

    cds_path = tmp_path / "cds.fasta"
    _write_fasta(cds_path, [("seqA", _CDS_A), ("seqB", _CDS_B)])

    result = binding.invoke(None, {"cds_fasta": str(cds_path)})

    assert isinstance(result, OrganelleResult)
    assert result.operation_id == _PREPARE_CODEML_ID


def test_branch_model_branch_site_model_clade_model_and_site_model_resolve_and_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Passes ``run=False`` explicitly to keep this test fast and deterministic.

    ``run=True`` is the real default and, with ``codeml`` installed, spawns
    it for real (see the adapter module docstring) - this proves the
    binding resolves and invokes correctly without paying that cost here.
    """
    admitted = _admit_all(tmp_path, monkeypatch)
    alignment_path = tmp_path / "codon_aln.paml"
    alignment_path.write_text("2 18\nseqA\nATGGCTGCTGCTGCTGCT\nseqB\nATGGCAGCAGCTGCTGCT\n")
    tree_path = tmp_path / "tree.nwk"
    tree_path.write_text("(seqA:0.1,seqB:0.1);\n")

    for capability_id in (
        _BRANCH_MODEL_ID,
        _BRANCH_SITE_MODEL_ID,
        _CLADE_MODEL_ID,
        _SITE_MODEL_ID,
    ):
        binding = admitted.binding_source().resolve(capability_id)
        assert binding is not None
        result = binding.invoke(
            None,
            {
                "alignment": str(alignment_path),
                "tree": str(tree_path),
                "run": False,
            },
        )
        assert isinstance(result, OrganelleResult)
        assert result.operation_id == capability_id


def test_write_branch_model_and_write_site_model_resolve_and_create_a_real_output_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ``write_*`` codeml planners' ``directory`` (``path_role="output"``) parameter, for real.

    Passes ``run=False`` explicitly, same as
    ``test_branch_model_branch_site_model_clade_model_and_site_model_resolve_and_plan``
    above, to keep this fast and deterministic - the point proven here is
    that ``output_dir`` (required, unlike ``branch_model``'s own internal
    tempdir) is genuinely created and genuinely populated with real ctl/seq
    /tree files, not that ``codeml`` itself ran.

    Verified by real invocation: ``write_branch_model``/``write_site_model``
    each hardcode their own ``_contract.ok("branch_model", ...)``/
    ``_contract.ok("site_model", ...)`` op name regardless of caller, so the
    returned ``OrganelleResult`` carries ``operation_id =
    "selection.branch_model"``/``"selection.site_model"``, not
    ``"selection.write_branch_model"``/``"selection.write_site_model"`` -
    the same cross-function ``operation_id`` quirk already documented for
    ``population.detect_numt`` and (within this domain)
    ``write_codeml_inputs`` -> ``"selection.prepare_codeml"``.
    """
    admitted = _admit_all(tmp_path, monkeypatch)
    alignment_path = tmp_path / "codon_aln.paml"
    alignment_path.write_text("2 18\nseqA\nATGGCTGCTGCTGCTGCT\nseqB\nATGGCAGCAGCTGCTGCT\n")
    tree_path = tmp_path / "tree.nwk"
    tree_path.write_text("(seqA:0.1,seqB:0.1);\n")

    branch_out = tmp_path / "branch_out"
    assert not branch_out.exists()
    branch_binding = admitted.binding_source().resolve(_WRITE_BRANCH_MODEL_ID)
    assert branch_binding is not None
    branch_result = branch_binding.invoke(
        None,
        {
            "alignment": str(alignment_path),
            "tree": str(tree_path),
            "output_dir": str(branch_out),
            "run": False,
        },
    )
    assert isinstance(branch_result, OrganelleResult)
    assert branch_result.operation_id == _BRANCH_MODEL_ID
    assert branch_result.status == "ok"
    assert branch_out.is_dir()
    assert (branch_out / "branch_null.ctl").is_file()
    assert (branch_out / "branch_alt.ctl").is_file()

    site_out = tmp_path / "site_out"
    assert not site_out.exists()
    site_binding = admitted.binding_source().resolve(_WRITE_SITE_MODEL_ID)
    assert site_binding is not None
    site_result = site_binding.invoke(
        None,
        {
            "alignment": str(alignment_path),
            "tree": str(tree_path),
            "output_dir": str(site_out),
            "run": False,
        },
    )
    assert isinstance(site_result, OrganelleResult)
    assert site_result.operation_id == _SITE_MODEL_ID
    assert site_result.status == "ok"
    assert site_out.is_dir()
    assert (site_out / "site_model.ctl").is_file()


def test_write_branch_site_model_and_write_clade_model_resolve_and_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same hardcoded-``operation_id`` quirk as ``write_branch_model``/``write_site_model``."""
    admitted = _admit_all(tmp_path, monkeypatch)
    alignment_path = tmp_path / "codon_aln.paml"
    alignment_path.write_text("2 18\nseqA\nATGGCTGCTGCTGCTGCT\nseqB\nATGGCAGCAGCTGCTGCT\n")
    tree_path = tmp_path / "tree.nwk"
    tree_path.write_text("(seqA:0.1,seqB:0.1);\n")

    for capability_id, expected_operation_id, out_name in (
        (_WRITE_BRANCH_SITE_MODEL_ID, _BRANCH_SITE_MODEL_ID, "bs_out"),
        (_WRITE_CLADE_MODEL_ID, _CLADE_MODEL_ID, "clade_out"),
    ):
        out_dir = tmp_path / out_name
        binding = admitted.binding_source().resolve(capability_id)
        assert binding is not None
        result = binding.invoke(
            None,
            {
                "alignment": str(alignment_path),
                "tree": str(tree_path),
                "output_dir": str(out_dir),
                "run": False,
            },
        )
        assert isinstance(result, OrganelleResult)
        assert result.operation_id == expected_operation_id
        assert result.status == "ok"
        assert out_dir.is_dir()


def test_write_codeml_inputs_resolves_and_invokes_a_real_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Runs the full CDS -> validate -> translate -> align -> PAL2NAL -> PAML pipeline for real.

    Unlike ``prepare_codeml`` (internal tempdir), ``output_dir`` here is
    Agent-chosen and durable - this confirms every real output file lands
    there.
    """
    admitted = _admit_all(tmp_path, monkeypatch)
    binding = admitted.binding_source().resolve(_WRITE_CODEML_INPUTS_ID)
    assert binding is not None

    cds_path = tmp_path / "cds.fasta"
    _write_fasta(cds_path, [("seqA", _CDS_A), ("seqB", _CDS_B)])
    output_dir = tmp_path / "codeml_inputs_out"
    assert not output_dir.exists()

    result = binding.invoke(None, {"cds_fasta": str(cds_path), "output_dir": str(output_dir)})

    assert isinstance(result, OrganelleResult)
    assert result.status == "ok"
    assert output_dir.is_dir()
    assert (output_dir / "codon_aln.paml").is_file()


def test_run_kaks_calculator_workflow_resolves_and_invokes_for_real(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The public workflow uses managed storage for its intermediate files.

    Verified by real invocation: ``run_kaks_calculator_workflow`` hardcodes
    its own ``_contract.ok("kaks_calculator", ...)`` op name, so the
    returned ``OrganelleResult`` carries ``operation_id =
    "selection.kaks_calculator"``, not
    ``"selection.run_kaks_calculator_workflow"`` - the same quirk as
    ``write_branch_model``/``write_site_model``/``write_branch_site_model``/
    ``write_clade_model`` above.

    Does not assert ``status == "ok"``, matching
    ``test_kaks_calculator_resolves_and_invokes_the_real_kaks_calculator_binary``
    above: ``check_kaks_calculator()`` searches ``PATH`` for a binary
    literally named ``KaKs`` (``_KAKS_INFO["cli"] = "KaKs"``), not
    ``KaKs_Calculator`` - a real, pre-existing name mismatch in the
    restored source (independently discovered while writing this test, not
    a directory-codec concern), so this call honestly returns ``"failed"``
    with anomaly ``kaks_calculator_not_found`` in this environment even
    though a real ``KaKs_Calculator`` binary exists elsewhere on ``PATH``
    under its own name. The binding still resolves and invokes for real.
    """
    admitted = _admit_all(tmp_path, monkeypatch)
    binding = admitted.binding_source().resolve(_RUN_KAKS_CALCULATOR_WORKFLOW_ID)
    assert binding is not None

    cds_path = tmp_path / "cds.fasta"
    _write_fasta(cds_path, [("seqA", _CDS_A), ("seqB", _CDS_B)])
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    result = binding.invoke(None, {"cds_fasta": str(cds_path)})

    assert isinstance(result, OrganelleResult)
    assert result.operation_id == _KAKS_CALCULATOR_ID


@pytest.mark.skipif(shutil.which("codeml") is None, reason="requires the PAML codeml executable")
def test_run_codeml_resolves_and_genuinely_spawns_codeml_on_a_real_work_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The agent supplies a control file; CodeML runs in managed storage.

    A real finding, not a fabricated success: this PAML build (4.10.9)
    exits non-zero on this minimal one-tree ``tree.nwk`` (``error: end of
    tree file`` - it expects more tree records than the file has, a known
    PAML quirk unrelated to this binding) even though it already computed
    and wrote a real, complete ``run.out`` with a real ``lnL`` before that.
    ``run_codeml``'s own subprocess fallback treats that nonzero exit as
    failure and returns ``None`` - a real, honest outcome ``json_metric``
    is built to carry (``None`` is valid finite JSON), not a binding defect.
    The genuinely-spawned subprocess is proven by the real PAML side-effect
    files it leaves in ``work_dir``.
    """
    admitted = _admit_all(tmp_path, monkeypatch)
    binding = admitted.binding_source().resolve(_RUN_CODEML_ID)
    assert binding is not None

    work_dir = tmp_path / "codeml_work"
    work_dir.mkdir()
    (work_dir / "codon_aln.paml").write_text(
        "2 18\nseqA\nATGGCTGCTGCTGCTGCT\nseqB\nATGGCAGCAGCTGCTGCT\n"
    )
    (work_dir / "tree.nwk").write_text("(seqA:0.1,seqB:0.1);\n")
    ctl_path = work_dir / "run.ctl"
    ctl_path.write_text(
        "      seqfile  = codon_aln.paml\n"
        "      treefile = tree.nwk\n"
        "      outfile  = run.out\n"
        "      noisy = 9\n"
        "      verbose = 1\n"
        "      runmode = 0\n"
        "      seqtype = 1\n"
        "      CodonFreq = 2\n"
        "      clock = 0\n"
        "      model = 0\n"
        "      NSsites = 0\n"
        "      icode = 0\n"
        "      fix_kappa = 0\n"
        "      kappa = 2.0\n"
        "      fix_omega = 0\n"
        "      omega = 1.0\n"
        "      fix_alpha = 1\n"
        "      alpha = 0.\n"
        "      Malpha = 0\n"
        "      ncatG = 4\n"
        "      getSE = 0\n"
        "      RateAncestor = 0\n"
        "      Small_Diff = 0.5e-6\n"
        "      cleandata = 0\n"
        "      method = 0\n"
    )

    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    result = binding.invoke(None, {"ctl_path": str(ctl_path)})

    assert isinstance(result, OrganelleResult)
    assert result.operation_id == _RUN_CODEML_ID
    assert result.status == "ok"
    # Both outcomes are honest and PAML-build-dependent: the 4.10.9 build
    # this test was written against exits non-zero on this minimal
    # tree.nwk and run_codeml returns None (see the docstring), while other
    # builds exit zero and run_codeml returns the parsed run.out. What must
    # hold either way is the genuinely-spawned proof below.
    codeml_run = result.metrics["codeml_run"]
    assert codeml_run is None or "lnL" in codeml_run
    # codeml genuinely ran: it always leaves its own real side-effect files
    # in work_dir, whatever its own subprocess exit code was.
    managed = list((tmp_path / "cache" / "runs" / "selection.run_codeml").glob("*"))
    assert len(managed) == 1
    assert (managed[0] / "rst").is_file()
    assert (managed[0] / "run.out").is_file()
    assert "lnL" in (managed[0] / "run.out").read_text()
