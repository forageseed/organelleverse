import inspect
from pathlib import Path

import pytest
from pydantic import ValidationError

from organelleverse import assembly
from organelleverse.assembly import (
    AssemblyAuxiliary,
    ContigInput,
    LongReadLibrary,
    ShortReadLibrary,
)
from organelleverse.assembly import routing as routing_module
from organelleverse.assembly.backends import BACKENDS, AssemblyProfile, AuxiliaryRole
from organelleverse.assembly.contracts import (
    AssemblyRequest,
    GenomeSizeEvidence,
    LongTechnology,
    QualityState,
    ShortLayout,
)
from organelleverse.assembly.routing import (
    AssemblyRoute,
    AutoRouteRule,
    RejectedBackend,
    RoutingContext,
    classify_profile,
    prepare_routing_context,
    resolve_backend,
)
from organelleverse.core.data import OrganelleData
from organelleverse.core.errors import OrganelleInputError, OrganelleNeedsInput
from organelleverse.io import read_reads

_FULL_STATIC_CATALOG = tuple(BACKENDS.ids())


def _fastq(path: Path) -> Path:
    path.write_text("@r\nACGT\n+\n!!!!\n")
    return path


def _fasta(path: Path) -> Path:
    path.write_text(">ref\nACGTACGT\n")
    return path


def _config(path: Path) -> Path:
    path.write_text("correction_backend=canu\n")
    return path


def _illumina(
    tmp_path: Path,
    *,
    layout: ShortLayout = "paired_end",
) -> OrganelleData:
    read1 = _fastq(tmp_path / "R1.fastq")
    read2 = _fastq(tmp_path / "R2.fastq") if layout == "paired_end" else None
    return read_reads(
        short_libraries=(
            ShortReadLibrary(
                technology="illumina",
                layout=layout,
                read1=read1,
                read2=read2,
                read_length=150,
                insert_size=350 if layout == "paired_end" else None,
            ),
        )
    )


def _long(
    tmp_path: Path,
    technology: LongTechnology,
    quality_state: QualityState,
    auxiliary: AssemblyAuxiliary | None = None,
) -> OrganelleData:
    return read_reads(
        long_libraries=(
            LongReadLibrary(
                technology=technology,
                quality_state=quality_state,
                reads=_fastq(tmp_path / f"{technology}.fastq"),
            ),
        ),
        auxiliary=auxiliary or AssemblyAuxiliary(),
    )


@pytest.mark.parametrize(
    ("technology", "quality_state"),
    [
        ("pacbio_hifi", "ccs"),
        ("pacbio_clr", "raw"),
        ("pacbio_clr", "corrected"),
        ("ont", "raw"),
        ("ont", "corrected"),
        ("ont", "hq"),
        ("ont", "duplex"),
    ],
)
def test_himt_explicit_route_for_all_approved_profiles(
    tmp_path: Path,
    technology: LongTechnology,
    quality_state: QualityState,
) -> None:
    data = _long(tmp_path, technology, quality_state)
    route = resolve_backend(data, organelle="mitochondrion", method="himt")
    assert route.selected_backend == "himt"
    assert route.rule_id == "explicit.backend"


def test_plastid_hifi_auto_route_is_oatk(tmp_path: Path) -> None:
    data = _long(
        tmp_path,
        "pacbio_hifi",
        "ccs",
        AssemblyAuxiliary(hmm_profiles=_fasta(tmp_path / "profiles.hmm")),
    )
    route = resolve_backend(data, organelle="plastid")
    assert route.selected_backend == "oatk"
    assert route.rule_id == "auto.hifi_oatk"
    assert route.profile.value == "pacbio_hifi"


def test_mitochondrial_hifi_auto_route_requires_genome_size(tmp_path: Path) -> None:
    data = _long(tmp_path, "pacbio_hifi", "ccs")

    with pytest.raises(OrganelleNeedsInput) as raised:
        resolve_backend(data, organelle="mitochondrion")

    assert raised.value.needs_input.field == "auxiliary.genome_size"


def test_reads_and_contigs_cannot_be_mixed(tmp_path: Path) -> None:
    data = read_reads(
        short_libraries=(
            ShortReadLibrary(
                technology="illumina",
                layout="single_end",
                read1=_fastq(tmp_path / "short.fastq"),
                read_length=150,
            ),
        ),
        contig_inputs=(ContigInput(fasta=_fasta(tmp_path / "contigs.fasta")),),
    )

    with pytest.raises(OrganelleInputError) as raised:
        classify_profile(data)

    assert raised.value.code == "assembly.unsupported_data_profile"
    assert raised.value.as_dict()["details"]["contig_inputs"] == 1


def test_empty_structured_input_is_rejected() -> None:
    with pytest.raises(ValidationError, match="assembly input requires reads or contigs"):
        read_reads()


def test_mixed_illumina_layouts_are_rejected(tmp_path: Path) -> None:
    data = read_reads(
        short_libraries=(
            ShortReadLibrary(
                technology="illumina",
                layout="paired_end",
                read1=_fastq(tmp_path / "paired_R1.fastq"),
                read2=_fastq(tmp_path / "paired_R2.fastq"),
                read_length=150,
                insert_size=350,
            ),
            ShortReadLibrary(
                technology="illumina",
                layout="single_end",
                read1=_fastq(tmp_path / "single.fastq"),
                read_length=150,
            ),
        )
    )

    with pytest.raises(OrganelleInputError) as raised:
        classify_profile(data)

    assert raised.value.code == "assembly.unsupported_data_profile"
    assert raised.value.as_dict()["details"]["short_layouts"] == [
        "paired_end",
        "single_end",
    ]


def test_hifi_with_short_reads_remains_hifi_and_requires_genome_size(tmp_path: Path) -> None:
    data = read_reads(
        short_libraries=(
            ShortReadLibrary(
                technology="illumina",
                layout="single_end",
                read1=_fastq(tmp_path / "short.fastq"),
                read_length=150,
            ),
        ),
        long_libraries=(
            LongReadLibrary(
                technology="pacbio_hifi",
                quality_state="ccs",
                reads=_fastq(tmp_path / "hifi.fastq"),
            ),
        ),
        auxiliary=AssemblyAuxiliary(hmm_profiles=_fasta(tmp_path / "profiles.hmm")),
    )

    assert classify_profile(data) == AssemblyProfile.PACBIO_HIFI
    with pytest.raises(OrganelleNeedsInput):
        resolve_backend(data, organelle="mitochondrion")


def test_ont_duplex_has_no_automatic_route(tmp_path: Path) -> None:
    data = _long(tmp_path, "ont", "duplex")

    with pytest.raises(OrganelleInputError) as raised:
        resolve_backend(data, organelle="mitochondrion")

    error = raised.value.as_dict()
    assert error["error_code"] == "assembly.unsupported_data_profile"
    assert error["details"]["profile"] == "ont_duplex"
    assert error["details"]["compatible_candidates"] == ["himt", "pmat"]


def test_route_models_are_frozen_and_reject_unknown_fields(tmp_path: Path) -> None:
    route = resolve_backend(_long(tmp_path, "pacbio_hifi", "ccs"), organelle="plastid")

    with pytest.raises(ValidationError) as frozen_route:
        route.selected_backend = "other"
    assert frozen_route.value.errors()[0]["type"] == "frozen_instance"

    route_payload = route.model_dump(mode="python", round_trip=True)
    route_payload["unexpected"] = True
    with pytest.raises(ValidationError) as extra_route:
        AssemblyRoute.model_validate(route_payload)
    assert extra_route.value.errors()[0]["type"] == "extra_forbidden"

    rejected = RejectedBackend(
        backend_id="probe",
        reason_code="assembly.not_selected",
    )
    with pytest.raises(ValidationError) as frozen_rejected:
        rejected.reason_code = "other"
    assert frozen_rejected.value.errors()[0]["type"] == "frozen_instance"

    rejected_payload = rejected.model_dump(mode="python", round_trip=True)
    rejected_payload["unexpected"] = True
    with pytest.raises(ValidationError) as extra_rejected:
        RejectedBackend.model_validate(rejected_payload)
    assert extra_rejected.value.errors()[0]["type"] == "extra_forbidden"


@pytest.mark.parametrize(
    ("technology", "quality_state", "expected_rule"),
    [
        ("pacbio_clr", "raw", "auto.raw_clr_pmat"),
        ("ont", "raw", "auto.raw_ont_pmat"),
    ],
)
def test_raw_long_reads_auto_route_pmat(
    tmp_path: Path,
    technology: LongTechnology,
    quality_state: QualityState,
    expected_rule: str,
) -> None:
    data = _long(
        tmp_path,
        technology,
        quality_state,
        AssemblyAuxiliary(
            genome_size=500_000_000,
            correction_config=_config(tmp_path / "correction.cfg"),
        ),
    )
    route = resolve_backend(
        data,
        organelle="mitochondrion",
        released_backend_ids=_FULL_STATIC_CATALOG,
    )
    assert route.selected_backend == "pmat"
    assert route.rule_id == expected_rule


@pytest.mark.parametrize("layout", ["paired_end", "single_end"])
def test_illumina_auto_route_getorganelle(
    tmp_path: Path,
    layout: ShortLayout,
) -> None:
    route = resolve_backend(
        _illumina(tmp_path, layout=layout),
        organelle="plastid",
        released_backend_ids=_FULL_STATIC_CATALOG,
    )
    assert route.selected_backend == "getorganelle"


def test_corrected_clr_has_candidates_but_no_auto_backend(tmp_path: Path) -> None:
    data = _long(
        tmp_path,
        "pacbio_clr",
        "corrected",
        AssemblyAuxiliary(genome_size=500_000_000),
    )
    with pytest.raises(OrganelleInputError) as raised:
        resolve_backend(data, organelle="mitochondrion")
    error = raised.value.as_dict()
    assert error["error_code"] == "assembly.unsupported_data_profile"
    assert error["details"]["compatible_candidates"] == ["himt", "pmat"]


def test_explicit_oatk_rejects_ont_hq(tmp_path: Path) -> None:
    data = _long(tmp_path, "ont", "hq")
    with pytest.raises(OrganelleInputError) as raised:
        resolve_backend(data, organelle="mitochondrion", method="oatk")
    assert raised.value.code == "assembly.unsupported_data_profile"


def test_unknown_explicit_backend_has_stable_input_error(tmp_path: Path) -> None:
    data = _long(tmp_path, "ont", "hq")
    with pytest.raises(OrganelleInputError) as raised:
        resolve_backend(data, organelle="mitochondrion", method="not-a-backend")
    assert raised.value.code == "assembly.unknown_backend"
    assert raised.value.as_dict()["details"]["backend"] == "not-a-backend"


def test_oatk_routes_without_hmm_profiles_using_the_managed_default(tmp_path: Path) -> None:
    """hmm_profiles is not a hard routing requirement: Oatk's managed OatkDB
    profile is provisioned automatically by the assembly service. Supplying
    AssemblyAuxiliary.hmm_profiles overrides that managed default rather than
    gating whether the route is reachable at all."""
    data = _long(tmp_path, "pacbio_hifi", "ccs")
    route = resolve_backend(data, organelle="plastid", method="oatk")
    assert route.selected_backend == "oatk"


def test_oatk_accepts_an_explicit_hmm_profiles_override(tmp_path: Path) -> None:
    data = _long(
        tmp_path,
        "pacbio_hifi",
        "ccs",
        AssemblyAuxiliary(hmm_profiles=_fasta(tmp_path / "profiles.hmm")),
    )
    route = resolve_backend(data, organelle="plastid", method="oatk")
    assert route.selected_backend == "oatk"


def test_pmat_raw_reads_require_only_correction_config(tmp_path: Path) -> None:
    """PMAT no longer unconditionally requires genome_size; only the raw CLR/ONT
    correction_config remains a conditional routing requirement."""
    data = _long(tmp_path, "ont", "raw")
    with pytest.raises(OrganelleInputError) as raised:
        resolve_backend(
            data,
            organelle="mitochondrion",
            method="pmat",
            released_backend_ids=_FULL_STATIC_CATALOG,
        )
    assert raised.value.as_dict()["details"]["missing"] == ["correction_config"]


@pytest.mark.parametrize(
    ("technology", "quality_state", "expected_rule"),
    [
        ("pacbio_clr", "raw", "auto.raw_clr_pmat"),
        ("ont", "raw", "auto.raw_ont_pmat"),
    ],
)
def test_pmat_raw_routes_without_genome_size_when_correction_config_present(
    tmp_path: Path,
    technology: LongTechnology,
    quality_state: QualityState,
    expected_rule: str,
) -> None:
    """Genome size is supplied as traceable evidence, not gated at routing: a
    raw long-read input with only a correction route resolves to PMAT."""
    data = _long(
        tmp_path,
        technology,
        quality_state,
        AssemblyAuxiliary(correction_config=_config(tmp_path / "correction.cfg")),
    )
    route = resolve_backend(
        data,
        organelle="mitochondrion",
        released_backend_ids=_FULL_STATIC_CATALOG,
    )
    assert route.selected_backend == "pmat"
    assert route.rule_id == expected_rule


def test_routing_does_not_import_install_or_scan_environment(tmp_path: Path) -> None:
    data = _long(tmp_path, "pacbio_hifi", "ccs")
    route = resolve_backend(data, organelle="plastid")
    assert route.selected_backend == "oatk"
    source = inspect.getsource(routing_module)
    assert "assembly.install" not in source
    assert "shutil.which" not in source


def test_assembly_exports_pure_routing_api() -> None:
    assert assembly.AssemblyRoute is routing_module.AssemblyRoute
    assert assembly.classify_profile is classify_profile
    assert assembly.compatible_backends is routing_module.compatible_backends
    assert assembly.resolve_backend is resolve_backend


def test_auxiliary_role_registry_covers_genome_size_evidence_roles() -> None:
    """The closed auxiliary role registry must expose the traceable genome-size
    evidence and PMAT2 companion-executable roles so that downstream tasks can
    reference them by canonical name instead of free-form strings."""
    assert AuxiliaryRole.GENOME_SIZE_REPORT.value == "genome_size_report"
    assert AuxiliaryRole.CANU_EXECUTABLE.value == "canu_executable"
    assert AuxiliaryRole.NEXTDENOVO_EXECUTABLE.value == "nextdenovo_executable"
    # The new roles are optional and must not destabilize the released routing
    # surface: every released backend id still resolves through the registry.
    assert "oatk" in routing_module.BACKENDS
    assert "himt" in routing_module.BACKENDS


# ---------------------------------------------------------------------------
# Release-aware declarative routing: RoutingContext / AutoRouteRule
# ---------------------------------------------------------------------------


_RELEASED_WITH_PMAT = {"oatk", "himt", "pmat"}
_RELEASED_WITHOUT_PMAT = {"oatk", "himt"}


def _hifi_request(
    tmp_path: Path,
    *,
    genome_size: int | None,
    taxon_group: str = "plant",
) -> AssemblyRequest:
    auxiliary = AssemblyAuxiliary()
    if genome_size is not None:
        auxiliary = AssemblyAuxiliary(
            genome_size=genome_size,
            genome_size_evidence=GenomeSizeEvidence(
                source="user",
                genome_size_bp=genome_size,
            ),
        )
    data = read_reads(
        long_libraries=(
            LongReadLibrary(
                technology="pacbio_hifi",
                quality_state="ccs",
                reads=_fastq(tmp_path / "hifi.fastq"),
            ),
        ),
        auxiliary=auxiliary,
    )
    return AssemblyRequest(
        data=data,
        organelle="mitochondrion",
        taxon_group=taxon_group,  # type: ignore[arg-type]
    )


def _context(
    *,
    total_bases: int | None,
    genome_size_bp: int | None,
    taxon_group: str = "plant",
) -> RoutingContext:
    return RoutingContext(
        profile=AssemblyProfile.PACBIO_HIFI,
        taxon_group=taxon_group,  # type: ignore[arg-type]
        total_bases=total_bases,
        genome_size_bp=genome_size_bp,
        threshold_multiplier=3 if genome_size_bp is not None else None,
    )


def test_default_routing_uses_only_the_released_runtime_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        routing_module,
        "_released_backend_ids",
        lambda: tuple(sorted(_RELEASED_WITHOUT_PMAT)),
    )
    with pytest.raises(OrganelleInputError) as raised:
        resolve_backend(_illumina(tmp_path), organelle="plastid")
    assert raised.value.code == "assembly.unsupported_data_profile"
    error = raised.value.as_dict()
    assert error["details"]["compatible_candidates"] == []
    assert error["suggested_action"]["select_explicit_backend"] == []


def test_default_route_does_not_advertise_unreleased_compatible_backends(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        routing_module,
        "_released_backend_ids",
        lambda: tuple(sorted(_RELEASED_WITHOUT_PMAT)),
    )
    route = resolve_backend(
        _long(tmp_path, "pacbio_hifi", "ccs"),
        organelle="mitochondrion",
    )
    assert route.compatible_candidates == ("oatk", "himt")
    assert {item.backend_id for item in route.rejected_candidates} == {"himt"}


def test_routing_rejects_a_context_for_a_different_data_profile(tmp_path: Path) -> None:
    data = _long(tmp_path, "pacbio_hifi", "ccs")
    context = RoutingContext(profile=AssemblyProfile.ILLUMINA_PE, taxon_group="plant")
    with pytest.raises(OrganelleInputError) as raised:
        resolve_backend(
            data,
            organelle="mitochondrion",
            context=context,
            released_backend_ids=_RELEASED_WITHOUT_PMAT,
        )
    assert raised.value.code == "assembly.routing_context_mismatch"


def test_routing_context_and_rule_models_are_closed_and_frozen() -> None:
    context = _context(total_bases=10, genome_size_bp=4)
    with pytest.raises(ValidationError):
        context.total_bases = 99  # type: ignore[misc]

    payload = context.model_dump(mode="python", round_trip=True)
    payload["unexpected"] = True
    with pytest.raises(ValidationError):
        RoutingContext.model_validate(payload)

    rule = AutoRouteRule(
        rule_id="auto.mito_hifi_lte_3x_pmat",
        backend_id="pmat",
        organelle="mitochondrion",
        profiles=(AssemblyProfile.PACBIO_HIFI,),
        taxon_group="plant",
        coverage_predicate="lte_3x",
    )
    with pytest.raises(ValidationError):
        rule.backend_id = "other"  # type: ignore[misc]
    rule_payload = rule.model_dump(mode="python", round_trip=True)
    rule_payload["unexpected"] = True
    with pytest.raises(ValidationError):
        AutoRouteRule.model_validate(rule_payload)


def test_auto_rules_are_declared_in_the_approved_priority_order() -> None:
    rule_ids = [rule.rule_id for rule in routing_module.AUTO_RULES]
    assert rule_ids == [
        "auto.mito_hifi_lte_3x_pmat",
        "auto.mito_hifi_gt_3x_oatk",
        "auto.hifi_oatk",
        "auto.raw_clr_pmat",
        "auto.raw_ont_pmat",
        "auto.illumina_pe_getorganelle",
        "auto.illumina_se_getorganelle",
    ]


def test_exact_3x_boundary_routes_to_pmat(tmp_path: Path) -> None:
    data = read_reads(
        long_libraries=(
            LongReadLibrary(
                technology="pacbio_hifi",
                quality_state="ccs",
                reads=_fastq(tmp_path / "hifi.fastq"),
            ),
        )
    )
    context = _context(total_bases=3_000, genome_size_bp=1_000)
    route = resolve_backend(
        data,
        organelle="mitochondrion",
        context=context,
        released_backend_ids=_RELEASED_WITH_PMAT,
    )
    assert route.rule_id == "auto.mito_hifi_lte_3x_pmat"
    assert route.selected_backend == "pmat"


def test_one_base_over_the_3x_boundary_routes_to_oatk(tmp_path: Path) -> None:
    data = read_reads(
        long_libraries=(
            LongReadLibrary(
                technology="pacbio_hifi",
                quality_state="ccs",
                reads=_fastq(tmp_path / "hifi.fastq"),
            ),
        )
    )
    context = _context(total_bases=3_001, genome_size_bp=1_000)
    route = resolve_backend(
        data,
        organelle="mitochondrion",
        context=context,
        released_backend_ids=_RELEASED_WITH_PMAT,
    )
    assert route.rule_id == "auto.mito_hifi_gt_3x_oatk"
    assert route.selected_backend == "oatk"


def test_missing_genome_size_raises_needs_input_before_any_side_effect(
    tmp_path: Path,
) -> None:
    data = read_reads(
        long_libraries=(
            LongReadLibrary(
                technology="pacbio_hifi",
                quality_state="ccs",
                reads=_fastq(tmp_path / "hifi.fastq"),
            ),
        )
    )
    context = _context(total_bases=3_000, genome_size_bp=None)
    with pytest.raises(OrganelleNeedsInput) as raised:
        resolve_backend(
            data,
            organelle="mitochondrion",
            context=context,
            released_backend_ids=_RELEASED_WITH_PMAT,
        )
    request = raised.value.needs_input
    assert request.field == "auxiliary.genome_size"
    assert request.choices == (
        "use_default_oatk",
        "provide_genome_size",
        "lookup_by_species",
    )
    assert request.request_id.startswith("sha256:")


def test_without_pmat_released_hifi_uses_oatk_and_does_not_ask_for_input(
    tmp_path: Path,
) -> None:
    data = read_reads(
        long_libraries=(
            LongReadLibrary(
                technology="pacbio_hifi",
                quality_state="ccs",
                reads=_fastq(tmp_path / "hifi.fastq"),
            ),
        )
    )
    context = _context(total_bases=3_000, genome_size_bp=None)
    route = resolve_backend(
        data,
        organelle="mitochondrion",
        context=context,
        released_backend_ids=_RELEASED_WITHOUT_PMAT,
    )
    assert route.rule_id == "auto.hifi_oatk"
    assert route.selected_backend == "oatk"


def test_non_plant_hifi_falls_through_to_auto_hifi_oatk(tmp_path: Path) -> None:
    data = read_reads(
        long_libraries=(
            LongReadLibrary(
                technology="pacbio_hifi",
                quality_state="ccs",
                reads=_fastq(tmp_path / "hifi.fastq"),
            ),
        )
    )
    context = _context(total_bases=3_000, genome_size_bp=1_000, taxon_group="animal")
    route = resolve_backend(
        data,
        organelle="mitochondrion",
        context=context,
        released_backend_ids=_RELEASED_WITH_PMAT,
    )
    assert route.rule_id == "auto.hifi_oatk"
    assert route.selected_backend == "oatk"


def test_plastid_hifi_falls_through_to_auto_hifi_oatk(tmp_path: Path) -> None:
    data = read_reads(
        long_libraries=(
            LongReadLibrary(
                technology="pacbio_hifi",
                quality_state="ccs",
                reads=_fastq(tmp_path / "hifi.fastq"),
            ),
        )
    )
    context = _context(total_bases=3_000, genome_size_bp=1_000)
    route = resolve_backend(
        data,
        organelle="plastid",
        context=context,
        released_backend_ids=_RELEASED_WITH_PMAT,
    )
    assert route.rule_id == "auto.hifi_oatk"
    assert route.selected_backend == "oatk"


def test_every_automatic_result_selects_only_a_released_backend(
    tmp_path: Path,
) -> None:
    # Illumina PE has an automatic rule -> getorganelle, but getorganelle is not
    # in the injected release set, so no released automatic rule matches.
    data = _illumina(tmp_path)
    context = RoutingContext(
        profile=AssemblyProfile.ILLUMINA_PE,
        taxon_group="plant",
    )
    with pytest.raises(OrganelleInputError) as raised:
        resolve_backend(
            data,
            organelle="plastid",
            context=context,
            released_backend_ids=_RELEASED_WITH_PMAT,
        )
    assert raised.value.code == "assembly.unsupported_data_profile"


def test_explicit_method_overrides_the_coverage_boundary(tmp_path: Path) -> None:
    data = read_reads(
        long_libraries=(
            LongReadLibrary(
                technology="pacbio_hifi",
                quality_state="ccs",
                reads=_fastq(tmp_path / "hifi.fastq"),
            ),
        )
    )
    context = _context(total_bases=3_000, genome_size_bp=1_000)
    pmat_route = resolve_backend(
        data,
        organelle="mitochondrion",
        method="pmat",
        context=context,
        released_backend_ids=_RELEASED_WITH_PMAT,
    )
    assert pmat_route.rule_id == "explicit.backend"
    assert pmat_route.selected_backend == "pmat"

    oatk_route = resolve_backend(
        data,
        organelle="mitochondrion",
        method="oatk",
        context=context,
        released_backend_ids=_RELEASED_WITH_PMAT,
    )
    assert oatk_route.rule_id == "explicit.backend"
    assert oatk_route.selected_backend == "oatk"


def test_explicit_unreleased_backend_is_rejected(tmp_path: Path) -> None:
    data = _long(tmp_path, "pacbio_hifi", "ccs")
    with pytest.raises(OrganelleInputError) as raised:
        resolve_backend(
            data,
            organelle="mitochondrion",
            method="getorganelle",
            released_backend_ids=_RELEASED_WITH_PMAT,
        )
    assert raised.value.code == "assembly.unknown_backend"


def test_routing_does_not_probe_software_or_environments_under_release_filtering(
    tmp_path: Path,
) -> None:
    data = read_reads(
        long_libraries=(
            LongReadLibrary(
                technology="pacbio_hifi",
                quality_state="ccs",
                reads=_fastq(tmp_path / "hifi.fastq"),
            ),
        )
    )
    context = _context(total_bases=3_000, genome_size_bp=1_000)
    resolve_backend(
        data,
        organelle="mitochondrion",
        context=context,
        released_backend_ids=_RELEASED_WITH_PMAT,
    )
    source = inspect.getsource(routing_module)
    assert "shutil.which" not in source
    assert "import subprocess" not in source
    assert "assembly.install" not in source


def test_prepare_routing_context_carries_taxon_group_and_evidence(tmp_path: Path) -> None:
    request = _hifi_request(tmp_path, genome_size=1_000)
    context = prepare_routing_context(request)
    assert context.profile == AssemblyProfile.PACBIO_HIFI
    assert context.taxon_group == "plant"
    assert context.genome_size_bp == 1_000
    assert context.genome_size_evidence is not None
    assert context.genome_size_evidence.source == "user"
    assert context.threshold_multiplier == 3
    assert len(context.read_statistics) == 1
    assert context.total_bases is not None and context.total_bases >= 0


def test_prepare_routing_context_default_request_requires_genome_size(
    tmp_path: Path,
) -> None:
    request = _hifi_request(tmp_path, genome_size=None)
    with pytest.raises(OrganelleNeedsInput) as raised:
        prepare_routing_context(request)
    assert raised.value.needs_input.choices == (
        "use_default_oatk",
        "provide_genome_size",
        "lookup_by_species",
    )


def test_prepare_routing_context_sums_total_bases_across_multiple_libraries(
    tmp_path: Path,
) -> None:
    one = tmp_path / "one.fastq"
    two = tmp_path / "two.fastq"
    one.write_text("@a\nACGTAC\n+\nIIIIII\n", encoding="utf-8")
    two.write_text("@b\nACGT\n+\nIIII\n@c\nACGT\n+\nIIII\n", encoding="utf-8")
    data = read_reads(
        long_libraries=(
            LongReadLibrary(technology="pacbio_hifi", quality_state="ccs", reads=one),
            LongReadLibrary(technology="pacbio_hifi", quality_state="ccs", reads=two),
        ),
    )
    request = AssemblyRequest(data=data, organelle="mitochondrion")
    context = prepare_routing_context(
        request,
        released_backend_ids=_RELEASED_WITHOUT_PMAT,
    )
    assert len(context.read_statistics) == 2
    assert context.total_bases == 6 + 4 + 4
    # Each statistic carries the full artifact SHA256 of its source.
    for stats, role in zip(
        context.read_statistics,
        ("long_0_reads", "long_1_reads"),
        strict=True,
    ):
        assert stats.artifact_sha256 == data.artifacts[role].sha256
