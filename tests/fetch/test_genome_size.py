"""``fetch.genome_size_candidates`` — exact-species NCBI genome-size lookup.

These tests pin the contract from the approved design
(``docs/superpowers/specs/2026-07-19-pmat2-low-coverage-routing-design.md`` §
Genome-size candidate operation): exact TaxID equality, deterministic ranking,
anomalous exclusion, scaffold/contig and phased/diploid non-selection, missing
optional statistics, canonical report bytes with a real SHA256, and a fail-closed
exact-name miss. No test performs a network request — ``FakeTransport`` serves the
immutable snapshot in ``fixtures/ncbi_genome_size_reports.json``.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from urllib.parse import parse_qs, quote, urlsplit

import pytest
from pydantic import ValidationError

from organelleverse.core.data import OrganelleData
from organelleverse.core.errors import (
    OrganelleExecutionError,
    OrganelleInputError,
    OrganelleParameterError,
)
from organelleverse.fetch import _agent_ops
from organelleverse.fetch import genome_size as genome_size_module
from organelleverse.fetch._http import HttpResponse
from organelleverse.fetch.genome_size import GenomeSizeCandidate, genome_size_candidates
from organelleverse.fetch.operations import FETCH_SPECS, GENOME_SIZE_CANDIDATES_SPEC
from organelleverse.operations import SideEffect
from organelleverse.operations.adapters import invoke_json

FIXTURE = Path(__file__).parent / "fixtures" / "ncbi_genome_size_reports.json"

# Expected deterministic ranking of the returned records. R9 (Complete Genome,
# reference, but ploidy unconfirmed) and R8 (diploid) are displayed ahead of
# lesser assembly levels yet remain non-selectable: ranking and auto-selection
# are separate decisions.
EXPECTED_ORDER = [
    "GCF_000001735.4",
    "GCF_999888.8",
    "GCF_999111.1",
    "GCA_999222.2",
    "GCF_999333.3",
    "GCF_999444.4",
    "GCF_999777.7",
    "GCA_999555.5",
    "GCA_999666.6",
]
EXPECTED_SELECTABLE = [
    "GCF_000001735.4",
    "GCF_999111.1",
    "GCA_999222.2",
    "GCF_999333.3",
    "GCF_999444.4",
]
EXCLUDED = ["GCF_999999.9", "GCF_888000.0"]


def _load_fixture() -> dict[str, object]:
    return json.loads(FIXTURE.read_text())


def _canonical_report_bytes(record: object) -> bytes:
    """Independent canonicalization: sorted keys, compact separators, one newline."""
    return (
        json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
        + b"\n"
    )


class FakeTransport:
    """Serves fixture bodies by URL path. Records every call for exclusion checks."""

    def __init__(
        self,
        fixture: dict[str, object],
        *,
        taxonomy: dict[str, object] | None = None,
        taxon_reports: dict[str, object] | None = None,
        taxonomy_status: int = 200,
        taxonomy_body: bytes | None = None,
        accession_reports: dict[str, dict[str, object]] | None = None,
        taxon_pages: tuple[dict[str, object], ...] | None = None,
    ) -> None:
        self.base_url = str(fixture["base_url"])
        self.species_name = str(fixture["species_name"])
        self.taxid = int(fixture["taxid"])
        self._taxonomy = taxonomy if taxonomy is not None else fixture["taxonomy_response"]  # type: ignore[assignment]
        self._taxon_reports = (
            taxon_reports if taxon_reports is not None else {"reports": fixture["reports"]}  # type: ignore[assignment]
        )
        self._taxonomy_status = taxonomy_status
        self._taxonomy_body = taxonomy_body
        self._taxon_pages = taxon_pages
        reports = fixture["reports"]
        self._by_accession = {str(r["accession"]): r for r in reports}  # type: ignore[union-attr]
        if accession_reports is not None:
            self._by_accession.update(accession_reports)
        self.calls: list[str] = []

    def get_response(
        self,
        url: str,
        *,
        method: str = "GET",
        data: bytes | None = None,
        timeout: float = 120.0,
        headers: object = None,
    ) -> HttpResponse:
        self.calls.append(url)
        path = url.split("?", 1)[0]
        if self._taxonomy_body is not None and path.endswith(
            f"/taxonomy/taxon/{quote(self.species_name, safe='')}"
        ):
            return HttpResponse(status=self._taxonomy_status, body=self._taxonomy_body)
        if path.endswith(f"/taxonomy/taxon/{quote(self.species_name, safe='')}"):
            body = json.dumps(self._taxonomy).encode("utf-8")
            return HttpResponse(status=self._taxonomy_status, body=body)
        if f"/genome/taxon/{self.taxid}/dataset_report" in path:
            report_page = self._taxon_reports
            if self._taxon_pages is not None:
                page_token = parse_qs(urlsplit(url).query).get("page_token", [None])[0]
                page_index = 0 if page_token is None else int(str(page_token).removeprefix("page-"))
                report_page = self._taxon_pages[page_index]
            body = json.dumps(report_page).encode("utf-8")
            return HttpResponse(status=200, body=body)
        if "/genome/accession/" in path and path.endswith("/dataset_report"):
            accession = path.split("/genome/accession/", 1)[1].split("/", 1)[0]
            record = self._by_accession.get(accession)
            body = json.dumps({"reports": [record] if record is not None else []}).encode("utf-8")
            return HttpResponse(status=200, body=body)
        return HttpResponse(status=404, body=b"")


@pytest.fixture
def cache_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "genome_size_cache"
    monkeypatch.setenv("ORGANELLEVERSE_GENOME_SIZE_CACHE", str(root))
    return root


@pytest.fixture
def fixture() -> dict[str, object]:
    return _load_fixture()


@pytest.fixture
def transport(fixture: dict[str, object]) -> FakeTransport:
    return FakeTransport(fixture)


def _candidates(data: OrganelleData) -> list[dict[str, object]]:
    return list(data.payload.get("candidates", []))  # type: ignore[union-attr]


def _accessions(data: OrganelleData) -> list[str]:
    return [str(c["assembly_accession"]) for c in _candidates(data)]


# ──────────────────────────────────────────────────────────────────────────
# Exact-species filtering and deterministic ranking
# ──────────────────────────────────────────────────────────────────────────


def test_returns_only_exact_taxid_records_in_deterministic_order(
    fixture: dict[str, object], transport: FakeTransport, cache_root: Path
) -> None:
    data = genome_size_candidates(str(fixture["species_name"]), transport=transport)
    assert isinstance(data, OrganelleData)
    assert data.modality == "genome_size_candidates"
    assert _accessions(data) == EXPECTED_ORDER
    # The near relative (different TaxID) is never substituted.
    assert "GCF_888000.0" not in _accessions(data)


def test_selectable_flag_marks_only_auto_selectable_records(
    fixture: dict[str, object], transport: FakeTransport, cache_root: Path
) -> None:
    data = genome_size_candidates(str(fixture["species_name"]), transport=transport)
    selectable = [str(c["assembly_accession"]) for c in _candidates(data) if c["selectable"]]
    assert selectable == EXPECTED_SELECTABLE


def test_complete_genome_ranks_above_chromosome(
    fixture: dict[str, object], transport: FakeTransport, cache_root: Path
) -> None:
    data = genome_size_candidates(str(fixture["species_name"]), transport=transport)
    levels = {str(c["assembly_accession"]): str(c["assembly_level"]) for c in _candidates(data)}
    complete_positions = [
        i for i, acc in enumerate(_accessions(data)) if levels[acc] == "Complete Genome"
    ]
    chromosome_positions = [
        i for i, acc in enumerate(_accessions(data)) if levels[acc] == "Chromosome"
    ]
    assert max(complete_positions) < min(chromosome_positions)


def test_reference_genome_ranks_above_representative_at_same_level(
    fixture: dict[str, object], transport: FakeTransport, cache_root: Path
) -> None:
    data = genome_size_candidates(str(fixture["species_name"]), transport=transport)
    by_accession = {str(c["assembly_accession"]): c for c in _candidates(data)}
    # GCF_000001735.4 (reference) is selectable #1; GCF_999111.1 (representative) is #2.
    assert by_accession["GCF_000001735.4"]["refseq_category"] == "reference genome"
    assert by_accession["GCF_999111.1"]["refseq_category"] == "representative genome"
    assert _accessions(data).index("GCF_000001735.4") < _accessions(data).index("GCF_999111.1")


def test_accession_is_the_final_ascending_tie_break(
    fixture: dict[str, object], transport: FakeTransport, cache_root: Path
) -> None:
    """R3 and R4 tie on level/category/N50/unplaced/date; accession breaks it."""
    data = genome_size_candidates(str(fixture["species_name"]), transport=transport)
    assert _accessions(data).index("GCA_999222.2") < _accessions(data).index("GCF_999333.3")


def test_release_date_tie_break_favors_newer_within_category(
    fixture: dict[str, object], cache_root: Path
) -> None:
    base = deepcopy(fixture["reports"][2])  # type: ignore[index]
    newer = deepcopy(base)
    older = deepcopy(base)
    newer["accession"] = "GCA_000010.1"
    older["accession"] = "GCA_000009.1"
    newer["assembly_info"]["release_date"] = "2024-02-29"  # type: ignore[index]
    older["assembly_info"]["release_date"] = "2020-01-01"  # type: ignore[index]
    for report in (newer, older):
        report["assembly_info"].pop("released_date", None)  # type: ignore[union-attr]
    transport = FakeTransport(
        fixture,
        taxon_reports={"reports": [older, newer]},
        accession_reports={str(item["accession"]): item for item in (older, newer)},
    )
    data = genome_size_candidates(str(fixture["species_name"]), transport=transport)
    assert _accessions(data) == ["GCA_000010.1", "GCA_000009.1"]


def test_anomalous_record_is_excluded_not_displayed(
    fixture: dict[str, object], transport: FakeTransport, cache_root: Path
) -> None:
    data = genome_size_candidates(str(fixture["species_name"]), transport=transport)
    assert "GCF_999999.9" not in _accessions(data)
    # The provider must not even fetch the report for an excluded record.
    assert not any("GCF_999999.9" in url for url in transport.calls)


def test_suppressed_partial_record_is_excluded_by_official_fields(
    fixture: dict[str, object], cache_root: Path
) -> None:
    partial = deepcopy(fixture["reports"][0])  # type: ignore[index]
    partial["assembly_info"]["assembly_status"] = "suppressed"  # type: ignore[index]
    partial["assembly_info"]["suppression_reason"] = "partial/incomplete assembly"  # type: ignore[index]
    valid = deepcopy(fixture["reports"][1])  # type: ignore[index]
    transport = FakeTransport(fixture, taxon_reports={"reports": [partial, valid]})
    data = genome_size_candidates(str(fixture["species_name"]), transport=transport)
    assert _accessions(data) == [str(valid["accession"])]
    assert not any(str(partial["accession"]) in url for url in transport.calls)


def test_all_taxon_report_pages_participate_in_ranking(
    fixture: dict[str, object], cache_root: Path
) -> None:
    first = deepcopy(fixture["reports"][4])  # type: ignore[index]
    best = deepcopy(fixture["reports"][0])  # type: ignore[index]
    pages = (
        {"reports": [first], "next_page_token": "page-1"},
        {"reports": [best]},
    )
    transport = FakeTransport(fixture, taxon_pages=pages)
    data = genome_size_candidates(str(fixture["species_name"]), transport=transport)
    assert _accessions(data)[0] == str(best["accession"])
    assert any("page_token=page-1" in url for url in transport.calls)


def test_phased_diploid_record_is_displayed_but_not_selectable(
    fixture: dict[str, object], transport: FakeTransport, cache_root: Path
) -> None:
    data = genome_size_candidates(str(fixture["species_name"]), transport=transport)
    by_accession = {str(c["assembly_accession"]): c for c in _candidates(data)}
    diploid = by_accession["GCF_999777.7"]
    assert diploid["selectable"] is False
    assert diploid["ambiguity_reason"] is not None
    assert (
        "diploid" in str(diploid["ambiguity_reason"]).lower()
        or "phas" in str(diploid["ambiguity_reason"]).lower()
    )


def test_ploidy_unconfirmed_record_is_not_auto_selected(
    fixture: dict[str, object], transport: FakeTransport, cache_root: Path
) -> None:
    """Fail-closed: a record with no ploidy evidence is shown but not selectable."""
    data = genome_size_candidates(str(fixture["species_name"]), transport=transport)
    by_accession = {str(c["assembly_accession"]): c for c in _candidates(data)}
    unconfirmed = by_accession["GCF_999888.8"]
    assert unconfirmed["selectable"] is False
    assert unconfirmed["ambiguity_reason"] is not None


def test_scaffold_and_contig_records_are_displayed_but_not_selectable(
    fixture: dict[str, object], transport: FakeTransport, cache_root: Path
) -> None:
    data = genome_size_candidates(str(fixture["species_name"]), transport=transport)
    by_accession = {str(c["assembly_accession"]): c for c in _candidates(data)}
    assert by_accession["GCA_999555.5"]["selectable"] is False
    assert by_accession["GCA_999666.6"]["selectable"] is False


def test_missing_optional_statistics_lower_rank_without_inventing_zero(
    fixture: dict[str, object], transport: FakeTransport, cache_root: Path
) -> None:
    data = genome_size_candidates(str(fixture["species_name"]), transport=transport)
    by_accession = {str(c["assembly_accession"]): c for c in _candidates(data)}
    missing_stats = by_accession["GCF_999888.8"]
    assert missing_stats["contig_n50"] is None
    assert missing_stats["scaffold_n50"] is None
    assert missing_stats["unplaced_proportion"] is None
    # Despite being a newer Complete Genome reference, missing stats drop it below
    # the fully-specified reference R1 (which has every statistic).
    assert _accessions(data).index("GCF_999888.8") > _accessions(data).index("GCF_000001735.4")


def test_fractional_required_genome_size_is_rejected_not_truncated(
    fixture: dict[str, object], cache_root: Path
) -> None:
    invalid = deepcopy(fixture["reports"][0])  # type: ignore[index]
    invalid["assembly_stats"]["total_sequence_length"] = 119668435.5  # type: ignore[index]
    transport = FakeTransport(fixture, taxon_reports={"reports": [invalid]})
    with pytest.raises(OrganelleInputError) as raised:
        genome_size_candidates(str(fixture["species_name"]), transport=transport)
    assert raised.value.code == "fetch.genome_size_taxon_miss"
    assert not any("/genome/accession/" in url for url in transport.calls)


def test_candidate_carries_resolved_identity_and_size(
    fixture: dict[str, object], transport: FakeTransport, cache_root: Path
) -> None:
    data = genome_size_candidates(str(fixture["species_name"]), transport=transport)
    top = _candidates(data)[0]
    assert top["assembly_accession"] == "GCF_000001735.4"
    assert top["taxon_id"] == int(fixture["taxid"])
    assert top["scientific_name"] == "Arabidopsis thaliana"
    assert top["assembly_name"] == "TAIR10.1"
    assert top["assembly_level"] == "Complete Genome"
    assert top["refseq_category"] == "reference genome"
    assert top["genome_size_bp"] == 119668435
    assert top["contig_n50"] == 10740370
    assert top["scaffold_n50"] == 23138553
    assert top["unplaced_proportion"] == pytest.approx(0.0001)
    assert top["release_date"] == "2022-09-20"


def test_max_candidates_caps_returned_records(
    fixture: dict[str, object], transport: FakeTransport, cache_root: Path
) -> None:
    data = genome_size_candidates(
        str(fixture["species_name"]), max_candidates=3, transport=transport
    )
    assert _accessions(data) == EXPECTED_ORDER[:3]


@pytest.mark.parametrize("invalid", [True, 0, 51, 1.5, "3"])
def test_direct_api_rejects_invalid_max_candidates_without_clamping(
    invalid: object,
    fixture: dict[str, object],
    transport: FakeTransport,
    cache_root: Path,
) -> None:
    with pytest.raises(OrganelleParameterError) as raised:
        genome_size_candidates(
            str(fixture["species_name"]),
            max_candidates=invalid,  # type: ignore[arg-type]
            transport=transport,
        )
    assert raised.value.code == "parameter.invalid_max_candidates"
    assert transport.calls == []


# ──────────────────────────────────────────────────────────────────────────
# Canonical report artifact + real SHA256
# ──────────────────────────────────────────────────────────────────────────


def test_report_artifact_is_canonical_sorted_compact_json_with_newline_and_hash(
    fixture: dict[str, object], transport: FakeTransport, cache_root: Path
) -> None:
    data = genome_size_candidates(str(fixture["species_name"]), transport=transport)
    reports_by_accession = {str(r["accession"]): r for r in fixture["reports"]}  # type: ignore[union-attr]
    for candidate in _candidates(data):
        accession = str(candidate["assembly_accession"])
        record = reports_by_accession[accession]
        expected_bytes = _canonical_report_bytes(record)
        expected_sha = hashlib.sha256(expected_bytes).hexdigest()

        assert candidate["report_sha256"] == expected_sha
        role = str(candidate["report_artifact_role"])
        assert candidate["report_artifact_role"] == "genome_size_report:" + accession
        artifact = data.artifacts[role]
        assert artifact.sha256 == expected_sha
        assert artifact.kind == "genome_size_report"
        assert artifact.format == "json"
        # report_uri is the exact official Datasets accession request URI.
        assert artifact.uri != candidate["report_uri"]
        assert str(candidate["report_uri"]).endswith(
            f"/genome/accession/{accession}/dataset_report"
        )
        # The canonical bytes are materialized at the artifact path.
        assert Path(artifact.uri).read_bytes() == expected_bytes


def test_report_artifacts_are_distinct_per_candidate(
    fixture: dict[str, object], transport: FakeTransport, cache_root: Path
) -> None:
    data = genome_size_candidates(str(fixture["species_name"]), transport=transport)
    roles = [str(c["report_artifact_role"]) for c in _candidates(data)]
    assert len(roles) == len(set(roles))
    assert set(roles) <= set(data.artifacts)


def test_existing_content_addressed_report_is_revalidated_not_overwritten(
    fixture: dict[str, object], transport: FakeTransport, cache_root: Path
) -> None:
    data = genome_size_candidates(
        str(fixture["species_name"]), max_candidates=1, transport=transport
    )
    artifact = next(iter(data.artifacts.values()))
    Path(artifact.uri).write_bytes(b"tampered\n")
    with pytest.raises(OrganelleExecutionError) as raised:
        genome_size_candidates(
            str(fixture["species_name"]),
            max_candidates=1,
            transport=FakeTransport(fixture),
        )
    assert raised.value.code == "fetch.genome_size_cache_corrupt"


# ──────────────────────────────────────────────────────────────────────────
# Provider/schema metadata stays out of candidate scientific identity
# ──────────────────────────────────────────────────────────────────────────


def test_retrieval_metadata_is_outside_candidate_identity(
    fixture: dict[str, object], transport: FakeTransport, cache_root: Path
) -> None:
    data = genome_size_candidates(str(fixture["species_name"]), transport=transport)
    assert data.metadata["provider"] == "ncbi_datasets"
    assert data.metadata["schema"] == "datasets/v2"
    assert isinstance(data.metadata["retrieved_at"], str)
    candidate_fields = set(GenomeSizeCandidate.model_fields)
    for candidate in _candidates(data):
        # No retrieval/provider/schema metadata leaks into a candidate record.
        assert set(candidate).issubset(candidate_fields)
        assert "retrieved_at" not in candidate
        assert "provider" not in candidate


# ──────────────────────────────────────────────────────────────────────────
# Fail-closed: exact-name miss and provider failures are typed errors
# ──────────────────────────────────────────────────────────────────────────


def test_exact_name_miss_raises_typed_error_without_substituting_a_relative(
    fixture: dict[str, object], cache_root: Path
) -> None:
    transport = FakeTransport(fixture, taxonomy=fixture["taxonomy_response_miss"])  # type: ignore[arg-type]
    with pytest.raises(OrganelleInputError) as raised:
        genome_size_candidates(str(fixture["species_name"]), transport=transport)
    assert raised.value.code == "fetch.genome_size_taxon_miss"
    assert raised.value.retryable is False


def test_real_v2_nested_taxonomy_shape_is_accepted(
    fixture: dict[str, object], cache_root: Path
) -> None:
    nested = {
        "taxonomy_nodes": [
            {
                "query": [fixture["species_name"]],
                "taxonomy": {
                    "tax_id": fixture["taxid"],
                    "organism_name": fixture["species_name"],
                    "rank": "SPECIES",
                },
            }
        ]
    }
    data = genome_size_candidates(
        str(fixture["species_name"]), transport=FakeTransport(fixture, taxonomy=nested)
    )
    assert data.payload["taxon_id"] == fixture["taxid"]


def test_taxonomy_near_name_is_rejected_even_when_provider_returns_a_node(
    fixture: dict[str, object], cache_root: Path
) -> None:
    wrong_name = {
        "taxonomy_nodes": [
            {
                "taxonomy": {
                    "tax_id": fixture["near_relative_taxid"],
                    "organism_name": "Arabidopsis lyrata",
                }
            }
        ]
    }
    with pytest.raises(OrganelleInputError) as raised:
        genome_size_candidates(
            str(fixture["species_name"]),
            transport=FakeTransport(fixture, taxonomy=wrong_name),
        )
    assert raised.value.code == "fetch.genome_size_taxon_miss"
    assert raised.value.retryable is False


def test_taxonomy_path_quotes_slashes_as_path_data() -> None:
    scientific_name = "Species / exact"

    class TaxonomyTransport:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def get_response(self, url: str, **_: object) -> HttpResponse:
            self.calls.append(url)
            return HttpResponse(
                status=200,
                body=json.dumps(
                    {
                        "taxonomy_nodes": [
                            {
                                "taxonomy": {
                                    "tax_id": 42,
                                    "organism_name": scientific_name,
                                }
                            }
                        ]
                    }
                ).encode(),
            )

    transport = TaxonomyTransport()
    assert genome_size_module._resolve_taxonomy(transport, scientific_name) == (  # type: ignore[arg-type]
        scientific_name,
        42,
    )
    assert transport.calls[0].endswith("/taxonomy/taxon/Species%20%2F%20exact")


def test_no_exact_taxid_candidate_raises_typed_error(
    fixture: dict[str, object], cache_root: Path
) -> None:
    near_relative = next(
        r
        for r in fixture["reports"]
        if int(r["organism"]["tax_id"]) == 4565  # type: ignore[union-attr]
    )
    transport = FakeTransport(fixture, taxon_reports={"reports": [near_relative]})
    with pytest.raises(OrganelleInputError) as raised:
        genome_size_candidates(str(fixture["species_name"]), transport=transport)
    assert raised.value.code == "fetch.genome_size_taxon_miss"


def test_inconsistent_taxonomy_response_raises_typed_error(
    fixture: dict[str, object], cache_root: Path
) -> None:
    transport = FakeTransport(
        fixture,
        taxonomy=fixture["taxonomy_response_inconsistent"],  # type: ignore[arg-type]
    )
    with pytest.raises(OrganelleExecutionError) as raised:
        genome_size_candidates(str(fixture["species_name"]), transport=transport)
    assert raised.value.code == "fetch.genome_size_taxon_inconsistent"
    assert raised.value.retryable is True


def test_accession_report_must_match_ranked_candidate_identity(
    fixture: dict[str, object], cache_root: Path
) -> None:
    top = deepcopy(fixture["reports"][0])  # type: ignore[index]
    mismatched = deepcopy(top)
    mismatched["assembly_stats"]["total_sequence_length"] = 1  # type: ignore[index]
    transport = FakeTransport(
        fixture,
        accession_reports={str(top["accession"]): mismatched},
    )
    with pytest.raises(OrganelleExecutionError) as raised:
        genome_size_candidates(str(fixture["species_name"]), max_candidates=1, transport=transport)
    assert raised.value.code == "fetch.genome_size_report_inconsistent"


def test_malformed_taxonomy_json_raises_retryable_network_error(
    fixture: dict[str, object], cache_root: Path
) -> None:
    transport = FakeTransport(fixture, taxonomy_body=b"<<<not json>>>")
    with pytest.raises(OrganelleExecutionError) as raised:
        genome_size_candidates(str(fixture["species_name"]), transport=transport)
    assert raised.value.code == "network.malformed_response"
    assert raised.value.retryable is True


def test_taxonomy_http_404_is_an_exact_name_miss(
    fixture: dict[str, object], cache_root: Path
) -> None:
    transport = FakeTransport(fixture, taxonomy_status=404)
    with pytest.raises(OrganelleInputError) as raised:
        genome_size_candidates(str(fixture["species_name"]), transport=transport)
    assert raised.value.code == "fetch.genome_size_taxon_miss"


def test_transient_status_exhausts_retries_then_raises_network_transient(
    fixture: dict[str, object], cache_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("organelleverse.fetch._http.time.sleep", lambda *args: None)
    transport = FakeTransport(fixture, taxonomy_status=503)
    with pytest.raises(OrganelleExecutionError) as raised:
        genome_size_candidates(str(fixture["species_name"]), transport=transport)
    assert raised.value.code == "network.transient"
    assert raised.value.retryable is True


# ──────────────────────────────────────────────────────────────────────────
# GenomeSizeCandidate is a strict, frozen, identity-only record
# ──────────────────────────────────────────────────────────────────────────


def test_genome_size_candidate_is_frozen_and_rejects_unknown_fields() -> None:
    candidate = GenomeSizeCandidate(
        taxon_id=3702,
        scientific_name="Arabidopsis thaliana",
        assembly_accession="GCF_000001735.4",
        assembly_name="TAIR10.1",
        assembly_level="Complete Genome",
        refseq_category="reference genome",
        genome_size_bp=119668435,
        contig_n50=10740370,
        scaffold_n50=23138553,
        unplaced_proportion=0.0001,
        release_date="2022-09-20",
        selectable=True,
        ambiguity_reason=None,
        report_uri="https://example.invalid/genome/accession/GCF_000001735.4/dataset_report",
        report_artifact_role="genome_size_report:GCF_000001735.4",
        report_sha256="a" * 64,
    )
    with pytest.raises(ValidationError):
        candidate.taxon_id = 1  # type: ignore[misc]
    with pytest.raises(ValidationError):
        GenomeSizeCandidate(
            taxon_id=3702,
            scientific_name="x",
            assembly_accession="GCF_x",
            assembly_name="n",
            assembly_level="Complete Genome",
            refseq_category=None,
            genome_size_bp=1,
            contig_n50=None,
            scaffold_n50=None,
            unplaced_proportion=None,
            release_date="2022-09-20",
            selectable=True,
            ambiguity_reason=None,
            report_uri="u",
            report_artifact_role="r",
            report_sha256="a" * 64,
            unexpected_field=True,  # type: ignore[call-arg]
        )


def test_genome_size_candidate_rejects_coercion_and_invalid_proportion() -> None:
    common = {
        "scientific_name": "Arabidopsis thaliana",
        "assembly_accession": "GCF_000001735.4",
        "assembly_name": "TAIR10.1",
        "assembly_level": "Chromosome",
        "refseq_category": "reference genome",
        "genome_size_bp": 119146348,
        "contig_n50": 11194537,
        "scaffold_n50": 23459830,
        "release_date": "2018-03-15",
        "selectable": True,
        "ambiguity_reason": None,
        "report_uri": "https://example.invalid/report",
        "report_artifact_role": "genome_size_report:GCF_000001735.4",
        "report_sha256": "a" * 64,
    }
    with pytest.raises(ValidationError):
        GenomeSizeCandidate(taxon_id="3702", unplaced_proportion=0.1, **common)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        GenomeSizeCandidate(taxon_id=3702, unplaced_proportion=1.01, **common)


# ──────────────────────────────────────────────────────────────────────────
# Operation / facade / catalog / Agent JSON all reach the same callable
# ──────────────────────────────────────────────────────────────────────────


def test_genome_size_spec_is_declared_and_registered() -> None:
    assert GENOME_SIZE_CANDIDATES_SPEC in FETCH_SPECS
    assert GENOME_SIZE_CANDIDATES_SPEC.operation_id == "fetch.genome_size_candidates"
    assert GENOME_SIZE_CANDIDATES_SPEC.callable_locator == (
        "organelleverse.fetch._agent_ops:op_genome_size_candidates"
    )


def test_facade_alias_is_the_public_python_callable() -> None:
    import organelleverse.fetch as fetch_pkg

    assert fetch_pkg.genome_size_candidates is genome_size_candidates
    assert "genome_size_candidates" in fetch_pkg.__all__


def test_catalog_registers_the_same_callable_as_the_locator(
    fixture: dict[str, object],
    cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from organelleverse.operations import registry

    binding = registry.require("fetch.genome_size_candidates")
    assert binding.implementation is _agent_ops.op_genome_size_candidates
    assert binding.spec == GENOME_SIZE_CANDIDATES_SPEC


def test_parameter_schema_constrains_max_candidates_and_requires_name() -> None:
    from organelleverse.operations import registry

    schema = registry.parameter_schema("fetch.genome_size_candidates")
    properties = schema["properties"]
    assert "scientific_name" in properties
    assert properties["max_candidates"]["minimum"] == 1
    assert properties["max_candidates"]["maximum"] == 50
    assert "transport" not in properties


def test_agent_json_invocation_reaches_the_operation_callable(
    fixture: dict[str, object],
    cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from organelleverse.operations import registry

    transport = FakeTransport(fixture)
    monkeypatch.setattr(genome_size_module, "default_transport", lambda: transport)
    response = invoke_json(
        {
            "operation_id": "fetch.genome_size_candidates",
            "input": None,
            "parameters": {
                "scientific_name": str(fixture["species_name"]),
                "max_candidates": 5,
            },
        },
        registry=registry,
        granted_side_effects={SideEffect.NETWORK, SideEffect.WRITE_FILES},
    )
    assert response["ok"] is True
    data = OrganelleData.model_validate(response["result"])
    assert isinstance(data, OrganelleData)
    assert data.modality == "genome_size_candidates"
    assert _accessions(data) == EXPECTED_ORDER[:5]


def test_transport_url_encodes_the_scientific_name_path(
    fixture: dict[str, object], transport: FakeTransport, cache_root: Path
) -> None:
    genome_size_candidates(str(fixture["species_name"]), transport=transport)
    quoted = quote(str(fixture["species_name"]))
    assert any(f"/taxonomy/taxon/{quoted}" in url for url in transport.calls)
