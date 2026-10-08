from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from organelleverse.capabilities.discovery import discover_capability_candidates
from organelleverse.capabilities.scaffold import scaffold
from organelleverse.capabilities.snapshot import build_admission_snapshot
from organelleverse.capabilities.trust import TrustStore, trust
from organelleverse.capabilities.verification import (
    CandidateEvidence,
    EquivalenceEvidence,
    LocalVerificationEnvironment,
    VerificationStore,
    verify_capability,
)


def _isolate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ORGANELLEVERSE_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    monkeypatch.delenv("ORGANELLEVERSE_TRUST", raising=False)


def _candidate_index(tmp_path: Path):
    search_root = tmp_path / "capabilities"
    scaffold(
        search_root / "demo-snapshot",
        capability_id="demo.snapshot",
        title="Snapshot demo",
        description="A deterministic admission snapshot fixture.",
    )
    return search_root, discover_capability_candidates(paths=[search_root], entry_points=())


def _store_bytes(root: Path) -> dict[str, bytes]:
    if not root.exists():
        return {}
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_fresh_snapshot_is_read_only_and_reports_missing_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate(monkeypatch, tmp_path)
    _, candidates = _candidate_index(tmp_path)
    verification_store = VerificationStore(tmp_path / "state" / "verifications")
    trust_store = TrustStore(tmp_path / "state" / "trust.json")

    before = _store_bytes(tmp_path / "state")
    snapshot = build_admission_snapshot(
        candidates,
        verification_store=verification_store,
        trust_store=trust_store,
    )
    after = _store_bytes(tmp_path / "state")

    assert before == after == {}
    assert snapshot.counts.model_dump() == {
        "discovered": 1,
        "admitted": 0,
        "not_admitted": 1,
        "agent_visible": 0,
        "admitted_but_untrusted": 0,
        "unsupported": 0,
        "conflicted_ids": 0,
        "discovery_diagnostics": 0,
    }
    assert snapshot.not_admitted_by_reason == {"capability.verification_missing": 1}
    assert snapshot.agent_hidden_by_reason == {}
    assert snapshot.capabilities[0].reason_code == "capability.verification_missing"
    assert snapshot.capabilities[0].agent_visible is False
    assert snapshot.platform.python


@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes only")
def test_rejected_candidate_does_not_consult_irrelevant_trust_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate(monkeypatch, tmp_path)
    _, candidates = _candidate_index(tmp_path)
    trust_path = tmp_path / "state" / "trust.json"
    trust_path.parent.mkdir(parents=True)
    trust_path.write_text('{"schema":"organelleverse.trust.v2","trusted":[]}\n')
    trust_path.chmod(0o666)

    snapshot = build_admission_snapshot(
        candidates,
        verification_store=VerificationStore(tmp_path / "state" / "verifications"),
        trust_store=TrustStore(trust_path),
    )

    assert snapshot.counts.admitted == 0
    assert snapshot.capabilities[0].trust_state == "untrusted"


def test_admitted_and_agent_visible_are_separate_trust_states(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate(monkeypatch, tmp_path)
    _, candidates = _candidate_index(tmp_path)
    verification_store = VerificationStore(tmp_path / "state" / "verifications")
    trust_store = TrustStore(tmp_path / "state" / "trust.json")
    verify_capability(
        "demo.snapshot",
        store=verification_store,
        environment=LocalVerificationEnvironment(candidates),
    )
    verified_bytes = _store_bytes(verification_store.root)

    first = build_admission_snapshot(
        candidates,
        verification_store=verification_store,
        trust_store=trust_store,
    )
    second = build_admission_snapshot(
        candidates,
        verification_store=verification_store,
        trust_store=trust_store,
    )

    assert first == second
    assert _store_bytes(verification_store.root) == verified_bytes
    assert not trust_store.path.exists()
    assert first.counts.admitted == 1
    assert first.counts.agent_visible == 0
    assert first.counts.admitted_but_untrusted == 1
    assert first.agent_hidden_by_reason == {"capability.untrusted": 1}
    assert first.capabilities[0].trust_state == "untrusted"

    entry = candidates.describe("demo.snapshot")
    assert entry.execution_identity is not None
    trust(entry.capability_id, entry.execution_identity, store=trust_store)
    trusted = build_admission_snapshot(
        candidates,
        verification_store=verification_store,
        trust_store=trust_store,
    )
    assert trusted.counts.admitted == 1
    assert trusted.counts.agent_visible == 1
    assert trusted.counts.admitted_but_untrusted == 0
    assert trusted.capabilities[0].trust_state == "trusted"


def test_snapshot_distinguishes_stale_platform_and_failed_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate(monkeypatch, tmp_path)
    search_root, candidates = _candidate_index(tmp_path)
    verification_store = VerificationStore(tmp_path / "state" / "verifications")
    trust_store = TrustStore(tmp_path / "state" / "trust.json")
    record = verify_capability(
        "demo.snapshot",
        store=verification_store,
        environment=LocalVerificationEnvironment(candidates),
    )

    mismatched = record.model_copy(
        update={
            "platform": record.platform.model_copy(update={"python": "0.0.0"}),
        }
    )
    verification_store.write(mismatched)
    platform_snapshot = build_admission_snapshot(
        candidates,
        verification_store=verification_store,
        trust_store=trust_store,
    )
    assert platform_snapshot.not_admitted_by_reason == {"capability.platform_mismatch": 1}

    failed_evidence = EquivalenceEvidence(
        case="failure",
        method="exact",
        evaluator_version="organelleverse.fixture-evaluator.v2",
        verdict="fail",
        fixture_dataset_hash=record.fixture_dataset_hash,
        candidate=CandidateEvidence(
            bundle_content_hash=record.bundle_content_hash,
            environment_key=record.environment_key,
            produced_hash="sha256:" + "0" * 64,
        ),
    )
    verification_store.write(record.model_copy(update={"equivalence": (failed_evidence,)}))
    failed_snapshot = build_admission_snapshot(
        candidates,
        verification_store=verification_store,
        trust_store=trust_store,
    )
    assert failed_snapshot.not_admitted_by_reason == {"capability.equivalence_failed": 1}

    bundle_readme = search_root / "demo-snapshot" / "README.md"
    bundle_readme.write_text(
        bundle_readme.read_text(encoding="utf-8") + "changed\n", encoding="utf-8"
    )
    changed = discover_capability_candidates(paths=[search_root], entry_points=())
    stale_snapshot = build_admission_snapshot(
        changed,
        verification_store=verification_store,
        trust_store=trust_store,
    )
    assert stale_snapshot.not_admitted_by_reason == {"capability.verification_stale": 1}


def test_snapshot_json_is_canonical_and_contains_no_timestamp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate(monkeypatch, tmp_path)
    _, candidates = _candidate_index(tmp_path)
    snapshot = build_admission_snapshot(
        candidates,
        verification_store=VerificationStore(tmp_path / "state" / "verifications"),
        trust_store=TrustStore(tmp_path / "state" / "trust.json"),
    )

    first = json.dumps(snapshot.model_dump(mode="json", by_alias=True), sort_keys=True)
    second = json.dumps(snapshot.model_dump(mode="json", by_alias=True), sort_keys=True)
    assert first == second
    assert "timestamp" not in first
    assert "verified_at" not in first
