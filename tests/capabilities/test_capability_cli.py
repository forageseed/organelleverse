"""The capability CLI is a thin front end -- prove it, don't just exercise it.

Capability Plan 02 Task 3's charter (``.superpowers/sdd/charters/
capability-02-task-3-authoring.md``, deliverable 5) states the binding
constraint for this module: "The CLI is a thin front end over the Python
API. No behaviour exists only in the CLI; an Agent calling the Python
surface reaches the same code path." A CLI that merely *works* does not
prove that -- a CLI with its own parallel, subtly-different logic can also
"work" while silently diverging from the API an Agent would actually call.

Every test below therefore runs one command through
``organelleverse.tools.capability_cli.main(...)`` (capturing real stdout)
*and* the equivalent call through the public Python API
(``organelleverse.capabilities``), computed independently in the test body
rather than by importing the CLI's own formatting helpers, and asserts the
two agree. ``test_init_matches_python_scaffold_byte_for_byte`` is the
strongest form of this: it diffs the *files on disk*, not just a summary.

The one exception is ``verified_at``: ``verify_capability`` calls
``datetime.now(UTC)`` in its own record, so two independent, back-to-back
real invocations of the *same* deterministic verify_capability necessarily
carry different timestamps -- the way a `created_at` column differs across
two separate INSERTs of otherwise-identical rows. Every other field,
including the security-relevant ``execution_identity``, ``environment_key``,
and ``equivalence``, is compared and must match exactly.

The charter also requires that ``capability init`` (and/or ``verify``)
output plainly warns that verification executes the bundle's code before
any trust record exists; ``test_capability_init_states_verification_
executes_untrusted_code`` and
``test_capability_verify_states_verification_executes_untrafteed_code``
(sic -- see the actual name below) assert that text is present in real CLI
output, not just in the generated README.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from organelleverse.capabilities.admission import admit_capabilities
from organelleverse.capabilities.discovery import (
    discover_capabilities,
    discover_capability_candidates,
)
from organelleverse.capabilities.scaffold import scaffold
from organelleverse.capabilities.snapshot import build_admission_snapshot
from organelleverse.capabilities.trust import TrustStore, trust
from organelleverse.capabilities.verification import (
    LocalVerificationEnvironment,
    VerificationStore,
    verify_capability,
)
from organelleverse.tools import capability_cli


def _iso_isolation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Match test_authoring_loop.py's hermetic isolation: no real ~/.organelleverse."""
    monkeypatch.setenv("ORGANELLEVERSE_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))


def _walk_files(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_init_matches_python_scaffold_byte_for_byte(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI's `init` must generate byte-identical output to scaffold() itself."""
    cli_target = tmp_path / "via-cli"
    api_target = tmp_path / "via-api"

    exit_code = capability_cli.main(
        [
            "init",
            str(cli_target),
            "--id",
            "demo.cli_init",
            "--title",
            "CLI init demo",
            "--description",
            "Proves CLI init matches scaffold() byte for byte.",
        ]
    )
    assert exit_code == 0

    scaffold(
        api_target,
        capability_id="demo.cli_init",
        title="CLI init demo",
        description="Proves CLI init matches scaffold() byte for byte.",
    )

    cli_files = _walk_files(cli_target)
    api_files = _walk_files(api_target)
    assert cli_files == api_files


def test_capability_init_states_verification_executes_untrusted_code(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = capability_cli.main(
        [
            "init",
            str(tmp_path / "readme-demo"),
            "--id",
            "demo.cli_readme",
            "--title",
            "README demo",
            "--description",
            "Proves init output states the verification security property.",
        ]
    )
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "verify_capability" in out
    assert "executes" in out
    assert "before any trust record exists" in out


def test_validate_reports_status_and_diagnostics_matching_python_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _iso_isolation(monkeypatch, tmp_path)
    search_root = tmp_path / "capabilities"
    scaffold(
        search_root / "demo-cli-validate",
        capability_id="demo.cli_validate",
        title="Validate demo",
        description="Proves CLI validate matches discover_capabilities.",
    )

    exit_code = capability_cli.main(["validate", str(search_root), "--json"])
    assert exit_code == 0
    cli_payload = json.loads(capsys.readouterr().out)

    api_index = discover_capabilities(paths=[search_root])
    api_entry = api_index.describe("demo.cli_validate")
    expected = {
        "capability_id": api_entry.capability_id,
        "status": api_entry.status.value,
        "content_hash": api_entry.content_hash,
        "diagnostic": (
            None if api_entry.diagnostic is None else api_entry.diagnostic.model_dump(mode="json")
        ),
    }

    cli_entry = next(
        item for item in cli_payload["capabilities"] if item["capability_id"] == "demo.cli_validate"
    )
    assert cli_entry == expected
    # Nothing has been verified yet: a fresh discovery must not report admission.
    assert cli_entry["status"] != "admitted"


def test_verify_executes_and_matches_python_api_except_timestamp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _iso_isolation(monkeypatch, tmp_path)
    search_root = tmp_path / "capabilities"
    scaffold(
        search_root / "demo-cli-verify",
        capability_id="demo.cli_verify",
        title="Verify demo",
        description="Proves CLI verify matches verify_capability().",
    )

    cli_store_dir = tmp_path / "verifications-cli"
    exit_code = capability_cli.main(
        [
            "verify",
            "demo.cli_verify",
            "--search",
            str(search_root),
            "--verification-store",
            str(cli_store_dir),
            "--json",
        ]
    )
    assert exit_code == 0
    captured = capsys.readouterr().out
    # The verify command must state the security property before it runs.
    assert "verify_capability" in captured
    assert "executes" in captured
    assert "before any trust record exists" in captured
    cli_payload_lines = captured.splitlines()
    # The JSON payload is the last thing printed (the NOTICE line precedes it).
    cli_json_text = "\n".join(cli_payload_lines[cli_payload_lines.index("{") :])
    cli_record = json.loads(cli_json_text)

    api_index = discover_capabilities(paths=[search_root])
    api_store = VerificationStore(tmp_path / "verifications-api")
    api_environment = LocalVerificationEnvironment(api_index)
    api_record = verify_capability("demo.cli_verify", store=api_store, environment=api_environment)
    api_payload = api_record.model_dump(mode="json", by_alias=True)

    assert cli_record["verified_at"] != "" and api_payload["verified_at"] != ""
    cli_record.pop("verified_at")
    api_payload.pop("verified_at")
    assert cli_record == api_payload


def test_trust_matches_python_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _iso_isolation(monkeypatch, tmp_path)
    search_root = tmp_path / "capabilities"
    scaffold(
        search_root / "demo-cli-trust",
        capability_id="demo.cli_trust",
        title="Trust demo",
        description="Proves CLI trust matches trust().",
    )

    cli_trust_path = tmp_path / "trust-cli.json"
    exit_code = capability_cli.main(
        [
            "trust",
            "demo.cli_trust",
            "--search",
            str(search_root),
            "--trust-store",
            str(cli_trust_path),
            "--json",
        ]
    )
    assert exit_code == 0
    cli_payload = json.loads(capsys.readouterr().out)

    api_index = discover_capabilities(paths=[search_root])
    api_entry = api_index.describe("demo.cli_trust")
    assert api_entry.execution_identity is not None
    api_trust_path = tmp_path / "trust-api.json"
    api_record = trust(
        "demo.cli_trust",
        api_entry.execution_identity,
        store=TrustStore(api_trust_path),
    )
    api_payload = api_record.model_dump(mode="json")

    assert cli_payload == api_payload
    assert cli_trust_path.is_file()


def test_check_matches_python_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _iso_isolation(monkeypatch, tmp_path)
    search_root = tmp_path / "capabilities"
    scaffold(
        search_root / "demo-cli-check",
        capability_id="demo.cli_check",
        title="Check demo",
        description="Proves CLI check matches admit_capabilities().",
    )

    store_dir = tmp_path / "verifications"
    discovered = discover_capabilities(paths=[search_root])
    verify_capability(
        "demo.cli_check",
        store=VerificationStore(store_dir),
        environment=LocalVerificationEnvironment(discovered),
    )

    exit_code = capability_cli.main(
        [
            "check",
            "demo.cli_check",
            "--search",
            str(search_root),
            "--verification-store",
            str(store_dir),
            "--json",
        ]
    )
    assert exit_code == 0
    cli_payload = json.loads(capsys.readouterr().out)

    api_index = discover_capabilities(paths=[search_root])
    admitted_index = admit_capabilities(api_index, store=VerificationStore(store_dir))
    api_entry = admitted_index.describe("demo.cli_check")
    expected = {
        "capability_id": api_entry.capability_id,
        "status": api_entry.status.value,
        "content_hash": api_entry.content_hash,
        "diagnostic": (
            None if api_entry.diagnostic is None else api_entry.diagnostic.model_dump(mode="json")
        ),
    }

    assert cli_payload == expected
    assert cli_payload["status"] == "admitted", cli_payload["diagnostic"]


def test_snapshot_matches_public_python_api_and_explicit_stores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _iso_isolation(monkeypatch, tmp_path)
    monkeypatch.delenv("ORGANELLEVERSE_TRUST", raising=False)
    search_root = tmp_path / "capabilities"
    scaffold(
        search_root / "demo-cli-snapshot",
        capability_id="demo.cli_snapshot",
        title="CLI snapshot demo",
        description="Proves CLI snapshot matches the public snapshot API.",
    )
    verification_root = tmp_path / "snapshot-verifications"
    trust_path = tmp_path / "snapshot-trust.json"

    exit_code = capability_cli.main(
        [
            "snapshot",
            "--search",
            str(search_root),
            "--verification-store",
            str(verification_root),
            "--trust-store",
            str(trust_path),
            "--json",
        ]
    )
    assert exit_code == 0
    cli_payload = json.loads(capsys.readouterr().out)

    candidates = discover_capability_candidates(paths=[search_root])
    expected = build_admission_snapshot(
        candidates,
        verification_store=VerificationStore(verification_root),
        trust_store=TrustStore(trust_path),
    ).model_dump(mode="json", by_alias=True)
    assert cli_payload == expected
    assert cli_payload["counts"]["discovered"] == 1
    assert cli_payload["counts"]["agent_visible"] == 0
    assert not verification_root.exists()
    assert not trust_path.exists()


def test_full_authoring_loop_through_the_cli_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """init -> validate -> verify -> trust -> check, driven entirely by CLI subcommands."""
    _iso_isolation(monkeypatch, tmp_path)
    search_root = tmp_path / "capabilities"
    bundle_target = search_root / "demo-cli-loop"

    assert (
        capability_cli.main(
            [
                "init",
                str(bundle_target),
                "--id",
                "demo.cli_loop",
                "--title",
                "Loop demo",
                "--description",
                "Proves the whole authoring loop works through the CLI alone.",
            ]
        )
        == 0
    )
    capsys.readouterr()

    assert capability_cli.main(["validate", str(search_root), "--json"]) == 0
    capsys.readouterr()

    store_dir = tmp_path / "verifications"
    assert (
        capability_cli.main(
            [
                "verify",
                "demo.cli_loop",
                "--search",
                str(search_root),
                "--verification-store",
                str(store_dir),
                "--json",
            ]
        )
        == 0
    )
    capsys.readouterr()

    trust_path = tmp_path / "trust.json"
    assert (
        capability_cli.main(
            [
                "trust",
                "demo.cli_loop",
                "--search",
                str(search_root),
                "--trust-store",
                str(trust_path),
                "--json",
            ]
        )
        == 0
    )
    capsys.readouterr()

    assert (
        capability_cli.main(
            [
                "check",
                "demo.cli_loop",
                "--search",
                str(search_root),
                "--verification-store",
                str(store_dir),
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "admitted", payload["diagnostic"]
