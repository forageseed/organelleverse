"""CLI for the external developer / Agent capability authoring loop.

Commands::

    organelleverse-capability init <target_dir> --id ID --title T --description D [--json]
    organelleverse-capability validate <search_path>... [--json]
    organelleverse-capability verify <capability_id> --search <search_path>...
        [--verification-store PATH] [--json]
    organelleverse-capability verify-all [--search <search_path>...]
        [--verification-store PATH] [--continue-on-error] [--json]
    organelleverse-capability trust <capability_id> --search <search_path>...
        [--trust-store PATH] [--json]
    organelleverse-capability check <capability_id> --search <search_path>...
        [--verification-store PATH] --json
    organelleverse-capability snapshot [--search <search_path>...]
        [--verification-store PATH] [--trust-store PATH] [--json]

This module is a **thin front end**. Every command below is a direct call
into the same public Python API an external developer or Agent would use --
``organelleverse.capabilities.scaffold``, ``discover_capabilities``,
``verify_capability``, ``trust``, and ``admit_capabilities`` -- with no
behaviour that exists only here. See
``tests/capabilities/test_capability_cli.py`` for the tests that prove this
by running the same operation through this CLI and through the Python API
and asserting identical results, per Capability Plan 02 Task 3's charter
(``.superpowers/sdd/charters/capability-02-task-3-authoring.md``, deliverable
5).

Security notice text (required by the charter to appear in ``init`` and/or
``verify`` output): ``verify_capability`` executes the bundle's code inside
the controlled worker subprocess before any trust record exists. See
``organelleverse.capabilities.verification`` and the generated bundle's own
``README.md`` for the full statement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from organelleverse.capabilities.admission import admit_capabilities
from organelleverse.capabilities.discovery import (
    discover_capabilities,
    discover_capability_candidates,
)
from organelleverse.capabilities.index import CapabilityDiagnostic, CapabilityIndex
from organelleverse.capabilities.scaffold import ScaffoldParameter, scaffold
from organelleverse.capabilities.snapshot import (
    CapabilityAdmissionSnapshot,
    build_admission_snapshot,
)
from organelleverse.capabilities.trust import TrustStore, trust
from organelleverse.capabilities.verification import (
    LocalVerificationEnvironment,
    VerificationStore,
    default_verification_store,
    verify_capability,
)
from organelleverse.core.errors import OrganelleError

SECURITY_NOTICE = (
    "verify_capability executes this bundle's code inside the controlled "
    "worker before any trust record exists. Read the code under code/ "
    "before running `capability verify`."
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="organelleverse-capability",
        description=(
            "Scaffold, validate, verify, trust, check, and snapshot external capability "
            "bundles through the same public Python API an Agent would use."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    init_parser = sub.add_parser("init", help="scaffold a new capability bundle")
    init_parser.add_argument("target", help="directory to create the bundle in")
    init_parser.add_argument("--id", dest="capability_id", required=True, help="capability id")
    init_parser.add_argument("--title", required=True, help="human-readable title")
    init_parser.add_argument("--description", required=True, help="human-readable description")
    init_parser.add_argument("--bundle-version", default="1.0.0", help="bundle_version (semver)")
    init_parser.add_argument("--stage", default="analyze", help="pipeline stage")
    init_parser.add_argument(
        "--package-name", default=None, help="generated code/ package name (default: derived)"
    )
    init_parser.add_argument(
        "--parameter",
        dest="parameters",
        action="append",
        default=None,
        metavar="NAME:ANNOTATION:DEFAULT_JSON",
        help=(
            "one named parameter, repeatable, e.g. --parameter value:int:1 "
            "(default: scaffold's own single 'value: int = 1' parameter)"
        ),
    )
    init_parser.add_argument("--json", action="store_true", help="emit canonical JSON")

    validate_parser = sub.add_parser(
        "validate", help="discover bundles under one or more search paths and report status"
    )
    validate_parser.add_argument("paths", nargs="+", help="search paths to scan")
    validate_parser.add_argument("--json", action="store_true", help="emit canonical JSON")

    verify_parser = sub.add_parser(
        "verify", help="run verify_capability with the default local verification environment"
    )
    verify_parser.add_argument("capability_id", help="capability id to verify")
    verify_parser.add_argument(
        "--search", dest="search", nargs="+", required=True, help="search paths to scan"
    )
    verify_parser.add_argument(
        "--verification-store",
        default=None,
        help="verification store directory (default: ~/.organelleverse/verifications)",
    )
    verify_parser.add_argument("--json", action="store_true", help="emit canonical JSON")

    verify_all_parser = sub.add_parser(
        "verify-all",
        help=(
            "verify every discovered capability into the store (the owner-approved "
            "'verify once, reuse the record' batch form of verify)"
        ),
    )
    verify_all_parser.add_argument(
        "--search",
        dest="search",
        nargs="+",
        default=None,
        help="search paths to scan (default: the standard discovery channels)",
    )
    verify_all_parser.add_argument(
        "--verification-store",
        default=None,
        help="verification store directory (default: ~/.organelleverse/verifications)",
    )
    verify_all_parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="record failures and continue instead of stopping at the first one",
    )
    verify_all_parser.add_argument("--json", action="store_true", help="emit canonical JSON")

    trust_parser = sub.add_parser(
        "trust", help="authorize one discovered execution identity for invocation"
    )
    trust_parser.add_argument("capability_id", help="capability id to trust")
    trust_parser.add_argument(
        "--search", dest="search", nargs="+", required=True, help="search paths to scan"
    )
    trust_parser.add_argument(
        "--trust-store",
        default=None,
        help="trust store file path (default: ~/.organelleverse/trust.json)",
    )
    trust_parser.add_argument("--json", action="store_true", help="emit canonical JSON")

    check_parser = sub.add_parser(
        "check", help="discover and report admission status against a verification store"
    )
    check_parser.add_argument("capability_id", help="capability id to check")
    check_parser.add_argument(
        "--search", dest="search", nargs="+", required=True, help="search paths to scan"
    )
    check_parser.add_argument(
        "--verification-store",
        default=None,
        help="verification store directory (default: ~/.organelleverse/verifications)",
    )
    check_parser.add_argument("--json", action="store_true", help="emit canonical JSON")

    snapshot_parser = sub.add_parser(
        "snapshot",
        help="report deterministic admission and Agent-visibility counts without writing stores",
    )
    snapshot_parser.add_argument(
        "--search",
        dest="search",
        nargs="+",
        default=None,
        help="search paths to scan (default: the standard discovery channels)",
    )
    snapshot_parser.add_argument(
        "--verification-store",
        default=None,
        help="verification store directory (default: ~/.organelleverse/verifications)",
    )
    snapshot_parser.add_argument(
        "--trust-store",
        default=None,
        help="trust store file path (default: ~/.organelleverse/trust.json)",
    )
    snapshot_parser.add_argument("--json", action="store_true", help="emit canonical JSON")

    return parser


def _parse_scaffold_parameter(raw: str) -> ScaffoldParameter:
    parts = raw.split(":", 2)
    if len(parts) != 3:
        raise ValueError(f"--parameter must be NAME:ANNOTATION:DEFAULT_JSON, got {raw!r}")
    name, annotation, default_json = parts
    return ScaffoldParameter(name=name, annotation=annotation, default=json.loads(default_json))


def _verification_store(path: str | None) -> VerificationStore:
    return VerificationStore(Path(path)) if path is not None else default_verification_store()


def _trust_store(path: str | None) -> TrustStore:
    return TrustStore(Path(path)) if path is not None else TrustStore()


def _diagnostic_payload(diagnostic: CapabilityDiagnostic | None) -> dict[str, object] | None:
    return None if diagnostic is None else diagnostic.model_dump(mode="json")


def _entry_payload(index: CapabilityIndex, capability_id: str) -> dict[str, object]:
    entry = index.describe(capability_id)
    return {
        "capability_id": entry.capability_id,
        "status": entry.status.value,
        "content_hash": entry.content_hash,
        "diagnostic": _diagnostic_payload(entry.diagnostic),
    }


def _validate_payload(index: CapabilityIndex) -> dict[str, object]:
    capabilities = [
        {
            "capability_id": entry.capability_id,
            "status": entry.status.value,
            "content_hash": entry.content_hash,
            "diagnostic": _diagnostic_payload(entry.diagnostic),
        }
        for entry in index.entries
    ]
    conflicts = [conflict.model_dump(mode="json") for conflict in index.conflicts()]
    return {"capabilities": capabilities, "conflicts": conflicts}


def _cmd_init(args: argparse.Namespace) -> int:
    parameters = (
        tuple(_parse_scaffold_parameter(raw) for raw in args.parameters)
        if args.parameters is not None
        else None
    )
    kwargs: dict[str, object] = {
        "capability_id": args.capability_id,
        "title": args.title,
        "description": args.description,
        "bundle_version": args.bundle_version,
        "stage": args.stage,
        "package_name": args.package_name,
    }
    if parameters is not None:
        kwargs["parameters"] = parameters
    bundle_root = scaffold(Path(args.target), **kwargs)  # type: ignore[arg-type]
    payload = {
        "bundle_root": str(bundle_root),
        "capability_id": args.capability_id,
        "security_notice": SECURITY_NOTICE,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"scaffolded {args.capability_id} at {bundle_root}")
        print(f"NOTICE: {SECURITY_NOTICE}")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    index = discover_capabilities(paths=[Path(path) for path in args.paths])
    payload = _validate_payload(index)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for capability in payload["capabilities"]:
            print(
                f"{capability['capability_id']}\t{capability['status']}\t"
                f"{capability['content_hash']}"
            )
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    print(f"NOTICE: {SECURITY_NOTICE}")
    index = discover_capabilities(paths=[Path(path) for path in args.search])
    store = _verification_store(args.verification_store)
    environment = LocalVerificationEnvironment(index)
    record = verify_capability(args.capability_id, store=store, environment=environment)
    payload = record.model_dump(mode="json", by_alias=True)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            f"verified {record.capability_id} bundle_content_hash={record.bundle_content_hash} "
            f"environment_key={record.environment_key}"
        )
    return 0


def _cmd_trust(args: argparse.Namespace) -> int:
    index = discover_capabilities(paths=[Path(path) for path in args.search])
    entry = index.describe(args.capability_id)
    if entry.execution_identity is None:
        raise SystemExit(
            f"capability has no content-addressed execution identity: {args.capability_id}"
        )
    store = _trust_store(args.trust_store)
    record = trust(args.capability_id, entry.execution_identity, store=store)
    payload = record.model_dump(mode="json")
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"trusted {record.capability_id} execution_identity={record.execution_identity}")
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    index = discover_capabilities(paths=[Path(path) for path in args.search])
    store = _verification_store(args.verification_store)
    admitted_index = admit_capabilities(index, store=store)
    payload = _entry_payload(admitted_index, args.capability_id)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"{payload['capability_id']}\t{payload['status']}")
    return 0


def _cmd_verify_all(args: argparse.Namespace) -> int:
    print(f"NOTICE: {SECURITY_NOTICE}")
    if args.search is not None:
        index = discover_capabilities(paths=[Path(path) for path in args.search])
    else:
        index = discover_capabilities()
    store = _verification_store(args.verification_store)
    environment = LocalVerificationEnvironment(index)
    verified: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for entry in index.entries:
        capability_id = entry.capability_id
        try:
            record = verify_capability(capability_id, store=store, environment=environment)
        except OrganelleError as error:
            payload = {
                "capability_id": capability_id,
                "error_code": error.code,
                "message": error.message,
            }
            if not args.continue_on_error:
                if args.json:
                    print(json.dumps(payload, indent=2, sort_keys=True))
                else:
                    print(f"FAILED {capability_id} [{error.code}] {error.message}")
                return 1
            failures.append(payload)
            if not args.json:
                print(f"FAILED {capability_id} [{error.code}] {error.message}")
            continue
        verified.append(
            {
                "capability_id": record.capability_id,
                "bundle_content_hash": record.bundle_content_hash,
                "environment_key": record.environment_key,
            }
        )
        if not args.json:
            print(
                f"verified {record.capability_id} "
                f"bundle_content_hash={record.bundle_content_hash} "
                f"environment_key={record.environment_key}"
            )
    summary = {
        "discovered": len(index.entries),
        "verified": len(verified),
        "failed": len(failures),
        "verification_store": str(store.root),
    }
    if args.json:
        print(
            json.dumps(
                {"summary": summary, "verified": verified, "failures": failures},
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(
            f"verified {summary['verified']}/{summary['discovered']} capabilities "
            f"({summary['failed']} failed) into {summary['verification_store']}"
        )
    return 1 if failures else 0


def _print_snapshot(snapshot: CapabilityAdmissionSnapshot) -> None:
    platform = snapshot.platform
    counts = snapshot.counts
    print(
        "platform "
        f"os={platform.os} arch={platform.arch} libc={platform.libc} python={platform.python}"
    )
    print(
        f"discovered={counts.discovered} admitted={counts.admitted} "
        f"not_admitted={counts.not_admitted} agent_visible={counts.agent_visible} "
        f"admitted_but_untrusted={counts.admitted_but_untrusted}"
    )
    print(
        f"unsupported={counts.unsupported} conflicted_ids={counts.conflicted_ids} "
        f"discovery_diagnostics={counts.discovery_diagnostics}"
    )
    if snapshot.not_admitted_by_reason:
        print("not_admitted_by_reason")
        for reason, count in snapshot.not_admitted_by_reason.items():
            print(f"  {reason}={count}")
    if snapshot.agent_hidden_by_reason:
        print("agent_hidden_by_reason")
        for reason, count in snapshot.agent_hidden_by_reason.items():
            print(f"  {reason}={count}")


def _cmd_snapshot(args: argparse.Namespace) -> int:
    paths = None if args.search is None else [Path(path) for path in args.search]
    candidates = discover_capability_candidates(paths=paths)
    snapshot = build_admission_snapshot(
        candidates,
        verification_store=_verification_store(args.verification_store),
        trust_store=_trust_store(args.trust_store),
    )
    if args.json:
        print(
            json.dumps(
                snapshot.model_dump(mode="json", by_alias=True),
                indent=2,
                sort_keys=True,
            )
        )
    else:
        _print_snapshot(snapshot)
    return 0


_COMMANDS = {
    "init": _cmd_init,
    "validate": _cmd_validate,
    "verify": _cmd_verify,
    "verify-all": _cmd_verify_all,
    "trust": _cmd_trust,
    "check": _cmd_check,
    "snapshot": _cmd_snapshot,
}


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    handler = _COMMANDS[args.command]
    try:
        return handler(args)
    except OrganelleError as error:
        print(json.dumps(error.as_dict(), indent=2, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
