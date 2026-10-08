"""CLI for listing and locating registered environments.

Commands::

    organelleverse-environments list [--json]
    organelleverse-environments locate <backend> [--json]

The human-readable form prints the activation instruction
(``conda activate /absolute/provider/prefix``) verbatim. ``--json`` emits a
canonical JSON array of environment records.
"""

from __future__ import annotations

import argparse
import json

from organelleverse import environments


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="organelleverse-environments",
        description="List and locate verified OrganelleVerse backend environments.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    list_parser = sub.add_parser("list", help="list all registered environments")
    list_parser.add_argument(
        "--json",
        action="store_true",
        help="emit a canonical JSON array instead of human-readable text",
    )

    locate_parser = sub.add_parser("locate", help="locate a specific backend")
    locate_parser.add_argument("backend", help="backend id, e.g. oatk")
    locate_parser.add_argument(
        "--json",
        action="store_true",
        help="emit a canonical JSON array instead of human-readable text",
    )
    return parser


def _render_human(records: tuple[environments.EnvironmentRecord, ...]) -> str:
    lines: list[str] = []
    for record in records:
        lines.append(f"{record.backend_id}\t{record.version or '-'}\t{record.source}")
        if record.activation:
            lines.append(f"  {record.activation}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    records = environments.list() if args.command == "list" else environments.locate(args.backend)

    if args.json:
        payload = [record.model_dump(mode="json") for record in records]
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        text = _render_human(records)
        if text:
            print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
