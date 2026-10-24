"""Read the installed OV usage guide and its optional Agent Skills entry point.

Reading these resources does not discover operations or execute scientific work.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from importlib.resources import files

__all__ = ["main", "read_guide", "read_skill"]


def read_guide() -> str:
    """Return the usage instructions shipped with this installed OV version."""
    return files("organelleverse").joinpath("AGENTS.md").read_text(encoding="utf-8")


def read_skill() -> str:
    """Return the small SKILL.md that directs an agent to the installed guide.

    A host or user can save this text in its supported skill directory. Merely
    installing a Python distribution does not register a skill with any host.
    """
    return (
        files("organelleverse")
        .joinpath("agent_resources", "organelleverse", "SKILL.md")
        .read_text(encoding="utf-8")
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Print the guide, or print the skill with ``--skill``. No files are written."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skill", action="store_true", help="print the bundled SKILL.md")
    args = parser.parse_args(argv)
    sys.stdout.write(read_skill() if args.skill else read_guide())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
