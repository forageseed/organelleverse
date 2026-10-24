"""Python access to the same built-in analyses exposed in the workbench."""

from pathlib import Path
from typing import Any


def run(analysis: str, /, *, output: str | Path = "organelleverse-output", **parameters: Any) -> dict[str, Any]:
    """Run a built-in workbench analysis and return its report and output paths.

    Execute this on the desired compute device with its input paths. The desktop
    scheduler's ``compute_target_id`` is not a Python runner parameter.
    """
    from organelleverse.operations.builtin_tool_runners import BUILTIN_TOOL_RUNNERS

    if "compute_target_id" in parameters:
        raise ValueError("Run Python on the selected device; compute_target_id is a desktop setting")
    if analysis not in BUILTIN_TOOL_RUNNERS:
        raise ValueError(f"Unknown built-in workbench analysis: {analysis}")
    output_dir = Path(output).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    return BUILTIN_TOOL_RUNNERS[analysis](parameters, output_dir=output_dir, log=print)
