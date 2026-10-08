"""One fresh process per native workflow stage; no process-global RSS estimates."""

from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from ._runner import run_command
from .workflow_models import WorkflowRequest


@dataclass(frozen=True)
class StageOutcome:
    outputs: list[Path]
    note: str | None
    resources: dict | None


class StageFailure(RuntimeError):
    def __init__(self, message: str, resources: dict):
        super().__init__(message)
        self.resources = resources


def execute_stage(name: str, directory: Path, request: WorkflowRequest) -> StageOutcome:
    """Use the existing controlled runner and exact child accounting boundary."""
    with tempfile.TemporaryDirectory(prefix="organelleverse-stage-") as temporary:
        source = Path(temporary) / "request.json"
        target = Path(temporary) / "response.json"
        source.write_text(request.model_dump_json())
        numerical_threads = {
            key: str(request.threads)
            for key in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        }
        command = run_command(
            [
                sys.executable,
                "-m",
                "organelleverse.pangenome.stage_worker",
                name,
                str(directory),
                str(source),
                str(target),
            ],
            cwd=directory,
            env_overrides=numerical_threads,
        )
        resources = dict(command.resource_usage)
        resources["numerical_thread_limits"] = numerical_threads
        resources["cpu_scope"] = (
            "isolated stage process and its waited descendants; includes process startup"
        )
        if not command.ok:
            raise StageFailure(
                f"Stage {name} process exited with code {command.returncode}: {command.stderr}",
                resources,
            )
        response = json.loads(target.read_text())
        if response["error"] is not None:
            raise StageFailure(response["error"], resources)
        return StageOutcome(
            [Path(path) for path in response["outputs"]], response["note"], resources
        )


def main() -> None:
    from .workflow import _execute_stage_body
    from .workflow_store import STAGES

    name, directory, source, target = sys.argv[1:]
    if name not in STAGES:
        raise ValueError(f"Unknown pangenome workflow stage: {name}")
    request = WorkflowRequest.model_validate_json(Path(source).read_text())
    try:
        outputs, note = _execute_stage_body(name, Path(directory), request)
        response = {"outputs": [str(path) for path in outputs], "note": note, "error": None}
    except Exception as error:
        response = {"outputs": [], "note": None, "error": str(error)}
    Path(target).write_text(json.dumps(response, allow_nan=False))


if __name__ == "__main__":
    main()
