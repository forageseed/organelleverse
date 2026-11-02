"""Publish captured post hoc reports alongside an immutable optimization study."""

import argparse
import json
import shutil
from pathlib import Path

from organelleverse.core.artifacts import ArtifactRef
from organelleverse.pangenome.graph_selection import verified_artifact
from types import SimpleNamespace
from typing import Protocol
from organelleverse.optimization.evidence_models import RepeatEvidence

class StudyDiagnosticEvidence(Protocol):
    study_id: str
    status: str
    contract_digest: str
    attempts: tuple[RepeatEvidence, ...]



def publish_diagnostics(record: StudyDiagnosticEvidence, source: Path, destination: Path):
    summary = json.loads((source / "summary.json").read_text())
    if summary["study_id"] != record.study_id or record.status not in {
        "adopted",
        "baseline_retained",
    }:
        raise ValueError("Diagnostics must describe a completed matching study")
    if {r["run_id"] for r in summary["trials"]} != {a.run_id for a in record.attempts}:
        raise ValueError("Diagnostics must include every recorded attempt")
    sources = []
    for trial in summary["trials"]:
        if trial["contract_digest"] != record.contract_digest:
            raise ValueError("Diagnostics contract differs from study")
        artifact = ArtifactRef.model_validate(trial["source_evaluation_artifact"])
        verified_artifact(artifact)
        sources.append({"run_id": trial["run_id"], "artifact": artifact.model_dump(mode="json")})
    destination.mkdir(parents=True, exist_ok=True)
    supplementary_sources = summary.get("supplementary_sources", [])
    for source_ref in supplementary_sources:
        verified_artifact(ArtifactRef.model_validate(source_ref))
    artifacts = []
    for name in (
        "summary.json",
        "diagnostics-report.md",
        "diagnostics.pdf",
        "diagnostics.svg",
        "diagnostics.png",
        "retrieval.tsv",
        "groups.tsv",
        "accession-deletion.tsv",
        "ranked-pairs.tsv",
        "structural-readiness.md",
        "structural-audit.json",
    ):
        shutil.copyfile(source / name, destination / name)
        artifacts.append(
            ArtifactRef.from_path(
                destination / name, kind="diagnostic_report", format=Path(name).suffix[1:]
            ).model_dump(mode="json")
        )
    if "block_comparison" in summary:
        for extension in ("json", "tsv", "md", "svg", "pdf"):
            name = f"block-comparison.{extension}"
            shutil.copyfile(source / name, destination / name)
            artifacts.append(
                ArtifactRef.from_path(
                    destination / name, kind="diagnostic_report", format=extension
                ).model_dump(mode="json")
            )
    manifest = {
        "supplementary_sources": supplementary_sources,
        "study_id": record.study_id,
        "contract_digest": record.contract_digest,
        "scope": "post_hoc_conformation_diagnostics",
        "sources": sources,
        "artifacts": artifacts,
    }
    (destination / "index.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study_directory", type=Path)
    args = parser.parse_args()
    root = args.study_directory
    payload = json.loads((root / "study-result.json").read_text())
    record = SimpleNamespace(study_id=payload["study_id"], status=payload["status"],
                             contract_digest=payload["contract_digest"],
                             attempts=tuple(RepeatEvidence.model_validate(item) for item in payload["attempts"]))
    destination = root / "evidence" / record.study_id / "diagnostics"
    publish_diagnostics(record, root / "diagnostics-v1", destination)
    print(destination)


if __name__ == "__main__":
    main()
