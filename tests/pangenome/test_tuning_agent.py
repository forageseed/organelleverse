from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path

from organelleverse.capabilities.discovery import discover_capabilities
from organelleverse.capabilities.verification import (
    LocalVerificationEnvironment,
    VerificationStore,
    verify_capability,
)
from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.genome import OrganelleGenome, OrganelleMetadata
from organelleverse.core.result import OrganelleResult
from organelleverse.operations.adapters.json import invoke_json
from organelleverse.operations.registry import OperationRegistry
from organelleverse.pangenome import service, tuning_service
from organelleverse.pangenome._contract import make_provenance
from organelleverse.pangenome._runner import CommandRecord


def test_agent_hands_browser_safe_recommendation_unchanged_to_build(
    tmp_path: Path, monkeypatch
) -> None:
    home = tmp_path / "home"
    project = tmp_path / "project"
    home.mkdir()
    (project / ".organelleverse" / "capabilities").mkdir(parents=True)
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda **kwargs: ())
    monkeypatch.setenv("ORGANELLEVERSE_HOME", str(home))
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    monkeypatch.delenv("ORGANELLEVERSE_CAPABILITY_PATH", raising=False)
    monkeypatch.chdir(project)

    genomes: list[OrganelleGenome] = []
    for accession, sequence in (("agent_a", "ACGT" * 40), ("agent_b", "ACGA" * 40)):
        fasta = tmp_path / f"{accession}.fa"
        fasta.write_text(f">chr1\n{sequence}\n", encoding="utf-8")
        annotation = tmp_path / f"{accession}.gff3"
        annotation.write_text(
            "##gff-version 3\nchr1\ttest\tgene\t1\t4\t.\t+\t.\tID=gene1\n",
            encoding="utf-8",
        )
        genomes.append(
            OrganelleGenome(
                organelle="plastid",
                sequence=ArtifactRef.from_path(fasta, kind="sequence", format="fasta"),
                annotation=ArtifactRef.from_path(annotation, kind="annotation", format="gff3"),
                metadata=OrganelleMetadata(accession=accession),
            )
        )

    def fake_run(argv, *, stdout_path=None, cwd=None):
        command = tuple(str(item) for item in argv)
        if stdout_path is not None:
            target = Path(stdout_path)
            target.write_text(
                "mash 9.1\n" if "--version" in command else _sample_pair_output(Path(cwd)),
                encoding="utf-8",
            )
        if len(command) > 1 and command[1] == "sketch":
            Path(command[command.index("-o") + 1] + ".msh").write_bytes(b"mash")
        return CommandRecord(command, 0, None if stdout_path is None else str(stdout_path), "")

    def fake_builder(*args, **kwargs):
        graph = Path(kwargs["output_dir"]) / "pangenome.gfa"
        graph.write_text("H\tVN:Z:1.1\nS\tseg1\tACGT\n", encoding="utf-8")
        return OrganelleResult(
            operation_id="pangenome.build_graph",
            operation_version="0.1",
            scope="plastid",
            status="ok",
            flags=("graph_built",),
            artifacts=(ArtifactRef.from_path(graph, kind="pangenome_graph", format="gfa"),),
            provenance=make_provenance(
                operation_id="pangenome.build_graph",
                parameters={},
                requested_backend="pggb",
                actual_backend="pggb",
                attempted_backends=("pggb",),
            ),
        )

    monkeypatch.setattr(tuning_service, "_find_executable", lambda name: "/tools/mash")
    monkeypatch.setattr(tuning_service, "_run_command", fake_run)
    monkeypatch.setattr(service, "_core_build_graph", fake_builder)

    discovered = discover_capabilities()
    store = VerificationStore(home / "verifications")
    for capability_id in ("pangenome.recommend_parameters", "pangenome.build_graph"):
        verify_capability(
            capability_id,
            store=store,
            environment=LocalVerificationEnvironment(discovered),
        )
    admitted = discover_capabilities()
    registry = OperationRegistry()
    registry.attach_capability_source(admitted.binding_source())
    encoded_genomes = [genome.model_dump(mode="json") for genome in genomes]

    recommended = invoke_json(
        {
            "operation_id": "pangenome.recommend_parameters",
            "input": encoded_genomes,
            "parameters": {"run_repeatmasker": False},
        },
        registry=registry,
        granted_side_effects=["read_files", "write_files", "subprocess"],
    )
    assert recommended["ok"] is True, recommended.get("error")
    recommendation = recommended["result"]["metrics"]["recommendation"]
    browser_payload = json.loads(json.dumps(recommendation))

    built = invoke_json(
        {
            "operation_id": "pangenome.build_graph",
            "input": encoded_genomes,
            "parameters": {
                "method": "pggb",
                "identity": browser_payload["identity"],
                "segment_length": browser_payload["segment_length"],
                "recommendation": browser_payload,
            },
        },
        registry=registry,
        granted_side_effects=["read_files", "write_files", "subprocess"],
    )

    assert built["ok"] is True, built.get("error")
    result = built["result"]
    assert "graph_built" in result["flags"]
    adoption = next(
        artifact
        for artifact in result["artifacts"]
        if artifact["kind"] == "pangenome_parameter_adoption"
    )
    adopted = json.loads(Path(adoption["uri"]).read_text(encoding="utf-8"))
    assert adopted["recommendation_digest"] == browser_payload["recommendation_digest"]


def _sample_pair_output(workspace: Path) -> str:
    files = sorted((workspace / "mash_samples").glob("*.fa"))
    return "".join(
        f"{a}\t{b}\t{0 if a == b else 0.041}\t0\t100/1000\n" for a in files for b in files
    )
