from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.errors import OrganelleExecutionError, OrganelleInputError
from organelleverse.core.genome import OrganelleGenome, OrganelleMetadata
from organelleverse.core.result import ErrorDetail, OrganelleResult
from organelleverse.pangenome import service, tuning_service
from organelleverse.pangenome._contract import make_provenance
from organelleverse.pangenome._runner import CommandRecord


def test_repeatmasker_failure_message_includes_stderr_tail() -> None:
    record = CommandRecord(("RepeatMasker", "input.fa"), 2, None, "missing RepeatMasker library\n")
    with pytest.raises(OrganelleExecutionError) as caught:
        tuning_service._require_success(record, "RepeatMasker")
    assert "missing RepeatMasker library" in caught.value.message


def _genomes(tmp_path: Path) -> list[OrganelleGenome]:
    genomes: list[OrganelleGenome] = []
    for accession, sequence in (("sample_a", "ACGT" * 40), ("sample_b", "ACGA" * 40)):
        fasta = tmp_path / f"{accession}.fa"
        fasta.write_text(f">chr1\n{sequence}\n", encoding="utf-8")
        genomes.append(
            OrganelleGenome(
                organelle="plastid",
                sequence=ArtifactRef.from_path(fasta, kind="sequence", format="fasta"),
                metadata=OrganelleMetadata(accession=accession),
            )
        )
    return genomes


def test_recommendation_uses_controlled_tools_and_publishes_exact_metrics_copy(
    tmp_path: Path, monkeypatch
) -> None:
    genomes = _genomes(tmp_path)
    invocations: list[tuple[str, ...]] = []
    staged_headers: list[str] = []

    def fake_find(name: str) -> str:
        return f"/verified/tools/{name}"

    def fake_run(argv, *, stdout_path=None, cwd=None):
        command = tuple(str(item) for item in argv)
        invocations.append(command)
        if stdout_path is not None:
            target = Path(stdout_path)
            if "--version" in command or "-v" in command:
                target.write_text(f"{Path(command[0]).name} 9.1\n", encoding="utf-8")
            elif command[1] == "dist":
                target.write_text(_sample_pair_output(Path(cwd)), encoding="utf-8")
        if len(command) > 1 and command[1] == "sketch":
            for staged_name in command[command.index("-o") + 2 :]:
                staged_headers.extend(
                    line.strip()
                    for line in Path(staged_name).read_text(encoding="utf-8").splitlines()
                    if line.startswith(">")
                )
            Path(command[command.index("-o") + 1] + ".msh").write_bytes(b"mash")
        if Path(command[0]).name == "RepeatMasker" and "-v" not in command:
            output_dir = Path(command[command.index("-dir") + 1])
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / f"{Path(command[-1]).name}.out").write_text(
                "463 1.3 0.0 0.0 sample_a#1#1 10 50 (0) + r1 DNA/hAT 1 41 (0) 1\n",
                encoding="utf-8",
            )
        return CommandRecord(command, 0, None if stdout_path is None else str(stdout_path), "")

    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    monkeypatch.setenv("REPEATMASKER_DATABASE_LABEL", "rm-db-2025")
    monkeypatch.setattr(tuning_service, "_find_executable", fake_find)
    monkeypatch.setattr(tuning_service, "_run_command", fake_run)

    result = tuning_service.recommend_parameters(genomes, species="Arabidopsis thaliana")

    assert result.status == "ok"
    recommendation = result.model_dump(mode="json")["metrics"]["recommendation"]
    artifact = next(
        item for item in result.artifacts if item.kind == "pangenome_parameter_recommendation"
    )
    assert json.loads(Path(artifact.uri).read_text(encoding="utf-8")) == recommendation
    assert recommendation["identity"] == 94
    assert recommendation["segment_length"] == 5000
    assert result.metrics["segment_length_policy"]["derived"] == 50
    assert recommendation["mash_executable"] == "/verified/tools/mash"
    assert recommendation["mash_version"] == "mash 9.1"
    assert recommendation["repeatmasker_executable"] == "/verified/tools/RepeatMasker"
    assert recommendation["repeatmasker_version"] == "RepeatMasker 9.1"
    assert recommendation["repeatmasker_database_label"] == "rm-db-2025"
    assert recommendation["input_hashes"] == [
        genome.sequence.sha256 for genome in genomes if genome.sequence is not None
    ]
    assert staged_headers == [">sample_a#1#1", ">sample_b#1#1"]
    assert all(isinstance(argv, tuple) and ">" not in argv for argv in invocations)
    assert result.provenance is not None
    assert result.provenance.actual_backend == "mash+repeatmasker"

    # Identical sequence bytes with a different organelle must not reuse the
    # plastid-specific floor from the recommendation cache.
    mitochondrial = [g.model_copy(update={"organelle": "mitochondrion"}) for g in genomes]
    other = tuning_service.recommend_parameters(mitochondrial, species="Arabidopsis thaliana")
    assert other.metrics["recommendation"]["segment_length"] == 50
    assert other.metrics["segment_length_policy"]["floor"] is None


@pytest.mark.parametrize("mutation", ("tamper", "symlink", "extra"))
def test_recommendation_cache_reopen_rejects_unsafe_or_changed_tree(
    tmp_path: Path, monkeypatch, mutation: str
) -> None:
    genomes = _genomes(tmp_path)

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

    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    monkeypatch.setattr(tuning_service, "_find_executable", lambda name: "/tools/mash")
    monkeypatch.setattr(tuning_service, "_run_command", fake_run)
    first = tuning_service.recommend_parameters(genomes, run_repeatmasker=False)
    recommendation_path = next(
        Path(item.uri)
        for item in first.artifacts
        if item.kind == "pangenome_parameter_recommendation"
    )
    owner = recommendation_path.parent.parent
    if mutation == "tamper":
        payload = json.loads(recommendation_path.read_text(encoding="utf-8"))
        payload["identity"] = 93
        unsigned = dict(payload)
        unsigned.pop("recommendation_digest")
        payload["recommendation_digest"] = hashlib.sha256(
            json.dumps(
                unsigned,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
        ).hexdigest()
        recommendation_path.write_text(
            json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n",
            encoding="utf-8",
        )
    elif mutation == "symlink":
        outside = tmp_path / "outside-recommendation.json"
        shutil.copy2(recommendation_path, outside)
        recommendation_path.unlink()
        recommendation_path.symlink_to(outside)
    else:
        (owner / "undeclared.txt").write_text("extra\n", encoding="utf-8")

    with pytest.raises(OrganelleInputError):
        tuning_service.recommend_parameters(genomes, run_repeatmasker=False)


def _recommendation(genomes: list[OrganelleGenome]) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "organelleverse.pangenome.recommendation.v1",
        "policy_version": "minegraph-compatible.v1",
        "input_hashes": [
            genome.sequence.sha256 for genome in genomes if genome.sequence is not None
        ],
        "threads": 4,
        "run_repeatmasker": True,
        "species": "Arabidopsis thaliana",
        "identity_margin": 2.0,
        "repeat_multiplier": 1.2,
        "round_to": 10,
        "no_repeat_fallback": 5000,
        "include_rna": True,
        "mash_executable": "/verified/tools/mash",
        "mash_version": "mash 9.1",
        "repeatmasker_executable": "/verified/tools/RepeatMasker",
        "repeatmasker_version": "RepeatMasker 9.1",
        "repeatmasker_library_identity": "Arabidopsis thaliana",
        "repeatmasker_database_label": "rm-db-2025",
        "identity": 94.0,
        "segment_length": 5000,
        "max_mash_distance": 0.041,
        "longest_repeat_span": 41,
        "repeat_fallback_used": False,
        "mash_sketch_argv": ["/verified/tools/mash", "sketch"],
        "mash_distance_argv": ["/verified/tools/mash", "dist"],
        "repeatmasker_argv": ["/verified/tools/RepeatMasker"],
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode()
    payload["recommendation_digest"] = hashlib.sha256(encoded).hexdigest()
    return payload


@pytest.mark.parametrize(
    "automatic,with_recommendation", [(False, True), (True, True), (False, False)]
)
def test_managed_build_adopts_exact_recommendation_in_graph_owned_record(
    tmp_path: Path, monkeypatch, automatic, with_recommendation
) -> None:
    genomes = _genomes(tmp_path)
    recommendation = _recommendation(genomes)
    calls: list[dict[str, object]] = []

    def fake_core_build_graph(*args, **kwargs):
        calls.append(kwargs)
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

    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    monkeypatch.setattr(service, "_core_build_graph", fake_core_build_graph)

    result = service.build_graph(
        genomes,
        method="pggb",
        identity=90 if automatic else 94,
        segment_length=10000 if automatic else 5000 if with_recommendation else 500,
        recommendation=recommendation if with_recommendation else None,
        auto_adopt_recommendation=automatic,
    )

    assert calls[0]["identity"] == 94
    assert calls[0]["segment_length"] == 5000
    assert result.metrics["parameter_policy"]["auto_adopt_recommendation"] is automatic
    assert result.metrics["segment_length"] == 5000
    assert result.metrics["requested_segment_length"] == (5000 if with_recommendation else 500)
    assert result.metrics["parameter_policy"]["derived_segment_length"] == (
        50 if with_recommendation else None
    )
    artifacts = {artifact.kind: artifact for artifact in result.artifacts}
    if not with_recommendation:
        assert "pangenome_parameter_adoption" not in artifacts
        return
    adoption = json.loads(
        Path(artifacts["pangenome_parameter_adoption"].uri).read_text(encoding="utf-8")
    )
    assert adoption == {
        "schema_version": "organelleverse.pangenome.adoption.v1",
        "recommendation_digest": recommendation["recommendation_digest"],
        "input_hashes": recommendation["input_hashes"],
        "identity": 94.0,
        "segment_length": 5000,
        "graph_sha256": artifacts["pangenome_graph"].sha256,
    }
    owner = Path(artifacts["pangenome_run_record"].uri).parent
    assert all(Path(artifact.uri).is_relative_to(owner) for artifact in result.artifacts)


def test_recommendation_mismatch_fails_before_backend_execution(
    tmp_path: Path, monkeypatch
) -> None:
    genomes = _genomes(tmp_path)
    recommendation = _recommendation(genomes)
    calls = 0

    def forbidden_builder(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("backend must not execute")

    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    monkeypatch.setattr(service, "_core_build_graph", forbidden_builder)

    with pytest.raises(OrganelleInputError) as captured:
        service.build_graph(
            genomes,
            method="pggb",
            identity=93,
            segment_length=5000,
            recommendation=recommendation,
        )

    assert captured.value.code == "pangenome.recommendation_mismatch"
    assert calls == 0


def test_recommendation_strict_schema_rejects_wrong_field_types(
    tmp_path: Path, monkeypatch
) -> None:
    genomes = _genomes(tmp_path)
    recommendation = _recommendation(genomes)
    recommendation["mash_version"] = 9
    unsigned = dict(recommendation)
    unsigned.pop("recommendation_digest")
    recommendation["recommendation_digest"] = hashlib.sha256(
        json.dumps(
            unsigned,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    monkeypatch.setattr(
        service,
        "_core_build_graph",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not execute")),
    )

    with pytest.raises(OrganelleInputError) as captured:
        service.build_graph(
            genomes,
            method="pggb",
            identity=94,
            segment_length=5000,
            recommendation=recommendation,
        )

    assert captured.value.code == "pangenome.invalid_recommendation"


def test_failed_builder_cannot_leak_graph_or_adoption_success_evidence(
    tmp_path: Path, monkeypatch
) -> None:
    genomes = _genomes(tmp_path)
    recommendation = _recommendation(genomes)

    def failed_builder(*args, **kwargs):
        leaked = Path(kwargs["output_dir"]) / "leaked-adoption.json"
        leaked.write_text("{}\n", encoding="utf-8")
        return OrganelleResult(
            operation_id="pangenome.build_graph",
            operation_version="0.1",
            scope="plastid",
            status="failed",
            flags=("graph_built",),
            artifacts=(
                ArtifactRef.from_path(leaked, kind="pangenome_parameter_adoption", format="json"),
            ),
            errors=(ErrorDetail(code="test.backend_failed", message="backend failed"),),
        )

    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    monkeypatch.setattr(service, "_core_build_graph", failed_builder)

    result = service.build_graph(
        genomes,
        method="pggb",
        identity=94,
        segment_length=5000,
        recommendation=recommendation,
    )

    assert result.status == "failed"
    assert "graph_built" not in result.flags
    assert not result.artifacts
    assert not list((tmp_path / "cache").rglob("adoption.json"))


def _sample_pair_output(workspace: Path) -> str:
    files = sorted((workspace / "mash_samples").glob("*.fa"))
    return "".join(
        f"{a}\t{b}\t{0 if a == b else 0.041}\t0\t100/1000\n" for a in files for b in files
    )
