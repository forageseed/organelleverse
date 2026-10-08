from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.genome import OrganelleGenome, OrganelleMetadata
from organelleverse.pangenome.tuning_service import recommend_parameters

pytestmark = pytest.mark.integration

def test_installed_mash_and_repeatmasker_write_real_recommendation_evidence(
    tmp_path: Path, monkeypatch
) -> None:
    mash = shutil.which("mash")
    repeatmasker = shutil.which("RepeatMasker")
    assert mash is not None, "release gate requires an installed Mash executable"
    assert repeatmasker is not None, "release gate requires an installed RepeatMasker executable"
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))

    genomes: list[OrganelleGenome] = []
    sequences = ("ACGT" * 3000, "ACGT" * 2999 + "ACGA")
    for accession, sequence in zip(("real_a", "real_b"), sequences, strict=True):
        fasta = tmp_path / f"{accession}.fa"
        fasta.write_text(f">chr1\n{sequence}\n", encoding="utf-8")
        genomes.append(
            OrganelleGenome(
                organelle="plastid",
                sequence=ArtifactRef.from_path(fasta, kind="sequence", format="fasta"),
                metadata=OrganelleMetadata(accession=accession),
            )
        )

    result = recommend_parameters(genomes, threads=1)

    assert result.status == "ok"
    recommendation = result.model_dump(mode="json")["metrics"]["recommendation"]
    assert recommendation["mash_executable"] == str(Path(mash).resolve())
    assert recommendation["repeatmasker_executable"] == str(Path(repeatmasker).resolve())
    assert recommendation["mash_version"]
    assert recommendation["repeatmasker_version"].startswith("RepeatMasker version ")
    assert 50 <= recommendation["identity"] <= 100
    assert recommendation["segment_length"] > 0
    artifact = next(
        item for item in result.artifacts if item.kind == "pangenome_parameter_recommendation"
    )
    evidence = {
        "schema_version": 1,
        "mash": {
            "path": recommendation["mash_executable"],
            "version": recommendation["mash_version"],
        },
        "repeatmasker": {
            "path": recommendation["repeatmasker_executable"],
            "version": recommendation["repeatmasker_version"],
        },
        "recommendation_digest": recommendation["recommendation_digest"],
        "artifact_sha256": artifact.sha256,
        "identity": recommendation["identity"],
        "segment_length": recommendation["segment_length"],
    }
    evidence_path = tmp_path / "pangenome-tuning-evidence.json"
    evidence_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    assert json.loads(evidence_path.read_text(encoding="utf-8")) == evidence
