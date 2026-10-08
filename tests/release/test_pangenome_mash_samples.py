import json
import random
import shutil
from pathlib import Path

import pytest

from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.genome import OrganelleGenome, OrganelleMetadata
from organelleverse.pangenome.tuning_service import recommend_parameters

pytestmark = pytest.mark.integration


def test_actual_mash_compares_samples_and_preserves_multiple_molecules(tmp_path: Path, monkeypatch):
    assert shutil.which("mash"), "Actual Mash is required for this release gate"
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    rng = random.Random(20260908)
    first = "".join(rng.choice("ACGT") for _ in range(15000))
    second = list(first)
    for index in range(0, len(second), 73):
        second[index] = next(base for base in "ACGT" if base != second[index])
    extra = "".join(rng.choice("ACGT") for _ in range(4000))
    genomes = []
    for sample, sequence in [("a", first), ("b", "".join(second))]:
        fasta = tmp_path / f"{sample}.fa"
        fasta.write_text(f">main\n{sequence}\n>extra\n{extra}\n")
        genomes.append(
            OrganelleGenome(
                organelle="mitochondrion",
                sequence=ArtifactRef.from_path(fasta, kind="sequence", format="fasta"),
                metadata=OrganelleMetadata(accession=sample),
            )
        )
    result = recommend_parameters(genomes, threads=1, run_repeatmasker=False)
    evidence = result.model_dump(mode="json")["metrics"]["recommendation"]
    assert evidence["policy_version"] == "sample-mash-mitochondrial.v3"
    assert 0.002 < evidence["max_mash_distance"] < 0.05
    table = next(a for a in result.artifacts if a.kind == "pangenome_sample_distances").resolve()
    pairs = [line.split("\t")[:2] for line in table.read_text().splitlines()]
    assert len(pairs) == 4
    assert len({a for pair in pairs for a in pair}) == 2
    cached = recommend_parameters(genomes, threads=1, run_repeatmasker=False)
    assert cached.object_id == result.object_id
    (tmp_path / "sample-distance-evidence.json").write_text(
        json.dumps({"recommendation": evidence, "pairs": pairs}, indent=2)
    )
