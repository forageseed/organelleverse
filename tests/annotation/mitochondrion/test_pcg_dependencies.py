from types import SimpleNamespace

import pytest

from organelleverse import _losat
from organelleverse.annotation.mitochondrion import pcg
from organelleverse.annotation.mitochondrion.models.genome import GenomeSequence
from organelleverse.core.errors import OrganelleDependencyError


@pytest.mark.parametrize("stage", ["fallback", "missing_core", "boundary"])
@pytest.mark.parametrize("managed", [False, True])
def test_missing_search_tools_never_silently_skip_pcg(monkeypatch, stage, managed):
    monkeypatch.setenv("ORG_VERSE_LOSAT_BIN", "ncbi")
    monkeypatch.setattr(_losat.shutil, "which", lambda name: None)
    genome = GenomeSequence(seqid="genome", sequence="A" * 1000)
    db = SimpleNamespace()
    config = pcg.PCGConfig()
    kwargs = {"tool_paths": {}, "command_runner": SimpleNamespace()} if managed else {}
    with pytest.raises(OrganelleDependencyError, match="Install LOSAT or NCBI BLAST"):
        if stage == "fallback":
            pcg._blastn_fallback(genome, db, set(), config, **kwargs)
        elif stage == "missing_core":
            pcg._search_missing_core_genes_blast(genome, db, ["atp1"], config, **kwargs)
        else:
            pcg._refine_boundaries_reference([], genome, db, config, **kwargs)
