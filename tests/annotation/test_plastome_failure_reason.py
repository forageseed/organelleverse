"""A failed plastome annotation carries its cause in the message, not only in the details."""

import pytest

from organelleverse.annotation.backends import plastome as backend_module
from organelleverse.annotation.backends.base import AnnotationRequest
from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.errors import OrganelleExecutionError
from organelleverse.core.genome import OrganelleGenome, OrganelleMetadata

REASON = "No plastome features could be transferred from ref.gb: 80 reference gene queries"


class FailingPipeline:
    """Stands in for the pipeline: one sample, rejected with a reason."""

    def __init__(self, **kwargs):
        self.dependency_issues = []
        self.annotation_results = {}
        self.failed_samples = {"Ephedra_x": REASON}

    def verify_dependencies(self):
        return True

    def run_pipeline(self):
        return None


def test_the_failure_reason_is_in_the_message_and_the_details(monkeypatch, tmp_path):
    monkeypatch.setattr(backend_module, "PlastomeAnnotationPipeline", FailingPipeline)
    monkeypatch.setattr(backend_module, "_resolve_search_tools", lambda runner: ({}, {}))
    fasta = tmp_path / "plastome.fasta"
    fasta.write_text(">seq\n" + "ACGT" * 50 + "\n")
    genome = OrganelleGenome(
        organelle="plastid",
        sequence=ArtifactRef.from_path(
            fasta, kind="sequence", format="fasta", media_type="text/x-fasta"
        ),
        metadata=OrganelleMetadata(species="Ephedra x", accession="Ephedra_x"),
    )
    request = AnnotationRequest(
        backend="plastome", workspace=tmp_path / "ws", threads=1, stages=("pcg",)
    )
    with pytest.raises(OrganelleExecutionError) as caught:
        backend_module.PlastomeBackend().run(genome, request, tmp_path / "scratch")
    assert REASON in caught.value.message
    assert caught.value.details["reason"] == REASON
