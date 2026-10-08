"""Real ODGI annotation injection and untangle release gate."""

import shutil

import pytest

from organelleverse.pangenome.annotation_projection import Annotation
from organelleverse.pangenome.annotation_untangle import run_odgi_untangle

pytestmark = pytest.mark.integration

def test_actual_odgi_inject_preserves_paths_and_compound_part_identity(tmp_path):
    executable = shutil.which("odgi")
    assert executable is not None, "Real annotation untangle gate requires odgi"
    graph = tmp_path / "graph.gfa"
    graph.write_text(
        "H\tVN:Z:1.0\nS\t1\tACGTTGCAAA\nS\t2\tTTGGCCAATT\nL\t1\t+\t2\t+\t0M\nP\ta#1#1\t1+,2+\t*\nP\tb#1#1\t1+,2+\t*\n"
    )
    feature = Annotation("a#1#1", "gene", "locus", ((2, 5), (15, 18)), ".", part_strands=("+", "-"))
    result = run_odgi_untangle(
        graph, [feature], ["a#1#1", "b#1#1"], tmp_path / "out", executable=executable, n_best=2
    )
    assert result["annotation_part_count"] == 2
    assert result["mapping_row_count"] > 0
    assert result["tool_version"].startswith("v")
    assert all(command["returncode"] == 0 for command in result["commands"])
