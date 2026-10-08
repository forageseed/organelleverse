from pathlib import Path

import organelleverse as ov
from organelleverse.visualization.plot_object import OrganellePlot


def test_synteny_matrix_prepares_then_writes(tmp_path: Path) -> None:
    blocks = [
        {
            "genome_a": "A",
            "genome_b": "B",
            "a_start": 1,
            "a_end": 50,
            "b_start": 2,
            "b_end": 60,
            "orientation": "direct",
        }
    ]
    destination = tmp_path / "synteny.svg"
    plot = ov.visualization.synteny_matrix(blocks)
    assert isinstance(plot, OrganellePlot)
    assert plot.operation_id == "visualization.plot_synteny_matrix"
    assert plot.metrics["blocks"] == 1
    assert not destination.exists()

    published = ov.write(plot, destination)
    assert destination.is_file()
    assert published.artifacts[0].resolve() == destination
