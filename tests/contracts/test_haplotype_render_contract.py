from pathlib import Path

import organelleverse as ov
from organelleverse.visualization.plot_object import OrganellePlot


def test_haplotype_renderer_prepares_then_writes(tmp_path: Path) -> None:
    destination = tmp_path / "network.svg"
    plot = ov.phylogeny.render_network(
        [{"id": "h1"}], {"h1": ["sample1"]}, {"h1": 1}, []
    )
    assert isinstance(plot, OrganellePlot)
    assert plot.operation_id == "phylogeny.render_network"
    assert not destination.exists()

    published = ov.write(plot, destination)
    assert destination.is_file()
    assert published.artifacts[0].resolve() == destination
