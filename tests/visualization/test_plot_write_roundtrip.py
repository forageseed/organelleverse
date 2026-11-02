"""A plot survives validation of its own return value and can still be written.

Regression (2026-10-25): with the shared Python validation boundary, a public
plot call's return value is checked with ``TypeAdapter(OrganellePlot)``.
``OrganelleResult`` revalidates instances, so the check rebuilt the plot
without its private renderer and ``ov.write(plot_genome_map(...), "map.svg")``
raised ``visualization.unrenderable_result``.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import TypeAdapter

import organelleverse as ov
from organelleverse.visualization.plot_object import OrganellePlot

ARABIDOPSIS = Path(__file__).resolve().parents[1] / "data" / "Arabidopsis_thaliana.gb"


def test_validating_a_plot_returns_the_same_plot_with_its_renderer() -> None:
    plot = ov.visualization.plot_genome_map(genbank_path=ARABIDOPSIS)
    assert plot.renderer is not None

    checked = TypeAdapter(OrganellePlot).validate_python(plot, strict=True)
    assert checked is plot
    assert checked.renderer is not None


def test_plot_genome_map_is_written_as_svg_by_ov_write(tmp_path: Path) -> None:
    plot = ov.visualization.plot_genome_map(
        genbank_path=ARABIDOPSIS, title="Arabidopsis thaliana"
    )
    written = ov.write(plot, tmp_path / "map.svg")

    assert written.status == "ok"
    svg = tmp_path / "map.svg"
    assert svg.is_file()
    assert "<svg" in svg.read_text(encoding="utf-8")[:500]
    assert written.artifacts
