"""Public contract: compute -> prepare -> ``ov.write`` for genome identity."""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import pytest

import organelleverse as ov
from organelleverse.capabilities.adapters import comparative as comparative_adapter
from organelleverse.visualization.plot_object import OrganellePlot


def _fixture_fasta(tmp_path: Path) -> tuple[Path, Path, Path]:
    reference = tmp_path / "reference.fasta"
    query_a = tmp_path / "query_a.fasta"
    query_b = tmp_path / "query_b.fasta"
    reference.write_text(comparative_adapter._IDENTITY_REFERENCE_FASTA.content)
    query_a.write_text(comparative_adapter._IDENTITY_QUERY_A_FASTA.content)
    query_b.write_text(comparative_adapter._IDENTITY_QUERY_B_FASTA.content)
    return reference, query_a, query_b


def test_genome_identity_compute_then_plot_then_write(tmp_path: Path) -> None:
    reference, query_a, query_b = _fixture_fasta(tmp_path)
    result = ov.comparative.compute_genome_identity(reference, [query_a, query_b])
    assert result["samples"] == ["query_a", "query_b"]
    assert result["reference_length"] == 2400

    plot = ov.visualization.genome_identity(
        result["windows"],
        result["features"],
        reference_length=result["reference_length"],
        reference_name=result["reference_name"],
    )
    assert isinstance(plot, OrganellePlot)
    assert plot.operation_id == "visualization.plot_genome_identity"
    assert not (tmp_path / "identity.svg").exists()

    svg_destination = tmp_path / "identity.svg"
    published = ov.write(plot, svg_destination)
    assert svg_destination.is_file()
    assert published.artifacts[0].resolve() == svg_destination

    png_destination = tmp_path / "identity.png"
    plot.save(png_destination)
    assert png_destination.is_file()


def test_genome_identity_plot_zooms_to_region(tmp_path: Path) -> None:
    reference, query_a, _ = _fixture_fasta(tmp_path)
    result = ov.comparative.compute_genome_identity(reference, [query_a])
    plot = ov.visualization.plot_genome_identity(
        result["windows"],
        result["features"],
        reference_length=result["reference_length"],
        region=(500, 1500),
    )
    assert plot.metrics["region"] == (500, 1500)
    destination = tmp_path / "region.png"
    plot.save(destination)
    assert destination.is_file()


def test_genome_identity_plot_validates_thresholds(tmp_path: Path) -> None:
    reference, query_a, _ = _fixture_fasta(tmp_path)
    result = ov.comparative.compute_genome_identity(reference, [query_a])
    with pytest.raises(ValueError, match="min_identity"):
        ov.visualization.plot_genome_identity(
            result["windows"],
            min_identity=100.0,
        )
    with pytest.raises(ValueError, match="conserved_threshold"):
        ov.visualization.plot_genome_identity(
            result["windows"],
            min_identity=50.0,
            conserved_threshold=40.0,
        )


def test_genome_identity_plot_requires_windows(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one window"):
        ov.visualization.plot_genome_identity([])


@pytest.mark.parametrize("region", [(-1, 50), (100, 100), (0, 201), (150, 180)])
def test_invalid_or_empty_region_is_rejected_before_render(region):
    with pytest.raises(ValueError, match="region"):
        ov.visualization.genome_identity(
            [{"sample": "s", "start": 0, "end": 100, "identity": 90}],
            reference_length=200,
            region=region,
        )


def test_sliding_windows_have_monotone_curve_and_exact_feature_boundaries(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import to_rgba

    from organelleverse.visualization import suite_plots

    captured = []
    monkeypatch.setattr(suite_plots, "_save", lambda fig, out: captured.append(fig) or out)
    windows = [
        dict(sample="s", start=start, end=start + 100, identity=90) for start in (100, 50, 0)
    ]
    features = [dict(key="gene1", name="gene1", start=0, end=75, strand=-1, category="cds")]
    suite_plots._render_genome_identity(
        windows,
        features,
        tmp_path / "x.svg",
        reference_length=200,
        region=None,
        min_identity=50,
        conserved_threshold=70,
        reference_name=None,
        title=None,
    )
    fig = captured[0]
    ax = fig.axes[1]
    xs = ax.lines[0].get_xdata()
    assert all(a <= b for a, b in pairwise(xs))
    fills = next(c for c in ax.collections if isinstance(c, PolyCollection))
    for path, color in zip(fills.get_paths(), fills.get_facecolors(), strict=True):
        low, high = path.vertices[:, 0].min(), path.vertices[:, 0].max()
        category = "cds" if high <= 75 else "noncoding"
        assert low >= 75 or high <= 75
        assert tuple(color) == to_rgba(suite_plots._IDENTITY_CATEGORY_COLORS[category], 0.85)
    plt.close(fig)
