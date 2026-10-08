"""Tests for organelleverse.reporting: compute-only ``build`` + writer ``write``.

Report construction (``ov.report.build``) is compute-only: it captures the
genome, results, title, backend, and images in a typed ``OrganelleReport``
without creating any file. Materialization (``ov.report.write``) is the only
step that renders and publishes the report to a user-selected destination.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

import organelleverse as ov
from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.frozen import FrozenMap
from organelleverse.core.genome import OrganelleGenome, OrganelleMetadata
from organelleverse.core.result import ErrorDetail, OrganelleResult, ResultStatus
from organelleverse.operations.output_boundary import FORBIDDEN_FINAL_WRITE_PARAMS
from organelleverse.reporting import OrganelleReport, build_html_report

_PNG = (
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000d49444154789c63600100000005000156fed98a0000000049454e44ae426082"
)


def _genome(tmp_path: Path) -> OrganelleGenome:
    fa = tmp_path / "m.fa"
    fa.write_text(">m\nACGTACGTACGT\n")
    return OrganelleGenome(
        organelle="mitochondrion",
        sequence=ArtifactRef.from_path(fa, kind="sequence", format="fasta"),
        metadata=OrganelleMetadata(species="Msativa"),
    )


def _png(tmp_path: Path, name: str = "fig.png") -> Path:
    p = tmp_path / name
    p.write_bytes(bytes.fromhex(_PNG))
    return p


def _result(
    operation_id: str,
    *,
    status: ResultStatus = "ok",
    summary_text: str = "",
    metrics: dict[str, object] | None = None,
    artifacts: tuple[ArtifactRef, ...] = (),
    flags: tuple[str, ...] = (),
) -> OrganelleResult:
    errors = (
        (ErrorDetail(code="test.failed", message=summary_text or "failed"),)
        if status == "failed"
        else ()
    )
    return OrganelleResult(
        operation_id=operation_id,
        scope="mitochondrion",
        status=status,
        summary_text=summary_text,
        metrics=FrozenMap.from_json(metrics or {}),
        artifacts=artifacts,
        flags=flags,
        errors=errors,
    )


# -- construction is compute-only ----------------------------------------


def test_build_has_no_destination_parameter() -> None:
    """build is the scientific compute entry: no forbidden final-write param."""
    params = inspect.signature(ov.report.build).parameters
    forbidden = sorted(set(params) & FORBIDDEN_FINAL_WRITE_PARAMS)
    assert not forbidden, f"build still accepts forbidden output parameter(s): {forbidden}"


def test_build_creates_no_user_file(tmp_path: Path) -> None:
    """Construction captures inputs but writes nothing to disk."""
    g = _genome(tmp_path)
    before = {p.name for p in tmp_path.iterdir()}
    report = ov.report.build(g, [], title="No File", backend="jinja")
    after = {p.name for p in tmp_path.iterdir()}
    assert isinstance(report, OrganelleReport)
    assert before == after, "build created a user-visible file"


def test_build_preserves_every_scientific_option(tmp_path: Path) -> None:
    g = _genome(tmp_path)
    r1 = _result("qc.run", summary_text="ok")
    png = _png(tmp_path)
    report = ov.report.build(g, [r1], title="Title", backend="jinja", images=[png])
    assert report.genome is g
    assert report.results == (r1,)
    assert report.title == "Title"
    assert report.backend == "jinja"
    assert report.images == (png,)


def test_write_requires_a_built_report(tmp_path: Path) -> None:
    """``write`` takes a report value, not the old ``(genome, results)`` shape."""
    g = _genome(tmp_path)
    with pytest.raises(TypeError):
        ov.report.write(g, output=tmp_path / "x.html")  # type: ignore[arg-type]


# -- materialization: default backend ------------------------------------


def test_write_default_backend_is_multiqc_or_jinja(tmp_path: Path) -> None:
    """Default backend is MultiQC when installed, else Jinja2."""
    g = _genome(tmp_path)
    r1 = _result(
        "qc.run",
        metrics={"length": 12, "gc": 0.5},
        summary_text="QC passed.",
        flags=("ok",),
    )
    out = tmp_path / "report.html"
    report = ov.report.build(g, [r1], title="Test Report")
    r = ov.report.write(report, output=out)
    assert r.status == "ok"
    assert r.metrics["backend"] in ("multiqc", "jinja2")
    assert r.metrics["results_count"] == 1
    assert out.exists()


def test_generic_write_dispatches_built_report(tmp_path: Path) -> None:
    """The human convenience writer recognizes the typed report value."""
    report = ov.report.build(_genome(tmp_path), [], backend="jinja")
    out = tmp_path / "generic-report.html"

    written = ov.write(report, out)

    assert written.status == "ok"
    assert written.operation_id == "reporting.report"
    assert out.exists()


def test_write_jinja_backend_forced(tmp_path: Path) -> None:
    """backend='jinja' forces the in-tree template even with MultiQC installed."""
    g = _genome(tmp_path)
    r1 = _result("qc.run", summary_text="ok")
    out = tmp_path / "report.html"
    report = ov.report.build(g, [r1], backend="jinja")
    r = ov.report.write(report, output=out)
    assert r.metrics["backend"] == "jinja2"
    html = out.read_text()
    assert "<style>" in html
    assert "<section" in html  # one section per result


def test_write_multiqc_backend_forced(tmp_path: Path) -> None:
    """backend='multiqc' produces a MultiQC report (skips if MultiQC missing)."""
    pytest.importorskip("multiqc")
    g = _genome(tmp_path)
    r1 = _result("qc.run", summary_text="ok", metrics={"gc": 0.45})
    out = tmp_path / "mqc.html"
    report = ov.report.build(g, [r1], backend="multiqc")
    r = ov.report.write(report, output=out)
    assert r.metrics["backend"] == "multiqc"
    assert out.exists()
    assert "multiqc" in out.read_text().lower()


def test_write_multiqc_missing_fails_when_forced(tmp_path: Path, monkeypatch) -> None:
    """backend='multiqc' with MultiQC missing -> failed result, not silent fallback."""
    g = _genome(tmp_path)
    out = tmp_path / "report.html"
    import sys

    monkeypatch.setitem(sys.modules, "multiqc", None)
    report = ov.report.build(g, [], backend="multiqc")
    r = ov.report.write(report, output=out)
    assert r.status == "failed"
    assert r.errors[0].code == "reporting.multiqc_missing"
    assert not out.exists(), "a failed forced-multiqc build left a partial destination"


def test_write_multiqc_embeds_figures(tmp_path: Path) -> None:
    """MultiQC report embeds base64 figures referenced in artifacts."""
    pytest.importorskip("multiqc")
    g = _genome(tmp_path)
    png = _png(tmp_path)
    r1 = _result(
        "viz.plot",
        artifacts=(ArtifactRef.from_path(png, kind="figure", format="png"),),
        summary_text="Plot attached.",
    )
    out = tmp_path / "report.html"
    report = ov.report.build(g, [r1], backend="multiqc")
    ov.report.write(report, output=out)
    assert "data:image/png;base64," in out.read_text()


def test_write_multiple_results(tmp_path: Path) -> None:
    """Each OrganelleResult becomes its own section."""
    g = _genome(tmp_path)
    results = [
        _result("qc.run", summary_text="QC ok."),
        _result(
            "composition.gc_content",
            metrics={"gc_content": 0.45},
            summary_text="GC=0.45.",
        ),
        _result("demo.missing", status="failed", summary_text="Needs annotation."),
    ]
    out = tmp_path / "report.html"
    report = ov.report.build(g, results, backend="jinja")
    ov.report.write(report, output=out)
    html = out.read_text()
    assert html.count("<section") == 3
    # failed section gets the 'failed' class (red border per CSS).
    assert "class='failed'" in html


def test_write_embeds_base64_images(tmp_path: Path) -> None:
    """Figures referenced in result artifacts are embedded as base64 PNG."""
    g = _genome(tmp_path)
    png = _png(tmp_path)
    r1 = _result(
        "viz.plot",
        artifacts=(ArtifactRef.from_path(png, kind="figure", format="png"),),
        summary_text="Plot attached.",
    )
    out = tmp_path / "report.html"
    report = ov.report.build(g, [r1], backend="jinja")
    ov.report.write(report, output=out)
    assert "data:image/png;base64," in out.read_text()


def test_write_extra_images_argument(tmp_path: Path) -> None:
    """The images= argument embeds caller-supplied figures at the top."""
    g = _genome(tmp_path)
    png = _png(tmp_path, "extra.png")
    out = tmp_path / "report.html"
    report = ov.report.build(g, [], images=[png], backend="jinja")
    r = ov.report.write(report, output=out)
    assert r.metrics["figures_embedded"] == 1
    assert "data:image/png;base64," in out.read_text()


def test_write_status_colours(tmp_path: Path) -> None:
    """ok/warning/failed map to green/orange/red section classes (Jinja backend)."""
    g = _genome(tmp_path)
    results = [
        _result("a.b", summary_text="ok"),
        _result("a.c", status="failed", summary_text="bad"),
    ]
    out = tmp_path / "report.html"
    report = ov.report.build(g, results, backend="jinja")
    ov.report.write(report, output=out)
    html = out.read_text()
    assert "class='ok'" in html
    assert "class='failed'" in html


def test_write_no_results(tmp_path: Path) -> None:
    """Empty results list still produces a valid header-only report."""
    g = _genome(tmp_path)
    out = tmp_path / "report.html"
    report = ov.report.build(g, [], backend="jinja")
    r = ov.report.write(report, output=out)
    assert r.status == "ok"
    html = out.read_text()
    assert "Msativa" in html  # header still shows genome metadata
    assert "<section" not in html or html.count("<section") == 0


def test_write_self_contained_no_external_assets(tmp_path: Path) -> None:
    """The Jinja HTML must not reference external CSS/JS (portable single file)."""
    g = _genome(tmp_path)
    out = tmp_path / "report.html"
    report = ov.report.build(g, [], backend="jinja")
    ov.report.write(report, output=out)
    html = out.read_text()
    assert "<link rel=" not in html
    assert "<script src=" not in html


def test_write_is_repeatable_byte_for_byte(tmp_path: Path) -> None:
    """Repeated Jinja writes are byte-equivalent (deterministic, atomic publish)."""
    g = _genome(tmp_path)
    r1 = _result("qc.run", summary_text="ok")
    report = ov.report.build(g, [r1], backend="jinja")
    out = tmp_path / "report.html"
    ov.report.write(report, output=out)
    first = out.read_bytes()
    ov.report.write(report, output=out)
    assert out.read_bytes() == first


def test_short_report_facade_consumes_v1_contracts_compute_then_write(tmp_path: Path) -> None:
    fasta = tmp_path / "v1.fa"
    fasta.write_text(">v1\nACGTACGT\n")
    genome = OrganelleGenome(
        organelle="mitochondrion",
        sequence=ArtifactRef.from_path(fasta, kind="sequence", format="fasta"),
        metadata=OrganelleMetadata(species="V1 species", accession="V1"),
    )
    result = OrganelleResult(
        operation_id="quality_control.assess",
        scope="mitochondrion",
        status="ok",
        summary_text="QC passed.",
        metrics=FrozenMap.from_json({"gc": 0.5}),
        flags=("qc_passed",),
    )
    output = tmp_path / "v1-report.html"

    report = ov.report.build(genome, [result], backend="jinja")

    assert report.genome is genome
    assert report.results == (result,)
    assert not output.exists()

    written = ov.report.write(report, output)

    assert type(written) is OrganelleResult
    assert written.operation_id == "reporting.report"
    assert written.metrics["results_count"] == 1
    assert written.artifacts[0].resolve() == output
    assert output.exists()


def test_write_shows_error_details_and_section_navigation(tmp_path: Path) -> None:
    """Failed sections surface their error code/message; every section is linked."""
    g = _genome(tmp_path)
    results = [
        _result("qc.run", summary_text="QC ok."),
        _result("annotation.trna", status="failed", summary_text="tRNAscan-SE missing."),
    ]
    out = tmp_path / "report.html"
    ov.report.write(ov.report.build(g, results, backend="jinja"), output=out)
    html = out.read_text()
    assert "test.failed" in html and "tRNAscan-SE missing." in html
    assert "href='#r1'" in html and "id='r1'" in html
    assert "href='#r2'" in html and "id='r2'" in html


def test_build_html_report_core_uses_report_layout(tmp_path: Path) -> None:
    html = build_html_report(
        "Core", [{"title": "GC", "text": "GC ok.", "metrics": {"gc": 0.36}, "status": "warning"}]
    )
    assert html.count("<section") == 1
    assert "class='warning'" in html
    assert "0.36" in html


def test_write_jinja_header_embeds_the_desktop_logo(tmp_path: Path) -> None:
    out = tmp_path / "report.html"
    ov.report.write(ov.report.build(_genome(tmp_path), [], backend="jinja"), output=out)
    html = out.read_text()
    assert "class='ov-brand'><img src='data:image/png;base64," in html


def test_write_multiqc_uses_report_layout_logo_and_single_summary(tmp_path: Path) -> None:
    """MultiQC output carries the OV stylesheet and logo; each summary appears once."""
    pytest.importorskip("multiqc")
    g = _genome(tmp_path)
    results = [
        _result("qc.run", summary_text="QC summary sentence.", metrics={"gc": 0.45}),
        _result("annotation.trna", status="failed", summary_text="tRNAscan-SE missing."),
    ]
    out = tmp_path / "mqc.html"
    ov.report.write(ov.report.build(g, results, backend="multiqc"), output=out)
    html = out.read_text()
    assert 'class="custom_logo' in html
    assert ".ov-mqc .metrics" in html  # stylesheet included via custom_css_files
    assert "ov-stats" in html  # overview module
    assert html.count("QC summary sentence.") == 1
    assert "test.failed" in html
