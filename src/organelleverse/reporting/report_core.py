"""Typed cores for reporting (OmicVerse-style data-in/data-out)."""

from __future__ import annotations

from pathlib import Path

from .report import _render_page, _SectionView, _split_metrics


def build_html_report(
    title: str,
    sections: list[dict],
    *,
    images: list[str | Path] | None = None,
) -> str:
    """Build a styled, self-contained HTML report string.

    Each section is ``{"title", "text", "status"?, "metrics"?, "flags"?,
    "provenance"?, "figures"?}``. Returns the full HTML document.
    """
    views = []
    for s in sections:
        values, groups = _split_metrics(s.get("metrics") or {})
        provenance = s.get("provenance")
        views.append(
            _SectionView(
                group="",
                title=s.get("title", ""),
                status=s.get("status", "ok"),
                summary=s.get("text", ""),
                flags=tuple(s.get("flags") or ()),
                metrics=values,
                metric_groups=groups,
                figures=tuple(Path(fig) for fig in s.get("figures") or ()),
                provenance=(("method", provenance.get("method", "?")),) if provenance else (),
            )
        )
    return _render_page(title, views, images=[Path(img) for img in images or []])
