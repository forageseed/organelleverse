"""Packaged plastome reference data."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from ..data import plastome_data_dir, plastome_reference_dir, plastome_reference_manifest


def default_plastome_reference_dir() -> Path:
    return plastome_reference_dir()


def default_plastome_preprocessed_dir() -> Path:
    return plastome_data_dir() / "preprocessed"


def default_plastome_reference_manifest() -> Path:
    return plastome_reference_manifest()


@lru_cache
def load_product_map(reference_dir: str | Path | None = None) -> dict[str, str]:
    data_dir = (
        Path(reference_dir) if reference_dir is not None else default_plastome_reference_dir()
    )
    path = data_dir / "product.txt"
    if not path.exists():
        path = default_plastome_reference_dir() / "product.txt"
    products: dict[str, str] = {}
    if not path.exists():
        return products
    for line in path.read_text().splitlines():
        if not line.strip() or "\t" not in line:
            continue
        gene, product = line.split("\t", 1)
        products[gene.strip()] = product.strip()
    return products


def prepare_plastome_reference_cache(
    reference_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, int]:
    data_dir = (
        Path(reference_dir) if reference_dir is not None else default_plastome_reference_dir()
    )
    reference_files = sorted(data_dir.glob("*.gb")) + sorted(data_dir.glob("*.gbk"))
    if not reference_files:
        raise FileNotFoundError(f"No plastome GenBank references found in {data_dir}")
    if output_dir is not None:
        from .references import build_preprocessed_reference_cache

        return build_preprocessed_reference_cache(data_dir, Path(output_dir))
    from .references import build_plastome_queries, load_references

    references = load_references(reference_files)
    return {reference.path.stem: len(build_plastome_queries(reference)) for reference in references}
