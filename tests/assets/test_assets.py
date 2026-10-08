"""Knowledge asset layer: identity, content hashing, resolution, and drift."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import cast

import pytest

from organelleverse import assets
from organelleverse.assets import AssetLocator, AssetManifest, compute_content
from organelleverse.assets.manifest import manifest_path_for
from organelleverse.assets.resolvers import PackagedAssetResolver
from organelleverse.core.errors import OrganelleDependencyError, OrganelleInputError
from organelleverse.core.frozen import thaw_json

MANIFEST = """\
schema = "organelleverse.asset.v1"

[asset]
namespace = "demo"
name = "reference_set"
version = "1.0"
summary = "A demonstration asset."

[content]
files = {files}
bytes = {bytes}
merkle = "{merkle}"

[provenance]
source = "https://example.invalid/reference"
license = "CC0-1.0"
rebuild = ""
derived_from = []
notes = ""
"""


def _build_asset(root: Path, payload: str = "reference bases\n") -> Path:
    """Create a minimal on-disk asset whose manifest matches its content."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "reference.fasta").write_text(payload, encoding="utf-8")
    content = compute_content(root)
    (root / "asset.toml").write_text(
        MANIFEST.format(files=content.files, bytes=content.bytes, merkle=content.merkle),
        encoding="utf-8",
    )
    return root


class _StubResolver:
    """A resolver over one directory, for tests that need a second source."""

    def __init__(self, index: dict[str, Path], source: str = "stub") -> None:
        self._index = index
        self._source = source

    @property
    def source(self) -> str:
        return self._source

    def index(self) -> dict[str, Path]:
        return dict(self._index)


# --- identity -------------------------------------------------------------


def test_locator_roundtrips_through_parse_and_str() -> None:
    locator = AssetLocator.parse("ov-asset:mito/blast_refs@1.0")
    assert (locator.namespace, locator.name, locator.version) == ("mito", "blast_refs", "1.0")
    assert str(locator) == "ov-asset:mito/blast_refs@1.0"


@pytest.mark.parametrize(
    "text",
    [
        "blast_refs",
        "ov-asset:mito/blast_refs",
        "ov-asset:mito@1.0",
        "ov-asset:Mito/blast_refs@1.0",
        "ov-asset:mito/blast_refs@",
        "asset:mito/blast_refs@1.0",
    ],
)
def test_malformed_locators_are_rejected(text: str) -> None:
    with pytest.raises(OrganelleInputError) as error:
        AssetLocator.parse(text)
    assert error.value.code == "asset.locator_invalid"


def test_locator_is_immutable() -> None:
    locator = AssetLocator.parse("ov-asset:mito/blast_refs@1.0")
    with pytest.raises(ValueError):
        locator.version = "2.0"  # type: ignore[misc]


# --- content hashing ------------------------------------------------------


def test_content_hash_is_stable_across_recomputation(tmp_path: Path) -> None:
    root = _build_asset(tmp_path / "asset")
    assert compute_content(root) == compute_content(root)


def test_content_hash_excludes_the_manifest_itself(tmp_path: Path) -> None:
    """Declaring the hash must not change the hash it declares."""
    root = tmp_path / "asset"
    root.mkdir()
    (root / "reference.fasta").write_text("reference bases\n", encoding="utf-8")
    before = compute_content(root)
    (root / "asset.toml").write_text('schema = "organelleverse.asset.v1"\n', encoding="utf-8")
    assert compute_content(root) == before


def test_content_hash_changes_when_bytes_change(tmp_path: Path) -> None:
    root = _build_asset(tmp_path / "asset")
    original = compute_content(root)
    (root / "reference.fasta").write_text("different bases\n", encoding="utf-8")
    assert compute_content(root).merkle != original.merkle


def test_content_hash_changes_when_a_file_is_renamed(tmp_path: Path) -> None:
    """Path is part of identity, so a rename is a different asset."""
    root = _build_asset(tmp_path / "asset")
    original = compute_content(root)
    (root / "reference.fasta").rename(root / "renamed.fasta")
    assert compute_content(root).merkle != original.merkle


def test_compute_content_rejects_a_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(OrganelleInputError) as error:
        compute_content(tmp_path / "absent")
    assert error.value.code == "asset.root_missing"


# --- manifests ------------------------------------------------------------


def test_manifest_reads_declared_identity_and_provenance(tmp_path: Path) -> None:
    root = _build_asset(tmp_path / "asset")
    manifest = AssetManifest.from_directory(root)
    assert str(manifest.locator) == "ov-asset:demo/reference_set@1.0"
    assert manifest.provenance.license == "CC0-1.0"
    assert manifest.content == compute_content(root)


def test_manifest_missing_is_reported(tmp_path: Path) -> None:
    root = tmp_path / "asset"
    root.mkdir()
    with pytest.raises(OrganelleInputError) as error:
        AssetManifest.from_directory(root)
    assert error.value.code == "asset.manifest_missing"


def test_manifest_without_asset_table_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "asset"
    root.mkdir()
    (root / "asset.toml").write_text('schema = "organelleverse.asset.v1"\n', encoding="utf-8")
    with pytest.raises(OrganelleInputError) as error:
        AssetManifest.from_directory(root)
    assert error.value.code == "asset.manifest_invalid"


def test_manifest_with_unsupported_schema_is_rejected(tmp_path: Path) -> None:
    root = _build_asset(tmp_path / "asset")
    text = (root / "asset.toml").read_text(encoding="utf-8")
    (root / "asset.toml").write_text(
        text.replace("organelleverse.asset.v1", "organelleverse.asset.v99"),
        encoding="utf-8",
    )
    with pytest.raises(OrganelleInputError) as error:
        AssetManifest.from_directory(root)
    assert error.value.code == "asset.manifest_unsupported_schema"


# --- resolution and drift -------------------------------------------------


@pytest.fixture
def registered_asset(tmp_path: Path) -> Iterator[Path]:
    """Register a throwaway asset for the duration of one test."""
    root = _build_asset(tmp_path / "asset")
    resolver = _StubResolver({"ov-asset:demo/reference_set@1.0": root})
    assets.register_resolver(resolver)
    try:
        yield root
    finally:
        assets.unregister_resolver(resolver)


def test_verify_accepts_content_matching_its_declaration(registered_asset: Path) -> None:
    assert assets.verify("ov-asset:demo/reference_set@1.0").content.files == 1


def test_verify_rejects_content_that_drifted_from_its_manifest(registered_asset: Path) -> None:
    """A swapped reference library must invalidate the declaration, not pass."""
    (registered_asset / "injected.fasta").write_text("swapped in\n", encoding="utf-8")
    with pytest.raises(OrganelleInputError) as error:
        assets.verify("ov-asset:demo/reference_set@1.0")
    assert error.value.code == "asset.content_mismatch"
    details = cast("Mapping[str, Mapping[str, object]]", thaw_json(error.value.details))
    assert details["declared"]["files"] == 1
    assert details["observed"]["files"] == 2


def test_resolvers_disagreeing_on_one_locator_is_a_conflict(
    registered_asset: Path, tmp_path: Path
) -> None:
    """Two sources claiming one locator must fail, not pick a winner by order."""
    other = _build_asset(tmp_path / "other", payload="different bases\n")
    conflicting = _StubResolver({"ov-asset:demo/reference_set@1.0": other}, source="other")
    assets.register_resolver(conflicting)
    try:
        with pytest.raises(OrganelleDependencyError) as error:
            assets.locate("ov-asset:demo/reference_set@1.0")
        assert error.value.code == "asset.conflict"
    finally:
        assets.unregister_resolver(conflicting)


def test_unresolvable_locator_is_a_typed_dependency_error() -> None:
    with pytest.raises(OrganelleDependencyError) as error:
        assets.locate("ov-asset:absent/nothing@1.0")
    assert error.value.code == "asset.unresolvable"


def test_packaged_resolver_indexes_shipped_assets() -> None:
    """Assets shipped in the tree are discovered without registration code."""
    index = PackagedAssetResolver().index()
    assert "ov-asset:trna/cm_models@1.0" in index
    assert index["ov-asset:trna/cm_models@1.0"].is_dir()


def test_shipped_assets_match_their_declared_content() -> None:
    """Every packaged asset's bytes still match what its manifest declares."""
    for manifest in assets.list_assets():
        assert assets.content_hash(manifest.locator) == manifest.content.merkle


# --- single-file assets and nesting -----------------------------------------


def _build_file_asset(path: Path, payload: str = "model weights\n") -> Path:
    """Create a single-file asset with a sibling manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    content = compute_content(path)
    manifest_path_for(path).write_text(
        MANIFEST.format(files=content.files, bytes=content.bytes, merkle=content.merkle),
        encoding="utf-8",
    )
    return path


def test_a_single_file_can_be_an_asset(tmp_path: Path) -> None:
    """A model checkpoint is one file; it cannot contain its own manifest."""
    weights = _build_file_asset(tmp_path / "model.hdf5")
    assert manifest_path_for(weights).name == "model.hdf5.asset.toml"
    manifest = AssetManifest.from_directory(weights)
    assert manifest.content.files == 1
    assert manifest.content == compute_content(weights)


def test_single_file_asset_hash_tracks_its_bytes(tmp_path: Path) -> None:
    weights = _build_file_asset(tmp_path / "model.hdf5")
    before = compute_content(weights)
    weights.write_text("different weights\n", encoding="utf-8")
    assert compute_content(weights).merkle != before.merkle


def test_nested_asset_is_excluded_from_its_container(tmp_path: Path) -> None:
    """Versioning a model directory must not re-version the set containing it."""
    container = tmp_path / "reference"
    nested = container / "models"
    _build_asset(nested, payload="model bytes\n")
    (container / "reference.fasta").write_text(">a\nACGT\n", encoding="utf-8")

    content = compute_content(container)
    assert content.files == 1, "container must see only its own file"

    (nested / "reference.fasta").write_text("changed model\n", encoding="utf-8")
    assert compute_content(container) == content, "nested change must not move the container"


def test_a_nested_asset_still_sees_all_of_its_own_files(tmp_path: Path) -> None:
    """The container boundary must not swallow the nested asset's own content."""
    nested = tmp_path / "reference" / "models"
    _build_asset(nested, payload="model bytes\n")
    (tmp_path / "reference" / "asset.toml").write_text(
        'schema = "organelleverse.asset.v1"\n', encoding="utf-8"
    )
    assert compute_content(nested).files == 1


def test_code_is_never_part_of_an_asset(tmp_path: Path) -> None:
    """A data directory that is also an importable package ships only its data."""
    root = tmp_path / "asset"
    _build_asset(root)
    before = compute_content(root)
    (root / "__init__.py").write_text("# package marker\n", encoding="utf-8")
    (root / "download.py").write_text("def fetch(): ...\n", encoding="utf-8")
    assert compute_content(root) == before


def test_shipped_model_weights_are_content_addressed() -> None:
    """The vendored ML weights carry identity, so a result can name them."""
    for locator in (
        "ov-asset:rna_editing/deepredmt@210520",
        "ov-asset:rna_editing/plantc2u@flank90",
    ):
        manifest = assets.verify(locator)
        assert manifest.content.bytes > 0
        assert manifest.provenance.derived_from


def test_a_single_file_asset_is_excluded_from_its_container(tmp_path: Path) -> None:
    """A sidecar manifest makes its file separately versioned, so the container stops there."""
    container = tmp_path / "reference"
    container.mkdir()
    (container / "ref.fasta").write_text(">a\nACGT\n", encoding="utf-8")
    weights = _build_file_asset(container / "weights.bin", payload="v1")

    content = compute_content(container)
    assert content.files == 1, "container must not hash a nested single-file asset"

    weights.write_text("v2", encoding="utf-8")
    assert compute_content(container) == content
    assert compute_content(weights).files == 1, "the nested asset still sees its own bytes"
