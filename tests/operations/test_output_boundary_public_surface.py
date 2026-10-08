"""Public output-boundary ratchet over the canonical ``ov.*`` facades. The legacy scientific entry points have been migrated to destination-free signatures; any new output-path parameter now fails this inventory."""

from __future__ import annotations

import organelleverse as ov
from organelleverse.operations.output_boundary import (
    FORBIDDEN_FINAL_WRITE_PARAMS,
    public_surface_final_write_violations,
)

FROZEN_PUBLIC_SURFACE_VIOLATIONS: dict[str, tuple[str, ...]] = {
}


def test_out_dir_is_recognized_as_a_final_write_destination() -> None:
    """Keep out_dir forbidden even after migrating population entry points."""
    assert "out_dir" in FORBIDDEN_FINAL_WRITE_PARAMS
    violations = public_surface_final_write_violations()
    assert "population.prepare_gemma_input" not in violations


def test_forbidden_final_write_params_are_frozen() -> None:
    """The forbidden set is the exact spelling list from the design contract."""
    assert (
        frozenset(
            {
                "output",
                "output_dir",
                "out_dir",
                "destination",
                "workspace",
                "work_dir",
                "output_path",
                "output_prefix",
                "save_path",
                "export_path",
            }
        )
        == FORBIDDEN_FINAL_WRITE_PARAMS
    )


def test_public_surface_matches_frozen_inventory() -> None:
    """No unreviewed additions; later tasks shrink both sides in lockstep."""
    actual = public_surface_final_write_violations()
    extra = {key: actual[key] for key in actual.keys() - FROZEN_PUBLIC_SURFACE_VIOLATIONS.keys()}
    assert not extra, f"unreviewed new public-surface violations: {extra}"
    assert actual == FROZEN_PUBLIC_SURFACE_VIOLATIONS, (
        "public surface inventory drifted from the frozen Task-1 snapshot; "
        "if you shrank the set, remove the matching entries here too"
    )


def test_writers_are_not_reported_as_scientific_violations() -> None:
    """Explicit ``write`` materializers may accept a destination."""
    violations = public_surface_final_write_violations()
    assert "annotation.write" not in violations
    assert "qc.write" not in violations
    assert "assembly.write" not in violations


def test_reader_fetch_primitives_are_not_reported() -> None:
    """``fetch.merge_with_baseline`` has ``output_path`` but is a fetch primitive.

    This is a classification check, not an emptiness check: it must hold
    regardless of how many *scientific* violations the frozen inventory
    currently admits, so it asserts against
    ``FROZEN_PUBLIC_SURFACE_VIOLATIONS`` rather than against ``{}``.
    ``test_public_surface_matches_frozen_inventory`` already covers the full
    dict; this test's own job is narrower and must not also demand the
    inventory be empty.
    """
    violations = public_surface_final_write_violations()
    assert "fetch.merge_with_baseline" not in violations
    assert "io.read_fasta" not in violations
    assert "coevolution.run_coevolution" not in violations
    assert "coevolution.run_orthofinder" not in violations
    assert violations == FROZEN_PUBLIC_SURFACE_VIOLATIONS


def test_scanner_inspects_every_canonical_facade() -> None:
    """The scanner follows the canonical facade set, not a stale hardcoded list.

    Compared dynamically against ``organelleverse._PUBLIC_MODULES`` (31
    entries today, and growing) rather than a literal enumeration, so this
    test does not need editing every time a suite is admitted.
    """
    from organelleverse.operations.output_boundary import (
        _public_facades,  # pyright: ignore[reportPrivateUsage]
    )

    public_modules = set(getattr(ov, "_PUBLIC_MODULES", ()))
    assert public_modules, "organelleverse._PUBLIC_MODULES must be non-empty"
    assert set(_public_facades()) == public_modules

    # Short aliases (ov.viz, ov.erc, ...) are a routing layer over
    # _PUBLIC_MODULES, not additional facades: none of their names may appear
    # as a scanned facade, and resolving every alias must not add, remove, or
    # rename a single violation.
    short_aliases = set(getattr(ov, "_SHORT_MODULE_ALIASES", {}))
    assert short_aliases.isdisjoint(set(_public_facades()))

    before = public_surface_final_write_violations()
    for short_name in short_aliases:
        getattr(ov, short_name)
    after = public_surface_final_write_violations()
    assert after == before == FROZEN_PUBLIC_SURFACE_VIOLATIONS

    # visualization.plot_synteny_matrix and visualization.synteny_matrix are
    # two public names for the exact same function: one real violation, not
    # two independent ones.
    import organelleverse.visualization as visualization

    assert visualization.plot_synteny_matrix is visualization.synteny_matrix

    # localization.predict and localization.localize are likewise the exact
    # same function object (the archive's own `localize = predict` alias).
    import organelleverse.localization as localization

    assert localization.predict is localization.localize
