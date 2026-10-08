"""Explicit adapter/codec decisions for non-canonical restored-capability signatures.

``scripts/capabilities/build_restored_bundles.py`` handles every restored
capability whose Python signature already matches the framework's generic
``canonical_core`` shape (a bare ``OrganelleGenome``/``OrganelleData``/
``OrganelleResult`` first parameter, a matching return annotation) without
consulting this package at all - see the generator's ``_detect_canonical_core``.

For everything else - a plain-JSON or file-path parameter list, a ``custom``
result shape with no self-evident ``ResultCodec`` - the generator refuses to
guess. It looks up an explicit, human-authored override in the matching
``adapters.<domain>`` module instead (``OVERRIDES: dict[str,
NamedParameterOverride]``, keyed by capability id), and fails closed with a
structured diagnostic for any capability that module does not cover.

One module per scientific domain (``docs/operations/restored-capabilities.toml``'s
``domain`` field), per Capability Plan 03 Task 2's design (22 domains total).
This slice ships exactly one: :mod:`.format_conversion`.

**Fixtures are adapter data too.** A domain module may additionally export
``FIXTURES: dict[str, tuple[FixtureCase, ...]]``, keyed by capability id. Each
:class:`FixtureCase` names real input files to write and real parameter
values to call the capability's *actual* implementation with; the generator
(``build_restored_bundles.py``'s ``_capture_fixtures``) is the one place that
actually imports and runs that implementation - a deliberate, explicit step
kept separate from every AST-only decision in this package - and freezes
whatever it *really* returns as that fixture's ``expect`` file. No
``FIXTURES`` entry is ever a value someone reasoned their way to; it is
either the real output of a real run, or absent.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FixtureFile:
    """One real input file a fixture case writes before its capability runs.

    ``relative_path`` is relative to the fixture case's own ``input/``
    directory (e.g. ``"genome.fasta"``) - never an absolute path, and never
    escaping that directory.
    """

    relative_path: str
    content: str


@dataclass(frozen=True)
class FixtureCase:
    """One ``[[fixture]]`` the generator actually runs to capture its expect value.

    ``files`` are written under ``fixtures/<case>/input/`` before the real
    callable is invoked. ``parameters`` supplies exactly one value per
    declared binding parameter used by this case: for a ``path``-codec
    parameter, the value must equal one of ``files``' own
    ``relative_path`` (resolved to a real file for invocation, and rendered
    bundle-root-relative in the emitted ``[[fixture]]`` block, matching
    ``LocalVerificationEnvironment.evaluate_fixture``'s own resolution
    convention); for every other codec, the value is the literal JSON-safe
    value both invoked with and recorded as-is. Every capability this
    module's ``FIXTURES`` covers has ``input_kind == "none"`` (the only
    shape the fixture evaluator supports today), so ``input`` is always
    ``{"kind": "none"}`` and is not a field here at all - the generator
    supplies it.
    """

    case: str
    parameters: dict[str, object]
    files: tuple[FixtureFile, ...] = ()
    equivalence: str = "exact"
    tolerance: float | None = None


__all__ = ["FixtureCase", "FixtureFile"]
