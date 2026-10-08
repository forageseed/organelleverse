"""``ov.capabilities`` must be reachable, and reaching for it must stay lazy.

Both halves matter and they pull in opposite directions, which is why they are
asserted together. Task 3 exists so an Agent can drive the authoring loop through
the public surface, and an Agent that types ``ov.capabilities`` and gets
``AttributeError`` has hit a wall on its first move. But the capability package
pulls in the bundle models, discovery, admission, trust and the worker, and none
of that belongs in the cost of ``import organelleverse``.

``__getattr__`` resolves lazily, so both hold at once: the name is available, and
nothing is imported until somebody actually asks for it. The laziness assertion
is here so that stays true — it is the property that made the reachability change
safe in the first place, and it would regress silently.
"""

from __future__ import annotations

import subprocess
import sys

from tests._paths import child_env


def test_bare_import_does_not_load_the_capability_package() -> None:
    probe = "import sys, organelleverse; print('organelleverse.capabilities' in sys.modules)"
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
        env=child_env(),
    )
    assert completed.stdout.strip() == "False", (
        "importing organelleverse eagerly loaded the capability package; the "
        "attribute must resolve through the lazy __getattr__, never at import time"
    )


def test_capabilities_is_reachable_as_an_attribute() -> None:
    import organelleverse as ov

    assert ov.capabilities.discover_capabilities is not None
    assert "capabilities" in dir(ov)


def test_the_attribute_and_the_explicit_import_are_the_same_module() -> None:
    import organelleverse as ov
    import organelleverse.capabilities as explicit

    assert ov.capabilities is explicit
