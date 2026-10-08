"""T-C2: the optional qupath measurement backend — fail-closed bridge rules."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from organelleverse.core.errors import OrganelleDependencyError, OrganelleInputError
from organelleverse.morphology import qupath_bridge
from organelleverse.morphology.measure import measure


def test_native_is_the_default_and_auto_never_picks_qupath() -> None:
    lab = np.zeros((12, 12), dtype=np.int32)
    lab[2:6, 2:6] = 1
    assert measure(lab, min_area=1)["n_total"] == 1
    assert measure(lab, min_area=1, backend="native")["n_total"] == 1
    # auto routes to native even when no bridge exists
    assert measure(lab, min_area=1, backend="auto")["n_total"] == 1


def test_unknown_backend_fails_closed() -> None:
    with pytest.raises(OrganelleInputError) as raised:
        measure(np.zeros((4, 4), dtype=np.int32), backend="imagej")
    assert raised.value.code == "morphology.unknown_measure_backend"


def test_qupath_without_bridge_fails_closed_and_never_falls_back(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # No paquo in this environment: the explicit request must fail, and the
    # native path must NOT be silently consulted.
    monkeypatch.setitem(sys.modules, "paquo", None)
    import numpy as np
    from PIL import Image

    label_path = tmp_path / "label.png"
    Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(label_path)
    with pytest.raises(OrganelleDependencyError) as raised:
        measure(label_path, backend="qupath")
    assert raised.value.code == "qupath.bridge_unavailable"
    assert "paquo" in raised.value.message


def test_qupath_rejects_in_memory_arrays() -> None:
    with pytest.raises(OrganelleInputError) as raised:
        measure(np.zeros((4, 4), dtype=np.int32), backend="qupath")
    assert raised.value.code == "qupath.requires_label_map_path"


def test_out_of_range_qupath_version_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """A QuPath outside the declared range fails — never 'try anyway'."""
    fake_paquo = types.ModuleType("paquo")
    fake_java = types.ModuleType("paquo.java")
    fake_java.qupath_version = "0.7.0"  # outside >=0.5,<0.6
    fake_paquo.java = fake_java
    monkeypatch.setitem(sys.modules, "paquo", fake_paquo)
    monkeypatch.setitem(sys.modules, "paquo.java", fake_java)

    status = qupath_bridge.check_qupath_bridge()
    assert status["available"] is False
    assert "outside the declared range" in status["reason"]


def test_in_range_qupath_version_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_paquo = types.ModuleType("paquo")
    fake_java = types.ModuleType("paquo.java")
    fake_java.qupath_version = "0.5.1"
    fake_paquo.java = fake_java
    monkeypatch.setitem(sys.modules, "paquo", fake_paquo)
    monkeypatch.setitem(sys.modules, "paquo.java", fake_java)

    status = qupath_bridge.check_qupath_bridge()
    assert status["available"] is True
    assert status["qupath_version"] == "0.5.1"


def test_jvm_start_failure_is_named(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_paquo = types.ModuleType("paquo")
    fake_java = types.ModuleType("paquo.java")

    class _Broken:
        def __getattr__(self, name):
            raise RuntimeError("JVM failed to start: no Java runtime")

    fake_java.__getattr__ = lambda name: (_ for _ in ()).throw(RuntimeError("JVM failed to start"))
    fake_paquo.java = property(lambda self: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setitem(sys.modules, "paquo", fake_paquo)
    monkeypatch.setitem(sys.modules, "paquo.java", None)  # import of paquo.java fails

    status = qupath_bridge.check_qupath_bridge()
    assert status["available"] is False
    assert status["reason"]


def test_qupath_rows_match_the_native_schema() -> None:
    """Schema parity is static and pinned: the bridge row template must carry
    exactly the keys native measure() produces (values may be None where the
    definitions diverge — documented in morphology_metrics.md)."""
    lab = np.zeros((12, 12), dtype=np.int32)
    lab[2:6, 2:6] = 1
    native_keys = set(measure(lab, min_area=1)["per_object"][0].keys())
    assert native_keys  # sanity

    import inspect

    source = inspect.getsource(qupath_bridge.measure_via_qupath)
    # every native key must appear in the bridge row construction
    missing = [key for key in native_keys if f'"{key}"' not in source]
    assert missing == []


def test_locked_project_fails_without_waiting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """paquo surfaces a locked project as OSError; the bridge names it and
    never waits or forces the lock."""
    import sys
    import types

    # paquo importable and in-range...
    fake_paquo = types.ModuleType("paquo")
    fake_java = types.ModuleType("paquo.java")
    fake_java.qupath_version = "0.5.1"
    fake_paquo.java = fake_java
    fake_projects = types.ModuleType("paquo.projects")

    class _LockedProject:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            raise OSError("project file is locked")

        def __exit__(self, *a):
            return False

    fake_projects.QuPathProject = _LockedProject
    fake_shapely = types.ModuleType("shapely")
    fake_shapely_geom = types.ModuleType("shapely.geometry")
    fake_shapely_geom.Polygon = object
    fake_shapely.geometry = fake_shapely_geom
    monkeypatch.setitem(sys.modules, "shapely", fake_shapely)
    monkeypatch.setitem(sys.modules, "shapely.geometry", fake_shapely_geom)
    monkeypatch.setitem(sys.modules, "paquo", fake_paquo)
    monkeypatch.setitem(sys.modules, "paquo.java", fake_java)
    monkeypatch.setitem(sys.modules, "paquo.projects", fake_projects)

    import numpy as np
    from PIL import Image

    label_path = tmp_path / "label.png"
    Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(label_path)
    with pytest.raises(OrganelleDependencyError) as raised:
        measure(label_path, backend="qupath")
    assert raised.value.code == "qupath.project_locked"


def test_jvm_stays_out_of_the_core_import_graph() -> None:
    """Importing the morphology suite must not touch paquo/jpype — the JVM is
    an optional-backend concern, loaded only inside the bridge call."""
    import subprocess
    import sys

    from tests._paths import child_env

    probe = (
        "import sys, organelleverse.morphology; "
        "print('paquo' in sys.modules or 'jpype' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        env=child_env(),
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "False"
