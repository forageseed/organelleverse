"""T3: micro-SAM as the second morphology.segment backend (task brief 2026-08-17).

Routing is a pure function of declared input metadata, never of what happens
to be installed; an explicitly pinned but unavailable backend fails closed
and the other backend's executor is never invoked.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import organelleverse.morphology.install as install_module
from organelleverse.morphology.orgseg import _route_backend, segment


def _write_image(path: Path) -> Path:
    Image.fromarray(np.zeros((32, 32), dtype=np.uint8)).save(path)
    return path


def _fake_check(installed: dict[str, bool]):
    def check(name: str, *, scan_envs: bool = True) -> dict:
        return {
            "name": name,
            "installed": installed.get(name, False),
            "path": "/fake/python" if installed.get(name, False) else None,
            "env": None,
            "version": "1.8.9" if installed.get(name, False) else None,
            "note": "",
        }

    return check


def _write_label_png(path: Path, values: np.ndarray) -> None:
    Image.fromarray(values).save(path)


def _orgseg_executor(out_dir: Path) -> list[Path]:
    label = np.zeros((32, 32), dtype=np.uint8)
    label[4:12, 4:12] = 1  # Chloroplast
    label[20:26, 20:26] = 2  # Mitochondria
    p = Path(out_dir) / "img_label.png"
    _write_label_png(p, label)
    return [p]


def _micro_sam_executor(out_dir: Path) -> list[Path]:
    # Instance ids, not class indices: two separated instances.
    label = np.zeros((32, 32), dtype=np.uint16)
    label[4:12, 4:12] = 1
    label[20:26, 20:26] = 2
    p = Path(out_dir) / "img_label.png"
    _write_label_png(p, label)
    return [p]


def test_auto_routing_is_deterministic_on_declared_metadata() -> None:
    assert _route_backend("auto", is_3d_stack=False) == "orgseg"
    assert _route_backend("auto", is_3d_stack=True) == "micro_sam"
    # Same metadata, same route — every time.
    for _ in range(5):
        assert _route_backend("auto", is_3d_stack=True) == "micro_sam"
    assert _route_backend("micro_sam", is_3d_stack=False) == "micro_sam"
    assert _route_backend("bogus", is_3d_stack=False) is None


def test_auto_route_does_not_depend_on_installed_backends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Only micro_sam "installed": 2D auto must still route to orgseg (and then
    # take the plan path, since orgseg is absent) — never silently re-route.
    monkeypatch.setattr(install_module, "check_backend", _fake_check({"micro_sam": True}))
    image = _write_image(tmp_path / "img.png")
    result = segment(image, backend="auto", is_3d_stack=False)
    assert result.status == "warning"
    assert result.metrics["backend"] == "orgseg"
    assert result.metrics["ran"] is False


def test_explicit_unavailable_backend_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(install_module, "check_backend", _fake_check({"orgseg": True}))
    calls: list[tuple] = []

    def orgseg_executor(*args: object) -> list[Path]:
        calls.append(args)
        return _orgseg_executor(tmp_path / "out")

    image = _write_image(tmp_path / "img.png")
    result = segment(
        image, backend="micro_sam", executor=orgseg_executor, output_dir=tmp_path / "out"
    )
    assert result.status == "failed"
    assert result.errors[0].code == "morphology.backend_unavailable"
    assert result.errors[0].details["backend"] == "micro_sam"
    assert calls == [], "the available backend's executor must never be invoked"


def test_unknown_backend_fails_closed(tmp_path: Path) -> None:
    image = _write_image(tmp_path / "img.png")
    result = segment(image, backend="laser_microscope_3000")
    assert result.status == "failed"
    assert result.errors[0].code == "morphology.unknown_backend"


def test_both_backends_share_the_result_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        install_module,
        "check_backend",
        _fake_check({"orgseg": True, "micro_sam": True}),
    )
    image = _write_image(tmp_path / "img.png")

    orgseg_result = segment(
        image,
        backend="orgseg",
        checkpoint=tmp_path / "user_finetuned.pth",  # user checkpoint: not publisher-gated
        executor=lambda cfg, ckpt, dev, paths, out: _orgseg_executor(Path(out)),
        output_dir=tmp_path / "out_orgseg",
    )
    msam_result = segment(
        image,
        backend="micro_sam",
        executor=lambda model, dev, paths, out: _micro_sam_executor(Path(out)),
        output_dir=tmp_path / "out_msam",
    )

    assert orgseg_result.status == "ok"
    assert msam_result.status == "ok"
    assert set(orgseg_result.metrics.keys()) == set(msam_result.metrics.keys())

    # OrgSegNet: semantic classes measured by name (zero-count classes are
    # always present in the counts mapping — pre-existing measure() shape).
    orgseg_positive = {k: v for k, v in orgseg_result.metrics["total_counts"].items() if v > 0}
    assert orgseg_positive == {"Chloroplast": 1, "Mitochondria": 1}
    # micro-SAM: instances measured under the generic class, never fabricated.
    assert msam_result.metrics["total_counts"] == {"object": 2}

    # Provenance records the actual backend and its probed version.
    assert orgseg_result.provenance.actual_backend == "orgseg"
    assert msam_result.provenance.actual_backend == "micro_sam"
    assert msam_result.provenance.software_versions["micro_sam"] == "1.8.9"
    assert "backend_micro_sam" in msam_result.flags
    assert "backend_orgseg" in orgseg_result.flags


def test_segment_marks_outputs_as_model_sourced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(install_module, "check_backend", _fake_check({"orgseg": True}))
    image = _write_image(tmp_path / "img.png")
    result = segment(
        image,
        backend="orgseg",
        checkpoint=tmp_path / "user_finetuned.pth",  # user checkpoint: not publisher-gated
        executor=lambda cfg, ckpt, dev, paths, out: _orgseg_executor(Path(out)),
        output_dir=tmp_path / "out",
    )
    rows = result.metrics["per_image"][0]["per_object"]
    assert rows and all(row["label_source"] == "model" for row in rows)


def test_missing_official_checkpoint_fails_with_guidance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The official OrgSegNet checkpoint is license-gated: missing means an
    explicit failure with download guidance — never a silent micro-SAM swap."""
    monkeypatch.setattr(install_module, "check_backend", _fake_check({"orgseg": True}))
    image = _write_image(tmp_path / "img.png")
    calls: list[tuple] = []
    result = segment(
        image,
        backend="orgseg",
        checkpoint=tmp_path / "OrgSegNet_iter_Version1.pth",  # official name, absent
        executor=lambda *args: calls.append(args),
        output_dir=tmp_path / "out",
    )
    assert result.status == "failed"
    assert result.errors[0].code == "morphology.weights_missing"
    assert "zenodo" in result.errors[0].details["next_step"].lower()
    assert calls == [], "the executor must not run without the gated weights"


def test_corrupt_official_checkpoint_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(install_module, "check_backend", _fake_check({"orgseg": True}))
    image = _write_image(tmp_path / "img.png")
    corrupt = tmp_path / "OrgSegNet_iter_Version1.pth"
    corrupt.write_bytes(b"truncated download")
    result = segment(
        image,
        backend="orgseg",
        checkpoint=corrupt,
        executor=lambda *args: (_ for _ in ()).throw(AssertionError("must not run")),
        output_dir=tmp_path / "out",
    )
    assert result.status == "failed"
    assert result.errors[0].code == "morphology.weights_size_mismatch"
