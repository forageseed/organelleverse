"""T5: device resolution — deterministic auto-routing, fail-closed explicit.

No silent CPU fallback anywhere: an explicit device that is unavailable or
unsupported by the selected backend fails with the feasible list, and the
resolved device lands in provenance.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import organelleverse.morphology.device as device_module
import organelleverse.morphology.install as install_module
from organelleverse.morphology.device import (
    DeviceInfo,
    detect_devices,
    resolve_device,
)
from organelleverse.morphology.orgseg import segment
from organelleverse.morphology.train import train

CUDA = DeviceInfo(kind="cuda", name="RTX 4090", total_vram_gb=24.0, runtime_version="12.4")
IGPU = DeviceInfo(kind="xpu", name="Arc iGPU", total_vram_gb=None, runtime_version="2.5")
MPS = DeviceInfo(kind="mps", name="apple-silicon-mps", total_vram_gb=None, runtime_version="2.5")
CPU = DeviceInfo(kind="cpu", name="cpu", total_vram_gb=None, runtime_version="2.5")


def test_auto_routing_is_deterministic_per_hardware_and_backend() -> None:
    detected = (CUDA, IGPU, CPU)
    first = resolve_device("auto", backend="micro_sam", detected=detected)
    for _ in range(5):
        again = resolve_device("auto", backend="micro_sam", detected=detected)
        assert again == first
    assert first.device is not None and first.device.kind == "cuda"

    # orgseg never routes to xpu/mps even when they are the only accelerators.
    only_igpu = resolve_device("auto", backend="orgseg", detected=(IGPU, CPU))
    assert only_igpu.device is not None and only_igpu.device.kind == "cpu"
    # micro_sam on the same hardware takes the accelerator.
    msam = resolve_device("auto", backend="micro_sam", detected=(IGPU, CPU))
    assert msam.device is not None and msam.device.kind == "xpu"


def test_explicit_unavailable_device_fails_with_feasible_list() -> None:
    resolution = resolve_device("cuda:0", backend="micro_sam", detected=(CPU,))
    assert not resolution.ok
    assert resolution.error_code == "morphology.device_unavailable"
    assert resolution.feasible == ("cpu",)


def test_orgseg_rejects_mps_and_xpu_pointing_at_micro_sam() -> None:
    for requested in ("mps", "xpu"):
        resolution = resolve_device(requested, backend="orgseg", detected=(MPS, IGPU, CPU))
        assert not resolution.ok
        assert resolution.error_code == "morphology.device_backend_unsupported"
        assert "micro_sam" in (resolution.error_message or "")


def test_integrated_gpu_training_is_rejected_before_running() -> None:
    resolution = resolve_device("auto", backend="orgseg", purpose="train", detected=(IGPU, CPU))
    assert not resolution.ok
    assert resolution.error_code == "morphology.train_device_infeasible"
    # A big discrete CUDA device passes the gate.
    ok = resolve_device("auto", backend="orgseg", purpose="train", detected=(CUDA, CPU))
    assert ok.ok and ok.device is not None and ok.device.kind == "cuda"


def test_no_silent_cpu_fallback_branch_exists() -> None:
    """Static: no 'device unavailable -> use cpu' logic anywhere in the suite."""
    suite = Path(__file__).resolve().parents[2] / "src" / "organelleverse" / "morphology"
    offenders: list[str] = []
    for path in suite.glob("*.py"):
        if path.name == "device.py":
            continue  # the declared resolution layer itself
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                value = node.value
                if value in {"cpu", "cpu:0"} and "device" not in path.name:
                    # A bare cpu string literal outside the resolution layer is
                    # only acceptable as documentation in tests, not suite code.
                    offenders.append(f"{path.name}: bare {value!r} literal")
    assert offenders == []


def test_detect_devices_never_requires_torch() -> None:
    detected = detect_devices()
    assert detected and detected[-1].kind == "cpu"


def _write_image(path: Path) -> Path:
    Image.fromarray(np.zeros((16, 16), dtype=np.uint8)).save(path)
    return path


def test_segment_records_resolved_device_in_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(device_module, "detect_devices", lambda: (CUDA, CPU))
    monkeypatch.setattr(
        install_module,
        "check_backend",
        lambda name, *, scan_envs=True: {
            "installed": True,
            "path": "/fake/python",
            "env": None,
            "version": "0.30.0",
            "note": "",
        },
    )
    seen: list[str] = []

    def executor(cfg: object, ckpt: object, device: str, paths: list[str], out: str) -> list[Path]:
        seen.append(device)
        label = np.zeros((16, 16), dtype=np.uint8)
        label[2:6, 2:6] = 1
        p = Path(out) / "img_label.png"
        Image.fromarray(label).save(p)
        return [p]

    result = segment(
        _write_image(tmp_path / "img.png"),
        checkpoint=tmp_path / "user_finetuned.pth",  # user checkpoint: not publisher-gated
        executor=executor,
    )
    assert result.status == "ok"
    assert seen == ["cuda:0"], "the executor must receive the resolved device"
    record = result.provenance.software_versions["device"]
    assert record["kind"] == "cuda"
    assert record["runtime_version"] == "12.4"
    assert result.metrics["device"] == "cuda:0"
    assert result.metrics["requested_device"] == "auto"


def test_segment_explicit_unavailable_device_never_executes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(device_module, "detect_devices", lambda: (CPU,))
    monkeypatch.setattr(
        install_module,
        "check_backend",
        lambda name, *, scan_envs=True: {
            "installed": True,
            "path": "/fake/python",
            "env": None,
            "version": "1.8.9",
            "note": "",
        },
    )
    calls: list[tuple] = []
    result = segment(
        _write_image(tmp_path / "img.png"),
        backend="micro_sam",
        device="cuda:0",
        executor=lambda *args: calls.append(args),
    )
    assert result.status == "failed"
    assert result.errors[0].code == "morphology.device_unavailable"
    assert "cpu" in result.errors[0].details["feasible_devices"]
    assert calls == [], "nothing ran on a substitute device"


def _minimal_dataset(root: Path) -> Path:
    (root / "image").mkdir(parents=True)
    (root / "label").mkdir(parents=True)
    (root / "splits").mkdir(parents=True)
    (root / "image" / "a.tif").write_bytes(b"\x00")
    (root / "label" / "a.png").write_bytes(b"\x00")
    for split in ("train", "val", "test"):
        (root / "splits" / f"{split}.txt").write_text("a\n")
    return root


def test_train_rejects_integrated_device_before_executor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(device_module, "detect_devices", lambda: (IGPU, CPU))
    monkeypatch.setattr(
        install_module,
        "check_backend",
        lambda name, *, scan_envs=True: {"installed": True, "path": "/fake", "env": None},
    )
    load_from = tmp_path / "start.pth"
    load_from.write_bytes(b"start")
    calls: list[dict] = []
    result = train(
        _minimal_dataset(tmp_path / "ds"),
        executor=lambda cfg: calls.append(cfg),
        load_from=load_from,
        work_dir=tmp_path / "work",
    )
    assert result.status == "failed"
    assert result.errors[0].code == "morphology.train_device_infeasible"
    assert calls == [], "training must be rejected before the executor runs"
