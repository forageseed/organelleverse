"""T1: morphology.train checkpoint provenance (task brief 2026-08-17).

The training wrapper must close the provenance chain at both endpoints: the
``load_from`` start checkpoint is hashed before the executor runs, and the
produced ``best_mIoU_iter_*.pth`` is resolved deterministically afterwards.
A run whose produced checkpoint cannot be addressed is a failure, never a
bare ``status="ok"``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from organelleverse.morphology import train as train_module
from organelleverse.morphology.train import train


def _write_dataset(root: Path, stems: tuple[str, ...] = ("img1",)) -> Path:
    """Minimal valid OrgSegNet fine-tune dataset layout."""
    (root / "image").mkdir(parents=True)
    (root / "label").mkdir(parents=True)
    (root / "splits").mkdir(parents=True)
    for stem in stems:
        (root / "image" / f"{stem}.tif").write_bytes(b"\x00")
        (root / "label" / f"{stem}.png").write_bytes(b"\x00")
    for split in ("train", "val", "test"):
        (root / "splits" / f"{split}.txt").write_text("\n".join(stems) + "\n")
    return root


@pytest.fixture()
def backend_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the OrgSegNet backend probe to report installed."""
    monkeypatch.setattr(
        train_module,
        "check_backend",
        lambda name, *, scan_envs=True: {"installed": True, "path": "/fake/orgseg"},
        raising=False,
    )
    # train() imports check_backend inside the function from .install, so the
    # install module attribute must be patched too.
    import organelleverse.morphology.install as install_module

    monkeypatch.setattr(
        install_module,
        "check_backend",
        lambda name, *, scan_envs=True: {"installed": True, "path": "/fake/orgseg"},
    )


@pytest.fixture()
def training_capable_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fake a discrete CUDA device meeting the declared training VRAM floor."""
    import organelleverse.morphology.device as device_module

    fake = (
        device_module.DeviceInfo(
            kind="cuda",
            name="Fake GPU",
            total_vram_gb=24.0,
            runtime_version="12.4",
        ),
        device_module.DeviceInfo(kind="cpu", name="cpu", total_vram_gb=None, runtime_version=None),
    )
    monkeypatch.setattr(device_module, "detect_devices", lambda: fake)


@pytest.fixture()
def run_ready(backend_present: None, training_capable_device: None) -> None:
    """Backend installed + training-feasible device: the run path is reachable."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_successful_run_records_both_endpoint_hashes(tmp_path: Path, run_ready: None) -> None:
    data_root = _write_dataset(tmp_path / "dataset")
    work_dir = tmp_path / "work"
    load_from = tmp_path / "OrgSegNet_iter_Version1.pth"
    load_from.write_bytes(b"pretrained-weights")

    def executor(config_diff: dict) -> None:
        wdir = Path(config_diff["work_dir"])
        wdir.mkdir(parents=True, exist_ok=True)
        (wdir / "best_mIoU_iter_2000.pth").write_bytes(b"best-weights")

    result = train(data_root, executor=executor, load_from=load_from, work_dir=work_dir)

    assert result.status == "ok"
    assert len(result.artifacts) == 1
    artifact = result.artifacts[0]
    assert artifact.kind == "model_checkpoint"
    assert artifact.sha256 == _sha256(work_dir / "best_mIoU_iter_2000.pth")
    assert result.metrics["checkpoint"] == str(work_dir / "best_mIoU_iter_2000.pth")
    assert result.metrics["checkpoint_iteration"] == 2000
    model_hashes = result.provenance.model_hashes
    assert model_hashes["load_from"]["sha256"] == _sha256(load_from)
    assert model_hashes["best_checkpoint"]["sha256"] == artifact.sha256


def test_missing_best_checkpoint_fails_closed(tmp_path: Path, run_ready: None) -> None:
    data_root = _write_dataset(tmp_path / "dataset")
    work_dir = tmp_path / "work"
    load_from = tmp_path / "start.pth"
    load_from.write_bytes(b"start")

    def executor(config_diff: dict) -> None:
        Path(config_diff["work_dir"]).mkdir(parents=True, exist_ok=True)
        # Executor "succeeds" but writes no best checkpoint.

    result = train(data_root, executor=executor, load_from=load_from, work_dir=work_dir)

    assert result.status == "failed"
    assert result.errors[0].code == "morphology.train_no_best_checkpoint"
    assert str(work_dir) in result.errors[0].message


def test_unparseable_best_checkpoint_never_guessed(tmp_path: Path, run_ready: None) -> None:
    data_root = _write_dataset(tmp_path / "dataset")
    work_dir = tmp_path / "work"
    load_from = tmp_path / "start.pth"
    load_from.write_bytes(b"start")

    def executor(config_diff: dict) -> None:
        wdir = Path(config_diff["work_dir"])
        wdir.mkdir(parents=True, exist_ok=True)
        (wdir / "best_mIoU_iter_final.pth").write_bytes(b"a")
        (wdir / "best_mIoU_iter_2000.pth").write_bytes(b"b")

    result = train(data_root, executor=executor, load_from=load_from, work_dir=work_dir)

    assert result.status == "failed"
    assert result.errors[0].code == "morphology.train_ambiguous_best_checkpoint"
    assert result.artifacts == ()


def test_multiple_parseable_checkpoints_resolve_to_highest_iteration(
    tmp_path: Path, run_ready: None
) -> None:
    data_root = _write_dataset(tmp_path / "dataset")
    work_dir = tmp_path / "work"
    load_from = tmp_path / "start.pth"
    load_from.write_bytes(b"start")

    def executor(config_diff: dict) -> None:
        wdir = Path(config_diff["work_dir"])
        wdir.mkdir(parents=True, exist_ok=True)
        # mmengine writes a new best only on improvement, iterations monotone.
        (wdir / "best_mIoU_iter_2000.pth").write_bytes(b"earlier-best")
        (wdir / "best_mIoU_iter_4000.pth").write_bytes(b"final-best")

    result = train(data_root, executor=executor, load_from=load_from, work_dir=work_dir)

    assert result.status == "ok"
    assert result.metrics["checkpoint_iteration"] == 4000
    assert result.artifacts[0].sha256 == _sha256(work_dir / "best_mIoU_iter_4000.pth")


def test_missing_load_from_refuses_before_executor_runs(tmp_path: Path, run_ready: None) -> None:
    data_root = _write_dataset(tmp_path / "dataset")
    calls: list[dict] = []

    result = train(
        data_root,
        executor=lambda config_diff: calls.append(config_diff),
        load_from=tmp_path / "does_not_exist.pth",
        work_dir=tmp_path / "work",
    )

    assert result.status == "failed"
    assert result.errors[0].code == "morphology.train_load_from_missing"
    assert calls == [], "executor must not run when the start checkpoint is unaddressable"


def test_plan_path_unchanged_without_executor(tmp_path: Path) -> None:
    """No executor -> the documented plan path, regardless of checkpoints."""
    data_root = _write_dataset(tmp_path / "dataset")
    result = train(data_root, work_dir=tmp_path / "work")
    assert result.status == "warning"
    assert result.metrics["ran"] is False
    assert result.artifacts == ()
