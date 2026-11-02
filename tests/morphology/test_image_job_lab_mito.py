"""lab_mito engine of the image job runner, against a FAKE inference script (no torch, no model is ever run)."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from organelleverse.core.errors import OrganelleInputError
from organelleverse.morphology import image_job, lab_mito_infer
from organelleverse.morphology.image_job import (
    lab_mito_availability,
    load_analysis_image,
    run_image_engine_probe,
    run_image_morphology,
)

_CHECKPOINT = b"fake lab checkpoint"
_WEIGHTS = lab_mito_infer.weights_matrix().tolist()


def _build_root(root: Path, *, skip: tuple[str, ...] = ()) -> None:
    files = {
        image_job._LAB_MITO_CHECKPOINT_REL: _CHECKPOINT,
        image_job._LAB_MITO_MODEL_REL: b"# fake model\n",
        image_job._LAB_MITO_PREDICT_REL: b"# fake predict\n",
        image_job._FUSION3_PYTHON_REL: b"#!/bin/sh\n",
    }
    for rel, content in files.items():
        if rel in skip:
            continue
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        if rel == image_job._FUSION3_PYTHON_REL:
            path.chmod(0o755)


def _arg(cmd: list[str], flag: str) -> str:
    return cmd[cmd.index(flag) + 1]


class _Runner:
    """Stands in for ``_run_lab_mito``: writes the mask the real script would write at 6 nm/px."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], str, float]] = []
        self.blobs: list[tuple[slice, slice]] = []  # in 6 nm/px coordinates
        # (rows, cols, p): mitochondrion probability written to <out>.prob.npy; None = no file (older script)
        self.probs: list[tuple[slice, slice, float]] | None = None
        self.returncode = 0

    def __call__(self, cmd: list[str], cwd: str, timeout: float) -> subprocess.CompletedProcess[str]:
        self.calls.append((cmd, cwd, timeout))
        if self.returncode:
            return subprocess.CompletedProcess(cmd, self.returncode, "loading checkpoint\n",
                                                "CUDA out of memory\n")
        import tifffile

        staged = tifffile.imread(_arg(cmd, "--image"))
        nm = float(_arg(cmd, "--nm"))
        mask = np.zeros(lab_mito_infer.resampled_shape(staged.shape[:2], nm), dtype=bool)
        for rows, cols in self.blobs:
            mask[rows, cols] = True
        out = Path(_arg(cmd, "--out"))
        np.save(out, mask)
        if self.probs is not None:
            prob = np.zeros(mask.shape, dtype=np.uint8)
            for rows, cols, value in self.probs:
                prob[rows, cols] = round(value * 255)
            np.save(out.with_name(out.stem + ".prob.npy"), prob)
        out.with_suffix(".json").write_text(json.dumps({
            "checkpoint_sha256": hashlib.sha256(_CHECKPOINT).hexdigest(), "scales": [1.0, 2.0, 3.0],
            "weights": _WEIGHTS, "nm": nm, "device": "fake-cuda", "mask_shape": list(mask.shape)}),
            encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "ok\n", "")


@pytest.fixture
def lab_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "lab-root"
    _build_root(root)
    monkeypatch.setenv("ORGANELLEVERSE_FUSION3_ROOT", str(root))
    return root


@pytest.fixture
def runner(monkeypatch: pytest.MonkeyPatch) -> _Runner:
    fake = _Runner()
    monkeypatch.setattr(image_job, "_run_lab_mito", fake)
    return fake


def _image(tmp_path: Path, shape: tuple[int, int] = (100, 120)) -> Path:
    import tifffile

    path = tmp_path / "em.tif"
    tifffile.imwrite(path, np.full(shape, 120, dtype=np.uint8))
    return path


def _job(tmp_path: Path, out: str = "out", **params: object) -> dict:
    return run_image_morphology({"image": str(_image(tmp_path)), "engine": "lab_mito", **params},
                                output_dir=tmp_path / out, log=lambda _: None)


def _objects(out: Path) -> list[dict]:
    return json.loads((out / "objects.json").read_text(encoding="utf-8"))["objects"]


def test_blob_is_mapped_back_to_original_coordinates(tmp_path: Path, lab_root: Path, runner: _Runner) -> None:
    # 12 nm/px -> factor 2: the 6 nm/px mask is 200 x 240 for the 100 x 120 original.
    runner.blobs = [(slice(40, 80), slice(60, 120))]  # -> original rows 20:40, cols 30:60 (20 x 30 = 600 px)
    result = _job(tmp_path, pixel_size_um=0.012, min_area=10)
    out = tmp_path / "out"
    labels = load_analysis_image(out / "labels.tif")
    assert labels.dtype == np.uint16 and labels.shape == (100, 120)
    assert (labels[20:40, 30:60] == 1).all() and int((labels > 0).sum()) == 600
    assert labels[19, 30] == 0 and labels[20, 29] == 0 and labels[40, 59] == 0 and labels[39, 60] == 0
    objects = _objects(out)
    assert [o["area_px"] for o in objects] == [600]
    assert result["summary_text"].startswith("lab_mito: 1 objects")


def test_non_integer_factor_keeps_exact_shape_and_location(tmp_path: Path, lab_root: Path,
                                                           runner: _Runner) -> None:
    # 10 nm/px -> factor 5/3: 60 x 90 original, 100 x 150 mask.
    import tifffile

    source = tmp_path / "em.tif"
    tifffile.imwrite(source, np.full((60, 90), 7, dtype=np.uint8))
    runner.blobs = [(slice(50, 100), slice(75, 150))]  # lower-right quadrant of the model grid
    run_image_morphology({"image": str(source), "engine": "lab_mito", "pixel_size_um": 0.01, "min_area": 5},
                         output_dir=tmp_path / "out", log=lambda _: None)
    labels = load_analysis_image(tmp_path / "out" / "labels.tif")
    assert labels.shape == (60, 90)
    assert (labels[30:, 45:] == 1).all() and int((labels > 0).sum()) == 30 * 45


def test_small_object_floor_is_in_square_micrometres(tmp_path: Path, lab_root: Path, runner: _Runner) -> None:
    # 12 nm/px: floor = ceil(0.05 / 0.012^2) = 348 px. 15 x 15 = 225 px is dropped, 30 x 30 = 900 px kept.
    runner.blobs = [(slice(10, 40), slice(10, 40)), (slice(100, 160), slice(100, 160))]
    _job(tmp_path, pixel_size_um=0.012, min_area=10)
    assert [o["area_px"] for o in _objects(tmp_path / "out")] == [900]
    evidence = json.loads((tmp_path / "out" / "evidence.json").read_text(encoding="utf-8"))
    assert evidence["engine"]["min_area_px_applied"] == 348
    # An explicit min_area above the µm² floor wins.
    _job(tmp_path, "out2", pixel_size_um=0.012, min_area=1000)
    assert _objects(tmp_path / "out2") == []
    # At 6 nm/px the floor is ceil(0.05 / 0.006^2) = 1389 px.
    runner.blobs = [(slice(10, 40), slice(10, 40)), (slice(50, 90), slice(50, 90))]  # 900 and 1600 px
    _job(tmp_path, "out3", pixel_size_um=0.006, min_area=10)
    assert [o["area_px"] for o in _objects(tmp_path / "out3")] == [1600]


def test_touching_blobs_merge_with_eight_connectivity(tmp_path: Path, lab_root: Path, runner: _Runner) -> None:
    runner.blobs = [(slice(10, 50), slice(10, 50)), (slice(50, 90), slice(50, 90))]  # touch at a corner
    _job(tmp_path, pixel_size_um=0.006, min_area=10)
    assert len(_objects(tmp_path / "out")) == 1


def test_each_object_carries_its_mean_probability(tmp_path: Path, lab_root: Path, runner: _Runner,
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    # Rule off (argmax, no opening, no floor) so only the confidence computation is tested.
    monkeypatch.setattr(image_job, "_LAB_MITO_THRESHOLD", None)
    monkeypatch.setattr(image_job, "_LAB_MITO_OPENING_PX", 0)
    monkeypatch.setattr(image_job, "_LAB_MITO_MIN_CONFIDENCE", 0.0)
    runner.blobs = [(slice(5, 45), slice(5, 45)), (slice(55, 95), slice(60, 100))]
    runner.probs = [(slice(5, 45), slice(5, 45), 0.9), (slice(55, 95), slice(60, 100), 0.55),
                    (slice(55, 75), slice(60, 100), 0.75)]   # upper half of the second blob is higher
    _job(tmp_path, pixel_size_um=0.006, min_area=10)
    rows = sorted(_objects(tmp_path / "out"), key=lambda r: r["label"])
    assert [round(r["confidence"], 3) for r in rows] == [round(230 / 255, 3), round((191 + 140) / 2 / 255, 3)]
    evidence = json.loads((tmp_path / "out" / "evidence.json").read_text(encoding="utf-8"))
    assert evidence["engine"]["rule"]["probability_available"] is True
    assert "object_confidence" not in evidence["engine"]   # per-object values live in the table only


def test_rule_threshold_opening_and_confidence_floor(tmp_path: Path, lab_root: Path, runner: _Runner,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    # The argmax mask holds three blobs; the probability decides which survive the configured rule.
    runner.blobs = [(slice(5, 45), slice(5, 45)), (slice(55, 95), slice(5, 45)), (slice(50, 95), slice(50, 120))]
    runner.probs = [(slice(5, 45), slice(5, 45), 0.95), (slice(55, 95), slice(5, 45), 0.65),
                    (slice(50, 95), slice(50, 120), 0.9), (slice(50, 95), slice(83, 87), 0.4)]
    monkeypatch.setattr(image_job, "_LAB_MITO_THRESHOLD", 0.7)
    monkeypatch.setattr(image_job, "_LAB_MITO_OPENING_PX", 2)
    _job(tmp_path, pixel_size_um=0.006, min_area=10)
    rows = _objects(tmp_path / "out")
    # 0.65 blob is below the threshold; the third blob is cut in two (45 x 33 each) by its 4-px
    # low-probability seam. The opening rounds each rectangle's corners by a few pixels.
    areas = sorted(r["area_px"] for r in rows)
    assert len(areas) == 3 and all(abs(a - e) <= 16 for a, e in zip(areas, [1485, 1485, 1600])), areas
    monkeypatch.setattr(image_job, "_LAB_MITO_THRESHOLD", None)
    monkeypatch.setattr(image_job, "_LAB_MITO_OPENING_PX", 0)
    monkeypatch.setattr(image_job, "_LAB_MITO_MIN_CONFIDENCE", 0.8)
    _job(tmp_path, "out2", pixel_size_um=0.006, min_area=10)
    rows = _objects(tmp_path / "out2")
    assert sorted(r["area_px"] for r in rows) == [1600, 3150]   # argmax mask, 0.65 blob dropped by its mean
    evidence = json.loads((tmp_path / "out2" / "evidence.json").read_text(encoding="utf-8"))
    assert evidence["engine"]["rule"]["min_confidence"] == 0.8 and evidence["engine"]["rule"]["threshold"] is None


def test_default_rule_is_the_one_chosen_on_validation() -> None:
    assert (image_job._LAB_MITO_THRESHOLD, image_job._LAB_MITO_OPENING_PX, image_job._LAB_MITO_MIN_CONFIDENCE) == (0.6, 2, 0.8)


def test_older_script_without_probability_keeps_the_argmax_rule(tmp_path: Path, lab_root: Path, runner: _Runner,
                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(image_job, "_LAB_MITO_THRESHOLD", 0.7)
    monkeypatch.setattr(image_job, "_LAB_MITO_OPENING_PX", 2)
    monkeypatch.setattr(image_job, "_LAB_MITO_MIN_CONFIDENCE", 0.8)
    runner.blobs = [(slice(5, 45), slice(5, 45))]
    _job(tmp_path, pixel_size_um=0.006, min_area=10)
    rows = _objects(tmp_path / "out")
    assert [r["area_px"] for r in rows] == [1600] and "confidence" not in rows[0]
    evidence = json.loads((tmp_path / "out" / "evidence.json").read_text(encoding="utf-8"))
    assert evidence["engine"]["rule"] | {"source": None} == {
        "threshold": None, "opening_px_at_6nm": 0, "min_confidence": 0.0, "probability_available": False, "source": None}


def test_command_and_evidence(tmp_path: Path, lab_root: Path, runner: _Runner) -> None:
    runner.blobs = [(slice(40, 80), slice(60, 120))]
    result = _job(tmp_path, pixel_size_um=0.012)
    cmd, cwd, timeout = runner.calls[0]
    assert cmd[0] == str(lab_root / "data/orgsegnet/training-venv/bin/python")
    assert Path(cmd[1]).name == "lab_mito_infer.py" and Path(cmd[1]).is_file()
    assert _arg(cmd, "--root") == str(lab_root)
    assert _arg(cmd, "--checkpoint") == str(lab_root / "experiments/plant-sam/lab-finetune-v1-s1234/best.pt")
    assert float(_arg(cmd, "--nm")) == pytest.approx(12.0)
    assert Path(_arg(cmd, "--image")).name == "source.tif" and Path(_arg(cmd, "--out")).suffix == ".npy"
    assert cwd == str(lab_root) and timeout >= 3600

    evidence = json.loads(Path(result["output_files"]["evidence"]).read_text(encoding="utf-8"))
    assert evidence["parameters"] == {"engine": "lab_mito", "min_area": 10, "pixel_size_um": 0.012}
    detail = evidence["engine"]
    assert detail["engine"] == "lab_mito" and detail["method"].startswith("lab_finetuned_unet_mitochondria")
    assert detail["checkpoint"] == str(lab_root / image_job._LAB_MITO_CHECKPOINT_REL)
    assert detail["checkpoint_sha256"] == hashlib.sha256(_CHECKPOINT).hexdigest()
    assert detail["model_nm_per_px"] == 6.0 and detail["resample_factor"] == pytest.approx(2.0)
    assert detail["scales"] == [1.0, 2.0, 3.0] and detail["weights"] == _WEIGHTS
    assert detail["inference"]["device"] == "fake-cuda"
    assert detail["connectivity"] == 8 and detail["min_area_um2"] == 0.05
    assert "Dice 0.903" in detail["scope"] and "recall 0.919" in detail["scope"]
    assert "mitochondria only" in detail["scope"] and "precision not measured" in detail["scope"]
    assert "organelle" not in detail and evidence["outputs"]["objects"] == 1


def test_non_tiff_input_is_staged_as_tiff(tmp_path: Path, lab_root: Path, runner: _Runner) -> None:
    from PIL import Image

    source = tmp_path / "em.png"
    Image.fromarray(np.full((50, 60), 90, dtype=np.uint8)).save(source)
    runner.blobs = [(slice(10, 60), slice(10, 60))]
    run_image_morphology({"image": str(source), "engine": "lab_mito", "pixel_size_um": 0.012, "min_area": 5},
                         output_dir=tmp_path / "out", log=lambda _: None)
    staged = tmp_path / "out" / "lab_mito" / "in" / "source.tif"
    assert staged.is_file() and load_analysis_image(staged).shape == (50, 60)


@pytest.mark.parametrize("pixel", [None, 0, ""])
def test_uncalibrated_image_is_rejected_without_running(tmp_path: Path, lab_root: Path, runner: _Runner,
                                                        pixel: object) -> None:
    with pytest.raises(OrganelleInputError) as error:
        _job(tmp_path, pixel_size_um=pixel)
    assert error.value.code == "image_job.calibration_required"
    assert error.value.details["model_nm_per_px"] == 6.0
    assert runner.calls == [] and not (tmp_path / "out" / "labels.tif").exists()


@pytest.mark.parametrize("skip", [
    image_job._LAB_MITO_CHECKPOINT_REL, image_job._LAB_MITO_MODEL_REL, image_job._LAB_MITO_PREDICT_REL,
    image_job._FUSION3_PYTHON_REL,
])
def test_missing_files_are_reported_and_never_fall_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                        runner: _Runner, skip: str) -> None:
    root = tmp_path / "partial"
    _build_root(root, skip=(skip,))
    monkeypatch.setenv("ORGANELLEVERSE_FUSION3_ROOT", str(root))
    with pytest.raises(OrganelleInputError) as error:
        _job(tmp_path, pixel_size_um=0.012)
    assert error.value.code == "image_job.engine_unavailable"
    assert error.value.details["engine"] == "lab_mito" and error.value.details["root"] == str(root)
    assert list(error.value.details["missing"]) == [skip]
    assert runner.calls == [] and not (tmp_path / "out" / "labels.tif").exists()


def test_failed_inference_carries_exit_code_and_output_tails(tmp_path: Path, lab_root: Path,
                                                             runner: _Runner) -> None:
    runner.returncode = 1
    with pytest.raises(OrganelleInputError) as error:
        _job(tmp_path, pixel_size_um=0.012)
    assert error.value.code == "image_job.lab_mito_failed"
    assert error.value.details["exit_code"] == 1
    assert "CUDA out of memory" in error.value.details["stderr_tail"]
    assert "loading checkpoint" in error.value.details["stdout_tail"]


def test_missing_output_mask_is_a_failure(tmp_path: Path, lab_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(image_job, "_run_lab_mito",
                        lambda cmd, cwd, timeout: subprocess.CompletedProcess(cmd, 0, "done\n", ""))
    with pytest.raises(OrganelleInputError) as error:
        _job(tmp_path, pixel_size_um=0.012)
    assert error.value.code == "image_job.lab_mito_failed" and "no mask" in error.value.message


def test_timeout_is_retryable_failure(tmp_path: Path, lab_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def hang(cmd: list[str], cwd: str, timeout: float) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd, timeout)

    monkeypatch.setattr(image_job, "_run_lab_mito", hang)
    with pytest.raises(OrganelleInputError) as error:
        _job(tmp_path, pixel_size_um=0.012)
    assert error.value.code == "image_job.lab_mito_failed" and error.value.retryable


def test_probe_reports_lab_mito_availability(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "root"
    monkeypatch.setenv("ORGANELLEVERSE_FUSION3_ROOT", str(root))
    report = json.loads(Path(run_image_engine_probe({}, output_dir=tmp_path / "a", log=lambda _: None)
                             ["output_files"]["engines"]).read_text(encoding="utf-8"))
    missing = report["engines"]["lab_mito"]
    assert missing["available"] is False and "not installed" in missing["reason"]
    assert str(root) == missing["root"] and lab_mito_availability()["available"] is False

    _build_root(root)
    report = json.loads(Path(run_image_engine_probe({}, output_dir=tmp_path / "b", log=lambda _: None)
                             ["output_files"]["engines"]).read_text(encoding="utf-8"))
    ready = report["engines"]["lab_mito"]
    assert ready["available"] is True and ready["reason"] is None and ready["root"] == str(root)
    assert lab_mito_availability()["missing"] == []


def test_other_engines_are_unchanged(tmp_path: Path, lab_root: Path, runner: _Runner) -> None:
    out = tmp_path / "out"
    run_image_morphology({"image": str(_image(tmp_path)), "engine": "classical", "min_area": 10,
                          "pixel_size_um": 0.012}, output_dir=out, log=lambda _: None)
    evidence = json.loads((out / "evidence.json").read_text(encoding="utf-8"))
    assert evidence["parameters"] == {"engine": "classical", "min_area": 10, "pixel_size_um": 0.012}
    assert evidence["engine"]["method"] == "otsu_threshold+connected_components"
    assert "checkpoint" not in evidence["engine"]
    assert runner.calls == []
    with pytest.raises(OrganelleInputError) as error:
        run_image_morphology({"image": str(_image(tmp_path)), "engine": "nope"}, output_dir=tmp_path / "x",
                             log=lambda _: None)
    assert error.value.code == "image_job.engine_unknown" and "lab_mito" in error.value.details["known"]


# --- lab_mito_infer.py preprocessing (no torch) -------------------------------------------------------------

def test_stretch_uses_0p1_and_99p9_percentiles() -> None:
    raw = np.arange(10000, dtype=np.uint16).reshape(100, 100)
    img8 = lab_mito_infer.stretch_to_uint8(raw)
    lo, hi = np.percentile(raw.astype(np.float32), [0.1, 99.9])
    expected = (np.clip((raw.astype(np.float32) - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    assert img8.dtype == np.uint8 and (img8 == expected).all()
    assert img8[0, 0] == 0 and img8[-1, -1] == 255 and img8[0, 5] == 0  # clipped below the 0.1 percentile
    flat = lab_mito_infer.stretch_to_uint8(np.full((8, 8), 500, dtype=np.uint16))
    assert (flat == 0).all()


def test_resample_size_follows_nm_over_6_with_a_floor_of_32() -> None:
    assert lab_mito_infer.resampled_shape((50, 70), 12.0) == (100, 140)
    assert lab_mito_infer.resampled_shape((50, 70), 6.0) == (50, 70)
    assert lab_mito_infer.resampled_shape((60, 90), 10.0) == (100, 150)
    assert lab_mito_infer.resampled_shape((50, 70), 1.0) == (32, 32)  # round(8.3) and round(11.7) < 32
    assert lab_mito_infer.resampled_shape((600, 640), 1.0) == (100, 107)


def test_preprocess_returns_uint8_plane_at_model_scale() -> None:
    rng = np.random.default_rng(3)
    raw = rng.integers(0, 60000, size=(50, 70), dtype=np.uint16)
    img8, notes = lab_mito_infer.preprocess(raw, 12.0)
    assert img8.dtype == np.uint8 and img8.shape == (100, 140)
    assert notes["resample_factor"] == 2.0 and notes["model_shape"] == [100, 140]
    same, _ = lab_mito_infer.preprocess(raw, 6.0)
    assert (same == lab_mito_infer.stretch_to_uint8(raw)).all()  # factor 1: bilinear resize is the identity
    assert 0 <= int(img8.min()) and int(img8.max()) <= 255


def test_weights_rows_are_normalised_like_evaluate_lab() -> None:
    weights = lab_mito_infer.weights_matrix()
    assert weights.shape == (5, 3) and np.allclose(weights.sum(1), 1.0)
    assert np.allclose(weights[0], 1 / 3) and np.allclose(weights[2], [1, 0, 0])
    assert lab_mito_infer.SCALES == (1.0, 2.0, 3.0) and lab_mito_infer.MITO_CLASS == 2


def test_inference_module_does_not_import_torch_at_top_level() -> None:
    assert not hasattr(lab_mito_infer, "torch")
