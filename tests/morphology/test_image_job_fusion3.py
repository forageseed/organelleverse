"""fusion3 engine of the image job runner, against a FAKE delivery (no model is ever run)."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from organelleverse.core.errors import OrganelleInputError
from organelleverse.morphology import image_job
from organelleverse.morphology.fusion3 import FUSION3_CONFIG_REL, FUSION3_ENTRY_REL
from organelleverse.morphology.image_job import (
    fusion3_availability,
    load_analysis_image,
    run_image_engine_probe,
    run_image_morphology,
)

_FAKE_ENTRY = '''
import argparse
import json
from pathlib import Path

import numpy as np
import tifffile

parser = argparse.ArgumentParser()
parser.add_argument("--image-dir", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--names", nargs="+", required=True)
parser.add_argument("--clean", action="store_true")
args = parser.parse_args()
out = Path(args.output)
out.mkdir(parents=True, exist_ok=True)
for name in args.names:
    shape = tifffile.imread(Path(args.image_dir) / f"{name}.tif").shape[:2]
    classes = np.zeros(shape, dtype=np.uint8)
    classes[5:15, 5:15] = 1    # chloroplast A (100 px)
    classes[5:15, 15:25] = 1   # chloroplast B touches A; too small to split (peaks < min_distance) -> 200 px
    classes[30:40, 40:50] = 1  # separate chloroplast (100 px)
    classes[45:46, 5:8] = 1    # 3 px chloroplast speck
    classes[50:55, 60:70] = 2  # mitochondrion (50 px)
    classes[20:25, 60:70] = 3  # vacuole
    tifffile.imwrite(out / f"{name}.class.tif", classes)
(out / "delivery-run.json").write_text(json.dumps({"device": "fake-cpu", "names": args.names}))
'''

_FAILING_ENTRY = "import sys\nprint('hash mismatch in member 2', file=sys.stderr)\nsys.exit(3)\n"


def _build_delivery(root: Path, *, entry: str = _FAKE_ENTRY, interpreters: bool = True) -> None:
    config = root / FUSION3_CONFIG_REL
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps({"name": "fusion3-v1"}), encoding="utf-8")
    script = root / FUSION3_ENTRY_REL
    script.parent.mkdir(parents=True)
    script.write_text(entry, encoding="utf-8")
    if interpreters:
        for rel in (image_job._FUSION3_PYTHON_REL, image_job._FUSION3_BASELINE_PYTHON_REL):
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("#!/bin/sh\n", encoding="utf-8")
            path.chmod(0o755)


@pytest.fixture
def calls() -> list[tuple[list[str], str, float]]:
    return []


@pytest.fixture
def delivery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
             calls: list[tuple[list[str], str, float]]) -> Path:
    root = tmp_path / "delivery-root"
    _build_delivery(root)
    monkeypatch.setenv("ORGANELLEVERSE_FUSION3_ROOT", str(root))

    def fake_run(cmd: list[str], cwd: str, timeout: float) -> subprocess.CompletedProcess[str]:
        calls.append((cmd, cwd, timeout))
        # Same command, but the (shell-script) venv interpreter is replaced by this Python.
        return subprocess.run([sys.executable, *cmd[1:]], cwd=cwd, capture_output=True, text=True,
                              timeout=timeout, check=False)

    monkeypatch.setattr(image_job, "_run_delivery", fake_run)
    return root


def _image(tmp_path: Path) -> Path:
    import tifffile

    path = tmp_path / "em.tif"
    tifffile.imwrite(path, np.full((60, 80), 120, dtype=np.uint8))
    return path


def _areas(out: Path) -> list[int]:
    objects = json.loads((out / "objects.json").read_text(encoding="utf-8"))
    return sorted(o["area_px"] for o in objects["objects"])


def test_chloroplast_job_merges_touching_blobs_and_drops_speck(tmp_path: Path, delivery: Path, calls: list) -> None:
    source = _image(tmp_path)
    out = tmp_path / "out"
    result = run_image_morphology({"image": str(source), "engine": "fusion3", "organelle": "chloroplast",
                                   "min_area": 10, "pixel_size_um": 0.01}, output_dir=out, log=lambda _: None)
    assert _areas(out) == [100, 200]
    labels = load_analysis_image(out / "labels.tif")
    assert labels.dtype == np.uint16 and labels.shape == (60, 80)
    assert sorted(np.unique(labels).tolist()) == [0, 1, 2]
    assert result["summary_text"].startswith("fusion3: 2 objects")

    cmd, cwd, timeout = calls[0]
    assert cmd[0] == str(delivery / "data/orgsegnet/training-venv/bin/python")
    assert cmd[1] == "scripts/plant-sam/deliver_fusion3.py"
    assert cmd[cmd.index("--names") + 1] == "source" and "--clean" in cmd
    assert cwd == str(delivery) and timeout >= 3600
    assert Path(cmd[cmd.index("--image-dir") + 1], "source.tif").is_file()

    evidence = json.loads(Path(result["output_files"]["evidence"]).read_text(encoding="utf-8"))
    config_sha = hashlib.sha256((delivery / FUSION3_CONFIG_REL).read_bytes()).hexdigest()
    assert evidence["parameters"] == {"engine": "fusion3", "min_area": 10, "pixel_size_um": 0.01,
                                      "organelle": "chloroplast"}
    detail = evidence["engine"]
    assert detail["engine"] == "fusion3" and detail["delivery"] == "fusion3-v1"
    assert detail["organelle"] == "chloroplast" and detail["class_id"] == 1
    assert detail["root"] == str(delivery) and detail["config_sha256"] == config_sha
    assert detail["delivery_run"]["device"] == "fake-cpu"
    assert "V6-corpus" in detail["scope"] and "split by a distance watershed" in detail["scope"]
    assert evidence["outputs"]["objects"] == 2


def test_mitochondria_job_uses_class_two(tmp_path: Path, delivery: Path) -> None:
    out = tmp_path / "out"
    run_image_morphology({"image": str(_image(tmp_path)), "engine": "fusion3", "organelle": "mitochondria",
                          "min_area": 10}, output_dir=out, log=lambda _: None)
    assert _areas(out) == [50]
    evidence = json.loads((out / "evidence.json").read_text(encoding="utf-8"))
    assert evidence["engine"]["class_id"] == 2 and evidence["parameters"]["organelle"] == "mitochondria"


def test_non_tiff_input_is_staged_as_tiff(tmp_path: Path, delivery: Path) -> None:
    from PIL import Image

    source = tmp_path / "em.png"
    Image.fromarray(np.full((60, 80), 90, dtype=np.uint8)).save(source)
    out = tmp_path / "out"
    run_image_morphology({"image": str(source), "engine": "fusion3", "organelle": "mitochondria"},
                         output_dir=out, log=lambda _: None)
    staged = out / "fusion3" / "in" / "source.tif"
    assert staged.is_file() and load_analysis_image(staged).shape == (60, 80)
    assert _areas(out) == [50]


def test_missing_delivery_is_reported_and_never_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "partial"
    _build_delivery(root, interpreters=False)
    monkeypatch.setenv("ORGANELLEVERSE_FUSION3_ROOT", str(root))

    def must_not_run(*_: object) -> None:
        raise AssertionError("delivery must not run when it is not installed")

    monkeypatch.setattr(image_job, "_run_delivery", must_not_run)
    out = tmp_path / "out"
    with pytest.raises(OrganelleInputError) as error:
        run_image_morphology({"image": str(_image(tmp_path)), "engine": "fusion3",
                              "organelle": "chloroplast"}, output_dir=out, log=lambda _: None)
    assert error.value.code == "image_job.engine_unavailable"
    assert "fusion3-v1 delivery is not installed on this compute target" in error.value.message
    assert error.value.details["root"] == str(root)
    assert sorted(error.value.details["missing"]) == sorted([
        "data/orgsegnet/training-venv/bin/python", "data/orgsegnet/orgsegnet-baseline-venv/bin/python"])
    assert not (out / "labels.tif").exists()


def test_failed_delivery_carries_exit_code_and_output_tail(
    tmp_path: Path, delivery: Path,
) -> None:
    (delivery / FUSION3_ENTRY_REL).write_text(_FAILING_ENTRY, encoding="utf-8")
    with pytest.raises(OrganelleInputError) as error:
        run_image_morphology({"image": str(_image(tmp_path)), "engine": "fusion3",
                              "organelle": "chloroplast"}, output_dir=tmp_path / "out", log=lambda _: None)
    assert error.value.code == "image_job.fusion3_failed"
    assert error.value.details["exit_code"] == 3
    assert "hash mismatch" in error.value.details["stderr_tail"]


def test_class_map_with_wrong_shape_is_rejected(tmp_path: Path, delivery: Path) -> None:
    (delivery / FUSION3_ENTRY_REL).write_text(
        _FAKE_ENTRY.replace(".tif\").shape[:2]", ".tif\").shape[:2][::-1]"), encoding="utf-8")
    with pytest.raises(OrganelleInputError) as error:
        run_image_morphology({"image": str(_image(tmp_path)), "engine": "fusion3",
                              "organelle": "chloroplast"}, output_dir=tmp_path / "out", log=lambda _: None)
    assert error.value.code == "image_job.mask_dimension_mismatch"


def test_fusion3_requires_an_organelle(tmp_path: Path, delivery: Path, calls: list) -> None:
    for organelle in (None, "", "vacuole"):
        with pytest.raises(OrganelleInputError) as error:
            run_image_morphology({"image": str(_image(tmp_path)), "engine": "fusion3", "organelle": organelle},
                                 output_dir=tmp_path / "out", log=lambda _: None)
        assert error.value.code == "image_job.organelle_required"
    assert calls == []


def test_probe_reports_fusion3_availability(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "root"
    monkeypatch.setenv("ORGANELLEVERSE_FUSION3_ROOT", str(root))
    report = json.loads(Path(run_image_engine_probe({}, output_dir=tmp_path / "a", log=lambda _: None)
                             ["output_files"]["engines"]).read_text(encoding="utf-8"))
    missing = report["engines"]["fusion3"]
    assert missing["available"] is False and "not installed" in missing["reason"]
    assert fusion3_availability()["available"] is False

    _build_delivery(root)
    report = json.loads(Path(run_image_engine_probe({}, output_dir=tmp_path / "b", log=lambda _: None)
                             ["output_files"]["engines"]).read_text(encoding="utf-8"))
    ready = report["engines"]["fusion3"]
    assert ready["available"] is True and ready["reason"] is None and ready["root"] == str(root)
    assert fusion3_availability()["missing"] == []


def test_other_engines_keep_their_parameters_and_evidence(tmp_path: Path, delivery: Path, calls: list) -> None:
    out = tmp_path / "out"
    run_image_morphology({"image": str(_image(tmp_path)), "engine": "classical", "min_area": 10,
                          "organelle": "chloroplast"}, output_dir=out, log=lambda _: None)
    evidence = json.loads((out / "evidence.json").read_text(encoding="utf-8"))
    assert evidence["parameters"] == {"engine": "classical", "min_area": 10, "pixel_size_um": None}
    assert evidence["engine"]["method"] == "otsu_threshold+connected_components"
    assert "organelle" not in evidence["engine"] and "delivery" not in evidence["engine"]
    assert calls == []


def test_large_touching_chloroplasts_are_split() -> None:
    from organelleverse.morphology.image_job import _split_touching

    yy, xx = np.mgrid[0:1000, 0:1800]
    mask = ((yy - 500) ** 2 + (xx - 480) ** 2 <= 400 ** 2) | ((yy - 500) ** 2 + (xx - 1280) ** 2 <= 400 ** 2)
    mask[900:903, 1700:1703] = True  # 9 px speck the coarse grid may miss: stays its own object
    labels = _split_touching(mask, downsample=4, min_distance=140)
    big = [int((labels == k).sum()) for k in np.unique(labels[labels > 0])]
    big = sorted(a for a in big if a > 1000)
    assert len(big) == 2 and abs(big[0] - big[1]) < 0.02 * big[0]
    assert labels[500, 480] != labels[500, 1280]
    assert labels[901, 1701] not in (labels[500, 480], labels[500, 1280])
    assert ((labels > 0) == mask).all()
