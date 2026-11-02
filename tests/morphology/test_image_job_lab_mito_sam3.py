"""lab_mito + SAM 3 refinement (``sam3_recover``), against FAKE lab and SAM 3 scripts (no torch, no model)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from organelleverse.core.errors import OrganelleInputError
from organelleverse.morphology import image_job, lab_mito_infer, lab_mito_sam3, sam3_infer
from organelleverse.morphology.image_job import load_analysis_image, run_image_engine_probe, run_image_morphology

SHAPE = (200, 240)  # original grid; at 6 nm/px it is also the model grid
STRONG = (slice(20, 80), slice(20, 80))        # p = 0.9, 3600 px
FAINT = (slice(120, 170), slice(150, 200))     # p = 0.45, below the loose threshold, 2500 px


def _arg(cmd: list[str], flag: str) -> str:
    return cmd[cmd.index(flag) + 1]


def _build_root(root: Path, *, sam3: bool = True) -> None:
    files = {image_job._LAB_MITO_CHECKPOINT_REL: b"ck", image_job._LAB_MITO_MODEL_REL: b"#",
             image_job._LAB_MITO_PREDICT_REL: b"#", image_job._FUSION3_PYTHON_REL: b"#!/bin/sh\n"}
    if sam3:
        files |= {lab_mito_sam3.SAM3_PYTHON_REL: b"#!/bin/sh\n", lab_mito_sam3.SAM3_CHECKPOINT_REL: b"sam3",
                  lab_mito_sam3.SAM3_BPE_REL: b"bpe"}
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        if rel.endswith("bin/python"):
            path.chmod(0o755)


class _Lab:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, cmd, cwd, timeout):
        self.calls.append(cmd)
        import tifffile

        staged = tifffile.imread(_arg(cmd, "--image"))
        shape = lab_mito_infer.resampled_shape(staged.shape[:2], float(_arg(cmd, "--nm")))
        prob = np.zeros(shape, np.uint8)
        prob[STRONG] = round(0.9 * 255)
        prob[FAINT] = round(0.45 * 255)
        out = Path(_arg(cmd, "--out"))
        np.save(out, prob >= 128)
        np.save(out.with_name(out.stem + ".prob.npy"), prob)
        if "--save-image" in cmd:
            np.save(out.with_name(out.stem + ".img.npy"), np.full(shape, 100, np.uint8))
        out.with_suffix(".json").write_text(json.dumps({"checkpoint_sha256": "x"}), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")


class _Sam3:
    """Writes what sam3_infer.py would: 3 hypotheses per candidate, exemplar proposals on the faint blob."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.returncode = 0

    def __call__(self, cmd, cwd, timeout):
        self.calls.append(cmd)
        if self.returncode:
            return subprocess.CompletedProcess(cmd, self.returncode, "", "CUDA error")
        modes = _arg(cmd, "--mode").split(",")
        out = Path(_arg(cmd, "--out"))
        cand = np.load(_arg(cmd, "--candidates")) if "refine" in modes else np.zeros(SHAPE, np.int32)
        hyps = []
        for k in range(1, int(cand.max()) + 1):
            m = cand == k
            ys, xs = np.nonzero(m)
            tight = np.zeros_like(m)
            tight[ys.min() + 2:ys.max() - 1, xs.min() + 2:xs.max() - 1] = True
            hyps += [sam3_infer.to_proposal(tight, (0, 0), 0.9, "refine0", k),
                     sam3_infer.to_proposal(np.ones_like(m), (0, 0), 0.99, "refine1", k),
                     sam3_infer.to_proposal(m, (0, 0), 0.5, "refine2", k)]
        faint = np.zeros(cand.shape, bool)
        faint[FAINT] = True
        pools = {"refine": hyps, "exemplar": [sam3_infer.to_proposal(faint, (0, 0), 0.6, "exemplar", 0)]}
        for mode in modes:
            path = out if len(modes) == 1 else out.with_name(f"{out.stem}.{mode}.npz")
            np.savez_compressed(path, **sam3_infer.pack(pools[mode]))
        out.with_suffix(".json").write_text(json.dumps({"modes": modes}), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")


@pytest.fixture
def fakes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[_Lab, _Sam3, Path]:
    root = tmp_path / "root"
    _build_root(root)
    monkeypatch.setenv("ORGANELLEVERSE_FUSION3_ROOT", str(root))
    lab, sam = _Lab(), _Sam3()
    monkeypatch.setattr(image_job, "_run_lab_mito", lab)
    monkeypatch.setattr(lab_mito_sam3, "_run_sam3", sam)
    return lab, sam, root


def _run(tmp_path: Path, out: str = "out", **params) -> Path:
    import tifffile

    src = tmp_path / "em.tif"
    tifffile.imwrite(src, np.full(SHAPE, 120, np.uint8))
    run_image_morphology({"image": str(src), "engine": "lab_mito", "pixel_size_um": 0.006, "min_area": 10, **params},
                         output_dir=tmp_path / out, log=lambda _: None)
    return tmp_path / out


def test_default_rule_keeps_the_lab_objects_and_recovers_a_faint_one(tmp_path, fakes) -> None:
    lab, sam, _ = fakes
    out = _run(tmp_path, sam3_recover=True)
    assert "--save-image" in lab.calls[0]
    assert _arg(sam.calls[0], "--mode") == "exemplar" and "--candidates" not in sam.calls[0]
    exemplars = json.loads(Path(_arg(sam.calls[0], "--exemplars")).read_text(encoding="utf-8"))
    assert exemplars == [[20, 20, 80, 80]]  # only the confident object is an exemplar
    labels = load_analysis_image(out / "labels.tif")
    # object 1 = the production lab_mito object (corners rounded by the r2 opening), object 2 = SAM's find
    assert int(labels.max()) == 2 and (labels[25:75, 25:75] == 1).all() and (labels[FAINT] == 2).all()
    objects = json.loads((out / "objects.json").read_text(encoding="utf-8"))["objects"]
    assert [round(o["confidence"], 2) for o in objects] == [0.9, 0.45]
    evidence = json.loads((out / "evidence.json").read_text(encoding="utf-8"))
    assert evidence["parameters"]["sam3_recover"] is True
    assert evidence["engine"]["method"] == "lab_finetuned_unet_mitochondria+sam3_recovery"
    sam3 = evidence["engine"]["sam3"]
    assert sam3["recovered"] == 1 and sam3["refined"] == 0 and sam3["model_objects"] == 1
    assert sam3["rule"]["recover"] == [0.5, 0.2] and sam3["rule"]["refine"] is False


def test_refine_rule_replaces_the_candidate_outline(tmp_path, fakes, monkeypatch) -> None:
    _, sam, _ = fakes
    monkeypatch.setitem(lab_mito_sam3.SAM3_RULE, "refine", True)
    monkeypatch.setitem(lab_mito_sam3.SAM3_RULE, "recover", None)
    out = _run(tmp_path, sam3_recover=True)
    assert _arg(sam.calls[0], "--mode") == "refine"
    labels = load_analysis_image(out / "labels.tif")
    # the agreeing, best-scored hypothesis (rows/cols 22..77 of the 60 x 60 square) wins over the whole-image one
    assert int(labels.max()) == 1 and int((labels > 0).sum()) == 56 * 56 and (labels[22:78, 22:78] == 1).all()
    sam3 = json.loads((out / "evidence.json").read_text(encoding="utf-8"))["engine"]["sam3"]
    assert sam3["refined"] == 1 and sam3["recovered"] == 0


def test_without_sam3_recover_the_lab_rule_is_unchanged(tmp_path, fakes) -> None:
    lab, sam, _ = fakes
    out = _run(tmp_path)
    assert "--save-image" not in lab.calls[0] and sam.calls == []
    labels = load_analysis_image(out / "labels.tif")
    # production rule: p >= 0.6 then fill + open r2 (rounds the square's corners), faint blob dropped
    assert int(labels.max()) == 1 and (labels[25:75, 25:75] == 1).all() and (labels[FAINT] == 0).all()


def test_sam3_errors_are_reported(tmp_path, fakes, monkeypatch) -> None:
    _, sam, root = fakes
    sam.returncode = 3
    with pytest.raises(OrganelleInputError) as failed:
        _run(tmp_path, sam3_recover=True)
    assert failed.value.code == "image_job.sam3_failed"
    (root / lab_mito_sam3.SAM3_CHECKPOINT_REL).unlink()
    with pytest.raises(OrganelleInputError) as missing:
        _run(tmp_path, "out2", sam3_recover=True)
    assert missing.value.code == "image_job.engine_unavailable"
    assert lab_mito_sam3.SAM3_CHECKPOINT_REL in missing.value.details["missing"]
    with pytest.raises(OrganelleInputError) as wrong:
        run_image_morphology({"image": str(tmp_path / "em.tif"), "engine": "classical", "sam3_recover": True},
                             output_dir=tmp_path / "out3", log=lambda _: None)
    assert wrong.value.code == "image_job.sam3_needs_lab_mito"


def test_probe_reports_sam3_recover(tmp_path, fakes) -> None:
    _, _, root = fakes
    run_image_engine_probe({}, output_dir=tmp_path / "probe", log=lambda _: None)
    engines = json.loads((tmp_path / "probe" / "engines.json").read_text(encoding="utf-8"))["engines"]
    assert engines["sam3_recover"]["available"] is True and engines["sam3_recover"]["engine"] == "lab_mito"


def _sam3_install(root: Path, python_rel: str) -> None:
    for rel, content in {python_rel: b"py", lab_mito_sam3.SAM3_CHECKPOINT_REL: b"sam3",
                         lab_mito_sam3.SAM3_BPE_REL: b"bpe"}.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(content)
        (root / rel).chmod(0o755)


def test_sam3_is_found_in_the_workstation_home_with_a_windows_interpreter(tmp_path) -> None:
    lab_root, home = tmp_path / "lab", tmp_path / "home"
    lab_root.mkdir()
    _sam3_install(home, "data/sam3/env/python.exe")
    found = lab_mito_sam3.sam3_availability(lab_root, {"ORGANELLEVERSE_HOME": str(home)})
    assert found["available"] is True and found["root"] == str(home)
    assert found["python"] == str(home / "data/sam3/env/python.exe")
    assert found["searched"] == [str(lab_root), str(home)]


def test_an_install_at_the_model_root_wins_even_when_incomplete(tmp_path) -> None:
    lab_root, home = tmp_path / "lab", tmp_path / "home"
    _sam3_install(home, lab_mito_sam3.SAM3_PYTHON_REL)
    _sam3_install(lab_root, "data/sam3/env/Scripts/python.exe")
    (lab_root / lab_mito_sam3.SAM3_CHECKPOINT_REL).unlink()
    found = lab_mito_sam3.sam3_availability(lab_root, {"ORGANELLEVERSE_HOME": str(home)})
    assert found["available"] is False and found["root"] == str(lab_root)
    assert found["missing"] == [lab_mito_sam3.SAM3_CHECKPOINT_REL]


def test_sam3_root_variable_is_the_only_place_looked(tmp_path) -> None:
    lab_root, chosen = tmp_path / "lab", tmp_path / "chosen"
    _sam3_install(lab_root, lab_mito_sam3.SAM3_PYTHON_REL)
    found = lab_mito_sam3.sam3_availability(lab_root, {"ORGANELLEVERSE_SAM3_ROOT": str(chosen)})
    assert found["available"] is False and found["searched"] == [str(chosen)]
    assert found["missing"][0] == lab_mito_sam3.SAM3_PYTHON_REL


def test_cpu_build_redirects_the_hardcoded_cuda_tensors() -> None:
    torch = pytest.importorskip("torch")
    assert sam3_infer.resolve_device("cpu") == "cpu"
    assert sam3_infer.resolve_device("auto") == ("cuda" if torch.cuda.is_available() else "cpu")
    zeros = torch.zeros
    with sam3_infer._hardcoded_cuda_on("cpu"):
        assert torch.zeros((1, 2), device="cuda").device.type == "cpu"
        assert torch.arange(3, device="cuda").device.type == "cpu"
    assert torch.zeros is zeros


def test_a_windows_run_is_refused_up_front_when_memory_is_short() -> None:
    with pytest.raises(OrganelleInputError) as short:
        lab_mito_sam3.ensure_memory_headroom(r"C:\sam\env\python.exe", headroom_gb=lambda: 3.2)
    assert short.value.code == "image_job.sam3_memory" and short.value.details["free_gb"] == 3.2
    lab_mito_sam3.ensure_memory_headroom(r"C:\sam\env\python.exe", headroom_gb=lambda: 20.0)
    lab_mito_sam3.ensure_memory_headroom("/srv/sam/env/bin/python", headroom_gb=lambda: 0.1)


def test_a_native_crash_is_reported_as_memory_exhaustion() -> None:
    crashed = subprocess.CompletedProcess([], 3221225477, "", "")
    paging = subprocess.CompletedProcess([], 1, "", "OSError: ... (os error 1455)")
    plain = subprocess.CompletedProcess([], 2, "", "Traceback")
    assert "ran out of memory" in lab_mito_sam3.failure_message(crashed)
    assert "ran out of memory" in lab_mito_sam3.failure_message(paging)
    assert lab_mito_sam3.failure_message(plain) == "SAM 3 failed (exit 2)"


def test_safetensors_weights_are_read_straight_into_the_model(tmp_path) -> None:
    torch = pytest.importorskip("torch")

    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = torch.nn.Linear(3, 2)
            self.register_buffer("count", torch.zeros((), dtype=torch.int64))
            self.inst_interactive_predictor = None

    weight = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    bias = torch.tensor([0.5, -1.0], dtype=torch.float16)  # stored narrower: copied with a cast
    tensors = {"detector.lin.weight": ("F32", weight), "detector.lin.bias": ("F16", bias),
               "tracker.other": ("F32", torch.ones(1))}
    header, blobs, offset = {}, [], 0
    for key, (dtype, value) in tensors.items():
        data = value.numpy().tobytes()
        header[key] = {"dtype": dtype, "shape": list(value.shape), "data_offsets": [offset, offset + len(data)]}
        blobs.append(data)
        offset += len(data)
    raw = json.dumps(header).encode()
    path = tmp_path / "w.safetensors"
    path.write_bytes(len(raw).to_bytes(8, "little") + raw + b"".join(blobs))
    model = Tiny()
    missing = sam3_infer._load_streamed(model, path)
    assert missing == ["count"]
    assert torch.equal(model.lin.weight, weight) and torch.equal(model.lin.bias, bias.float())
