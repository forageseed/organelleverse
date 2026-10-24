"""Remote OrgSegNet executor: run the official checkpoint on the compute host.

The local machine rarely has OrgSegNet's pinned stack (torch 1.13.1 +
mmcv 2.0.0rc4 + mmseg 1.0.0). The fusion3 delivery host, however, already
ships a self-contained OrgSegNet environment — the released Zenodo
checkpoint, the authors' config, and a venv running the delivery's
``scripts/plant-sam/predict_orgsegnet.py`` (mmcv-lite + an explicit
``mmcv._ext`` stub, so no compiled mmcv is needed). When the local stack is
absent, the orgseg backend executes THERE instead of failing closed with an
install guide.

Paths below mirror ``experiments/plant-sam/delivery-fusion3-v1/delivery-config.json``
(member ``orgseg``) — the single delivery source of truth. The remote script
applies the authors' own test protocol (Resize to 768x512 keep-ratio,
ImageNet mean/std on a 3-channel-replicated grayscale, whole-image mode) and
writes one ``<name>.class.tif`` uint8 label map per image, which this module
converts to the same ``<stem>_label.png`` contract the local executors
return, so measurement consumes local and remote runs identically.

Custom (fine-tuned) checkpoints: when the caller passes a checkpoint that
exists locally, it is uploaded and used in place of the official one; mind
the ~490 MB upload cost per run.

Scope note (from the delivery README): the official checkpoint was trained
on the public V6 corpus; on the lab's own samples it is NOT a substitute for
the lab fine-tuned model.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from . import remote_copy
from .fusion3 import (
    DEFAULT_FUSION3_HOST,
    DEFAULT_FUSION3_ROOT,
    _ssh_base,
    fusion3_env,
)

__all__ = [
    "check_orgseg_local",
    "check_orgseg_remote",
    "default_orgseg_delivery_executor",
    "orgseg_local_paths",
]

#: Delivery-relative locations of the OrgSegNet member (fusion3-v1 config).
ORGSEG_REMOTE_SCRIPT = "scripts/plant-sam/predict_orgsegnet.py"
ORGSEG_REMOTE_REPO = "_validation/2026-09-08-plant-sam/orgsegnet-baseline-v1/OrgSegNet"
ORGSEG_REMOTE_CONFIG = (
    "_validation/2026-09-08-plant-sam/orgsegnet-baseline-v1/OrgSegNet"
    "/configs/OrgsegNet_resolution/OrgSeg_PlantCell_768x512.py"
)
ORGSEG_REMOTE_CHECKPOINT = (
    "_validation/2026-09-08-plant-sam/orgsegnet-baseline-v1/OrgSegNet_iter_Version1.pth"
)
ORGSEG_REMOTE_PYTHON = "data/orgsegnet/orgsegnet-baseline-venv/bin/python"

#: Local deployment of the same delivery (OrgSegNet 先行本地部署). When these
#: exist on the service's own machine, orgseg runs HERE — no SSH, no network.
#: Layout mirrors the delivery: <root>/OrgSegNet_iter_Version1.pth,
#: <root>/OrgSegNet (repo with vendored mmseg + configs), <root>/scripts/plant-sam/
#: predict_orgsegnet.py, plus a python venv that has torch/mmengine/mmcv-lite.
DEFAULT_ORGSEG_LOCAL_ROOT = str(Path.home() / "organelleverse-orgseg-local")
DEFAULT_ORGSEG_LOCAL_PYTHON = str(Path.home() / "orgsegnet-local-venv" / "bin" / "python")


def _env(environ):
    return os.environ if environ is None else environ


def orgseg_local_paths(environ=None):
    """Resolve the local delivery layout; returns None when not configured."""
    env = _env(environ)
    root = env.get("ORGANELLEVERSE_ORGSEG_LOCAL_ROOT", "")
    python = env.get("ORGANELLEVERSE_ORGSEG_LOCAL_PYTHON", "")
    if not root:
        root = DEFAULT_ORGSEG_LOCAL_ROOT
        python = python or DEFAULT_ORGSEG_LOCAL_PYTHON
    if not (Path(root).is_dir() and Path(python).exists()):
        return None
    return {
        "root": root,
        "python": python or "python3",
        "script": f"{root}/scripts/plant-sam/predict_orgsegnet.py",
        "repo": f"{root}/OrgSegNet",
        "config": f"{root}/OrgSegNet/configs/OrgsegNet_resolution/OrgSeg_PlantCell_768x512.py",
        "checkpoint": f"{root}/OrgSegNet_iter_Version1.pth",
    }


def check_orgseg_local(environ=None):
    """Probe the local delivery deployment (pure filesystem, no network)."""
    layout = orgseg_local_paths(environ)
    if layout is None:
        return {"name": "orgseg-local", "installed": False, "path": None, "note": "local orgseg delivery not configured"}
    needed = [layout["python"], layout["script"], layout["config"], layout["checkpoint"]]
    ready = all(Path(n).exists() for n in needed) and Path(layout["repo"]).is_dir()
    return {
        "name": "orgseg-local",
        "installed": ready,
        "path": layout["root"] if ready else None,
        "note": f"local OrgSegNet delivery at {layout['root']} (CPU/GPU auto)"
        if ready
        else f"local orgseg delivery at {layout['root']} is incomplete",
    }

_PROBE_TIMEOUT_S = 15
_RUN_TIMEOUT_S = 3600


def check_orgseg_remote(
    host: str | None = None,
    root: str | None = None,
    *,
    environ: dict[str, str] | None = None,
    _runner: Any = subprocess.run,
) -> dict[str, Any]:
    """Probe the compute host for a runnable remote OrgSegNet stack."""
    cfg = fusion3_env(environ)
    host = host or cfg["host"]
    root = root or cfg["root"]
    probe = (
        f"test -x {root}/{ORGSEG_REMOTE_PYTHON}"
        f" && test -f {root}/{ORGSEG_REMOTE_SCRIPT}"
        f" && test -f {root}/{ORGSEG_REMOTE_CONFIG}"
        f" && test -f {root}/{ORGSEG_REMOTE_CHECKPOINT}"
        f" && test -d {root}/{ORGSEG_REMOTE_REPO}"
        f" && echo ORGSEG_REMOTE_READY"
    )
    try:
        result = _runner(
            [*_ssh_base(host), probe],
            capture_output=True, text=True, timeout=_PROBE_TIMEOUT_S,
        )
        ready = result.returncode == 0 and "ORGSEG_REMOTE_READY" in result.stdout
    except (OSError, subprocess.TimeoutExpired):
        ready = False
    return {
        "name": "orgseg-remote",
        "installed": ready,
        "host": host,
        "root": root,
        "path": f"{host}:{root}" if ready else None,
        "note": (
            f"official OrgSegNet checkpoint runs remotely at {host}:{root} "
            "(delivery venv; authors' 768x512 whole-image protocol)"
            if ready
            else f"remote OrgSegNet delivery not reachable at {host}:{root}"
        ),
    }


def default_orgseg_delivery_executor(
    config: str | Path | None,
    checkpoint: str | Path | None,
    device: str,
    image_paths: list[str],
    out_dir: str | Path,
    *,
    host: str | None = None,
    root: str | None = None,
    force_ssh: bool = False,
    clean_remote: bool = True,
    environ: dict[str, str] | None = None,
    _runner: Any = subprocess.run,
) -> list[Path]:
    """Run the delivery's OrgSegNet inference on the compute host.

    Matches the local orgseg executor contract
    ``(config, checkpoint, device, image_paths, out_dir) -> label paths``.
    ``config``/``checkpoint`` refer to LOCAL paths: when they exist they are
    uploaded and used (fine-tuned weights; ~490 MB upload), otherwise the
    delivery's official checkpoint and the authors' config run remotely.
    ``device`` is a local concept and is ignored — the remote venv picks GPU
    when present.
    """
    cfg = fusion3_env(environ)
    host = host or cfg["host"]
    root = root or cfg["root"]
    images = [Path(p) for p in image_paths]
    for image in images:
        if not image.is_file():
            raise RuntimeError(f"remote orgseg input image not found: {image}")
    stems = [image.stem for image in images]
    if len(set(stems)) != len(stems):
        raise RuntimeError("orgseg delivery needs unique image file names")

    # Local deployment first: when the delivery runs on this very machine the
    # images are already local — execute directly, no SSH, no network.
    local = orgseg_local_paths(environ)
    if local is not None and not force_ssh:
        return _run_orgseg_local(
            local, config, checkpoint, images, stems, out_dir, _runner
        )
    if shutil.which("scp") is None or shutil.which("ssh") is None:
        raise RuntimeError("remote orgseg executor needs ssh/scp on PATH")

    uploads: dict[str, Path] = {f"{stem}.tif": image for stem, image in zip(stems, images)}
    remote_checkpoint = ORGSEG_REMOTE_CHECKPOINT
    local_checkpoint = Path(checkpoint) if checkpoint else None
    if local_checkpoint is not None and local_checkpoint.is_file():
        uploads[local_checkpoint.name] = local_checkpoint
        remote_checkpoint = local_checkpoint.name
    remote_config = ORGSEG_REMOTE_CONFIG
    local_config = Path(config) if config else None
    if local_config is not None and local_config.is_file():
        uploads[local_config.name] = local_config
        remote_config = local_config.name

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    remote = _runner(
        [*_ssh_base(host), "mktemp -d /tmp/orgseg.XXXXXX"],
        capture_output=True, text=True, timeout=_PROBE_TIMEOUT_S,
    ).stdout.strip()
    if not remote:
        raise RuntimeError(f"remote orgseg could not create a work dir on {host}")

    def _cleanup() -> None:
        if clean_remote and remote:
            try:
                _runner([*_ssh_base(host), f"rm -rf -- {remote}"],
                        capture_output=True, text=True, timeout=_PROBE_TIMEOUT_S)
            except Exception:
                pass

    manifest_dir: Path | None = None
    try:
        # scp keeps the source basename, but the delivery script expects
        # <stem>.tif inputs (and remote-named weights/config): upload each
        # file to its explicit remote path.
        for remote_name, src in uploads.items():
            remote_copy.upload_as(_runner, host, src, f"{remote}/{remote_name}", timeout=1800)
        manifest_dir = Path(tempfile.mkdtemp(prefix="orgseg-manifest-"))
        manifest_local = manifest_dir / "manifest.json"
        manifest_local.write_text(json.dumps({"deliver": stems}), encoding="utf-8")
        remote_copy.upload_as(_runner, host, manifest_local, f"{remote}/manifest.json", timeout=60)
        names_arg = " ".join(stems)
        cmd = [
            *_ssh_base(host),
            f"cd {root} && {ORGSEG_REMOTE_PYTHON} {ORGSEG_REMOTE_SCRIPT}"
            f" --repo {ORGSEG_REMOTE_REPO} --config {remote_config}"
            f" --checkpoint {remote_checkpoint}"
            f" --manifest {remote}/manifest.json --split deliver"
            f" --image-dir {remote} --output {remote}/out --names {names_arg}",
        ]
        run = _runner(cmd, capture_output=True, text=True, timeout=_RUN_TIMEOUT_S)
        if run.returncode != 0:
            raise RuntimeError(
                f"remote orgseg failed on {host} (exit {run.returncode}): {(run.stdout or '')[-2000:]}"
            )
        staged = Path(tempfile.mkdtemp(prefix="orgseg-out-"))
        remote_copy.download(
            _runner, host, [f"{remote}/out/{s}.class.tif" for s in stems], staged, timeout=600,
        )
    except Exception:
        _cleanup()
        raise
    finally:
        if manifest_dir is not None:
            shutil.rmtree(manifest_dir, ignore_errors=True)

    label_paths: list[Path] = []
    try:
        import numpy as np
        from PIL import Image

        for stem in stems:
            class_map = np.array(Image.open(staged / f"{stem}.class.tif"))
            if class_map.ndim != 2:
                raise RuntimeError(f"remote orgseg label map for {stem} is not 2D")
            label = out / f"{stem}_label.png"
            Image.fromarray(class_map.astype("uint8")).save(str(label))
            label_paths.append(label)
    finally:
        _cleanup()
        shutil.rmtree(staged, ignore_errors=True)
    return label_paths


def _run_orgseg_local(
    local: dict[str, str],
    config: str | Path | None,
    checkpoint: str | Path | None,
    images: list[Path],
    stems: list[str],
    out_dir: str | Path,
    _runner,
) -> list[Path]:
    """Local-delivery execution: same stages as the remote path, all on this
    machine. The official checkpoint is md5-verified (delivery parity)."""
    from .install import verify_orgseg_checkpoint

    # Delivery parity: when the official released checkpoint is what runs,
    # hash-verify it before inference (custom fine-tuned weights carry their
    # own provenance and skip this check).
    if not (checkpoint and Path(checkpoint).is_file()):
        check = verify_orgseg_checkpoint(local["checkpoint"])
        if not check["ok"]:
            raise RuntimeError(
                f"OrgSegNet checkpoint verification failed: {check['reason']}: {local['checkpoint']}"
            )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="orgseg-local-"))
    try:
        uploads: dict[str, Path] = {f"{stem}.tif": image for stem, image in zip(stems, images)}
        used_checkpoint = local["checkpoint"]
        local_checkpoint = Path(checkpoint) if checkpoint else None
        if local_checkpoint is not None and local_checkpoint.is_file():
            from .install import ORGSEG_CHECKPOINT_NAME

            if local_checkpoint.name == ORGSEG_CHECKPOINT_NAME:
                check = verify_orgseg_checkpoint(local_checkpoint)
                if not check["ok"]:
                    raise RuntimeError(f"OrgSegNet checkpoint verification failed: {check['reason']}")
            used_checkpoint = str(local_checkpoint)
        for remote_name, src in uploads.items():
            target = work / remote_name
            shutil.copy2(src, target)
        (work / "manifest.json").write_text(json.dumps({"deliver": stems}), encoding="utf-8")
        cmd = [
            local["python"], local["script"],
            "--repo", local["repo"],
            "--config", local["config"],
            "--checkpoint", used_checkpoint,
            "--manifest", str(work / "manifest.json"),
            "--split", "deliver",
            "--image-dir", str(work),
            "--output", str(work / "out"),
            "--names", *stems,
        ]
        run = _runner(cmd, capture_output=True, text=True, timeout=3600)
        if run.returncode != 0:
            raise RuntimeError(
                f"local orgseg delivery failed (exit {run.returncode}): {(run.stdout or '')[-2000:]}"
            )
        import numpy as np
        from PIL import Image

        label_paths: list[Path] = []
        for stem in stems:
            class_map = np.array(Image.open(work / "out" / f"{stem}.class.tif"))
            if class_map.ndim != 2:
                raise RuntimeError(f"local orgseg label map for {stem} is not 2D")
            label = out / f"{stem}_label.png"
            Image.fromarray(class_map.astype("uint8")).save(str(label))
            label_paths.append(label)
        return label_paths
    finally:
        shutil.rmtree(work, ignore_errors=True)
