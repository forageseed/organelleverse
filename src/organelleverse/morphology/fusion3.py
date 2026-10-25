"""fusion3-v1: three-member fusion segmentation backend (remote executor).

The delivered ``fusion3-v1`` configuration (our multi-scale U-Net + official
OrgSegNet + fine-tuned Mask2Former, fused per class, then the shared
post-processing) is the best-validated 5-class plant-organelle segmentation
pipeline on the held-out OrgSegNet V6 test set (see
``_validation/2026-09-08-plant-sam/`` on the compute host). The delivery
fixed every deciding parameter — member checkpoint sha256 hashes, inference
scales, per-class fusion weights, post-processing — in
``experiments/plant-sam/delivery-fusion3-v1/delivery-config.json``; the entry
script ``scripts/plant-sam/deliver_fusion3.py`` verifies the hashes before it
runs anything and stops rather than fusing a partial result set.

The delivery lives on the GPU compute host (the two venvs it needs are
``data/orgsegnet/training-venv`` and ``data/orgsegnet/orgsegnet-baseline-venv``
inside the PhytoOrga_AI_c project tree). This backend therefore executes
remotely over SSH: images are copied up, ``deliver_fusion3.py`` runs there,
and the produced class maps are copied back and converted to the label-PNG
contract the rest of the morphology suite consumes.

Host and remote project root are configurable through the environment
(``ORGANELLEVERSE_FUSION3_HOST`` / ``ORGANELLEVERSE_FUSION3_ROOT``) so the
same code runs against a different compute box without edits.

Scope note (from the delivery README): the members were validated only on
OrgSegNet V6-corpus images; V6-trained members without fine-tuning found
41.7% of the lab's traced mitochondria. For the lab's own EM images the
ring-only lab fine-tuned model remains the tool — this backend is the
V6-corpus specialist.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from . import remote_copy

__all__ = [
    "DEFAULT_FUSION3_HOST",
    "DEFAULT_FUSION3_ROOT",
    "FUSION3_CLASS_NAMES",
    "FUSION3_CONFIG_REL",
    "FUSION3_DELIVERY_NAME",
    "FUSION3_ENTRY_REL",
    "check_fusion3_remote",
    "default_fusion3_executor",
    "fusion3_env",
]

#: Delivery identity, mirrored from delivery-config.json ``name``.
FUSION3_DELIVERY_NAME = "fusion3-v1"

#: Location of the delivery inside the remote project root.
FUSION3_CONFIG_REL = "experiments/plant-sam/delivery-fusion3-v1/delivery-config.json"
FUSION3_ENTRY_REL = "scripts/plant-sam/deliver_fusion3.py"

#: 5-class semantic layout shared by the whole delivery (uint8 label maps).
FUSION3_CLASS_NAMES = ("background", "chloroplast", "mitochondria", "vacuole", "nucleus")

DEFAULT_FUSION3_HOST = "forageseed-235"
# SSH commands resolve a relative deployment root in the remote account home.
DEFAULT_FUSION3_ROOT = "work/PhytoOrga_AI_c"

#: Remote sanity probe budget. A failed/slow host must not wedge the UI.
_SSH_PROBE_TIMEOUT_S = 15
#: Generous run budget: three model loads (~45 s) + ~12 s/image + copy time.
_RUN_TIMEOUT_S = 3600


def fusion3_env(environ: dict[str, str] | None = None) -> dict[str, str]:
    """Resolve (host, remote root) from the environment or the defaults."""
    env = os.environ if environ is None else environ
    return {
        "host": env.get("ORGANELLEVERSE_FUSION3_HOST", DEFAULT_FUSION3_HOST),
        "root": env.get("ORGANELLEVERSE_FUSION3_ROOT", DEFAULT_FUSION3_ROOT),
    }


def _ssh_base(host: str) -> list[str]:
    return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host]


def check_fusion3_remote(
    host: str | None = None,
    root: str | None = None,
    *,
    environ: dict[str, str] | None = None,
    _runner: Any = subprocess.run,
) -> dict[str, Any]:
    """Probe the compute host for a runnable fusion3 delivery.

    Checks, over one SSH invocation: the delivery config and entry script
    exist, and both interpreter venvs the delivery names are present. The
    member-weight sha256 verification itself is the entry script's job at
    run time (it refuses to run on a mismatch); probing four hashes here
    would add seconds to every readiness poll.

    Returns ``{name, installed, host, root, path, note}`` — the same shape
    :func:`organelleverse.morphology.install.check_backend` returns for the
    local backends, so status UIs can treat all backends uniformly.
    """
    cfg = fusion3_env(environ)
    host = host or cfg["host"]
    root = root or cfg["root"]
    probe = (
        f"test -f {root}/{FUSION3_CONFIG_REL}"
        f" && test -f {root}/{FUSION3_ENTRY_REL}"
        f" && test -x {root}/data/orgsegnet/training-venv/bin/python"
        f" && test -x {root}/data/orgsegnet/orgsegnet-baseline-venv/bin/python"
        f" && echo FUSION3_READY"
    )
    try:
        result = _runner(
            [*_ssh_base(host), probe],
            capture_output=True,
            text=True,
            timeout=_SSH_PROBE_TIMEOUT_S,
        )
        ready = result.returncode == 0 and "FUSION3_READY" in result.stdout
    except (OSError, subprocess.TimeoutExpired):
        ready = False
    return {
        "name": "fusion3",
        "installed": ready,
        "host": host,
        "root": root,
        "path": f"{host}:{root}" if ready else None,
        "note": (
            f"{FUSION3_DELIVERY_NAME} delivery at {host}:{root}"
            if ready
            else f"{FUSION3_DELIVERY_NAME} delivery not reachable at {host}:{root} "
            "(SSH alias, config, entry script, or venvs missing)"
        ),
    }


def default_fusion3_executor(
    image_paths: list[str],
    out_dir: str | Path,
    *,
    host: str | None = None,
    root: str | None = None,
    clean_remote: bool = True,
    environ: dict[str, str] | None = None,
    _runner: Any = subprocess.run,
) -> list[Path]:
    """Run the delivered fusion3 pipeline on the compute host.

    Copies the input images to a fresh remote work directory, runs
    ``deliver_fusion3.py`` there (hash-check on, work dir cleaned), copies
    the produced ``<name>.class.tif`` maps back, converts each to the
    label-PNG contract the OrgSegNet/micro-SAM executors return, and removes
    the remote work directory. Returns one label-map path per input image,
    in input order.

    The local ``device`` concept does not apply: the remote script picks GPU
    when present (peak ~1.4 GB) and the choice is recorded in its
    ``delivery-run.json``, not here.
    """
    cfg = fusion3_env(environ)
    host = host or cfg["host"]
    root = root or cfg["root"]
    images = [Path(p) for p in image_paths]
    for image in images:
        if not image.is_file():
            raise RuntimeError(f"fusion3 input image not found: {image}")
    if shutil.which("scp") is None or shutil.which("ssh") is None:
        raise RuntimeError("fusion3 remote executor needs ssh/scp on PATH")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    remote = _runner(
        [*_ssh_base(host), "mktemp -d /tmp/fusion3.XXXXXX"],
        capture_output=True, text=True, timeout=_SSH_PROBE_TIMEOUT_S,
    ).stdout.strip()
    if not remote:
        raise RuntimeError(f"fusion3 could not create a remote work dir on {host}")

    def _cleanup() -> None:
        if clean_remote and remote:
            try:
                _runner([*_ssh_base(host), f"rm -rf -- {remote}"],
                        capture_output=True, text=True, timeout=_SSH_PROBE_TIMEOUT_S)
            except Exception:
                pass

    try:
        remote_copy.upload(_runner, host, images, remote, timeout=600)
        names = [image.stem for image in images]
        cmd = [
            *_ssh_base(host),
            f"cd {root} && data/orgsegnet/training-venv/bin/python "
            f"{FUSION3_ENTRY_REL} --image-dir {remote} --output {remote}/out "
            f"--names {' '.join(names)} --clean",
        ]
        run = _runner(cmd, capture_output=True, text=True, timeout=_RUN_TIMEOUT_S)
        if run.returncode != 0:
            tail = (run.stdout or "")[-2000:]
            raise RuntimeError(
                f"fusion3 delivery failed on {host} (exit {run.returncode}): {tail}"
            )
        staged = Path(tempfile.mkdtemp(prefix="fusion3-"))
        remote_copy.download(
            _runner, host, [f"{remote}/out/{n}.class.tif" for n in names], staged, timeout=600,
        )
    except Exception:
        _cleanup()
        raise

    label_paths: list[Path] = []
    try:
        import numpy as np
        from PIL import Image

        for name in names:
            class_map = np.array(Image.open(staged / f"{name}.class.tif"))
            if class_map.ndim != 2:
                raise RuntimeError(f"fusion3 class map for {name} is not a 2D label map")
            label = out / f"{name}_label.png"
            Image.fromarray(class_map.astype("uint8")).save(str(label))
            label_paths.append(label)
    finally:
        _cleanup()
        shutil.rmtree(staged, ignore_errors=True)
    return label_paths
