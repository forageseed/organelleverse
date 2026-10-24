"""Backend installer + auto-locate for the OrgSegNet EM-segmentation tool.

OrgSegNet is a fork of MMSegmentation v1.0.0 and requires an old stack
(Python 3.8, torch 1.13.1, mmcv 2.0.0rc4, mmseg 1.0.0) that is not pip-
installable into modern environments. The official checkpoint is published
under CC BY 4.0; the application can fetch and verify it independently of
the separately distributed OrgSegNet source and Python environment.

The built-in :func:`.measure` morphometric quantifier needs none of this and
runs offline on any label map.
"""

from __future__ import annotations

import shutil
import os
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

from ..assembly.install import _list_conda_envs

__all__ = [
    "MORPHOLOGY_BACKEND_INFO",
    "ORGSEG_BACKEND_INFO",
    "ORGSEG_CHECKPOINT_MD5",
    "ORGSEG_CHECKPOINT_NAME",
    "ORGSEG_CHECKPOINT_SIZE",
    "orgseg_model_path",
    "download_orgseg_checkpoint",
    "check_all_backends",
    "check_backend",
    "install_hint",
    "verify_orgseg_checkpoint",
]

#: The released OrgSegNet checkpoint (Zenodo record 8419877), with the
#: publisher-provided size and md5. The Zenodo record licenses the weight
#: file CC BY 4.0; attribution to Feng et al. and the record is required.
ORGSEG_CHECKPOINT_NAME = "OrgSegNet_iter_Version1.pth"
ORGSEG_CHECKPOINT_SIZE = 489_381_591
ORGSEG_CHECKPOINT_MD5 = "9c91a0f5e1eb90beac7e320c4e1ae3e7"
ORGSEG_CHECKPOINT_URL = (
    "https://zenodo.org/api/records/8419877/files/"
    "OrgSegNet_iter_Version1.pth/content"
)
ORGSEG_ATTRIBUTION = (
    "OrgSegNet_iter_Version1.pth — Zeyu Yu (2023), Plantorganelle Hunter model. "
    "https://doi.org/10.5281/zenodo.8419877\n"
    "Checkpoint license: Creative Commons Attribution 4.0 International "
    "(CC BY 4.0). https://creativecommons.org/licenses/by/4.0/\n"
)


def orgseg_model_path() -> Path:
    """Managed model location, overrideable for installations and tests."""
    root = os.environ.get("ORGANELLEVERSE_MODEL_DIR")
    if root:
        base = Path(root)
    else:
        cache = os.environ.get("ORGANELLEVERSE_CACHE_ROOT")
        base = Path(cache) / "models" if cache else Path.home() / ".cache" / "organelleverse" / "models"
    return base / "orgsegnet" / ORGSEG_CHECKPOINT_NAME


def download_orgseg_checkpoint(destination: str | Path | None = None) -> Path:
    """Fetch the official CC BY 4.0 weight atomically and verify its MD5.

    A completed file is never replaced by a partial or mismatched download.
    The source code and its separate environment are not installed here.
    """
    target = Path(destination) if destination else orgseg_model_path()
    if verify_orgseg_checkpoint(target)["ok"]:
        (target.parent / "ATTRIBUTION.txt").write_text(ORGSEG_ATTRIBUTION, encoding="utf-8")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".orgsegnet-", suffix=".pth", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as output, urllib.request.urlopen(
            ORGSEG_CHECKPOINT_URL, timeout=60
        ) as response:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        result = verify_orgseg_checkpoint(temporary)
        if not result["ok"]:
            raise ValueError(f"OrgSegNet checkpoint verification failed: {result['reason']}")
        os.replace(temporary, target)
        (target.parent / "ATTRIBUTION.txt").write_text(ORGSEG_ATTRIBUTION, encoding="utf-8")
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return target


MORPHOLOGY_BACKEND_INFO: dict[str, dict[str, str]] = {
    "orgseg": {
        "url": "https://github.com/yzy0102/OrgSegNet",
        "cli": "tools/test.py",
        "probe": "mmseg",
        "note": (
            "OrgSegNet (Plantorganelle Hunter, Feng et al. Nat. Plants 2023) -- "
            "PSPNet-R50-d8 with a custom OrgSeg_Head decoder; 5 classes "
            "(background, Chloroplast, Mitochondria, Vacuole, Nucleus). It is a "
            "fork of MMSegmentation v1.0.0 and requires Python 3.8 + torch "
            "1.13.1 + mmcv 2.0.0rc4 + mmseg 1.0.0. After cloning, run "
            "`pip install -v -e .` in the repo to register the custom "
            "OrgSeg_Head / PlantCellDataset with mmseg. Checkpoint: "
            "OrgSegNet_iter_Version1.pth (Zenodo DOI 10.5281/zenodo.8419877, "
            "~489 MB, CC BY 4.0). The app downloads and verifies this weight; "
            "the OrgSegNet source and Python environment are separate."
        ),
        "tier": "manual",
    },
    "micro_sam": {
        "url": "https://github.com/computational-cell-analytics/micro-sam",
        # Real registered console script (micro-sam 1.8.9 entry_points):
        # micro_sam.automatic_segmentation = micro_sam.automatic_segmentation:main
        "cli": "micro_sam.automatic_segmentation",
        "probe": "micro_sam",
        "note": (
            "micro-SAM (uSAM, Archit et al. Nat. Methods 2025) -- Segment "
            "Anything fine-tuned on 17,000+ microscopy images, with dedicated "
            "EM organelle models (vit_*_em_organelles) and 3D-stack support. "
            "Installable into modern environments: `pip install micro-sam`. "
            "Batch inference entry point: `micro_sam.automatic_segmentation` "
            "(real console script); library API: micro_sam.instance_segmentation. "
            "Model weights download on first use via its own registry -- "
            "weights are NOT bundled into OrganelleVerse."
        ),
        "tier": "pip",
    },
    "fusion3": {
        "url": "internal delivery (experiments/plant-sam/delivery-fusion3-v1)",
        "cli": "scripts/plant-sam/deliver_fusion3.py",
        "probe": "fusion3-remote",
        "note": (
            "fusion3-v1 -- delivered three-member geometric fusion (our "
            "multi-scale U-Net + official OrgSegNet + fine-tuned Mask2Former) "
            "with fixed per-class fusion weights and shared post-processing; "
            "5 classes (background, chloroplast, mitochondria, vacuole, "
            "nucleus). Best-validated pipeline on the held-out OrgSegNet V6 "
            "test set. It runs remotely over SSH on the compute host where "
            "the delivery lives (see ORGANELLEVERSE_FUSION3_HOST / "
            "ORGANELLEVERSE_FUSION3_ROOT); member weights are sha256-verified "
            "by the entry script on every run. Validated on V6-corpus images "
            "only -- not a replacement for the lab fine-tuned model on the "
            "lab's own samples."
        ),
        "tier": "remote",
    },
}

#: Backward-compatible alias: the registry started OrgSegNet-only.
ORGSEG_BACKEND_INFO = MORPHOLOGY_BACKEND_INFO


def check_backend(name: str, *, scan_envs: bool = True) -> dict[str, Any]:
    """Check if a morphology segmentation backend is available.

    ``orgseg`` / ``micro_sam`` look for an importable probe module in the
    current env + all conda/micromamba envs. ``fusion3`` is different: it is
    a delivered pipeline that runs remotely over SSH, so availability means
    the delivery config, entry script, and both venvs are reachable on the
    compute host (see :mod:`organelleverse.morphology.fusion3`).

    Returns ``{name, installed, path, env, note}`` for the local backends and
    ``{name, installed, host, root, path, note}`` for fusion3.
    """
    if name == "fusion3":
        from .fusion3 import check_fusion3_remote

        return check_fusion3_remote()

    info = ORGSEG_BACKEND_INFO.get(name)
    if name == "orgseg":
        # Priority: a local pip mmseg stack, then the delivered pipeline on
        # THIS machine (no network), then the delivery host over SSH. Any of
        # the three counts as installed so callers can execute instead of
        # being shown an install guide.
        from .orgseg_remote import check_orgseg_local, check_orgseg_remote

        local_delivery = check_orgseg_local()
        if local_delivery["installed"]:
            return {
                "name": name,
                "installed": True,
                "path": local_delivery.get("path"),
                "env": "local-delivery",
                "note": f"{info.get('note', '')} [local delivery: {local_delivery.get('path')}]".strip(),
            }
        remote = check_orgseg_remote()
        if remote["installed"]:
            return {
                "name": name,
                "installed": True,
                "path": remote.get("path"),
                "env": "remote",
                "note": f"{info.get('note', '')} [remote: {remote.get('host')}]".strip(),
            }
    if not info:
        return {
            "name": name,
            "installed": False,
            "path": None,
            "env": None,
            "note": f"Unknown backend: {name}",
        }

    def _env_has_module(env_bin: Path | None, module: str) -> str | None:
        """Return the env's python path if `module` imports there, else None.

        The probe module comes from the backend registry's ``probe`` field
        (mmseg for orgseg, micro_sam for micro-sam) — probing the wrong
        module reported a freshly pip-installed micro-sam as missing.
        """
        import subprocess

        py = (env_bin / "python") if env_bin else shutil.which("python")
        if not py:
            return None
        try:
            r = subprocess.run(
                [str(py), "-c", f"import {module}; print({module}.__file__)"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if r.returncode == 0 and r.stdout.strip():
                return str(py)
        except Exception:
            pass
        return None

    probe_module = str(info.get("probe", "mmseg"))

    # Current env first — probed in-process (importlib): PATH's `python` may
    # be a different interpreter than the one running this service (venv).
    import importlib.util

    current_spec = importlib.util.find_spec(probe_module)
    if current_spec is not None and current_spec.origin:
        found = str(current_spec.origin)
    else:
        import sys

        found = _env_has_module(
            Path(sys.executable).parent if sys.executable else None, probe_module
        )
    if found:
        return {
            "name": name,
            "installed": True,
            "path": found,
            "env": None,
            "note": info.get("note", ""),
        }
    if scan_envs:
        for env_bin in _list_conda_envs():
            found = _env_has_module(env_bin, probe_module)
            if found:
                parts = env_bin.parts
                env_name = (
                    parts[parts.index("envs") + 1]
                    if "envs" in parts and parts.index("envs") + 1 < len(parts)
                    else None
                )
                note = f"{info.get('note', '')} [env: {env_name}]".strip()
                return {
                    "name": name,
                    "installed": True,
                    "path": found,
                    "env": env_name,
                    "note": note,
                }
    return {
        "name": name,
        "installed": False,
        "path": None,
        "env": None,
        "note": info.get("note", ""),
    }


def check_all_backends(*, scan_envs: bool = True) -> dict[str, dict[str, Any]]:
    """Check all morphology backends. Returns {name: status_dict}."""
    return {name: check_backend(name, scan_envs=scan_envs) for name in ORGSEG_BACKEND_INFO}


def verify_orgseg_checkpoint(path: str | Path) -> dict[str, Any]:
    """Verify a placed OrgSegNet checkpoint against the publisher's record.

    Only the official redistributed checkpoint (``ORGSEG_CHECKPOINT_NAME``)
    is hash-checked against the Zenodo record; user fine-tuned checkpoints
    carry their own provenance (T1 hashes them at train time) and are not
    publisher artifacts. Returns a status dict; ``ok`` is False with a
    machine-readable ``reason`` (``missing`` / ``size_mismatch`` /
    ``hash_mismatch``) when verification fails.
    """
    import hashlib

    candidate = Path(path)
    if not candidate.is_file():
        return {"ok": False, "reason": "missing", "path": str(candidate)}
    size = candidate.stat().st_size
    if size != ORGSEG_CHECKPOINT_SIZE:
        return {
            "ok": False,
            "reason": "size_mismatch",
            "path": str(candidate),
            "expected_size": ORGSEG_CHECKPOINT_SIZE,
            "actual_size": size,
        }
    digest = hashlib.md5()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != ORGSEG_CHECKPOINT_MD5:
        return {
            "ok": False,
            "reason": "hash_mismatch",
            "path": str(candidate),
            "expected_md5": ORGSEG_CHECKPOINT_MD5,
        }
    return {"ok": True, "reason": "", "path": str(candidate)}


def install_hint(name: str) -> str:
    """Return a human-readable install guide with URLs."""
    info = ORGSEG_BACKEND_INFO.get(name)
    if not info:
        return f"Unknown backend: {name}"
    url = info.get("url", "?")
    lines = [f"Backend '{name}' not found. Setup:"]
    lines.append(f"  repo: {url}")
    if name == "orgseg":
        lines.append("  env:  conda create -n orgseg python=3.8 -y && conda activate orgseg")
        lines.append(
            "  deps: pip install torch==1.13.1 torchvision==0.14.1 "
            "(+cu116 if GPU); install mmcv==2.0.0rc4 and mmseg==1.0.0"
        )
        lines.append("        (confirm platform-specific build steps in repo markdowns/PrepareEnvironment.md)")
        lines.append(
            "  code: git clone https://github.com/yzy0102/OrgSegNet && "
            "cd OrgSegNet && pip install -v -e ."
        )
        lines.append(
            "  weights: OrgSegNet_iter_Version1.pth from Zenodo "
            "DOI 10.5281/zenodo.8419877 (~489 MB)"
        )
        lines.append(
            "  run:  deeploc2-style: init_model(cfg, ckpt, device) -> inference_model(model, img)"
        )
        lines.append(
            "  license: official weight CC BY 4.0 (Zeyu Yu, Zenodo record 8419877); "
            "check the source repository separately for code terms"
        )
    elif name == "micro_sam":
        lines.append("  env:  pip install micro-sam  (modern stack; no conda pin needed)")
        lines.append(
            "  weights: EM organelle models (vit_*_em_organelles) download on "
            "first use via micro_sam's own registry"
        )
        lines.append("  batch entry: micro_sam.automatic_segmentation (console script)")
        lines.append("  license: MIT (code); model weights per micro-SAM release notes")
    elif name == "fusion3":
        lines.append("  remote: runs over SSH on the compute host (no local install)")
        lines.append(
            "  needs: delivery config + entry script + both venvs on the host; "
            "set ORGANELLEVERSE_FUSION3_HOST / ORGANELLEVERSE_FUSION3_ROOT to point elsewhere"
        )
        lines.append("  weights: member sha256 verified by the entry script every run")
        lines.append("  scope: OrgSegNet V6-corpus images; lab samples keep the lab fine-tuned model")
    if info.get("note"):
        lines.append(f"  note: {info['note']}")
    lines.append("  fallback: ov.morph.measure() runs offline on any label map")
    return "\n".join(lines)
