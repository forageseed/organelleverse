"""Backend installer + auto-locate for the OrgSegNet EM-segmentation tool.

OrgSegNet is a fork of MMSegmentation v1.0.0 and requires an old stack
(Python 3.8, torch 1.13.1, mmcv 2.0.0rc4, mmseg 1.0.0) that is not pip-
installable into modern environments. Its trained checkpoint is released under
an academic license (non-commercial, no redistribution), so neither the code
nor the weights are bundled here -- this module auto-locates an installed
OrgSegNet repo / mmseg and gives the real setup guide when absent.

The built-in :func:`.measure` morphometric quantifier needs none of this and
runs offline on any label map.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from ..assembly.install import _list_conda_envs

__all__ = [
    "MORPHOLOGY_BACKEND_INFO",
    "ORGSEG_BACKEND_INFO",
    "ORGSEG_CHECKPOINT_MD5",
    "ORGSEG_CHECKPOINT_NAME",
    "ORGSEG_CHECKPOINT_SIZE",
    "check_all_backends",
    "check_backend",
    "install_hint",
    "verify_orgseg_checkpoint",
]

#: The released OrgSegNet checkpoint (Zenodo record 8419877), with the
#: publisher-provided size and md5 (Zenodo publishes md5). The license is
#: academic non-redistributable: the app guides the user to the official
#: page and verifies what they placed — it never downloads it for them.
ORGSEG_CHECKPOINT_NAME = "OrgSegNet_iter_Version1.pth"
ORGSEG_CHECKPOINT_SIZE = 489_381_591
ORGSEG_CHECKPOINT_MD5 = "9c91a0f5e1eb90beac7e320c4e1ae3e7"


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
            "~489 MB). Academic license (non-commercial, no redistribution) -- "
            "weights/code are NOT bundled into OrganelleVerse."
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
}

#: Backward-compatible alias: the registry started OrgSegNet-only.
ORGSEG_BACKEND_INFO = MORPHOLOGY_BACKEND_INFO


def check_backend(name: str, *, scan_envs: bool = True) -> dict[str, Any]:
    """Check if the OrgSegNet backend is available.

    Looks for an importable ``mmseg`` (with the OrgSegNet custom registrations)
    in the current env + all conda/micromamba envs. Returns
    {name, installed, path, env, note}.

    Note: importing mmseg only confirms the framework; whether the OrgSegNet
    custom head is registered depends on the repo being pip-installed. The
    caller must pass the correct ``config`` / ``checkpoint`` to :func:`.segment`.
    """
    info = ORGSEG_BACKEND_INFO.get(name)
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
            "(+cu116 if GPU); pip install mmcv-full==1.7.0 mmseg==0.30.0"
        )
        lines.append("        (or the exact pins in repo markdowns/PrepareEnvironment.md)")
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
            "  license: academic (non-commercial, no redistribution -- "
            "weights/code NOT bundled here)"
        )
    elif name == "micro_sam":
        lines.append("  env:  pip install micro-sam  (modern stack; no conda pin needed)")
        lines.append(
            "  weights: EM organelle models (vit_*_em_organelles) download on "
            "first use via micro_sam's own registry"
        )
        lines.append("  batch entry: micro_sam.automatic_segmentation (console script)")
        lines.append("  license: MIT (code); model weights per micro-SAM release notes")
    if info.get("note"):
        lines.append(f"  note: {info['note']}")
    lines.append("  fallback: ov.morph.measure() runs offline on any label map")
    return "\n".join(lines)
