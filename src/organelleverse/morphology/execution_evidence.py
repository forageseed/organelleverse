"""Capture reproducible runtime evidence for an executed morphology result."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from .fusion3 import FUSION3_CONFIG_REL, FUSION3_ENTRY_REL, fusion3_env, _ssh_base
from .orgseg_remote import (
    ORGSEG_REMOTE_CHECKPOINT,
    ORGSEG_REMOTE_CONFIG,
    ORGSEG_REMOTE_PYTHON,
    ORGSEG_REMOTE_SCRIPT,
)
from .orgseg_wsl import _prefix, _settings

_SNAPSHOT_CODE = """
import hashlib, importlib, importlib.metadata, json, platform, sys
from pathlib import Path

if len(sys.argv) > 4 and sys.argv[4] != '-':
    sys.path.insert(0, sys.argv[4])

def digest(path):
    file = Path(path)
    if not file.is_file(): return None
    h = hashlib.sha256()
    with file.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

packages = {}
for name in ('torch', 'torchvision', 'mmseg', 'mmengine', 'mmcv', 'numpy', 'Pillow', 'tifffile'):
    try: packages[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        try: packages[name] = getattr(importlib.import_module(name), '__version__', None)
        except Exception: packages[name] = None
try:
    import torch
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
except Exception:
    device = 'unknown'
print(json.dumps({
    'hostname': platform.node(), 'system': platform.system(),
    'python': sys.version.split()[0], 'packages': packages,
    'device': device, 'model_sha256': digest(sys.argv[1]),
    'config_sha256': digest(sys.argv[2]), 'script_sha256': digest(sys.argv[3]),
}, sort_keys=True))
"""


def capture_orgseg_environment(site: str, *, ssh_host: str | None = None) -> dict[str, Any]:
    """Read exact runtime package versions and file hashes from the chosen host."""
    if site == "wsl":
        cfg = _settings()
        root = cfg["root"]
        argv = [
            *_prefix(cfg), cfg["python"], "-c", _SNAPSHOT_CODE,
            f"{root}/OrgSegNet_iter_Version1.pth",
            f"{root}/OrgSegNet/configs/OrgsegNet_resolution/OrgSeg_PlantCell_768x512.py",
            f"{root}/scripts/plant-sam/predict_orgsegnet.py",
            f"{root}/OrgSegNet",
        ]
    elif site == "ssh":
        cfg = fusion3_env()
        root = cfg["root"]
        host = ssh_host or cfg["host"]
        parts = [
            f"{root}/{ORGSEG_REMOTE_PYTHON}", "-c", _SNAPSHOT_CODE,
            f"{root}/{ORGSEG_REMOTE_CHECKPOINT}",
            f"{root}/{ORGSEG_REMOTE_CONFIG}",
            f"{root}/{ORGSEG_REMOTE_SCRIPT}",
            f"{root}/OrgSegNet",
        ]
        argv = [*_ssh_base(host), " ".join(shlex.quote(part) for part in parts)]
    elif site == "local":
        return {
            "hostname": os.environ.get("COMPUTERNAME") or os.environ.get("HOSTNAME"),
            "system": sys.platform, "python": sys.version.split()[0],
            "packages": {}, "device": "local", "model_sha256": None,
            "config_sha256": None, "script_sha256": None,
        }
    else:
        raise ValueError(f"unsupported execution site: {site}")
    result = subprocess.run(
        argv, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"runtime evidence probe failed at {site}: {(result.stderr or result.stdout)[-500:]}")
    return json.loads(result.stdout.strip().splitlines()[-1])


def capture_fusion3_environment(*, ssh_host: str | None = None) -> dict[str, Any]:
    cfg = fusion3_env()
    root = cfg["root"]
    host = ssh_host or cfg["host"]
    parts = [
        f"{root}/data/orgsegnet/training-venv/bin/python", "-c", _SNAPSHOT_CODE,
        "-", f"{root}/{FUSION3_CONFIG_REL}", f"{root}/{FUSION3_ENTRY_REL}",
    ]
    result = subprocess.run(
        [*_ssh_base(host), " ".join(shlex.quote(part) for part in parts)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"fusion3 runtime evidence probe failed: {(result.stderr or result.stdout)[-500:]}")
    return json.loads(result.stdout.strip().splitlines()[-1])
