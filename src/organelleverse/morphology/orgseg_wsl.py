"""Run the pinned OrgSegNet delivery in WSL from a Windows desktop service.

This is a narrow execution adapter: the desktop remains the only state owner,
and images cross the boundary through the shared, per-run cache directory.
The launched argv is fixed and shell-free. Missing WSL dependencies fail
closed; an explicitly selected model is never replaced by another model.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

DEFAULT_DISTRIBUTION = "Ubuntu"
# An empty user selects the distribution's configured default account.
DEFAULT_USER = ""
DEFAULT_ROOT = ""
DEFAULT_PYTHON = ""
_PROBE_TIMEOUT = 20
_RUN_TIMEOUT = 3600


def _settings(
    environ: dict[str, str] | None = None, *, _runner: Any = subprocess.run,
) -> dict[str, str]:
    env = os.environ if environ is None else environ
    config = {
        "distribution": env.get("ORGANELLEVERSE_WSL_DISTRIBUTION", DEFAULT_DISTRIBUTION),
        "user": env.get("ORGANELLEVERSE_WSL_USER", DEFAULT_USER),
        "root": env.get("ORGANELLEVERSE_WSL_ORGSEG_ROOT", DEFAULT_ROOT),
        "python": env.get("ORGANELLEVERSE_WSL_ORGSEG_PYTHON", DEFAULT_PYTHON),
    }

    if not config["root"] or not config["python"]:
        home = _runner(
            [*_prefix(config), "python3", "-c", "from pathlib import Path; print(Path.home())"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=_PROBE_TIMEOUT,
        )
        remote_home = home.stdout.strip()
        if home.returncode != 0 or not PurePosixPath(remote_home).is_absolute():
            raise RuntimeError("Cannot resolve the selected WSL account home; configure the OrgSegNet root and interpreter explicitly")
        config["root"] = config["root"] or str(PurePosixPath(remote_home) / "organelleverse-orgseg-local")
        config["python"] = config["python"] or str(PurePosixPath(remote_home) / "orgsegnet-local-venv/bin/python")
    return config


def _prefix(config: dict[str, str]) -> list[str]:
    prefix = ["wsl.exe", "--distribution", config["distribution"]]
    if config["user"]:
        prefix.extend(["--user", config["user"]])
    return [*prefix, "--exec"]


def check_orgseg_wsl(*, environ: dict[str, str] | None = None, _runner: Any = subprocess.run) -> dict[str, Any]:
    """Probe the exact interpreter, script, config and checkpoint used by runs."""
    if sys.platform != "win32":
        return {"installed": False, "path": None, "note": "WSL execution needs Windows"}
    try:
        cfg = _settings(environ, _runner=_runner)
    except (OSError, subprocess.TimeoutExpired, RuntimeError) as error:
        return {"installed": False, "path": None, "note": str(error)}
    root = cfg["root"]
    paths = [
        f"{root}/scripts/plant-sam/predict_orgsegnet.py",
        f"{root}/OrgSegNet/configs/OrgsegNet_resolution/OrgSeg_PlantCell_768x512.py",
        f"{root}/OrgSegNet_iter_Version1.pth",
        f"{root}/OrgSegNet",
    ]
    code = (
        "import pathlib,sys; "
        "ok=all(pathlib.Path(p).exists() for p in sys.argv[1:]); "
        "import torch,mmengine,tifffile; "
        "print('ORGSEG_WSL_READY' if ok else 'ORGSEG_WSL_MISSING')"
    )
    try:
        result = _runner(
            [*_prefix(cfg), cfg["python"], "-c", code, *paths],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=_PROBE_TIMEOUT,
        )
        ready = result.returncode == 0 and "ORGSEG_WSL_READY" in result.stdout
    except (OSError, subprocess.TimeoutExpired):
        ready = False
    return {
        "installed": ready,
        "path": f"WSL:{cfg['distribution']}:{root}" if ready else None,
        "note": "OrgSegNet WSL environment ready" if ready else "OrgSegNet WSL environment incomplete",
    }


def _wsl_path(path: Path, cfg: dict[str, str], runner: Any) -> str:
    result = runner(
        [*_prefix(cfg), "wslpath", "-a", str(path.resolve())],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=_PROBE_TIMEOUT,
    )
    if result.returncode != 0 or not result.stdout.strip().startswith("/"):
        raise RuntimeError(f"WSL cannot access staged path {path}: {(result.stderr or '')[-500:]}")
    return result.stdout.strip()


def default_orgseg_wsl_executor(
    image_paths: list[str], out_dir: str | Path, *,
    environ: dict[str, str] | None = None, _runner: Any = subprocess.run,
) -> list[Path]:
    """Infer in WSL and return Windows-local uint8 label PNGs in input order."""
    if sys.platform != "win32":
        raise RuntimeError("OrgSegNet WSL executor is only available from Windows")
    from PIL import Image
    import numpy as np

    cfg = _settings(environ, _runner=_runner)
    env = os.environ if environ is None else environ
    cache = Path(env.get("ORGANELLEVERSE_CACHE_ROOT", r"D:\organelleverse-data\cache"))
    cache.mkdir(parents=True, exist_ok=True)
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = [Path(p) for p in image_paths]
    if not paths or any(not p.is_file() for p in paths):
        raise RuntimeError("OrgSegNet WSL input image missing")

    labels: list[Path] = []
    with tempfile.TemporaryDirectory(prefix="orgseg-wsl-", dir=cache) as temporary:
        work = Path(temporary)
        image_dir = work / "images"
        result_dir = work / "results"
        image_dir.mkdir()
        result_dir.mkdir()
        names = [f"image-{index:04d}" for index in range(len(paths))]
        for source, name in zip(paths, names, strict=True):
            with Image.open(source) as image:
                image.save(image_dir / f"{name}.tif")
        manifest = work / "manifest.json"
        manifest.write_text(json.dumps({"run": names}), encoding="utf-8")
        wsl_work = _wsl_path(work, cfg, _runner)
        root = cfg["root"]
        argv = [
            *_prefix(cfg), cfg["python"], f"{root}/scripts/plant-sam/predict_orgsegnet.py",
            "--repo", f"{root}/OrgSegNet",
            "--config", f"{root}/OrgSegNet/configs/OrgsegNet_resolution/OrgSeg_PlantCell_768x512.py",
            "--checkpoint", f"{root}/OrgSegNet_iter_Version1.pth",
            "--manifest", f"{wsl_work}/manifest.json",
            "--split", "run", "--image-dir", f"{wsl_work}/images",
            "--output", f"{wsl_work}/results", "--names", *names,
            "--device", "auto",
        ]
        result = _runner(argv, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=_RUN_TIMEOUT)
        if result.returncode != 0:
            raise RuntimeError(f"OrgSegNet WSL inference failed: {(result.stderr or result.stdout)[-2000:]}")
        import tifffile

        for source, name in zip(paths, names, strict=True):
            class_map = np.asarray(tifffile.imread(result_dir / f"{name}.class.tif"))
            if class_map.ndim != 2 or not np.isin(class_map, [0, 1, 2, 3, 4]).all():
                raise RuntimeError(f"OrgSegNet WSL returned invalid class map for {source.name}")
            destination = output / f"{source.stem}_label.png"
            Image.fromarray(class_map.astype("uint8")).save(destination)
            labels.append(destination)
    return labels
