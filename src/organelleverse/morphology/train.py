"""OrgSegNet fine-tuning wrapper.

OrgSegNet (Plantorganelle Hunter, Feng et al. *Nat. Plants* 2023) is trained
and fine-tuned via the MMSegmentation / MMEngine programmatic API rather than a
documented CLI flag for ``--load-from``. The canonical flow (from the repo's
``demo/Fine-tune_OrgSegNet_demo.ipynb``) is::

    from mmengine import Config
    from mmengine.runner import Runner

    cfg = Config.fromfile("configs/OrgSegNet/OrgSeg_PlantCell_768x512.py")
    cfg.load_from = "../checkpoints/OrgSegNet_iter_Version1.pth"
    cfg.train_dataloader.dataset.data_root = "../FinetuneDataset001"
    cfg.train_dataloader.dataset.data_prefix = dict(img_path="image", seg_map_path="label")
    cfg.train_dataloader.dataset.ann_file = "splits/train.txt"
    # ... val/test dataloaders, work_dir, max_iters, batch_size ...
    runner = Runner.from_cfg(cfg)
    runner.train()

This module wraps that flow: it validates the fine-tune dataset layout,
constructs the config-diff, and either runs training via an executor hook
(callable that does the ``Config.fromfile -> mutate -> Runner.from_cfg ->
runner.train()`` dance inside the user's ``orgseg`` conda env) or returns a
plan with a paste-ready code snippet when no executor is supplied.

Dataset layout expected (PrepareData.md)::

    <data_root>/
      image/   img1.tif ...        # img_suffix=.tif
      label/   img1.png ...        # seg_map_suffix=.png, pixel values 0-4
      splits/
        train.txt  val.txt  test.txt   # bare stems, one per line

References
----------
- Feng, X. et al. (2023) *Plantorganelle Hunter.* Nat. Plants 9:1705-1717.
- OrgSegNet repo: https://github.com/yzy0102/OrgSegNet
- Fine-tune demo: demo/Fine-tune_OrgSegNet_demo.ipynb
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..core import ArtifactRef, ErrorDetail, Finding, OrganelleResult
from ._contract import (
    MORPHOLOGY_OPERATION_VERSION,
    MORPHOLOGY_SCOPE,
    collect_artifacts,
    make_provenance,
    parameters_hash,
    utc_now,
)

__all__ = ["train", "training_code_snippet", "validate_dataset"]

OPERATION_ID = "morphology.train"

# Default fine-tune config (PSPNet-R50-d8 + OrgSeg_Head, 512x512 slide window).
DEFAULT_TRAIN_CONFIG = "configs/OrgSegNet/OrgSeg_PlantCell_768x512.py"
DEFAULT_CHECKPOINT = "OrgSegNet_iter_Version1.pth"
ZENODO_URL = "https://zenodo.org/records/8419877"


def validate_dataset(data_root: str | Path) -> dict[str, Any]:
    """Validate an OrgSegNet fine-tune dataset directory layout.

    Checks for ``image/``, ``label/``, ``splits/{train,val,test}.txt`` and
    that image/label stems match. Returns a dict with ``valid`` (bool),
    ``n_train``, ``n_val``, ``n_test``, and ``problems`` (list of issues).
    """
    root = Path(data_root)
    img_dir = root / "image"
    lab_dir = root / "label"
    splits_dir = root / "splits"
    problems: list[str] = []

    if not root.is_dir():
        problems.append(f"data_root does not exist: {root}")
        return {
            "valid": False,
            "n_train": 0,
            "n_val": 0,
            "n_test": 0,
            "problems": problems,
            "data_root": str(root),
        }
    for d, name in ((img_dir, "image"), (lab_dir, "label"), (splits_dir, "splits")):
        if not d.is_dir():
            problems.append(f"missing directory: {name}/")

    n_train = n_val = n_test = 0
    for split in ("train", "val", "test"):
        f = splits_dir / f"{split}.txt"
        if not f.exists():
            problems.append(f"missing splits/{split}.txt")
            continue
        stems = [ln.strip() for ln in f.read_text().splitlines() if ln.strip()]
        if split == "train":
            n_train = len(stems)
        elif split == "val":
            n_val = len(stems)
        else:
            n_test = len(stems)

    # Check image/label stem pairing (if both dirs exist).
    if img_dir.is_dir() and lab_dir.is_dir():
        img_stems = {
            p.stem
            for p in img_dir.iterdir()
            if p.suffix.lower() in (".tif", ".tiff", ".png", ".jpg", ".jpeg")
        }
        lab_stems = {p.stem for p in lab_dir.iterdir() if p.suffix.lower() == ".png"}
        missing_labels = img_stems - lab_stems
        if missing_labels:
            problems.append(
                f"{len(missing_labels)} image(s) without a matching label/*.png "
                f"(e.g. {sorted(missing_labels)[:3]})"
            )

    return {
        "valid": not problems,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "problems": problems,
        "data_root": str(root),
    }


def training_code_snippet(
    data_root: str | Path,
    *,
    config: str = DEFAULT_TRAIN_CONFIG,
    checkpoint: str = DEFAULT_CHECKPOINT,
    max_iters: int = 40000,
    batch_size: int = 2,
) -> str:
    """Return a training template with a fixed, editable run directory."""
    return _training_code_snippet(
        data_root,
        config=config,
        checkpoint=checkpoint,
        work_dir="work_dirs/finetune",
        max_iters=max_iters,
        batch_size=batch_size,
    )


def _training_code_snippet(
    data_root: str | Path,
    *,
    config: str,
    checkpoint: str,
    work_dir: str,
    max_iters: int,
    batch_size: int,
) -> str:
    """Return a paste-ready Python snippet that fine-tunes OrgSegNet.

    This mirrors the canonical ``Fine-tune_OrgSegNet_demo.ipynb`` flow:
    ``Config.fromfile`` -> mutate fields -> ``Runner.from_cfg`` -> ``train()``.
    """
    return f'''# OrgSegNet fine-tuning (Plantorganelle Hunter, Nat. Plants 2023)
# Run inside the orgseg conda env (py3.8 + torch1.13 + mmcv2.0rc4 + mmseg1.0)
# after `pip install -v -e .` in the OrgSegNet repo.
from mmengine import Config
from mmengine.runner import Runner

cfg = Config.fromfile("{config}")
cfg.load_from = "{checkpoint}"               # pretrained OrgSegNet weights
cfg.work_dir = "{work_dir}"

DATA_ROOT = "{data_root}"
cfg.train_dataloader.dataset.data_root = DATA_ROOT
cfg.train_dataloader.dataset.data_prefix = dict(img_path="image", seg_map_path="label")
cfg.train_dataloader.dataset.ann_file = "splits/train.txt"
cfg.val_dataloader.dataset.data_root = DATA_ROOT
cfg.val_dataloader.dataset.data_prefix = dict(img_path="image", seg_map_path="label")
cfg.val_dataloader.dataset.ann_file = "splits/val.txt"
cfg.test_dataloader.dataset.data_root = DATA_ROOT
cfg.test_dataloader.dataset.data_prefix = dict(img_path="image", seg_map_path="label")
cfg.test_dataloader.dataset.ann_file = "splits/test.txt"

cfg.train_cfg.max_iters = {max_iters}
cfg.train_cfg.val_interval = {max_iters // 10}
cfg.train_cfg.val_begin = 1000
cfg.train_cfg.dynamic_intervals = None
cfg.default_hooks.checkpoint.interval = 2000
cfg.default_hooks.checkpoint.max_keep_ckpts = 3
# single-GPU: keep BatchNorm (do NOT swap to SyncBN)
cfg.param_scheduler = [dict(type="PolyLR", power=0.9, end=0, eta_min=1e-4,
                            by_epoch=False, begin=0, end={max_iters})]
# batch size
for dl in (cfg.train_dataloader, cfg.val_dataloader, cfg.test_dataloader):
    dl.batch_size = {batch_size}
cfg["randomness"] = dict(seed=0)

runner = Runner.from_cfg(cfg)
runner.train()
# Best checkpoint -> cfg.work_dir / "best_mIoU_iter_*.pth"
'''


_BEST_CKPT_PATTERN = re.compile(r"^best_mIoU_iter_(?P<iteration>\d+)\.pth$")


def _resolve_best_checkpoint(work_dir: Path) -> tuple[Path | None, str]:
    """Deterministically resolve the fine-tune run's best checkpoint.

    mmengine's ``CheckpointHook(save_best=\"mIoU\")`` writes
    ``best_mIoU_iter_<N>.pth`` under ``cfg.work_dir`` only when the metric
    improves, and training iterations increase monotonically, so the highest
    iteration suffix is the final best checkpoint. Returns ``(path, "")`` on
    a determinate resolution, or ``(None, reason)`` with a machine-readable
    reason when the produced checkpoint cannot be addressed honestly:
    ``"no_best_checkpoint"`` when nothing matches, or
    ``"ambiguous_best_checkpoint"`` when a match's iteration cannot be parsed
    (a name this wrapper cannot interpret must never be guessed at).
    """
    candidates: list[tuple[int, Path]] = []
    for path in sorted(work_dir.glob("best_mIoU_iter_*.pth")):
        match = _BEST_CKPT_PATTERN.match(path.name)
        if match is None:
            return None, "ambiguous_best_checkpoint"
        candidates.append((int(match.group("iteration")), path))
    if not candidates:
        return None, "no_best_checkpoint"
    return max(candidates)[1], ""


def train(
    data_root: str | Path,
    *,
    executor: Callable[[dict[str, Any]], None] | None = None,
    config: str | Path | None = None,
    load_from: str | Path | None = None,
    work_dir: str | Path | None = None,
    max_iters: int = 40000,
    batch_size: int = 2,
    device: str = "auto",
    orgseg_root: str | Path | None = None,
) -> OrganelleResult:
    """Fine-tune OrgSegNet on a custom plant-EM dataset.

    Parameters
    ----------
    data_root
        Dataset root containing ``image/``, ``label/``, ``splits/``.
    executor
        Callable ``executor(config_diff)`` that performs the actual training
        inside the orgseg env (``Config.fromfile -> mutate -> Runner.from_cfg
        -> runner.train()``). When ``None`` (default), returns a plan with a
        paste-ready code snippet.
    config
        Base config path. Defaults to
        ``configs/OrgSegNet/OrgSeg_PlantCell_768x512.py``.
    load_from
        Pretrained ``.pth`` checkpoint. Defaults to
        ``OrgSegNet_iter_Version1.pth`` (Zenodo 10.5281/zenodo.8419877).
    work_dir
        Output directory for logs/checkpoints.
    max_iters
        Training iterations (default 40000; official full train is 160000).
    batch_size
        Per-GPU batch size.
    device
        Compute device. ``"auto"`` (default) resolves deterministically via
        :mod:`.device`; fine-tuning fails closed unless the resolved device
        is a discrete CUDA device meeting the declared VRAM floor.
    orgseg_root
        Path to the OrgSegNet repo (resolves default config/checkpoint paths).

    Returns
    -------
    OrganelleResult
        Canonical immutable result, ``operation_id="morphology.train"``,
        ``scope="mixed"``. Status is ``"ok"`` when the executor completed
        **and** the produced best checkpoint could be resolved
        deterministically, ``"warning"`` when only a plan could be produced
        (no executor / backend absent / dataset invalid), and ``"failed"``
        when the executor raised, when the ``load_from`` start checkpoint is
        not addressable, or when the produced checkpoint is missing or
        ambiguous. On success the resolved ``best_mIoU_iter_*.pth`` is
        attached as a content-addressed artifact and both endpoint hashes
        (start ``load_from`` and produced best checkpoint) are recorded in
        ``provenance.model_hashes``.
    """
    from .install import check_backend, install_hint

    started_at = utc_now()

    cfg_path = Path(config) if config else Path(orgseg_root or ".") / DEFAULT_TRAIN_CONFIG
    ckpt = Path(load_from) if load_from else Path(orgseg_root or ".") / DEFAULT_CHECKPOINT
    wdir = Path(work_dir) if work_dir else Path("work_dirs") / "finetune"

    ds = validate_dataset(data_root)
    warning = ""
    if not ds["valid"]:
        warning = "\n  ⚠ dataset issues:\n    " + "\n    ".join(ds["problems"])

    loc = check_backend("orgseg", scan_envs=True)
    backend_found = bool(loc.get("installed"))
    if not backend_found:
        warning += "\n  ⚠ " + install_hint("orgseg")

    config_diff = {
        "load_from": str(ckpt),
        "work_dir": str(wdir),
        "data_root": str(data_root),
        "data_prefix": dict(img_path="image", seg_map_path="label"),
        "ann_files": dict(train="splits/train.txt", val="splits/val.txt", test="splits/test.txt"),
        "max_iters": max_iters,
        "batch_size": batch_size,
        "device": device,
        "config": str(cfg_path),
    }
    snippet = _training_code_snippet(
        data_root,
        config=str(cfg_path),
        checkpoint=str(ckpt),
        work_dir=str(wdir),
        max_iters=max_iters,
        batch_size=batch_size,
    )
    # CLI alternative (mmseg tools/train.py with --cfg-options).
    cli_argv = [
        "python",
        "tools/train.py",
        str(cfg_path),
        "--work-dir",
        str(wdir),
        "--cfg-options",
        f"load_from={ckpt}",
        f"train_dataloader.dataset.data_root={data_root}",
        "train_dataloader.dataset.ann_file=splits/train.txt",
        f"train_cfg.max_iters={max_iters}",
    ]

    params_hash = parameters_hash(
        {
            "config": str(cfg_path),
            "load_from": str(ckpt),
            "data_root": str(data_root),
            "max_iters": max_iters,
            "batch_size": batch_size,
            "device": device,
            "orgseg_root": str(orgseg_root) if orgseg_root else None,
        }
    )

    # ── Plan-only path ───────────────────────────────────────────────
    if executor is None or not backend_found or not ds["valid"]:
        plan = {
            "config": str(cfg_path),
            "checkpoint": str(ckpt),
            "work_dir": str(wdir),
            "config_diff": config_diff,
            "cli_argv": cli_argv,
            "code_snippet": snippet,
            "dataset": ds,
            "orgseg_root": str(orgseg_root) if orgseg_root else None,
        }
        reason = (
            "no executor"
            if executor is None
            else "tool not installed"
            if not backend_found
            else "dataset invalid"
        )
        return OrganelleResult(
            operation_id=OPERATION_ID,
            operation_version=MORPHOLOGY_OPERATION_VERSION,
            scope=MORPHOLOGY_SCOPE,
            # Nothing was trained: a plan is not an "ok" training run.
            status="warning",
            summary_text=(
                f"OrgSegNet fine-tune planned for "
                f"{ds['n_train']} train / {ds['n_val']} val / "
                f"{ds['n_test']} test images (not run: {reason})." + warning
            ),
            metrics={
                "data_root": str(data_root),
                "backend_found": backend_found,
                "ran": False,
                "n_train": ds["n_train"],
                "n_val": ds["n_val"],
                "n_test": ds["n_test"],
                "plan": plan,
            },
            findings=(
                Finding(code="morphology.train_plan", metric="reason", value=reason),
                Finding(code="morphology.train_plan", metric="config", value=str(cfg_path)),
            ),
            flags=("morph_train_planned", "backend_orgseg"),
            provenance=make_provenance(
                OPERATION_ID,
                params_hash=params_hash,
                argv=tuple(cli_argv),
                attempted_backends=("orgseg",),
                started_at=started_at,
                finished_at=utc_now(),
            ),
        )

    # ── Run path ─────────────────────────────────────────────────────
    # Device gate before anything runs: training declares its feasibility
    # floor up front (T5) — an integrated/CPU device is rejected here, not
    # after 40 minutes of OOM.
    from .device import resolve_device

    resolution = resolve_device(device, backend="orgseg", purpose="train")
    if not resolution.ok:
        return OrganelleResult(
            operation_id=OPERATION_ID,
            operation_version=MORPHOLOGY_OPERATION_VERSION,
            scope=MORPHOLOGY_SCOPE,
            status="failed",
            summary_text=f"OrgSegNet fine-tune refused: {resolution.error_message}",
            metrics={
                "data_root": str(data_root),
                "ran": False,
                "requested_device": device,
            },
            flags=("morph_train_failed", "backend_orgseg"),
            provenance=make_provenance(
                OPERATION_ID,
                params_hash=params_hash,
                argv=tuple(cli_argv),
                attempted_backends=("orgseg",),
                started_at=started_at,
                finished_at=utc_now(),
            ),
            errors=(
                ErrorDetail(
                    code=resolution.error_code or "morphology.train_device_infeasible",
                    message=resolution.error_message or "device resolution failed",
                    details={
                        "requested_device": device,
                        "backend": "orgseg",
                        "feasible_devices": list(resolution.feasible),
                    },
                    retryable=False,
                ),
            ),
        )
    assert resolution.device is not None
    torch_device = resolution.device.torch_device
    device_record = resolution.device.provenance_record()
    config_diff["device"] = torch_device

    # Provenance contract: both endpoints of the fine-tune must be
    # content-addressable. The start checkpoint is hashed *before* the
    # executor runs (it is the run's input state); a start point that does
    # not exist on disk can never be addressed, so the run refuses to start
    # rather than producing an unverifiable "success".
    if not ckpt.is_file():
        message = f"load_from checkpoint does not exist: {ckpt}"
        return OrganelleResult(
            operation_id=OPERATION_ID,
            operation_version=MORPHOLOGY_OPERATION_VERSION,
            scope=MORPHOLOGY_SCOPE,
            status="failed",
            summary_text=f"OrgSegNet fine-tune refused: {message}",
            metrics={
                "data_root": str(data_root),
                "ran": False,
                "load_from": str(ckpt),
            },
            flags=("morph_train_failed", "backend_orgseg"),
            provenance=make_provenance(
                OPERATION_ID,
                params_hash=params_hash,
                argv=tuple(cli_argv),
                actual_backend="orgseg",
                attempted_backends=("orgseg",),
                started_at=started_at,
                finished_at=utc_now(),
            ),
            errors=(
                ErrorDetail(
                    code="morphology.train_load_from_missing",
                    message=message,
                    details={"load_from": str(ckpt), "work_dir": str(wdir)},
                    retryable=False,
                ),
            ),
        )
    load_from_ref = ArtifactRef.from_path(ckpt, kind="pretrained_checkpoint", format="pth")

    run_err = ""
    try:
        executor(config_diff)
        ran = True
    except Exception as exc:
        run_err = str(exc)
        ran = False

    best_ckpt: Path | None = None
    resolve_reason = ""
    if ran:
        best_ckpt, resolve_reason = _resolve_best_checkpoint(wdir)
        if best_ckpt is None:
            ran = False
            run_err = (
                f"training executor completed but the produced best checkpoint "
                f"could not be resolved under {wdir} ({resolve_reason}); refusing "
                f"to report an unaddressable training run as success"
            )

    if not ran:
        return OrganelleResult(
            operation_id=OPERATION_ID,
            operation_version=MORPHOLOGY_OPERATION_VERSION,
            scope=MORPHOLOGY_SCOPE,
            status="failed",
            summary_text=f"OrgSegNet fine-tune failed: {run_err}",
            metrics={
                "data_root": str(data_root),
                "ran": False,
                "error": run_err,
                "plan": {"code_snippet": snippet},
            },
            flags=("morph_train_failed", "backend_orgseg"),
            provenance=make_provenance(
                OPERATION_ID,
                params_hash=params_hash,
                argv=tuple(cli_argv),
                actual_backend="orgseg",
                attempted_backends=("orgseg",),
                started_at=started_at,
                finished_at=utc_now(),
            ),
            errors=(
                ErrorDetail(
                    code="morphology.train_failed"
                    if not resolve_reason
                    else f"morphology.train_{resolve_reason}",
                    message=run_err or "OrgSegNet training executor raised.",
                    details={
                        "config": str(cfg_path),
                        "load_from": str(ckpt),
                        "work_dir": str(wdir),
                        "data_root": str(data_root),
                    },
                    retryable=False,
                ),
            ),
        )

    assert best_ckpt is not None  # guaranteed by the resolution gate above
    checkpoint_artifacts = collect_artifacts(
        [best_ckpt],
        kind="model_checkpoint",
        format="pth",
    )
    best_iteration = int(_BEST_CKPT_PATTERN.match(best_ckpt.name).group("iteration"))  # type: ignore[union-attr]

    return OrganelleResult(
        operation_id=OPERATION_ID,
        operation_version=MORPHOLOGY_OPERATION_VERSION,
        scope=MORPHOLOGY_SCOPE,
        status="ok",
        summary_text=(
            f"OrgSegNet fine-tuned on {ds['n_train']} train / "
            f"{ds['n_val']} val images; best checkpoint {best_ckpt.name}."
        ),
        metrics={
            "data_root": str(data_root),
            "ran": True,
            "backend_found": True,
            "work_dir": str(wdir),
            "checkpoint": str(best_ckpt),
            "checkpoint_iteration": best_iteration,
            "load_from": str(ckpt),
            "n_train": ds["n_train"],
            "n_val": ds["n_val"],
            "n_test": ds["n_test"],
            "max_iters": max_iters,
        },
        findings=(
            Finding(
                code="morphology.trained",
                metric="best_checkpoint",
                value=best_ckpt.name,
                evidence_artifact_ids=tuple(item.object_id for item in checkpoint_artifacts),
            ),
            Finding(
                code="morphology.trained",
                metric="checkpoint_iteration",
                value=best_iteration,
                unit="iterations",
            ),
            Finding(
                code="morphology.trained",
                metric="max_iters",
                value=int(max_iters),
                unit="iterations",
            ),
        ),
        flags=("morph_train_ran", "backend_orgseg"),
        artifacts=checkpoint_artifacts,
        provenance=make_provenance(
            OPERATION_ID,
            params_hash=params_hash,
            argv=tuple(cli_argv),
            actual_backend="orgseg",
            attempted_backends=("orgseg",),
            software_versions={"device": device_record},
            model_hashes={
                "load_from": {
                    "path": str(ckpt),
                    "sha256": load_from_ref.sha256,
                },
                "best_checkpoint": {
                    "path": str(best_ckpt),
                    "sha256": checkpoint_artifacts[0].sha256,
                },
            },
            started_at=started_at,
            finished_at=utc_now(),
        ),
    )
