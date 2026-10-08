"""Device resolution for the morphology suite (T5).

One shared resolution layer used by ``segment`` / ``train`` / the workbench
panel — the three call sites must never grow their own device logic.

Hard constraints (task brief 2026-08-17, T5):

- The OrgSegNet stack is pinned to torch 1.13.1 + mmcv 2.0.0rc4, so the
  ``orgseg`` backend can only ever be CUDA or CPU — no MPS, no XPU. The
  portable path is ``micro_sam`` (pure PyTorch, follows modern torch).
  This is declared in :data:`BACKEND_DEVICE_SUPPORT`, not discovered.
- ``device="auto"`` routes deterministically on (detected hardware,
  backend capability): the same combination always resolves the same way.
- An explicit device that is unavailable fails closed with the feasible
  list. There is no silent fallback to CPU — a 100-200x slowdown nobody
  asked for is exactly the silent failure this project exists to kill.
- Fine-tuning declares its feasibility floor up front
  (:data:`TRAIN_MIN_VRAM_GB`): integrated GPUs share system memory and
  would OOM or crawl, so training on anything but a sufficiently large
  discrete CUDA device is rejected *before* the executor runs.
- The resolved device (kind, backend, runtime version) goes into
  provenance: CPU and GPU numerics legitimately differ, and an unrecorded
  device makes results non-reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "AUTO_PREFERENCE",
    "BACKEND_DEVICE_SUPPORT",
    "TRAIN_MIN_VRAM_GB",
    "DeviceInfo",
    "DeviceResolution",
    "detect_devices",
    "resolve_device",
]

#: Device kinds in deterministic auto-routing preference order.
AUTO_PREFERENCE = ("cuda", "mps", "xpu", "cpu")

#: Declared per-backend device support. ``orgseg`` is torch-1.13.1-pinned
#: (no XPU, immature MPS, mmcv compiled CUDA ops missing on other devices);
#: ``micro_sam`` follows modern torch. This is a declaration of fact about
#: the two stacks, not a probe.
BACKEND_DEVICE_SUPPORT: dict[str, tuple[str, ...]] = {
    "orgseg": ("cuda", "cpu"),
    "micro_sam": ("cuda", "mps", "xpu", "cpu"),
}

#: Declared VRAM floor (GiB) for OrgSegNet fine-tuning (PSPNet-R50-d8 at
#: 768x512). A policy floor for fail-fast rejection, not a measurement;
#: operators tune it here, in one place.
TRAIN_MIN_VRAM_GB = 8.0


@dataclass(frozen=True)
class DeviceInfo:
    """One detected compute device."""

    kind: str  # "cuda" | "mps" | "xpu" | "cpu"
    name: str
    total_vram_gb: float | None  # None when unified/shared or unknown
    runtime_version: str | None  # cuda version / torch version / platform

    @property
    def torch_device(self) -> str:
        """The torch device string executors receive (``cuda:0`` / ``mps`` / ...)."""
        if self.kind in {"cpu", "mps"}:
            return self.kind
        return f"{self.kind}:0"

    def provenance_record(self) -> dict[str, str | float | None]:
        """The provenance payload for this resolved device."""
        return {
            "kind": self.kind,
            "name": self.name,
            "total_vram_gb": self.total_vram_gb,
            "runtime_version": self.runtime_version,
            "torch_device": self.torch_device,
        }


@dataclass(frozen=True)
class DeviceResolution:
    """The outcome of resolving a requested device for one backend."""

    device: DeviceInfo | None
    requested: str
    backend: str
    purpose: str
    error_code: str | None
    error_message: str | None
    feasible: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return self.device is not None


def detect_devices() -> tuple[DeviceInfo, ...]:
    """Detect available compute devices, lazily importing torch.

    Without torch installed, only ``cpu`` is reported: neither backend can
    execute without torch anyway, and CUDA/MPS/XPU presence cannot honestly
    be known without it.
    """
    devices: list[DeviceInfo] = []
    try:
        import torch
    except ImportError:
        return (DeviceInfo(kind="cpu", name="cpu", total_vram_gb=None, runtime_version=None),)

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        devices.append(
            DeviceInfo(
                kind="cuda",
                name=torch.cuda.get_device_name(0),
                total_vram_gb=round(props.total_memory / (1024**3), 2),
                runtime_version=torch.version.cuda,
            )
        )
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        devices.append(
            DeviceInfo(
                kind="mps",
                name="apple-silicon-mps",
                total_vram_gb=None,  # unified memory: no discrete VRAM to report
                runtime_version=torch.__version__,
            )
        )
    xpu = getattr(torch, "xpu", None)
    if xpu is not None and xpu.is_available():
        devices.append(
            DeviceInfo(
                kind="xpu",
                name=xpu.get_device_name(0),
                total_vram_gb=None,
                runtime_version=torch.__version__,
            )
        )
    devices.append(
        DeviceInfo(kind="cpu", name="cpu", total_vram_gb=None, runtime_version=torch.__version__)
    )
    return tuple(devices)


def resolve_device(
    requested: str,
    *,
    backend: str,
    purpose: str = "inference",
    detected: tuple[DeviceInfo, ...] | None = None,
) -> DeviceResolution:
    """Resolve ``requested`` to a concrete device for ``backend``.

    Pure and deterministic given (requested, backend, purpose, detected):
    the same inputs always produce the same resolution. Failures are data,
    not exceptions — the caller renders them into a ``status="failed"``
    result with the feasible device list.
    """
    available = detected if detected is not None else detect_devices()
    by_kind = {device.kind: device for device in available}
    supported = BACKEND_DEVICE_SUPPORT.get(backend, ("cpu",))

    def _failure(code: str, message: str) -> DeviceResolution:
        return DeviceResolution(
            device=None,
            requested=requested,
            backend=backend,
            purpose=purpose,
            error_code=code,
            error_message=message,
            feasible=tuple(kind for kind in supported if kind in by_kind),
        )

    if requested == "auto":
        for kind in AUTO_PREFERENCE:
            if kind in supported and kind in by_kind:
                resolved = by_kind[kind]
                break
        else:
            return _failure(
                "morphology.device_unavailable",
                f"no detected device is supported by backend {backend!r}",
            )
    else:
        kind = requested.split(":", 1)[0]
        if kind not in supported:
            suggestion = (
                " Use the micro_sam backend for mps/xpu devices."
                if kind in {"mps", "xpu"} and backend == "orgseg"
                else ""
            )
            return _failure(
                "morphology.device_backend_unsupported",
                f"backend {backend!r} does not support device {requested!r}"
                f" (supported: {list(supported)}).{suggestion}",
            )
        if kind not in by_kind:
            return _failure(
                "morphology.device_unavailable",
                f"device {requested!r} is not available on this machine"
                f" (detected: {[d.kind for d in available]})",
            )
        resolved = by_kind[kind]

    if purpose == "train":
        if resolved.kind != "cuda":
            return _failure(
                "morphology.train_device_infeasible",
                f"fine-tuning requires a discrete CUDA device with "
                f">= {TRAIN_MIN_VRAM_GB} GiB VRAM; resolved device is "
                f"{resolved.kind!r} (integrated/CPU training would OOM or "
                f"never finish)",
            )
        vram = resolved.total_vram_gb
        if vram is not None and vram < TRAIN_MIN_VRAM_GB:
            return _failure(
                "morphology.train_device_infeasible",
                f"fine-tuning requires >= {TRAIN_MIN_VRAM_GB} GiB VRAM; "
                f"{resolved.name!r} has {vram} GiB",
            )

    return DeviceResolution(
        device=resolved,
        requested=requested,
        backend=backend,
        purpose=purpose,
        error_code=None,
        error_message=None,
        feasible=tuple(kind for kind in supported if kind in by_kind),
    )
