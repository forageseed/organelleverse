"""mmcv.ops compatibility shim for inference on the modern stack.

mmcv 2.0.0 (the only version mmseg 1.x accepts) ships compiled ops that do
not build against torch 2.x. The OrgSegNet model (PSPNet-R50 + custom head)
uses none of them at inference time — but mmseg imports every model file
eagerly, dragging five ``mmcv.ops`` names in at import time. This module
installs pure-torch implementations where they are load-bearing for
training-era losses and fail-closed placeholders for heads OrgSegNet never
instantiates (calling those raises loudly, never silently).
"""

from __future__ import annotations

import sys
import types


def _sigmoid_focal_loss(pred, target, weight=None, gamma=2.0, alpha=0.25,
                        reduction="mean", avg_factor=None):  # type: ignore[no-untyped-def]
    import torch
    import torch.nn.functional as F

    p = torch.sigmoid(pred)
    ce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    pt = p * target + (1 - p) * (1 - target)
    loss = ce * ((1 - pt) ** gamma)
    if weight is not None:
        loss = loss * weight
    if reduction == "mean":
        return loss.sum() / (avg_factor if avg_factor else loss.numel())
    if reduction == "sum":
        return loss.sum()
    return loss


def _point_sample(input, points, align_corners=False, **kwargs):  # type: ignore[no-untyped-def]
    import torch.nn.functional as F

    # mmcv point_sample: points in [0, 1] image coords as (y, x); F.grid_sample
    # wants (x, y) in [-1, 1].
    xy = points.clone()
    xy[..., 0] = points[..., 1] * 2 - 1
    xy[..., 1] = points[..., 0] * 2 - 1
    return F.grid_sample(input, xy, align_corners=align_corners, **kwargs)


def _fail_closed(name: str):  # type: ignore[no-untyped-def]
    def _init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise NotImplementedError(
            f"{name} is not available in the mmcv.ops shim (compiled mmcv "
            "ops absent). OrgSegNet inference never instantiates it."
        )

    return type(name, (), {"__init__": _init})


def install_mmcv_ops_shim() -> None:
    """Make `from mmcv.ops import X` work for the names mmseg imports."""
    try:
        import mmcv.ops  # noqa: F401  # compiled ops present — nothing to do
        return
    except Exception:
        pass
    stub = types.ModuleType("mmcv.ops")
    stub.sigmoid_focal_loss = _sigmoid_focal_loss
    stub._sigmoid_focal_loss = _sigmoid_focal_loss
    stub.point_sample = _point_sample
    stub.CrissCrossAttention = _fail_closed("CrissCrossAttention")
    stub.PSAMask = _fail_closed("PSAMask")
    sys.modules["mmcv.ops"] = stub
