"""Standalone inference script of the ``lab_mito`` image-job engine.

It is executed as a script by the delivery interpreter
(``<root>/data/orgsegnet/training-venv/bin/python``), which has torch but not
this package, so it imports nothing from ``organelleverse``. Run it with::

    python lab_mito_infer.py --root R --checkpoint best.pt --image in.tif --nm 6.0 --out mask.npy

The preprocessing and inference reproduce the lab fine-tune exactly:

* ``scripts/plant-sam/build_lab_roi_labels.py``: percentile stretch to uint8
  (0.1 / 99.9 %), then bilinear resampling to 6 nm/px;
* ``scripts/plant-sam/evaluate_lab.py``: scales 1, 2, 3, per-class scale
  weights (rows normalised to sum 1), mitochondrion = ``argmax == 2``.

Output: a boolean mask on the 6 nm/px grid (``--out``, .npy), the mitochondrion
probability averaged over relative scales 1 and 2 as uint8 (round(p * 255)) next
to it (``<out stem>.prob.npy``) and a JSON (``--out`` with the suffix ``.json``). torch is imported inside
:func:`main` so the preprocessing helpers are importable (and testable) on a
machine without it.
"""

from __future__ import annotations

import os
import sys

# The interpreter puts this file's directory first on sys.path; that directory holds modules of the
# organelleverse package whose names (train, device, ...) must not shadow third-party imports.
if __name__ == "__main__":
    _HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402

MODEL_NM_PER_PX = 6.0
SCALES = (1.0, 2.0, 3.0)
#: Per-class (rows 0..4) scale weights as in evaluate_lab.py; columns are the scales above.
WEIGHTS_SPEC = "1,1,1;0,1,0;1,0,0;0,1,1;1,0,0"
MITO_CLASS = 2
#: Scales whose mitochondrion probabilities are averaged into the saved probability map.
PROB_SCALES = (1.0, 2.0)
PERCENTILES = (0.1, 99.9)
MIN_SIDE = 32


def weights_matrix(spec: str = WEIGHTS_SPEC) -> np.ndarray:
    """Class x scale weight matrix with every row normalised to sum 1 (as evaluate_lab.py)."""
    matrix = np.array([[float(x) for x in row.split(",")] for row in spec.split(";")], np.float32)
    return matrix / np.maximum(1e-6, matrix.sum(1, keepdims=True))


def stretch_to_uint8(raw: np.ndarray) -> np.ndarray:
    """Percentile (0.1 / 99.9) stretch of the raw plane to uint8, as build_lab_roi_labels.py."""
    values = raw.astype(np.float32)
    lo, hi = np.percentile(values, list(PERCENTILES))
    return (np.clip((values - lo) / max(1e-6, hi - lo), 0, 1) * 255).astype(np.uint8)


def resampled_shape(shape: tuple[int, int], nm: float) -> tuple[int, int]:
    """Height/width of the image at 6 nm/px: ``max(32, round(side * nm / 6))``."""
    factor = nm / MODEL_NM_PER_PX
    return max(MIN_SIDE, round(shape[0] * factor)), max(MIN_SIDE, round(shape[1] * factor))


def resample_to_model_scale(img8: np.ndarray, nm: float) -> np.ndarray:
    """Bilinear (PIL) resampling of the uint8 plane to 6 nm/px."""
    from PIL import Image

    th, tw = resampled_shape((img8.shape[0], img8.shape[1]), nm)
    return np.asarray(Image.fromarray(img8).resize((tw, th), Image.BILINEAR))


def preprocess(raw: np.ndarray, nm: float) -> tuple[np.ndarray, dict[str, Any]]:
    """Raw plane + pixel size (nm) -> uint8 plane at 6 nm/px and what was done."""
    img8 = resample_to_model_scale(stretch_to_uint8(raw), nm)
    return img8, {"nm": nm, "resample_factor": nm / MODEL_NM_PER_PX, "input_shape": list(raw.shape),
                  "model_shape": list(img8.shape)}


def read_plane(path: Path) -> np.ndarray:
    """Read the staged image as one 2-D plane with its native dtype (RGB -> BT.601 luminance)."""
    try:
        import tifffile

        array = np.asarray(tifffile.imread(path))
    except ImportError:
        from PIL import Image

        with Image.open(path) as image:
            array = np.asarray(image)
    if array.ndim == 3 and array.shape[-1] in (3, 4):
        array = array[..., :3].astype(np.float64) @ np.array([0.299, 0.587, 0.114])
    if array.ndim != 2:
        raise SystemExit(f"lab_mito needs one 2-D plane, got shape {list(array.shape)}")
    return array


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--nm", required=True, type=float)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if not args.nm > 0:
        raise SystemExit(f"--nm must be positive, got {args.nm}")
    started = time.time()

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("lab_mito needs a CUDA device; torch.cuda.is_available() is False on this machine")
    sys.path.insert(0, str(Path(args.root) / "scripts" / "plant-sam"))
    from organelle_unet import build_model  # type: ignore[import-not-found]
    from predict_organelle_unet import infer_logits  # type: ignore[import-not-found]

    checkpoint = Path(args.checkpoint)
    ck = torch.load(checkpoint, map_location="cuda", weights_only=False)  # CUDA is required above
    cargs = ck["config"]["args"]
    model = build_model(cargs["encoder"], pretrained=False, psp=cargs.get("psp", False),
                        aux_head=cargs.get("aux_head", False)).cuda().eval()
    model.load_state_dict(ck["model"])
    model = model.to(memory_format=torch.channels_last)

    img8, notes = preprocess(read_plane(Path(args.image)), args.nm)
    weights = weights_matrix()
    weights_t = torch.from_numpy(weights).cuda()
    acc = None
    prob = None
    with torch.no_grad():
        for index, scale in enumerate(SCALES):
            probs = infer_logits(model, img8, max(1.0, scale))   # softmax probabilities per class
            if scale in PROB_SCALES:
                mito = probs[MITO_CLASS].float()
                prob = mito if prob is None else prob + mito
            part = (probs * weights_t[:, index][:, None, None]).half()
            del probs
            acc = part if acc is None else acc + part
    mask = (acc.argmax(0) == MITO_CLASS).cpu().numpy().astype(bool)
    prob8 = (prob / len(PROB_SCALES)).clamp(0, 1).mul(255).round().to(torch.uint8).cpu().numpy()
    del acc, prob
    torch.cuda.empty_cache()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, mask)
    prob_path = out.with_name(out.stem + ".prob.npy")
    np.save(prob_path, prob8)
    summary = {
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_of(checkpoint),
        "encoder": cargs.get("encoder"), "psp": bool(cargs.get("psp", False)),
        "aux_head": bool(cargs.get("aux_head", False)),
        "scales": list(SCALES), "weights_spec": WEIGHTS_SPEC, "weights": weights.tolist(),
        "mitochondrion_class": MITO_CLASS, "mask_shape": list(mask.shape),
        "mask_pixels": int(mask.sum()), "device": torch.cuda.get_device_name(0),
        "probability": {"file": prob_path.name, "scales": list(PROB_SCALES), "encoding": "uint8 = round(p * 255)",
                        "definition": "mean mitochondrion softmax probability over the listed scales"},
        "torch": torch.__version__, "seconds": round(time.time() - started, 2), **notes,
    }
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"lab_mito: {summary['mask_pixels']} mitochondrion pixels on {mask.shape[1]}x{mask.shape[0]} "
          f"at 6 nm/px in {summary['seconds']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
