"""Standalone SAM 3 inference script of the ``sam3`` image-job engine.

It is executed as a script by the isolated SAM 3 interpreter (``<root>/data/sam3/env/bin/python``, or
``env/python.exe`` on a Windows workstation; Python 3.12 + torch + Meta's ``sam3`` package; ``--device auto``
runs on CUDA when there is a GPU, else on the CPU), which does not have this package, so it imports
nothing from ``organelleverse``. Weights are Meta's ``sam3.pt`` (SAM License) stored at
``<root>/data/sam3/weights/sam3.pt``.

Three modes, all on 8-bit grey images (percentile stretch 0.1 / 99.9 %, replicated to RGB):

``text``
    Zero-shot concept segmentation: every instance matching a short noun phrase, run on overlapping
    tiles so small organelles are seen at a usable size. The organelle is given as the user names it
    (``--target 线粒体``) and translated to the phrase measured to ground on that modality
    (``concept_phrase``); ``--text`` passes a phrase verbatim.
``exemplar``
    Same tiled detector, prompted with example boxes (x0, y0, x1, y1 pixels) drawn on the image;
    only tiles that contain an exemplar are searched with it.
``refine``
    Box + interior-point prompt for every object of a candidate instance map (another engine's
    output) on a context crop around it at full resolution; returns SAM's three hypotheses with
    their predicted IoU so the caller decides which, if any, replaces the candidate.

Output (``--out`` .npz) holds the proposals as bounding boxes + bit-packed crop masks, scores and
the mode; a JSON with the run facts sits next to it (``--out`` with suffix ``.json``). torch and
sam3 are imported lazily so the geometry helpers are importable (and testable) without them.
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
import contextlib  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402

PERCENTILES = (0.1, 99.9)
SAM3_RESOLUTION = 1008
BPE_NAME = "bpe_simple_vocab_16e6.txt.gz"

# People name the organelle ("线粒体", "mitochondria"); SAM 3 needs the phrase that grounds on the image.
# Its concepts come from natural images, so on EM the biological noun can find nothing: on the lab EM VAL
# set (28 images, 529 mitochondria, 6 nm/px) "mitochondrion" gave 0 proposals while "membrane-bound
# organelle" alone found 81.5 % at score >= 0.3 (.agent-work/image-workbench/sam3, bench_prompts.py).
# The phrase finds organelle-like objects, not mitochondria specifically; identity still needs the
# lab model or review. Pairs not measured fall back to the English noun and are reported unverified.
ORGANELLE_ALIASES = {
    "mitochondrion": ("mitochondrion", "mitochondria", "mito", "线粒体"),
    "chloroplast": ("chloroplast", "chloroplasts", "叶绿体"),
    "plastid": ("plastid", "plastids", "质体"),
}
CONCEPT_PHRASES = {("mitochondrion", "em"): "membrane-bound organelle"}


def concept_phrase(target: str, modality: str = "em") -> tuple[str, bool]:
    """Text prompt for an organelle named by the user and whether that phrase was measured to work."""
    key = target.strip().lower()
    name = next((n for n, names in ORGANELLE_ALIASES.items() if key in names), None)
    if name is None:
        return target.strip(), False
    phrase = CONCEPT_PHRASES.get((name, modality.lower()))
    return (phrase, True) if phrase else (name, False)


@dataclass
class Proposal:
    """One mask on the full image: inclusive-exclusive box (y0, x0, y1, x1) and the mask inside it."""

    box: tuple[int, int, int, int]
    mask: np.ndarray  # bool, shape (y1 - y0, x1 - x0)
    score: float
    source: str = ""
    ref: int = 0  # refine: candidate label; exemplar/text: tile index

    @property
    def area(self) -> int:
        return int(self.mask.sum())


def stretch_to_uint8(raw: np.ndarray) -> np.ndarray:
    """Percentile stretch of a 2-D (or RGB, averaged) image to uint8."""
    arr = np.asarray(raw)
    if arr.ndim == 3:
        arr = arr[..., :3].mean(axis=-1)
    arr = arr.astype(np.float32)
    lo, hi = np.percentile(arr, PERCENTILES)
    return (np.clip((arr - lo) / max(1e-6, float(hi - lo)), 0, 1) * 255).astype(np.uint8)


def tile_grid(height: int, width: int, size: int, overlap: int) -> list[tuple[int, int, int, int]]:
    """Overlapping tiles (y0, x0, y1, x1) covering the image; the last row/column is flush with the edge."""
    if size <= overlap:
        raise ValueError("tile size must exceed the overlap")

    def starts(extent: int) -> list[int]:
        if extent <= size:
            return [0]
        step = size - overlap
        out = list(range(0, extent - size, step))
        out.append(extent - size)
        return sorted(set(out))

    return [(y, x, min(height, y + size), min(width, x + size)) for y in starts(height) for x in starts(width)]


def crop_box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys = np.flatnonzero(mask.any(1))
    if ys.size == 0:
        return None
    xs = np.flatnonzero(mask.any(0))
    return int(ys[0]), int(xs[0]), int(ys[-1]) + 1, int(xs[-1]) + 1


def to_proposal(mask: np.ndarray, offset: tuple[int, int], score: float, source: str, ref: int) -> Proposal | None:
    box = crop_box(mask)
    if box is None:
        return None
    y0, x0, y1, x1 = box
    oy, ox = offset
    return Proposal((y0 + oy, x0 + ox, y1 + oy, x1 + ox), mask[y0:y1, x0:x1].copy(), float(score), source, ref)


def pair_iou(a: Proposal, b: Proposal) -> tuple[float, float, float]:
    """IoU, fraction of a inside b, fraction of b inside a."""
    y0, x0 = max(a.box[0], b.box[0]), max(a.box[1], b.box[1])
    y1, x1 = min(a.box[2], b.box[2]), min(a.box[3], b.box[3])
    if y1 <= y0 or x1 <= x0:
        return 0.0, 0.0, 0.0
    ma = a.mask[y0 - a.box[0]:y1 - a.box[0], x0 - a.box[1]:x1 - a.box[1]]
    mb = b.mask[y0 - b.box[0]:y1 - b.box[0], x0 - b.box[1]:x1 - b.box[1]]
    inter = int((ma & mb).sum())
    if inter == 0:
        return 0.0, 0.0, 0.0
    aa, ab = a.area, b.area
    return inter / (aa + ab - inter), inter / max(1, aa), inter / max(1, ab)


def suppress(props: list[Proposal], *, iou: float = 0.5, containment: float = 0.87,
             max_ratio: float = 2.2) -> list[Proposal]:
    """Greedy score-ordered de-duplication across tiles.

    A proposal is dropped when it overlaps a kept one with IoU >= ``iou`` or when >= ``containment`` of
    it lies inside a kept proposal at most ``max_ratio`` times its size (a tile-edge fragment of the
    same object). The containment rule follows the sam3-segmentation skill (87 %, 2.2x).
    """
    kept: list[Proposal] = []
    for p in sorted(props, key=lambda q: -q.score):
        dup = False
        for k in kept:
            i, a_in_b, b_in_a = pair_iou(p, k)
            if i >= iou or (a_in_b >= containment and k.area <= max_ratio * p.area) \
                    or (b_in_a >= containment and p.area <= max_ratio * k.area):
                dup = True
                break
        if not dup:
            kept.append(p)
    return kept


def touches_inner_edge(mask: np.ndarray, tile: tuple[int, int, int, int], shape: tuple[int, int]) -> bool:
    """True if the tile mask touches a tile border that is not the image border (cut-off object)."""
    y0, x0, y1, x1 = tile
    h, w = shape
    return bool((y0 > 0 and mask[0].any()) or (x0 > 0 and mask[:, 0].any())
                or (y1 < h and mask[-1].any()) or (x1 < w and mask[:, -1].any()))


def pack(props: list[Proposal]) -> dict[str, np.ndarray]:
    boxes = np.array([p.box for p in props], np.int32).reshape(-1, 4)
    flat = [np.packbits(p.mask.ravel()) for p in props]
    offsets = np.cumsum([0] + [f.size for f in flat]).astype(np.int64)
    return {"boxes": boxes, "scores": np.array([p.score for p in props], np.float32),
            "refs": np.array([p.ref for p in props], np.int32),
            "sources": np.array([p.source for p in props], dtype="U32"),
            "bits": np.concatenate(flat) if flat else np.zeros(0, np.uint8), "offsets": offsets}


def unpack(data: Any) -> list[Proposal]:
    out = []
    for i, (y0, x0, y1, x1) in enumerate(np.asarray(data["boxes"]).tolist()):
        n = (y1 - y0) * (x1 - x0)
        bits = data["bits"][data["offsets"][i]:data["offsets"][i + 1]]
        mask = np.unpackbits(bits)[:n].astype(bool).reshape(y1 - y0, x1 - x0)
        out.append(Proposal((y0, x0, y1, x1), mask, float(data["scores"][i]), str(data["sources"][i]),
                            int(data["refs"][i])))
    return out


def paint(props: list[Proposal], shape: tuple[int, int]) -> np.ndarray:
    """Instance map from proposals; later (lower-priority) proposals never overwrite earlier pixels."""
    labels = np.zeros(shape, np.int32)
    for k, p in enumerate(props, 1):
        y0, x0, y1, x1 = p.box
        view = labels[y0:y1, x0:x1]
        view[(view == 0) & p.mask] = k
    return labels


def resolve_device(device: str) -> str:
    """The device kind for ``device`` (``auto`` or explicit) by the suite's resolution layer, device.py.

    device.py (standard library + lazily torch) is loaded by path because this interpreter has no
    organelleverse; it sees this interpreter's torch, the only one that knows its devices. ``auto`` follows
    AUTO_PREFERENCE within the ``sam3`` backend's declared devices; an unavailable explicit device fails
    with the feasible list.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("_organelleverse_device",
                                                  Path(__file__).with_name("device.py"))
    device_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = device_module  # its dataclasses look their module up there
    spec.loader.exec_module(device_module)
    resolution = device_module.resolve_device(device, backend="sam3")
    if not resolution.ok:
        raise SystemExit(f"{resolution.error_code}: {resolution.error_message} "
                         f"(feasible: {list(resolution.feasible)})")
    # the kind, not "cuda:0": sam3's builder moves the model to the GPU only for device == "cuda"
    return resolution.device.kind


@contextlib.contextmanager
def _hardcoded_cuda_on(device: str):
    """While the model is built, tensor factories asked for ``device="cuda"`` use ``device`` instead.

    sam3 0.1.4 precomputes position encodings and box coordinates with ``device="cuda"`` written in
    (position_encoding.py, decoder.py), so a CPU build would fail without this.
    """
    import torch

    if device.startswith("cuda"):
        yield
        return
    names = ("zeros", "ones", "empty", "arange", "linspace", "tensor", "full")
    originals = {name: getattr(torch, name) for name in names}

    def redirect(factory):
        def call(*args, **kwargs):
            if str(kwargs.get("device", "")).startswith("cuda"):
                kwargs["device"] = device
            return factory(*args, **kwargs)
        return call

    for name, factory in originals.items():
        setattr(torch, name, redirect(factory))
    try:
        yield
    finally:
        for name, factory in originals.items():
            setattr(torch, name, factory)


_SAFETENSORS_DTYPES = {"F32": "float32", "F16": "float16", "BF16": "bfloat16", "F64": "float64", "I64": "int64",
                       "I32": "int32", "I16": "int16", "I8": "int8", "U8": "uint8", "BOOL": "bool",
                       "C64": "complex64"}


def _load_streamed(model, path: Path) -> list[str]:
    """Read ``sam3.safetensors`` straight into the built model's tensors; returns the model keys not set.

    Same key mapping as sam3's own ``_load_checkpoint`` (detector -> image model, tracker -> the
    interactive predictor). sam3 itself reads the whole ``.pt`` into memory next to the model, and a
    mapped file (torch ``mmap=True``, the safetensors library) is a copy-on-write view that Windows
    charges in full against the commit limit; reading each tensor's bytes from the file into the
    parameter keeps the peak at the model's own size -- what lets a workstation with ~8 GB of commit
    charge left run it.
    """
    import torch

    own = model.state_dict()
    loaded: set[str] = set()
    with open(path, "rb") as handle, torch.no_grad():
        size = int.from_bytes(handle.read(8), "little")
        header = json.loads(handle.read(size))
        base = 8 + size
        keys = [k for k in header if k != "__metadata__"]
        renames = [(k, k.replace("detector.", "")) for k in keys if "detector" in k]
        if model.inst_interactive_predictor is not None:
            renames += [(k, k.replace("tracker.", "inst_interactive_predictor.model."))
                        for k in keys if "tracker" in k]
        for key, name in renames:
            target = own.get(name)
            if target is None:
                continue
            entry = header[key]
            dtype = getattr(torch, _SAFETENSORS_DTYPES[entry["dtype"]])
            start, stop = entry["data_offsets"]
            if tuple(entry["shape"]) != tuple(target.shape):
                raise ValueError(f"{key}: shape {entry['shape']} does not match the model's {tuple(target.shape)}")
            direct = target.dtype == dtype and target.is_contiguous()
            buffer = target if direct else torch.empty(entry["shape"], dtype=dtype)
            handle.seek(base + start)
            view = buffer.reshape(-1).view(torch.uint8).numpy()
            if view.nbytes != stop - start or handle.readinto(view) != view.nbytes:
                raise ValueError(f"{key}: truncated or inconsistent tensor data in {path.name}")
            if not direct:
                target.copy_(buffer)
            loaded.add(name)
    return sorted(set(own) - loaded)


def _stub_triton_if_missing() -> bool:
    """Let ``import sam3`` succeed where Triton does not exist (Windows, CPU-only torch).

    sam3 0.1.4 imports one Triton kernel at module level, the distance transform of its video tracker
    (``sam3.model.edt``); the image model used here never calls it, and its other Triton paths
    (CUDA NMS, connected components) are imported only on CUDA. That one module is replaced by a
    stand-in whose kernel raises. A fake ``triton`` package is not used: torch would detect it.
    """
    import importlib.util
    import types

    if importlib.util.find_spec("triton") is not None or "sam3.model.edt" in sys.modules:
        return False

    def edt_triton(data):
        raise RuntimeError("Triton is not installed here; the SAM 3 video tracker needs a CUDA machine")

    edt = types.ModuleType("sam3.model.edt")
    edt.edt_triton = edt_triton
    sys.modules["sam3.model.edt"] = edt
    return True


class Sam3:
    """Thin wrapper over Meta's sam3 image model + processor."""

    def __init__(self, checkpoint: str, device: str = "auto", bpe_path: str | None = None) -> None:
        import torch

        self.triton_stub = _stub_triton_if_missing()
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        device = resolve_device(device)
        self.torch = torch
        self.device = device
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # The PyPI sam3 wheel ships without its CLIP BPE vocabulary; it is kept next to the weights.
        bpe = bpe_path or str(Path(checkpoint).with_name(BPE_NAME))
        if not Path(bpe).is_file():
            raise FileNotFoundError(f"SAM 3 text tokenizer vocabulary not found: {bpe}")
        # Off CUDA a converted copy next to the checkpoint (same tensors) is streamed in, see _load_streamed.
        streamed = Path(checkpoint).with_suffix(".safetensors")
        stream = not device.startswith("cuda") and streamed.is_file()
        self.weights_path = str(streamed if stream else checkpoint)
        with _hardcoded_cuda_on(device):
            self.model = build_sam3_image_model(bpe_path=bpe, checkpoint_path=None if stream else checkpoint,
                                                load_from_HF=False, enable_inst_interactivity=True, device=device,
                                                compile=False)
        self.missing_keys = _load_streamed(self.model, streamed) if stream else None
        if not device.startswith("cuda"):
            # sam3 pins box tensors before an async copy to the GPU (geometry_encoders.py); without an
            # accelerator pinning raises, and the copy it prepares is a no-op on the CPU.
            torch.Tensor.pin_memory = lambda tensor, *args, **kwargs: tensor
        self.model.eval()
        self.processor = Sam3Processor(self.model, device=device, confidence_threshold=0.0)

    def _ctx(self):
        # bf16 autocast (the precision SAM 3 is released for). On the CPU without the weight-cast cache:
        # on the demo field vs fp32, box masks IoU >= 0.992 and text proposals (score >= 0.3) 52/52 matched
        # at IoU 0.997, 27 % faster and a lower memory peak.
        kind = self.torch.device(self.device).type
        return self.torch.autocast(kind, dtype=self.torch.bfloat16, cache_enabled=kind == "cuda")

    @staticmethod
    def _pil(gray: np.ndarray):
        from PIL import Image

        return Image.fromarray(np.repeat(gray[..., None], 3, axis=2))

    def detect(self, tile: np.ndarray, *, text: str | None, boxes_xyxy: list[list[float]] | None = None,
               threshold: float = 0.0) -> list[tuple[np.ndarray, float]]:
        """Concept detection on one tile: text phrase and/or positive exemplar boxes (tile pixels)."""
        h, w = tile.shape
        with self.torch.inference_mode(), self._ctx():
            state = self.processor.set_image(self._pil(tile))
            self.processor.reset_all_prompts(state)
            if text:
                state = self.processor.set_text_prompt(prompt=text, state=state)
            for x0, y0, x1, y1 in boxes_xyxy or []:
                box = [(x0 + x1) / 2 / w, (y0 + y1) / 2 / h, (x1 - x0) / w, (y1 - y0) / h]
                state = self.processor.add_geometric_prompt(box=box, label=True, state=state)
            if "masks" not in state:
                return []
            masks = state["masks"].squeeze(1).cpu().numpy().astype(bool)
            scores = state["scores"].float().cpu().numpy()
        return [(m, float(s)) for m, s in zip(masks, scores, strict=True) if s >= threshold]

    def refine(self, crop: np.ndarray, box_xyxy: list[float], points: list[list[float]],
               labels: list[int]) -> list[tuple[np.ndarray, float]]:
        """SAM-style box + point prompt on one crop; returns the three hypotheses with predicted IoU."""
        with self.torch.inference_mode(), self._ctx():
            state = self.processor.set_image(self._pil(crop))
            masks, scores, _ = self.model.predict_inst(
                state, point_coords=np.asarray(points, np.float32) if points else None,
                point_labels=np.asarray(labels, np.int32) if points else None,
                box=np.asarray(box_xyxy, np.float32), multimask_output=True, return_logits=False)
        return [(np.asarray(m).astype(bool), float(s)) for m, s in zip(masks, scores, strict=True)]


def tiled_detect(sam: Sam3, image: np.ndarray, *, text: str | None, tile: int, overlap: int,
                 threshold: float, exemplars: list[tuple[int, int, int, int]] | None = None,
                 max_exemplars: int = 4, source: str = "text") -> list[Proposal]:
    """Run concept detection on every tile and return de-duplicated full-image proposals.

    Masks cut by an inner tile border are dropped: the overlap (larger than a typical object) means
    a neighbouring tile holds the whole object. ``exemplars`` are (y0, x0, y1, x1) image boxes in
    priority order; a tile is searched with the first ``max_exemplars`` it fully contains and
    skipped if it contains none.
    """
    shape = image.shape
    props: list[Proposal] = []
    for t, (y0, x0, y1, x1) in enumerate(tile_grid(shape[0], shape[1], tile, overlap)):
        boxes = None
        if exemplars is not None:
            boxes = [[bx0 - x0, by0 - y0, bx1 - x0, by1 - y0] for by0, bx0, by1, bx1 in exemplars
                     if by0 >= y0 and bx0 >= x0 and by1 <= y1 and bx1 <= x1][:max_exemplars]
            if not boxes:
                continue
        for mask, score in sam.detect(image[y0:y1, x0:x1], text=text, boxes_xyxy=boxes, threshold=threshold):
            if touches_inner_edge(mask, (y0, x0, y1, x1), shape):
                continue
            p = to_proposal(mask, (y0, x0), score, source, t)
            if p is not None:
                props.append(p)
    return suppress(props)


def interior_point(mask: np.ndarray) -> tuple[int, int]:
    """(y, x) of the mask pixel farthest from the background."""
    from scipy import ndimage as ndi

    dist = ndi.distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]
    y, x = np.unravel_index(int(np.argmax(dist)), dist.shape)
    return int(y), int(x)


def refine_candidates(sam: Sam3, image: np.ndarray, candidates: np.ndarray, *, context: float = 1.0,
                      min_crop: int = 256) -> list[Proposal]:
    """Three SAM hypotheses per candidate object, prompted by its box and its deepest interior point.

    Each object is seen on its own crop (box grown by ``context`` x its side on every edge, at least
    ``min_crop`` px), so a 100-px mitochondrion reaches SAM at several times its native size.
    """
    from scipy import ndimage as ndi

    h, w = image.shape
    out: list[Proposal] = []
    for k, sl in enumerate(ndi.find_objects(candidates), 1):
        if sl is None:
            continue
        by0, by1, bx0, bx1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
        side = max(by1 - by0, bx1 - bx0)
        pad = max(int(context * side), (min_crop - side) // 2, 8)
        cy0, cx0 = max(0, by0 - pad), max(0, bx0 - pad)
        cy1, cx1 = min(h, by1 + pad), min(w, bx1 + pad)
        obj = candidates[cy0:cy1, cx0:cx1] == k
        py, px = interior_point(obj)
        box = [bx0 - cx0, by0 - cy0, bx1 - cx0, by1 - cy0]
        for j, (mask, score) in enumerate(sam.refine(image[cy0:cy1, cx0:cx1], box, [[px, py]], [1])):
            p = to_proposal(mask, (cy0, cx0), score, f"refine{j}", k)
            if p is not None:
                out.append(p)
    return out


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_image(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path, allow_pickle=False)
    if path.suffix.lower() in {".tif", ".tiff"}:
        import tifffile

        return np.asarray(tifffile.imread(path))
    from PIL import Image

    with Image.open(path) as im:
        return np.asarray(im)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--image", required=True, type=Path)
    ap.add_argument("--mode", required=True,
                    help="text, exemplar or refine; several comma-separated modes share one model load and "
                         "write <out stem>.<mode>.npz each")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--target", default="mitochondrion",
                    help="text mode: organelle as the user names it (线粒体, chloroplast, ...); translated by "
                         "concept_phrase()")
    ap.add_argument("--modality", default="em", help="em or fluorescence; selects the measured phrase")
    ap.add_argument("--text", default=None, help="text mode: exact phrase, overrides --target")
    ap.add_argument("--exemplars", type=Path, help="JSON list of [y0, x0, y1, x1] image boxes")
    ap.add_argument("--candidates", type=Path, help="refine: .npy instance map on the image grid")
    ap.add_argument("--scale", type=float, default=1.0, help="resample factor applied before SAM (0.5 = half size)")
    ap.add_argument("--tile", type=int, default=SAM3_RESOLUTION)
    ap.add_argument("--overlap", type=int, default=256)
    ap.add_argument("--threshold", type=float, default=0.05)
    ap.add_argument("--context", type=float, default=1.0)
    ap.add_argument("--device", default="auto", help="auto (CUDA if available, else CPU), cuda, cpu, ...")
    ap.add_argument("--bpe", default=None, help=f"CLIP BPE vocabulary (default: {BPE_NAME} next to the checkpoint)")
    a = ap.parse_args(argv)

    modes = [m.strip() for m in a.mode.split(",") if m.strip()]
    unknown = sorted(set(modes) - {"text", "exemplar", "refine"})
    if unknown or not modes:
        raise SystemExit(f"unknown mode(s): {', '.join(unknown) or '(none)'}")
    if "exemplar" in modes and a.exemplars is None:
        raise SystemExit("--mode exemplar needs --exemplars")
    if "refine" in modes and a.candidates is None:
        raise SystemExit("--mode refine needs --candidates")
    text, text_verified = (a.text, None) if a.text else concept_phrase(a.target, a.modality)
    t0 = time.time()
    image = stretch_to_uint8(_read_image(a.image))
    full_shape = image.shape
    if a.scale != 1.0:
        from PIL import Image

        size = (max(1, round(full_shape[1] * a.scale)), max(1, round(full_shape[0] * a.scale)))
        image = np.asarray(Image.fromarray(image).resize(size, Image.BILINEAR))
    sam = Sam3(a.checkpoint, a.device, a.bpe)
    t_load = time.time() - t0
    a.out.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    seconds: dict[str, float] = {}
    for mode in modes:
        t = time.time()
        if mode == "text":
            props = tiled_detect(sam, image, text=text, tile=a.tile, overlap=a.overlap, threshold=a.threshold)
        elif mode == "exemplar":
            boxes = [tuple(round(v * a.scale) for v in b) for b in json.loads(a.exemplars.read_text())]
            props = tiled_detect(sam, image, text=None, tile=a.tile, overlap=a.overlap, threshold=a.threshold,
                                 exemplars=boxes, source="exemplar")
        else:
            cand = np.load(a.candidates, allow_pickle=False).astype(np.int32)
            if cand.shape != full_shape:
                raise SystemExit(f"candidates {cand.shape} do not match the image {full_shape}")
            if a.scale != 1.0:
                rows = np.minimum((np.arange(image.shape[0]) / a.scale).astype(int), full_shape[0] - 1)
                cols = np.minimum((np.arange(image.shape[1]) / a.scale).astype(int), full_shape[1] - 1)
                cand = cand[np.ix_(rows, cols)]
            props = refine_candidates(sam, image, cand, context=a.context)
        path = a.out if len(modes) == 1 else a.out.with_name(f"{a.out.stem}.{mode}.npz")
        np.savez_compressed(path, **pack(props), shape=np.array(image.shape, np.int32),
                            full_shape=np.array(full_shape, np.int32), scale=np.float32(a.scale))
        counts[mode] = len(props)
        seconds[mode] = round(time.time() - t, 1)
    import importlib.metadata as md

    info = {"modes": modes, "proposals": counts, "mode_seconds": seconds, "scale": a.scale, "tile": a.tile,
            "overlap": a.overlap, "threshold": a.threshold, "text": text if "text" in modes else None,
            "target": a.target if "text" in modes and not a.text else None, "modality": a.modality,
            "text_phrase_verified": text_verified if "text" in modes else None,
            "context": a.context, "checkpoint_sha256": _sha256(Path(a.checkpoint)),
            "weights_read": Path(sam.weights_path).name,
            "missing_keys": None if sam.missing_keys is None else len(sam.missing_keys),
            "sam3_version": md.version("sam3"), "torch": md.version("torch"), "device": sam.device,
            "triton_stub": sam.triton_stub, "load_s": round(t_load, 1),
            "total_s": round(time.time() - t0, 1)}
    a.out.with_suffix(".json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(json.dumps(info), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
