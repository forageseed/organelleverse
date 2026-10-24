"""Phenotypic color traits for segmented biological images (BioIMA-style).

BioIMA (Qumu et al. 2026, bioRxiv doi:10.64898/2026.08.30.747465) is a
desktop tool for standardized phenotyping of plant images: prompt-based
segmentation plus automated trait computation. The segmentation side is
already covered by the workbench's existing backends (micro-SAM prompts,
OrgSegNet, fusion3); this module ports the trait side that the morphology
suite did not have — **color phenotyping**:

- per-object mean RGB + HEX + CIE L*a*b* (D65/2° via skimage);
- K-means color distribution over the object pixels (cluster centers in
  RGB/HEX + population share), BioIMA's "color distribution" trait;
- white/gray reference correction: per-channel gains from a user-supplied
  reference region (white card / gray card in the scene) reduce lighting
  and scanner color bias before measuring, exactly the BioIMA correction
  step.

Geometry traits (area, perimeter, axes, circularity, equivalent diameter)
are already produced by :func:`organelleverse.morphology.measure.measure`
and are intentionally not duplicated here; callers combine both.

References
----------
- Qumu, X. et al. (2026) *BioIMA: a one-click desktop tool for standardized
  extraction of phenotypic traits from biological images.* bioRxiv
  doi:10.64898/2026.08.30.747465; code github.com/jingwanglab/BioIMA
  (Avalonia/.NET desktop app — capabilities re-implemented here, not wrapped).
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["DEFAULT_K", "color_traits", "white_reference_correction"]

#: Default cluster count for the K-means color distribution (BioIMA exposes
#: a selectable K in its UI; 5 matches its default granularity).
DEFAULT_K = 5


def _as_float_rgb(image: np.ndarray) -> np.ndarray:
    """Normalize an image array to float RGB in [0, 1], shape (H, W, 3)."""
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"expected a 2D grayscale or (H, W, >=3) image, got {arr.shape}")
    arr = arr[..., :3]
    if arr.dtype.kind == "f":
        return np.clip(arr.astype(np.float64), 0.0, 1.0)
    info = np.iinfo(arr.dtype) if arr.dtype.kind in "ui" else None
    scale = float(info.max) if info else 255.0
    return np.clip(arr.astype(np.float64) / scale, 0.0, 1.0)


def _hex(rgb01: np.ndarray) -> str:
    rgb8 = np.rint(np.clip(rgb01, 0.0, 1.0) * 255).astype(int)
    return "#{:02X}{:02X}{:02X}".format(*rgb8)


def white_reference_correction(
    image: np.ndarray,
    reference_mask: np.ndarray,
    *,
    target: float = 1.0,
) -> np.ndarray:
    """Per-channel gain correction from a white/gray reference region.

    Parameters
    ----------
    image
        Grayscale or RGB image (any dtype; floats assumed [0, 1]).
    reference_mask
        Boolean array, same H×W, marking the reference card pixels.
    target
        Value the reference should reach after correction (1.0 = white,
        0.5 = mid-gray card). Must be > 0.

    Returns
    -------
    np.ndarray
        Float RGB [0, 1], corrected. Gain = target / channel mean over the
        reference; each channel is scaled independently and clipped.
    """
    rgb = _as_float_rgb(image)
    mask = np.asarray(reference_mask, dtype=bool)
    if mask.shape != rgb.shape[:2]:
        raise ValueError(f"reference mask {mask.shape} does not match image {rgb.shape[:2]}")
    if not mask.any():
        raise ValueError("reference mask is empty")
    if target <= 0:
        raise ValueError("target must be > 0")
    means = rgb[mask].mean(axis=0)
    means = np.where(means < 1e-6, 1e-6, means)
    return np.clip(rgb * (target / means), 0.0, 1.0)


def _kmeans(
    pixels: np.ndarray, k: int, *, seed: int = 13, iters: int = 50
) -> tuple[np.ndarray, np.ndarray]:
    """Small deterministic K-means (scipy backend when available).

    Returns (centers k×3 in [0,1], counts k).
    """
    if pixels.shape[0] < k:
        k = max(1, pixels.shape[0])
    try:
        from scipy.cluster.vq import kmeans2

        centers, labels = kmeans2(
            pixels, k, minit="++", seed=seed, iter=iters
        )
        counts = np.bincount(labels, minlength=k)
        return np.clip(np.asarray(centers, dtype=np.float64), 0.0, 1.0), counts
    except ImportError:
        rng = np.random.default_rng(seed)
        centers = pixels[rng.choice(pixels.shape[0], size=k, replace=False)].copy()
        for _ in range(iters):
            dist = ((pixels[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
            labels = dist.argmin(axis=1)
            new_centers = np.array(
                [pixels[labels == i].mean(axis=0) if (labels == i).any() else centers[i] for i in range(k)]
            )
            if np.allclose(new_centers, centers):
                break
            centers = new_centers
        counts = np.bincount(labels, minlength=k)
        return np.clip(centers, 0.0, 1.0), counts


def color_traits(
    image: np.ndarray,
    label_map: np.ndarray | None = None,
    *,
    k: int = DEFAULT_K,
    white_reference: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute BioIMA-style color traits for an image / its objects.

    Parameters
    ----------
    image
        Grayscale or RGB image array (uint8/uint16/float).
    label_map
        Optional integer instance/semantic label map (0 = background). When
        given, per-object color means are reported for every label > 0;
        when None, the whole image is treated as one object.
    k
        K-means cluster count for the color distribution.
    white_reference
        Optional boolean mask of a white/gray reference region. When given,
        all color traits are computed on the corrected image.

    Returns
    -------
    dict
        ``image_hex``, ``image_lab``, ``image_kmeans`` (centers+shares),
        and ``objects`` — a list of ``{label, n_pixels, rgb, hex, lab,
        kmeans}`` entries. ``lab`` is ``[L*, a*, b*]`` (D65/2°, skimage).
    """
    rgb = _as_float_rgb(image)
    if white_reference is not None:
        rgb = white_reference_correction(rgb, white_reference)

    from skimage.color import rgb2lab

    lab_all = rgb2lab(rgb)

    labels = np.asarray(label_map) if label_map is not None else np.ones(rgb.shape[:2], dtype=np.int32)
    if labels.shape != rgb.shape[:2]:
        raise ValueError(f"label map {labels.shape} does not match image {rgb.shape[:2]}")

    def _traits_for(mask: np.ndarray) -> dict[str, Any]:
        pixels = rgb[mask]
        if pixels.size == 0:
            return {"n_pixels": 0, "rgb": None, "hex": None, "lab": None, "kmeans": []}
        mean_rgb = pixels.mean(axis=0)
        mean_lab = lab_all[mask].mean(axis=0)
        centers, counts = _kmeans(pixels, k)
        order = np.argsort(-counts)
        total = float(counts.sum())
        clusters = [
            {
                "rgb": [round(float(c), 4) for c in centers[i]],
                "hex": _hex(centers[i]),
                "share": round(float(counts[i]) / total, 4),
            }
            for i in order
        ]
        return {
            "n_pixels": int(mask.sum()),
            "rgb": [round(float(v), 4) for v in mean_rgb],
            "hex": _hex(mean_rgb),
            "lab": [round(float(v), 2) for v in mean_lab],
            "kmeans": clusters,
        }

    whole = _traits_for(np.ones(rgb.shape[:2], dtype=bool))
    objects = []
    for label in sorted(int(v) for v in np.unique(labels) if v > 0):
        entry = {"label": label}
        entry.update(_traits_for(labels == label))
        objects.append(entry)

    return {
        "k": k,
        "white_reference_corrected": white_reference is not None,
        "image": whole,
        "objects": objects,
    }
