"""micro-SAM segmentation backend executor for the morphology suite.

micro-SAM (uSAM, Archit et al. *Nat. Methods* 2025) fine-tunes Segment
Anything on 17,000+ microscopy images and ships dedicated EM organelle models
(``vit_*_em_organelles``), explicitly supporting organelle segmentation in
electron microscopy, including 3D stacks. Unlike OrgSegNet it installs into
modern environments (``pip install micro-sam``); its batch entry point is the
registered console script ``micro_sam.automatic_segmentation``.

This module mirrors :func:`organelleverse.morphology.orgseg.default_executor`:
a single default executor callable that :func:`.segment` may receive via its
``executor`` hook when ``backend="micro_sam"``. Nothing here is imported at
module import time beyond the standard library — the micro-SAM / torch stack
is lazily imported inside the executor so the morphology suite stays cheap to
import without the backend installed.

References
----------
- Archit, A. et al. (2025) *Segment Anything for Microscopy.* Nat. Methods
  22:579-591. doi:10.1038/s41592-024-02580-4
- micro-SAM repo: https://github.com/computational-cell-analytics/micro-sam
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["DEFAULT_EM_ORGANELLE_MODEL", "default_micro_sam_executor"]

#: Default model series for plant EM organelle segmentation: micro-SAM's
#: generalist EM-organelle model (not the light-microscopy series).
DEFAULT_EM_ORGANELLE_MODEL = "vit_l_em_organelles"


def default_micro_sam_executor(
    model_type: str,
    device: str,
    image_paths: list[str],
    out_dir: str | Path,
) -> list[Path]:
    """Default micro-SAM automatic-segmentation executor.

    Runs Automatic Mask Generation (AMG) with a micro-SAM EM organelle model
    over each image and saves each label map as a PNG, returning the label-map
    paths — the same return contract as the OrgSegNet default executor, so
    downstream measurement consumes both backends identically.

    AMG produces *instance* masks (one integer id per detected object, 0 =
    background) without semantic classes. The saved label map keeps those
    instance ids; class assignment is a downstream, explicitly separate
    decision this executor never fabricates. Consumers measure instance maps
    with ``measure(..., instance_map=True)``.
    """
    import numpy as np
    from micro_sam.instance_segmentation import (  # type: ignore
        AutomaticMaskGenerator,
    )
    from micro_sam.util import get_sam_model  # type: ignore
    from PIL import Image

    predictor = get_sam_model(model_type=model_type, device=device)
    # micro-sam 1.8.x: construct the AMG directly (the old get_amg() factory
    # is gone); threshold/point defaults are the library's own.
    amg: AutomaticMaskGenerator = AutomaticMaskGenerator(predictor)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    label_paths: list[Path] = []
    for img_path in image_paths:
        # micro-sam 1.8.x treats a 3-axis array as a (Z, Y, X) stack — an
        # (H, W, 3) RGB image would be embedded as H slices. EM input is
        # single-channel anyway: pass 2D (H, W); the library expands
        # channels internally.
        image = np.array(Image.open(img_path).convert("L"))
        amg.initialize(image, verbose=False)
        generated = amg.generate()
        if isinstance(generated, np.ndarray):
            # 1.8.x default output_mode: already-merged instance map
            label_map = generated.astype(np.uint32)
        else:
            # older dict form: one entry per instance with a binary mask
            label_map = np.zeros(image.shape[:2], dtype=np.uint32)
            for instance_id, instance in enumerate(generated, start=1):
                label_map[instance["segmentation"]] = instance_id
        p = out / (Path(img_path).stem + "_label.png")
        Image.fromarray(label_map.astype(np.uint16)).save(str(p))
        label_paths.append(p)
    return label_paths
