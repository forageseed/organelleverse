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
    over each 2D image and saves each label map as a uint32 TIFF, returning the label-map
    paths — the same return contract as the OrgSegNet default executor, so
    downstream measurement consumes both backends identically.

    AMG produces *instance* masks (one integer id per detected object, 0 =
    background) without semantic classes. The saved label map keeps those
    instance ids; class assignment is a downstream, explicitly separate
    decision this executor never fabricates. Consumers measure instance maps
    with ``measure(..., instance_map=True)``.
    """
    import importlib.util
    if importlib.util.find_spec("micro_sam") is None:
        import subprocess
        from .install import check_backend
        runtime = check_backend("micro_sam", scan_envs=True)
        if not runtime["installed"] or not runtime.get("env"):
            raise RuntimeError("microSAM runtime is not installed")
        subprocess.run([runtime["path"], "-m", "organelleverse.morphology.microsam", "--worker", model_type, device, str(out_dir), *image_paths], check=True)
        outputs = [Path(out_dir) / (Path(path).stem + "_label.tif") for path in image_paths]
        if not all(path.is_file() for path in outputs):
            raise RuntimeError("microSAM did not produce the requested label maps")
        return outputs
    import numpy as np
    import torch
    from micro_sam.instance_segmentation import (  # type: ignore
        AutomaticMaskGenerator,
    )
    from micro_sam.util import get_sam_model  # type: ignore
    from .mask_io import save_instances, spatial_image

    predictor = get_sam_model(model_type=model_type, device=torch.device(device))
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
        image = spatial_image(img_path, is_3d_stack=False)
        if image.ndim == 3:
            image = image[..., :3].mean(axis=-1)
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
        p = save_instances(label_map, out / (Path(img_path).stem + "_label.tif"))
        label_paths.append(p)
    return label_paths


def execute_offline_micro_sam(model_type, checkpoint, decoder, device, image_paths, out_dir,
                              *, is_3d_stack=False):
    """micro-SAM 1.8.14 AIS with local encoder/decoder; preserve full volumes."""
    from micro_sam.util import get_sam_model
    import torch
    from micro_sam.automatic_segmentation import get_predictor_and_segmenter, automatic_instance_segmentation
    from .mask_io import save_instances, spatial_image
    model_device = torch.device(device)
    predictor, state = get_sam_model(model_type=model_type, checkpoint_path=checkpoint,
                                    decoder_path=decoder, device=model_device, return_state=True)
    predictor, segmenter = get_predictor_and_segmenter(
        model_type, predictor=predictor, state=state, device=model_device, segmentation_mode="ais")
    outputs = []
    for index, path in enumerate(image_paths):
        image = spatial_image(path, is_3d_stack=is_3d_stack)
        labels = automatic_instance_segmentation(predictor, segmenter, image,
                                                ndim=3 if is_3d_stack else 2, verbose=False)
        outputs.append(save_instances(labels, Path(out_dir)/f"{index:04d}_{Path(path).stem}_label.tif"))
    return outputs


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 6 or sys.argv[1] != "--worker":
        raise SystemExit("Expected --worker MODEL DEVICE OUTPUT IMAGE...")
    default_micro_sam_executor(sys.argv[2], sys.argv[3], sys.argv[5:], sys.argv[4])
