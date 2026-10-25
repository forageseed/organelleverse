"""Native Cellpose 4.2.1.1 inference; generic instance masks, no invented classes."""
from pathlib import Path

from .mask_io import save_instances, spatial_image


def execute_cellpose(checkpoint: Path, device: str, image_paths: list[str], out_dir: str | Path,
                     *, is_3d_stack: bool = False, voxel_size_um=None) -> list[Path]:
    import torch
    from cellpose.models import CellposeModel
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if is_3d_stack and voxel_size_um is not None and voxel_size_um[1] != voxel_size_um[2]:
        raise ValueError("Cellpose 3D requires equal X/Y pixel sizes; resample explicitly before segmentation")
    model = CellposeModel(pretrained_model=str(checkpoint), device=torch.device(device))
    outputs = []
    for index, path in enumerate(image_paths):
        image = spatial_image(path, is_3d_stack=is_3d_stack)
        labels, _flows, _styles = model.eval(
            image, do_3D=is_3d_stack, channel_axis=-1 if image.ndim == 3 and not is_3d_stack else None,
            z_axis=0 if is_3d_stack else None,
            anisotropy=voxel_size_um[0]/voxel_size_um[1] if is_3d_stack and voxel_size_um else None,
        )
        outputs.append(save_instances(labels, Path(out_dir)/f"{index:04d}_{Path(path).stem}_label.tif"))
    return outputs
