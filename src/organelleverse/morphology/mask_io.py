"""Lossless microscopy array I/O with explicit spatial dimensionality."""
from pathlib import Path
import numpy as np


def read_image(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() in (".tif", ".tiff"):
        import tifffile
        return tifffile.imread(path)
    from PIL import Image
    with Image.open(path) as image:
        return np.array(image)


def spatial_image(path: str | Path, *, is_3d_stack: bool) -> np.ndarray:
    image = read_image(path)
    if is_3d_stack:
        if image.ndim != 3:
            raise ValueError("A 3D input must be a single-channel (Z, Y, X) volume")
    elif image.ndim != 2 and not (image.ndim == 3 and image.shape[-1] in (3, 4)):
        raise ValueError("A 2D input must be (Y, X) or RGB(A); declare is_3d_stack=True for a volume")
    if image.ndim == 3 and not is_3d_stack and image.shape[-1] == 4:
        image = image[..., :3]
    return image


def save_instances(labels: np.ndarray, path: Path) -> Path:
    """Save integer instance IDs as uint32 TIFF, never a truncated uint16 PNG."""
    import tifffile
    labels = np.asarray(labels)
    if labels.ndim not in (2, 3) or labels.dtype.kind not in "uib" or np.any(labels < 0):
        raise ValueError("Instance labels must be a nonnegative integer 2D/3D array")
    if labels.size and int(labels.max()) > np.iinfo(np.uint32).max:
        raise ValueError("Instance IDs exceed uint32 TIFF capacity")
    path = path.with_suffix(".tif")
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, labels.astype(np.uint32), photometric="minisblack",
                     metadata={"axes": "ZYX" if labels.ndim == 3 else "YX"})
    return path
