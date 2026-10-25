"""Three-dimensional instance measurement (axes Z,Y,X)."""
import numpy as np


def measure_volume(labels, *, voxel_size_um, min_area, class_names, instance_map, label_source):
    from scipy import ndimage as ndi
    from skimage.measure import marching_cubes, mesh_surface_area, regionprops
    spacing = None if voxel_size_um is None else tuple(float(x) for x in voxel_size_um)
    if spacing is not None and (len(spacing) != 3 or not np.all(np.isfinite(spacing)) or min(spacing) <= 0):
        raise ValueError("voxel_size_um must contain positive finite (z, y, x) spacings")
    rows, counts = [], {}
    classes = [(1, class_names[1] if len(class_names)>1 else "object")] if instance_map else list(enumerate(class_names))[1:]
    for class_id, name in classes:
        instances = labels if instance_map else ndi.label(labels == class_id)[0]
        counts[name] = 0
        for region in regionprops(instances):
            if region.area < min_area:
                continue
            row = dict(label=int(region.label), **{"class": class_id}, class_name=name,
                       volume_voxels=int(region.area), centroid_zyx=list(map(float, region.centroid)),
                       bbox_zyx=list(map(int, region.bbox)))
            if spacing is not None:
                volume = float(region.area)*float(np.prod(spacing))
                vertices, faces, _, _ = marching_cubes(np.pad(region.image.astype(np.uint8), 1), .5, spacing=spacing)
                row.update(volume_um3=volume, surface_area_um2=float(mesh_surface_area(vertices, faces)),
                           equivalent_diameter_um=float((6*volume/np.pi)**(1/3)))
            rows.append(row)
            counts[name] += 1
    return dict(per_object=rows, counts=counts, n_total=len(rows),
                n_classes_present=sum(v>0 for v in counts.values()), dimensionality=3,
                voxel_size_um=spacing, label_source=label_source,
                surface_method="marching_cubes_at_half_voxel" if spacing else None)
