# Microscopy segmentation and measurement

OV first segments an image, then measures the resulting mask. Area and shape
measurements do not establish segmentation accuracy or biological mechanisms
such as mitochondrial fusion. Inspect overlays and validate representative
images against independently annotated masks before interpreting results.

The built-in catalog is pinned as checked on 2026-10-25:

| Profile | Runtime | Appropriate use | Mask semantics |
| --- | --- | --- | --- |
| `orgsegnet-v1` | upstream legacy OrgSegNet stack | Plant TEM thin sections | Four organelle classes |
| `vit_b_em_organelles`, `vit_l_em_organelles` | micro-SAM 1.8.14 | EM organelle instances | Instance IDs; no organelle classes |
| `cpsam_v2` | Cellpose 4.2.1.1 | General light microscopy cells/objects | Instance IDs |
| `cpdino`, `cpdino-vitb` | Cellpose 4.2.1.1 | General light microscopy cells/objects | Instance IDs |

Cellpose is not an organelle classifier. Its suitability for small, branching
or dim organelles requires task-specific validation. Newer general SAM video
models are not silently substituted for these scientific microscopy models.

## Installation and portable offline weights

Ordinary OV installation includes the catalog, Python adapters and this usage
documentation. Large weights and incompatible upstream runtimes are separate.
Use a separate environment per backend: legacy OrgSegNet cannot share a modern
PyTorch stack, and micro-SAM and Cellpose depend on different distributions of
the `segment_anything` namespace. No model is downloaded implicitly by the
`model_id` inference path.

```bash
# In the compute server's chosen environment, install the appropriate extra:
uv pip install 'organelleverse[morphology-cellpose,morphology-cpdino,morphology-models]'
# A separate micro-SAM environment:
uv pip install 'organelleverse[morphology-microsam,morphology-models]'

# Prepare all catalog weights explicitly on the server (several GB):
python -m organelleverse.morphology.models download --directory /data/ov-models
# Or select profiles with repeated --model options.
```

The bundle contains original checkpoints, version/source/license metadata and
publisher-provided integrity checks. `HF_ENDPOINT` can explicitly select a
Hugging Face download mirror. Copy the bundle to an offline compute machine
and set `ORGANELLEVERSE_MODEL_DIR`, or pass `model_dir`. Each model keeps its
upstream license; OV's license does not relicense weights. OrgSegNet also
requires its upstream implementation and configuration; the existing Python
executor/configuration interface remains available.

CP-DINO additionally needs the pinned upstream DINOv3 implementation; the
`morphology-cpdino` extra installs it. DINOv3 retains its own license agreement,
linked in each CP-DINO profile, in addition to the Cellpose project's notices.
For Cellpose 3D, X/Y pixels must have equal spacing; Z anisotropy is passed to
the model from the declared voxel calibration.

```python
from organelleverse import morphology as morph

print(morph.list_models())
print(morph.model_status('vit_b_em_organelles', '/data/ov-models'))
result = morph.segment('plant-tem.tif', model_id='vit_b_em_organelles',
                       model_dir='/data/ov-models', pixel_size_um=0.01)
print(result.status, result.metrics)

# Fluorescence cell/object images, after validating the chosen model:
result = morph.segment('fluorescence.tif', model_id='cpdino-vitb',
                       model_dir='/data/ov-models', pixel_size_um=0.2)
```

With `model_id` omitted, the existing explicit backend/executor interface keeps
its planning behavior. Python-only executor injection remains supported. A
plan (`ran=False`) is not a completed segmentation. Always inspect status,
errors, artifacts and label semantics.

2D calibration is positive finite micrometers per pixel. For a single-channel
3D `(Z,Y,X)` TIFF declare `is_3d_stack=True`; measurements additionally require
positive `voxel_size_um=(z,y,x)`. 3D output reports volume and marching-cubes
surface area rather than 2D area/perimeter. Intensity measurements currently
support 2D only. Instance masks are saved as uint32 TIFF to retain IDs above
65535 and all Z planes. No first plane is selected implicitly.

Upstream references: [OrgSegNet](https://github.com/yzy0102/OrgSegNet),
[micro-SAM 1.8.14](https://github.com/computational-cell-analytics/micro-sam/releases/tag/v1.8.14),
[Cellpose model documentation](https://cellpose.readthedocs.io/en/latest/models.html),
[Cellpose weights](https://huggingface.co/mouseland/cellpose-sam).

For an instance-mask overlay, call `morph.overlay(image, mask, instance_map=True)`.
Instance colors do not identify organelle classes. Select a Z plane explicitly
for visualization of a volume.
