"""Regressions for lossless masks, calibration, model identity and public invocation."""
from pathlib import Path
import inspect
import numpy as np
import pytest
from PIL import Image

from organelleverse import morphology
from organelleverse.morphology.mask_io import read_image, save_instances, spatial_image


def test_catalog_pins_current_models_and_distinguishes_semantics():
    rows = {r['id']: r for r in morphology.list_models()}
    assert {'cpsam_v2', 'cpdino', 'cpdino-vitb', 'vit_b_em_organelles', 'vit_l_em_organelles', 'orgsegnet-v1'} == rows.keys()
    assert rows['cpdino']['runtime_version'] == '4.2.1.1'
    assert rows['vit_b_em_organelles']['runtime_version'] == '1.8.14'
    assert rows['orgsegnet-v1']['mask_kind'] == 'semantic'
    assert rows['cpdino']['mask_kind'] == 'instance'
    assert '/resolve/main/' not in str(rows['cpdino']['files'])


def test_uint32_ids_and_multipage_tiff_are_lossless(tmp_path):
    volume = np.zeros((4, 12, 13), dtype=np.uint32)
    volume[1:3, 2:10, 3:11] = 70001
    path = save_instances(volume, tmp_path/'mask.tif')
    assert np.array_equal(read_image(path), volume)
    assert np.array_equal(spatial_image(path, is_3d_stack=True), volume)
    with pytest.raises(ValueError, match='2D input'):
        spatial_image(path, is_3d_stack=False)


def test_palette_decodes_class_instead_of_using_red_channel():
    label = np.zeros((20, 20, 3), dtype=np.uint8)
    label[2:12, 2:12] = [0, 128, 0]
    result = morphology.measure(label, watershed_classes=(), network_metrics=False)
    assert result['counts']['Mitochondria'] == 1
    assert result['per_object'][0]['area_px'] == 100


@pytest.mark.parametrize('value', [-1, 1.5, float('nan')])
def test_noninteger_or_negative_masks_are_rejected(value):
    label = np.full((8, 8), value)
    with pytest.raises(Exception, match='integer'):
        morphology.measure(label)


@pytest.mark.parametrize('scale', [0, -1, float('nan'), float('inf')])
def test_physical_scale_is_positive_and_finite(scale):
    with pytest.raises(Exception):
        morphology.measure(np.zeros((8, 8), np.uint8), pixel_size_um=scale)


def test_calibrated_2d_and_3d_measurements_use_different_units(tmp_path):
    mask = np.zeros((20, 20), np.uint32)
    mask[2:12, 3:13] = 70001
    result = morphology.measure(mask, instance_map=True, pixel_size_um=.2, network_metrics=False)
    assert result['per_object'][0]['area_um2'] == pytest.approx(4)
    volume = np.zeros((4, 14, 14), np.uint32)
    volume[1:3, 2:12, 2:12] = 70001
    path = save_instances(volume, tmp_path/'volume.tif')
    result = morphology.measure(path, instance_map=True, voxel_size_um=(2., .5, .5))
    row = result['per_object'][0]
    assert result['dimensionality'] == 3
    assert row['volume_um3'] == pytest.approx(100)
    assert row['surface_area_um2'] > 0
    assert 'area_um2' not in row and 'perimeter_um' not in row


def test_input_domain_and_shape_errors_do_not_silently_drop_objects():
    mask = np.full((10, 10), 12, np.uint8)
    with pytest.raises(Exception, match='instance_map'):
        morphology.measure(mask)
    with pytest.raises(Exception, match='identical spatial shape'):
        morphology.measure(np.zeros((10, 10), np.uint8), intensity_image=np.zeros((9, 9)))


def test_missing_bundle_is_explicit_and_does_not_download(tmp_path):
    from organelleverse.morphology.models import make_executor
    status = morphology.model_status('cpdino-vitb', tmp_path)
    assert status['weights_present'] is False
    with pytest.raises(FileNotFoundError):
        make_executor('cpdino-vitb', tmp_path)


def test_pinned_model_rejects_a_replaced_checkpoint(tmp_path, monkeypatch):
    import organelleverse.morphology.models as models
    checkpoint=tmp_path/'cpsam_v2'
    checkpoint.write_bytes(b'not the publisher checkpoint')
    row=next(r for r in models.list_models() if r['id']=='cpsam_v2')
    monkeypatch.setattr(models, 'model_status', lambda *_: dict(model=row, weights_present=True,
        runtime_matches=True, files=[str(checkpoint)]))
    with pytest.raises(ValueError,match='publisher checkpoint'):
        models.make_executor('cpsam_v2',tmp_path)


def test_packaged_guide_matches_documentation_and_is_readable():
    import organelleverse
    from organelleverse._layout import checkout_root
    root=checkout_root()
    assert morphology.model_guide() == (root/'docs/microscopy-models.md').read_text(encoding='utf-8')


def test_instance_overlay_retains_large_ids_without_semantic_colors():
    label=np.zeros((12,12),np.uint32)
    label[2:10,2:10]=70001
    image=np.zeros((12,12),np.uint8)
    with pytest.raises(Exception,match='instance_map'):
        morphology.overlay(image,label)
    overlay=morphology.overlay(image,label,instance_map=True)
    assert overlay.shape==(12,12,3)
    assert overlay[4,4].any()


def test_public_model_call_keeps_injected_executor_and_tiff_metadata(tmp_path):
    image = tmp_path/'image.png'
    Image.fromarray(np.zeros((20,20), np.uint8)).save(image)
    seen = []
    def injected(model_type, device, paths, out):
        seen.append((model_type, paths))
        labels = np.zeros((20,20), np.uint32)
        labels[2:12,2:12] = 70001
        return [save_instances(labels, Path(out)/'labels.tif')]
    result = morphology.segment(image, model_id='cpdino-vitb', executor=injected,
                                device='cpu', pixel_size_um=.2)
    assert result.status == 'ok'
    assert seen[0][0] == 'cpdino-vitb'
    assert result.metrics['backend'] == 'cellpose'
    assert result.metrics['model']['id'] == 'cpdino-vitb'
    assert result.metrics['model']['execution'] == 'injected'
    assert result.metrics['per_image'][0]['counts'] == {'object': 1}
    assert result.artifacts[0].format == 'tiff'


def test_registered_parameters_follow_public_signatures():
    from organelleverse.operations import registry
    for op, function, fields in [
        ('morphology.segment', morphology.segment, ('model_id','model_dir','voxel_size_um')),
        ('morphology.measure', morphology.measure, ('voxel_size_um',)),
    ]:
        bound = registry.require(op)
        names = {p.name for p in bound.spec.binding.parameters}
        assert set(fields) <= names
        assert set(fields) <= inspect.signature(function).parameters.keys()


def test_morphology_generator_and_registry_use_the_same_calibration_contract():
    import importlib.util
    import sys
    import tomllib
    import organelleverse
    from organelleverse.operations import registry
    from organelleverse._layout import checkout_root
    root = checkout_root()
    spec = importlib.util.spec_from_file_location('_morphology_generator', root/'scripts/capabilities/build_restored_bundles.py')
    generator = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = generator
    spec.loader.exec_module(generator)
    overrides = generator._load_overrides('morphology')
    records = generator._load_domain_records(root/'docs/operations/restored-capabilities.toml', 'morphology')
    for record in records:
        if record['id'] not in ('morphology.measure','morphology.segment','morphology.overlay'):continue
        source_root = Path(organelleverse.__file__).resolve().parent
        plan = generator._plan_bundle(record,source_root,overrides)
        assert isinstance(plan,generator.PlannedBundle), plan
        actual = registry.require(record['id']).spec.binding.parameters
        assert [(p.name,p.codec,p.path_role) for p in plan.parameters] == [(p.name,p.codec,p.path_role) for p in actual]
