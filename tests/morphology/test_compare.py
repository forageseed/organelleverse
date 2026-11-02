import csv

import pytest
from scipy import stats

from organelleverse.core.errors import OrganelleInputError
from organelleverse.morphology.compare import compare_groups, markdown_tables, write_tables


def _obj(area, *, bbox=(10, 10, 20, 20), cls="rod"):
    return {"area_um2": area, "equivalent_diameter_um": 2 * (area / 3.14159) ** 0.5, "aspect_ratio_imagej": 1.5,
            "circularity": 0.8, "bbox": bbox, "morphology_class": cls}


def _image(name, sample, group, areas, **extra):
    return {"image": name, "sample": sample, "group": group, "pixel_size_um": 0.01, "image_shape": (100, 100),
            "objects": [_obj(a) for a in areas], **extra}


def _design():
    images = []
    for g, scale in (("WT", 1.0), ("mut", 1.6)):
        for s in range(3):
            for i in range(2):
                base = scale * (0.20 + 0.02 * s)
                images.append(_image(f"{g}-{s}-{i}", f"{g}-plant{s}", g, [base, base + 0.02 * i, base + 0.05]))
    return images


def test_groups_are_compared_on_samples_not_images_or_objects():
    result = compare_groups(_design(), reference_group="WT")
    assert [g["samples"] for g in result["group_table"]] == [3, 3]
    assert [g["images"] for g in result["group_table"]] == [6, 6]
    area = next(c for c in result["comparisons"] if c["metric"] == "area_um2")
    assert (area["n"], area["n_reference"]) == (3, 3)
    mut = [r["area_um2"] for r in result["sample_table"] if r["group"] == "mut"]
    wt = [r["area_um2"] for r in result["sample_table"] if r["group"] == "WT"]
    expected = stats.ttest_ind(mut, wt, equal_var=False)
    assert area["p"] == pytest.approx(expected.pvalue)
    assert area["ci95_low"] < area["difference"] < area["ci95_high"]
    assert area["ratio"] == pytest.approx(sum(mut) / sum(wt))
    # Holm across the metrics of the mut-vs-WT family never lowers a p value.
    assert all(c["p_holm"] is None or c["p_holm"] >= c["p"] for c in result["comparisons"])
    # Image metrics: 3 objects on 1 µm² -> 300 per 100 µm².
    assert result["image_table"][0]["objects_per_100um2"] == pytest.approx(300.0)
    assert "objects_per_100um2" in result["metrics"] and result["metrics"]["area_um2"]["unit"] == "µm²"


def test_one_sample_per_group_is_descriptive_only():
    images = [_image("a", "p1", "WT", [0.2, 0.3]), _image("b", "p2", "mut", [0.4, 0.5])]
    result = compare_groups(images)
    assert result["reference_group"] == "WT"
    assert {c["status"] for c in result["comparisons"]} == {"descriptive_only"}
    assert all(c["p"] is None for c in result["comparisons"])
    assert any("descriptive" in w for w in result["warnings"])


def test_missing_samples_are_flagged_as_images_used_as_replicates():
    images = [{**_image("a", None, "WT", [0.2]), "sample": None}, {**_image("b", None, "mut", [0.3]), "sample": None}]
    result = compare_groups(images)
    assert any("independent biological replicate" in w for w in result["warnings"])


def test_a_sample_cannot_sit_in_two_groups():
    with pytest.raises(OrganelleInputError) as error:
        compare_groups([_image("a", "p1", "WT", [0.2]), _image("b", "p1", "mut", [0.3])])
    assert error.value.code == "morphology.compare_sample_in_two_groups"


def test_edge_objects_are_counted_but_can_be_left_out_of_sizes():
    image = _image("a", "p1", "WT", [0.2])
    image["objects"].append(_obj(5.0, bbox=(0, 40, 30, 60)))
    kept = compare_groups([image])
    left = compare_groups([image], exclude_edge=True)
    assert kept["image_table"][0]["edge_objects"] == 1
    assert kept["image_table"][0]["objects_per_image"] == left["image_table"][0]["objects_per_image"] == 2
    assert kept["image_table"][0]["area_um2"] == pytest.approx(2.6)
    assert left["image_table"][0]["area_um2"] == pytest.approx(0.2)


def test_uncalibrated_images_are_compared_in_pixels_with_a_warning():
    image = {"image": "a", "group": "WT", "objects": [{"area_px": 400, "circularity": 0.7}]}
    result = compare_groups([image])
    assert result["calibrated"] is False
    assert result["metrics"]["area_um2"]["field"] == "area_px" and result["metrics"]["area_um2"]["unit"] == "px²"
    assert "objects_per_100um2" not in result["metrics"]
    assert any("pixels" in w for w in result["warnings"])


def test_intensity_is_compared_only_when_asked():
    images = [_image("a", "p1", "WT", [0.2])]
    images[0]["objects"][0]["mean_intensity"] = 120.0
    assert "mean_intensity" not in compare_groups(images)["metrics"]
    asked = compare_groups(images, compare_intensity=True)
    assert "mean_intensity" in asked["metrics"] and any("acquisition" in w for w in asked["warnings"])


def test_tables_are_written_and_rendered(tmp_path):
    result = compare_groups(_design(), reference_group="WT")
    paths = write_tables(result, tmp_path)
    with open(paths["comparisons.csv"], encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["metric"] for row in rows} >= {"area_um2", "objects_per_image"}
    text = markdown_tables(result)
    assert "| Area | µm² |" in text and "mut vs WT (n=3 vs 3)" in text
