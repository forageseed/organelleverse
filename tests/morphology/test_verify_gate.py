"""T4b: the multimodal verification gate (task brief 2026-08-17).

All model access goes through an injected client — these tests stub it and
never touch the network. The gate's honest-contract rules are pinned:
verdicts only (never mask writes), pinned verifier identity per record,
declared vision capability (never probed), cancel -> not_verified,
cross-model aggregation refused, batch resume via state file.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from organelleverse.capabilities.parser import parse_capability_bundle
from organelleverse.morphology.verify import (
    MODEL_CAPABILITIES,
    model_supports_vision,
    summarize_verdicts,
    verify,
    verify_batch,
)
from tests._paths import PROJECT_ROOT


def _semantic_setup(tmp_path: Path) -> tuple[Path, Path]:
    """One EM image with a two-class semantic label map (3 objects)."""
    image = np.full((48, 48), 120, dtype=np.uint8)
    label = np.zeros((48, 48), dtype=np.uint8)
    label[4:12, 4:12] = 1  # Chloroplast object A
    label[18:24, 4:12] = 1  # Chloroplast object B (separate component)
    label[30:36, 30:36] = 2  # Mitochondria
    image[4:12, 4:12] = 200
    image[18:24, 4:12] = 200
    image[30:36, 30:36] = 220
    label_path = tmp_path / "label.png"
    image_path = tmp_path / "image.png"
    Image.fromarray(label).save(label_path)
    Image.fromarray(image).save(image_path)
    return label_path, image_path


def _stub_client(responses: dict[str, str], calls: list[tuple]):
    def client(model: str, prompt: str, crop_path: Path) -> str:
        calls.append((model, prompt, crop_path.name))
        return responses.get(crop_path.name, "agree")

    return client


def test_bundle_declares_honest_contract_flags() -> None:
    bundle = parse_capability_bundle(
        PROJECT_ROOT / "src/organelleverse/capabilities/morphology-verify/capability.toml"
    )
    spec = bundle.contract
    assert spec.operation_id == "morphology.verify"
    assert spec.deterministic is False
    assert spec.idempotent is False
    assert any(effect.value == "network" for effect in spec.side_effects)
    assert spec.fallback.allowed is False


def test_verdict_flow_with_stubbed_client(tmp_path: Path) -> None:
    label_path, image_path = _semantic_setup(tmp_path)
    calls: list[tuple] = []
    # Object ids: class_idx*1000 + component. 1001/1002 chloroplasts, 2001 mito.
    client = _stub_client({"object-2001.png": "disagree"}, calls)
    result = verify(
        label_path,
        image_path,
        client=client,
        model="gpt-4o",
        model_version="gpt-4o-2026-01",
    )
    assert result.status == "ok"
    assert result.metrics["verified"] is True
    assert result.metrics["n_objects"] == 3
    assert result.metrics["n_model_calls"] == 3
    verdicts = {v["object_label"]: v for v in result.metrics["verdicts"]}
    assert verdicts[1001]["verdict"] == "agree"
    assert verdicts[2001]["verdict"] == "disagree"
    assert tuple(result.metrics["flagged_for_correction"]) == (2001,)
    # The model was asked per-object, with the closed question — never the
    # whole image.
    assert len(calls) == 3
    assert all("agree / disagree / uncertain" in prompt for _, prompt, _ in calls)


def test_every_record_pins_model_version_prompt_hash_and_crop(tmp_path: Path) -> None:
    label_path, image_path = _semantic_setup(tmp_path)
    result = verify(
        label_path,
        image_path,
        client=_stub_client({}, []),
        model="gpt-4o",
        model_version="gpt-4o-2026-01",
    )
    for record in result.metrics["verdicts"]:
        assert record["model_id"] == "gpt-4o"
        assert record["model_version"] == "gpt-4o-2026-01"
        assert len(record["prompt_hash"]) == 64
        assert len(record["bbox"]) == 4
        assert record["verified_at"]
    assert result.provenance.software_versions["verifier_model"] == "gpt-4o"


def test_disagree_with_fail_strategy_fails_closed(tmp_path: Path) -> None:
    label_path, image_path = _semantic_setup(tmp_path)
    client = _stub_client({"object-1001.png": "uncertain"}, [])
    result = verify(
        label_path,
        image_path,
        client=client,
        model="gpt-4o",
        model_version="v1",
        on_disagree="fail",
    )
    assert result.status == "failed"
    assert result.errors[0].code == "morphology.verify_contested"
    assert tuple(result.errors[0].details["contested_labels"]) == (1001,)


def test_uncertain_is_routed_to_humans_never_counted_as_pass(tmp_path: Path) -> None:
    label_path, image_path = _semantic_setup(tmp_path)
    client = _stub_client({"object-1002.png": "I think maybe it looks fine?"}, [])
    result = verify(
        label_path,
        image_path,
        client=client,
        model="gpt-4o",
        model_version="v1",
    )
    assert result.status == "ok"
    verdicts = {v["object_label"]: v for v in result.metrics["verdicts"]}
    # An unparseable response is not a verdict: recorded uncertain, raw kept.
    assert verdicts[1002]["verdict"] == "uncertain"
    assert "maybe" in verdicts[1002]["raw_response"]
    assert tuple(result.metrics["flagged_for_correction"]) == (1002,)


def test_undeclared_model_fails_without_probing(tmp_path: Path) -> None:
    label_path, image_path = _semantic_setup(tmp_path)
    calls: list[tuple] = []
    result = verify(
        label_path,
        image_path,
        client=_stub_client({}, calls),
        model="text-only-model-9",
        model_version="v1",
    )
    assert result.status == "failed"
    assert result.errors[0].code == "morphology.verifier_not_multimodal"
    offered = result.errors[0].details["vision_capable_models"]
    assert "gpt-4o" in offered
    assert calls == [], "no probe call may be sent to find out"
    assert model_supports_vision("text-only-model-9") is False
    assert model_supports_vision("gpt-4o") is True
    assert set(MODEL_CAPABILITIES)  # registry is declared, not empty


def test_cancelled_verification_marks_not_verified_never_agree(tmp_path: Path) -> None:
    label_path, image_path = _semantic_setup(tmp_path)
    result = verify(label_path, image_path, client=None)
    assert result.status == "ok"
    assert result.metrics["verified"] is False
    verdicts = result.metrics["verdicts"]
    assert verdicts and all(v["verdict"] == "not_verified" for v in verdicts)


def test_verdicts_are_reused_unless_reverify(tmp_path: Path) -> None:
    label_path, image_path = _semantic_setup(tmp_path)
    calls: list[tuple] = []
    client = _stub_client({}, calls)
    store = tmp_path / "verdicts.json"
    first = verify(
        label_path,
        image_path,
        client=client,
        model="gpt-4o",
        model_version="v1",
        store_path=store,
    )
    assert first.metrics["n_model_calls"] == 3
    second = verify(
        label_path,
        image_path,
        client=client,
        model="gpt-4o",
        model_version="v1",
        store_path=store,
    )
    assert second.metrics["n_model_calls"] == 0, "recorded verdicts are reused"
    assert all(v["reused"] for v in second.metrics["verdicts"])
    third = verify(
        label_path,
        image_path,
        client=client,
        model="gpt-4o",
        model_version="v1",
        store_path=store,
        reverify=True,
    )
    assert third.metrics["n_model_calls"] == 3, "reverify=True asks again"


def test_cross_model_aggregation_is_refused() -> None:
    verdicts = [
        {"model_id": "gpt-4o", "verdict": "agree"},
        {"model_id": "claude-sonnet-4", "verdict": "disagree"},
    ]
    with pytest.raises(ValueError, match="not comparable"):
        summarize_verdicts(verdicts)
    single = summarize_verdicts(verdicts[:1])
    assert single["model_id"] == "gpt-4o"
    assert single["counts"]["agree"] == 1


def test_no_code_path_writes_back_to_the_label_map() -> None:
    """Static: verify.py reads the label map but never saves/writes one."""
    import importlib

    verify_module = importlib.import_module("organelleverse.morphology.verify")
    tree = ast.parse(Path(verify_module.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"save", "write_text", "write_bytes"}
        ):
            rendered = ast.dump(node.func.value)
            assert "label" not in rendered, "a label map is being written"


def test_batch_resume_skips_completed_items(tmp_path: Path) -> None:
    label_path, image_path = _semantic_setup(tmp_path)
    state_dir = tmp_path / "batch"
    calls: list[tuple] = []
    client = _stub_client({}, calls)
    items = [(label_path, image_path), (label_path, image_path)]
    # Distinct items need distinct keys; duplicate the pair under a second name.
    label2 = tmp_path / "label2.png"
    label2.write_bytes(label_path.read_bytes())
    items = [(label_path, image_path), (label2, image_path)]

    first = verify_batch(
        items, state_dir=state_dir, client=client, model="gpt-4o", model_version="v1"
    )
    assert first == {"done": 2, "failed": 0, "skipped": 0}
    assert (state_dir / "progress_state.json").is_file()
    calls_after_first = len(calls)

    second = verify_batch(
        items, state_dir=state_dir, client=client, model="gpt-4o", model_version="v1"
    )
    assert second == {"done": 0, "failed": 0, "skipped": 2}
    assert len(calls) == calls_after_first, "resume must not re-verify completed items"
