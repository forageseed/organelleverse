"""Multimodal verification gate for segmentation objects (T4b).

A wrong mask looks exactly as confident as a right one. This module asks a
*declared* multimodal model — the desktop's configured LLM — a constrained
closed question per object crop ("this region is labelled class X; does the
image agree?") and records the verdict as provenance. The model only ever
judges: it never generates or modifies a mask, and no code path here writes
a verdict back into a label map.

Design rules (task brief 2026-08-17, T4b):

- **Per-object crops, zoomed** — never one whole-image question.
- **Closed questions only** — verdicts are exactly ``agree`` / ``disagree``
  / ``uncertain``; ``uncertain`` must stay a real option and routes to a
  human, because with a single verifier it carries the
  verifier-unreliability signal.
- **The verifier is pinned** — model id, model version, prompt hash, crop
  region, verdict, timestamp on every record. Verdicts from different
  models are not comparable: aggregation across models is refused
  (:func:`summarize_verdicts`).
- **Declared capability, never probed** — a model is treated as
  vision-capable only when the declared registry says so
  (:data:`MODEL_CAPABILITIES`); an unlisted or unknown model is unsupported,
  full stop. Nothing here sends a probe image to find out.
- **Disableable** — the segmentation pipeline never calls this gate unless
  asked, and a declined/cancelled verification marks objects
  ``not_verified``, never ``agree``.
- **Honest contract** — ``deterministic=False``, network side effect,
  ``idempotent=False``. Recorded verdicts are reused across reruns unless
  ``reverify=True``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ..core import ErrorDetail, Finding, OrganelleResult
from ._contract import (
    MORPHOLOGY_OPERATION_VERSION,
    MORPHOLOGY_SCOPE,
    make_provenance,
    parameters_hash,
    utc_now,
)

__all__ = [
    "DISAGREE_STRATEGIES",
    "MODEL_CAPABILITIES",
    "VERDICTS",
    "VerdictStore",
    "closed_question_prompt",
    "model_supports_vision",
    "summarize_verdicts",
    "verify",
    "verify_batch",
]

OPERATION_ID = "morphology.verify"

VERDICTS = ("agree", "disagree", "uncertain")
DISAGREE_STRATEGIES = ("fail", "flag_for_correction")

#: Declared model capabilities. A model is vision-capable only when listed
#: here with ``supports_vision=True``; anything unlisted is treated as
#: unsupported (declaration over inference). Entries are vendor-documented
#: multimodal model ids; extend deliberately, never by probing.
MODEL_CAPABILITIES: dict[str, dict[str, Any]] = {
    "gpt-4o": {"supports_vision": True, "vendor": "openai"},
    "gpt-4o-mini": {"supports_vision": True, "vendor": "openai"},
    "claude-sonnet-4": {"supports_vision": True, "vendor": "anthropic"},
    "claude-opus-4": {"supports_vision": True, "vendor": "anthropic"},
    "qwen2.5-vl-72b-instruct": {"supports_vision": True, "vendor": "alibaba"},
}

#: The one closed question every crop is asked. Pinned: its hash travels
#: with every verdict, so a prompt edit visibly invalidates comparability.
PROMPT_TEMPLATE = (
    "This electron-microscopy crop is labelled as organelle class "
    '"{class_name}". Does the image content agree with that label? '
    "Answer with exactly one of: agree / disagree / uncertain."
)

#: Verifier client protocol: given a model id, a rendered prompt, and a crop
#: image path, return the model's raw text response. Injected by the caller
#: (the desktop wires its configured LLM; tests stub it). This module itself
#: performs no network I/O.
VerifierClient = Callable[[str, str, Path], str]


def model_supports_vision(model: str) -> bool:
    """Declared-capability check only — never a probe."""
    entry = MODEL_CAPABILITIES.get(model)
    return bool(entry and entry.get("supports_vision"))


def closed_question_prompt(class_name: str) -> str:
    """Render the pinned closed question for one object class."""
    return PROMPT_TEMPLATE.format(class_name=class_name)


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _parse_verdict(raw: str) -> str:
    """Map a raw model response onto the closed verdict set.

    Anything outside the three tokens is not a verdict: it is recorded as
    ``uncertain`` (routing to a human) with the raw response preserved —
    never coerced to ``agree``.
    """
    normalized = raw.strip().lower()
    for verdict in VERDICTS:
        if normalized == verdict:
            return verdict
    return "uncertain"


@dataclass(frozen=True)
class _ObjectCrop:
    label: int
    class_name: str
    bbox: tuple[int, int, int, int]
    crop_path: Path


def _crop_objects(
    label_map_path: Path,
    image_path: Path,
    out_dir: Path,
    *,
    instance_map: bool,
    zoom_padding: int = 16,
) -> list[_ObjectCrop]:
    """Cut one zoomed crop per segmented object (never a whole-image ask).

    Semantic maps (OrgSegNet: pixel value = class index) are split into
    connected components per class and keep their class names; instance maps
    (micro-SAM: pixel value = instance id) are already one id per object and
    honestly report no class semantics.
    """
    import numpy as np
    from PIL import Image
    from skimage import measure as skmeasure

    from .measure import ORGSEG_CLASSES, _load_image

    labels = _load_image(label_map_path)
    if labels.ndim == 3:
        labels = labels[..., 0]
    image = _load_image(image_path)
    if image.ndim == 3:
        image = image.mean(axis=-1)
    lo, hi = float(image.min()), float(image.max())
    image = ((image - lo) / (hi - lo + 1e-9) * 255.0).astype(np.uint8)

    out_dir.mkdir(parents=True, exist_ok=True)
    crops: list[_ObjectCrop] = []
    height, width = labels.shape

    # (object_id, class_name, object mask) triples, in a deterministic order.
    objects: list[tuple[int, str, Any]] = []
    if instance_map:
        for object_id in sorted(int(v) for v in np.unique(labels) if v != 0):
            objects.append((object_id, f"object_{object_id}", labels == object_id))
    else:
        for class_idx in sorted(int(v) for v in np.unique(labels) if v != 0):
            class_name = (
                ORGSEG_CLASSES[class_idx]
                if class_idx < len(ORGSEG_CLASSES)
                else f"class_{class_idx}"
            )
            components = skmeasure.label(labels == class_idx)
            for component_id in sorted(int(v) for v in np.unique(components) if v != 0):
                object_id = class_idx * 1000 + component_id
                objects.append((object_id, class_name, components == component_id))

    for object_id, class_name, mask in objects:
        rows, cols = np.nonzero(mask)
        r0 = max(0, int(rows.min()) - zoom_padding)
        r1 = min(height, int(rows.max()) + 1 + zoom_padding)
        c0 = max(0, int(cols.min()) - zoom_padding)
        c1 = min(width, int(cols.max()) + 1 + zoom_padding)
        crop_path = out_dir / f"object-{object_id:04d}.png"
        Image.fromarray(image[r0:r1, c0:c1]).save(crop_path)
        crops.append(
            _ObjectCrop(
                label=object_id,
                class_name=class_name,
                bbox=(r0, c0, r1, c1),
                crop_path=crop_path,
            )
        )
    return crops


class VerdictStore:
    """File-backed verdict reuse: same (crop, model, prompt) → same verdict.

    Reruns never silently re-call the model; only ``reverify=True`` bypasses
    the store. The store lives inside the project directory, never /tmp.
    """

    def __init__(self, path: Path) -> None:
        self._path = path

    @staticmethod
    def _key(crop_path: Path, model: str, prompt: str) -> str:
        payload = crop_path.read_bytes() + model.encode("utf-8") + prompt.encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _load(self) -> dict[str, Any]:
        if not self._path.is_file():
            return {}
        return json.loads(self._path.read_text(encoding="utf-8"))

    def get(self, crop_path: Path, model: str, prompt: str) -> dict[str, Any] | None:
        return self._load().get(self._key(crop_path, model, prompt))

    def put(self, crop_path: Path, model: str, prompt: str, record: dict[str, Any]) -> None:
        records = self._load()
        records[self._key(crop_path, model, prompt)] = record
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(records, indent=1, sort_keys=True), encoding="utf-8")


def summarize_verdicts(verdicts: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate verdicts **per model** — cross-model aggregation is refused.

    Verdicts from different verifiers are not comparable (the user can switch
    models at any time), so a mixed-model input fails closed instead of
    producing a meaningless pooled statistic.
    """
    models = {str(v.get("model_id", "")) for v in verdicts}
    if len(models) > 1:
        raise ValueError(
            "verdicts from different models are not comparable; "
            f"group by model_id instead of aggregating (got {sorted(models)})"
        )
    counts = {verdict: 0 for verdict in VERDICTS}
    counts["not_verified"] = 0
    for v in verdicts:
        counts[str(v.get("verdict", "not_verified"))] = (
            counts.get(str(v.get("verdict", "not_verified")), 0) + 1
        )
    return {"model_id": next(iter(models), None), "n": len(verdicts), "counts": counts}


def verify(
    label_map: str | Path,
    image: str | Path,
    *,
    client: VerifierClient | None = None,
    model: str = "",
    model_version: str = "",
    on_disagree: Literal["fail", "flag_for_correction"] = "flag_for_correction",
    store_path: str | Path | None = None,
    reverify: bool = False,
    instance_map: bool = False,
) -> OrganelleResult:
    """Verify every segmented object with the configured multimodal model.

    Parameters
    ----------
    label_map
        Instance/class label map produced by ``segment`` (never modified).
    image
        The source electron micrograph the objects are cropped from.
    client
        The verifier callable (model, prompt, crop_path) -> raw response.
        ``None`` means verification is declined/unavailable: every object is
        marked ``not_verified`` — never ``agree`` — and the result stays a
        honest, complete record of *not* having verified.
    model
        The pinned verifier model id. Must be declared vision-capable in
        :data:`MODEL_CAPABILITIES`; anything else fails closed with the
        vision-capable choices listed (the desktop turns this into the
        model-picker dialog).
    model_version
        The verifier's exact version, recorded verbatim per verdict.
    on_disagree
        Declared strategy for ``disagree``/``uncertain`` verdicts:
        ``"fail"`` fails the whole result; ``"flag_for_correction"`` marks
        the objects for human correction (T4a). Never silently ignored.
    store_path
        Verdict reuse store location (inside the project directory).
    reverify
        Bypass the verdict store and ask again.
    instance_map
        Declare the label map's convention (instance ids from micro-SAM vs
        semantic class indices from OrgSegNet); never inferred.
    """
    started_at = utc_now()
    label_path = Path(label_map)
    image_path = Path(image)
    # Crops derive deterministically from the label map's location — a
    # caller-chosen output path is a frozen public-surface violation for a
    # new operation, so there is deliberately no output_dir parameter.
    out_dir = label_path.parent / f"{label_path.stem}_verify_crops"
    params_hash = parameters_hash(
        {
            "label_map": str(label_path),
            "image": str(image_path),
            "model": model,
            "model_version": model_version,
            "on_disagree": on_disagree,
            "reverify": reverify,
            "instance_map": instance_map,
        }
    )

    def _failed(code: str, message: str, details: dict[str, Any]) -> OrganelleResult:
        return OrganelleResult(
            operation_id=OPERATION_ID,
            operation_version=MORPHOLOGY_OPERATION_VERSION,
            scope=MORPHOLOGY_SCOPE,
            status="failed",
            summary_text=f"Verification refused: {message}",
            metrics={"verified": False},
            flags=("morph_verify_failed",),
            provenance=make_provenance(
                OPERATION_ID,
                params_hash=params_hash,
                started_at=started_at,
                finished_at=utc_now(),
            ),
            errors=(ErrorDetail(code=code, message=message, details=details, retryable=False),),
        )

    if on_disagree not in DISAGREE_STRATEGIES:
        return _failed(
            "morphology.verify_invalid_strategy",
            f"on_disagree must be one of {DISAGREE_STRATEGIES}",
            {"on_disagree": on_disagree},
        )
    if not label_path.is_file() or not image_path.is_file():
        missing = str(label_path if not label_path.is_file() else image_path)
        return _failed(
            "morphology.verify_input_missing",
            f"input does not exist: {missing}",
            {"missing": missing},
        )

    if client is None:
        # Declined or unconfigured verification: honest not_verified, never agree.
        crops = _crop_objects(label_path, image_path, out_dir, instance_map=instance_map)
        verdicts = [
            {
                "object_label": crop.label,
                "class_name": crop.class_name,
                "bbox": crop.bbox,
                "verdict": "not_verified",
                "model_id": model or None,
                "model_version": model_version or None,
                "prompt_hash": None,
                "verified_at": None,
                "routed_to_human": False,
                "blocks_downstream": False,
            }
            for crop in crops
        ]
        return OrganelleResult(
            operation_id=OPERATION_ID,
            operation_version=MORPHOLOGY_OPERATION_VERSION,
            scope=MORPHOLOGY_SCOPE,
            status="ok",
            summary_text=(
                f"Verification not performed ({len(crops)} objects marked not_verified)."
            ),
            metrics={"verified": False, "n_objects": len(crops), "verdicts": verdicts},
            flags=("morph_not_verified",),
            provenance=make_provenance(
                OPERATION_ID,
                params_hash=params_hash,
                started_at=started_at,
                finished_at=utc_now(),
            ),
        )

    if not model or not model_supports_vision(model):
        return _failed(
            "morphology.verifier_not_multimodal",
            f"model {model!r} is not declared vision-capable; choose one that is",
            {
                "model": model,
                "vision_capable_models": sorted(
                    name for name, meta in MODEL_CAPABILITIES.items() if meta["supports_vision"]
                ),
            },
        )

    crops = _crop_objects(label_path, image_path, out_dir, instance_map=instance_map)
    store = VerdictStore(Path(store_path)) if store_path else None
    verdicts: list[dict[str, Any]] = []
    calls = 0
    for crop in crops:
        prompt = closed_question_prompt(crop.class_name)
        cached = None if reverify else (store.get(crop.crop_path, model, prompt) if store else None)
        if cached is not None:
            record = dict(cached)
            record["reused"] = True
            verdicts.append(record)
            continue
        raw = client(model, prompt, crop.crop_path)
        calls += 1
        verdict = _parse_verdict(raw)
        record = {
            "object_label": crop.label,
            "class_name": crop.class_name,
            "bbox": crop.bbox,
            "verdict": verdict,
            "model_id": model,
            "model_version": model_version,
            "prompt_hash": _prompt_hash(prompt),
            "raw_response": raw,
            "reused": False,
            "verified_at": utc_now().isoformat(),
            # disagree/uncertain always route to a human and block downstream
            # measurement until adjudicated; agree/not_verified do not.
            "routed_to_human": verdict in {"disagree", "uncertain"},
            "blocks_downstream": verdict in {"disagree", "uncertain"},
        }
        if store:
            store.put(crop.crop_path, model, prompt, record)
        verdicts.append(record)

    contested = [v for v in verdicts if v["verdict"] in {"disagree", "uncertain"}]
    if contested and on_disagree == "fail":
        return OrganelleResult(
            operation_id=OPERATION_ID,
            operation_version=MORPHOLOGY_OPERATION_VERSION,
            scope=MORPHOLOGY_SCOPE,
            status="failed",
            summary_text=(
                f"Verifier contested {len(contested)} of {len(verdicts)} objects (strategy=fail)."
            ),
            metrics={"verified": True, "n_objects": len(verdicts), "verdicts": verdicts},
            flags=("morph_verify_contested",),
            provenance=make_provenance(
                OPERATION_ID,
                params_hash=params_hash,
                software_versions={"verifier_model": model, "verifier_version": model_version},
                started_at=started_at,
                finished_at=utc_now(),
            ),
            errors=(
                ErrorDetail(
                    code="morphology.verify_contested",
                    message=f"{len(contested)} objects require human adjudication",
                    details={
                        "contested_labels": [v["object_label"] for v in contested],
                        "strategy": on_disagree,
                    },
                    retryable=False,
                ),
            ),
        )

    return OrganelleResult(
        operation_id=OPERATION_ID,
        operation_version=MORPHOLOGY_OPERATION_VERSION,
        scope=MORPHOLOGY_SCOPE,
        status="ok",
        summary_text=(
            f"Verified {len(verdicts)} objects with {model}: "
            f"{sum(1 for v in verdicts if v['verdict'] == 'agree')} agree, "
            f"{len(contested)} flagged for human correction."
        ),
        metrics={
            "verified": True,
            "n_objects": len(verdicts),
            "n_model_calls": calls,
            "verdicts": verdicts,
            "flagged_for_correction": [v["object_label"] for v in contested],
        },
        findings=(
            Finding(
                code="morphology.verified",
                metric="contested",
                value=len(contested),
                unit="objects",
            ),
        ),
        flags=("morph_verified",) if not contested else ("morph_verify_contested",),
        provenance=make_provenance(
            OPERATION_ID,
            params_hash=params_hash,
            software_versions={"verifier_model": model, "verifier_version": model_version},
            started_at=started_at,
            finished_at=utc_now(),
        ),
    )


def verify_batch(
    items: list[tuple[str | Path, str | Path]],
    *,
    state_dir: str | Path,
    client: VerifierClient | None,
    model: str,
    model_version: str = "",
    on_disagree: Literal["fail", "flag_for_correction"] = "flag_for_correction",
    reverify: bool = False,
    instance_map: bool = False,
) -> dict[str, Any]:
    """Batch-verify (label_map, image) pairs with resume support.

    Batch discipline (brief, stage-2 rules): a ``progress_state.json`` state
    file inside ``state_dir`` tracks per-item outcomes; reruns skip items
    already ``done``. The final summary reports done/failed/skipped counts.
    """
    state_path = Path(state_dir) / "progress_state.json"
    state: dict[str, Any] = {"done": {}, "failed": {}}
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))

    summary = {"done": 0, "failed": 0, "skipped": 0}
    for label_map, image in items:
        key = f"{label_map}|{image}"
        if key in state["done"]:
            summary["skipped"] += 1
            continue
        result = verify(
            label_map,
            image,
            client=client,
            model=model,
            model_version=model_version,
            on_disagree=on_disagree,
            store_path=Path(state_dir) / "verdicts.json",
            reverify=reverify,
            instance_map=instance_map,
        )
        if (
            result.status == "failed"
            and result.errors
            and result.errors[0].code not in {"morphology.verify_contested"}
        ):
            state["failed"][key] = result.errors[0].message
            summary["failed"] += 1
        else:
            state["done"][key] = {
                "status": result.status,
                "n_objects": result.metrics.get("n_objects"),
            }
            summary["done"] += 1
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state, indent=1, sort_keys=True), encoding="utf-8")
    return summary
