"""Subcellular-localization prediction for plant proteins.

Wraps the standard plant subcellular-localization predictors and provides a
dependency-free built-in heuristic core so the module is useful even when no
external tool is installed.

Backends
--------
- ``heuristic`` (default): pure-Python scorers in :mod:`.localize_core`
  (signal-peptide PWM from von Heijne 1986, transit-peptide composition
  heuristics, NLS/PTS motifs, GES-scale transmembrane prediction). Offline.
- ``deeploc``: DeepLoc 2.0 multi-label deep-learning predictor via its
  ``deeploc2`` CLI (``epcm18/deeploc2-fyp`` fork). Requires ESM1b/ProtT5
  weights -- see :func:`.install.check_backend`.
- ``targetp``: TargetP 2.0 N-terminal sorting-signal predictor (``targetp
  -org pl`` for plants).
- ``auto``: use ``deeploc`` if installed, else ``targetp``, else ``heuristic``.

Input is a protein FASTA (``str | Path``). Output is an
:class:`~organelleverse.core.OrganelleResult` whose ``key_findings`` list one
entry per protein (top compartment + probability) and whose
``observed_metrics['per_protein_predictions']`` carries the full per-protein
prediction dicts.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any
from uuid import uuid4

from .._bio import read_fasta
from ..core.frozen import FrozenMap
from ..core.provenance import ResultProvenance
from ..core.result import Finding, OrganelleResult, ResultStatus
from ..runtime import managed_run_path
from .localize_core import COMPARTMENTS, LocalizationScores, predict_heuristic

__all__ = ["predict", "localize", "COMPARTMENTS"]

_OPERATION_ID = "localization.predict"
_OPERATION_VERSION = "1.0"


def _package_version() -> str:
    try:
        return version("organelleverse")
    except PackageNotFoundError:
        return "0.0.1"


def _parameters_hash(parameters: dict[str, Any]) -> str:
    payload = json.dumps(
        parameters,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> tuple[str, ...]:
    """Content identity of the input, so a Result is not tied to a file path."""
    if not path.is_file():
        return ()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return (digest.hexdigest(),)


def _provenance(
    parameters: dict[str, Any],
    source: Path,
    *,
    requested_backend: str,
    actual_backend: str,
    attempted_backends: tuple[str, ...],
    argv: tuple[str, ...] = (),
) -> ResultProvenance:
    return ResultProvenance(
        operation_id=_OPERATION_ID,
        operation_version=_OPERATION_VERSION,
        package_version=_package_version(),
        git_commit=os.environ.get("ORG_VERSE_GIT_COMMIT", ""),
        input_artifact_hashes=_file_sha256(source),
        parameters_hash=_parameters_hash(parameters),
        requested_backend=requested_backend,
        actual_backend=actual_backend,
        attempted_backends=attempted_backends,
        argv=argv,
    )


def _localization_findings(summaries: list[dict[str, Any]]) -> tuple[Finding, ...]:
    """One Finding per protein: the top predicted compartment and its probability."""
    findings: list[Finding] = []
    for summary in summaries:
        probabilities = summary.get("probabilities") or {}
        probability = probabilities.get(summary["predicted"], 0.0)
        findings.append(
            Finding(
                code="localization.top_compartment",
                metric=str(summary["id"]),
                value=str(summary["predicted"]),
                confidence=min(1.0, max(0.0, float(probability))),
            )
        )
    return tuple(findings)


# DeepLoc 2.1 compartment / signal columns -> OrganelleVerse canonical names.
# DeepLoc 2.1 writes a 17-column CSV whose header is fixed:
#   Protein_ID, Localizations, Signals, Membrane types,
#   Cytoplasm, Nucleus, Extracellular, Cell membrane, Mitochondrion, Plastid,
#   Endoplasmic reticulum, Lysosome/Vacuole, Golgi apparatus, Peroxisome,
#   Peripheral, Transmembrane, Lipid anchor, Soluble
_DEEPLOC_MAP: dict[str, str] = {
    "Nucleus": "Nucleus",
    "Cytoplasm": "Cytoplasm",
    "Extracellular": "Secretory",
    "Cell membrane": "Membrane",
    "Mitochondrion": "Mitochondrion",
    "Plastid": "Plastid",
    "Endoplasmic reticulum": "ER",
    "Golgi apparatus": "Golgi",
    "Peroxisome": "Peroxisome",
    "Lysosome/Vacuole": "Vacuole",
    # DeepLoc 2.1 adds membrane-association descriptors; map "Membrane" loosely.
    "Peripheral": "Membrane",
    "Transmembrane": "Membrane",
    "Lipid anchor": "Membrane",
    "Soluble": "Cytoplasm",
}
# DeepLoc 2.1 probability columns (in output order), excluding the first four
# text columns (Protein_ID, Localizations, Signals, Membrane types).
_DEEPLOC_PROB_COLS = (
    "Cytoplasm",
    "Nucleus",
    "Extracellular",
    "Cell membrane",
    "Mitochondrion",
    "Plastid",
    "Endoplasmic reticulum",
    "Lysosome/Vacuole",
    "Golgi apparatus",
    "Peroxisome",
    "Peripheral",
    "Transmembrane",
    "Lipid anchor",
    "Soluble",
)
# DeepLoc "Localizations" cell is a multi-label string like
# "Mitochondrion|Nucleus"; it can also be a single label.
_DEEPLOC_SEP = re.compile(r"[|;,]")


def _heuristic_predictions(
    records: list[tuple[str, str]],
) -> tuple[list[dict[str, Any]], list[LocalizationScores]]:
    """Run the heuristic core over FASTA records; return (summary_dicts, scores)."""
    summaries: list[dict[str, Any]] = []
    scores: list[LocalizationScores] = []
    for seqid, seq in records:
        sc = predict_heuristic(seq, sequence_id=seqid)
        scores.append(sc)
        top = sc.predicted[0] if sc.predicted else "Cytoplasm"
        summaries.append(
            {
                "id": seqid,
                "length": sc.length,
                "predicted": top,
                "predicted_all": sc.predicted,
                "probabilities": {k: round(v, 3) for k, v in sc.probabilities.items()},
                "has_sp": bool(sc.signal_peptide.get("has_sp")),
                "tm_count": int(sc.transmembrane.get("tm_count", 0)),
                "has_nls": bool(sc.nls.get("has_nls")),
                "has_pts": bool(sc.pts.get("has_pts")),
            }
        )
    return summaries, scores


def _parse_deeploc_output(path: Path) -> dict[str, dict[str, Any]]:
    """Parse a DeepLoc 2.1 results CSV into {id: {compartment: prob, ...}}.

    DeepLoc 2.1 (``deeploc2`` CLI, conda package ``deeploc2-2.1.0``) writes a
    comma-separated ``results_<timestamp>.csv`` with 17 columns: 4 text columns
    (Protein_ID, Localizations, Signals, Membrane types) followed by 13
    probability columns. ``Localizations`` is a multi-label cell joined by
    ``|`` (e.g. "Mitochondrion|Nucleus").
    """
    out: dict[str, dict[str, Any]] = {}
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            return out
        header = [h.strip() for h in header]
        if not header or header[0].lower() not in ("protein_id", "id"):
            # Not a DeepLoc 2.1 results header; bail out.
            return out
        # Map header -> column index for the probability columns.
        prob_idx: list[tuple[int, str]] = []
        for i, col in enumerate(header):
            if col in _DEEPLOC_PROB_COLS:
                prob_idx.append((i, col))
        loc_col = header.index("Localizations") if "Localizations" in header else 1
        sig_col = header.index("Signals") if "Signals" in header else 2
        mem_col = header.index("Membrane types") if "Membrane types" in header else 3
        for row in reader:
            if not row or not row[0].strip():
                continue
            seqid = row[0].strip().split()[0]
            # Per-compartment probabilities (mapped to OrganelleVerse names,
            # taking the max when two DeepLoc columns map to the same name).
            mapped: dict[str, float] = {}
            raw: dict[str, float] = {}
            for i, col in prob_idx:
                if i >= len(row):
                    continue
                try:
                    p = float(row[i])
                except ValueError:
                    continue
                raw[col] = p
                name = _DEEPLOC_MAP.get(col, col)
                mapped[name] = max(mapped.get(name, 0.0), p)
            # Predicted = the DeepLoc-reported "Localizations" cell (authoritative
            # multi-label), mapped to our names.
            loc_cell = row[loc_col].strip() if loc_col < len(row) else ""
            labels = [s for s in _DEEPLOC_SEP.split(loc_cell) if s]
            predicted = (
                _DEEPLOC_MAP.get(labels[0], labels[0])
                if labels
                else (max(mapped, key=mapped.get) if mapped else "Cytoplasm")
            )
            out[seqid] = {
                "predicted": predicted,
                "predicted_all": [_DEEPLOC_MAP.get(l, l) for l in labels],
                "probabilities": {k: round(v, 3) for k, v in mapped.items()},
                "localizations": loc_cell,
                "signals": row[sig_col].strip() if sig_col < len(row) else "",
                "membrane_type": row[mem_col].strip() if mem_col < len(row) else "",
            }
    return out


def _parse_targetp_output(path: Path) -> dict[str, dict[str, Any]]:
    """Parse a TargetP 2.0 summary file into {id: {compartment: prob, ...}}.

    TargetP 2.0 (``targetp -org pl``) predicts plant N-terminal sorting signals:
    chloroplast transit peptide (cTP/Plastid), mitochondrial targeting peptide
    (mTP/Mitochondrion), signal peptide (SP/Secretory), and OTHER. The summary
    file is tab-delimited with a header; the plant columns are ``CSci CScp CSm
    CSmtp CSSp CSother`` (the leading ``CS`` are per-class probabilities).
    """
    # TargetP 2.0 column labels (plant mode) -> OrganelleVerse compartment.
    col_map = {
        # Probability columns (TargetP 2.0 naming).
        "CSci": "Secretory",
        "CScp": "Plastid",
        "CSm": "Mitochondrion",
        "CSmtp": "Mitochondrion",
        "CSSp": "Secretory",
        "CSother": "Cytoplasm",
        # Legacy TargetP 1.1 names.
        "SP": "Secretory",
        "mTP": "Mitochondrion",
        "cTP": "Plastid",
        "Chl": "Plastid",
        "Other": "Cytoplasm",
        "CSec": "Secretory",
        "CmTP": "Mitochondrion",
        "CcTP": "Plastid",
        "COther": "Cytoplasm",
    }
    out: dict[str, dict[str, Any]] = {}
    with open(path) as fh:
        header: list[str] = []
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if not header:
                # First non-comment, non-empty line is the header.
                header = parts
                continue
            if len(parts) < len(header):
                continue
            seqid = parts[0]
            row = dict(zip(header[1:], parts[1:]))
            probs: dict[str, float] = {}
            for col, val in row.items():
                mapped = col_map.get(col)
                if mapped is None:
                    continue
                try:
                    p = float(val)
                except ValueError:
                    continue
                probs[mapped] = max(probs.get(mapped, 0.0), p)
            top = max(probs, key=probs.get) if probs else "Cytoplasm"
            out[seqid] = {
                "predicted": top if probs.get(top, 0.0) >= 0.5 else "Cytoplasm",
                "probabilities": {k: round(v, 3) for k, v in probs.items()},
            }
    return out


def _run_external(
    argv: list[str],
    executor: Callable[[list[str]], Any] | None,
) -> tuple[bool, str]:
    """Run an external CLI via the executor hook. Returns (success, stderr/err)."""
    if executor is None:
        return False, "no executor"
    try:
        executor(list(argv))
        return True, ""
    except Exception as exc:  # noqa: BLE001 -- executor is user-supplied
        return False, str(exc)


def predict(
    fasta: str | Path,
    *,
    backend: str = "auto",
    executor: Callable[[list[str]], Any] | None = None,
    deeploc_model: str = "Fast",
    deeploc_device: str = "cpu",
    organism: str = "plant",
) -> OrganelleResult:
    """Predict subcellular localization for proteins in a FASTA file.

    Parameters
    ----------
    fasta
        Protein FASTA path.
    backend
        ``"auto"`` (default), ``"heuristic"``, ``"deeploc"``, or ``"targetp"``.
        ``auto`` picks the first installed external tool, falling back to the
        built-in heuristic core so the call always returns predictions.
    executor
        Optional ``Callable[[list[str]], Any]`` that runs an external CLI argv
        (e.g. a subprocess wrapper). When omitted, external backends only
        return a plan.
    deeploc_model
        DeepLoc 2.1 model variant: ``"Fast"`` (ESM1b) or ``"Accurate"`` (ProtT5).
    deeploc_device
        DeepLoc 2.1 compute device: ``"cpu"``, ``"cuda"``, or ``"mps"``.
    organism
        ``"plant"`` or ``"non-plant"`` for TargetP organism mode (DeepLoc 2.1
        does not take an organism flag -- it is a single eukaryotic model).

    Returns
    -------
    OrganelleResult
        ``suite="localization"``, ``op="localize"``. ``key_findings`` list one
        entry per protein (id + top compartment + probability).
    """
    from .install import check_backend, install_hint

    fasta_path = Path(fasta)
    records = read_fasta(fasta_path)
    n_prot = len(records)

    backend_lc = backend.lower()
    chosen = backend_lc
    warning = ""
    parameters: dict[str, Any] = {
        "backend": backend_lc,
        "deeploc_model": deeploc_model,
        "deeploc_device": deeploc_device,
        "organism": organism,
    }
    # ``auto`` is a selection policy, not a backend identity: only an explicit
    # backend request is recorded as ``requested_backend``.
    requested_backend = "" if backend_lc == "auto" else backend_lc

    # ── Auto-select an installed external backend ────────────────────
    if backend_lc == "auto":
        for cand in ("deeploc", "targetp"):
            if check_backend(cand, scan_envs=True).get("installed"):
                chosen = cand
                break
        else:
            chosen = "heuristic"

    # ── Heuristic backend (offline, always available) ────────────────
    if chosen == "heuristic":
        summaries, _scores = _heuristic_predictions(records)
        return OrganelleResult(
            operation_id=_OPERATION_ID,
            operation_version=_OPERATION_VERSION,
            scope="none",
            status="ok",
            summary_text=(
                f"Localization predicted for {n_prot} protein(s) via the "
                "built-in heuristic core (von Heijne 1986 SP PWM + transit-"
                "peptide composition + NLS/PTS motifs + GES TM scale). "
                "Heuristic -- not DeepLoc/TargetP."
            ),
            metrics=FrozenMap.from_json(
                {
                    "n_proteins": n_prot,
                    "backend": "heuristic",
                    "organism": organism,
                    "per_protein_predictions": summaries,
                }
            ),
            findings=_localization_findings(summaries),
            flags=("localize_heuristic",),
            artifacts=(),
            provenance=_provenance(
                parameters,
                fasta_path,
                requested_backend=requested_backend,
                actual_backend="heuristic",
                attempted_backends=("heuristic",),
                argv=("heuristic_core",),
            ),
        )

    # ── External backends (deeploc / targetp) ────────────────────────
    loc = check_backend(chosen, scan_envs=True)
    backend_path = loc.get("path") if loc.get("installed") else None
    bin_name = backend_path or chosen
    outdir = managed_run_path(_OPERATION_ID, uuid4().hex)
    if backend_path and executor:
        outdir.mkdir(parents=True, exist_ok=False)
    if not backend_path:
        warning = "\n  ⚠ " + install_hint(chosen)

    if chosen == "deeploc":
        # DeepLoc 2.1 CLI: deeploc2 -f <fa> -o <outdir> -m Fast|Accurate
        #                        -d cpu|cuda|mps  (no organism flag)
        out_file = outdir  # -o is an output *directory*
        argv = [
            bin_name,
            "-f",
            str(fasta_path),
            "-o",
            str(outdir),
            "-m",
            deeploc_model,
            "-d",
            deeploc_device,
        ]
        ran = False
        ext_preds: dict[str, dict[str, Any]] = {}
        if backend_path and executor:
            ok, err = _run_external(argv, executor)
            ran = ok
            if ok:
                # DeepLoc writes results_<timestamp>.csv in the output dir.
                csvs = sorted(outdir.glob("results_*.csv"))
                if csvs:
                    ext_preds = _parse_deeploc_output(csvs[-1])
                else:
                    warning += "\n  deeploc produced no results_*.csv"
            else:
                warning += f"\n  deeploc run failed: {err}"
    else:  # targetp
        org_flag = "pl" if organism == "plant" else "no"
        out_file = outdir / (fasta_path.stem + "_targetp_summary.txt")
        argv = [bin_name, "-org", org_flag, str(fasta_path)]
        ran = False
        ext_preds = {}
        if backend_path and executor:
            ok, err = _run_external(argv, executor)
            ran = ok
            if ok and out_file.exists():
                ext_preds = _parse_targetp_output(out_file)
            elif not ok:
                warning += f"\n  targetp run failed: {err}"

    if ran and ext_preds:
        # External tool produced predictions.
        summaries = []
        for seqid, seq in records:
            pred = ext_preds.get(seqid, {"predicted": "unknown", "probabilities": {}})
            summaries.append(
                {
                    "id": seqid,
                    "length": len(seq),
                    "predicted": pred["predicted"],
                    "predicted_all": pred.get("predicted_all", [pred["predicted"]]),
                    "probabilities": pred.get("probabilities", {}),
                    "localizations": pred.get("localizations", ""),
                    "signals": pred.get("signals", ""),
                    "membrane_type": pred.get("membrane_type", ""),
                }
            )
        flags = ("localize_ran", f"backend_{chosen}")
        summary = f"Localization predicted for {n_prot} protein(s) via {chosen} ({ran})."
    else:
        # External tool not available or produced no parseable output:
        # fall back to the heuristic core so callers always get predictions,
        # but flag that the requested backend did not run.
        summaries, _ = _heuristic_predictions(records)
        flags = ("localize_planned", f"backend_requested_{chosen}", "fallback_heuristic")
        summary = (
            f"{chosen} not available/ran -> heuristic fallback for "
            f"{n_prot} protein(s); argv planned." + warning
        )

    actual_backend = chosen if ran else "heuristic"
    attempted_backends = (chosen,) if ran else (chosen, "heuristic")
    # A requested backend that did not run and silently degraded to the
    # heuristic core is a fallback: the canonical contract requires that to be
    # reported as ``warning`` rather than ``ok``.
    status: ResultStatus = "ok" if ran else "warning"
    return OrganelleResult(
        operation_id=_OPERATION_ID,
        operation_version=_OPERATION_VERSION,
        scope="none",
        status=status,
        summary_text=summary,
        metrics=FrozenMap.from_json(
            {
                "n_proteins": n_prot,
                "backend": actual_backend,
                "backend_requested": chosen,
                "backend_found": backend_path is not None,
                "ran_external": ran,
                "organism": organism,
                "per_protein_predictions": summaries,
            }
        ),
        findings=_localization_findings(summaries),
        flags=flags,
        artifacts=(),
        provenance=_provenance(
            parameters,
            fasta_path,
            requested_backend=requested_backend,
            actual_backend=actual_backend,
            attempted_backends=attempted_backends,
            argv=tuple(argv) if "argv" in locals() else (),
        ),
    )


# ``localize`` is the public entry point name used elsewhere in OrganelleVerse
# (each suite exposes a verb named after the operation); ``predict`` is the
# domain-standard alias.
localize = predict
