"""TMbed topology execution and strict import of its directed 3-line output."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from ...core.artifacts import ArtifactRef
from ...core.errors import OrganelleDependencyError, OrganelleInputError, OrganelleParameterError
from ...core.external import run_external
from ...core.result import OrganelleResult


def _invalid(message):
    return OrganelleInputError(code="cms.invalid_topology", message=message)


def load_tmbed_predictions(path: Path, proteins: dict[str, str]) -> dict:
    """Require TMbed --out-format 0 IDs, complete sequences and residue labels.

    H/h and B/b are directed transmembrane segments; S is a signal peptide.
    Non-membrane positions use '.', never a fabricated topology assignment.
    All provided query proteins must be represented exactly once.
    """
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise _invalid(str(error)) from error
    if len(lines) % 3:
        raise _invalid("Expected TMbed directed 3-line output (--out-format 0)")
    predictions = {}
    for index in range(0, len(lines), 3):
        header, sequence, labels = lines[index : index + 3]
        if not header.startswith(">") or not header[1:].split():
            raise _invalid("Invalid TMbed prediction header")
        name = header[1:].split()[0]
        if name in predictions or name not in proteins:
            raise _invalid(f"Unexpected or duplicate topology protein ID: {name}")
        if (
            sequence != proteins[name]
            or len(labels) != len(sequence)
            or set(labels) - set("HhBbS.")
        ):
            raise _invalid(f"{name}: topology sequence/labels do not match the protein")
        segments = []
        start = 0
        while start < len(labels):
            label = labels[start]
            end = start + 1
            while end < len(labels) and labels[end] == label:
                end += 1
            if label != ".":
                segments.append(
                    {
                        "start": start + 1,
                        "end": end,
                        "type": {
                            "H": "alpha_helix",
                            "h": "alpha_helix",
                            "B": "beta_strand",
                            "b": "beta_strand",
                            "S": "signal_peptide",
                        }[label],
                        "orientation": "inside_to_outside"
                        if label in "HB"
                        else "outside_to_inside"
                        if label in "hb"
                        else None,
                    }
                )
            start = end
        predictions[name] = {
            "predictor": "TMbed",
            "format": "directed_3line_0",
            "labels": labels,
            "segments": segments,
            "transmembrane_segment_count": sum(s["type"] != "signal_peptide" for s in segments),
            "signal_peptide_predicted": any(s["type"] == "signal_peptide" for s in segments),
            "coordinate_system": "protein 1-based inclusive",
            "interpretation": "Model prediction; neither experimental membrane topology nor CMS causality",
        }
    if set(predictions) != set(proteins):
        raise _invalid("TMbed predictions must cover every query protein")
    return predictions


def predict_cms_topology(
    protein_fasta: Path,
    *,
    model_dir: Path,
    tmbed_path: str = "tmbed",
    threads: int = 4,
    timeout: int = 1800,
) -> OrganelleResult:
    """Run the mature TMbed model locally on CPU, without device fallback.

    Supply the upstream ProtT5 encoder model directory explicitly. Model/tool
    absence or partial output is an error. Protein input may contain the standard
    20 residues or X; do not silently transform stops or ambiguous residues.
    Validated predictions are returned as an artifact in managed run storage.
    """
    from .evidence import _proteins

    for name, value in (("threads", threads), ("timeout", timeout)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise OrganelleParameterError(
                code="cms.invalid_topology_parameter", message=f"{name} must be a positive integer"
            )
    proteins = _proteins(protein_fasta)
    if any(set(seq) - set("ACDEFGHIKLMNPQRSTVWYX") for seq in proteins.values()):
        raise _invalid("TMbed input must contain standard amino acids or X, without internal stops")
    executable = shutil.which(tmbed_path)
    if executable is None:
        raise OrganelleDependencyError(
            code="cms.tmbed_missing", message=f"TMbed not found: {tmbed_path}"
        )
    model_dir = Path(model_dir)
    if not model_dir.is_dir() or not (model_dir / "config.json").is_file():
        raise OrganelleDependencyError(
            code="cms.tmbed_model_missing",
            message="Expected the upstream ProtT5 encoder directory with config.json",
        )
    from ...runtime import managed_run_path

    output = managed_run_path("phenotype.predict_cms_topology", uuid4().hex)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "tmbed.pred"
    # TMbed's published CLI supplies this exact directed output format. Its
    # CPU/device fallback is explicitly disabled; no custom model is substituted.
    env = os.environ.copy()
    # Upstream ignores --model-dir when HF_HOME is set. The explicit local model
    # is required here; inference must not fetch a different remote snapshot.
    env.pop("HF_HOME", None)
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    with TemporaryDirectory(prefix=".cms-tmbed-", dir=output) as scratch:
        from ..._bio import write_fasta

        query = write_fasta(Path(scratch) / "proteins.fasta", list(proteins.items()))
        predicted = Path(scratch) / "tmbed.pred"
        argv = [
            executable,
            "predict",
            "-f",
            str(query),
            "-p",
            str(predicted),
            "--out-format",
            "0",
            "--no-use-gpu",
            "--no-cpu-fallback",
            "--threads",
            str(threads),
            "--model-dir",
            str(model_dir),
        ]
        run_external(argv, timeout=timeout, env=env, code="cms.tmbed_failed", tool="TMbed")
        predictions = load_tmbed_predictions(predicted, proteins)
        predicted.replace(destination)
    return OrganelleResult(
        operation_id="phenotype.predict_cms_topology",
        scope="mitochondrion",
        status="ok",
        summary_text=f"TMbed topology predictions for {len(predictions)} proteins.",
        metrics={
            "prediction_count": len(predictions),
            "predictions": predictions,
            "predictor": "TMbed",
            "model_directory": str(model_dir),
            "argv": argv,
            "device": "cpu",
        },
        flags=("topology_is_model_prediction", "cms_causality_not_established"),
        artifacts=(
            ArtifactRef.from_path(
                destination, kind="topology", format="tmbed_3line", media_type="text/plain"
            ),
        ),
    )
