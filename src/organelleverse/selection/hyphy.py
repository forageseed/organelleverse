"""HyPhy selection analyses: BUSTED, aBSREL, RELAX, MEME and FEL.

HyPhy (Kosakovsky Pond et al. 2005/2020) is the reference implementation of
these random-effects codon models; there is no maintained Python
re-implementation, and re-writing its optimiser, rate-class mixtures and
branch-site likelihood machinery is far beyond "a small algorithm", so this
module drives the real ``hyphy`` CLI (>= 2.5) through
:func:`organelleverse.core.external.run_external` and parses its JSON.

Inputs reuse the selection suite's preparation steps: an in-frame codon
alignment (``selection.pal2nal`` output as FASTA, or the PAML sequential file
the codeml models consume) plus a Newick tree. Foreground branches follow the
codeml models' convention: ``foreground_labels`` names tips to mark (the
codeml models write ``name #1``; HyPhy receives ``name{Foreground}``), and a
tree that already carries codeml marks (``#1`` on a branch, ``$1`` on a
clade) or HyPhy ``{Foreground}`` tags is converted as-is. ``mark_clade=True``
additionally marks every branch of the smallest clade spanning the labels
(stem included), which is what a lineage-level RELAX test usually wants.

Each analysis returns an :class:`~organelleverse.core.result.OrganelleResult`
whose ``metrics`` hold the parsed key results and whose ``artifacts`` keep
HyPhy's raw JSON (plus the exact alignment/tree/log it ran on).

======== =================================================================
BUSTED   gene-wide episodic diversifying selection (on foreground or all)
aBSREL   branches with episodic diversifying selection (Holm-corrected p)
RELAX    selection intensity k on test vs reference branches (k>1 intensified)
MEME     sites under episodic diversifying selection
FEL      sites under pervasive diversifying / purifying selection
======== =================================================================
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..core.errors import OrganelleDependencyError, OrganelleInputError
from ..core.external import run_external
from ..core.result import OrganelleResult
from . import _contract
from ._tree import (
    _Node,
    _label_tips,
    _parse_newick,
    _propagate_clade_marks,
    _smallest_clade,
)

__all__ = [
    "HYPHY_METHODS",
    "check_hyphy",
    "hyphy_absrel",
    "hyphy_busted",
    "hyphy_fel",
    "hyphy_install_hint",
    "hyphy_meme",
    "hyphy_relax",
    "label_tree_for_hyphy",
    "parse_hyphy_json",
]

#: CLI analysis name -> batch file shipped in ``TemplateBatchFiles/SelectionAnalyses``.
HYPHY_METHODS: dict[str, str] = {
    "busted": "BUSTED.bf",
    "absrel": "aBSREL.bf",
    "relax": "RELAX.bf",
    "meme": "MEME.bf",
    "fel": "FEL.bf",
}

_MIN_VERSION = (2, 5, 0)
_ENV_VAR = "ORGANELLEVERSE_HYPHY"
_FOREGROUND = "Foreground"
_STOP_CODONS = frozenset({"TAA", "TAG", "TGA"})
_VALID_NT = re.compile(r"^[ACGTNRYKMSWBDHV?\-]*$")


# =========================================================================
# Locating HyPhy
# =========================================================================


def hyphy_install_hint() -> str:
    """How to make HyPhy available to this package."""
    return (
        "HyPhy (>= 2.5) not found. Install options:\n"
        "  conda:  conda install -c conda-forge -c bioconda hyphy\n"
        "  mamba:  micromamba create -n hyphy -c conda-forge -c bioconda hyphy\n"
        "  source: https://github.com/veg/hyphy\n"
        f"Then put `hyphy` on PATH, set {_ENV_VAR}=/path/to/hyphy, or pass hyphy_path=."
    )


def _candidate_binaries(hyphy_path: str | Path | None) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    if hyphy_path:
        candidates.append((str(hyphy_path), "argument"))
        return candidates
    env_value = os.environ.get(_ENV_VAR)
    if env_value:
        candidates.append((env_value, f"env:{_ENV_VAR}"))
        return candidates
    on_path = shutil.which("hyphy")
    if on_path:
        candidates.append((on_path, "PATH"))
    try:
        from ..assembly.install import _list_conda_envs

        for env_bin in _list_conda_envs():
            candidate = env_bin / "hyphy"
            if candidate.is_file():
                candidates.append((str(candidate), f"conda:{env_bin.parent.name}"))
    except Exception:  # pragma: no cover - env scanning is best effort
        pass
    return candidates


def _batch_file_root(binary: Path) -> Path | None:
    """``<prefix>/share/hyphy/TemplateBatchFiles/SelectionAnalyses`` for a conda-style install."""
    resolved = binary.resolve()
    for prefix in (resolved.parent.parent, binary.parent.parent):
        root = prefix / "share" / "hyphy" / "TemplateBatchFiles" / "SelectionAnalyses"
        if root.is_dir():
            return root
    return None


def _parse_version(text: str) -> tuple[int, int, int] | None:
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", text)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def check_hyphy(hyphy_path: str | Path | None = None) -> dict[str, Any]:
    """Locate HyPhy, report its version and which selection batch files exist.

    Search order: ``hyphy_path`` argument, ``$ORGANELLEVERSE_HYPHY``, ``PATH``,
    then every conda/micromamba environment the package can enumerate.
    """
    for raw, source in _candidate_binaries(hyphy_path):
        binary = Path(raw)
        if not binary.is_file():
            continue
        try:
            completed = run_external(
                [str(binary), "--version"], timeout=60, input_data="", tool="hyphy"
            )
            version_text = (completed.stdout or completed.stderr).strip().splitlines()[0]
        except Exception as error:  # a broken binary is reported, not hidden
            return {
                "installed": False,
                "path": str(binary),
                "source": source,
                "error": str(error),
                "hint": hyphy_install_hint(),
            }
        version = _parse_version(version_text)
        root = _batch_file_root(binary)
        batch_files = {
            method: bool(root is not None and (root / name).is_file())
            for method, name in HYPHY_METHODS.items()
        }
        return {
            "installed": True,
            "path": str(binary),
            "source": source,
            "version": ".".join(map(str, version)) if version else version_text,
            "version_text": version_text,
            "version_ok": bool(version and version >= _MIN_VERSION),
            "batch_file_root": str(root) if root else None,
            "batch_files": batch_files,
        }
    return {"installed": False, "path": None, "source": None, "hint": hyphy_install_hint()}


def _require_hyphy(method: str, hyphy_path: str | Path | None) -> dict[str, Any]:
    info = check_hyphy(hyphy_path)
    if not info.get("installed"):
        raise OrganelleDependencyError(
            code="selection.hyphy_not_found",
            message=hyphy_install_hint(),
            details={k: v for k, v in info.items() if k != "hint"},
            suggested_action={"install": "conda install -c conda-forge -c bioconda hyphy"},
        )
    if not info.get("version_ok"):
        raise OrganelleDependencyError(
            code="selection.hyphy_version_unsupported",
            message=(
                f"HyPhy {info.get('version')} at {info['path']} is older than "
                f"{'.'.join(map(str, _MIN_VERSION))}; the named-argument CLI is required.\n"
                + hyphy_install_hint()
            ),
            details={"path": info["path"], "version": info.get("version")},
        )
    if info.get("batch_file_root") is not None and not info["batch_files"].get(method):
        raise OrganelleDependencyError(
            code="selection.hyphy_batch_file_missing",
            message=(
                f"HyPhy at {info['path']} has no {HYPHY_METHODS[method]} in "
                f"{info['batch_file_root']}; reinstall HyPhy."
            ),
            details={"path": info["path"], "method": method},
        )
    return info


# =========================================================================
# Inputs: codon alignment + labelled tree
# =========================================================================


def _read_alignment(path: Path) -> list[tuple[str, str]]:
    """Read a codon alignment from FASTA or PAML sequential format."""
    text = path.read_text()
    stripped = text.lstrip()
    if stripped.startswith(">"):
        from .._bio import read_fasta

        return [(name, seq) for name, seq in read_fasta(path)]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    header = lines[0].split() if lines else []
    if len(header) < 2 or not header[0].isdigit() or not header[1].isdigit():
        raise OrganelleInputError(
            code="selection.hyphy_alignment_format",
            message=f"alignment is neither FASTA nor PAML sequential: {path}",
            details={"alignment": str(path)},
        )
    n_seqs, length = int(header[0]), int(header[1])
    records: list[tuple[str, str]] = []
    index = 1
    while index < len(lines) and len(records) < n_seqs:
        name = lines[index]
        index += 1
        chunks: list[str] = []
        while index < len(lines) and sum(map(len, chunks)) < length:
            chunks.append(lines[index].replace(" ", ""))
            index += 1
        records.append((name, "".join(chunks)))
    return records


def _validate_codon_alignment(records: list[tuple[str, str]], path: Path) -> tuple[list[tuple[str, str]], list[str]]:
    notes: list[str] = []
    if len(records) < 3:
        raise OrganelleInputError(
            code="selection.hyphy_too_few_sequences",
            message=f"HyPhy selection analyses need >= 3 sequences; got {len(records)}.",
            details={"alignment": str(path), "n_sequences": len(records)},
        )
    names = [name for name, _ in records]
    if len(set(names)) != len(names):
        raise OrganelleInputError(
            code="selection.hyphy_duplicate_names",
            message="alignment sequence names must be unique",
            details={"alignment": str(path)},
        )
    seqs = [seq.upper().replace("U", "T").replace(".", "-") for _, seq in records]
    lengths = {len(seq) for seq in seqs}
    if len(lengths) != 1:
        raise OrganelleInputError(
            code="selection.hyphy_unaligned",
            message="sequences differ in length; supply a codon alignment (selection.pal2nal)",
            details={"alignment": str(path), "lengths": sorted(lengths)},
        )
    length = lengths.pop()
    if length == 0 or length % 3:
        raise OrganelleInputError(
            code="selection.hyphy_not_in_frame",
            message=f"alignment length {length} is not a positive multiple of 3",
            details={"alignment": str(path), "length": length},
        )
    for name, seq in zip(names, seqs, strict=True):
        if not _VALID_NT.match(seq):
            raise OrganelleInputError(
                code="selection.hyphy_invalid_characters",
                message=f"sequence {name!r} contains non-nucleotide characters",
                details={"alignment": str(path), "sequence": name},
            )
    # A shared terminal stop column is removed (HyPhy rejects stop codons);
    # an internal stop is a frame/annotation error and is refused.
    last = [seq[-3:] for seq in seqs]
    if all(codon in _STOP_CODONS or codon == "---" for codon in last) and any(
        codon in _STOP_CODONS for codon in last
    ):
        seqs = [seq[:-3] for seq in seqs]
        notes.append("terminal_stop_codon_removed")
    for name, seq in zip(names, seqs, strict=True):
        for pos in range(0, len(seq), 3):
            if seq[pos : pos + 3] in _STOP_CODONS:
                raise OrganelleInputError(
                    code="selection.hyphy_internal_stop_codon",
                    message=(
                        f"sequence {name!r} has a stop codon at codon {pos // 3 + 1}; "
                        "HyPhy needs an in-frame alignment without stops"
                    ),
                    details={"alignment": str(path), "sequence": name, "codon": pos // 3 + 1},
                )
    return list(zip(names, seqs, strict=True)), notes


def _serialize(node: _Node, *, is_root: bool) -> str:
    text = ""
    if node.children:
        text = "(" + ",".join(_serialize(child, is_root=False) for child in node.children) + ")"
        # internal labels are usually support values; HyPhy would treat them as
        # node names (and choke on duplicates), so they are dropped
    else:
        text = node.name
    if node.marks and not is_root:
        # HyPhy accepts one tag per branch; Foreground wins over other groups.
        tag = _FOREGROUND if _FOREGROUND in node.marks else sorted(node.marks)[0]
        text += "{" + tag + "}"
    if node.length is not None and not is_root:
        text += ":" + node.length
    return text


def label_tree_for_hyphy(
    tree: str | Path,
    foreground_labels: Sequence[str] | None = None,
    *,
    mark_clade: bool = False,
) -> tuple[str, dict[str, Any]]:
    """Return a HyPhy-annotated Newick string and a labelling summary.

    ``foreground_labels`` are exact tip names (codeml's ``#1`` convention for
    ``selection.branch_site_model``); ``mark_clade=True`` also marks all
    branches of the smallest clade spanning them (stem included). Existing
    codeml ``#n``/``$n`` marks and HyPhy ``{Tag}`` marks are converted.
    """
    candidate = Path(str(tree))
    tree_text = candidate.read_text() if candidate.is_file() else str(tree)
    root = _parse_newick(tree_text)
    _propagate_clade_marks(root)
    labels = list(foreground_labels or ())
    tip_names = _label_tips(
        root, {label: {_FOREGROUND} for label in labels}, error_prefix="hyphy"
    )
    if mark_clade and labels:
        clade = _smallest_clade(root, set(labels))
        if clade is not None and clade is not root:
            clade.clade_marks.add(_FOREGROUND)
            _propagate_clade_marks(clade)
        elif clade is root:
            raise OrganelleInputError(
                code="selection.hyphy_foreground_clade_is_root",
                message=(
                    "the smallest clade spanning the foreground labels is the whole tree "
                    "under its current rooting; reroot the tree or mark tips only"
                ),
                details={"labels": labels},
            )

    def count(node: _Node, is_root: bool) -> tuple[int, int, list[str]]:
        marked = int(_FOREGROUND in node.marks and not is_root)
        internal = int(marked and bool(node.children))
        names = [node.name] if marked and not node.children else []
        for child in node.children:
            m, i, n = count(child, False)
            marked, internal, names = marked + m, internal + i, names + n
        return marked, internal, names

    n_marked, n_internal, fg_tips = count(root, True)
    summary = {
        "n_tips": len(tip_names),
        "n_foreground_branches": n_marked,
        "n_foreground_internal_branches": n_internal,
        "foreground_tips": sorted(fg_tips),
        "tip_names": tip_names,
    }
    return _serialize(root, is_root=True) + ";", summary


# =========================================================================
# JSON parsing
# =========================================================================


def _finite(value: Any) -> Any:
    """Make HyPhy numbers JSON-safe for the frozen result contract."""
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return value
    if isinstance(value, Mapping):
        return {str(k): _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    return value


def _rate_classes(distribution: Mapping[str, Any] | None) -> list[dict[str, float]]:
    if not isinstance(distribution, Mapping):
        return []
    rows = []
    for key in sorted(distribution, key=lambda k: int(k) if str(k).isdigit() else 0):
        item = distribution[key]
        if isinstance(item, Mapping) and "omega" in item:
            rows.append({"omega": item["omega"], "proportion": item.get("proportion")})
    return rows


def _analysis_meta(data: Mapping[str, Any]) -> dict[str, Any]:
    analysis = data.get("analysis", {}) or {}
    inputs = data.get("input", {}) or {}
    tested = (data.get("tested") or {}).get("0", {})
    n_test = sum(1 for role in tested.values() if str(role).lower() in {"test", "foreground"})
    return {
        "batch_version": analysis.get("version"),
        "settings": analysis.get("settings", {}),
        "n_sequences": inputs.get("number of sequences"),
        "n_codons": inputs.get("number of sites"),
        "n_tested_branches": n_test,
        "n_branches": len(tested),
    }


def _mle_rows(data: Mapping[str, Any]) -> tuple[list[str], list[list[Any]]]:
    mle = data.get("MLE", {}) or {}
    headers = [str(h[0]) if isinstance(h, (list, tuple)) else str(h) for h in mle.get("headers", [])]
    # MEME headers carry HTML ("&beta;<sup>+</sup>"); normalise to "beta+".
    clean = [
        re.sub(r"<[^>]+>", "", h.replace("&alpha;", "alpha").replace("&beta;", "beta"))
        for h in headers
    ]
    content = mle.get("content", {}) or {}
    rows = content.get("0", []) if isinstance(content, Mapping) else content
    return clean, list(rows)


def _parse_busted(data: Mapping[str, Any], alpha: float) -> dict[str, Any]:
    test = data.get("test results", {}) or {}
    fits = data.get("fits", {}) or {}
    unconstrained = fits.get("Unconstrained model", {}) or {}
    constrained = fits.get("Constrained model", {}) or {}
    test_dist = _rate_classes((unconstrained.get("Rate Distributions") or {}).get("Test"))
    bg_dist = _rate_classes((unconstrained.get("Rate Distributions") or {}).get("Background"))
    ers = ((data.get("Evidence Ratios") or {}).get("optimized null") or [[]])[0]
    er_sites = sorted(
        (
            {"site": index + 1, "evidence_ratio": value}
            for index, value in enumerate(ers)
            if isinstance(value, (int, float)) and value >= 10
        ),
        key=lambda row: -row["evidence_ratio"],
    )
    p_value = test.get("p-value")
    top = max(test_dist, key=lambda r: r["omega"]) if test_dist else {}
    return {
        "lrt": test.get("LRT"),
        "p_value": p_value,
        "significant": bool(p_value is not None and p_value <= alpha),
        "lnL_unconstrained": unconstrained.get("Log Likelihood"),
        "lnL_constrained": constrained.get("Log Likelihood"),
        "omega_distribution_test": test_dist,
        "omega_distribution_background": bg_dist,
        "omega_max": top.get("omega"),
        "omega_max_proportion": top.get("proportion"),
        "evidence_ratio_sites": er_sites,
        "n_evidence_ratio_sites": len(er_sites),
    }


def _parse_absrel(data: Mapping[str, Any], alpha: float) -> dict[str, Any]:
    attributes = (data.get("branch attributes") or {}).get("0", {}) or {}
    tested = (data.get("tested") or {}).get("0", {}) or {}
    rows = []
    for name, attrs in attributes.items():
        if not isinstance(attrs, Mapping) or "Corrected P-value" not in attrs:
            continue
        if tested and str(tested.get(name, "test")).lower() not in {"test", "foreground"}:
            continue
        distribution = attrs.get("Rate Distributions") or []
        rows.append(
            {
                "branch": attrs.get("original name", name),
                "lrt": attrs.get("LRT"),
                "p_uncorrected": attrs.get("Uncorrected P-value"),
                "p_corrected": attrs.get("Corrected P-value"),
                "rate_classes": attrs.get("Rate classes"),
                "omega_baseline": attrs.get("Baseline MG94xREV omega ratio"),
                "omega_distribution": [
                    {"omega": pair[0], "proportion": pair[1]}
                    for pair in distribution
                    if isinstance(pair, (list, tuple)) and len(pair) == 2
                ],
                "selected": bool(
                    attrs.get("Corrected P-value") is not None
                    and attrs["Corrected P-value"] <= alpha
                ),
            }
        )
    rows.sort(key=lambda row: (row["p_corrected"] if row["p_corrected"] is not None else 2.0))
    test = data.get("test results", {}) or {}
    selected = [row for row in rows if row["selected"]]
    return {
        "branches": rows,
        "selected_branches": [row["branch"] for row in selected],
        "n_selected_branches": len(selected),
        "n_tested": test.get("tested", len(rows)),
        "hyphy_positive_test_results": test.get("positive test results"),
        "significant": bool(selected),
    }


def _parse_relax(data: Mapping[str, Any], alpha: float) -> dict[str, Any]:
    test = data.get("test results", {}) or {}
    fits = data.get("fits", {}) or {}
    alternative = fits.get("RELAX alternative", {}) or {}
    null = fits.get("RELAX null", {}) or {}
    k = test.get("relaxation or intensification parameter")
    p_value = test.get("p-value")
    significant = bool(p_value is not None and p_value <= alpha)
    if k is None:
        direction = "unknown"
    elif k > 1:
        direction = "intensified"
    elif k < 1:
        direction = "relaxed"
    else:
        direction = "unchanged"
    distributions = alternative.get("Rate Distributions") or {}
    settings = (data.get("analysis") or {}).get("settings", {}) or {}
    warnings = [key for key in settings if str(key).startswith("convergence")]
    return {
        "k": k,
        "lrt": test.get("LRT"),
        "p_value": p_value,
        "significant": significant,
        "direction": direction,
        "conclusion": (f"selection {direction} on test branches" if significant else "no significant change"),
        "lnL_alternative": alternative.get("Log Likelihood"),
        "lnL_null": null.get("Log Likelihood"),
        "omega_distribution_test": _rate_classes(distributions.get("Test")),
        "omega_distribution_reference": _rate_classes(distributions.get("Reference")),
        "convergence_warnings": warnings,
    }


def _column(headers: list[str], *names: str) -> int | None:
    for name in names:
        if name in headers:
            return headers.index(name)
    return None


def _parse_meme(data: Mapping[str, Any], alpha: float) -> dict[str, Any]:
    headers, rows = _mle_rows(data)
    col = {
        "alpha": _column(headers, "alpha"),
        "beta_plus": _column(headers, "beta+"),
        "p_plus": _column(headers, "p+"),
        "lrt": _column(headers, "LRT"),
        "p_value": _column(headers, "p-value"),
        "n_branches": _column(headers, "# branches under selection"),
    }
    sites = []
    for index, row in enumerate(rows):
        p_col = col["p_value"]
        if p_col is None or row[p_col] is None or row[p_col] > alpha:
            continue
        sites.append(
            {"site": index + 1, **{k: row[c] for k, c in col.items() if c is not None}}
        )
    return {
        "sites": sites,
        "n_sites": len(sites),
        "n_codons_tested": len(rows),
        "significant": bool(sites),
    }


def _parse_fel(data: Mapping[str, Any], alpha: float) -> dict[str, Any]:
    headers, rows = _mle_rows(data)
    a_col, b_col = _column(headers, "alpha"), _column(headers, "beta")
    lrt_col, p_col = _column(headers, "LRT"), _column(headers, "p-value")
    positive, negative = [], []
    for index, row in enumerate(rows):
        if p_col is None or a_col is None or b_col is None:
            break
        if row[p_col] is None or row[p_col] > alpha:
            continue
        entry = {
            "site": index + 1,
            "alpha": row[a_col],
            "beta": row[b_col],
            "lrt": row[lrt_col] if lrt_col is not None else None,
            "p_value": row[p_col],
        }
        (positive if row[b_col] > row[a_col] else negative).append(entry)
    return {
        "positive_sites": positive,
        "negative_sites": negative,
        "n_positive_sites": len(positive),
        "n_negative_sites": len(negative),
        "n_codons_tested": len(rows),
        "significant": bool(positive),
    }


_PARSERS = {
    "busted": _parse_busted,
    "absrel": _parse_absrel,
    "relax": _parse_relax,
    "meme": _parse_meme,
    "fel": _parse_fel,
}


def _detect_method(data: Mapping[str, Any]) -> str:
    test = data.get("test results", {}) or {}
    if "relaxation or intensification parameter" in test:
        return "relax"
    if "positive test results" in test:
        return "absrel"
    if "Evidence Ratios" in data or "Unconstrained model" in (data.get("fits") or {}):
        return "busted"
    headers, _ = _mle_rows(data)
    if "beta+" in headers:
        return "meme"
    if "alpha=beta" in headers:
        return "fel"
    raise OrganelleInputError(
        code="selection.hyphy_json_unrecognized",
        message="JSON is not a BUSTED/aBSREL/RELAX/MEME/FEL result",
        details={"keys": sorted(map(str, data))[:20]},
    )


def parse_hyphy_json(
    json_path: str | Path,
    *,
    method: str | None = None,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Parse the key results of a HyPhy BUSTED/aBSREL/RELAX/MEME/FEL JSON.

    Works on JSON from this module, a local HyPhy run or Datamonkey. The
    method is detected from the JSON layout unless ``method`` is given.
    ``alpha`` is the significance threshold (aBSREL uses Holm-corrected p).
    """
    data = json.loads(Path(json_path).read_text())
    key = (method or _detect_method(data)).lower()
    if key not in _PARSERS:
        raise OrganelleInputError(
            code="selection.hyphy_method_unknown",
            message=f"unknown HyPhy method {method!r}; use one of {sorted(_PARSERS)}",
            details={"method": method},
        )
    parsed = _PARSERS[key](data, alpha)
    return _finite({"method": key, "alpha": alpha, **_analysis_meta(data), **parsed})


# =========================================================================
# Running an analysis
# =========================================================================


def _prepare_run_dir(op: str, output_dir: str | Path | None) -> Path:
    if output_dir is not None:
        out = Path(output_dir)
    else:
        from ..runtime import managed_run_path

        out = managed_run_path(_contract.operation_id(op), uuid4().hex)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _run(
    method: str,
    *,
    alignment: str | Path,
    tree: str | Path,
    foreground_labels: Sequence[str] | None,
    mark_clade: bool,
    output_dir: str | Path | None,
    genetic_code: str,
    threads: int,
    alpha: float,
    extra_args: Sequence[str],
    branches_required: bool,
    hyphy_path: str | Path | None,
    timeout: float | None,
    parameters: Mapping[str, Any],
) -> OrganelleResult:
    op = f"hyphy_{method}"
    info = _require_hyphy(method, hyphy_path)
    alignment_path = Path(alignment)
    records, notes = _validate_codon_alignment(_read_alignment(alignment_path), alignment_path)
    labelled, labelling = label_tree_for_hyphy(tree, foreground_labels, mark_clade=mark_clade)
    aln_names = {name for name, _ in records}
    tree_names = set(labelling["tip_names"])
    if aln_names != tree_names:
        raise OrganelleInputError(
            code="selection.hyphy_names_mismatch",
            message="alignment sequence names and tree tip names differ",
            details={
                "only_in_alignment": sorted(aln_names - tree_names),
                "only_in_tree": sorted(tree_names - aln_names),
            },
        )
    has_foreground = labelling["n_foreground_branches"] > 0
    if branches_required and not has_foreground:
        raise OrganelleInputError(
            code="selection.hyphy_foreground_required",
            message=(
                f"HyPhy {method.upper()} needs test (foreground) branches: pass "
                "foreground_labels or a tree marked with #1/$1/{Foreground}"
            ),
            details={"method": method},
        )

    out = _prepare_run_dir(op, output_dir)
    aln_file = out / "codon_alignment.fasta"
    aln_file.write_text("".join(f">{name}\n{seq}\n" for name, seq in records))
    tree_file = out / "tree.hyphy.nwk"
    tree_file.write_text(labelled + "\n")
    json_file = out / f"{method}.json"
    log_file = out / f"{method}.log"
    if json_file.exists():
        json_file.unlink()

    argv = [
        info["path"],
        f"CPU={max(1, int(threads))}",
        method,
        "--alignment",
        str(aln_file),
        "--tree",
        str(tree_file),
        "--code",
        genetic_code,
    ]
    if method == "relax":
        argv += ["--test", _FOREGROUND]
    else:
        argv += ["--branches", _FOREGROUND if has_foreground else "All"]
    argv += [*extra_args, "--output", str(json_file)]

    completed = run_external(
        argv,
        cwd=out,
        timeout=timeout,
        input_data="",  # never let HyPhy block on an interactive prompt
        code="selection.hyphy_failed",
        tool="hyphy",
        extra_details={"method": method, "run_dir": str(out)},
    )
    log_file.write_text(completed.stdout + ("\n" + completed.stderr if completed.stderr else ""))
    if not json_file.is_file():
        from ..core.errors import OrganelleExecutionError
        from ..core.external import tail_lines

        raise OrganelleExecutionError(
            code="selection.hyphy_no_json",
            message=f"HyPhy {method} exited 0 but wrote no JSON: {tail_lines(completed.stdout, 5)}",
            details={"argv": argv, "stdout_tail": tail_lines(completed.stdout, 40), "run_dir": str(out)},
        )

    parsed = parse_hyphy_json(json_file, method=method, alpha=alpha)
    metrics = {
        **parsed,
        "hyphy_version": info.get("version"),
        "hyphy_path": info["path"],
        "genetic_code": genetic_code,
        "branch_set": "Foreground" if has_foreground else "All",
        "foreground_tips": labelling["foreground_tips"],
        "n_foreground_branches": labelling["n_foreground_branches"],
        "input_notes": notes,
        "raw_json": str(json_file),
    }
    flags = [f"hyphy:{method}"]
    if parsed.get("significant"):
        flags.append(
            {
                "busted": "episodic_diversifying_selection",
                "absrel": "positive_selection_branches",
                "relax": f"selection_{parsed.get('direction')}",
                "meme": "episodic_selection_sites",
                "fel": "pervasive_positive_sites",
            }[method]
        )
    flags.extend(notes)
    if method == "relax" and parsed.get("convergence_warnings"):
        flags.append("hyphy_convergence_warning")

    return _contract.ok(
        op,
        summary_text=_summary(method, parsed),
        metrics=_finite(metrics),
        findings=_contract.findings(op, _findings(method, parsed)),
        flags=flags,
        artifacts=_contract.artifacts((json_file, aln_file, tree_file, log_file)),
        method=f"hyphy_{method}",
        parameters=parameters,
        software_versions={"hyphy": info.get("version")},
        argv=[str(a) for a in argv],
    )


def _fmt(value: Any, spec: str = ".4g") -> str:
    return format(value, spec) if isinstance(value, (int, float)) else str(value)


def _summary(method: str, parsed: Mapping[str, Any]) -> str:
    if method == "busted":
        verdict = "evidence" if parsed["significant"] else "no evidence"
        return (
            f"BUSTED: {verdict} of episodic diversifying selection "
            f"(LRT={_fmt(parsed['lrt'])}, p={_fmt(parsed['p_value'])}; "
            f"omega_max={_fmt(parsed['omega_max'])} on {_fmt(parsed['omega_max_proportion'])} of sites)."
        )
    if method == "absrel":
        return (
            f"aBSREL: {parsed['n_selected_branches']}/{parsed['n_tested']} tested branches under "
            f"episodic diversifying selection (Holm-corrected p <= {parsed['alpha']})."
        )
    if method == "relax":
        return (
            f"RELAX: k={_fmt(parsed['k'])}, LRT={_fmt(parsed['lrt'])}, p={_fmt(parsed['p_value'])} "
            f"-> {parsed['conclusion']}."
        )
    if method == "meme":
        return f"MEME: {parsed['n_sites']} sites under episodic diversifying selection (p <= {parsed['alpha']})."
    return (
        f"FEL: {parsed['n_positive_sites']} positively and {parsed['n_negative_sites']} negatively "
        f"selected sites (p <= {parsed['alpha']})."
    )


def _findings(method: str, parsed: Mapping[str, Any]) -> tuple[tuple[str, Any], ...]:
    if method == "busted":
        return (("lrt", parsed["lrt"]), ("p_value", parsed["p_value"]), ("omega_max", parsed["omega_max"]))
    if method == "absrel":
        return (
            ("n_selected_branches", parsed["n_selected_branches"]),
            *[(f"branch:{row['branch']}", row["p_corrected"]) for row in parsed["branches"][:10]],
        )
    if method == "relax":
        return (("k", parsed["k"]), ("lrt", parsed["lrt"]), ("p_value", parsed["p_value"]))
    if method == "meme":
        return (("n_sites", parsed["n_sites"]), *[(f"site:{s['site']}", s.get("p_value")) for s in parsed["sites"][:10]])
    return (
        ("n_positive_sites", parsed["n_positive_sites"]),
        ("n_negative_sites", parsed["n_negative_sites"]),
        *[(f"positive_site:{s['site']}", s["p_value"]) for s in parsed["positive_sites"][:10]],
    )


def _common_parameters(**values: Any) -> dict[str, Any]:
    return {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in values.items()}


def hyphy_busted(
    *,
    alignment: str | Path,
    tree: str | Path,
    foreground_labels: list[str] | None = None,
    mark_clade: bool = False,
    srv: bool = True,
    alpha: float = 0.05,
    genetic_code: str = "Universal",
    threads: int = 1,
    hyphy_path: str | Path | None = None,
    timeout: float | None = None,
) -> OrganelleResult:
    """BUSTED: gene-wide test for episodic diversifying selection.

    Tests the foreground branches (``foreground_labels`` / a marked tree) or,
    without any foreground, all branches. ``metrics`` carry the LRT, p-value,
    the unconstrained omega distribution and sites with evidence ratio >= 10;
    the raw HyPhy JSON is kept as an artifact (``metrics["raw_json"]``).
    The run lives in package-managed storage (output-boundary contract);
    ``metrics`` and ``artifacts`` carry the paths.
    """
    params = _common_parameters(
        foreground_labels=foreground_labels or [], mark_clade=mark_clade, srv=srv,
        alpha=alpha, genetic_code=genetic_code, threads=threads,
    )
    return _run(
        "busted", alignment=alignment, tree=tree, foreground_labels=foreground_labels,
        mark_clade=mark_clade, output_dir=None, genetic_code=genetic_code,
        threads=threads, alpha=alpha, extra_args=["--srv", "Yes" if srv else "No"],
        branches_required=False, hyphy_path=hyphy_path, timeout=timeout, parameters=params,
    )


def hyphy_absrel(
    *,
    alignment: str | Path,
    tree: str | Path,
    foreground_labels: list[str] | None = None,
    mark_clade: bool = False,
    alpha: float = 0.05,
    genetic_code: str = "Universal",
    threads: int = 1,
    hyphy_path: str | Path | None = None,
    timeout: float | None = None,
) -> OrganelleResult:
    """aBSREL: which branches show episodic diversifying selection.

    Tests the foreground branches, or all branches when none are marked
    (exploratory mode; the Holm correction then spans every branch).
    ``metrics["branches"]`` lists every tested branch with LRT, uncorrected
    and Holm-corrected p-values and its omega distribution.
    """
    params = _common_parameters(
        foreground_labels=foreground_labels or [], mark_clade=mark_clade, alpha=alpha,
        genetic_code=genetic_code, threads=threads,
    )
    return _run(
        "absrel", alignment=alignment, tree=tree, foreground_labels=foreground_labels,
        mark_clade=mark_clade, output_dir=None, genetic_code=genetic_code,
        threads=threads, alpha=alpha, extra_args=[], branches_required=False,
        hyphy_path=hyphy_path, timeout=timeout, parameters=params,
    )


def hyphy_relax(
    *,
    alignment: str | Path,
    tree: str | Path,
    foreground_labels: list[str] | None = None,
    mark_clade: bool = False,
    models: str = "Minimal",
    alpha: float = 0.05,
    genetic_code: str = "Universal",
    threads: int = 1,
    hyphy_path: str | Path | None = None,
    timeout: float | None = None,
) -> OrganelleResult:
    """RELAX: is selection relaxed (k<1) or intensified (k>1) on test branches?

    Test branches are the foreground (required); every unlabelled branch is
    the reference set. ``models="All"`` also fits the descriptive
    partitioned model (slower). ``metrics`` carry k, LRT, p-value and the
    direction; HyPhy convergence notes are surfaced as a flag.
    """
    if models not in {"Minimal", "All"}:
        raise OrganelleInputError(
            code="selection.hyphy_relax_models",
            message="models must be 'Minimal' or 'All'",
            details={"models": models},
        )
    params = _common_parameters(
        foreground_labels=foreground_labels or [], mark_clade=mark_clade, models=models,
        alpha=alpha, genetic_code=genetic_code, threads=threads,
    )
    return _run(
        "relax", alignment=alignment, tree=tree, foreground_labels=foreground_labels,
        mark_clade=mark_clade, output_dir=None, genetic_code=genetic_code,
        threads=threads, alpha=alpha, extra_args=["--models", models],
        branches_required=True, hyphy_path=hyphy_path, timeout=timeout, parameters=params,
    )


def hyphy_meme(
    *,
    alignment: str | Path,
    tree: str | Path,
    foreground_labels: list[str] | None = None,
    mark_clade: bool = False,
    alpha: float = 0.1,
    genetic_code: str = "Universal",
    threads: int = 1,
    hyphy_path: str | Path | None = None,
    timeout: float | None = None,
) -> OrganelleResult:
    """MEME: sites under episodic diversifying selection (HyPhy default p <= 0.1)."""
    params = _common_parameters(
        foreground_labels=foreground_labels or [], mark_clade=mark_clade, alpha=alpha,
        genetic_code=genetic_code, threads=threads,
    )
    return _run(
        "meme", alignment=alignment, tree=tree, foreground_labels=foreground_labels,
        mark_clade=mark_clade, output_dir=None, genetic_code=genetic_code,
        threads=threads, alpha=alpha, extra_args=["--pvalue", str(alpha)],
        branches_required=False, hyphy_path=hyphy_path, timeout=timeout, parameters=params,
    )


def hyphy_fel(
    *,
    alignment: str | Path,
    tree: str | Path,
    foreground_labels: list[str] | None = None,
    mark_clade: bool = False,
    srv: bool = True,
    alpha: float = 0.1,
    genetic_code: str = "Universal",
    threads: int = 1,
    hyphy_path: str | Path | None = None,
    timeout: float | None = None,
) -> OrganelleResult:
    """FEL: sites under pervasive positive or negative selection (p <= 0.1 default)."""
    params = _common_parameters(
        foreground_labels=foreground_labels or [], mark_clade=mark_clade, srv=srv,
        alpha=alpha, genetic_code=genetic_code, threads=threads,
    )
    return _run(
        "fel", alignment=alignment, tree=tree, foreground_labels=foreground_labels,
        mark_clade=mark_clade, output_dir=None, genetic_code=genetic_code,
        threads=threads, alpha=alpha,
        extra_args=["--srv", "Yes" if srv else "No", "--pvalue", str(alpha)],
        branches_required=False, hyphy_path=hyphy_path, timeout=timeout, parameters=params,
    )
