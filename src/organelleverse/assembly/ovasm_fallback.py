"""Never-fail escalation around the OVASM runner.

An organelle assembly must not fail silently, and when something goes wrong it must say why,
adjust where it can, and still deliver the organelle. :func:`run_ovasm_resilient` wraps
``ovasm_runner.run_ovasm`` for the two callers (the desktop assembly tools and the canonical
``ovasm`` backend):

* HiFi whole-genome runs (built-in or custom seeds) go through up to three attempts, each in
  its own directory ``attempts/<name>/`` of the output directory: ``requested`` (the run as
  asked), ``also_discover`` (the same plus the seed-free depth clusters called as the target)
  and ``discover`` (no seeds at all). An attempt is accepted when :func:`assess` finds
  neither a problem nor a concern in its summary; the first accepted attempt ends the
  escalation, so a run that works the first time costs what it always cost.
* The best attempt (see :func:`score`) is published in the output directory exactly where the
  runner writes a plain run, and its ``summary.json`` gains ``status``, ``attempts``,
  ``chosen_attempt``, ``reasons``, ``suggestions`` and ``concerns``.
* Every other run (target reads, Illumina, noisy long reads, ``fallback: false``) is a single
  run as before; it gets the same structured report, and a failure is written to
  ``summary.json`` before the error is raised.

The module imports nothing from ``ovasm_runner`` at import time (the runner imports
:func:`choose_target_clusters` from here).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

# --- relaxed call of a depth cluster ------------------------------------------------------

#: `ovasm identify` calls a depth cluster the target only at >= 0.3 conserved-gene segments per
#: read (and >= 5 distinct genes, >= 2 segments per gene). Carex laevigata and C. sylvatica
#: hold their mitochondria in one cluster each (15,457 of 16,520 reads, ~58x for laevigata)
#: whose mitochondrial density is 0.25-0.28 because the chloroplast-derived and nuclear
#: reads in it dilute it: no cluster was called and the run failed. A cluster the target is
#: clearly ahead in is taken when no cluster is called the target. Calibrated on the 20 seen
#: Oatk-record taxa (`run-union/*/discover/identify.json`, 40 organelle x taxon cases):
#: it fires in exactly the five cases where no cluster was called the mitochondrion (Carex
#: laevigata and C. sylvatica, the "none" clusters; Solanum dulcamara, Populus nigra and
#: Alnus glutinosa, the "mixed" ones) and picks one cluster each (the one holding 4,000-8,700
#: mitochondrial gene segments); no plastid case needed it. Without the segment floor it
#: would also take a 69-segment cluster of Hypericum androsaemum for the plastid.
RELAXED_MIN_GENES = 5
RELAXED_MIN_SEGMENTS = 300
RELAXED_MIN_RATIO = 2.0
RELAXED_MIN_DENSITY = 0.1


def _other(organelle: str) -> str:
    return "plastid" if organelle == "mitochondrion" else "mitochondrion"


def relaxed_target_call(cluster: dict[str, Any], organelle: str) -> bool:
    """Whether the identify statistics of a depth cluster show the target clearly ahead."""
    stats = cluster.get("stats")
    if not stats or cluster.get("call") == _other(organelle):
        return False
    target, other = stats[organelle], stats[_other(organelle)]
    return (
        target["genes"] >= RELAXED_MIN_GENES
        and target["segments"] >= RELAXED_MIN_SEGMENTS
        and target["segments"] >= RELAXED_MIN_RATIO * other["segments"]
        and target["density"] >= RELAXED_MIN_DENSITY
    )


def choose_target_clusters(
    called: list[dict[str, Any]], organelle: str, *, relaxed: bool
) -> list[dict[str, Any]]:
    """The depth clusters that are the target: those called so; with `relaxed`, and none
    called so, those the target is clearly ahead in (flagged `weak_call`)."""
    chosen = [c for c in called if c["call"] == organelle]
    if chosen or not relaxed:
        return chosen
    chosen = [c for c in called if relaxed_target_call(c, organelle)]
    for cluster in chosen:
        cluster["weak_call"] = True
    return chosen


# --- what an attempt gave -------------------------------------------------------------------

#: How many more k the runner tries after an accepted k that another k could cure (see
#: `k_curable`); the search ends at the first complete result, and never revisits a k.
MAX_EXTRA_K = 4

#: A mitochondrion whose discovery pool (the seed-recruited reads plus the depth clusters
#: called as the mitochondrion) is at least this many times the seed pool gets an
#: `also_discover` attempt even when the requested run was accepted. Seen-data calibration
#: (Oatk-record taxa, seed-only vs seed + discovery, union reads / seed reads): every
#: mitochondrion whose seed-only result stayed under 97% of the published genome had >= 1.35
#: (Hedera 1.35, Quercus cerris 1.4, Calendula 1.46, E. peplus 1.49, Balanites 2.11,
#: Hypericum 3.4, E. lathyris 4.1, Adonis 6.7, Galeopsis 9.5, Inga 51) and the healthy
#: Solanum 1.0. Plastids showed no usable separation (1.05-1.44), so only the mitochondrion
#: is checked.
POOL_RATIO_MIN = 1.30


def k_curable(organelle: str, molecules: list[dict[str, Any]], linear: dict[str, Any]) -> bool:
    """Whether an accepted result at one k leaves what a different k could cure: a plastid
    without a circular molecule (Draba incana: one linear 82,858 bp molecule at k=1001, the
    153,566 bp circle at k=701..251), or a mitochondrion with graph components or paths
    that no molecule covers."""
    if not molecules:
        return False
    if organelle == "plastid":
        return not any(m.get("circular") for m in molecules)
    return bool(linear.get("unsolved_components") or linear.get("partial_paths"))


#: An assembly shorter than this share of the closest builtin reference genome is suspect.
MIN_REFERENCE_SHARE = 0.5

#: Graph segments shorter than this are not delivered as unitigs when longer ones exist.
UNITIG_MIN_LENGTH = 1000


def _reference_length(identity: dict[str, Any] | None, organelle: str) -> int | None:
    """Length of the closest builtin reference genome of an assembly, if it can be told."""
    try:
        from organelleverse._ovasm_seeddb import bundled_seed

        accession = identity["closest"]["accession"]  # type: ignore[index]
        _, seed = bundled_seed(organelle)
        manifest = json.loads(Path(seed["manifest"]).read_text(encoding="utf-8"))
        for ref in manifest["databases"][organelle]["references"]:
            if ref["accession"] == accession:
                return int(ref["genome_length_bp"])
    except Exception:
        return None
    return None


def assess(summary: dict[str, Any], organelle: str) -> dict[str, Any]:
    """Judge a finished run from its summary.json.

    `problems` make an attempt unacceptable: no molecule; no k explained 50% of the graph
    (`k_accepted` false: the molecules are not to be trusted); the identity check says the
    assembly is the other organelle or no organelle; a plastid without a circular molecule
    (a plastome closes into one circle; seeds alone left Carex laevigata, C. sylvatica and
    Eleocharis plastids linear at 85%, 71%, 79% of the published genome).

    `concerns` are signs of an incomplete assembly that a healthy one can show too, so they
    do not make a run fail, but they do start the next attempt and keep the status at
    `partial` when no attempt gets rid of them: graph components without a representative
    sequence, an assembly under half the length of the closest builtin reference genome
    (Balanites mitochondrion: one 122 kb circle, 31% of the published genome, closest
    reference 365 kb).

    `notes` are informational only (a mitochondrion without a circular molecule): they are
    reported in `concerns` of the summary but neither start an attempt nor change the status.
    """
    molecules = int(summary.get("molecules") or 0)
    lengths = [int(x) for x in summary.get("lengths") or []]
    circular = [bool(x) for x in summary.get("circular") or []]
    problems: list[str] = []
    concerns: list[str] = []
    notes: list[str] = []
    tried = summary.get("k_tried") or []
    kept = tried[int(summary.get("kept_attempt") or 0)] if tried else {}
    if not molecules:
        paths = int(summary.get("partial_paths") or 0)
        problems.append(
            "the graph gave no molecule"
            + (f" (only {paths} partial path(s))" if paths else "")
            + f" at any k tried ({[t['k'] for t in tried]})"
            if tried
            else "the graph gave no molecule"
        )
    elif not summary.get("k_accepted", True):
        problems.append(
            f"no k gave molecules explaining {summary.get('min_explained', 0.5):.0%} of the graph "
            f"(best {float(kept.get('explained', 0)):.0%} at k={kept.get('k')}): the molecules "
            "are not to be trusted"
        )
    for warning in summary.get("warnings") or []:
        problems.append(warning)
    if molecules and not any(circular):
        text = f"none of the {molecules} molecule(s) is circular"
        if organelle == "plastid":
            problems.append(text + " (a plastid genome closes into one circle)")
        else:
            # only a note: healthy linear and multi-chromosome mitochondria are common (four
            # seen ones that were 99.9-100% of the published genome), and this started
            # escalations that cost 2.5x wall time and changed nothing
            notes.append(text + " (a multi-chromosome mitochondrion may have none)")
    if molecules and summary.get("unsolved_components"):
        concerns.append(
            f"{summary['unsolved_components']} graph component(s) have no representative sequence"
        )
    bases = sum(lengths)
    reference = _reference_length(summary.get("identity"), organelle) if molecules else None
    if reference and bases < MIN_REFERENCE_SHARE * reference:
        concerns.append(
            f"the molecules total {bases} bp, under {MIN_REFERENCE_SHARE:.0%} of the closest "
            f"builtin reference genome ({reference} bp)"
        )
    return {
        "problems": problems,
        "concerns": concerns,
        "notes": notes,
        "acceptable": not problems and not concerns,
        "metrics": {
            "molecules": molecules,
            "molecule_bases": bases,
            "circular_molecules": sum(circular),
            "k": summary.get("k"),
            "k_accepted": summary.get("k_accepted"),
            "explained": kept.get("explained"),
            "graph_bases": kept.get("total_length"),
            "unsolved_components": summary.get("unsolved_components"),
            "partial_paths": summary.get("partial_paths"),
            "recruited_reads": summary.get("recruited_reads"),
            "closest_reference_bp": reference,
        },
    }


def score(record: dict[str, Any], organelle: str = "plastid") -> list[Any]:
    """Rank of an attempt, compared lexicographically (larger is better; ties keep the
    earliest attempt): no problems; trusted molecules (a k explained >= 50% of the graph);
    no concerns; then for a plastid a circular molecule before the molecule bases, for a
    mitochondrion the molecule bases before a circular molecule; bases of partial paths;
    bases of the assembly graph. An attempt that raised ranks last.

    The mitochondrion's circle does not come first: Schoenoplectus lacustris' seed-only run
    was seven molecules with one small circle at 78% of the published genome, the run with
    discovery four linear molecules at 99.5%."""
    if record["outcome"] == "error":
        return [False, False, False, False, 0, 0, 0]
    m = record["metrics"]
    circle = bool(m["circular_molecules"])
    head = [
        not record["problems"],
        bool(m["molecules"] and m["k_accepted"] is not False),
        not record["concerns"],
    ]
    body = [circle, m["molecule_bases"]] if organelle == "plastid" else [m["molecule_bases"], circle]
    return [*head, *body, record.get("partial_bases", 0), int(m["graph_bases"] or 0)]


# --- plain-language reasons -------------------------------------------------------------------


def explain_error(error: BaseException, source: str) -> str:
    """The cause of a failed attempt, said plainly."""
    text = str(error)
    few = re.search(r"recruited (\d+) reads \((\d+) bp\)", text)
    if few:
        return (
            f"{source} recruited only {few.group(1)} reads ({few.group(2)} bp): the seeds are "
            "too distant from this species"
        )
    if "recruited no" in text:
        return (
            f"{source} recruited no reads: the seeds are too distant from this species, or the "
            "library holds almost none of this organelle"
        )
    if "no depth cluster" in text:
        return (
            "no depth cluster was called by its conserved proteins as this organelle, and none "
            "was clearly ahead: the organelle is too shallow to stand out from the nuclear "
            "genome, or its reads share a cluster with the other organelle's"
        )
    return f"{type(error).__name__}: {text[:300]}"


def explain_attempt(record: dict[str, Any]) -> str:
    if record["outcome"] == "error":
        return record["error"]
    return "; ".join([*record["problems"], *record["concerns"]]) or "accepted"


def _source_label(params: dict[str, Any]) -> str:
    if str(params.get("seed_source") or "") == "custom" or str(params.get("seed") or "").strip():
        return "the supplied seed"
    return "the built-in seeds"


# --- the escalation ---------------------------------------------------------------------------

ATTEMPT_NAMES = ("requested", "also_discover", "discover")


def _truthy(value: Any, default: bool) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def escalates(params: dict[str, Any], mate: Path | None) -> bool:
    """Whether a run is one that escalates: HiFi whole-genome reads, fallback not switched off."""
    return (
        _truthy(params.get("fallback"), True)
        and str(params.get("strategy") or "hifi_only") == "hifi_only"
        and str(params.get("read_set") or "whole_genome") == "whole_genome"
        and mate is None
    )


def plan(params: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """The attempts of a run, in order: (name, parameter overrides)."""
    source = str(params.get("seed_source") or ("custom" if params.get("seed") else "builtin"))
    steps: list[tuple[str, dict[str, Any]]] = [("requested", {})]
    if source == "discover":
        # seed-free was asked for: the next step is seeds plus discovery
        steps.append(
            ("also_discover", {"seed_source": "builtin", "seed": "", "also_discover": True})
        )
        return steps
    if not params.get("also_discover"):
        steps.append(("also_discover", {"also_discover": True}))
    steps.append(("discover", {"seed_source": "discover", "seed": "", "also_discover": False}))
    return steps


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _partial_bases(summary: dict[str, Any], adir: Path) -> int:
    fasta = adir / "molecules.partial.fasta"
    candidates = [fasta, *adir.glob("*partial*.fasta")]
    for path in candidates:
        if path.is_file():
            return _fasta_bases(path)
    return 0


def _fasta_bases(path: Path) -> int:
    total = 0
    with path.open("rb") as handle:
        for line in handle:
            if not line.startswith(b">"):
                total += len(line.strip())
    return total


def _record(
    name: str,
    overrides: dict[str, Any],
    adir: Path,
    root: Path,
    organelle: str,
    seconds: float,
    summary: dict[str, Any] | None,
    error: BaseException | None,
    source: str,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "name": name,
        "parameters": overrides,
        "directory": str(adir.relative_to(root)) if adir != root else ".",
        "seconds": round(seconds, 1),
    }
    if summary is None:
        record.update(
            outcome="error",
            error=explain_error(error, source) if error is not None else "no summary.json",
            raw_error=str(error)[:600] if error is not None else None,
            error_type=type(error).__name__ if error is not None else None,
            problems=[],
            concerns=[],
            notes=[],
            metrics={},
        )
        return record
    judged = assess(summary, organelle)
    record.update(
        outcome="accepted" if judged["acceptable"] else "not_accepted",
        error=None,
        problems=judged["problems"],
        concerns=judged["concerns"],
        notes=judged["notes"],
        metrics=judged["metrics"],
        partial_bases=_partial_bases(summary, adir),
        weak_cluster_call=_weak_clusters(summary),
    )
    return record


def _weak_clusters(summary: dict[str, Any]) -> list[str]:
    seed_db = summary.get("seed_database") or {}
    clusters = (seed_db.get("discovery") or {}).get("clusters") or seed_db.get("clusters") or []
    return [c["name"] for c in clusters if c.get("weak_call")]


def _link(src: Path, dst: Path) -> None:
    """Hard link `src` as `dst` (a copy where the file system cannot)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _graph_unitigs(gfa: Path, out: Path) -> int:
    """Write the graph's longest sequences as FASTA (all >= UNITIG_MIN_LENGTH bp, else the 20
    longest); returns the number written."""
    segments: list[tuple[int, str, str, str]] = []
    with gfa.open() as handle:
        for line in handle:
            if not line.startswith("S\t"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3 or fields[2] == "*":
                continue
            depth = next((f[5:] for f in fields[3:] if f.startswith("DP:f:")), "")
            segments.append((len(fields[2]), fields[1], fields[2], depth))
    segments.sort(key=lambda s: -s[0])
    chosen = [s for s in segments if s[0] >= UNITIG_MIN_LENGTH] or segments[:20]
    if not chosen:
        return 0
    with out.open("w") as handle:
        for n, (length, name, seq, depth) in enumerate(chosen, 1):
            tag = f" depth={depth}" if depth else ""
            handle.write(f">unitig.{n} segment={name} length={length}{tag} complete=false\n{seq}\n")
    return len(chosen)


def _deliver_partial(adir: Path, summary: dict[str, Any], log: Callable[[str], None]) -> str | None:
    """The sequence an attempt without a molecule can still deliver: the paths its reads join,
    else the longest unitigs of its graph. Returns what it is, or None."""
    if summary.get("molecules"):
        return "molecules"
    for path in adir.glob("*partial*.fasta"):
        if path.is_file() and _fasta_bases(path):
            return "partial_paths"
    for name in ("assembly.filtered.gfa", "assembly.gfa"):
        gfa = adir / name
        if gfa.is_file():
            written = _graph_unitigs(gfa, adir / "unitigs.fasta")
            if written:
                log(
                    f"OVASM: no molecule and no partial path: delivering the {written} longest unitigs"
                )
                return "graph_unitigs"
    return None


def _suggestions(
    records: list[dict[str, Any]], status: str, organelle: str, weak: list[str]
) -> list[str]:
    text = " ".join(explain_attempt(r) for r in records)
    out: list[str] = []
    if "too distant" in text or "recruited no" in text:
        out.append(
            "The built-in seeds recruit little here: give a close relative's organelle genome as "
            "a custom seed (seed source: custom), or use seed-free discovery."
        )
    if "no depth cluster" in text or "too shallow" in text:
        out.append(
            "The organelle did not stand out by depth: add reads or give an organelle-enriched "
            "read set (read set: target reads)."
        )
    if "not to be trusted" in text or "no molecule" in text:
        out.append(
            "The read graph did not resolve into molecules: more or longer reads help; "
            "assembly.gfa and linearization.json show where it breaks."
        )
    if "is circular" in text:
        out.append(
            "No closed molecule: the genome may be incomplete or multi-chromosomal; check "
            "linearization.json and, if it should close, add reads."
        )
    if "under 50%" in text:
        out.append(
            "The assembly is much shorter than the closest reference: compare with the related "
            "genome (identity.json) before relying on it as the complete genome."
        )
    if weak:
        out.append(
            "A depth cluster was taken on weaker evidence than the usual call "
            f"({', '.join(weak)}): check identity.json and annotate the result."
        )
    if status == "failed":
        out.append(
            f"Check that the library is whole-genome sequencing of the sample and holds the {organelle}; "
            "the reads recruited nothing that assembles."
        )
    return out


def _failure_report(
    output_dir: Path,
    organelle: str,
    params: dict[str, Any],
    records: list[dict[str, Any]],
    message: str,
) -> None:
    report = {
        "backend": "ovasm",
        "organelle": organelle,
        "status": "failed",
        "chosen_attempt": None,
        "attempts": records,
        "reasons": [f"{r['name']}: {explain_attempt(r)}" for r in records],
        "suggestions": _suggestions(records, "failed", organelle, []),
        "error": message,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


def _rewrite(files: dict[str, str], src_root: Path, dst_root: Path) -> dict[str, str]:
    """Hard link the published files of an attempt into the output directory."""
    out: dict[str, str] = {}
    for role, value in files.items():
        path = Path(value)
        try:
            relative = path.relative_to(src_root)
        except ValueError:
            out[role] = value
            continue
        if path.is_file():
            _link(path, dst_root / relative)
        out[role] = str(dst_root / relative)
    return out


def _read_names(path: Path) -> set[bytes]:
    """The read names of a FASTQ file (first token of each header)."""
    names: set[bytes] = set()
    with path.open("rb") as handle:
        for n, line in enumerate(handle):
            if n % 4 == 0:
                names.add(line.split(None, 1)[0] if line.strip() else line)
    return names


def pool_check_applies(params: dict[str, Any], organelle: str) -> bool:
    """The seed-pool check is for a mitochondrion recruited with seeds and no discovery."""
    return (
        organelle == "mitochondrion"
        and str(params.get("seed_source") or ("custom" if params.get("seed") else "builtin"))
        != "discover"
        and not params.get("also_discover")
    )


def survey_pool(
    ovasm_runner: Any,
    params: dict[str, Any],
    organelle: str,
    reads: Path,
    output_dir: Path,
    attempt_dir: Path,
    seed_reads: int | None,
    log: Callable[[str], None],
) -> dict[str, Any]:
    """Compare the seed pool of the requested run with the discovery pool.

    Discovery and identification run once on the whole read set, into `output_dir/discovery`
    (the later attempts reuse them). The discovery pool is the seed-recruited reads plus the
    depth clusters called as the target, the relaxed call included, each read once.
    """
    survey = output_dir / "discovery"
    called = ovasm_runner._called_clusters(
        reads, organelle, survey, int(params.get("threads") or 4), {}, log
    )
    chosen = choose_target_clusters(called, organelle, relaxed=True)
    cluster_names: set[bytes] = set()
    for cluster in chosen:
        cluster_names |= _read_names(Path(cluster["output"]))
    seed_file = attempt_dir / "recruit" / f"{organelle}.fastq"
    seed_names = _read_names(seed_file) if seed_file.is_file() else None
    if seed_names is not None:
        seed_pool = len(seed_names)
        union = len(seed_names | cluster_names)
    else:
        seed_pool = int(seed_reads or 0)
        union = seed_pool + len(cluster_names)
    ratio = union / seed_pool if seed_pool else float("inf")
    pool = {
        "seed_pool_reads": seed_pool,
        "discovery_reads": len(cluster_names),
        "union_reads": union,
        "pool_ratio": None if ratio == float("inf") else round(ratio, 3),
        "threshold": POOL_RATIO_MIN,
        "triggered": ratio >= POOL_RATIO_MIN,
        "clusters": [c["name"] for c in chosen],
        "weak_call": [c["name"] for c in chosen if c.get("weak_call")],
        "directory": (survey.relative_to(output_dir) / "discover").as_posix(),
    }
    if pool["triggered"]:
        pool["reason"] = (
            f"the seed pool is {seed_pool} reads but the depth clusters called as the "
            f"{organelle} bring it to {union} ({pool['pool_ratio']}x; at least {POOL_RATIO_MIN}x "
            "means the seeds recruited an incomplete set): tried the seeds plus discovery"
        )
    return pool


def run_ovasm_resilient(
    params: dict[str, Any],
    *,
    organelle: str,
    reads: Path,
    output_dir: Path,
    log: Callable[[str], None],
    mate: Path | None = None,
    coverage_tools: Any = None,
    read_origins: Path | None = None,
    original_read_artifact_sha256: str | None = None,
) -> dict[str, Any]:
    """`run_ovasm` with the escalation described in the module docstring."""
    from . import ovasm_pipeline as ovasm_runner

    run_context = {key: value for key, value in {
        "coverage_tools": coverage_tools, "read_origins": read_origins,
        "original_read_artifact_sha256": original_read_artifact_sha256,
    }.items() if value is not None}
    source = _source_label(params)
    if not escalates(params, mate):
        return _single(params, organelle, reads, output_dir, log, mate, source, ovasm_runner, run_context)

    records: list[dict[str, Any]] = []
    results: dict[str, dict[str, Any]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    reuse: Path | None = None
    pool: dict[str, Any] | None = None
    steps = plan(params)
    for index, (name, overrides) in enumerate(steps):
        adir = output_dir / "attempts" / name
        attempt_params = {**params, **overrides, "relaxed_call": True, "k_search": True}
        if reuse is not None:
            attempt_params["reuse_discovery"] = str(reuse)
        t0 = time.monotonic()
        summary = error = None
        try:
            results[name] = ovasm_runner.run_ovasm(
                attempt_params,
                organelle=organelle,
                reads=reads,
                output_dir=adir,
                log=lambda m, name=name: log(f"[attempt {name}] {m}"),
                **run_context,
            )
            summary = _read_json(adir / "summary.json")
        except Exception as exc:
            if not adir.exists():
                # refused before anything started: a bad request or a missing binary, not an
                # assembly that went wrong; there is nothing to escalate
                raise
            error = exc
            log(f"[attempt {name}] failed: {type(exc).__name__}: {str(exc)[:300]}")
        label = source if name == "requested" else "the seeds"
        record = _record(
            name,
            overrides,
            adir,
            output_dir,
            organelle,
            time.monotonic() - t0,
            summary,
            error,
            label,
        )
        records.append(record)
        if summary is not None:
            summaries[name] = summary
        if (adir / "discover" / "identify.json").is_file() and (
            adir / "discover" / "discover.json"
        ).is_file():
            reuse = adir / "discover"
        if name == "requested" and summary is not None and pool_check_applies(params, organelle):
            try:
                pool = survey_pool(
                    ovasm_runner,
                    params,
                    organelle,
                    reads,
                    output_dir,
                    adir,
                    summary.get("recruited_reads"),
                    lambda m: log(f"[pool check] {m}"),
                )
            except Exception as exc:
                pool = {"error": f"{type(exc).__name__}: {str(exc)[:300]}", "triggered": False}
                log(f"OVASM: the seed-pool check failed ({pool['error']}); going on without it")
            else:
                reuse = output_dir / pool["directory"]
                log(
                    f"OVASM: seed pool {pool['seed_pool_reads']} reads, discovery pool "
                    f"{pool['union_reads']} (x{pool['pool_ratio']}, threshold {POOL_RATIO_MIN})"
                )
            if pool.get("triggered"):
                record["pool_trigger"] = pool["reason"]
        if record["outcome"] == "accepted" and not record.get("pool_trigger"):
            break
        later = steps[index + 1][0] if index + 1 < len(steps) else None
        log(
            f"OVASM: attempt '{name}' "
            + (
                "accepted, but " + record["pool_trigger"]
                if record["outcome"] == "accepted"
                else f"not accepted: {explain_attempt(record)}"
            )
            + (f"; escalating to '{later}'" if later else "")
        )

    ranked = max(range(len(records)), key=lambda i: (score(records[i], organelle), -i))
    chosen = records[ranked]
    chosen_dir = output_dir / chosen["directory"]
    delivered = None
    if chosen["name"] in summaries:
        delivered = _deliver_partial(chosen_dir, summaries[chosen["name"]], log)
    if delivered is None:
        # the best-ranked attempt has nothing; another may have a partial path or a graph
        for i, rec in enumerate(records):
            if rec["name"] in summaries:
                delivered = _deliver_partial(
                    output_dir / rec["directory"], summaries[rec["name"]], log
                )
                if delivered is not None:
                    ranked, chosen, chosen_dir = i, rec, output_dir / rec["directory"]
                    break
    reasons = [
        f"{r['name']}: "
        + (
            f"accepted, but {r['pool_trigger']}"
            if r["outcome"] == "accepted"
            else explain_attempt(r)
        )
        for r in records
        if r["outcome"] != "accepted" or r.get("pool_trigger")
    ]
    if delivered is None:
        # the runner's own messages, verbatim and in order (the plain-language reasons are in
        # summary.json): the first one is what the run as requested ran into
        message = (
            f"OVASM could not assemble the {organelle} in {len(records)} attempt(s): "
            + " | ".join(
                f"{r['name']}: {r.get('raw_error') or explain_attempt(r)}" for r in records
            )
            + " (see attempts/<name>/ in the output directory and summary.json)"
        )
        _failure_report(output_dir, organelle, params, records, message)
        log(f"OVASM: {message}")
        raise RuntimeError(message)

    summary = dict(summaries[chosen["name"]])
    result = results[chosen["name"]]
    files = _rewrite(result["output_files"], chosen_dir, output_dir)
    assembly_json = chosen_dir / "assembly.json"
    if assembly_json.is_file():
        _link(assembly_json, output_dir / "assembly.json")
    if delivered == "partial_paths":
        partial = next(p for p in chosen_dir.glob("*partial*.fasta") if p.is_file())
        _link(partial, output_dir / partial.name)
        files["assembly_fasta"] = str(output_dir / partial.name)
        files["partial_fasta"] = str(output_dir / partial.name)
    elif delivered == "graph_unitigs":
        _link(chosen_dir / "unitigs.fasta", output_dir / "unitigs.fasta")
        files["assembly_fasta"] = str(output_dir / "unitigs.fasta")
    status = "complete" if chosen["outcome"] == "accepted" else "partial"
    weak = chosen.get("weak_cluster_call") or []
    suggestions = _suggestions(records, status, organelle, weak)
    summary.update(
        status=status,
        delivered=delivered,
        chosen_attempt=chosen["name"],
        attempts=records,
        reasons=reasons,
        suggestions=suggestions,
        concerns=[*chosen["concerns"], *chosen["notes"]],
        notes=chosen["notes"],
        problems=chosen["problems"],
        weak_cluster_call=weak,
        seed_pool_reads=(pool or {}).get("seed_pool_reads"),
        discovery_reads=(pool or {}).get("discovery_reads"),
        pool_ratio=(pool or {}).get("pool_ratio"),
        pool=pool,
        escalation={
            "attempts": [name for name, _ in steps],
            "accepted_when": "at least one molecule; some k explained >= 50% of the graph; the "
            "identity check raises no warning; a plastid has a circular molecule; and no concern "
            "(graph components without a representative sequence, molecules under 50% of the "
            "closest reference genome); a mitochondrion also needs a discovery pool under "
            f"{POOL_RATIO_MIN}x its seed pool, or the seeds plus discovery are tried too",
            "ranked_by": [
                "no problems",
                "trusted molecules",
                "no concerns",
                "a circular molecule, then molecule bases (a mitochondrion: bases, then circle)",
                "partial-path bases",
                "graph bases",
            ],
        },
    )
    top = output_dir / "summary.json"
    top.unlink(missing_ok=True)  # a hard link to the attempt's own summary: never write through it
    top.write_text(json.dumps(summary, indent=2) + "\n")
    files["summary"] = str(top)
    text = result["summary_text"]
    text += f"; STATUS {status.upper()}: attempt '{chosen['name']}' of {len(records)}"
    if delivered != "molecules":
        text += f"; delivered {delivered.replace('_', ' ')} (not molecules)"
    if reasons:
        text += "; earlier attempts: " + " | ".join(reasons)
    if weak:
        text += f"; WEAK depth-cluster call: {', '.join(weak)}"
    log(
        f"OVASM: attempt '{chosen['name']}' chosen ({status}); "
        + (
            "accepted"
            if chosen["outcome"] == "accepted"
            else "best of the attempts, not accepted: " + explain_attempt(chosen)
        )
    )
    return {"summary_text": text, "output_files": files}


def _single(
    params: dict[str, Any],
    organelle: str,
    reads: Path,
    output_dir: Path,
    log: Callable[[str], None],
    mate: Path | None,
    source: str,
    ovasm_runner: Any,
    run_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One run as before (no escalation), with the structured report."""
    t0 = time.monotonic()
    try:
        more = {**(run_context or {}), **({} if mate is None else {"mate": mate})}
        result = ovasm_runner.run_ovasm(
            params, organelle=organelle, reads=reads, output_dir=output_dir, log=log, **more
        )
    except Exception as exc:
        if output_dir.is_dir():
            record = _record(
                "requested",
                {},
                output_dir,
                output_dir,
                organelle,
                time.monotonic() - t0,
                None,
                exc,
                source,
            )
            _failure_report(output_dir, organelle, params, [record], f"{type(exc).__name__}: {exc}")
        raise
    summary = _read_json(output_dir / "summary.json")
    if summary is None:
        return result
    record = _record(
        "requested",
        {},
        output_dir,
        output_dir,
        organelle,
        time.monotonic() - t0,
        summary,
        None,
        source,
    )
    # a single run delivers what the runner always delivered (molecules; the partial paths are
    # listed as `partial_fasta`): only the report is new
    delivered = (
        "molecules"
        if summary.get("molecules")
        else "partial_paths"
        if summary.get("partial_paths")
        else None
    )
    status = "complete" if record["outcome"] == "accepted" else "partial"
    if delivered is None:
        status = "failed"
    reasons = [f"requested: {explain_attempt(record)}"] if record["outcome"] != "accepted" else []
    summary.update(
        status=status,
        delivered=delivered,
        chosen_attempt="requested",
        attempts=[record],
        reasons=reasons,
        suggestions=_suggestions(
            [record], status, organelle, record.get("weak_cluster_call") or []
        ),
        concerns=[*record["concerns"], *record["notes"]],
        notes=record["notes"],
        problems=record["problems"],
        weak_cluster_call=record.get("weak_cluster_call") or [],
        escalation=None,
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    text = result["summary_text"] + f"; STATUS {status.upper()}"
    if reasons:
        text += ": " + "; ".join(reasons)
    return {"summary_text": text, "output_files": dict(result["output_files"])}
