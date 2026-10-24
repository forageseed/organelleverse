"""Native OVASM execution for the desktop assembly tools."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from organelleverse import _ovasm
from organelleverse.core.errors import OrganelleExecutionError
from organelleverse.assembly.ovasm_fallback import MAX_EXTRA_K, choose_target_clusters, k_curable

#: Fewest recruited bases that can hold an organelle genome at all (the smallest are tens of
#: kb). Luzula sylvatica's mitochondrion (v3 holdout) recruited 4 reads, 4 kb, from distant
#: seeds and failed in unringing with "every segment is a repeat"; ordinary runs recruit
#: millions of bases.
MIN_RECRUITED_BASES = 100_000


# the binary's automatic-k ladder (`K_LADDER` in rust/ovasm/src/assemble.rs)
K_LADDER = (1001, 701, 501, 351, 251, 201, 151, 101)

#: The molecules of an attempt must explain at least this share of the assembly graph for the
#: attempt to count (`MIN_EXPLAINED` in rust/ovasm/src/pipeline.rs, which `ovasm run` uses).
MIN_EXPLAINED = 0.5

#: Read types (the `strategy` field of the assembly tools). The noisy ones are recruited with
#: the `ont` preset in one pass and corrected (`ovasm correct`) before assembly; `hybrid` is
#: noisy long reads plus Illumina reads of the same sample, whose k-mers correct them.
READ_TYPES = ("hifi_only", "short_read", "ont_only", "clr_only", "hybrid")
NOISY = ("ont_only", "clr_only", "hybrid")

#: First k for corrected noisy reads, as validated: Nipponbare ONT and Salvia CLR
#: (docs/benchmarks/ovasm-nipponbare.md, ovasm-salvia.md) and gold mode all assemble them at
#: 501; the automatic k looks at depth only and was never run on corrected reads.
NOISY_K = 501

#: `ovasm correct` rounds, as in gold mode: the reads' own k-mers, small k first; with short
#: reads their k-mers at larger k (the binary then adds self rounds for stretches the short
#: reads miss).
SELF_KS = (15, 21, 25)
HYBRID_KS = (21, 25, 31)

#: `--numt-min-block` that no read reaches: noisy recruitment writes no NUMT candidates. At
#: k = 15 the multi-species seed database hits nuclear reads by chance every ~130 bp, and
#: "partially organelle-like" stops meaning anything: 997,070 of 2.3 M Salvia CLR reads were
#: flagged, a 21 GB file next to 0.5 GB of recruited reads. The runner never used that file.
NO_NUMT_BLOCK = 1_000_000_000


def _component_filter_summary(components: dict[str, Any]) -> dict[str, Any]:
    """Report retention separately from marker evidence and physical origin.

    Depth alone can retain both gene-free organelle components and nuclear repeats.
    Target panel hits are sequence evidence, including in transferred copies; they
    do not identify the physical compartment in which the sequenced DNA occurred.
    Lengths here count graph segments, before path overlaps are removed.
    """
    retained = [
        {
            "id": c["id"],
            "segments": c["segments"],
            "length": c["length"],
            "depth": c["depth"],
            "target_genes": c["target_genes"],
            "other_genes": c["other_genes"],
            "reason": c["reason"],
            **({"read_origin_evidence": c["read_origin_evidence"]} if "read_origin_evidence" in c else {}),
            **({"read_origin_status": c["read_origin_status"]} if "read_origin_status" in c else {}),
            **({"target_assignment": c["target_assignment"]} if "target_assignment" in c else {}),
            "target_marker_detected": bool(c["target_genes"]),
            "physical_origin_identified": False,
        }
        for c in components["components"]
        if c["kept"]
    ]
    without_markers = [c for c in retained if not c["target_marker_detected"]]
    return {
        **({"read_origin_input": components["read_origin_input"]} if "read_origin_input" in components else {}),
        "reference_depth": components["reference_depth"],
        "kept_components": components["kept_components"],
        "kept_length": components["kept_length"],
        "removed_components": components["removed_components"],
        "removed_length": components["removed_length"],
        "length_scope": "graph_segments_before_path_overlap_removal",
        "retained_without_target_markers": len(without_markers),
        "retained_without_target_markers_length": sum(c["length"] for c in without_markers),
        "retained": retained,
        "removed": [
            {
                "segments": len(c["segments"]),
                "length": c["length"],
                "depth": c["depth"],
                "target_genes": len(c["target_genes"]),
                "other_genes": len(c["other_genes"]),
                "reason": c["reason"],
            }
            for c in components["components"]
            if not c["kept"]
        ],
    }


def _require_enough(recruitment: dict[str, Any], organelle: str, seed_info: dict[str, Any]) -> None:
    """Stop before assembly when recruitment left too little to assemble anything."""
    target = recruitment["outputs"][_target_index(recruitment, organelle)]
    if target["kept_bases"] >= MIN_RECRUITED_BASES:
        return
    source = seed_info.get("name") or seed_info.get("file") or seed_info["source"]
    raise RuntimeError(
        f"OVASM recruited {target['kept_reads']} reads ({target['kept_bases']} bp) "
        f"for the {organelle} from {source} {seed_info.get('version', '')}: "
        "too little to assemble anything. The seeds are probably too distant from "
        "this species; supply a closer reference seed, use seed-free discovery, or give "
        "the target reads themselves."
    )


def _called_clusters(
    reads: Path,
    organelle: str,
    output_dir: Path,
    threads: int,
    files: dict[str, str],
    log: Callable[[str], None],
    reuse: Path | None = None,
) -> list[dict[str, Any]]:
    """Depth clusters of the reads, each called by its conserved organelle proteins.

    `reuse`: the `discover` directory of an earlier attempt on the same reads (discovery and
    identification are the costly part and give the same clusters again).
    """
    target = output_dir / "discover"
    if (
        reuse is not None
        and (reuse / "discover.json").is_file()
        and (reuse / "identify.json").is_file()
    ):
        log(f"OVASM: reusing the depth clusters found in {reuse}")
        discovery = json.loads((reuse / "discover.json").read_text())
        identification = json.loads((reuse / "identify.json").read_text())
        target.mkdir(parents=True, exist_ok=True)
        for name in ("discover.json", "identify.json"):
            (target / name).write_text((reuse / name).read_text())
    else:
        log(f"OVASM: finding {organelle} reads by depth, without seeds")
        discovery = _ovasm.run_discover([reads], out_dir=target, threads=threads)
        log("OVASM: telling the depth clusters apart by their conserved organelle proteins")
        identification = _ovasm.run_panel(
            [c["output"] for c in discovery["clusters"]],
            out_json=target / "identify.json",
            threads=threads,
        )
    files["discovery"] = str(target / "discover.json")
    files["identification"] = str(target / "identify.json")
    return [
        {
            "name": c["name"],
            "reads": c["kept_reads"],
            "bases": s["bases"],
            "estimated_depth": c["estimated_depth"],
            "call": s["call"],
            # the identify statistics the call (and the relaxed call) rests on
            "stats": {
                org: {
                    "genes": s.get(org, {}).get("genes", 0),
                    "segments": s.get(org, {}).get("segments", 0),
                    "density": s.get(f"{org}_density", 0.0),
                }
                for org in ("mitochondrion", "plastid")
            },
            "output": c["output"],
        }
        for c, s in zip(discovery["clusters"], identification["sets"], strict=True)
    ]


def _pool_fastq(sources: list[str | Path], dest: Path) -> tuple[int, int]:
    """Concatenate FASTQ files into `dest`, a read once (by its name); returns reads, bases."""
    seen: set[bytes] = set()
    reads = bases = 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as out:
        for source in sources:
            if not Path(source).is_file():
                continue  # seeds that recruited nothing (also_discover) left no file
            with open(source, "rb") as src:
                while True:
                    record = [src.readline() for _ in range(4)]
                    if not record[0]:
                        break
                    name = record[0].split(None, 1)[0]
                    if name in seen:
                        continue
                    seen.add(name)
                    out.writelines(record)
                    reads += 1
                    bases += len(record[1].rstrip(b"\r\n"))
    return reads, bases


def _summary(called: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: v for k, v in c.items() if k != "output"} for c in called]


def _discover_reads(
    reads: Path,
    organelle: str,
    output_dir: Path,
    threads: int,
    files: dict[str, str],
    log: Callable[[str], None],
    *,
    relaxed: bool = False,
    reuse: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Organelle reads without seeds: the clusters called as the target, pooled."""
    called = _called_clusters(reads, organelle, output_dir, threads, files, log, reuse)
    chosen = choose_target_clusters(called, organelle, relaxed=relaxed)
    if chosen and not any(c["call"] == organelle for c in chosen):
        log(
            f"OVASM: no depth cluster was called {organelle}; taking "
            f"{[c['name'] for c in chosen]} where its genes clearly lead (weak call)"
        )
    if not chosen:
        raise RuntimeError(
            f"OVASM found no depth cluster whose reads encode the conserved {organelle} "
            f"proteins (clusters: {[(c['name'], c['call']) for c in called]}); the organelle "
            "may be too shallow to stand out from the nuclear genome, or its reads may share a "
            "depth cluster with the other organelle's; try seeds or give the target reads"
        )
    pooled = output_dir / "recruit" / f"{organelle}.fastq"
    n_reads, n_bases = _pool_fastq([c["output"] for c in chosen], pooled)
    log(f"OVASM: {len(chosen)} cluster(s) hold the {organelle}: {n_reads} reads")
    recruitment = {
        "targets": [{"name": organelle}],
        "outputs": [{"kept_reads": n_reads, "kept_bases": n_bases}],
        "clusters": _summary(called),
    }
    return pooled, recruitment


def _add_discovered(
    whole_genome: Path,
    seed_reads: Path,
    seed_n: int,
    organelle: str,
    output_dir: Path,
    threads: int,
    files: dict[str, str],
    log: Callable[[str], None],
    *,
    relaxed: bool = False,
    reuse: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Seed-recruited reads plus the depth clusters called as the target organelle.

    Seeds give a clean core where a relative is in the database; discovery reaches what seeds
    cannot (v3 holdout, Luzula plastid: seeds took 6,225 reads and missed a 57 kb stretch
    the library holds at 400x; its cluster has them all). A cluster called "mixed" holds both
    organelles' reads and is left to the seeds.
    """
    called = _called_clusters(whole_genome, organelle, output_dir, threads, files, log, reuse)
    chosen = choose_target_clusters(called, organelle, relaxed=relaxed)
    if chosen and not any(c["call"] == organelle for c in chosen):
        log(
            f"OVASM: no depth cluster was called {organelle}; taking "
            f"{[c['name'] for c in chosen]} where its genes clearly lead (weak call)"
        )
    union = output_dir / "recruit" / f"{organelle}.union.fastq"
    n_reads, n_bases = _pool_fastq([seed_reads, *[c["output"] for c in chosen]], union)
    log(
        f"OVASM: seeds {seed_n} reads + {len(chosen)} discovered cluster(s) "
        f"-> {n_reads} reads for the {organelle}"
    )
    recruitment = {
        "targets": [{"name": organelle}],
        "outputs": [{"kept_reads": n_reads, "kept_bases": n_bases}],
        "discovery": {
            "clusters": _summary(called),
            "seed_reads": seed_n,
            "union_reads": n_reads,
        },
    }
    return union, recruitment


def _keep_target_components(
    graph: Path,
    organelle: str,
    output_dir: Path,
    files: dict[str, str],
    log: Callable[[str], None],
    *,
    reject_other_only: bool = False,
    read_origins: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    """The assembly graph without the components that are not the target organelle.

    Reads pooled from depth clusters bring whatever shares the organelle's cluster (v3
    holdout, Climacium mitochondrion: one 104,863 bp circle at depth 80 beside 27 components
    of nuclear repeats, 1.3 Mb at depth 7-36, gave 41 molecules). `assembly.gfa` stays as
    assembled; `components.json` lists every component with its length, depth, genes and why
    it was kept or removed. The rule's thresholds were set on seen cases (see
    rust/ovasm/src/components.rs).
    """
    log(f"OVASM: keeping the graph components that are the {organelle}")
    kept = output_dir / "assembly.filtered.gfa"
    report = _ovasm.run_components(
        graph, organelle=organelle, out_gfa=kept, out_json=output_dir / "components.json",
        **({"reject_other_only": True} if reject_other_only else {}),
        **({"read_origins": read_origins} if read_origins is not None else {}),
        **({"out_assigned_gfa": output_dir / "assembly.assigned.gfa",
            "out_unassigned_gfa": output_dir / "assembly.unassigned.gfa"}
           if read_origins is not None else {}),
    )
    files["filtered_assembly_graph"] = str(kept)
    files["component_filter"] = str(output_dir / "components.json")
    if report["removed_components"]:
        log(
            f"OVASM: removed {report['removed_components']} of {len(report['components'])} "
            f"graph components ({report['removed_length']} bp) that are not the {organelle}; "
            "see components.json"
        )
    return kept, report


def _target_index(recruitment: dict[str, Any], organelle: str) -> int:
    return [t["name"] for t in recruitment["targets"]].index(organelle)


def _recruited(recruitment: dict[str, Any] | None, organelle: str) -> int | None:
    if recruitment is None:
        return None
    return recruitment["outputs"][_target_index(recruitment, organelle)]["kept_reads"]


#: Least containment in the organelle's own references for an organelle genome. Correct
#: assemblies of the holdouts ranged 0.014 (Carex mitochondrion vs Zea) to 0.61; a runaway of
#: nuclear sequence (Jasione, 4.3 Mb) shared 0.0002.
ORGANELLE_MIN_CONTAINMENT = 0.005


def _family(taxonomy: list[str]) -> str | None:
    return next((t for t in reversed(taxonomy) if t.endswith("aceae")), None)


def _identify(molecules: Path, organelle: str, out_json: Path) -> tuple[dict[str, Any], list[str]]:
    """Closest builtin SeedDB references of the assembly, and warnings about its target."""
    from organelleverse._ovasm_seeddb import bundled_seed

    paths, info = {}, {}
    for org in ("mitochondrion", "plastid"):
        paths[org], seed = bundled_seed(org)
        manifest = json.loads(Path(seed["manifest"]).read_text(encoding="utf-8"))
        for ref in manifest["databases"][org]["references"]:
            info[ref["accession"]] = ref
    report = _ovasm.run_identify(molecules, references=paths, out_json=out_json)
    best = {}
    for org in ("mitochondrion", "plastid"):
        hit = next(h for h in report["references"] if h["set"] == org)
        ref = info[hit["id"]]
        best[org] = {
            "accession": hit["id"],
            "organism": ref["organism"],
            "family": _family(ref["taxonomy"]),
            "containment": hit["containment"],
            "estimated_identity": hit["estimated_identity"],
        }
    other = "plastid" if organelle == "mitochondrion" else "mitochondrion"
    resembles = max(best, key=lambda org: best[org]["containment"])
    warnings = []
    if resembles != organelle:
        warnings.append(
            f"the assembled {organelle} resembles the {other} references more than the "
            f"{organelle} references (containment {best[other]['containment']:.3f} vs "
            f"{best[organelle]['containment']:.3f}): the recruited reads are likely {other}"
        )
    if best[organelle]["containment"] < ORGANELLE_MIN_CONTAINMENT:
        warnings.append(
            f"the assembly shares almost no k-mers with any {organelle} reference "
            f"(containment {best[organelle]['containment']:.4f}): it is probably not an "
            "organelle genome"
        )
    return {
        "method": "k-mer containment (k=21) against the builtin SeedDB; identity = C^(1/k)",
        "closest": best[organelle],
        "best_by_database": best,
        "resembles": resembles,
        "interpretation": "a near-identical match to a reference of another species can mean "
        "contamination or a mislabelled sample; compare with the species sequenced",
    }, warnings


#: Depth each discover cluster is downsampled to before it is assembled for labelling.
LABEL_CLUSTER_DEPTH = 200.0


def _sample_seeds(
    reads: Path,
    output_dir: Path,
    threads: int,
    seeds: dict[str, list[Path]],
    log: Callable[[str], None],
) -> dict[str, Any]:
    """Graph-first target assignment: add the sample's own labelled unitigs to ``seeds``.

    The builtin seeds come from other species; where they are distant, the wrong reads or none
    are recruited (v1 holdout: plastid reads as the Ajuga mitochondrion; v2: 37 of 1,916 Ajuga
    mitochondrial reads). ``discover`` finds the organelle-like reads without seeds, ``label``
    assembles each depth cluster and keeps the unitigs whose genes say mitochondrion or plastid.
    """
    from organelleverse._ovasm_seeddb import bundled_gene_db

    log("OVASM: finding organelle-like reads by depth, without seeds")
    discovered = _ovasm.run_discover(
        [reads],
        out_dir=output_dir / "discover",
        target_depth=LABEL_CLUSTER_DEPTH,
        threads=threads,
    )
    clusters = [c["output"] for c in discovered["clusters"] if c["kept_reads"]]
    genes, gene_info = bundled_gene_db()
    info: dict[str, Any] = {
        "gene_database": gene_info,
        "organelle_like_reads": discovered["organelle_like_reads"],
        "clusters": len(clusters),
        "sample_seed_bp": {"mitochondrion": 0, "plastid": 0},
    }
    if not clusters:
        log("OVASM: no organelle-like read clusters; recruiting with the builtin seeds only")
        return info
    log("OVASM: assembling each depth cluster and labelling unitigs by organelle genes")
    labels = _ovasm.run_label(clusters, genes=genes, out_dir=output_dir / "labels", threads=threads)
    for org, path, bp in zip(
        ("mitochondrion", "plastid"), labels["seeds"], labels["seed_bp"], strict=True
    ):
        info["sample_seed_bp"][org] = bp
        if path:
            seeds[org].insert(0, Path(path))
    return info


def _short_read_files(params: dict[str, Any], strategy: str) -> list[Path]:
    """The Illumina file(s) of a hybrid run (`short_reads`, and the mate in `short_reads_2`)."""
    given = [str(params.get(name) or "").strip() for name in ("short_reads", "short_reads_2")]
    if strategy != "hybrid":
        if any(given):
            raise ValueError(
                "OVASM uses a short-read file only with the hybrid read type; select hybrid "
                "or remove the short-read file"
            )
        return []
    if not given[0]:
        raise ValueError(
            "OVASM hybrid correction needs a short-read file (Illumina reads of the same sample)"
        )
    files = [Path(value).expanduser() for value in given if value]
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(f"OVASM short-read file not found: {path}")
    return files


def _correct_noisy(
    reads: Path,
    *,
    organelle: str,
    short_reads: list[Path],
    recruit_short: bool,
    other_seed: Path | None,
    output_dir: Path,
    threads: int,
    files: dict[str, str],
    log: Callable[[str], None],
) -> tuple[Path, dict[str, Any]]:
    """Corrected reads of noisy organelle reads (ONT, CLR), and what was done to them.

    Without short reads: self-correction against the reads' own solid k-mers. With short
    reads (hybrid), as in gold mode: the self-corrected long reads bait the sample's own
    short reads (a reference seed left a 1 kb stretch of the Nipponbare mitochondrion without
    short reads; the sample's own reads cover it), then the short reads' k-mers correct the
    long reads. `recruit_short` is off for target reads: the short reads are then taken as
    given, like the long ones.
    """
    out = output_dir / "correct"
    out.mkdir(parents=True, exist_ok=True)
    done: dict[str, Any] = {"mode": "hybrid" if short_reads else "self"}
    corrected = out / f"{organelle}.corrected.fasta"
    if not short_reads:
        log(f"OVASM: correcting the noisy reads against their own k-mers (k = {SELF_KS})")
        report = _ovasm.run_correct(
            [reads], out_fasta=corrected, out_json=out / "correct.json", ks=SELF_KS, threads=threads
        )
    else:
        if recruit_short:
            log("OVASM: self-correcting the long reads to bait the sample's short reads")
            bait = out / f"{organelle}.self.fasta"
            _ovasm.run_correct(
                [reads], out_fasta=bait, out_json=out / "self.json", ks=SELF_KS, threads=threads
            )
            files["self_correction"] = str(out / "self.json")
            log(f"OVASM: recruiting the {organelle} short reads with the corrected long reads")
            recruited = _ovasm.run_recruit(
                short_reads,
                # the other organelle keeps its reference seed, so that its far deeper reads
                # are claimed by their own target (see the long-read recruitment)
                seeds={} if other_seed is None else {_other(organelle): [other_seed]},
                seed_reads={organelle: [bait]},
                out_dir=output_dir / "recruit_short",
                preset="sr",
                threads=threads,
            )
            files["short_read_recruitment"] = str(output_dir / "recruit_short/recruit.json")
            target = recruited["outputs"][_target_index(recruited, organelle)]
            done["short_reads_recruited"] = target["kept_reads"]
            if not target["kept_reads"]:
                raise RuntimeError(
                    f"OVASM recruited no {organelle} short reads with the sample's corrected "
                    "long reads; are the two files from the same sample? Use the ONT or CLR "
                    "read type to correct the long reads against themselves"
                )
            short_reads = [output_dir / "recruit_short" / f"{organelle}.fastq"]
        log(f"OVASM: correcting the long reads with the short reads' k-mers (k = {HYBRID_KS})")
        report = _ovasm.run_correct(
            [reads],
            out_fasta=corrected,
            out_json=out / "correct.json",
            ks=HYBRID_KS,
            short_reads=short_reads,
            threads=threads,
        )
    files["correction"] = str(out / "correct.json")
    rounds = report["rounds"]
    done["rounds"] = [{"k": r["k"], "hybrid": r["hybrid"]} for r in rounds]
    done["reads_in"] = rounds[0]["reads_in"]
    done["reads_out"] = rounds[-1]["reads_out"]
    if not done["reads_out"]:
        raise RuntimeError(
            f"OVASM correction kept none of the {done['reads_in']} noisy reads: too shallow "
            "for their own k-mers to be told from errors; add reads or use hybrid correction"
        )
    log(f"OVASM: {done['reads_out']} of {done['reads_in']} reads corrected")
    return corrected, done


def _other(organelle: str) -> str:
    return "plastid" if organelle == "mitochondrion" else "mitochondrion"


def _preserve_closed_cover(summary: dict[str, Any], linear: dict[str, Any], graph: Path) -> bool:
    """A resolved circular cover must account for every physical graph segment.

    Graph acceptance alone is insufficient: truncated searches and linear paths can
    be accepted while their chromosome configuration is still unresolved.
    """
    if not summary.get("k_accepted") or not summary.get("decisive"):
        return False
    if any(summary.get(key) for key in (
        "unsolved_components", "partial_paths", "search_truncated", "cover_size_limited",
        "unbridged_anchors", "skipped_anchors",
    )) or linear.get("unplaced_segments") or linear.get("repeats_not_placed"):
        return False
    circular = summary.get("circular", [])
    if not circular or not all(circular):
        return False
    segments = {row.split("\t")[1] for row in graph.read_text().splitlines() if row.startswith("S\t")}
    covered = {
        step[:-1]
        for component in linear.get("components", [])
        for molecule in (component.get("best") or {}).get("molecules", [])
        for step in molecule["path"].split(",")
    }
    return bool(segments) and covered == segments


def run_ovasm(
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
    strategy = str(params.get("strategy") or "hifi_only")
    graph_mode = str(params.get("graph_mode") or "sparse")
    if graph_mode not in {"sparse", "multichrom"}:
        raise ValueError("OVASM graph_mode must be sparse or multichrom")
    multichrom = graph_mode == "multichrom"
    graph_min_count = params.get("graph_min_count")
    if graph_min_count == "":
        graph_min_count = None
    automatic_coverage = graph_min_count == "auto"
    if multichrom and (strategy != "hifi_only" or graph_min_count is None
                      or (not automatic_coverage and int(graph_min_count) < 1)):
        raise ValueError("OVASM multichrom needs HiFi reads and graph_min_count from read coverage")
    if not multichrom and graph_min_count is not None:
        raise ValueError("OVASM graph_min_count is used only with graph_mode multichrom")
    if strategy not in READ_TYPES:
        raise ValueError(
            "OVASM supports HiFi, ONT, CLR, hybrid (ONT or CLR plus Illumina) or Illumina "
            "reads; select the matching read type"
        )
    k_ladder_start = str(params.get("k_ladder_start") or "auto")
    if k_ladder_start not in {"auto", "top"}:
        raise ValueError("OVASM k_ladder_start must be auto or top")
    if k_ladder_start == "top" and strategy != "hifi_only":
        raise ValueError("OVASM k_ladder_start top requires HiFi reads")
    max_molecules = int(params.get("max_molecules", 4))
    if max_molecules < 1:
        raise ValueError("OVASM max_molecules must be positive")
    short_reads = _short_read_files(params, strategy)
    if mate is not None:
        if strategy != "short_read":
            raise ValueError("OVASM takes a second read file only with the Illumina read type")
        if not mate.is_file():
            raise ValueError("OVASM requires a FASTQ file, optionally gzip compressed")
    mode = str(params.get("read_set") or "whole_genome")
    if mode not in {"whole_genome", "target_reads"}:
        raise ValueError("OVASM read set must be whole_genome or target_reads")
    if automatic_coverage and mode != "whole_genome":
        raise ValueError("Automatic nuclear coverage needs original whole-genome reads before recruitment")
    if not reads.is_file():
        raise ValueError("OVASM requires a FASTQ file, optionally gzip compressed")
    if read_origins is not None:
        if not multichrom or mode != "whole_genome":
            raise ValueError("Original-read component evidence requires whole-genome multichrom HiFi input")
        origin_document = json.loads(read_origins.read_text(encoding="utf-8"))
        if original_read_artifact_sha256 is not None:
            if origin_document.get("original_reads_sha256") != original_read_artifact_sha256:
                raise ValueError("Component read-origin evidence does not match the canonical read artifact")
        elif Path(origin_document["original_reads"]).resolve() != reads.resolve():
            raise ValueError("Component read-origin evidence belongs to a different original read input")
    threads = int(params.get("threads") or 4)
    if not 1 <= threads <= 256:
        raise ValueError("OVASM threads must be between 1 and 256")
    coverage_profile = None
    if automatic_coverage:
        from organelleverse.assembly.nuclear_coverage import fit_original_wgs

        coverage_profile = fit_original_wgs(
            reads, output_dir / "nuclear-coverage", threads=threads,
            ploidy=int(params.get("nuclear_ploidy", 2)), tools=coverage_tools, log=log,
        )
        graph_min_count = coverage_profile["graph_min_count"]
    seed = None
    seed_info = {"source": "none"}
    also_discover = bool(params.get("also_discover"))
    # set by `ovasm_fallback.run_ovasm_resilient` only (see there): the relaxed cluster call,
    # the discovery directory of an earlier attempt to reuse, and larger k after the ladder
    relaxed_call = bool(params.get("relaxed_call"))
    k_search = bool(params.get("k_search"))
    reuse_discovery = Path(params["reuse_discovery"]) if params.get("reuse_discovery") else None
    if also_discover and (strategy != "hifi_only" or mode != "whole_genome"):
        raise ValueError("OVASM discovery alongside seeds needs HiFi whole-genome reads")
    if also_discover and str(params.get("seed_source") or "") == "discover":
        raise ValueError("seed_source discover already uses discovery; drop also_discover")
    if mode == "whole_genome":
        value = str(params.get("seed") or "").strip()
        source = str(params.get("seed_source") or ("custom" if value else "builtin"))
        if source not in {"builtin", "custom", "discover"}:
            raise ValueError("OVASM seed source must be builtin, custom or discover")
        if source == "discover":
            # no seeds at all: organelle reads stand out by depth, and conserved proteins say
            # which depth cluster is which organelle (works far from any reference)
            if strategy != "hifi_only":
                raise ValueError("OVASM seed-free discovery supports HiFi reads only")
            seed_info = {"source": "discover"}
        elif source == "builtin":
            from organelleverse._ovasm_seeddb import bundled_seed

            seed, seed_info = bundled_seed(organelle)
        elif not value:
            raise ValueError("OVASM custom seeds require a seed FASTA for the selected organelle")
        else:
            seed = Path(value).expanduser()
            seed_info = {"source": "custom", "file": seed.name}
        if seed is not None and not seed.is_file():
            raise FileNotFoundError(f"OVASM seed FASTA not found: {seed}")
    if _ovasm.resolve_ovasm() is None:
        raise RuntimeError(
            "OVASM executable is unavailable; install the native binary or set ORG_VERSE_OVASM_BIN"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    # noisy reads: `ont` (k = 15, one pass) until they are corrected, `hifi` from then on
    preset = "sr" if strategy == "short_read" else "ont" if strategy in NOISY else "hifi"
    # Depth kept with a custom (single-genome) seed. Noisy reads get twice the HiFi depth, as
    # in gold mode: their correction counts k-mers across reads, and the validated noisy runs
    # had 300x (Nipponbare ONT) and 212x (Salvia CLR), never 150x.
    depth = 300.0 if strategy in NOISY else 150.0
    files: dict[str, str] = {}
    if seed_info["source"] == "builtin":
        manifest = output_dir / "seed_database.json"
        manifest.write_bytes(Path(seed_info["manifest"]).read_bytes())
        seed_info["manifest"] = manifest.name
        files["seed_database"] = str(manifest)
    recruitment = None
    whole_genome = reads
    graph_first_info = None
    if seed is not None:
        from organelleverse._ovasm_seeddb import bundled_seed

        # The other organelle is always seeded too, so that its reads are claimed by
        # their own target. Seeded alone, mitochondrial seeds (whole genomes holding
        # plastid-derived MTPT stretches) extended into the plastid genome, whose reads
        # are 10-30x deeper: the Ajuga and Alisma "mitochondria" of the frozen holdout
        # were their plastid genomes.
        other = "plastid" if organelle == "mitochondrion" else "mitochondrion"
        other_seed, other_info = bundled_seed(other)
        seed_info["competing_seed"] = {
            "organelle": other,
            "version": other_info["version"],
        }
        seeds: dict[str, list[Path]] = {organelle: [seed], other: [other_seed]}
        graph_first = preset == "hifi" and seed_info["source"] == "builtin"
        if graph_first:
            graph_first_info = _sample_seeds(reads, output_dir, threads, seeds, log)
            files["labels"] = str(output_dir / "labels/labels.json")
        log(f"OVASM: recruiting {organelle} reads with {seed_info}")
        recruitment = _ovasm.run_recruit(
            [reads] if mate is None else [reads, mate],
            seeds=seeds,
            out_dir=output_dir / "recruit",
            preset=preset,
            # A multi-species marker database's summed length is not a sample
            # genome size. Do not cap depth using that denominator.
            target_depth=None if seed_info["source"] == "builtin" else depth,
            numt_min_block=NO_NUMT_BLOCK if preset == "ont" else None,
            threads=threads,
            # With the sample's own seeds, extension runs until no bait grows (a relative
            # growth floor stopped the Buddleja mitochondrion at 3,401 of 4,539 reads),
            # learning only from organelle-deep reads (a stray seed sequence led the Carex
            # plastid 75,858 reads into the nucleus without that gate).
            saturation=0.0 if graph_first else None,
            extend_gate=graph_first,
        )
        kept = recruitment["outputs"][_target_index(recruitment, organelle)]["kept_reads"]
        if kept == 0 and not also_discover:
            raise ValueError(
                f"OVASM recruited no {organelle} reads: the library holds almost none of this "
                "organelle, or its seeds do not match it (see recruit/recruit.json)"
            )
        seed_reads = output_dir / "recruit" / f"{organelle}.fastq"
        files["recruitment"] = str(output_dir / "recruit/recruit.json")
        if also_discover:
            reads, recruitment = _add_discovered(
                whole_genome,
                seed_reads,
                _recruited(recruitment, organelle) or 0,
                organelle,
                output_dir,
                threads,
                files,
                log,
                relaxed=relaxed_call,
                reuse=reuse_discovery,
            )
            seed_info["discovery"] = recruitment.pop("discovery")
        else:
            reads = seed_reads
        _require_enough(recruitment, organelle, seed_info)
    elif seed_info["source"] == "discover":
        reads, recruitment = _discover_reads(
            reads, organelle, output_dir, threads, files, log,
            relaxed=relaxed_call, reuse=reuse_discovery,
        )  # fmt: skip
        seed_info["clusters"] = recruitment.pop("clusters")
        _require_enough(recruitment, organelle, seed_info)
    else:
        log("OVASM: using the supplied target reads; whole-genome recruitment was not run")

    correction = None
    if strategy in NOISY:
        from organelleverse._ovasm_seeddb import bundled_seed

        reads, correction = _correct_noisy(
            reads,
            organelle=organelle,
            short_reads=short_reads,
            recruit_short=mode == "whole_genome",
            other_seed=bundled_seed(_other(organelle))[0] if mode == "whole_genome" else None,
            output_dir=output_dir,
            threads=threads,
            files=files,
            log=log,
        )
        # corrected reads are assembled and scored like HiFi reads (gold mode does the same)
        preset = "hifi"

    # A seed pool cannot represent disconnected chromosomes without seed hits.
    # Multichrom WGS construction and junction evidence use the original library;
    # the ordinary preflight below still assembles its recruited target pool.
    graph_reads = whole_genome if multichrom and mode == "whole_genome" else reads
    assembly_reads = [graph_reads] if mode == "whole_genome" or mate is None else [graph_reads, mate]

    # Retain a fully supported closed cover before trying a depth-filtered graph.
    # The graph filter can otherwise remove genuine low-depth organelle sequence.
    preflight_summary: dict[str, Any] | None = None
    ordinary: dict[str, Any] | None = None
    preflight = output_dir / "sparse-preflight"

    def promote_ordinary(selection: str, failure: str | None = None) -> dict[str, Any]:
        assert ordinary is not None and preflight_summary is not None
        # Only promote files this nested run produced, preserving original recruitment.
        for source in preflight.iterdir():
            if source.is_file():
                shutil.copy2(source, output_dir / source.name)
        selected = dict(preflight_summary)
        selected.update(
            read_set=mode, seed_database=seed_info, graph_first=graph_first_info,
            recruited_reads=_recruited(recruitment, organelle),
            graph_mode="sparse", requested_graph_mode="multichrom",
            multichrom_selection=selection,
            requested_graph_min_count=int(graph_min_count),
            graph_input_context="recruited_target_reads" if mode == "whole_genome" else "supplied_target_reads",
        )
        if mode == "whole_genome":
            # This retained target-pool result cannot certify WGS completeness.
            selected["complete"] = False
        if coverage_profile is not None:
            selected["nuclear_coverage"] = coverage_profile
            selected["graph_min_count_auto"] = True
            files["nuclear_coverage"] = str(output_dir / "nuclear-coverage/coverage-profile.json")
        from organelleverse.assembly.configuration import infer_configurations

        promoted_graph = output_dir / (
            "assembly.filtered.gfa" if (preflight / "assembly.filtered.gfa").is_file()
            else "assembly.gfa"
        )
        configuration = infer_configurations(
            promoted_graph, json.loads((output_dir / "evidence.json").read_text())
        )
        configuration_file = output_dir / "configuration.json"
        configuration_file.write_text(json.dumps(configuration, indent=2) + "\n")
        selected["configuration"] = {
            "graph_order_identified": configuration["graph_order_identified"],
            "physical_topology_identified": configuration["physical_topology_identified"],
        }
        selected["physical_molecule_count"] = None
        files["configuration"] = str(configuration_file)
        if failure is not None:
            selected["multichrom_failure"] = failure
            selected["warnings"] = [*selected.get("warnings", []),
                "Multichrom graph failed; the earlier ordinary assembly was retained. "
                "Inspect its graph and configuration uncertainty."]
        if recruitment is not None:
            recruited = recruitment["outputs"][_target_index(recruitment, organelle)]
            selected["recruited_depth"] = recruited["kept_bases"] / sum(selected["lengths"])
        (output_dir / "summary.json").write_text(json.dumps(selected, indent=2) + "\n")
        promoted = {
            key: str(output_dir / Path(value).relative_to(preflight))
            for key, value in ordinary["output_files"].items()
        }
        ordinary["output_files"] = {**files, **promoted}
        ordinary["summary_text"] += (
            "; preserved supported closed cover" if failure is None
            else "; WARNING: multichrom graph failed, retained earlier ordinary assembly"
        )
        log(ordinary["summary_text"])
        return ordinary

    if multichrom:
        log("OVASM: checking whether the ordinary graph already has a supported complete circular cover")
        try:
            ordinary = run_ovasm(
                {**params, "read_set": "target_reads", "graph_mode": "sparse",
                 "graph_min_count": None, "also_discover": False},
                organelle=organelle, reads=reads, output_dir=preflight, log=log,
            )
            preflight_summary = json.loads((preflight / "summary.json").read_text())
            preflight_linear = json.loads((preflight / "linearization.json").read_text())
        except OrganelleExecutionError as error:
            log(f"OVASM: ordinary graph preflight did not produce a cover: {error}")
        else:
            if mode == "target_reads" and _preserve_closed_cover(
                preflight_summary, preflight_linear, preflight / "assembly.gfa"
            ):
                return promote_ordinary("supported_closed_cover_preserved")
        log(
            "OVASM: continuing with the original WGS multichrom graph; recruitment does not establish WGS completeness"
            if mode == "whole_genome" else
            "OVASM: ordinary cover is incomplete or unresolved; continuing with the multichrom graph"
        )

    graph = output_dir / "assembly.gfa"
    # reads found by depth: the graph goes on without the components that are not the target
    # (seed-only and target-read runs keep the whole graph, as before)
    discovered = also_discover or seed_info["source"] == "discover"

    def attempt(k: int | None) -> dict[str, Any]:
        """Assemble at `k` (None: ovasm picks it), filter, score, unring; what that gave.

        Short reads: k comes from the read length; HiFi: the largest k deep enough,
        self-correcting the reads at low depth (see `ovasm assemble --help`).
        """
        log("OVASM: assembling the read graph" + (f" at k={k}" if k else ""))
        graph_options = {"multichrom_graph": True, "min_count": int(graph_min_count)} if multichrom else {}
        assembly = _ovasm.run_assemble(
            assembly_reads,
            out_gfa=graph,
            out_json=output_dir / "assembly.json",
            k=k,
            dense=preset == "sr",
            threads=threads,
            **graph_options,
        )
        kept_graph = graph
        components: dict[str, Any] | None = None
        unassigned: dict[str, Any] | None = None
        if discovered or multichrom:
            kept_graph, components = _keep_target_components(
                graph, organelle, output_dir, files, log,
                **({"reject_other_only": True} if multichrom else {}),
                **({"read_origins": read_origins} if read_origins is not None else {}),
            )
        log("OVASM: evaluating read support and repeat pairings")
        evidence = _ovasm.run_evidence(
            kept_graph,
            assembly_reads,
            out_json=output_dir / "evidence.json",
            preset=preset,
            min_anchor=40 if preset == "sr" else 500,
            threads=threads,
        )
        log("OVASM: generating representative sequences and retaining configuration uncertainty")
        if multichrom:
            primary_graph = kept_graph
            if read_origins is not None:
                primary_graph = output_dir / "assembly.assigned.gfa"
                files["assigned_graph"] = str(primary_graph)
                files["unassigned_graph"] = str(output_dir / "assembly.unassigned.gfa")
                unassigned = _ovasm.run_primary(
                    output_dir / "assembly.unassigned.gfa",
                    out_fasta=output_dir / "unassigned.fasta",
                    out_json=output_dir / "unassigned.json",
                )
                unassigned_fasta = output_dir / "unassigned.fasta"
                unassigned_fasta.write_text(
                    unassigned_fasta.read_text().replace(">primary.", ">unassigned.")
                )
                files["unassigned_sequences"] = str(output_dir / "unassigned.fasta")
                files["unassigned_report"] = str(output_dir / "unassigned.json")
                log(f"OVASM: preserved {len(unassigned['records'])} records with competing origins separately")
            linear = _ovasm.run_primary(
                primary_graph, out_fasta=output_dir / "molecules.fasta",
                out_json=output_dir / "linearization.json",
            )
            molecules = linear["records"]
        else:
            linear = _ovasm.run_linearize(
                kept_graph,
                output_dir / "evidence.json",
                out_fasta=output_dir / "molecules.fasta",
                out_json=output_dir / "linearization.json",
                max_molecules=max_molecules,
            )
            molecules = [
                m for c in linear["components"] for m in (c.get("best") or {}).get("molecules", [])
            ]
        # what the molecules have to explain: the graph, or with the component filter the
        # components it kept (the nuclear repeats beside the organelle are not theirs to explain)
        total_length = int(
            components["kept_length"] if components is not None else assembly.get("total_length", 0)
        )
        molecule_bases = sum(int(m["length"]) for m in molecules)
        explained = molecule_bases / total_length if molecules and total_length else 0.0
        outcome = {
            "k": assembly["k"],
            "molecules": len(molecules),
            "molecule_bases": molecule_bases,
            "total_length": total_length,
            "explained": explained,
            "accepted": len(molecules) >= 1 and explained >= MIN_EXPLAINED,
        }
        if not outcome["accepted"]:
            log(
                f"OVASM: k={outcome['k']} gave {len(molecules)} molecule(s) of {molecule_bases} "
                f"bp in a graph of {total_length} bp: under {MIN_EXPLAINED:.0%} explained"
            )
        return {
            "outcome": outcome,
            "assembly": assembly,
            "kept_graph": kept_graph,
            "components": components,
            "unassigned": unassigned,
            "evidence": evidence,
            "linear": linear,
            "molecules": molecules,
            "curable": k_curable(organelle, molecules, linear),
            "has_circle": any(m.get("circular") for m in molecules),
        }

    # k ladder (same rule as `ovasm run`, rust/ovasm/src/pipeline.rs). A large k can leave a
    # gap the rescue cannot fill (Zou AT101/AT105 plastids: k=1001 gave 2 unitigs and no
    # molecule, k=701 down to 251 all gave the exact plastome), and the automatic k only looks
    # at depth. "A molecule" is not enough either: Carex laevigata's mitochondrion (2.76 Mb
    # graph) gave 2 molecules of 11,560 and 349 bp at k=251 that share nothing with the
    # published genome. An attempt is accepted when it has a molecule and its molecules
    # explain at least MIN_EXPLAINED of the graph; otherwise the next smaller k is tried.
    # When none is accepted, the attempt that explains the largest share is kept (the
    # earliest of equals), re-run last so that the files are its, and the result is flagged.
    # Dense (short-read) assembly takes k from the read length, so it is not retried.
    # Reads that were corrected (noisy long reads) start at NOISY_K, the k validated for them
    # (the automatic k looks at depth only), and the ladder goes down from there; the correction
    # is done once, before the ladder.
    k_tried: list[dict[str, Any]] = []
    k_info: list[tuple[bool, bool]] = []  # per k tried: (another k could cure it, has a circle)
    best = 0
    k: int | None = NOISY_K if correction is not None else None
    if k_ladder_start == "top":
        k = K_LADDER[0]
        log(f"OVASM: starting the HiFi k ladder at {k}")
    if multichrom:
        k = 1001
        log("OVASM: building the multichrom graph at k=1001 compressed runs with the supplied read-coverage cutoff")
    while True:
        try:
            result = attempt(k)
        except OrganelleExecutionError as error:
            if read_origins is not None or not multichrom or ordinary is None or not preflight_summary or not preflight_summary.get("molecules"):
                raise
            log(f"OVASM: multichrom attempt failed; retaining the existing ordinary assembly: {error}")
            return promote_ordinary("ordinary_retained_after_multichrom_failure", str(error))
        outcome = result["outcome"]
        if not k_tried or outcome["explained"] > k_tried[best]["explained"]:
            best = len(k_tried)
        k_tried.append(outcome)
        k_info.append((result["curable"], result["has_circle"]))
        smaller = [x for x in K_LADDER if x < outcome["k"]]
        if multichrom or outcome["accepted"] or preset == "sr" or not smaller:
            break
        k = smaller[0]
    # More k (`k_search`, set by the escalation wrapper only; a plain run never gets here).
    # Larger k than the automatic one when nothing was accepted: the automatic k only looks at
    # depth, and Schoenoplectus lacustris' mitochondrion shattered at its automatic 201/151
    # while a start at 1001/701 worked. These are the k a `k_ladder_start="top"` run tries
    # first, here reached inside the same attempt (the recruited reads are kept) and only after
    # the downward ladder failed. Largest first; an error here ends the search.
    ended_clean = True
    explored = False
    started_accepted = k_tried[-1]["accepted"]
    # also when the first accepted k leaves what another k could cure (a plastid without a
    # circle: Draba incana's plastid at k=1001 is one linear 82,858 bp molecule, 66% of the
    # graph, "accepted", while k=701..251 close the 153,566 bp circle; a mitochondrion with
    # graph components or paths left over). The k not tried yet, smaller then larger, at most
    # MAX_EXTRA_K of them in that case, ending at the first complete result.
    if k_search and preset != "sr" and not multichrom and (not started_accepted or k_info[-1][0]):
        tried_ks = {o["k"] for o in k_tried}
        smaller = [x for x in K_LADDER if x < k_tried[-1]["k"] and x not in tried_ks]
        bigger = [x for x in K_LADDER if x > k_tried[0]["k"] and x not in tried_ks]
        budget = MAX_EXTRA_K if started_accepted else None
        for k_next in smaller + bigger:
            if budget == 0:
                break
            if started_accepted:
                log(f"OVASM: k={k_tried[0]['k']} left a gap another k may close; trying k={k_next}")
            else:
                log(f"OVASM: no k was accepted; trying k={k_next}")
            try:
                result = attempt(k_next)
            except Exception as error:
                log(f"OVASM: k={k_next} failed ({type(error).__name__}: {error})")
                ended_clean = False
                break
            explored = True
            k_tried.append(result["outcome"])
            k_info.append((result["curable"], result["has_circle"]))
            if budget is not None:
                budget -= 1
            if result["outcome"]["accepted"] and not result["curable"]:
                break
    if explored:
        # accepted first, then whole (nothing another k could cure), then a circle, then the
        # share of the graph the molecules explain; the earliest of equals
        def rank(i: int) -> tuple[Any, ...]:
            accepted = k_tried[i]["accepted"]
            return (
                accepted,
                (not k_info[i][0]) if accepted else True,
                k_info[i][1] if accepted else False,
                k_tried[i]["explained"],
                -i,
            )

        best = max(range(len(k_tried)), key=rank)
        k_accepted = k_tried[best]["accepted"]
        kept_attempt = best
    else:
        k_accepted = k_tried[-1]["accepted"]
        kept_attempt = len(k_tried) - 1 if k_accepted else best
    if kept_attempt != len(k_tried) - 1 or not ended_clean:
        log(
            f"OVASM: keeping k={k_tried[kept_attempt]['k']}, the best of the {len(k_tried)} k "
            "tried" + ("" if k_accepted else " (no k was accepted: the molecules explain most of its graph)")
        )
        result = attempt(k_tried[kept_attempt]["k"])
    assembly = result["assembly"]
    kept_graph = result["kept_graph"]
    components = result["components"]
    unassigned = result["unassigned"]
    evidence = result["evidence"]
    linear = result["linear"]
    molecules = result["molecules"]
    log("OVASM: exporting the unified organelle graph")
    _ovasm.run_unify(
        kept_graph,
        sample=str(params.get("sample") or "desktop"),
        organelle=organelle,
        backend="ovasm",
        out_gfa=output_dir / "organelle.gfa",
        out_json=output_dir / "organelle.json",
        evidence_json=output_dir / "evidence.json",
        linearize_json=output_dir / "linearization.json",
    )
    identity, warnings = None, []
    if molecules:
        log("OVASM: comparing the assembly with the builtin organelle references")
        identity, warnings = _identify(
            output_dir / "molecules.fasta", organelle, output_dir / "identity.json"
        )
        files["identity"] = str(output_dir / "identity.json")
    recruited_bases = None
    if recruitment is not None:
        out = recruitment["outputs"][_target_index(recruitment, organelle)]
        recruited_bases = out["kept_bases"]
    report = {
        "backend": "ovasm",
        "organelle": organelle,
        "read_set": mode,
        "read_type": strategy,
        "seed_database": seed_info,
        "recruited_reads": _recruited(recruitment, organelle),
        "k": assembly["k"],
        "k_tried": k_tried,
        "graph_first": graph_first_info,
        # false: no k explained MIN_EXPLAINED of its graph with molecules; the files are those
        # of the attempt that came closest and its molecules are not to be trusted
        "k_accepted": k_accepted,
        "min_explained": MIN_EXPLAINED,
        "kept_attempt": kept_attempt,
        "unitigs": assembly["unitigs"],
        "supported_links": evidence["links_supported"],
        "links": len(evidence["links"]),
        "molecules": len(molecules),
        "lengths": [m["length"] for m in molecules],
        "circular": [m["circular"] for m in molecules],
        # paths the reads join where no cover of molecules was found: not molecules
        "partial_paths": len(linear.get("partial_paths") or []),
        "decisive": linear["decisive"],
        "unsolved_components": linear.get("unsolved_components", 0),
        "unbridged_anchors": linear.get("unbridged_anchors", []),
        "skipped_anchors": linear.get("skipped_anchors", []),
        "search_truncated": linear.get("search_truncated", False),
        "cover_size_limited": bool(linear.get("cover_size_limited", False)),
        "max_molecules": max_molecules,
        "k_ladder_start": k_ladder_start,
        # recruited bases per assembled base: the read depth behind the molecules
        "recruited_depth": recruited_bases / sum(m["length"] for m in molecules)
        if recruited_bases is not None and molecules
        else None,
        "identity": identity,
        "warnings": warnings,
        "scope": "native assembly and read evidence; no independent assembler certification or polishing",
    }
    if multichrom:
        from organelleverse.assembly.configuration import infer_configurations

        configuration = infer_configurations(kept_graph, evidence)
        configuration_file = output_dir / "configuration.json"
        configuration_file.write_text(json.dumps(configuration, indent=2) + "\n")
        files["configuration"] = str(configuration_file)
        report.update(
            graph_mode=graph_mode,
            multichrom_selection="unresolved_cover_primary_representation",
            ordinary_preflight=preflight_summary,
            graph_min_count=int(graph_min_count),
            graph_min_count_auto=automatic_coverage,
            multichrom_input_protocol="2",
            graph_input_context="original_wgs" if mode == "whole_genome" else "supplied_target_reads",
            graph_input_read_count=assembly.get("reads"),
            graph_input_bases=assembly.get("bases"),
            graph_engine=assembly["graph_engine"],
            graph_engine_source=assembly["upstream"],
            graph_engine_revision=assembly["upstream_revision"],
            graph_engine_patches=assembly.get("upstream_patches", []),
            representation=linear["representation"],
            assembly_sequences=len(molecules),
            primary_paths=sum(m["kind"] == "primary_path" for m in molecules),
            unplaced_segments=linear["unplaced_segments"],
            physical_molecule_count=None,
            complete=False,
            configuration={
                "graph_order_identified": configuration["graph_order_identified"],
                "physical_topology_identified": configuration["physical_topology_identified"],
            },
        )
        if coverage_profile is not None:
            report["nuclear_coverage"] = coverage_profile
            files["nuclear_coverage"] = str(output_dir / "nuclear-coverage/coverage-profile.json")
    if components is not None:
        report["component_filter"] = _component_filter_summary(components)
        if read_origins is not None:
            report["source_assignment"] = {
                "method": "abstain_on_observed_competing_origin",
                "scope": "reference-assisted target candidates; all unassigned sequence preserved",
                "unassigned_components": sum(c["target_assignment"] == "unassigned" for c in components["components"]),
                "unassigned_records": len(unassigned["records"]),
                "unassigned_bases": sum(m["length"] for m in unassigned["records"]),
                "all_retained_sequence_requires_scoring": True,
                "physical_origin_identified": False,
            }
        uncertain = report["component_filter"]
        if uncertain["retained_without_target_markers"]:
            warnings.append(
                f"{uncertain['retained_without_target_markers']} retained graph components "
                f"({uncertain['retained_without_target_markers_length']} bp) have no target "
                "panel markers; retention does not establish organelle origin"
            )
    if correction is not None:
        report["correction"] = correction
    summary = output_dir / "summary.json"
    summary.write_text(json.dumps(report, indent=2) + "\n")
    files.update(
        {
            "summary": str(summary),
            "assembly_graph": str(graph),
            "organelle_graph": str(output_dir / "organelle.gfa"),
            "graph_metadata": str(output_dir / "organelle.json"),
            "evidence": str(output_dir / "evidence.json"),
            "linearization": str(output_dir / "linearization.json"),
        }
    )
    if molecules:
        files["assembly_fasta"] = str(output_dir / "molecules.fasta")
    if linear.get("partial_paths") and linear.get("partial_fasta"):
        files["partial_fasta"] = str(linear["partial_fasta"])
    message = f"OVASM: {len(molecules)} representative molecules; supported links {evidence['links_supported']}/{len(evidence['links'])}; decisive={linear['decisive']}"
    if multichrom:
        message = (
            f"OVASM: {len(molecules)} assembly sequences ({report['primary_paths']} primary paths, "
            f"{report['unplaced_segments']} unplaced segments); physical molecule configuration unresolved"
        )
    if components is not None and components["removed_components"]:
        message += (
            f"; {components['removed_components']} graph components "
            f"({components['removed_length']} bp) removed as not the {organelle}"
        )
    if not molecules:
        message += "; no representative sequence, inspect the graph and evidence"
    else:
        c = identity["closest"]
        family = f" ({c['family']})" if c["family"] else ""
        message += (
            f"; closest reference {c['organism']}{family}, ~{c['estimated_identity']:.1%} identity"
        )
        if linear.get("unsolved_components"):
            # the molecules above are only the solved part of the graph (noisy reads are the usual
            # cause: self-correction alone left Nipponbare ONT in 8 fragments)
            message += (
                f"; {linear['unsolved_components']} graph components have no representative"
                " sequence (linearization.json: unsolved_components, unbridged_anchors)"
            )
    for warning in warnings:
        message += f"; WARNING: {warning}"
    if not k_accepted:
        best_try = k_tried[kept_attempt]
        message += (
            f"; NOT TRUSTWORTHY: no k gave molecules explaining {MIN_EXPLAINED:.0%} of the "
            f"graph (best: {best_try['explained']:.0%} at k={best_try['k']})"
        )
    log(message)
    return {"summary_text": message, "output_files": files}
