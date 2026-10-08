"""Gold mode: an organelle reference with a certificate (ovasm stage S5).

One run per species reference. ovasm builds the evidence graph from the most accurate reads
(HiFi, else noisy long reads corrected with short reads, else short reads alone), and every
other platform checks it independently:

1. recruit each platform's organelle reads (short reads seeded by the sample's own long
   reads when there are any);
2. build the graph and score it with every platform's reads (``ovasm evidence``): support of
   each link and the minority share at each branch, per platform;
3. unring it with the long reads' phasing and write OV-GFA;
4. cross-validate: external assemblers (HiMT, Oatk, PMAT, GetOrganelle) on the same recruited
   reads, and a short-read ovasm graph when short reads came with long ones, compared junction
   by junction in both directions (sequence flanks, not names);
5. polish the molecules with the most accurate platform (haploid calls) and count what the
   other platforms would still change;
6. compare with an earlier reference (SNVs, indels, alignment blocks);
7. write ``certificate.json`` and ``certificate.md``: every number above and a list of checks.

External assemblers run in their managed environments; one that is missing or fails is
recorded, not fatal. The orchestration lives here rather than in the Rust binary because
those assemblers are Python-managed conda environments; ovasm itself stays a single file.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .._ovasm import (
    run_assemble,
    run_correct,
    run_evidence,
    run_linearize,
    run_recruit,
    run_unify,
)
from .graph_compare import compare_linear, parse_gfa, score_graphs

#: Which recruited reads each external assembler takes.
EXTERNAL_INPUT = {"oatk": "hifi", "himt": "hifi", "pmat": "hifi", "getorganelle": "short"}

#: Junction flank for cross-validation (bp each side), as in the gold-graph scoring.
FLANK = 500


@dataclass(frozen=True)
class GoldInputs:
    sample: str
    out_dir: Path
    seeds: Mapping[str, Sequence[Path]]
    hifi: Sequence[Path] = ()
    ont: Sequence[Path] = ()
    clr: Sequence[Path] = ()
    short: Sequence[Path] = ()
    organelle: str = "mitochondrion"
    reference: Path | None = None
    external: Sequence[str] = ("oatk", "himt", "pmat", "getorganelle")
    threads: int = 16
    target_depth: float = 150.0


@dataclass
class _Run:
    out: Path
    record: dict[str, Any] = field(default_factory=lambda: {"steps": []})

    def step(self, name: str, fn: Callable[[], Any], *, fatal: bool = True) -> Any:
        started = time.time()
        entry: dict[str, Any] = {"name": name, "status": "running"}
        self.record["steps"].append(entry)
        try:
            value = fn()
        except Exception as error:
            entry.update(
                status="failed",
                seconds=round(time.time() - started, 1),
                error=f"{type(error).__name__}: {error}",
            )
            (self.out / "logs").mkdir(exist_ok=True)
            (self.out / "logs" / f"{name}.traceback.txt").write_text(traceback.format_exc())
            if fatal:
                raise
            return None
        entry.update(status="completed", seconds=round(time.time() - started, 1))
        return value


def _tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise FileNotFoundError(f"{name} is required for this step and is not on PATH")
    return path


def _sh(argv: Sequence[str], stdout: Path | None = None) -> None:
    with open(stdout, "wb") if stdout else open("/dev/null", "wb") as out:
        subprocess.run(list(argv), check=True, stdout=out, stderr=subprocess.PIPE)


# --- reads -------------------------------------------------------------------------------


def _recruit_and_correct(g: GoldInputs, run: _Run) -> dict[str, list[Path]]:
    """Organelle reads per platform: hifi, noisy (corrected ONT/CLR), short."""
    org = g.organelle
    reads: dict[str, list[Path]] = {}
    out = g.out_dir / "reads"
    if g.hifi:
        run.step(
            "recruit_hifi",
            lambda: run_recruit(
                g.hifi,
                seeds=g.seeds,
                out_dir=out / "hifi",
                preset="hifi",
                target_depth=g.target_depth,
                threads=g.threads,
            ),
        )
        reads["hifi"] = [out / "hifi" / f"{org}.fastq"]
    noisy_raw = list(g.ont) + list(g.clr)
    self_corrected = None
    if noisy_raw:
        run.step(
            "recruit_noisy",
            lambda: run_recruit(
                noisy_raw,
                seeds=g.seeds,
                out_dir=out / "noisy",
                preset="ont",
                target_depth=2 * g.target_depth,
                threads=g.threads,
            ),
        )
        self_corrected = out / "noisy_self.fa"
        run.step(
            "correct_noisy_self",
            lambda: run_correct(
                [out / "noisy" / f"{org}.fastq"],
                out_fasta=self_corrected,
                out_json=out / "noisy_self.json",
                threads=g.threads,
            ),
        )
    if g.short:
        # the sample's own long reads are the best bait for its short reads
        long_seed = reads.get("hifi") or ([self_corrected] if self_corrected else None)
        seeds = {k: v for k, v in g.seeds.items() if k != org} if long_seed else g.seeds
        run.step(
            "recruit_short",
            lambda: run_recruit(
                g.short,
                seeds=seeds,
                seed_reads={org: long_seed} if long_seed else None,
                out_dir=out / "short",
                preset="sr",
                threads=g.threads,
            ),
        )
        reads["short"] = [out / "short" / f"{org}.fastq"]
    if noisy_raw:
        if g.short:
            hybrid = out / "noisy_hybrid.fa"
            run.step(
                "correct_noisy_hybrid",
                lambda: run_correct(
                    [out / "noisy" / f"{org}.fastq"],
                    out_fasta=hybrid,
                    out_json=out / "noisy_hybrid.json",
                    ks=(21, 25, 31),
                    short_reads=reads["short"],
                    threads=g.threads,
                ),
            )
            reads["noisy"] = [hybrid]
        else:
            reads["noisy"] = [self_corrected]
    return reads


# --- graph -------------------------------------------------------------------------------

_EVIDENCE = {"hifi": ("hifi", 500), "noisy": ("hifi", 500), "short": ("sr", 40)}


def _build(reads: dict[str, list[Path]], g: GoldInputs, run: _Run, platform: str, tag: str):
    d = g.out_dir / tag
    d.mkdir(parents=True, exist_ok=True)
    k = {"hifi": None, "noisy": 501, "short": None}[platform]
    report = run.step(
        f"assemble_{tag}",
        lambda: run_assemble(
            reads[platform],
            out_gfa=d / "asm.gfa",
            out_json=d / "asm.json",
            k=k,
            dense=platform == "short",
            threads=g.threads,
        ),
    )
    return d / "asm.gfa", report


def _evidence(gfa: Path, reads: dict[str, list[Path]], g: GoldInputs, run: _Run, tag: str):
    out = {}
    for platform, files in reads.items():
        preset, anchor = _EVIDENCE[platform]
        path = gfa.parent / f"evidence_{platform}.json"
        rep = run.step(
            f"evidence_{tag}_{platform}",
            lambda files=files, preset=preset, anchor=anchor, path=path: run_evidence(
                gfa,
                files,
                out_json=path,
                preset=preset,
                min_anchor=anchor,
                threads=g.threads,
            ),
            fatal=False,
        )
        if rep is not None:
            out[platform] = (path, rep)
    return out


def _multi_platform(evidence: dict[str, tuple[Path, dict]]) -> dict[str, Any]:
    """Per link and per branch: reads and minority share on every platform.

    A platform's reads can support a link, cross it without telling it from the other ways
    at the same end (``ambiguous``: short reads inside a repeat longer than they are), or
    not reach it at all (``silent``). Only support counts as confirmation; the other two are
    listed, not held against the link.
    """
    links: dict[str, dict[str, int]] = {}
    ambiguous: dict[str, dict[str, int]] = {}
    branches: dict[str, dict[str, Any]] = {}
    for platform, (_, rep) in evidence.items():
        for link in rep["links"]:
            key = f"{link['from']} -> {link['to']}"
            links.setdefault(key, {})[platform] = link["reads"]
            ambiguous.setdefault(key, {})[platform] = link.get("ambiguous_reads", 0)
        for b in rep["branches"]:
            branches.setdefault(b["from"], {})[platform] = {
                "alternatives": {a["to"]: a["reads"] for a in b["alternatives"]},
                "minority_fraction": b["minority_fraction"],
                "minority_ci95": b.get("minority_ci95"),
            }
    platforms = list(evidence)
    per_platform = {
        p: {
            "supported": sum(1 for v in links.values() if v.get(p, 0) > 0),
            "ambiguous_only": sum(
                1 for k, v in links.items() if v.get(p, 0) == 0 and ambiguous[k].get(p, 0) > 0
            ),
            "silent": sum(
                1 for k, v in links.items() if v.get(p, 0) == 0 and ambiguous[k].get(p, 0) == 0
            ),
        }
        for p in platforms
    }
    confirmed = sum(1 for v in links.values() if sum(v.get(p, 0) > 0 for p in platforms) >= 2)
    return {
        "platforms": platforms,
        "links": links,
        "ambiguous_reads": ambiguous,
        "links_total": len(links),
        "links_confirmed_by_two_platforms": confirmed,
        "per_platform": per_platform,
        "branches": branches,
        "phased_paths": {p: len(rep.get("phased_paths", [])) for p, (_, rep) in evidence.items()},
    }


# --- cross-validation --------------------------------------------------------------------


def _external(g: GoldInputs, reads: dict[str, list[Path]], run: _Run) -> dict[str, Path]:
    """Graphs of the external assemblers that could run, by tool."""
    from ..io_reads import read_long_reads, read_reads
    from .api import assemble, write
    from .contracts import ShortReadLibrary

    graphs: dict[str, Path] = {}
    for tool in g.external:
        need = EXTERNAL_INPUT.get(tool)
        if need is None or need not in reads:
            run.record.setdefault("external_skipped", {})[tool] = f"needs {need} reads"
            continue
        out = g.out_dir / "external" / tool

        def go(tool=tool, need=need, out=out):
            if need == "hifi":
                data = read_long_reads(
                    reads["hifi"][0], technology="pacbio_hifi", quality_state="ccs"
                )
            else:
                data = read_reads(
                    short_libraries=(
                        ShortReadLibrary(
                            technology="illumina",
                            layout="single_end",
                            read1=reads["short"][0],
                            read_length=150,
                        ),
                    )
                )
            result = assemble(data, organelle=g.organelle, method=tool, threads=g.threads)
            if result.status == "failed":
                raise RuntimeError(f"{tool} failed: {result.model_dump_json()[:2000]}")
            write(result, output=out)
            found = sorted(out.rglob("normalized/assembly.gfa")) or sorted(out.rglob("*.gfa"))
            if not found:
                raise RuntimeError(f"{tool} wrote no GFA")
            return found[0]

        gfa = run.step(f"external_{tool}", go, fatal=False)
        if gfa is not None:
            graphs[tool] = gfa
    return graphs


def _concordance(primary: Path, others: dict[str, Path]) -> dict[str, Any]:
    """Junction concordance of ovasm's graph with every other graph, both ways."""
    minimap2 = _tool("minimap2")
    _, links = parse_gfa(primary)
    per_junction: dict[str, list[str]] = {
        f"{lk.source}{lk.source_orient} -> {lk.target}{lk.target_orient}": [] for lk in links
    }
    names = list(per_junction)
    tools: dict[str, Any] = {}
    for tool, gfa in others.items():
        if not parse_gfa(gfa)[1]:
            tools[tool] = {"note": "graph has no links"}
            continue
        forward = score_graphs(primary, gfa, FLANK, "align", minimap2)
        backward = score_graphs(gfa, primary, FLANK, "align", minimap2)
        missing = set(forward["missing_junctions_tolerant"])
        for i, name in enumerate(names, 1):
            if f"J{i:02d}" not in missing:
                per_junction[name].append(tool)
        tools[tool] = {
            "ovasm_junctions_found": forward["junctions_recovered_tolerant"],
            "ovasm_junctions": forward["gold_junctions"],
            "its_junctions_found_in_ovasm": backward["junctions_recovered_tolerant"],
            "its_junctions": backward["gold_junctions"],
        }
    confirmed = sum(1 for v in per_junction.values() if v)
    independent = {
        name: [tool for tool in supporting if tool in EXTERNAL_INPUT]
        for name, supporting in per_junction.items()
    }
    return {
        "tools": tools,
        "per_junction": per_junction,
        "junctions_confirmed_by_another_graph": confirmed,
        "independent_assemblers": [tool for tool in tools if tool in EXTERNAL_INPUT],
        "per_junction_independent": independent,
        "junctions_confirmed_by_independent_assembler": sum(bool(v) for v in independent.values()),
        "junctions": len(names),
    }


# --- polishing and comparison ------------------------------------------------------------

_MAP_PRESET = {"hifi": "map-hifi", "noisy": "map-hifi", "short": "sr"}


def _call(fasta: Path, reads: list[Path], platform: str, work: Path, threads: int) -> list[tuple]:
    """Haploid variant calls of ``reads`` against ``fasta``: (contig, pos1, ref, alt)."""
    minimap2, samtools, bcftools = _tool("minimap2"), _tool("samtools"), _tool("bcftools")
    work.mkdir(parents=True, exist_ok=True)
    bam = work / f"{platform}.bam"
    align = subprocess.Popen(
        [minimap2, "-ax", _MAP_PRESET[platform], "-t", str(threads), str(fasta), *map(str, reads)],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        [samtools, "sort", "-@", "4", "-o", str(bam), "-"],
        stdin=align.stdout,
        check=True,
        stderr=subprocess.DEVNULL,
    )
    align.wait()
    _sh([samtools, "index", str(bam)])
    _sh([samtools, "faidx", str(fasta)])
    pileup = subprocess.Popen(
        [
            bcftools,
            "mpileup",
            "-f",
            str(fasta),
            "-q",
            "20",
            "-Q",
            "13",
            "-a",
            "AD,DP",
            "-d",
            "100000",
            "-Ou",
            str(bam),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    called = subprocess.run(
        [bcftools, "call", "-m", "-v", "--ploidy", "1", "-Ov"],
        stdin=pileup.stdout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    pileup.wait()
    (work / f"{platform}.vcf").write_text(called)
    out = []
    for line in called.splitlines():
        if line.startswith("#"):
            continue
        f = line.split("\t")
        qual = float(f[5]) if f[5] != "." else 0.0
        fmt, sample = f[8].split(":"), f[9].split(":")
        ad = [int(x) for x in sample[fmt.index("AD")].split(",")] if "AD" in fmt else []
        depth = sum(ad)
        alt_share = (ad[1] / depth) if len(ad) > 1 and depth else 0.0
        if qual >= 30 and depth >= 10 and alt_share >= 0.8 and "," not in f[4]:
            out.append((f[0], int(f[1]), f[3], f[4]))
    return out


def _apply(fasta: Path, edits: list[tuple], dest: Path) -> None:
    seqs: dict[str, str] = {}
    headers: dict[str, str] = {}
    name = None
    for line in fasta.read_text().splitlines():
        if line.startswith(">"):
            name = line[1:].split()[0]
            headers[name] = line
            seqs[name] = ""
        elif name is not None:
            seqs[name] += line.strip()
    by_contig: dict[str, list[tuple]] = {}
    for c, pos, ref, alt in edits:
        by_contig.setdefault(c, []).append((pos, ref, alt))
    for c, rows in by_contig.items():
        s = seqs[c]
        for pos, ref, alt in sorted(rows, reverse=True):
            i = pos - 1
            if s[i : i + len(ref)].upper() == ref.upper():
                s = s[:i] + alt + s[i + len(ref) :]
        seqs[c] = s
    dest.write_text(
        "".join(
            f"{headers[n]}\n"
            + "".join(seqs[n][i : i + 80] + "\n" for i in range(0, len(seqs[n]), 80))
            for n in seqs
        )
    )


def _polish(fasta: Path, reads: dict[str, list[Path]], g: GoldInputs, run: _Run) -> dict:
    """Polish with the most accurate platform; count what the others would still change."""
    order = [p for p in ("hifi", "short", "noisy") if p in reads]
    work = g.out_dir / "polish"
    best = order[0]
    edits = _call(fasta, reads[best], best, work / "round1", g.threads)
    polished = g.out_dir / "polished.fa"
    _apply(fasta, edits, polished)
    kinds = {"snv": 0, "insertion": 0, "deletion": 0}
    for _, _, ref, alt in edits:
        kinds[
            "snv" if len(ref) == len(alt) else "insertion" if len(alt) > len(ref) else "deletion"
        ] += 1
    residual = {}
    for p in order:
        rows = _call(polished, reads[p], p, work / f"check_{p}", g.threads)
        residual[p] = len(rows)
    return {
        "polished_with": best,
        "edits": len(edits),
        "edit_kinds": kinds,
        "edit_list": [{"contig": c, "pos": p, "ref": r, "alt": a} for c, p, r, a in edits[:500]],
        "residual_calls_after_polishing": residual,
        "polished_fasta": str(polished),
    }


def _compare_reference(fasta: Path, reference: Path) -> dict:
    """SNVs, indels and alignment blocks of the molecules against an earlier reference."""
    minimap2 = _tool("minimap2")
    paf = subprocess.run(
        [minimap2, "-cx", "asm5", "--cs", str(reference), str(fasta)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    snv = ins = dele = ins_bp = del_bp = large = blocks = 0
    for row in paf.splitlines():
        f = row.split("\t")
        if "tp:A:P" not in f:
            continue
        blocks += 1
        cs = next((t[5:] for t in f[12:] if t.startswith("cs:Z:")), "")
        for op, body in re.findall(r"([:*+\-~])([^:*+\-~]*)", cs):
            if op == "*":
                snv += 1
            elif op == "+":
                ins += 1
                ins_bp += len(body)
                large += len(body) >= 50
            elif op == "-":
                dele += 1
                del_bp += len(body)
                large += len(body) >= 50
    linear = compare_linear(reference, fasta, minimap2)
    return {
        "reference": str(reference),
        "snvs": snv,
        "insertions": ins,
        "insertion_bp": ins_bp,
        "deletions": dele,
        "deletion_bp": del_bp,
        "indels_50bp_or_more": large,
        "primary_alignment_blocks": blocks,
        "reference_aligned_fraction": linear["gold_aligned_fraction"],
        "alignment_identity": linear["alignment_identity"],
    }


# --- certificate -------------------------------------------------------------------------


def _checks(cert: dict) -> list[dict]:
    checks = []
    mp = cert.get("multi_platform") or {}
    primary_ev = cert.get("evidence_primary") or {}
    if primary_ev:
        ok = primary_ev["links_supported"] == primary_ev["links"]
        checks.append(
            {
                "name": "every link supported by the primary platform's reads",
                "passed": ok,
                "detail": f"{primary_ev['links_supported']}/{primary_ev['links']}",
            }
        )
    if mp and len(mp["platforms"]) > 1:
        checks.append(
            {
                "name": "every link confirmed by a second platform",
                "passed": mp["links_confirmed_by_two_platforms"] == mp["links_total"],
                "detail": f"{mp['links_confirmed_by_two_platforms']}/{mp['links_total']}; "
                + "; ".join(
                    f"{p}: {c['supported']} supported, {c['ambiguous_only']} ambiguous only, "
                    f"{c['silent']} silent"
                    for p, c in mp["per_platform"].items()
                ),
            }
        )
    un = cert.get("unringing") or {}
    if un:
        checks.append(
            {
                "name": "the graph unrings into molecules",
                "passed": un.get("unsolved_components", 1) == 0 and un.get("molecules", 0) > 0,
                "detail": f"{un.get('molecules')} molecules, decisive={un.get('decisive')}",
            }
        )
    cv = cert.get("cross_validation") or {}
    if cv.get("tools"):
        independent = cv["junctions_confirmed_by_independent_assembler"]
        checks.append(
            {
                "name": "junctions confirmed by at least one independent graph",
                "passed": cv["junctions"] > 0 and independent == cv["junctions"],
                "detail": f"{independent}/{cv['junctions']}"
                f" (independent assemblers: {', '.join(cv['independent_assemblers']) or 'none'}; "
                f"all additional graphs: {cv['junctions_confirmed_by_another_graph']}/{cv['junctions']})",
            }
        )
    po = cert.get("polishing") or {}
    if po:
        left = po["residual_calls_after_polishing"]
        checks.append(
            {
                "name": "no confident change left after polishing",
                "passed": all(v == 0 for v in left.values()),
                "detail": ", ".join(f"{k}: {v}" for k, v in left.items()),
            }
        )
    if cert.get("steps"):
        failed = [s["name"] for s in cert["steps"] if s["status"] != "completed"]
        skipped = cert.get("external_skipped") or {}
        checks.append(
            {
                "name": "all requested validation steps completed",
                "passed": not failed and not skipped,
                "detail": f"failed: {', '.join(failed) or 'none'}; "
                f"skipped external tools: {', '.join(skipped) or 'none'}",
            }
        )
    return checks


def _markdown(cert: dict) -> str:
    lines = [f"# Organelle reference certificate: {cert['sample']} ({cert['organelle']})", ""]
    lines += ["## Checks", "", "| check | result | detail |", "|---|---|---|"]
    for c in cert["checks"]:
        lines.append(f"| {c['name']} | {'pass' if c['passed'] else 'FAIL'} | {c['detail']} |")
    un = cert.get("unringing") or {}
    if un:
        lines += ["", "## Molecules", "", "| molecule | length | circular |", "|---|---|---|"]
        for i, m in enumerate(un.get("molecule_list", []), 1):
            lines.append(f"| {i} | {m['length']:,} | {m['circular']} |")
        lines.append(f"\nDecisive: {un.get('decisive')} (margin ln {un.get('margin_ln')})")
    mp = cert.get("multi_platform") or {}
    if mp:
        lines += ["", "## Branches (minority share per platform)", ""]
        plats = mp["platforms"]
        lines += ["| branch | " + " | ".join(plats) + " |", "|---|" + "---|" * len(plats)]
        for b, per in sorted(mp["branches"].items()):
            cells = []
            for p in plats:
                v = per.get(p)
                cells.append(
                    "-"
                    if not v or v["minority_fraction"] is None
                    else f"{100 * v['minority_fraction']:.1f}%"
                )
            lines.append(f"| {b} | " + " | ".join(cells) + " |")
        lines.append(
            "\nPhased paths: " + ", ".join(f"{k} {v}" for k, v in mp["phased_paths"].items())
        )
    cv = cert.get("cross_validation") or {}
    if cv.get("tools"):
        lines += [
            "",
            "## Cross-validation",
            "",
            "| graph | ovasm junctions it has | its junctions ovasm has |",
            "|---|---|---|",
        ]
        for t, v in cv["tools"].items():
            if "note" in v:
                lines.append(f"| {t} | {v['note']} | |")
            else:
                lines.append(
                    f"| {t} | {v['ovasm_junctions_found']}/{v['ovasm_junctions']} | "
                    f"{v['its_junctions_found_in_ovasm']}/{v['its_junctions']} |"
                )
    po = cert.get("polishing") or {}
    if po:
        lines += [
            "",
            "## Polishing",
            "",
            f"With {po['polished_with']} reads: {po['edits']} edits {po['edit_kinds']}; "
            f"confident calls left: {po['residual_calls_after_polishing']}",
        ]
    rc = cert.get("reference_comparison") or {}
    if rc:
        lines += [
            "",
            "## Against the earlier reference",
            "",
            f"{rc['reference']}: {rc['snvs']} SNVs, {rc['insertions']} insertions "
            f"({rc['insertion_bp']} bp), {rc['deletions']} deletions ({rc['deletion_bp']} bp), "
            f"{rc['indels_50bp_or_more']} indels >= 50 bp, {rc['primary_alignment_blocks']} "
            f"alignment blocks, {100 * rc['reference_aligned_fraction']:.2f}% of the reference aligned.",
        ]
    lines += ["", "## Steps", "", "| step | status | seconds |", "|---|---|---|"]
    for s in cert["steps"]:
        lines.append(f"| {s['name']} | {s['status']} | {s.get('seconds', '')} |")
    return "\n".join(lines) + "\n"


def run_gold_mode(g: GoldInputs) -> dict[str, Any]:
    """Run the gold mode; write and return the certificate."""
    if not (g.hifi or g.ont or g.clr or g.short):
        raise ValueError("gold mode needs reads")
    g.out_dir.mkdir(parents=True, exist_ok=True)
    run = _Run(g.out_dir)
    cert: dict[str, Any] = run.record
    cert.update(
        {
            "sample": g.sample,
            "organelle": g.organelle,
            "inputs": {k: [str(p) for p in getattr(g, k)] for k in ("hifi", "ont", "clr", "short")},
        }
    )
    reads = _recruit_and_correct(g, run)
    primary = "hifi" if "hifi" in reads else "noisy" if "noisy" in reads else "short"
    cert["primary_platform"] = primary
    gfa, asm = _build(reads, g, run, primary, "ovasm")
    cert["assembly"] = {k: asm.get(k) for k in ("k", "unitigs", "links", "total_length", "n50")}
    evidence = _evidence(gfa, reads, g, run, "ovasm")
    cert["multi_platform"] = _multi_platform(evidence)
    ev_path, ev_rep = evidence[primary]
    cert["evidence_primary"] = {
        "platform": primary,
        "links": len(ev_rep["links"]),
        "links_supported": ev_rep["links_supported"],
        "repeats": ev_rep["repeat_segments"],
    }
    d = gfa.parent
    lin = run.step(
        "unring",
        lambda: run_linearize(gfa, ev_path, out_fasta=d / "molecules.fa", out_json=d / "lin.json"),
    )
    molecules = [
        m for c in lin.get("components", []) if c.get("best") for m in c["best"]["molecules"]
    ]
    cert["unringing"] = {
        "molecules": lin.get("molecules"),
        "unsolved_components": lin.get("unsolved_components"),
        "decisive": lin.get("decisive"),
        "margin_ln": lin.get("margin_ln"),
        "molecule_list": [
            {"length": m["length"], "circular": m["circular"], "path": m["path"]} for m in molecules
        ],
    }
    run.step(
        "unify",
        lambda: run_unify(
            gfa,
            sample=g.sample,
            out_gfa=g.out_dir / f"{g.sample}.ov.gfa",
            out_json=g.out_dir / f"{g.sample}.ov.json",
            backend="ovasm",
            organelle=g.organelle,
            evidence_json=ev_path,
            linearize_json=d / "lin.json",
        ),
    )
    others = _external(g, reads, run)
    if "short" in reads and primary != "short":
        short_gfa = run.step(
            "assemble_short_check",
            lambda: _build(reads, g, run, "short", "ovasm_short")[0],
            fatal=False,
        )
        if short_gfa is not None:
            others["ovasm-short"] = short_gfa
    if others:
        cert["cross_validation"] = run.step(
            "cross_validation", lambda: _concordance(gfa, others), fatal=False
        )
    if (d / "molecules.fa").is_file():
        cert["polishing"] = run.step(
            "polish", lambda: _polish(d / "molecules.fa", reads, g, run), fatal=False
        )
        if g.reference is not None:
            final = (
                Path(cert["polishing"]["polished_fasta"])
                if cert.get("polishing")
                else d / "molecules.fa"
            )
            cert["reference_comparison"] = run.step(
                "reference_comparison", lambda: _compare_reference(final, g.reference), fatal=False
            )
    cert["checks"] = _checks(cert)
    (g.out_dir / "certificate.json").write_text(json.dumps(cert, indent=2, default=str) + "\n")
    (g.out_dir / "certificate.md").write_text(_markdown(cert))
    return cert


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="ovasm gold mode: organelle reference + certificate")
    p.add_argument("--sample", required=True)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--seed", action="append", default=[], help="NAME=FASTA (repeatable)")
    p.add_argument("--hifi", nargs="*", type=Path, default=[])
    p.add_argument("--ont", nargs="*", type=Path, default=[])
    p.add_argument("--clr", nargs="*", type=Path, default=[])
    p.add_argument("--short", nargs="*", type=Path, default=[])
    p.add_argument("--organelle", default="mitochondrion", choices=("mitochondrion", "plastid"))
    p.add_argument("--reference", type=Path, help="earlier reference to compare against")
    p.add_argument("--external", nargs="*", default=["oatk", "himt", "pmat", "getorganelle"])
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--target-depth", type=float, default=150.0)
    a = p.parse_args(argv)
    seeds: dict[str, list[Path]] = {}
    for spec in a.seed:
        name, path = spec.split("=", 1)
        seeds.setdefault(name, []).append(Path(path))
    cert = run_gold_mode(
        GoldInputs(
            sample=a.sample,
            out_dir=a.out,
            seeds=seeds,
            hifi=a.hifi,
            ont=a.ont,
            clr=a.clr,
            short=a.short,
            organelle=a.organelle,
            reference=a.reference,
            external=a.external,
            threads=a.threads,
            target_depth=a.target_depth,
        )
    )
    for c in cert["checks"]:
        print(f"[{'pass' if c['passed'] else 'FAIL'}] {c['name']}: {c['detail']}")


if __name__ == "__main__":
    main()
