"""Chloroplast inverted-repeat boundary analysis (IRscope-equivalent).

Locates the four junction sites (JLA = LSC/IRa, JLB = IRb/LSC, JSA = SSC/IRa,
JSB = IRb/SSC) from GenBank repeat_region features, reports the genes flanking
each junction, and compares contraction/expansion across genomes. SVG output is
handled by ``write_boundary_svg``.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, cast

from ..annotation.genbank import parse_genbank
from ..core.artifacts import ArtifactRef
from ..core.errors import OrganelleInputError
from ..core.frozen import thaw_json
from ..core.genome import OrganelleGenome
from ..core.provenance import ResultProvenance
from ..core.result import ErrorDetail, Finding, OrganelleResult

_OPERATION_VERSION = "1.0"


def _package_version() -> str:
    try:
        return version("organelleverse")
    except PackageNotFoundError:
        return "0.0.1"


def _sha256_json(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _provenance(
    *,
    parameters: dict[str, object],
    started_at: datetime,
    finished_at: datetime,
    input_object_ids: tuple[str, ...] = (),
    input_artifact_hashes: tuple[str, ...] = (),
) -> ResultProvenance:
    return ResultProvenance(
        operation_id="ir_boundary.ir_boundary",
        operation_version=_OPERATION_VERSION,
        package_version=_package_version(),
        git_commit=os.environ.get("ORG_VERSE_GIT_COMMIT", ""),
        input_object_ids=input_object_ids,
        input_artifact_hashes=input_artifact_hashes,
        parameters_hash=_sha256_json(parameters),
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=max(0.0, (finished_at - started_at).total_seconds()),
    )


def ir_boundary(
    genomes: list[OrganelleGenome],
) -> OrganelleResult:
    """Analyze IR junction boundaries across ≥1 chloroplast genome, with sequence-level IR fallback.

    IRs are read from annotated repeat features only when plausible (both
    arms >= ~400 bp); otherwise the pair is detected from the sequence itself
    (revcomp seed-chaining, IRscope-style), because most modern NCBI plastid
    records carry no repeat_region features. A genuinely IR-less genome
    (Medicago NC_003119, papilionoid IR loss) honestly reports ir_length 0.

    For each genome, parse repeat_region features to locate IRa/IRb coordinates,
    infer LSC/SSC, and identify genes at each of the 4 junctions. When ≥2
    genomes are given, reports contraction/expansion events.
    """
    started_at = datetime.now(UTC)
    if not genomes:
        return OrganelleResult(
            operation_id="ir_boundary.ir_boundary",
            operation_version=_OPERATION_VERSION,
            scope="plastid",
            status="failed",
            summary_text="ir_boundary() requires >=1 genome.",
            provenance=_provenance(
                parameters={},
                started_at=started_at,
                finished_at=datetime.now(UTC),
            ),
            errors=(
                ErrorDetail(
                    code="ir_boundary.no_genomes",
                    message="ir_boundary() requires at least one OrganelleGenome.",
                ),
            ),
        )
    per_genome = [_analyze_one(g) for g in genomes]
    events: list[dict] = []
    if len(genomes) >= 2:
        events = _compare_boundaries(per_genome)
    ir_lengths = [pg["ir_length"] for pg in per_genome if pg["ir_length"]]
    lsc_lengths = [pg["lsc_length"] for pg in per_genome if pg["lsc_length"]]
    ssc_lengths = [pg["ssc_length"] for pg in per_genome if pg["ssc_length"]]
    return OrganelleResult(
        operation_id="ir_boundary.ir_boundary",
        operation_version=_OPERATION_VERSION,
        scope="plastid",
        status="ok",
        summary_text=(
            f"Analyzed IR boundaries for {len(genomes)} genome(s); "
            f"{len(events)} contraction/expansion event(s)."
        ),
        metrics={
            "genome_count": len(genomes),
            "ir_lengths": ir_lengths,
            "lsc_lengths": lsc_lengths,
            "ssc_lengths": ssc_lengths,
            "contraction_expansion_events": len(events),
            "events": events,
            "per_genome": per_genome,
        },
        findings=tuple(
            Finding(
                code="ir_boundary.ir_length",
                metric=f"ir_length.{pg['name']}",
                value=pg["ir_length"],
                unit="bp",
            )
            for pg in per_genome
        )
        + tuple(
            Finding(
                code=f"ir_boundary.{e['type']}",
                metric=f"ir_length_delta.{e['genome_a']}_vs_{e['genome_b']}",
                value=e["ir_length_delta"],
                unit="bp",
            )
            for e in events[:5]
        ),
        flags=("ir_detected",) if any(pg["ir_length"] for pg in per_genome) else (),
        provenance=_provenance(
            parameters={"genome_count": len(genomes)},
            input_object_ids=tuple(g.object_id for g in genomes),
            input_artifact_hashes=tuple(
                g.annotation.sha256 for g in genomes if g.annotation is not None
            ),
            started_at=started_at,
            finished_at=datetime.now(UTC),
        ),
    )


def materialize_result(result: OrganelleResult, output: str | Path) -> OrganelleResult:
    """Canonical write boundary for ``ir_boundary.*`` results.

    Never recomputes: it renders the SVG from the metrics already carried by a
    successful Result and returns that Result with the SVG artifact attached.
    """

    if not isinstance(result, OrganelleResult):
        raise OrganelleInputError(
            code="ir_boundary.unsupported_value",
            message="ir_boundary.write accepts a canonical OrganelleResult",
            details={"type": type(result).__name__},
        )
    if result.operation_id != "ir_boundary.ir_boundary":
        raise OrganelleInputError(
            code="ir_boundary.unsupported_operation",
            message="ir_boundary.write accepts Results from ir_boundary.ir_boundary",
            details={"operation_id": result.operation_id},
        )
    if result.status == "failed":
        raise OrganelleInputError(
            code="ir_boundary.failed_result",
            message="ir_boundary.write requires a successful ir_boundary result",
            details={"operation_id": result.operation_id},
        )
    path = write_boundary_svg(result, output)
    artifact = ArtifactRef.from_path(
        path,
        kind="ir_boundary_svg",
        format="svg",
        media_type="image/svg+xml",
    )
    return result.evolve(artifacts=(*result.artifacts, artifact))


def write_boundary_svg(
    result: OrganelleResult | Mapping[str, Any],
    output: str | Path,
) -> Path:
    """Write an IRscope-style SVG from an ``ir_boundary()`` result."""
    if isinstance(result, OrganelleResult):
        metrics = cast(dict[str, Any], thaw_json(result.metrics))
    else:
        metrics = dict(result)
    per_genome = list(metrics.get("per_genome", ()))
    if not per_genome:
        raise ValueError("write_boundary_svg() requires an ir_boundary result with per_genome data")
    return _write_svg(Path(output), per_genome)


# ---------------------------------------------------------------------------
# per-genome analysis
# ---------------------------------------------------------------------------


def _analyze_one(g: OrganelleGenome) -> dict:
    records = parse_genbank(g.annotation.resolve()).records if g.annotation is not None else ()
    name = g.metadata.species or "genome"
    total_len = sum(len(record.sequence) for record in records)
    ir_regions: list[dict] = []
    genes: list[dict] = []
    for record in records:
        for feature in record.features:
            ft = feature.type.casefold()
            bounding_start = min(part.start for part in feature.parts)
            bounding_end = max(part.end for part in feature.parts)
            if ft in ("repeat_region", "inverted_repeat", "repeat"):
                ir_regions.append(
                    {
                        "start": bounding_start,
                        "end": bounding_end,
                        "strand": feature.parts[0].strand,
                    }
                )
            elif ft == "cds":
                values = feature.qualifier_values("gene")
                if values and values[0]:
                    genes.append(
                        {
                            "name": values[0],
                            "start": bounding_start,
                            "end": bounding_end,
                            "strand": feature.parts[0].strand,
                        }
                    )
    # infer IR coordinates from repeat regions
    ira, irb = _infer_ir_pair(ir_regions, total_len)
    sequence = "".join(record.sequence for record in records).upper()
    # Modern NCBI plastid records usually carry no repeat_region features (25
    # of the 31 packaged references), and the few that do may annotate small
    # tandem repeats instead of the true IR (Oryza 66 bp). Fall back to
    # sequence-level inverted-repeat detection whenever the feature-based pair
    # is missing or implausibly short.
    feature_ir_length = min(
        (r["end"] - r["start"] for r in (ira, irb) if r), default=0
    )
    if sequence and (ira is None or feature_ir_length < _MIN_PLAUSIBLE_IR):
        detected = _detect_ir_from_sequence(sequence)
        if detected is not None:
            ira, irb = detected
    ir_length = 0
    lsc_length = 0
    ssc_length = 0
    junctions: dict[str, dict] = {}
    if ira and irb:
        ir_length = ira["end"] - ira["start"]
        # typical cp layout: LSC ... IRb ... SSC ... IRa (circular)
        irb_start, irb_end = irb["start"], irb["end"]
        ira_start, ira_end = ira["start"], ira["end"]
        # circular cp layout: LSC wraps from IRa_end around origin to IRb_start
        lsc_length = (total_len - ira_end + irb_start) if ira_end < total_len else irb_start
        ssc_length = max(0, ira_start - irb_end)  # SSC = irb_end..ira_start
        junctions = {
            "JLB": _gene_at(genes, irb_start),  # IRb/LSC
            "JSB": _gene_at(genes, irb_end),  # IRb/SSC
            "JSA": _gene_at(genes, ira_start),  # SSC/IRa
            "JLA": _gene_at(genes, ira_end),  # IRa/LSC
        }
    return {
        "name": name,
        "total_length": total_len,
        "ir_length": ir_length,
        "lsc_length": lsc_length,
        "ssc_length": ssc_length,
        "junctions": junctions,
        "genes": genes,
    }


def _infer_ir_pair(ir_regions: list[dict], total_len: int) -> tuple[dict | None, dict | None]:
    """Infer the IRa/IRb pair from repeat_region features.

    Plant cp has two IR copies (inverted). We pick the two largest repeat regions.
    """
    if len(ir_regions) < 2:
        # fallback: no IR annotated
        return None, None
    sorted_ir = sorted(ir_regions, key=lambda r: r["end"] - r["start"], reverse=True)
    irb, ira = sorted_ir[0], sorted_ir[1]
    # ensure irb comes before ira in coordinates
    if irb["start"] > ira["start"]:
        irb, ira = ira, irb
    return ira, irb


#: Below this length an annotated repeat pair is more plausibly a small tandem
#: repeat than the true IR (Pinaceae's highly reduced real IR is ~0.5 kb, so
#: the threshold must stay under that).
_MIN_PLAUSIBLE_IR = 400

#: Sequence-level IR detection parameters: exact k-mer seeds on one side, a
#: coarser probe stride on the other (a shared stride on both sides can alias
#: against the pair's offset invariant and miss the IR entirely).
_IR_SEED = 63
_IR_INDEX_STRIDE = 1
_IR_PROBE_STRIDE = 7
_IR_MAX_RUN_MISMATCH = 4
_IR_SEED_GAP = 500
_IR_MIN_SEED_COVERAGE = 0.4

_COMPLEMENT = str.maketrans("ACGTN", "TGCAN")


def _detect_ir_from_sequence(
    sequence: str,
    *,
    min_ir: int = _MIN_PLAUSIBLE_IR,
    circular: bool = False,
) -> tuple[dict, dict] | None:
    """Locate the largest inverted-repeat pair by revcomp self-seeding.

    Seeds exact k-mer matches between the sequence and its reverse complement,
    chains them on a shared offset invariant, then extends the chain outward
    while identity stays above ``_IR_MIN_IDENTITY``. Returns the two arms as
    coordinate dicts (IRb first); ``None`` when no pair reaches ``min_ir``
    (genuinely IR-less genomes such as the papilionoid IR-loss lineages).
    With ``circular=True``, coordinates refer to three concatenated turns;
    both returned arms are complete and may be reduced modulo input length.
    """
    # Three turns contain both full copies for any input origin. Reject
    # terminal fragments below instead of selecting a truncated first chain.
    circle_length = len(sequence) if circular else None
    if circular:
        sequence = sequence * 3
    n = len(sequence)
    if n < 2 * min_ir + _IR_SEED:
        return None
    reverse = sequence.translate(_COMPLEMENT)[::-1]
    table: dict[str, list[int]] = {}
    for i in range(0, n - _IR_SEED + 1, _IR_INDEX_STRIDE):
        kmer = sequence[i : i + _IR_SEED]
        if "N" in kmer:
            continue
        table.setdefault(kmer, []).append(i)
    if not table:
        return None

    # A reverse-complement match between arms [a1,a2) and [b1,b2) preserves
    # the invariant a1 + b2 == a2 + b1; bucket seed pairs by that constant.
    chains: dict[int, list[tuple[int, int]]] = {}
    for j in range(0, n - _IR_SEED + 1, _IR_PROBE_STRIDE):
        kmer = reverse[j : j + _IR_SEED]
        hits = table.get(kmer)
        if not hits:
            continue
        # reverse[j:j+seed] == sequence[i:i+seed] pairs arm A at i with arm B
        # covering [n-j-seed, n-j).
        for i in hits:
            if abs(i - (n - j - _IR_SEED)) < 5 * _IR_SEED:
                continue  # self-matches near the diagonal are not an IR pair
            chains.setdefault(i + (n - j), []).append((i, n - j - _IR_SEED))
    if not chains:
        return None

    # The offset invariant is symmetric, so one chain contains BOTH copies of
    # the repeat (each copy seeds once as arm A and once as arm B). Rank
    # chains by seed count, then split on large a-coordinate gaps: inside one
    # copy exact seeds are dense; the copies are separated by the SSC span.
    for constant, pairs in sorted(chains.items(), key=lambda kv: -len(kv[1]))[:4]:
        if len(pairs) < 2:
            continue
        pairs.sort()
        groups: list[list[tuple[int, int]]] = [[pairs[0]]]
        for previous, current in zip(pairs, pairs[1:]):
            if current[0] - previous[0] > _IR_SEED_GAP:
                groups.append([])
            groups[-1].append(current)
        for group in groups:
            if len(group) < 2:
                continue
            a_start = min(a for a, _ in group)
            a_end = max(a for a, _ in group) + _IR_SEED
            b_start = min(b for _, b in group)
            b_end = max(b for _, b in group) + _IR_SEED
            # Exact-seed coverage replaces an extension-identity gate: a real
            # IR is near-identical, so seeds blanket it; random offset
            # collisions cannot.
            coverage = len(group) * _IR_SEED / max(1, a_end - a_start)
            if coverage < _IR_MIN_SEED_COVERAGE:
                continue
            (start_a, end_a), (start_b, end_b) = _extend_arm(
                sequence, a_start, a_end, b_start, b_end
            )
            length = min(end_a - start_a, end_b - start_b)
            if length < min_ir:
                continue
            if circle_length is not None:
                if min(start_a, start_b) == 0 or max(end_a, end_b) == n:
                    continue
                if length >= circle_length or abs(start_a - start_b) >= circle_length:
                    continue
            first, second = sorted(
                ({"start": start_a, "end": end_a}, {"start": start_b, "end": end_b}),
                key=lambda r: r["start"],
            )
            return second, first  # (IRa, IRb) by _infer_ir_pair's convention
    return None


def _extend_arm(
    sequence: str,
    a_start: int,
    a_end: int,
    b_start: int,
    b_end: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Grow one copy's seed span outward until mismatches run out.

    Arm B runs opposite to arm A under the offset invariant: extending A
    leftward pairs with extending B rightward, and vice versa. Stops after
    ``_IR_MAX_RUN_MISMATCH`` consecutive mismatches on either side, which
    lands within a few bases of the true single-copy junction.
    """
    n = len(sequence)
    a, b = a_start - 1, b_end
    run = 0
    while a >= 0 and b < n and run <= _IR_MAX_RUN_MISMATCH:
        run = 0 if sequence[a] == sequence[b].translate(_COMPLEMENT) else run + 1
        a -= 1
        b += 1
    start_a, end_b = a + run + 1, b - run

    a, b = a_end, b_start - 1
    run = 0
    while a < n and b >= 0 and run <= _IR_MAX_RUN_MISMATCH:
        run = 0 if sequence[a] == sequence[b].translate(_COMPLEMENT) else run + 1
        a += 1
        b -= 1
    end_a, start_b = a - run, b + run + 1
    return (start_a, end_a), (start_b, end_b)


def _gene_at(genes: list[dict], pos: int, window: int = 500) -> dict:
    """Find the gene nearest to a junction position."""
    best = None
    best_dist = window
    for g in genes:
        center = (g["start"] + g["end"]) // 2
        dist = abs(center - pos)
        if dist < best_dist:
            best_dist = dist
            best = g
    if best:
        return {
            "gene": best["name"],
            "distance": best_dist,
            "start": best["start"],
            "end": best["end"],
        }
    return {"gene": "intergenic", "distance": -1}


def _compare_boundaries(per_genome: list[dict]) -> list[dict]:
    """Detect IR contraction/expansion events between genomes."""
    events = []
    ref = per_genome[0]
    for _i, pg in enumerate(per_genome[1:], 1):
        if ref["ir_length"] and pg["ir_length"]:
            delta = pg["ir_length"] - ref["ir_length"]
            if abs(delta) > 50:
                events.append(
                    {
                        "genome_a": ref["name"],
                        "genome_b": pg["name"],
                        "ir_length_delta": delta,
                        "type": "expansion" if delta > 0 else "contraction",
                    }
                )
    return events


# ---------------------------------------------------------------------------
# SVG output
# ---------------------------------------------------------------------------


def _write_svg(path: Path, per_genome: list[dict]) -> Path:
    """Draw a simple IR boundary comparison SVG (IRscope-style)."""
    h = 120 * len(per_genome) + 60
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="700" height="{h}">',
        '<text x="350" y="25" text-anchor="middle" font-size="14">IR Boundary Comparison</text>',
    ]
    y = 50
    for pg in per_genome:
        parts.append(f'<text x="10" y="{y}" font-size="11">{pg["name"]}</text>')
        # draw 4-quadrant bar: LSC (green), IRb (blue), SSC (orange), IRa (blue)
        total = pg["total_length"] or 1
        x = 10
        w_scale = 660 / total
        regions = [
            ("LSC", pg["lsc_length"] or 0, "#90EE90"),
            ("IRb", pg["ir_length"] or 0, "#87CEEB"),
            ("SSC", pg["ssc_length"] or 0, "#FFA500"),
            ("IRa", pg["ir_length"] or 0, "#87CEEB"),
        ]
        for label, length, color in regions:
            w = max(2, int(length * w_scale))
            parts.append(
                f'<rect x="{x}" y="{y + 5}" width="{w}" height="30" fill="{color}" stroke="black"/>'
            )
            parts.append(
                f'<text x="{x + w // 2}" y="{y + 50}" text-anchor="middle" font-size="9">{label}</text>'
            )
            x += w
        y += 70
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts))
    return path
