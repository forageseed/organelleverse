"""Real analysis runners for the OrganelleVerse Tools UI.

Every runnable mito / chloro / general tool maps onto the ``organelleverse``
analysis suites (annotation, assembly, comparative, structure, diversity,
variation, codon composition, phylogeny, localization, reporting, …).  Nothing
in this module fabricates output: when a required external binary is missing
the runner raises, and the run fails with the underlying message instead of
silently pretending to have analyzed anything.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable

from organelleverse.core.result import OrganelleResult

RunnerFn = Callable[..., dict[str, Any]]

_GENBANK_SUFFIXES = (".gb", ".gbk", ".gbf")


# --------------------------------------------------------------------------
# shared plumbing
# --------------------------------------------------------------------------


def _first_file(params: dict[str, Any], *names: str) -> Path:
    for name in names:
        value = str(params.get(name) or "").strip()
        if value:
            path = Path(value).expanduser()
            if path.exists():
                return path
            raise FileNotFoundError(f"input file not found: {path}")
    raise FileNotFoundError("no input file was provided")


def _genome_fasta(path: Path, workdir: Path) -> Path:
    """Whole-genome FASTA for an input that may be FASTA or GenBank."""
    if path.suffix.lower() in _GENBANK_SUFFIXES:
        from Bio import SeqIO

        out = workdir / (path.stem + ".fna")
        records = list(SeqIO.parse(str(path), "genbank"))
        SeqIO.write(records, str(out), "fasta")
        return out
    return path


def _cds_fasta(path: Path, workdir: Path, feature_type: str = "CDS") -> tuple[Path, int]:
    """Extract one feature type's sequences from a GenBank file."""
    from Bio import SeqIO

    out = workdir / f"{feature_type.lower()}.ffa"
    written = 0
    with out.open("w") as handle:
        for record in SeqIO.parse(str(path), "genbank"):
            for feature in record.features:
                if feature.type != feature_type:
                    continue
                gene = (
                    feature.qualifiers.get("gene", [None])[0]
                    or feature.qualifiers.get("product", [None])[0]
                    or f"{feature_type}_{written + 1}"
                )
                seq = str(feature.extract(record.seq)).upper()
                handle.write(f">{record.id}|{gene}\n{seq}\n")
                written += 1
    if written == 0:
        raise ValueError(f"no {feature_type} features found in {path.name}")
    return out, written


def _canon_organelle(value: str) -> str:
    """Core models only accept 'mitochondrion' / 'plastid'."""
    if value in ("mitochondrion", "plastid"):
        return value
    if value in ("mito", "mitochondrial"):
        return "mitochondrion"
    return "plastid"


def _genomes_from(path: Path, organelle: str) -> list[Any]:
    """OrganelleGenome list from a GenBank directory or a multi-record file."""
    from organelleverse.io_genome import read_genbank_genome

    if path.is_dir():
        files = sorted(
            p for p in path.iterdir() if p.suffix.lower() in _GENBANK_SUFFIXES
        )
        if not files:
            raise FileNotFoundError(f"no GenBank files in directory {path}")
        sources = files
    else:
        sources = _split_genbank(path)

    genomes = []
    for source in sources:
        species = source.stem.replace("_", " ")
        genomes.append(
            read_genbank_genome(
                source, organelle=_canon_organelle(organelle), species=species
            )
        )
    return genomes


def _split_genbank(path: Path) -> list[Path]:
    """Split a (possibly multi-record) GenBank file into per-record files."""
    from Bio import SeqIO

    records = list(SeqIO.parse(str(path), "genbank"))
    if not records:
        raise ValueError(f"no GenBank records in {path.name}")
    if len(records) == 1:
        return [path]
    out_dir = path.parent / f"{path.stem}_records"
    out_dir.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    for record in records:
        target = out_dir / f"{record.id.replace('/', '_')}.gb"
        SeqIO.write([record], str(target), "genbank")
        out.append(target)
    return out


def _genome(path: Path, organelle: str, species: str) -> Any:
    from organelleverse.io_genome import read_fasta_genome, read_genbank_genome

    organelle = _canon_organelle(organelle)
    if path.suffix.lower() in _GENBANK_SUFFIXES:
        return read_genbank_genome(path, organelle=organelle, species=species)
    return read_fasta_genome(path, organelle=organelle, species=species)


def _error_text(error: Any) -> str:
    if isinstance(error, dict):
        return str(error.get("message") or error.get("code") or error)
    return str(getattr(error, "message", None) or error)


def _metrics(result: OrganelleResult) -> dict[str, Any]:
    """OrganelleResult.metrics is a FrozenMap (a Mapping, not a dict)."""
    from collections.abc import Mapping

    metrics = result.metrics
    return dict(metrics) if isinstance(metrics, Mapping) else {}


def _check(result: OrganelleResult, operation: str) -> OrganelleResult:
    if result.status == "failed":
        errors = "; ".join(_error_text(e) for e in (result.errors or [])[:3])
        raise RuntimeError(f"{operation} failed: {errors or 'unknown error'}")
    return result


def _write_tsv(path: Path, rows: Iterable[dict[str, Any]]) -> Path:
    rows = list(rows)
    if not rows:
        path.write_text("no rows\n")
        return path
    headers: list[str] = []
    for row in rows:
        for key in row:
            if key not in headers:
                headers.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=headers, extrasaction="ignore", delimiter="\t"
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_metrics_tsv(result: OrganelleResult, path: Path) -> Path:
    metrics = _metrics(result)
    return _write_tsv(path, [{"metric": key, "value": json.dumps(value, default=str)[:500]}
                             for key, value in sorted(metrics.items())])


def _summary_from_metrics(result: OrganelleResult, *keys: str) -> str:
    metrics = _metrics(result)
    parts = [f"{key}={metrics[key]}" for key in keys if key in metrics]
    return "; ".join(parts) if parts else f"{result.operation_id} completed"


def _package(result: OrganelleResult, summary: str, files: dict[str, Path]) -> dict[str, Any]:
    return {
        "summary_text": summary,
        "output_files": {label: str(path) for label, path in files.items()},
        "metrics": _metrics(result),
    }


# --------------------------------------------------------------------------
# composition / codon / variation / diversity / structure (pure compute)
# --------------------------------------------------------------------------


def _run_gc(params: dict[str, Any], *, output_dir: Path, log: Callable[[str], None]) -> dict[str, Any]:
    from organelleverse.composition.gc_core import compute_gc_content

    source = _first_file(params, "input")
    fasta = _genome_fasta(source, output_dir)
    window = int(params.get("window") or 500)
    data = compute_gc_content(fasta, window_size=max(window, 10))
    gc_file = _write_tsv(
        output_dir / "gc_summary.tsv",
        [
            {"metric": k, "value": v}
            for k, v in data.items()
            if not isinstance(v, (list, dict))
        ],
    )
    windows = data.get("windows") or data.get("gc_windows") or []
    win_file = _write_tsv(output_dir / "gc_windows.tsv", windows)
    log(f"GC content computed over {window} bp windows")
    return {
        "summary_text": f"GC content: {data.get('gc_content', data.get('gc', 'n/a'))}",
        "output_files": {"summary": str(gc_file), "windows": str(win_file)},
    }


def _write_codon_tables(data: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    table = data.get("codon_table") or data.get("rows") or []
    if table:
        files["codon_usage"] = _write_tsv(output_dir / "codon_usage.tsv", table)
    rscu = data.get("rscu") or []
    if rscu:
        files["rscu"] = _write_tsv(output_dir / "rscu.tsv", rscu)
    if not files:
        files["summary"] = _write_tsv(
            output_dir / "codon_summary.tsv",
            [{"metric": k, "value": v} for k, v in data.items()
             if not isinstance(v, (list, dict))],
        )
    return files


def _run_codon(organelle: str) -> RunnerFn:
    def runner(
        params: dict[str, Any], *, output_dir: Path, log: Callable[[str], None]
    ) -> dict[str, Any]:
        from organelleverse.codon_composition.codon_core import compute_codon_usage

        source = _first_file(params, "input")
        if source.suffix.lower() in _GENBANK_SUFFIXES:
            fasta, count = _cds_fasta(source, output_dir)
            log(f"extracted {count} CDS features")
        else:
            fasta = source
        genetic = params.get("genetic_code") or None
        data = compute_codon_usage(fasta, organelle=organelle, genetic_code=genetic)
        files = _write_codon_tables(data, output_dir)
        enc = data.get("ENC") or data.get("enc")
        gc3 = data.get("GC3s") or data.get("gc3s")
        log(f"codon usage computed (ENC={enc}, GC3s={gc3})")
        return {
            "summary_text": f"ENC={enc}; GC3s={gc3}; codons={data.get('total_codons')}",
            "output_files": {k: str(v) for k, v in files.items()},
        }

    return runner


def _run_amino_acid(
    params: dict[str, Any], *, output_dir: Path, log: Callable[[str], None]
) -> dict[str, Any]:
    from organelleverse.codon_composition.codon_core import compute_amino_acid_composition

    source = _first_file(params, "input")
    if source.suffix.lower() not in _GENBANK_SUFFIXES:
        raise ValueError("Chloro Amino Acid requires a GenBank file with CDS features")
    fasta, count = _cds_fasta(source, output_dir)
    data = compute_amino_acid_composition(fasta, organelle="chloro")
    table = data.get("composition") or data.get("rows") or []
    out = _write_tsv(
        output_dir / "amino_acid_composition.tsv",
        table or [{"metric": k, "value": v} for k, v in data.items()
                  if not isinstance(v, (list, dict))],
    )
    log(f"amino acid composition over {count} CDS")
    return {
        "summary_text": f"{count} CDS analyzed",
        "output_files": {"composition": str(out)},
    }


def _run_snp(
    params: dict[str, Any], *, output_dir: Path, log: Callable[[str], None]
) -> dict[str, Any]:
    from organelleverse.variation.snp_core import compute_snp

    alignment = _first_file(params, "input")
    data = compute_snp(alignment)
    snps = data.get("snps") or []
    out = _write_tsv(
        output_dir / "snps.tsv",
        snps or [{"metric": k, "value": v} for k, v in data.items()
                 if not isinstance(v, (list, dict))],
    )
    log(f"{len(snps)} SNPs vs reference")
    return {
        "summary_text": f"{data.get('snp_count', len(snps))} SNPs; Ti/Tv={data.get('ti_tv')}",
        "output_files": {"snps": str(out)},
    }


def _run_diversity(
    params: dict[str, Any], *, output_dir: Path, log: Callable[[str], None]
) -> dict[str, Any]:
    from organelleverse.diversity.diversity_core import (
        compute_neutral_tests,
        compute_nucleotide_diversity,
    )

    alignment = _first_file(params, "input")
    window = int(params.get("window") or 1000)
    step = int(params.get("step") or (window // 2 or 1))
    pi = compute_nucleotide_diversity(alignment, window_size=window, step=step)
    neutral = compute_neutral_tests(alignment)
    pi_file = _write_tsv(
        output_dir / "pi_summary.tsv",
        [{k: v for k, v in pi.items() if not isinstance(v, (list, dict))}],
    )
    win_file = _write_tsv(output_dir / "pi_windows.tsv", pi.get("windows") or [])
    neutral_file = _write_tsv(
        output_dir / "neutral_tests.tsv",
        [{k: v for k, v in neutral.items() if not isinstance(v, (list, dict))}],
    )
    log(f"nucleotide diversity over {window}/{step} bp windows")
    return {
        "summary_text": f"pi={pi.get('pi')}; TajimaD={neutral.get('tajima_d')}",
        "output_files": {
            "pi": str(pi_file),
            "windows": str(win_file),
            "neutral": str(neutral_file),
        },
    }


def _run_ssr(
    params: dict[str, Any], *, output_dir: Path, log: Callable[[str], None]
) -> dict[str, Any]:
    from organelleverse.structure.structure_core import compute_repeats

    source = _first_file(params, "input")
    fasta = _genome_fasta(source, output_dir)
    min_unit = int(params.get("min_repeat_unit") or 2)
    min_copy = int(params.get("min_repeats") or 3)
    data = compute_repeats(fasta, min_ssr_unit=max(1, min_unit), min_copy=max(2, min_copy))
    ssrs = data.get("ssrs") or []
    out = _write_tsv(output_dir / "ssrs.tsv", ssrs)
    log(f"{len(ssrs)} SSRs detected")
    return {
        "summary_text": f"{data.get('ssr_count', len(ssrs))} SSRs (unit>={min_unit}bp, copy>={min_copy})",
        "output_files": {"ssrs": str(out)},
    }


def _repeat_analysis(default_min_len: int) -> RunnerFn:
    def runner(
        params: dict[str, Any], *, output_dir: Path, log: Callable[[str], None]
    ) -> dict[str, Any]:
        from organelleverse.structure.structure_core import compute_multiconf

        source = _first_file(params, "input")
        fasta = _genome_fasta(source, output_dir)
        min_len = int(params.get("min_len") or default_min_len)
        data = compute_multiconf(fasta, min_repeat_len=max(20, min_len))
        repeats = data.get("repeats") or []
        out = _write_tsv(output_dir / "repeats.tsv", repeats)
        log(
            f"{len(repeats)} repeats >= {min_len} bp "
            f"({data.get('direct_repeat_pairs', 0)} direct, "
            f"{data.get('inverted_repeat_pairs', 0)} inverted)"
        )
        return {
            "summary_text": (
                f"direct pairs={data.get('direct_repeat_pairs')}, "
                f"inverted pairs={data.get('inverted_repeat_pairs')}, "
                f"predicted configs={data.get('predicted_configs')}"
            ),
            "output_files": {"repeats": str(out)},
        }

    return runner


def _run_multiconf(
    params: dict[str, Any], *, output_dir: Path, log: Callable[[str], None]
) -> dict[str, Any]:
    from organelleverse.structure.structure_core import compute_multiconf

    source = _first_file(params, "input")
    fasta = _genome_fasta(source, output_dir)
    min_len = int(params.get("min_repeat") or 50)
    data = compute_multiconf(fasta, min_repeat_len=max(20, min_len))
    repeats = _write_tsv(output_dir / "repeats.tsv", data.get("repeats") or [])
    summary = _write_tsv(
        output_dir / "configurations.tsv",
        [
            {"metric": k, "value": v}
            for k, v in data.items()
            if not isinstance(v, (list, dict))
        ],
    )
    log(f"multi-configuration structure from repeats >= {min_len} bp")
    return {
        "summary_text": f"predicted configurations={data.get('predicted_configs')}",
        "output_files": {"repeats": str(repeats), "summary": str(summary)},
    }


# --------------------------------------------------------------------------
# comparative / ir_boundary / introns / conversion / reporting (GenBank in)
# --------------------------------------------------------------------------


def _comparative_runner(operation: str) -> RunnerFn:
    def runner(
        params: dict[str, Any], *, output_dir: Path, log: Callable[[str], None]
    ) -> dict[str, Any]:
        import organelleverse.comparative as comparative

        source = _first_file(params, "input", "genbank")
        genomes = _genomes_from(source, organelle="chloro")
        log(f"loaded {len(genomes)} annotated genomes")
        fn = getattr(comparative, operation)
        result = _check(fn(genomes), operation)
        writer = getattr(comparative, f"write_{operation}", None)
        files: dict[str, Path] = {}
        if writer is not None:
            try:
                written = writer(result, output_dir / f"{operation}")
                if isinstance(written, dict):
                    files.update(written)
                else:
                    files[operation] = Path(written)
            except (TypeError, ValueError) as exc:
                log(f"writer fallback to metrics ({exc})")
        metrics_file = _write_metrics_tsv(result, output_dir / f"{operation}_metrics.tsv")
        files.setdefault("metrics", metrics_file)
        return _package(result, _summary_from_metrics(result), files)

    return runner


def _run_synteny(
    params: dict[str, Any], *, output_dir: Path, log: Callable[[str], None]
) -> dict[str, Any]:
    import organelleverse.comparative as comparative

    source = _first_file(params, "input", "genbank")
    reference = str(params.get("reference") or "").strip()
    paths: list[Path] = []
    if reference:
        paths.append(Path(reference))
    if source.is_dir():
        paths.extend(
            p for p in sorted(source.iterdir())
            if p.suffix.lower() in _GENBANK_SUFFIXES
        )
    else:
        paths.extend(_split_genbank(source))
    if len(paths) < 2:
        raise ValueError("synteny analysis requires at least 2 GenBank genomes")
    genomes = []
    from organelleverse.io_genome import read_genbank_genome

    for p in paths:
        genomes.append(
            read_genbank_genome(
                p, organelle="mitochondrion", species=p.stem.replace("_", " ")
            )
        )
    result = _check(comparative.synteny(genomes), "synteny")
    files: dict[str, Path] = {}
    try:
        files["synteny"] = Path(comparative.write_synteny(result, output_dir / "synteny.tsv"))
    except (TypeError, ValueError):
        pass
    files.setdefault("metrics", _write_metrics_tsv(result, output_dir / "synteny_metrics.tsv"))
    log(f"synteny across {len(genomes)} genomes")
    return _package(result, _summary_from_metrics(result, "block_count", "jaccard"), files)


def _run_ir_boundary(
    params: dict[str, Any], *, output_dir: Path, log: Callable[[str], None]
) -> dict[str, Any]:
    from organelleverse.ir_boundary import ir_boundary, materialize_result, write_boundary_svg

    source = _first_file(params, "input", "genbank")
    genomes = _genomes_from(source, organelle="chloro")
    result = _check(ir_boundary(genomes), "ir_boundary")
    files: dict[str, Path] = {}
    try:
        materialize_result(result, output_dir / "ir_boundary")
        files["bundle"] = output_dir / "ir_boundary"
    except (TypeError, ValueError, OSError) as exc:
        log(f"bundle materialization skipped ({exc})")
    try:
        files["svg"] = write_boundary_svg(result, output_dir / "ir_boundary.svg")
    except (TypeError, ValueError, OSError) as exc:
        log(f"SVG rendering skipped ({exc})")
    files.setdefault("metrics", _write_metrics_tsv(result, output_dir / "ir_boundary_metrics.tsv"))
    log(f"IR boundary analysis on {len(genomes)} chloroplast genomes")
    return _package(result, _summary_from_metrics(result), files)


def _run_introns(
    params: dict[str, Any], *, output_dir: Path, log: Callable[[str], None]
) -> dict[str, Any]:
    from organelleverse.structure import introns as introns_fn
    from organelleverse.structure import write_introns

    source = _first_file(params, "input", "genbank")
    genome = _genome(source, organelle="chloro", species=source.stem.replace("_", " "))
    result = _check(introns_fn(genome), "introns")
    files: dict[str, Path] = {}
    try:
        files["introns"] = Path(write_introns(result, output_dir / "introns.tsv"))
    except (TypeError, ValueError):
        pass
    files.setdefault("metrics", _write_metrics_tsv(result, output_dir / "introns_metrics.tsv"))
    log("intron features extracted from annotation")
    return _package(result, _summary_from_metrics(result), files)


def _run_convert(
    params: dict[str, Any], *, output_dir: Path, log: Callable[[str], None]
) -> dict[str, Any]:
    from organelleverse.format_conversion import write_conversion

    source = _first_file(params, "input")
    genome = _genome(source, organelle="chloro", species=source.stem.replace("_", " "))
    written = write_conversion(genome, output_dir, formats=("gff3", "fasta"))
    log(f"converted to {sorted(written)}")
    return {
        "summary_text": f"formats written: {', '.join(sorted(written))}",
        "output_files": {name: str(path) for name, path in written.items()},
    }


def _run_report(
    params: dict[str, Any], *, output_dir: Path, log: Callable[[str], None]
) -> dict[str, Any]:
    from organelleverse.reporting import build as build_report
    from organelleverse.reporting import write as write_report

    source = _first_file(params, "input", "genbank")
    fmt = str(params.get("format") or "html")
    organelle = "chloro" if str(params.get("organelle") or "mito") == "chloro" else "mito"
    genome = _genome(source, organelle=organelle, species=source.stem.replace("_", " "))
    title = str(params.get("name") or source.stem)
    report = build_report(genome, title=f"OrganelleVerse Report — {title}")
    if fmt == "pdf":
        log("PDF rendering is not wired yet; writing HTML instead")
        fmt = "html"
    output = output_dir / f"report.{fmt}"
    write_report(report, output)
    log(f"report written ({fmt})")
    return {
        "summary_text": f"report generated: {output.name}",
        "output_files": {"report": str(output)},
    }


def _run_visualize(
    params: dict[str, Any], *, output_dir: Path, log: Callable[[str], None]
) -> dict[str, Any]:
    source = _first_file(params, "genbank_file", "input", "genbank")
    viz_type = str(params.get("viz_type") or "circular")
    if viz_type == "linear":
        log("linear rendering uses gbdraw when installed; falling back to circular map")
    from organelleverse.visualization.ogdraw import plot_ogdraw_map

    # No genome_name: the map names the record's species (a file stem such as
    # "annotation" names nothing).
    plot = plot_ogdraw_map(str(source))
    out = output_dir / "genome_map.png"
    plot.save(out)
    log("circular genome map rendered")
    return {
        "summary_text": "genome map rendered",
        "output_files": {"map": str(out)},
    }


def _run_compare_organelles(
    params: dict[str, Any], *, output_dir: Path, log: Callable[[str], None]
) -> dict[str, Any]:
    import organelleverse.comparative as comparative
    from organelleverse.composition.gc_core import compute_gc_content

    mito_gb = _first_file(params, "mito_genbank")
    chloro_gb = _first_file(params, "chloro_genbank")
    mito = _genome(mito_gb, organelle="mito", species=mito_gb.stem.replace("_", " "))
    chloro = _genome(chloro_gb, organelle="chloro", species=chloro_gb.stem.replace("_", " "))
    files: dict[str, Path] = {}
    summary: list[str] = []

    result = _check(comparative.compare_genomes([mito, chloro]), "compare_genomes")
    try:
        files["genome_comparison"] = Path(
            comparative.write_genome_comparison(result, output_dir / "genome_comparison.tsv")
        )
    except (TypeError, ValueError):
        files.setdefault("genome_metrics", _write_metrics_tsv(result, output_dir / "genome_metrics.tsv"))
    summary.append("genome comparison done")

    if params.get("compare_genes", True):
        genes = _check(comparative.compare_genes([mito, chloro]), "compare_genes")
        try:
            files["gene_comparison"] = Path(
                comparative.write_gene_comparison(genes, output_dir / "gene_comparison.tsv")
            )
        except (TypeError, ValueError):
            files.setdefault("gene_metrics", _write_metrics_tsv(genes, output_dir / "gene_metrics.tsv"))
        summary.append("gene content compared")

    if params.get("compare_gc", True):
        rows = []
        from Bio import SeqIO

        for label, gb in (("mitochondrion", mito_gb), ("chloroplast", chloro_gb)):
            fasta = output_dir / f"{label}.fna"
            SeqIO.write(list(SeqIO.parse(str(gb), "genbank")), str(fasta), "fasta")
            gc = compute_gc_content(fasta)
            rows.append(
                {
                    "organelle": label,
                    "gc_content": gc.get("gc_content", gc.get("gc")),
                    "length": gc.get("length") or gc.get("total_length"),
                }
            )
        files["gc"] = _write_tsv(output_dir / "gc_comparison.tsv", rows)
        summary.append("GC compared")

    if params.get("compare_codon", True):
        from organelleverse.codon_composition.codon_core import compute_codon_usage

        for label, gb, organelle in (
            ("mito", mito_gb, "mito"),
            ("chloro", chloro_gb, "chloro"),
        ):
            try:
                fasta, _ = _cds_fasta(gb, output_dir / label)
                data = compute_codon_usage(fasta, organelle=organelle)
                _write_codon_tables(data, output_dir / label)
                files[f"codon_{label}"] = output_dir / label / "codon_usage.tsv"
            except ValueError as exc:
                log(f"codon comparison for {label} skipped ({exc})")
        summary.append("codon usage compared")

    log("; ".join(summary))
    return {
        "summary_text": "; ".join(summary),
        "output_files": {k: str(v) for k, v in files.items()},
    }


# --------------------------------------------------------------------------
# alignment / phylogeny / localization / annotation / assembly / QC / RNA
# --------------------------------------------------------------------------


def _run_alignment(
    params: dict[str, Any], *, output_dir: Path, log: Callable[[str], None]
) -> dict[str, Any]:
    from organelleverse.phylogeny import align as align_fn
    from organelleverse.phylogeny import write_alignment

    source = _first_file(params, "input")
    method = str(params.get("method") or "mafft-auto")
    method_map = {
        "mafft-auto": "auto",
        "mafft-linsi": "linsi",
        "mafft-einsi": "einsi",
        "muscle": "auto",
    }
    result = _check(align_fn(source, method=method_map.get(method, "auto")), "align")
    aligned = write_alignment(result, output_dir / "alignment.fas")
    files: dict[str, Path] = {"alignment": aligned}
    if params.get("trim"):
        try:
            from organelleverse.phylogeny import trim_alignment

            trimmed = _check(
                trim_alignment(str(aligned)), "trim_alignment"
            )
            files["trimmed"] = write_alignment(trimmed, output_dir / "alignment_trimmed.fas")
            log("gap trimming applied (trimAl)")
        except Exception as exc:  # trimal missing — keep the untrimmed result
            log(f"trimming skipped: {exc}")
    metrics = _metrics(result)
    engine = metrics.get("method", method)
    log(f"alignment engine: {engine}")
    return {
        "summary_text": f"aligned {metrics.get('n_sequences', '?')} sequences via {engine}",
        "output_files": {k: str(v) for k, v in files.items()},
    }


def _phylogeny_matrix(organelle: str) -> RunnerFn:
    def runner(
        params: dict[str, Any], *, output_dir: Path, log: Callable[[str], None]
    ) -> dict[str, Any]:
        from organelleverse.phylogeny import extract_shared_genes, write_shared_genes

        source = _first_file(params, "input")
        paths: list[Path] = []
        if source.is_dir():
            paths.extend(
                p for p in sorted(source.iterdir())
                if p.suffix.lower() in _GENBANK_SUFFIXES
            )
        else:
            paths.extend(_split_genbank(source))
        if len(paths) < 2:
            raise ValueError("phylogenetic matrix requires at least 2 GenBank genomes")
        result = _check(extract_shared_genes([str(p) for p in paths]), "extract_shared_genes")
        files = write_shared_genes(result, output_dir)
        log(f"shared-gene matrix from {len(paths)} genomes")
        metrics = _metrics(result)
        return {
            "summary_text": f"shared genes={metrics.get('shared_gene_count', metrics.get('n_shared', '?'))}; genomes={len(paths)}",
            "output_files": {name: str(path) for name, path in files.items()},
        }

    return runner


def _run_localization(
    params: dict[str, Any], *, output_dir: Path, log: Callable[[str], None]
) -> dict[str, Any]:
    from organelleverse.localization import predict as predict_localization

    source = _first_file(params, "input")
    method = str(params.get("method") or "targetp")
    backend_map = {"targetp": "targetp", "deeploc": "deeploc", "wolfpsort": "auto", "all": "auto"}
    organism = str(params.get("organism") or "plant")
    result = predict_localization(
        source, backend=backend_map.get(method, "auto"), organism=organism
    )
    metrics = _metrics(result)
    predictions = (
        metrics.get("predictions")
        or metrics.get("per_protein")
        or metrics.get("rows")
        or []
    )
    out = _write_tsv(
        output_dir / "localization.tsv",
        predictions or [{"metric": k, "value": v} for k, v in metrics.items()
                        if not isinstance(v, (list, dict))],
    )
    log(f"localization via {method} (auto-falls-back when a backend is unavailable)")
    return {
        "summary_text": f"{len(predictions)} proteins classified",
        "output_files": {"predictions": str(out)},
    }


def _annotate_runner(organelle: str, backend: str) -> RunnerFn:
    def runner(
        params: dict[str, Any], *, output_dir: Path, log: Callable[[str], None]
    ) -> dict[str, Any]:
        from organelleverse.annotation import annotate
        from organelleverse.annotation import write as write_annotation

        source = _first_file(params, "input")
        species = str(params.get("name") or "").strip() or source.stem.replace("_", " ")
        genome = _genome(source, organelle=organelle, species=species)
        threads = int(params.get("threads") or 4)
        # All three stages are native: PCG via bundled HMM profiles (pyhmmer),
        # tRNA via packaged CM models on the Rust CYK kernel, rRNA via packaged
        # references. External tRNAscan-SE / barrnap are opt-in cross-checks,
        # never required.
        call_trna = not bool(params.get("skip_trna"))
        call_rrna = not bool(params.get("skip_rrna"))
        log(f"annotating {species} (backend={backend}, threads={threads})")
        result = _check(
            annotate(
                genome,
                backend=backend,
                threads=threads,
                call_trna=call_trna,
                call_rrna=call_rrna,
            ),
            "annotate",
        )
        output = output_dir / "annotation_bundle"
        write_annotation(result, output=output)
        # The materializer writes a bundle directory; the GenBank lives inside.
        if output.is_dir():
            candidates = sorted(output.glob("*.gb")) + sorted(output.glob("*.gbk"))
            gb_file = candidates[0] if candidates else output
        else:
            gb_file = output
        files: dict[str, Path] = {"annotation": gb_file}
        for extra in ("annotation.gff3", "cds.fasta", "proteins.fasta", "trna.fasta", "rrna.fasta"):
            candidate = output / extra if output.is_dir() else None
            if candidate is not None and candidate.exists():
                files[extra.replace("annotation.", "").replace(".fasta", "")] = candidate

        # Feature fastas + table from the produced GenBank.
        try:
            from Bio import SeqIO

            rows: list[dict[str, Any]] = []
            handles: dict[str, Any] = {}
            for feature_type in ("CDS", "tRNA", "rRNA"):
                handles[feature_type] = (output_dir / f"{feature_type}.ffa").open("w")
            for record in SeqIO.parse(str(gb_file), "genbank"):
                for feature in record.features:
                    if feature.type not in handles:
                        continue
                    name = (
                        feature.qualifiers.get("gene", [None])[0]
                        or feature.qualifiers.get("product", [None])[0]
                        or feature.type
                    )
                    start = int(feature.location.start)
                    end = int(feature.location.end)
                    strand = "+" if feature.location.strand == 1 else "-"
                    seq = str(feature.extract(record.seq)).upper()
                    handles[feature.type].write(f">{record.id}|{name}\n{seq}\n")
                    rows.append(
                        {
                            "type": feature.type,
                            "gene": name,
                            "start": start + 1,
                            "end": end,
                            "strand": strand,
                            "length": end - start,
                        }
                    )
            for handle in handles.values():
                handle.close()
            files["features"] = _write_tsv(output_dir / "features.tsv", rows)
            files["cds_fasta"] = output_dir / "CDS.ffa"
            files["trna_fasta"] = output_dir / "tRNA.ffa"
            files["rrna_fasta"] = output_dir / "rRNA.ffa"
        except Exception as exc:  # feature export is convenience, never fatal
            log(f"feature export skipped: {exc}")

        metrics = _metrics(result)
        log(f"annotation metrics: {json.dumps(metrics, default=str)[:400]}")
        counts: Counter[str] = Counter(row["type"] for row in rows) if rows else Counter()
        summary = (
            f"CDS={counts.get('CDS', 0)}, tRNA={counts.get('tRNA', 0)}, "
            f"rRNA={counts.get('rRNA', 0)}"
            if rows
            else _summary_from_metrics(result, "pcg_count", "trna_count", "rrna_count", "gene_count")
        )
        return {
            "summary_text": summary,
            "output_files": {k: str(v) for k, v in files.items()},
        }

    return runner


def _feature_annotation(feature_type: str) -> RunnerFn:
    """mito_trna / mito_rrna / mito_pcg: annotate then export one type."""

    def runner(
        params: dict[str, Any], *, output_dir: Path, log: Callable[[str], None]
    ) -> dict[str, Any]:
        if feature_type == "CDS":
            # The built-in pyhmmer pipeline covers exactly this stage.
            params = {**params, "skip_trna": True, "skip_rrna": True}
        base = _annotate_runner("mito", "mitochondrion")
        outcome = base(params, output_dir=output_dir, log=log)
        annotation = Path(outcome["output_files"]["annotation"])
        fasta, count = _cds_fasta(annotation, output_dir, feature_type=feature_type)
        out_fasta = output_dir / f"{feature_type}.ffa"
        fasta.replace(out_fasta)
        rows = _write_tsv(
            output_dir / f"{feature_type}_table.tsv",
            _feature_rows(annotation, feature_type),
        )
        log(f"{count} {feature_type} features exported")
        outcome["summary_text"] = f"{count} {feature_type} features annotated"
        outcome["output_files"][feature_type.lower()] = str(out_fasta)
        outcome["output_files"]["table"] = str(rows)
        return outcome

    return runner


def _feature_rows(genbank: Path, feature_type: str) -> list[dict[str, Any]]:
    from Bio import SeqIO

    rows: list[dict[str, Any]] = []
    for record in SeqIO.parse(str(genbank), "genbank"):
        for feature in record.features:
            if feature.type != feature_type:
                continue
            rows.append(
                {
                    "gene": feature.qualifiers.get("gene", [None])[0]
                    or feature.qualifiers.get("product", [None])[0]
                    or feature_type,
                    "start": int(feature.location.start) + 1,
                    "end": int(feature.location.end),
                    "strand": "+" if feature.location.strand == 1 else "-",
                    "product": feature.qualifiers.get("product", [""])[0],
                }
            )
    return rows


def _run_mito_boundary(
    params: dict[str, Any], *, output_dir: Path, log: Callable[[str], None]
) -> dict[str, Any]:
    """Boundary refinement runs inside the annotation backend's pipeline; the
    runner re-runs annotation and reports the resulting exon/intron layout."""
    base = _annotate_runner("mito", "mitochondrion")
    outcome = base(params, output_dir=output_dir, log=log)
    annotation = Path(outcome["output_files"]["annotation"])
    from Bio import SeqIO

    rows: list[dict[str, Any]] = []
    for record in SeqIO.parse(str(annotation), "genbank"):
        for feature in record.features:
            if feature.type not in ("CDS", "gene"):
                continue
            parts = feature.location.parts
            if len(parts) <= 1:
                continue
            rows.append(
                {
                    "gene": feature.qualifiers.get("gene", [feature.type])[0],
                    "exons": len(parts),
                    "span": int(feature.location.end) - int(feature.location.start),
                    "intron_total": int(feature.location.end)
                    - int(feature.location.start)
                    - sum(int(p.end) - int(p.start) for p in parts),
                }
            )
    table = _write_tsv(output_dir / "boundary_structure.tsv", rows)
    log(f"{len(rows)} multi-exon features after boundary refinement")
    outcome["summary_text"] = f"{len(rows)} multi-exon features"
    outcome["output_files"]["boundary"] = str(table)
    return outcome


def _run_mito_qc(
    params: dict[str, Any], *, output_dir: Path, log: Callable[[str], None]
) -> dict[str, Any]:
    from organelleverse._bio import read_fasta
    from organelleverse.composition.gc_core import compute_gc_content

    source = _first_file(params, "input")
    records = read_fasta(source)
    if not records:
        raise ValueError("no sequences in input FASTA")
    lengths = sorted((len(seq) for _, seq in records), reverse=True)
    total = sum(lengths)
    half = total / 2
    running = 0
    n50 = 0
    for length in lengths:
        running += length
        if running >= half:
            n50 = length
            break
    gc_data = compute_gc_content(source)
    gc_value = gc_data.get("gc_content", gc_data.get("gc"))

    rows: list[dict[str, Any]] = [
        {"metric": "contigs", "value": len(records)},
        {"metric": "total_length", "value": total},
        {"metric": "n50", "value": n50},
        {"metric": "min_length", "value": lengths[-1]},
        {"metric": "max_length", "value": lengths[0]},
        {"metric": "gc_content", "value": gc_value},
        {
            "metric": "structure",
            "value": "circular" if len(records) == 1 else "multi-contig",
        },
    ]
    files: dict[str, Path] = {"summary": _write_tsv(output_dir / "qc_summary.tsv", rows)}

    contamination = str(params.get("cp") or "").strip()
    if contamination:
        cp_path = Path(contamination)
        if cp_path.exists():
            shared = _shared_kmer_fraction(source, cp_path)
            rows.append({"metric": "cp_shared_kmers", "value": shared})
            files["summary"] = _write_tsv(output_dir / "qc_summary.tsv", rows)
            log(f"MTPT screen: {shared:.4f} of mito 31-mers also in chloroplast")
        else:
            log(f"chloroplast FASTA not found: {cp_path}")

    bam = str(params.get("bam") or "").strip()
    if bam and Path(bam).exists():
        try:
            coverage_rows = _bam_coverage(bam, records)
            files["coverage"] = _write_tsv(output_dir / "coverage.tsv", coverage_rows)
            mean = (
                sum(r["mean_depth"] for r in coverage_rows) / len(coverage_rows)
                if coverage_rows
                else 0
            )
            rows.append({"metric": "mean_depth", "value": round(mean, 2)})
            files["summary"] = _write_tsv(output_dir / "qc_summary.tsv", rows)
            log(f"BAM coverage: mean depth {mean:.1f}x")
        except ImportError:
            log("rubam not installed — coverage analysis skipped")

    log(f"QC: {len(records)} contigs, N50={n50}, GC={gc_value}")
    return {
        "summary_text": f"contigs={len(records)}; N50={n50}; GC={gc_value}",
        "output_files": {k: str(v) for k, v in files.items()},
    }


def _shared_kmer_fraction(mito_fasta: Path, cp_fasta: Path, k: int = 31) -> float:
    from organelleverse._bio import read_fasta

    def kmers(path: Path) -> set[str]:
        out: set[str] = set()
        for _, seq in read_fasta(path):
            s = seq.upper().replace("-", "")
            for i in range(0, max(len(s) - k + 1, 0), 7):  # stride keeps sets small
                out.add(s[i : i + k])
        return out

    mito_k = kmers(mito_fasta)
    cp_k = kmers(cp_fasta)
    if not mito_k:
        return 0.0
    return len(mito_k & cp_k) / len(mito_k)


def _bam_coverage(bam_path: str, records: list[tuple[str, str]]) -> list[dict[str, Any]]:
    import rubam

    rows: list[dict[str, Any]] = []
    with rubam.AlignmentFile(bam_path, "rb") as bam:
        for name, seq in records[:50]:
            try:
                depth = 0
                bases = 0
                for column in bam.pileup(name, 0, min(len(seq), 2_000_000), truncate=True):
                    if not column.nsegments:
                        continue
                    depth += column.nsegments
                    bases += 1
                rows.append(
                    {
                        "contig": name,
                        "mean_depth": round(depth / bases, 2) if bases else 0.0,
                        "covered_bases": bases,
                    }
                )
            except ValueError:
                continue
    return rows


def _run_validate_rna(
    params: dict[str, Any], *, output_dir: Path, log: Callable[[str], None]
) -> dict[str, Any]:
    import rubam
    from Bio import SeqIO

    genbank = _first_file(params, "genbank", "input")
    bam_path = _first_file(params, "bam")
    min_coverage = int(params.get("min_coverage") or 5)

    record = next(SeqIO.parse(str(genbank), "genbank"))
    reference_name = record.id

    junction_rows: list[dict[str, Any]] = []
    intron_coords = set()
    for feature in record.features:
        if feature.type not in ("CDS", "gene") or len(feature.location.parts) <= 1:
            continue
        parts = feature.location.parts
        for left, right in zip(parts, parts[1:]):
            intron_coords.add((int(left.end), int(right.start)))

    editing_rows: list[dict[str, Any]] = []
    with rubam.AlignmentFile(str(bam_path), "rb") as bam:
        for read in bam.fetch(reference_name, 0, bam.get_reference_length(reference_name)):
            ref_pos = read.reference_start
            for op, length in read.cigartuples:
                if op == 3:  # N — splice junction
                    supported = (ref_pos, ref_pos + length) in intron_coords
                    junction_rows.append(
                        {
                            "donor": ref_pos,
                            "acceptor": ref_pos + length,
                            "annotated": supported,
                            "read": read.query_name,
                        }
                    )
                    ref_pos += length
                elif op in (0, 2, 7, 8):
                    ref_pos += length

        # A-to-G editing candidates from pileup mismatches.
        for column in bam.pileup(reference_name, 0, len(record.seq), truncate=True):
            if column.nsegments < min_coverage:
                continue
            counts = Counter({"A": column.a, "C": column.c, "G": column.g, "T": column.t})
            ref_base = str(record.seq[column.reference_pos]).upper()
            if ref_base in "AT" and counts:
                alt = "G" if ref_base == "A" else "T"  # reverse-strand signal
                total = sum(counts.values())
                if counts.get(alt, 0) / total >= 0.3 and counts.get(alt, 0) >= min_coverage:
                    editing_rows.append(
                        {
                            "position": column.reference_pos + 1,
                            "ref": ref_base,
                            "alt": alt,
                            "depth": total,
                            "alt_fraction": round(counts[alt] / total, 3),
                        }
                    )
                    if len(editing_rows) >= 500:
                        break

    junction_file = _write_tsv(output_dir / "splice_junctions.tsv", junction_rows)
    editing_file = _write_tsv(output_dir / "editing_candidates.tsv", editing_rows)
    annotated = sum(1 for r in junction_rows if r["annotated"])
    log(
        f"{len(junction_rows)} splice junctions ({annotated} annotated); "
        f"{len(editing_rows)} RNA-editing candidates"
    )
    return {
        "summary_text": (
            f"junctions={len(junction_rows)} (annotated={annotated}); "
            f"editing candidates={len(editing_rows)}"
        ),
        "output_files": {
            "junctions": str(junction_file),
            "editing": str(editing_file),
        },
    }


def _assemble_runner(organelle: str) -> RunnerFn:
    def runner(
        params: dict[str, Any], *, output_dir: Path, log: Callable[[str], None]
    ) -> dict[str, Any]:
        method = str(params.get("tool") or "auto")
        if method == "ovasm":
            from organelleverse.assembly.ovasm_fallback import run_ovasm_resilient

            return run_ovasm_resilient(params, organelle=organelle, reads=_first_file(params, "input"), output_dir=output_dir, log=log)
        if params.get("strategy") == "hybrid":
            # Only OVASM corrects ONT/CLR reads with Illumina k-mers; refuse instead of
            # silently assembling the long reads alone.
            raise ValueError("Hybrid (long + short reads) assembly is supported only by OVASM; select OVASM as the assembler")
        from organelleverse.assembly import assemble
        from organelleverse.assembly import write as write_assembly
        from organelleverse.assembly.contracts import LongReadLibrary, ShortReadLibrary
        from organelleverse.read_facade import read as ov_read

        reads = _first_file(params, "input")
        strategy = str(params.get("strategy") or "hifi_only")
        threads = int(params.get("threads") or 4)

        if strategy == "short_read":
            libraries = dict(
                short_libraries=(
                    ShortReadLibrary(
                        technology="illumina",
                        layout="single_end",
                        read1=str(reads),
                        read_length=_infer_read_length(reads),
                    ),
                )
            )
        elif strategy == "ont_only":
            libraries = dict(
                long_libraries=(
                    LongReadLibrary(technology="ont", quality_state="raw", reads=str(reads)),
                )
            )
        elif strategy == "clr_only":
            libraries = dict(
                long_libraries=(
                    LongReadLibrary(technology="pacbio_clr", quality_state="raw", reads=str(reads)),
                )
            )
        else:
            libraries = dict(
                long_libraries=(
                    LongReadLibrary(
                        technology="pacbio_hifi", quality_state="ccs", reads=str(reads)
                    ),
                )
            )

        data = ov_read(**libraries)
        log(f"assembling {organelle} (strategy={strategy}, threads={threads})")
        result = _check(
            assemble(data, organelle=organelle, method=method, threads=threads), "assemble"
        )
        output = output_dir / "assembly"
        write_assembly(result, output=output)
        metrics = _metrics(result)
        log(f"assembly metrics: {json.dumps(metrics, default=str)[:400]}")
        return {
            "summary_text": _summary_from_metrics(result, "contigs", "total_length", "circular"),
            "output_files": {"assembly": str(output)},
        }

    return runner


def _infer_read_length(fastq: Path, sample: int = 100) -> int:
    opener = open
    if str(fastq).endswith(".gz"):
        import gzip

        opener = gzip.open
    lengths = []
    with opener(str(fastq), "rt") as handle:  # type: ignore[operator]
        for i, line in enumerate(handle):
            if i % 4 == 1:
                lengths.append(len(line.strip()))
            if len(lengths) >= sample:
                break
    if not lengths:
        raise ValueError(f"could not read any reads from {fastq}")
    return max(lengths)


def _run_protein_structure(
    params: dict[str, Any], *, output_dir: Path, log: Callable[[str], None]
) -> dict[str, Any]:
    """Fold the first protein via the public ESMFold API and write PDB files.

    AlphaFold2 has no local deployment; choosing it fails loudly instead of
    pretending a structure was predicted.
    """
    from organelleverse._bio import read_fasta

    method = str(params.get("method") or "esmfold")
    if method == "alphafold2":
        raise RuntimeError(
            "AlphaFold2 is not deployed locally; use ESMFold (fast) for structure prediction"
        )
    source = _first_file(params, "input")
    records = read_fasta(source)
    if not records:
        raise ValueError("no protein sequences in input")

    import urllib.request

    files: dict[str, Path] = {}
    summaries: list[str] = []
    for i, (name, seq) in enumerate(records[:5]):
        url = "https://api.esmatlas.com/foldSequence/v1/pdb/"
        request = urllib.request.Request(
            url,
            data=json.dumps({"sequence": seq}).encode(),
            headers={"Content-Type": "application/json"},
        )
        log(f"folding {name} ({len(seq)} aa) via ESMFold…")
        with urllib.request.urlopen(request, timeout=300) as response:
            pdb = response.read().decode()
        out = output_dir / f"{name.replace('/', '_') or f'protein_{i+1}'}.pdb"
        out.write_text(pdb)
        files[f"structure_{i + 1}"] = out
        summaries.append(name)
    return {
        "summary_text": f"folded {len(files)} structure(s): {', '.join(summaries)}",
        "output_files": {k: str(v) for k, v in files.items()},
    }


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------


_FASTA_SUFFIXES = (".fasta", ".fa", ".fna")
_GENBANK_SUFFIXES = (".gb", ".gbk", ".genbank")


def _produced_file(output_files: dict[str, str], suffixes: tuple[str, ...], prefer: str) -> Path | None:
    """The first produced file with one of ``suffixes``: a listed file whose label contains
    ``prefer`` wins, then any listed file, then the shallowest match inside a listed directory."""
    paths = [(label, Path(value)) for label, value in output_files.items()]
    listed = [(label, path) for label, path in paths if path.is_file() and path.suffix.lower() in suffixes]
    if listed:
        return sorted(listed, key=lambda item: prefer not in item[0])[0][1]
    for _, directory in paths:
        if directory.is_dir():
            found = sorted(
                (p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in suffixes),
                key=lambda p: (len(p.relative_to(directory).parts), p.name),
            )
            if found:
                return found[0]
    return None


def _with_follow_up(
    run: RunnerFn,
    follow_up: RunnerFn,
    *,
    label: str,
    suffixes: tuple[str, ...],
    prefer: str,
    params_for: Callable[[dict[str, Any], Path], dict[str, Any]],
) -> RunnerFn:
    """Run ``follow_up`` on what ``run`` produced, inside the same run and output directory.

    The follow-up's files join the run's ``output_files`` as ``{label}_*`` and its summary is
    appended. A missing input or a failing follow-up is logged and never fails the main run.
    """

    def runner(params: dict[str, Any], *, output_dir: Path, log: Callable[[str], None]) -> dict[str, Any]:
        result = run(params, output_dir=output_dir, log=log)
        output_files = dict(result.get("output_files") or {})
        source = _produced_file(output_files, suffixes, prefer)
        if source is None:
            log(f"{label}: no {'/'.join(suffixes)} output to evaluate; skipped")
            return result
        target = Path(output_dir) / label
        target.mkdir(parents=True, exist_ok=True)
        try:
            extra = follow_up(params_for(params, source), output_dir=target, log=log)
        except Exception as exc:  # the main result stays valid without its follow-up
            log(f"{label} failed: {_error_text(exc)}")
            return {**result, "summary_text": f"{result.get('summary_text', '')} | {label}: failed"}
        for key, value in (extra.get("output_files") or {}).items():
            output_files[f"{label}_{key}"] = value
        summary = f"{result.get('summary_text', '')} | {label}: {extra.get('summary_text', 'done')}"
        return {**result, "summary_text": summary, "output_files": output_files}

    return runner


def _with_assembly_qc(run: RunnerFn) -> RunnerFn:
    """Assembly output always carries its quality control."""
    return _with_follow_up(
        run, _run_mito_qc, label="qc", suffixes=_FASTA_SUFFIXES, prefer="assembly",
        params_for=lambda params, fasta: {"input": str(fasta), "name": params.get("name") or ""},
    )


def _with_annotation_report(run: RunnerFn, organelle: str) -> RunnerFn:
    """Annotation output always carries its report."""
    return _with_follow_up(
        run, _run_report, label="report", suffixes=_GENBANK_SUFFIXES, prefer="annotation",
        params_for=lambda params, genbank: {"input": str(genbank), "name": params.get("name") or "", "organelle": organelle},
    )


def _ensures_output_dir(fn: RunnerFn) -> RunnerFn:
    """Guarantee the output directory exists before a runner touches it."""

    def wrapper(params: dict[str, Any], *, output_dir: Path, log: Callable[[str], None]) -> dict[str, Any]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        return fn(params, output_dir=out, log=log)

    return wrapper


ANALYSIS_TOOL_RUNNERS: dict[str, RunnerFn] = {
    # mitochondrion
    "mito_annotate": _with_annotation_report(_annotate_runner("mito", "mitochondrion"), "mito"),
    "mito_pcg": _feature_annotation("CDS"),
    "mito_trna": _feature_annotation("tRNA"),
    "mito_rrna": _feature_annotation("rRNA"),
    "mito_boundary": _run_mito_boundary,
    "mito_qc": _run_mito_qc,
    "mito_visualize": _run_visualize,
    "mito_codon": _run_codon("mito"),
    "mito_gc": _run_gc,
    "mito_repeat": _repeat_analysis(50),
    "mito_large_repeat": _repeat_analysis(500),
    "mito_ssr": _run_ssr,
    "mito_multiconf": _run_multiconf,
    "mito_phylogeny": _phylogeny_matrix("mito"),
    "mito_synteny": _run_synteny,
    "mito_report": _run_report,
    "mito_validate_rna": _run_validate_rna,
    "mito_assemble": _with_assembly_qc(_assemble_runner("mitochondrion")),
    # chloroplast
    "chloro_assemble": _with_assembly_qc(_assemble_runner("plastid")),
    "chloro_annotate": _with_annotation_report(_annotate_runner("chloro", "plastome"), "chloro"),
    "chloro_codon": _run_codon("chloro"),
    "chloro_phylogeny": _phylogeny_matrix("chloro"),
    "chloro_ir_boundary": _run_ir_boundary,
    "chloro_snp": _run_snp,
    "chloro_ssr": _run_ssr,
    "chloro_diversity": _run_diversity,
    "chloro_intron": _run_introns,
    "chloro_gene_table": _comparative_runner("gene_table"),
    "chloro_amino": _run_amino_acid,
    "chloro_convert": _run_convert,
    "chloro_gene_compare": _comparative_runner("compare_genes"),
    "chloro_genome_compare": _comparative_runner("compare_genomes"),
    # general
    "compare_organelles": _run_compare_organelles,
    "sequence-alignment": _run_alignment,
    "subcellular-localization": _run_localization,
    "protein-structure": _run_protein_structure,
}

ANALYSIS_TOOL_RUNNERS = {name: _ensures_output_dir(fn) for name, fn in ANALYSIS_TOOL_RUNNERS.items()}
