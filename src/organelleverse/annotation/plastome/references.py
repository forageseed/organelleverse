"""Reference extraction for the OrganelleVerse plastome backend."""

from __future__ import annotations

import hashlib
import json
import re
import zlib
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqFeature import CompoundLocation, FeatureLocation, SeqFeature
from Bio.SeqRecord import SeqRecord

from .models import ReferenceFeature, ReferenceQuery, ReferenceRecord

FEATURE_TYPES = frozenset({"gene", "CDS", "tRNA", "rRNA", "repeat_region"})
# 2: multi-exon CDS parts on the wrong strand are repaired before caching
# (per record from /translation, across the reference set from homology).
PREPROCESSED_FORMAT_VERSION = 2
# A spliced reference CDS whose translation agrees with its /translation below
# this share is misannotated; RNA editing changes far fewer residues.
_TRANSLATION_AGREEMENT = 0.9


def load_references(paths: list[Path]) -> tuple[ReferenceRecord, ...]:
    key = tuple(_reference_cache_key(path) for path in sorted(paths))
    return _load_references_cached(key)


@lru_cache(maxsize=128)
def _load_reference_cached(
    key: tuple[str, int, int, str, int, int],
) -> ReferenceRecord:
    path_text, _, _, manifest_text, _, _ = key
    path = Path(path_text)
    manifest = Path(manifest_text) if manifest_text else None
    if manifest and manifest.exists():
        return _load_preprocessed_reference(path, manifest)
    return _parse_reference(path)


@lru_cache(maxsize=8)
def _load_references_cached(
    key: tuple[tuple[str, int, int, str, int, int], ...],
) -> tuple[ReferenceRecord, ...]:
    references = repair_reversed_trans_parts([_load_reference_cached(item) for item in key])
    usable = tuple(reference for reference in references if reference.features)
    if not usable:
        raise ValueError("No usable plastome reference features found")
    return usable


# Cross-reference check of trans-spliced CDS. Some NCBI records put rps12 exon 1
# on the strand of exons 2-3 and compute /translation from that location, so the
# record agrees with itself (Sorghum, Vitis, Eucalyptus copies: MHPSTCSS... for
# the universal MPTIKQLIRN...). Homology with the other references exposes it.
# Reversed rps12 exon 1 in the shipped set: 0.62-0.68 as annotated, 0.80-0.98
# flipped; no correctly annotated copy gains from flipping any part.
_TRANS_HOMOLOGS = 12
_TRANS_MIN_GAIN = 0.15
_TRANS_MIN_SIMILARITY = 0.75


@lru_cache(maxsize=1)
def _protein_aligner():
    from Bio.Align import PairwiseAligner, substitution_matrices

    aligner = PairwiseAligner(mode="local", open_gap_score=-11, extend_gap_score=-1)
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    return aligner


def _protein_similarity(query: str, others: list[str]) -> float:
    """Median of local BLOSUM62 score over each homolog's self-score."""
    aligner = _protein_aligner()
    clean = "".join(c if c.isalpha() and c != "*" else "X" for c in query.rstrip("*"))
    values = []
    for other in others:
        o = "".join(c if c.isalpha() else "X" for c in other.rstrip("*"))
        self_score = aligner.score(o, o)
        values.append(aligner.score(clean, o) / self_score if self_score > 0 and clean else 0.0)
    values.sort()
    return values[len(values) // 2] if values else 0.0


def _spliced_translation(location, sequence: Seq) -> str:
    coding = str(location.extract(sequence))
    coding = coding[: len(coding) - len(coding) % 3]
    return str(Seq(coding).translate(table=11)) if coding else ""


def repair_reversed_trans_parts(references: list[ReferenceRecord]) -> list[ReferenceRecord]:
    """Re-parse references whose trans-spliced CDS has a part on the wrong strand.

    A part is flipped when that raises the spliced translation's similarity to
    the same gene's proteins in other references by _TRANS_MIN_GAIN to at least
    _TRANS_MIN_SIMILARITY. Only records parsed from GenBank (with sequence) are
    checked; CDS that a record's own /translation already repaired pass.
    """
    proteins: dict[str, list[tuple[int, str]]] = {}
    for i, reference in enumerate(references):
        for feat in reference.features:
            if feat.feature_type == "CDS" and feat.gene:
                translation = feat.feature.qualifiers.get("translation", [""])[0]
                if translation:
                    proteins.setdefault(feat.gene, []).append((i, translation))
    out = list(references)
    for i, reference in enumerate(references):
        sequence = reference.record.seq
        if not len(sequence):
            continue
        flips: list[tuple[str, int]] = []
        for feat in reference.features:
            if feat.feature_type != "CDS" or feat.exon_count < 2:
                continue
            parts = list(feat.feature.location.parts)
            if len({p.strand for p in parts}) < 2 and "trans_splicing" not in feat.feature.qualifiers:
                continue
            others = [p for j, p in proteins.get(feat.gene, []) if j != i][:_TRANS_HOMOLOGS]
            if len(others) < 3:
                continue
            base = _protein_similarity(_spliced_translation(feat.feature.location, sequence), others)
            best = None
            for k, part in enumerate(parts):
                flipped = CompoundLocation(
                    [*parts[:k], FeatureLocation(part.start, part.end, strand=-(part.strand or 1)), *parts[k + 1 :]]
                )
                value = _protein_similarity(_spliced_translation(flipped, sequence), others)
                if value >= _TRANS_MIN_SIMILARITY and value >= base + _TRANS_MIN_GAIN and (best is None or value > best[0]):
                    best = (value, k)
            if best is not None:
                flips.append((feat.feature_id, best[1]))
        if flips:
            out[i] = _parse_reference(reference.path, flips=tuple(flips))
    return out


def _reference_cache_key(path: Path) -> tuple[str, int, int, str, int, int]:
    stat = path.stat()
    manifest = preprocessed_manifest_for(path)
    if manifest.exists():
        manifest_stat = manifest.stat()
        return (
            str(path.resolve()),
            stat.st_size,
            stat.st_mtime_ns,
            str(manifest.resolve()),
            manifest_stat.st_size,
            manifest_stat.st_mtime_ns,
        )
    return (str(path.resolve()), stat.st_size, stat.st_mtime_ns, "", 0, 0)


def preprocessed_manifest_for(path: Path) -> Path:
    return path.parent.parent / "preprocessed" / path.stem / "features.json"


_AA3_TO_1 = {
    "Ala": "A",
    "Arg": "R",
    "Asn": "N",
    "Asp": "D",
    "Cys": "C",
    "Gln": "Q",
    "Glu": "E",
    "Gly": "G",
    "His": "H",
    "Ile": "I",
    "Leu": "L",
    "Lys": "K",
    "Met": "M",
    "Phe": "F",
    "Pro": "P",
    "Ser": "S",
    "Thr": "T",
    "Trp": "W",
    "Tyr": "Y",
    "Val": "V",
    "fMet": "fM",
}
_RNA_COMPLEMENT = str.maketrans("ACGU", "UGCA")


def _trna_name_from(qualifiers: dict[str, list[str]]) -> str | None:
    """trnX-NNN from /product, /codon_recognized, /note or /anticodon; trnX if no anticodon."""
    text = " ".join(qualifiers.get("product", []) + qualifiers.get("note", []))
    match = re.search(r"tRNA-(fMet|[A-Z][a-z]{2})", text)
    amino_acid = _AA3_TO_1.get(match.group(1)) if match else None
    if amino_acid is None:
        return None
    anticodon = None
    codon = qualifiers.get("codon_recognized", [""])[0].upper().replace("T", "U")
    if len(codon) == 3:
        # Lysidine-modified trnI-CAU reads AUA; it is named by its gene anticodon.
        anticodon = (
            "CAU"
            if amino_acid == "I" and codon == "AUA"
            else codon[::-1].translate(_RNA_COMPLEMENT)
        )
    else:
        found = re.search(r"\(([ACGTUacgtu]{3})\)", text) or re.search(
            r"seq:([ACGTUacgtu]{3})", " ".join(qualifiers.get("anticodon", []))
        )
        if found:
            anticodon = found.group(1).upper().replace("T", "U")
    return f"trn{amino_acid}-{anticodon}" if anticodon else f"trn{amino_acid}"


def _fill_trna_gene_names(rec: SeqRecord) -> None:
    """Name tRNAs, and their gene features, that carry no /gene qualifier.

    Several NCBI records (Eucalyptus, Marchantia, Oryza, Populus, Cryptomeria)
    give tRNAs only /product="tRNA-Xxx". Without a name the transferred gene
    span cannot guide intron-tRNA splicing in the target genome.
    """
    by_tag: dict[str, str] = {}
    by_span: dict[tuple[int, int, int | None], str] = {}
    for feature in rec.features:
        if feature.type != "tRNA" or feature.qualifiers.get("gene"):
            continue
        name = _trna_name_from(feature.qualifiers)
        if name is None:
            continue
        feature.qualifiers["gene"] = [name]
        tag = feature.qualifiers.get("locus_tag", [None])[0]
        if tag:
            by_tag[tag] = name
        location = feature.location
        by_span[(int(location.start), int(location.end), location.strand)] = name
    for feature in rec.features:
        if feature.type != "gene" or feature.qualifiers.get("gene"):
            continue
        location = feature.location
        name = by_tag.get(feature.qualifiers.get("locus_tag", [None])[0]) or by_span.get(
            (int(location.start), int(location.end), location.strand)
        )
        if name:
            feature.qualifiers["gene"] = [name]


_PRODUCT_ALIASES = {"maturase": "matK"}


@lru_cache(maxsize=1)
def _gene_by_product() -> dict[str, str]:
    """Lower-cased product -> gene symbol from the shipped product table (products shared by two genes dropped)."""
    from .db import load_product_map

    seen: dict[str, set[str]] = {}
    for gene, product in load_product_map().items():
        seen.setdefault(product.lower(), set()).add(gene)
    table = {p: next(iter(g)) for p, g in seen.items() if len(g) == 1}
    table.update({p.lower(): g for p, g in _PRODUCT_ALIASES.items()})
    return table


def _fill_cds_gene_names(rec: SeqRecord) -> None:
    """Give a CDS that carries only /product (Marchantia, Azolla, ...) its gene symbol.

    Otherwise the product string ("photosystem I protein M", "maturase") becomes
    the gene name of every feature transferred from it and matches no truth gene.
    The gene feature sharing its locus_tag names it when there is one; the
    product table is the fallback.
    """
    by_tag = {
        f.qualifiers["locus_tag"][0]: f.qualifiers["gene"][0]
        for f in rec.features
        if f.type == "gene" and f.qualifiers.get("gene") and f.qualifiers.get("locus_tag")
    }
    table = _gene_by_product()
    for feature in rec.features:
        if feature.type != "CDS" or feature.qualifiers.get("gene"):
            continue
        name = by_tag.get(feature.qualifiers.get("locus_tag", [None])[0])
        if not name:
            name = table.get(feature.qualifiers.get("product", [""])[0].strip().lower())
        if name:
            feature.qualifiers["gene"] = [name]


def _parse_reference(path: Path, flips: tuple[tuple[str, int], ...] = ()) -> ReferenceRecord:
    """Parse one GenBank reference; ``flips`` = (feature_id, part index) pairs to put on the other strand."""
    rec = SeqIO.read(path, "genbank")
    flip_parts = dict(flips)
    _fill_trna_gene_names(rec)
    _fill_cds_gene_names(rec)
    features: list[ReferenceFeature] = []
    ref_name = path.stem
    # IR copies of a gene often carry /translation on one copy only.
    gene_proteins: dict[str, str] = {}
    for feat in rec.features:
        if feat.type == "CDS" and feat.qualifiers.get("translation") and feat.qualifiers.get("gene"):
            gene_proteins.setdefault(feat.qualifiers["gene"][0], feat.qualifiers["translation"][0])
    for idx, feat in enumerate(rec.features):
        if feat.type not in FEATURE_TYPES:
            continue
        gene = feat.qualifiers.get("gene", [""])[0]
        if feat.type == "repeat_region":
            gene = feat.qualifiers.get("note", [f"repeat_region_{idx}"])[0]
        if f"{ref_name}:{idx}" in flip_parts:
            k = flip_parts[f"{ref_name}:{idx}"]
            parts = list(feat.location.parts)
            parts[k] = FeatureLocation(parts[k].start, parts[k].end, strand=-(parts[k].strand or 1))
            feat.location = CompoundLocation(parts)
        elif feat.type == "CDS":
            _repair_part_strands(feat, rec.seq, gene_proteins.get(feat.qualifiers.get("gene", [""])[0], ""))
        seq = clean_seq(str(feat.extract(rec.seq)))
        if not seq:
            continue
        parts = list(getattr(feat.location, "parts", [feat.location]))
        features.append(
            ReferenceFeature(
                feature_id=f"{ref_name}:{idx}",
                feature=feat,
                sequence=seq,
                gene=gene,
                feature_type=feat.type,
                reference_name=ref_name,
                exon_count=max(1, len(parts)),
            )
        )
        if feat.type == "CDS" and len(parts) > 1:
            for exon_idx, part in enumerate(parts, start=1):
                exon_seq = clean_seq(str(part.extract(rec.seq)))
                if not exon_seq:
                    continue
                features.append(
                    ReferenceFeature(
                        feature_id=f"{ref_name}:{idx}:exon{exon_idx}",
                        feature=feat,
                        sequence=exon_seq,
                        gene=gene,
                        feature_type="CDS_exon",
                        reference_name=ref_name,
                        exon_index=exon_idx,
                        exon_count=len(parts),
                    )
                )
    return ReferenceRecord(path, rec, tuple(features), kmers=frozenset(sample_kmers(str(rec.seq))))


def _translation_agreement(feature: SeqFeature, sequence: Seq, protein: str) -> float:
    coding = str(feature.extract(sequence))
    coding = coding[: len(coding) - len(coding) % 3]
    if not coding or not protein:
        return 0.0
    translated = str(Seq(coding).translate(table=11)).rstrip("*")
    return sum(a == b for a, b in zip(translated, protein, strict=False)) / len(protein)


def _repair_part_strands(feature: SeqFeature, sequence: Seq, fallback_protein: str = "") -> None:
    """Flip the one part of a spliced CDS that the record put on the wrong strand.

    Several NCBI plastomes give trans-spliced rps12 exon 1 the strand of exons
    2-3 (Sorghum, Vitis, Eucalyptus, Helianthus, Ephedra): its sequence is then
    the reverse complement, and every genome annotated from such a reference
    inherits a reversed exon. The record's own /translation decides (that of
    another copy of the gene in the record when this one has none): a part is
    flipped only when that makes the spliced translation agree with it.
    """
    parts = list(getattr(feature.location, "parts", []))
    protein = feature.qualifiers.get("translation", [""])[0] or fallback_protein
    if len(parts) < 2 or not protein:
        return
    if _translation_agreement(feature, sequence, protein) >= _TRANSLATION_AGREEMENT:
        return
    best = None
    for k, part in enumerate(parts):
        flipped = [*parts[:k], FeatureLocation(part.start, part.end, strand=-(part.strand or 1)), *parts[k + 1 :]]
        candidate = SeqFeature(CompoundLocation(flipped), type=feature.type)
        score = _translation_agreement(candidate, sequence, protein)
        if score >= _TRANSLATION_AGREEMENT and (best is None or score > best[0]):
            best = (score, candidate.location)
    if best is not None:
        feature.location = best[1]


def _load_preprocessed_reference(path: Path, manifest: Path) -> ReferenceRecord:
    payload = json.loads(manifest.read_text())
    if payload.get("format_version") != PREPROCESSED_FORMAT_VERSION:
        return _parse_reference(path)
    features = []
    for item in payload.get("features", []):
        sequence = clean_seq(item.get("sequence", ""))
        if not sequence:
            continue
        qualifiers = {
            str(key): [str(value) for value in values]
            for key, values in item.get("qualifiers", {}).items()
        }
        template_type = item.get("template_type") or item.get("feature_type") or "gene"
        feature = SeqFeature(
            FeatureLocation(0, max(1, len(sequence)), strand=1),
            type=template_type,
            qualifiers=qualifiers,
        )
        features.append(
            ReferenceFeature(
                feature_id=str(item["feature_id"]),
                feature=feature,
                sequence=sequence,
                gene=str(item.get("gene", "")),
                feature_type=str(item["feature_type"]),
                reference_name=str(item.get("reference_name", path.stem)),
                exon_index=int(item.get("exon_index", 1)),
                exon_count=int(item.get("exon_count", 1)),
            )
        )
    record = SeqRecord(Seq(""), id=path.stem[:16], name=path.stem[:16])
    return ReferenceRecord(
        path=path,
        record=record,
        features=tuple(features),
        kmers=frozenset(str(kmer) for kmer in payload.get("kmers", [])),
    )


def rank_references(target_seq: str, references: tuple[ReferenceRecord, ...]) -> list[ReferenceRecord]:
    """References from closest to most distant (shared sampled k-mers, then length)."""
    target_kmers = sample_kmers(target_seq)
    scored = []
    for ref in references:
        ref_kmers = _reference_choice_kmers(ref)
        score = sum(1 for kmer in ref_kmers if kmer in target_kmers)
        length_delta = abs(_reference_length(ref) - len(clean_seq(target_seq)))
        scored.append((score, -length_delta, len(ref.features), ref))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [item[3] for item in scored]


def choose_reference(target_seq: str, references: tuple[ReferenceRecord, ...]) -> ReferenceRecord:
    return rank_references(target_seq, references)[0]


def exon_frame_offset(feature: ReferenceFeature) -> int:
    """Bases before the first complete codon in a biological-order exon."""
    parts = feature.feature.location.parts
    codon_start = int(feature.feature.qualifiers.get("codon_start", ["1"])[0]) - 1
    preceding = sum(len(part) for part in parts[: feature.exon_index - 1]) - codon_start
    return -preceding % 3


def build_plastome_queries(reference: ReferenceRecord) -> tuple[ReferenceQuery, ...]:
    queries: list[ReferenceQuery] = []
    for idx, ref_feature in enumerate(reference.features):
        if ref_feature.feature_type == "repeat_region":
            continue
        if ref_feature.feature_type in {"tRNA", "rRNA"}:
            group = "reference1"
            sequence = ref_feature.sequence
        elif ref_feature.feature_type == "gene":
            group = "reference2"
            sequence = ref_feature.sequence
        elif ref_feature.feature_type == "CDS" and ref_feature.exon_count == 1:
            group = "reference3"
            sequence = translate_nt(ref_feature.sequence)
        elif ref_feature.feature_type == "CDS_exon":
            group = "reference4"
            sequence = translate_nt(ref_feature.sequence[exon_frame_offset(ref_feature) :])
        else:
            continue
        if not sequence:
            continue
        queries.append(
            ReferenceQuery(
                query_id=f"q{idx:05d}",
                group=group,
                sequence=sequence,
                reference_feature=ref_feature,
            )
        )
    return tuple(queries)


def iter_fasta_inputs(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    for pattern in ("*.fasta", "*.fas", "*.fa", "*.fna"):
        yield from sorted(path.glob(pattern))


def clean_seq(seq: str) -> str:
    """Upper-case DNA with every non-ACGT symbol as N; whitespace dropped, length otherwise kept.

    Ambiguity codes used to be dropped, which shifted every coordinate after them
    (Panax ginseng NC_006290 carries 4 R and 1 Y: genes downstream of each came
    out one base further off).
    """
    return "".join(base if base in "ACGT" else "N" for base in seq.upper().replace("U", "T") if not base.isspace())


def sample_kmers(seq: str) -> set[str]:
    clean = clean_seq(seq)
    kmer_size = 31
    max_kmers = 1200
    if len(clean) < kmer_size:
        return {clean}
    circular = clean + clean[: kmer_size - 1]
    ranked = sorted(
        (
            _stable_kmer_hash(_canonical_kmer(circular[i : i + kmer_size])),
            _canonical_kmer(circular[i : i + kmer_size]),
        )
        for i in range(len(clean))
        if "N" not in circular[i : i + kmer_size]
    )
    if not ranked:
        return {clean[i : i + kmer_size] for i in range(0, len(clean) - kmer_size + 1)}
    return {kmer for _, kmer in ranked[:max_kmers]}


def _canonical_kmer(kmer: str) -> str:
    rc = str(Seq(kmer).reverse_complement())
    return kmer if kmer <= rc else rc


def _stable_kmer_hash(kmer: str) -> int:
    return zlib.crc32(kmer.encode("ascii"))


def _reference_choice_kmers(reference: ReferenceRecord) -> frozenset[str]:
    if reference.path.exists():
        stat = reference.path.stat()
        return _reference_choice_kmers_cached(
            str(reference.path.resolve()),
            stat.st_size,
            stat.st_mtime_ns,
        )
    if reference.record.seq:
        return frozenset(sample_kmers(str(reference.record.seq)))
    return reference.kmers or frozenset()


@lru_cache(maxsize=64)
def _reference_choice_kmers_cached(path_text: str, size: int, mtime_ns: int) -> frozenset[str]:
    del size, mtime_ns
    rec = SeqIO.read(path_text, "genbank")
    return frozenset(sample_kmers(str(rec.seq)))


def _reference_length(reference: ReferenceRecord) -> int:
    if reference.record.seq:
        return len(reference.record.seq)
    if reference.path.exists():
        return _reference_length_cached(
            str(reference.path.resolve()), reference.path.stat().st_mtime_ns
        )
    return 0


@lru_cache(maxsize=64)
def _reference_length_cached(path_text: str, mtime_ns: int) -> int:
    del mtime_ns
    return len(SeqIO.read(path_text, "genbank").seq)


def translate_nt(seq: str) -> str:
    usable = clean_seq(seq)
    usable = usable[: len(usable) - (len(usable) % 3)]
    if not usable:
        return ""
    return str(Seq(usable).translate(table=11, to_stop=False)).rstrip("*").replace("*", "X")


def build_preprocessed_reference_cache(reference_dir: Path, output_dir: Path) -> dict[str, int]:
    reference_files = sorted(reference_dir.glob("*.gb")) + sorted(reference_dir.glob("*.gbk"))
    if not reference_files:
        raise FileNotFoundError(f"No plastome GenBank references found in {reference_dir}")

    from .blast import write_fasta

    output_dir.mkdir(parents=True, exist_ok=True)
    query_counts: dict[str, int] = {}
    parsed = repair_reversed_trans_parts([_parse_reference(path) for path in reference_files])
    for reference_file, reference in zip(reference_files, parsed, strict=True):
        queries = build_plastome_queries(reference)
        ref_dir = output_dir / reference_file.stem
        ref_dir.mkdir(parents=True, exist_ok=True)
        payload = _reference_manifest_payload(reference)
        (ref_dir / "features.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        for group in ("reference1", "reference2", "reference3", "reference4"):
            group_queries = [query for query in queries if query.group == group]
            write_fasta(
                ref_dir / f"{group}.fasta",
                ((query.query_id, query.sequence) for query in group_queries),
                clean=False,
            )
        query_counts[reference_file.stem] = len(queries)
    return query_counts


def _reference_manifest_payload(reference: ReferenceRecord) -> dict:
    source = reference.path
    return {
        "format_version": PREPROCESSED_FORMAT_VERSION,
        "reference_name": source.stem,
        "source_file": source.name,
        "source_sha256": _sha256(source),
        "source_size": source.stat().st_size,
        "kmers": sorted(reference.kmers or sample_kmers(str(reference.record.seq))),
        "features": [_feature_payload(feature) for feature in reference.features],
    }


def _feature_payload(feature: ReferenceFeature) -> dict:
    return {
        "feature_id": feature.feature_id,
        "gene": feature.gene,
        "feature_type": feature.feature_type,
        "template_type": feature.feature.type,
        "qualifiers": {
            key: [str(value) for value in values]
            for key, values in feature.feature.qualifiers.items()
        },
        "sequence": feature.sequence,
        "reference_name": feature.reference_name,
        "exon_index": feature.exon_index,
        "exon_count": feature.exon_count,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
