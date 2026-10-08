"""Build whole-genome nucleotide recruitment seeds from pinned plant GenBank records.

No assembly truth or reads enter this builder. See docs/assembly/ovasm-seeddb.md.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from Bio import SeqIO

EXCLUDED_SPECIES = {"Arabidopsis thaliana", "Oryza sativa", "Salvia miltiorrhiza"}


def build_database(sources: dict[str, list[Path]], output: Path, version: str) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "name": "OVASM SeedDB",
        "version": version,
        "format_version": 1,
        "scope": "representative land-plant whole organelle genome recruitment seeds",
        "excluded_unverifiable_source": "Coffea arabica annotation record 163884508445366 has no versioned nucleotide accession",
        "excluded_benchmark_species": sorted(EXCLUDED_SPECIES),
        "databases": {},
    }
    for organelle, paths in sorted(sources.items()):
        references, seeds = [], []
        seen = set()
        for path in sorted(paths):
            for record in SeqIO.parse(path, "genbank"):
                organism = record.annotations["organism"]
                if record.id == "163884508445366":
                    continue
                if any(
                    organism == name or organism.startswith(name + " ") for name in EXCLUDED_SPECIES
                ):
                    continue
                if not re.fullmatch(r"[A-Z][A-Z_]*[0-9]+\.[0-9]+", record.id):
                    raise ValueError(f"Source lacks a versioned nucleotide accession: {record.id}")
                if record.id in seen:
                    raise ValueError(f"Duplicate source accession: {record.id}")
                seen.add(record.id)
                source = next(f for f in record.features if f.type == "source")
                declared = source.qualifiers.get("organelle", [])
                expected = (
                    {"mitochondrion"}
                    if organelle == "mitochondrion"
                    else {"plastid:chloroplast", "plastid"}
                )
                if not expected.intersection(declared):
                    raise ValueError(f"Wrong organelle for {record.id}: {declared}")
                reference = {
                    "accession": record.id,
                    "organism": organism,
                    "taxonomy": record.annotations.get("taxonomy", []),
                    "source_url": f"https://www.ncbi.nlm.nih.gov/nuccore/{record.id}",
                    "genome_length_bp": len(record),
                    "seeds": [],
                }
                sequence = str(record.seq).upper()
                seeds.append((record.id, sequence))
                reference["seeds"] = [
                    {
                        "id": record.id,
                        "start_1based": 1,
                        "end_1based": len(sequence),
                        "strand": 1,
                        "length_bp": len(sequence),
                    }
                ]
                references.append(reference)
        if not seeds:
            raise ValueError(f"No sources for {organelle}")
        filename = f"landplants.{organelle}.fasta"
        with (output / filename).open("w", encoding="ascii", newline="\n") as handle:
            for identifier, sequence in seeds:
                handle.write(f">{identifier}\n")
                for start in range(0, len(sequence), 80):
                    handle.write(sequence[start : start + 80] + "\n")
        manifest["databases"][organelle] = {
            "file": filename,
            "reference_count": len(references),
            "sequence_count": len(seeds),
            "total_bases": sum(len(seq) for _, seq in seeds),
            "references": references,
        }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mitochondrion-genbank", required=True, type=Path)
    parser.add_argument("--plastid-genbank", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--version", default="v20261001")
    args = parser.parse_args()
    manifest = build_database(
        {
            "mitochondrion": list(args.mitochondrion_genbank.glob("*.gb")),
            "plastid": list(args.plastid_genbank.glob("*.gb")),
        },
        args.output,
        args.version,
    )
    for organelle, database in manifest["databases"].items():
        print(
            organelle,
            database["reference_count"],
            database["sequence_count"],
            database["total_bases"],
        )


if __name__ == "__main__":
    main()
