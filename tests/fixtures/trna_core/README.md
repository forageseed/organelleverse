# tRNA Core Calibration Fixtures

These fixtures are OrganelleVerse-owned test inputs for the native tRNA core.

- `positives.tsv` contains 134 PMGA-supported Ranunculus mitochondrial tRNA loci.
- `hard_negatives.tsv` contains 134 current `OrganelleVerse-native` candidate loci treated as hard negatives because they are not supported by the PMGA baseline in the benchmark table.
- The builder script requires explicit CLI paths and must not bake developer-specific filesystem locations into the repository.

For local refresh, run `scripts/annotation_benchmarks/build_trna_core_fixtures.py` with explicit FASTA, PMGA GFF, benchmark TSV, and output paths.
