# Direction-aware assembly/reference comparison

`qc.compare_assembly_to_reference` compares independent candidate FASTA records
against one reference FASTA record. It returns an `OrganelleResult` in memory;
there is no output directory argument. Requires the existing optional `align`
extra (`mappy`; Linux/macOS/WSL). Missing mappy raises `OrganelleDependencyError`
with installation instructions. No external executable is needed by this API.

```python
from pathlib import Path
from organelleverse import qc

result = qc.compare_assembly_to_reference(
    Path("candidates.fa"), Path("reference.fa"), organelle="plastid",
)
for comparison in result.metrics["comparisons"]:
    print(comparison["sample"], comparison["raw"], comparison["normalized"])
    print(comparison["normalization"])
```

Each record is a **separate candidate**, not one contig of a pooled assembly.
Convert GenBank references to FASTA with the existing Biopython reader first.
The `generic` default compares each whole sequence without rotation or
normalization; its `normalized` and `normalization` fields are null.

For plastids, the existing circular IR detector and
`comparative.normalize_plastome_orientation` identify LSC/IRb/SSC/IRa, move the
origin to LSC, and orient LSC/SSC to the reference. The reference is also rotated
to LSC. Both raw and normalized comparisons map **each quadrant independently to
the whole reference** using mappy `asm5`. This prevents an IR-spanning chain from
suppressing the alternate SSC isomer: simply retaining secondary whole-genome
hits is insufficient for the NOVOPlasty Option 1 example. This partitioning is
the normal plastid algorithm, not a retry of low-coverage cases.

Returned fields:

- `raw` / `normalized`: quadrant summaries in their respective coordinate
  systems. `reference_coverage` is the union of all emitted forward and reverse
  reference intervals divided by the reference length. Overlaps count once.
  Alignment spans include reference deletions; this is breadth, not exact-base
  matching or read depth.
- `alignment_identity`: sum of mappy matching bases divided by sum of alignment
  block lengths, **including secondary and repeat alignments**. Repeat copies can
  contribute several times to this denominator. This is not one-to-one genome
  identity; `comparative.compute_genome_identity` provides a separate anchored
  one-to-one window statistic. No alignment yields zero coverage and null
  identity, with warning status.
- `alignment_blocks` / `inverted_blocks`: all blocks / their reverse-strand
  subset, including quadrant name, primary/secondary status and coordinates.
  Reverse IR cross-mappings are ambiguous repeat evidence, not a biological
  inversion call. Single-copy `anchor_strands` in `normalization` identify the
  LSC/SSC orientation relative to the reference.
- `whole_sequence_raw` / `whole_sequence_normalized`: whole-sequence summaries
  with the same strand and identity rules, retained so chaining effects remain
  visible. They are not mixed into the quadrant identity denominator.
- `normalization` / `reference_normalization`: original and normalized regions,
  anchor strands, and ordered replayable rotation/reverse-complement operations.
  Coordinates are zero-based, half-open; circular query intervals are split at
  the origin. Region descriptions use start plus length modulo sequence length.
- `unresolved`: samples lacking a quadripartite structure or unambiguous anchors.
  Such samples retain whole-sequence raw evidence but have null quadrant
  comparisons. A partial cohort warns; wholly unresolved inputs or an unresolved
  reference fail. They are never reported as normalized successes.

Boundary precision remains that of the existing seed/extension IR detector.
Independent quadrant alignment can leave a few terminal bases unmapped, and
normalization does not guarantee increasing coverage for divergent samples.
`asm5` is a close-reference comparison; absence of a hit is not proof of absent
sequence. Generic mode counts all emitted strands, but does not resolve an
arbitrary rearrangement or reconstruct a circular origin. These are sequence
comparisons, not evidence of graph correctness, repeat copy number or assembly
completeness without reference bias.

## Existing entry points and integration

`qc.assembly` consumes an assembly Result and evaluates read-to-assembly depth,
base errors, markers and graph support. Its reference positions are assembly
positions; it has no supplied gold-genome comparison contract. Assembly backends
accept references/seeds for assembly and annotation, without a common reference
breadth/identity evaluator. Consequently this comparison is a separate QC
capability, following `qc.read_statistics`' named-parameter Result API.

`comparative.compute_genome_identity` performs mappy anchoring, one-to-one block
selection and edlib window identity; it has different denominators and does not
expose raw versus normalized plastome QC. The existing normalization capability
is reused unchanged.

The graph scorer has the
linear gold/candidate evaluator and junction-alignment evaluator. The linear
summary now unions intervals separately per reference record, includes unaligned
reference records in the denominator, and reports reverse blocks explicitly.
Both CLI call sites use `core.external.run_external`. The graph scoring algorithm
is unchanged. An empty linear alignment has null identity and true reference
length, rather than fabricated zero-length reference statistics.

`validate_oriented_reference.py` runs the real-data acceptance cases and writes
raw/normalized JSON, replayed FASTA, quadrant FASTA, independent minimap2 PAF,
commands and stderr to a newly created validation directory. The CLI cross-check
uses an independent set-of-reference-positions breadth implementation. Only the
explicitly labelled legacy reproduction uses `--secondary=no`; SSC and all
acceptance comparisons keep secondary alignments.

## Real-data verification (2026-09-27)

mappy 2.31 and independent minimap2 CLI 2.31-r1302 gave identical quadrant
covered-base counts and block-weighted identities in all six before/after
comparisons. The reference for both NOVOPlasty candidates is NC_001320.1
(134,525 bp). The Hordeum reference is `Hordeum_vulgare_var__distichon`
(136,462 bp); its query is `Hordeum_brevisubulatum_subsp__`.

| Candidate | Whole raw breadth | Quadrant raw breadth | Quadrant normalized breadth | Quadrant identity raw → normalized |
| --- | ---: | ---: | ---: | ---: |
| NOVOPlasty Option 1 | 90.793533% | 99.996283% | 100.000000% | 99.837705% → 99.837709% |
| NOVOPlasty Option 2 | 100.000000% | 99.996283% | 100.000000% | 99.837705% → 99.837709% |
| Hordeum pair | 99.986810% | 99.940643% | 99.938444% | 98.916476% → 98.916458% |

Independent SSC-only CLI alignments found Option 1 on `-`, Option 2 on `+`
(both 12,307/12,320 matching/aligned bases; 12,311/12,311 reference bases covered).
The Hordeum query SSC was on `-`, with 12,517/12,690 matching/aligned bases and
12,634/12,663 reference bases covered. These coordinates use the existing IR
detector's SSC boundaries, explaining their difference from a manually selected
SSC interval in the earlier NOVOPlasty report.

Option 1 rotates left 20,757 bases and reverse-complements normalized SSC
[101413, 113732); Option 2 only rotates left 20,757 bases. The Hordeum query
rotates left 12 bases and reverse-complements SSC [102719, 115460). The Hordeum
quadrant breadth decreases by three bases after normalization; no threshold was
adjusted and no endpoint patch was added to erase this difference.
