# Released annotation contract

This report defines the annotation functionality released in OrganelleVerse
v0.0.1. The claim is deliberately bounded: the package provides a reproducible,
fail-closed mitochondrial annotation process for the scope below, exact compound
GenBank extraction, and the same implementation for Python and Agent callers. It
does not claim 100% biological accuracy, universal taxonomic coverage, or
equivalence to PMGA/PGA.

## Released scope

`ov.annotation` is the canonical v1 facade. It accepts immutable
`OrganelleGenome` inputs and returns `OrganelleResult` objects whose annotation,
run manifest, command evidence, tool versions, database hashes, and errors are
machine-readable.

The released backend set is exactly `auto | mitochondrion`. `auto` selects the
mitochondrial backend only for mitochondrial genomes. The backend requires
non-empty species metadata and performs these requested stages:

- protein-coding genes: packaged HMM/BLAST references with `blastn`,
  `makeblastdb`, and `tblastn`;
- tRNA: packaged plant-mitochondrial CM/HMM profiles, pyhmmer candidate
  filtering, and the OrganelleVerse native CYK/CM implementation;
- rRNA: packaged references and pyhmmer;
- canonical validation: exact zero-based half-open location parts, compound
  feature order, parent integrity, coordinate bounds, requested-stage
  completion, and CDS translation consistency;
- persistence: content-addressed canonical JSON and a semantic run manifest,
  followed by atomic materialization to JSON, GenBank, GFF3, and FASTA.

GenBank `join`, `order`, complement, fuzzy endpoints, and biological part order
are preserved. On the fixed Arabidopsis fixture, the compound `nad5` CDS is
extracted as 2,010 bp and matches Biopython byte-for-byte; it is never reduced to
its 333,312 bp bounding interval. `AnnotationFeature.genomic_position()` maps a
spliced biological offset back to its exact genomic base.

Required Python packages are Biopython 1.85 or newer and pyhmmer 0.7 or newer.
Required executables are only BLAST+ (`blastn`, `makeblastdb`, `tblastn`).
tRNA annotation does not search for or invoke tRNAscan-SE, Infernal `cmsearch`,
ARAGORN, or a BLAST fallback. Native zero-hit output remains an honest zero-hit
result for QC rather than changing algorithms according to the host PATH.
Missing required tools, non-zero commands, timeouts, malformed output,
incomplete requested stages, and changed input artifacts produce structured
failures; they are not converted to empty successful results.

## Fixed release gate

The non-skippable gate pins fixture content with
`tests/release/annotation_fixture_manifest.json`. It verifies checksums before
execution and then runs the real Python facade, the real Agent JSON adapter, and
a deterministic Python rerun. All three runs use the released backend and real
BLAST+ tools; tRNA execution is package-native.

The 2026-07-13 gate used BLAST+ 2.14.1+, tRNAscan-SE 2.0.12, and Infernal 1.1.4.
The Python, Agent, and isolated-process rerun produced identical canonical
document IDs and file SHA-256 values. The exact IDs are recorded by each
generated JSON report instead of being treated as permanent biological
constants. These figures predate the native-engine release and are retained
only as an external-engine historical baseline; they are not release evidence
for the native implementation. The same pinned gate must be rerun before native
tRNA metrics are published.

| Historical external-engine metric | Observed | Release condition |
|---|---:|---:|
| validated unique PCG recall vs Ranunculus PMGA | 36/39 (0.9231) | >= 0.85 |
| observed PCG name precision vs Ranunculus PMGA | 36/36 (1.0000) | >= 0.95 |
| recovered unique rRNA names | 3/3 | == 3 |
| evaluable PMGA tRNA loci supported | 110/111 (0.9910) | >= 0.85 |
| matched observed tRNA precision | 110/126 (0.8730) | >= 0.80 |
| tRNA hit ratio | 126/111 (1.1351) | <= 1.25 |
| invalid observed tRNA lengths | 0 | == 0 |
| canonical structural errors | 0 | == 0 |
| Arabidopsis compound `nad5` length | 2,010 bp | == 2,010 bp |

These measurements characterize the pinned fixtures and current reference/tool
versions. PCG and rRNA rows are unique-name comparisons, while tRNA uses
one-to-one, strand/anticodon-compatible reciprocal coordinate overlap. The
baseline contains 23 intron-spanning tRNA features of 599--2,597 bp; they are
reported but excluded from the 50--150 bp locus comparison. Nine malformed
CDS candidates are rejected before publication, including frame-shifted,
terminal-incomplete, or internal-stop candidates. Their strand, exon parts,
coordinates, issue codes, and messages are retained in
`metrics.rejected_cds_candidates`; because candidates were rejected and core
genes remain missing, the scientific result status is `warning`, not `ok`.
These values must not be presented as universal accuracy percentages.

The deterministic rerun executes in a fresh Python process, so the identity
check detects hash-seed and import-state drift. Injected post-backup failures
confirm that existing annotation and extraction outputs remain byte-identical.
A clean-wheel probe uses an isolated venv, installs declared dependencies, runs
`pip check`, imports the full facade outside the source checkout, lists the
production catalog, parses the packaged HMM database, compares synthetic Python
and Agent results, completes annotate/extract/write contracts, and reruns the
fixed Ranunculus annotation through the installed wheel's Agent adapter. The
installed artifact SHA-256 and document ID must match the source gate exactly.

Run the complete gate from the repository root:

```bash
# (the release-gate runner is not part of this repository)
```

BLAST+ executables should be on `PATH`. If BLAST+ is installed under a software
prefix, add `--software-dir /path/to/software`; the runner still fails if any
required executable cannot be resolved. It publishes deterministic JSON and
Markdown reports together under `--work-dir/annotation-release-report/` and
exits non-zero for a missing fixture,
checksum change, missing tool, skipped release test, failed threshold,
non-deterministic artifact, atomicity failure, or clean-wheel failure.

Ordinary offline pytest runs exclude `release_annotation`; maintainers do not
interpret that deselection as release evidence. The runner explicitly selects
the marker and rejects skips.

## Annotation quality control

`ov.qc.annotation(result)` audits one canonical `annotation.annotate` Result
without rerunning annotation. It evaluates two questions: are the source Result,
annotation artifact, run manifest, requested stages, coordinates, feature
hierarchy, CDS translations, and deterministic exports internally consistent;
and which expected profile genes were recovered, duplicated, partial, explicitly
exempted, or rejected. It does not compute a 0--100 score, a grade, or a
universal completeness percentage.

The decision is fixed-precedence and never forces `ready`:

- any failed contract check -> `not_ready`;
- otherwise any warning (rejected CDS candidate, missing expected core gene,
  duplicate, partial/pseudo feature, or explicit translation exception) ->
  `needs_review`;
- otherwise `ready`.

Reference-profile recovery is reported under that exact name against the fixed
`organelleverse.plant-mito-pcg.v1` profile (24 expected core protein-coding
genes). The 18 variable ribosomal-protein genes are listed separately, never
enter the expected-profile denominator, and never warn when absent. A missing
expected gene is a review signal, not an assertion that the genome is
biologically incomplete; plant mitochondrial gene content varies by lineage.

`ov.qc.write(qc_result, output=...)` is the single QC writer. It dispatches on
the input Result operation ID: `qc.assembly` uses the assembly-QC writer and
`qc.annotation` uses the annotation-QC writer; every other Result fails before
publication with `qc.write_input_contract`. A directory output contains:

- `annotation_qc_summary.md` -- human summary;
- `annotation_qc_report.json` -- the canonical report;
- `annotation_qc_run_manifest.json` -- the content-addressed run manifest;
- `summary.csv`, `checks.csv`, `features.csv`, `gene_profile.csv`, and
  `rejected_cds_candidates.csv` -- one row per record.

Single-file `.md`, `.json`, `.csv`, and `.xlsx` outputs follow the existing QC
writer convention; the `.csv` single file is the summary table and the `.xlsx`
carries all five tables. Publication is atomic and refuses conflicting existing
content unless it is byte-identical and reusable.

### Python caller

```python
from pathlib import Path

import organelleverse as ov

annotation_result = ov.annotation.annotate(genome, backend="mitochondrion")
annotation_result.raise_for_failure()
qc_result = ov.qc.annotation(annotation_result)
print(qc_result.metrics["qc_decision"])
print(qc_result.metrics["missing_profile_genes"])
written = ov.qc.write(qc_result, output=Path("sample/annotation_qc"))
```

### Agent caller

The released annotation-QC operation IDs are:

- `qc.annotation`: requires `read_files` and `write_files`; accepts a non-failed
  `annotation.annotate` Result and returns a `qc.annotation` Result whose
  `metrics["qc_decision"]` is one of `ready`, `needs_review`, `not_ready`, or
  `insufficient_evidence`;
- `qc.write`: requires `read_files` and `write_files`; dispatches to the
  annotation-QC writer for a `qc.annotation` Result.

```json
{
  "operation_id": "qc.annotation",
  "input": { "annotation.annotate Result, including its artifacts" },
  "parameters": {}
}
```

`qc.annotation` has no user-tunable scientific parameters. The exact profile and
decision rules are versioned in the annotation-QC policy and recorded in every
report and run manifest.

### Release evidence

The fixed release gate runs `qc.annotation` through the Python facade and the
Agent JSON adapter against the same direct Ranunculus annotation Result,
requires byte-identical results, requires deterministic reuse of exactly one
managed run, materializes the full writer bundle, reparses the report, run
manifest, and every CSV, and requires zero failed contract checks. The recorded
`annotation_qc_decision` is the honest result for the pinned fixture; it is
`needs_review` whenever rejected CDS candidates or missing expected core genes
are present and is never forced to `ready`. The latest focused suites (306
quality-control and operations tests, 18 release-unit tests, and 93 annotation
tests) and the complete default suite pass; the real gate passes only when the
pinned Ranunculus fixture, Arabidopsis GenBank, and BLAST+ environment are
available.

## Python caller

```python
from pathlib import Path

import organelleverse as ov
from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.genome import OrganelleGenome, OrganelleMetadata

fasta = Path("mitochondrion.fasta")
genome = OrganelleGenome(
    organelle="mitochondrion",
    sequence=ArtifactRef.from_path(fasta, kind="sequence", format="fasta"),
    metadata=OrganelleMetadata(species="Ranunculus species"),
)
result = ov.annotation.annotate(
    genome,
    backend="auto",
    threads=4,
    call_trna=True,
    call_rrna=True,
)
result.raise_for_failure()
# A warning still carries a validated artifact; inspect rejected candidates
# and missing core genes before deciding whether to publish it.
if result.status == "warning":
    print(result.metrics["rejected_cds_candidates"])
published = ov.annotation.write(result, output=Path("published-annotation"))
```

To extract from an existing canonical JSON or GenBank annotation, construct an
`OrganelleGenome` with an annotation `ArtifactRef`, then call
`ov.annotation.extract(...)`. Extraction does not rerun annotation.

## Agent caller

The production operation IDs are:

- `annotation.annotate`: requires `read_files`, `write_files`, and `subprocess`;
- `annotation.extract`: requires `read_files` and `write_files`;
- `annotation.write`: requires `read_files` and `write_files`.

The request below is the JSON-compatible shape accepted by `invoke_json`; the
`input` object includes the complete serialized `ArtifactRef` with SHA-256 and
size.

```json
{
  "operation_id": "annotation.annotate",
  "input": {
    "schema_version": "organelleverse.genome.v1",
    "kind": "genome",
    "organelle": "mitochondrion",
    "sequence": {
      "schema_version": "organelleverse.artifact.v1",
      "kind": "sequence",
      "uri": "/absolute/path/mitochondrion.fasta",
      "format": "fasta",
      "media_type": "application/octet-stream",
      "sha256": "<64 lowercase hexadecimal characters>",
      "size_bytes": 123456,
      "validated": true
    },
    "annotation": null,
    "metadata": {
      "kind": "metadata",
      "species": "Ranunculus species",
      "accession": "",
      "genetic_code": 1,
      "assembly_type": "",
      "plastid_type": "",
      "source": ""
    }
  },
  "parameters": {
    "backend": "auto",
    "threads": 4,
    "call_trna": true,
    "call_rrna": true
  }
}
```

Agents can discover the exact parameter contract with
`operations.parameter_schema("annotation.annotate")`. The schema exposes only
`auto` and `mitochondrion`; permission denial and dependency failure occur
before scientific execution where possible. Transport success is distinct from
scientific success: callers must inspect `response.ok` and then
`response.result.status`. A `warning` response is transport-successful and
contains a validated artifact, but its flags and metrics must remain visible to
the next step; `annotation.write` preserves the source warning state.

## Explicit limitations

- Pinus mitochondrial tRNA is outside the released scope. Current fixed
  evidence is below threshold, so a canonical Pinus request with tRNA enabled
  fails with `unsupported_annotation_scope`; PCG/rRNA can be requested without
  tRNA.
- Plastid annotation is not part of this release. Its former implementation and
  data bundle are not distributed and do not appear in Agent JSON Schema.

The process contract and provenance prevent silent execution, parsing, and
artifact failures. They do not prove that every biological prediction for every
species is correct.
