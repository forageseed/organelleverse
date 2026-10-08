# Assembly QC release report

**Date:** 2026-07-28
**Operations:** `qc.assembly`, `qc.write`
**Policy:** `organelleverse.assembly-qc-policy.v5`
**Report schema:** `organelleverse.assembly-qc.v1`
**Operation versions:** `qc.assembly` 1.5; `qc.write` 1.2

This report records what the assembly-quality-control release gate verifies and
what it does not. It is written to be factual: it does not claim a real-tool
gate ran when it did not.

## Operations shipped

`qc.assembly` assesses a `assembly.assemble` Result against its graph, source
reads, target-organelle evidence, and recorded provenance, and writes one
content-addressed canonical QC record under a managed workspace. `qc.write`
reverifies that canonical record and atomically materializes a directory bundle
or a single `.md`/`.json`/`.csv`/`.xlsx` view without recomputing QC. Both are
registered for direct Python, `OperationRegistry`, and Agent JSON invocation.

## What is verified hermetically

The 190-test hermetic suite (`tests/quality_control`) runs in the default
`pytest` invocation and covers:

- canonical Result, manifest, and artifact integrity, including FASTA/GFA
  referential integrity and write-time digest re-verification;
- read-mapping, junction-support, graph, repeat, marker-identity, and
  organelle-specific (plastid IR; mitochondrial multipartite) evidence;
- the four-state decision ladder (`ready` / `needs_review` / `not_ready` /
  `insufficient_evidence`) and compact metric keys, with `null` for unavailable
  numerics and no 0–100 score or A–F grade;
- managed-workspace reuse with full digest re-verification;
- directory and suffix-selected materialization, CSV/XLSX/BED projection,
  empty-table assessment-status semantics, and destination-conflict handling;
- `OperationSpec` discovery with closed parameter schemas and stable typed error
  codes.

Task 3 additions in this release:

- **Agent JSON equivalence** (`tests/quality_control/test_agent_json.py`):
  independent runs through direct Python, the registry, and `invoke_json` agree
  on every scientific report field; the typed `qc.input_contract_violation` and
  `qc.write_input_contract` envelopes are identical across all three paths and
  serialize unchanged through the Agent adapter; side-effect grant denial
  returns `permission.denied` without invoking science.
- **Error injection** (`tests/quality_control/test_error_injection.py`): a
  contradicted junction blocks release end-to-end (`not_ready`,
  `qc.contradicted_junction`, counted in metrics), and the marker identity
  ladder is verified at the `interpret_organelle` contract seam. Read and graph
  support alone no longer establishes target identity: when marker profiles
  were unavailable, identity is `not_assessed` and the overall decision is
  `insufficient_evidence`. An unintegrated conflicting marker fails
  (`qc.conflicting_organelle_marker`); explicit boundary-supported MTPT context
  can still be `context_supported`.
- **Release-pinned marker evidence**: `qc.assembly` reuses the assembly
  environment manager's content-addressed OatkDB `v20230921` cache (source
  commit `75e8db0ac4a7d508a9a518d900876003ceb70737`, MIT), scans the
  `embryophyta_mito.fam` and `embryophyta_pltd.fam` DNA HMMs directly with
  `nhmmer`, and records both exact profile artifacts in the report and run
  manifest. A real-profile smoke check recovered the expected mitochondrial
  `atp1` and plastid `accD` consensus targets. The report now preserves the
  exact expected-profile count, complete-profile count, recovery fraction, and
  duplicate complete-profile count. This is HMM-profile recovery, not genome
  completeness.
- **Read-supported base errors**: assembly-centered `small_collapse`,
  `small_expansion`, and substitution candidates require at least three
  primary-alignment observations, at least five assessed confident reads, and
  an observation fraction of at least 0.8. Each candidate records assessed
  depth, support fraction, supporting-library count, and evidence identity.
  The support fraction is not relabelled as calibrated confidence. The large
  BAM intermediates are still removed.
- **FASTA- and repeat-aware depth**: aligned bases without FASTQ quality values
  contribute to depth; bases with explicit FASTQ qualities still have to meet
  the base-quality floor. Ambiguous primary, secondary, and supplementary
  alignments contribute only to coverage, while QV and base-error candidates
  remain restricted to high-confidence primary alignments. The report exposes
  both tracks separately: exact any-alignment breadth/zero bases for
  repeat-aware gap evidence, and confident-primary breadth/zero bases for
  ambiguity review. It does not collapse them into one misleading coverage
  value.
- **Descriptive statistics and honest names**: sequence count, total length,
  largest sequence, N50/L50, GC, ambiguous bases, per-sequence depth
  percentiles, per-library mapping summaries, GFA topology, and supported
  alternatives are preserved in JSON, CSV, XLSX, and Markdown outputs.
  `read_assembly_agreement_qv` is explicitly a reads-versus-assembly agreement
  value, not Merqury consensus QV; `parallel_edge_count` is not called a bubble
  count; repeat `path_occurrences` is not called inferred copy number.
- **Conservative MTPT interpretation**: plastid marker evidence on a
  read-supported sequence that also has mitochondrial markers is reported as a
  target-linked probable MTPT and requires review when graph boundaries cannot
  resolve it. A separate plastid-only sequence still fails as conflicting
  organelle evidence.
- **Target-supported mixed markers**: a read-supported plastid sequence with
  direct plastid markers is not reclassified as mitochondrial contamination
  solely because conserved mitochondrial profiles also hit it. Those hits
  remain an explicit warning. A conflicting-only sequence still fails.

## Four-backend release gate (environment-controlled)

`tests/release/test_assembly_qc_{oatk,himt,getorganelle,pmat}.py` are marked
`release_assembly_qc` and excluded from the default run. Each gate consumes a
**real, already-produced assembly `Result`** (a serialized `result.json` from
that backend's assembly release gate) and pipes it through
`assembly Result -> qc.assembly -> qc.write(directory) -> reparse all outputs`.
The gates **do not re-run an assembly backend**; they reuse a successful
assembly Result, which is the design's required integration path. Each wrapper
also checks the exact recorded backend identity, so one backend Result cannot
silently satisfy another backend's gate.

### Policy-v5 real run (2026-07-28)

All four real assembly Results completed the policy-v5
`qc.assembly -> qc.write -> strict reparse` gate:

```text
4 passed, 2828 deselected in 470.26 s
```

| Backend | QC time | Decision | Any coverage / zero | Confident coverage / zero | Base-error candidates (v4 → v5) | Complete target profiles | Failures | Warnings |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: |
| Oatk | 147.77 s | `needs_review` | 1.0 / 0 | 1.0 / 0 | 2,578 → 35 | 40 / 81 | 0 | 3 |
| HiMT | 55.77 s | `needs_review` | 1.0 / 0 | 1.0 / 0 | 214 → 2 | 39 / 81 | 0 | 3 |
| GetOrganelle | 145.42 s | `needs_review` | 1.0 / 0 | 0.669131 / 51,112 | 1 → 1 | 102 / 130 | 0 | 3 |
| PMAT2 | 116.41 s | `needs_review` | 1.0 / 0 | 0.999997 / 1 | 180 → 1 | 39 / 81 | 0 | 4 |

The support-fraction/depth correction removed most low-fraction read errors:
Oatk retained 35 candidates, HiMT 2, and PMAT2 1. GetOrganelle retained its
single candidate. These are still review candidates, not validated assembly
errors. Full any-alignment coverage remained 1.0 with zero missing bases for
all four backends. GetOrganelle's confident-track gaps are retained as
repeat/ambiguity review evidence rather than misclassified as assembly gaps.

The marker values are recovery of complete target HMM profiles from the pinned
profile bundle, not genome-completeness percentages. Optional meryl evidence
was `not_assessed` because meryl was not available in this resolved QC
environment.

### Historical policy-v4 run

Before the policy-v5 correction, all four assembly Results passed the complete
real `qc.assembly -> qc.write` chain under policy v4. These numbers are retained
as historical evidence only; they are not a policy-v5 release result. Every
report returned
`needs_review`, not a fabricated `ready`, and every reported base had read
coverage:

| Backend | QC gate | Any-alignment coverage | Zero bases | Read/assembly agreement QV | Failures | Warnings |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Oatk | 146.54 s | 1.0 | 0 | 23.8394 | 0 | 3 |
| HiMT | 55.24 s | 1.0 | 0 | 22.4625 | 0 | 3 |
| GetOrganelle | 144.23 s | 1.0 | 0 | 27.0046 | 0 | 2 |
| PMAT2 | 116.64 s | 1.0 | 0 | 26.0319 | 0 | 3 |

Policy v4 retained the following raw support-count candidates:
2,578 for Oatk, 214 for HiMT, 1 for GetOrganelle, and 180 for PMAT2. The
mitochondrial reports also retain probable MTPT warnings; the plastid report
retains 114 conflicting-profile hits as review evidence.

The GetOrganelle gate additionally exposed and closed two integration defects:
long Conda prefixes now probe and execute with the managed prefix's own Python,
and the prepared database is represented by its canonical inventory JSON
rather than an invalid directory-shaped `ArtifactRef`. The official
Arabidopsis paired-end assembly and verified-reuse gate passed in 94.91 s after
those fixes; the final stable-cache rerun passed in 100.04 s.

### Verified gate mechanics

The gate mechanics are also covered hermetically. They verify exact source
backend provenance, run the full
`load_result -> qc.assembly -> qc.write` chain, strictly reparse the report and
run manifest, parse every CSV including explicit empty-table status, check the
Markdown summary, and recompute every materialized artifact digest. A real,
successful backend Result must still produce `ready` or `needs_review`.

### Low-depth PMAT2 zero-coverage guarantee

The PMAT2 gate additionally requires assessed coverage and exactly zero
zero-coverage bases. `None` is rejected because it means `not_assessed` and
cannot prove that policy mapping filters avoided an artificial interval. This
assertion passed on the real low-depth Malus PMAT2 Result. A read-supported
expansion candidate at a graph boundary remains in the report as a warning;
ambiguous alignments establish coverage there but do not contribute to QV or
base-error support.

## How to run the real gates

In the release environment, produce each backend's assembly Result (via its
`release_assembly_*` gate, which writes `result.json`), then point the matching
environment variable at it:

```bash
ORGANELLEVERSE_QC_OATK_RESULT=/path/to/oatk_result.json \
  PYTHONPATH=src pytest -m release_assembly_qc -q
ORGANELLEVERSE_QC_HIMT_RESULT=/path/to/himt_result.json \
  PYTHONPATH=src pytest tests/release/test_assembly_qc_himt.py -q
ORGANELLEVERSE_QC_GETORGANELLE_RESULT=/path/to/getorganelle_result.json \
  PYTHONPATH=src pytest tests/release/test_assembly_qc_getorganelle.py -q
ORGANELLEVERSE_QC_PMAT_RESULT=/path/to/pmat_result.json \
  PYTHONPATH=src pytest tests/release/test_assembly_qc_pmat.py -q
```

`minimap2` and `samtools` must be on `PATH`.

## Current release status

No four-backend assembly-QC release blocker remains. All four policy-v5 gates
passed with real assembly Results and no scientific failures. Every decision is
honestly `needs_review`; none was promoted to a fabricated `ready`.

## Known evidence limitations

These are reported explicitly and are not converted into invented values:

- The marker scanner uses the release-pinned OatkDB embryophyte profiles for
  both organelles. The first run may download the shared managed database; an
  unavailable network or cache leaves target identity explicitly
  `not_assessed` and prevents a `ready` decision, while digest or schema
  violations fail closed.
- When meryl is available, primary reads passing the mapping-quality filter are
  selected from each library alignment and compared with the assembly as
  canonical distinct 21-mers. `qc.kmer_evidence` reports the fraction of
  assembly k-mers present in those target reads and writes the exact counts to
  `kmer-metrics.tsv`. This is read support evidence, not a Merqury QV or a
  universal pass threshold. Missing meryl, no eligible target reads, or an
  optional meryl command failure remains explicitly `not_assessed`.
- The structural detector covers declared-path junction support,
  contradiction, explicit repeat copies, reverse orientation, circular-origin
  invariance, and read-supported small expansion/collapse events. Without a
  trusted structural reference, a contradicted large adjacency is deliberately
  not guessed to be a deletion, inversion, or translocation; it remains the
  evidence-level `contradicted_junction`.
- Base-error promotion thresholds are versioned conservative candidate filters,
  not empirically calibrated posterior probabilities. A candidate requires
  review or orthogonal validation; absence of candidates is not proof of a
  perfect assembly.
- N50/L50, GC, length, contig count, graph components, and path occurrence are
  descriptive. They never pass an assembly by themselves.

These limitations do not invalidate available read, graph, marker, provenance,
or artifact-integrity evidence. They define what the result can and cannot
claim.

## Recorded tool versions (this environment)

| Tool | Version |
| --- | --- |
| Python | 3.13.13 |
| minimap2 | 2.30-r1287 |
| samtools | 1.20 |
| meryl | 1.4.2 (temporary real-tool smoke environment) |
| pyright | 1.1.411 |
| ruff | 0.15.20 |

## Static gates

Fresh policy-v5 verification in this worktree:

```text
tests/quality_control: 191 passed
full default suite:     2796 passed, 15 skipped, 21 deselected
ruff check:             passed
ruff format --check:    passed
pyright:                0 errors
release_assembly_qc:    4 skipped (real Result environment variables unset)
```

Fresh environment-controlled rerun with the four managed real Result paths:

```text
release_assembly_qc:    4 passed, 2828 deselected, 470.26 s
```
