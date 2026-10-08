# NOVOPlasty 4.3.5

`assembly.assemble(method="novoplasty")` supports one Illumina paired-end library
for plastids and mitochondria. The existing assembly operation, runtime registry,
`OrganelleResult`, input artifacts and managed output contracts are reused; this
adds a backend, not another capability or execution framework.

```python
from pathlib import Path
from organelleverse.assembly import assemble
from organelleverse.assembly.contracts import (
    AssemblyAuxiliary, NovoplastyParameters, ShortReadLibrary,
)
from organelleverse.io_reads import read_reads

reads = read_reads(
    short_libraries=(ShortReadLibrary(
        technology="illumina", layout="paired_end",
        read1=Path("reads_1.fastq.gz"), read2=Path("reads_2.fastq.gz"),
        read_length=150, insert_size=300,
    ),),
    # Optional: use a related, organelle-specific plain FASTA seed.
    # auxiliary=AssemblyAuxiliary(seed_fasta=Path("seed.fasta")),
)
result = assemble(
    reads, organelle="plastid", method="novoplasty",
    backend_parameters=NovoplastyParameters(kmer_size=33),
    environment_source="managed", timeout_seconds=1800,
)
```

The Linux x86-64 Conda environment pins NOVOPlasty 4.3.5 and its dependencies by
exact package URLs and the package-manager checksums. It uses the existing
managed installer. An existing verified prefix is also supported. NOVOPlasty
has no version flag: the declared probe `-c ''` prints its banner and exits 2.
Only that exit is admitted for this probe; assembly execution still requires
success and nonempty sequence output. A missing tool or a failed execution
never becomes an empty successful result. NOVOPlasty is single threaded;
`threads` does not add parallelism to it.

## Inputs and defaults

Defaults follow the [official 4.3.5 configuration](https://github.com/ndierckx/NOVOPlasty/blob/master/config.txt)
and [upstream instructions](https://github.com/ndierckx/NOVOPlasty):

- K-mer: 33; insertion-size adjustment: yes; quality scores: no;
  extended logging: 0; save assembled reads: no; direct seed extension: no.
- Read length and approximate insert size are required library metadata.
  The upstream example's 151/300 values are examples, not assumed measurements.
- Plastid range: 120,000–200,000 bp. Animal mitochondrial range: 12,000–20,000 bp
  (the explanatory upstream default, rather than its 12,000–22,000 example).
  Other mitochondrial targets require an explicit `genome_range`.
- `memory_gb` maps to `Max memory`; unset leaves the upstream field blank.
- `AssemblyAuxiliary` carries `seed_fasta`, `reference_fasta`,
  `chloroplast_fasta`, and optionally `genome_range`. Conflicting ranges in
  auxiliary inputs and backend parameters are rejected.

Without a plastid seed, the adapter extracts the **rbcL CDS from the bundled
Arabidopsis_thaliana_chloroplast.gb reference**. This is a documented generic
plant plastid seed, not a taxon inference. A user seed takes precedence.
For mitochondrial assembly, supply an organelle-specific seed. Plant
mitochondria use upstream `mito_plant` and also require a previously assembled
chloroplast FASTA and an explicit mitochondrial size range. The chloroplast
seed is never used for mitochondria.

Heteroplasmy/variance calling, batch inputs, Ion Torrent, and direct seed
extension are outside this assembly backend's released scope.

## Results

All upstream `Circularized_assembly_<n>_organelleverse.fasta`,
`Option_<n>_organelleverse.fasta`, and `Contigs_<n>_organelleverse.fasta` files
are preserved as primary/alternate sequence artifacts. Circularized files
establish circular topology; contigs are linear fragments; options have
unknown topology. Multiple records are kept separate and sequences are not
post-processed to force circularization.

The primary slot uses the first circularized candidate, then the first option,
then the first contig file, in numeric file order. This is a serialization
choice, not a biological ranking. `novoplasty_report.json` records every
candidate's topology, length and record count, and the seed's source, accession
or user artifact, IDs and lengths. `novoplasty_seed.fasta` stores the actual
seed used. Absence of documented FASTAs, empty candidates and malformed
sequences fail explicitly. An incomplete linear assembly remains a real result;
its success status alone does not assert genome completeness.

## Isolation

For isolated validation, set all three variables before calling the API:
`ORGANELLEVERSE_HOME`, `ORGANELLEVERSE_CACHE_ROOT`, and
`ORGANELLEVERSE_TOOL_ROOT`. The first variable alone does not redirect assembly
caches. A Linux host that cannot run the existing `bwrap` sandbox may reject
unknown existing prefixes; a package-installed managed environment follows
the established trusted installation and registry path.
