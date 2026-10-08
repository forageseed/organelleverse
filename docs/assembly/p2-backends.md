# P2 assembly backend command contracts

This page records verified upstream invocation contracts and the integration
boundary for the planned TIPPo, ptGAUL, and Unicycler adapters.

## TIPPo

The upstream TIPPo v2.4 README documents a required long-read argument
`-f`, target selector `-g` (`chloroplast` or `organelle`), thread count
`-t`, and platform selector `-p` (`hifi`, `clr`, `ont`, `onthq`). It
does not document an output-directory option. Runs therefore need an isolated
working directory, and output collection must follow its input-prefixed output
naming contract. The helper in `assembly.backends.p2_cli` builds only this
documented argv; it does not run or install TIPPo.

## ptGAUL

The official ptGAUL 1.0.5 README documents a plastid-only pipeline that
requires a related plastome reference (`-r`) and raw long reads (`-l`).
Optional documented flags include `-t`, `-g`, `-c`, `-f`, and `-o`.
Its final output differs by detected graph topology: one-edge output is
`final_assembly.fasta`; three-edge output contains two alternative paths. A
production adapter must preserve both paths and must not choose one as the
biologically canonical plastome. The runtime adapter supports raw ONT only because the published script invokes
Flye with --nano-raw; its read-format parser accepts FASTA/FA or FASTQ/FQ
(including .fq.gz and .fastq.gz). A related plastome FASTA is mandatory.
The adapter counts S records in the documented Flye assembly GFA. One edge
uses final_assembly.fasta; three edges require both path1.fasta and
path2.fasta. Other counts fail with the graph path for manual review. In the
three-edge case, path 1 fills the API primary slot and path 2 is retained as
an alternate; neither is labeled the canonical biological haplotype. The
runtime requires the caller's existing ptGAUL environment and dependencies
(minimap2, seqkit, assembly-stats, seqtk, Flye, Python 3, and Biopython); the
package does not install them or bundle a managed environment lock. The
upstream script has no version flag, so runtime resolution cannot independently
verify the installed script version.

## Unicycler

Unicycler's own README states that it is designed for bacterial isolates and
is not intended for eukaryotic genomes. Direct organelle-only assembly is
therefore rejected by the scope validator. Do not expose Unicycler as a
validated plant organelle backend unless a separate benchmark establishes
that use case.

## Release gates

Both adapters are registered through the normal assembly.run service. They
still require benchmark verification before being treated as biologically
validated:

- benchmark TIPPo on the Japanese rice mitochondrial graph;
- benchmark ptGAUL on reference-guided plastid data and independently
  adjudicate its alternate paths against biological evidence.

Sources:

- [ptGAUL upstream script](https://raw.githubusercontent.com/Bean061/ptgaul/main/ptGAUL.sh)
- [ptGAUL path construction helper](https://raw.githubusercontent.com/Bean061/ptgaul/main/combine_gfa.py)
- [TIPPo upstream README](https://github.com/Wenfei-Xian/TIPP)
- [ptGAUL upstream README](https://github.com/Bean061/ptgaul)
- [Unicycler upstream README](https://github.com/rrwick/Unicycler)

