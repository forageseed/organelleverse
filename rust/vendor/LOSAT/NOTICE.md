# Vendored LOSAT

Source: <https://github.com/satoshikawato/LOSAT> (MIT, see `LICENSE`), the
0.1.0-era snapshot that OrganelleVerse was validated against. Its command line
(`--task`, `-q`, `-s`, `-n`, `-o`, `--word-size`, `tblastn --db-gencode` ...)
differs from upstream 0.2.0, which switched to NCBI-style single-dash options
and a pairwise-text default output, so OrganelleVerse keeps this snapshot
instead of tracking upstream.

Local change: `src/algorithm/blastn/alignment/gapped.rs`. The walk to the left
in `blast_get_start_for_gapped_alignment_nucl` also stops at the first subject
base (NCBI's subject buffer has a sentinel before position 0; here the index
underflowed and panicked on plastomes whose first gene starts at position 1),
plus a regression test. Not submitted upstream.

Binaries built from this directory are committed under
`src/organelleverse/_bin/<platform>/` by `.github/workflows/bundled-tools.yml`.
