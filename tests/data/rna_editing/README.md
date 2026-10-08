# Arabidopsis chloroplast editing reference

`arabidopsis_cp_known_34.tsv` transcribes the 34 biological sites in Table 1 of
Chateigner-Boutin & Small (2007), Nucleic Acids Research 35:e114,
https://doi.org/10.1093/nar/gkm640 (PMCID PMC2034463).
Source XML: https://www.ebi.ac.uk/europepmc/webservices/rest/PMC2034463/fullTextXML
Retrieved 2026-09-28. This is an historical reference list, not an exhaustive
modern truth set. No RNA observations were used to choose these positions.

`NC_000932.1.fasta` is the NCBI Arabidopsis thaliana plastid reference, copied
from the supplied `/tmp/ath_cp.fa`. The downloaded NCBI NC_000932.1 GenBank
sequence was verified identical. Coordinates are 1-based genomic positions;
alleles use reference orientation. The table retains original coordinates,
codons and amino acid labels in `source_*` columns.

One explicit source discrepancy: the publication lists rpl23 at **86056**, which
is A in NC_000932.1. Its stated target is codon 30, uCa (S>L). The GenBank CDS
`complement(85862..86143)` has TCA at codon 30; its second nucleotide maps to
**86055** (G on the reference). `position` records that annotation-derived
coordinate; `source_position` retains 86056. This normalization was made before
calling on RNA data, is not a threshold adjustment, and is tested explicitly.

The original table lists one representative of inverted-repeat loci; it does
not add the second genomic copy of rpl23 or ndhB. Validation must therefore
report literal coordinate precision against this fixed list, identifying its
limits rather than relabelling unlisted coordinates as proven novel biology.
