# RNA-seq organelle transcript summaries

These operations consume a coordinate-sorted, indexed RNA-seq BAM/CRAM with a
reference dictionary matching the annotation. Install `pysam` with
`pip install 'organelleverse[qc]'`.

## Splicing efficiency

`rna_seq.quantify_splicing_efficiency` takes a TSV with
`intron_id, gene_id, seqid, start, end, strand`. `start` and `end` are 1-based,
inclusive genomic coordinates on the forward reference for both `+` and `-`
strands; a negative-strand intron does not reverse its numeric coordinates.
Only alignments with an exact CIGAR `N` spanning the annotated intron and at
least 8 aligned reference bases on both sides are counted as spliced. An
unspliced alignment must continuously cover the entire intron plus the same
anchor length. Excision efficiency is spliced / (spliced + unspliced), so reads
with other junctions or insufficient anchors are omitted. Secondary,
supplementary, unmapped and QC-failed reads, low-MAPQ reads, and duplicates by
default are excluded. The result describes the observed alignment fraction at
provided introns; it is not transcript abundance, does not discover introns,
and may be biased by mapping ambiguity, RNA-seq library preparation, or
transcript-specific read coverage.

## Exon-overlap expression summary

`rna_seq.quantify_exon_expression` accepts a TSV with
`feature_id, gene_id, seqid, start, end, strand` and counts primary alignments
overlapping the union of a gene's annotated exons. Coordinates follow the same
1-based inclusive reference convention. An alignment is counted once per gene
even if it overlaps multiple exons; paired mates are counted as separate reads.
CPM is the gene read count divided by all primary mapped reads passing the same
MAPQ/duplicate filters in the BAM, multiplied by one million. This is a
library-size normalized read count, not TPM; it does not disambiguate reads
that overlap multiple genes, quantify isoforms, correct transcript length, or
correct organelle copy-number differences. Compare samples only when their
libraries and filters are appropriate for that comparison.
