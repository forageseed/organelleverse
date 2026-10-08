# Captured PAML output

Real PAML 4.10.9 output on the read-only 28-taxon, 25,674-site Poales alignment
used in `scripts/validation/validate_mcmctree_dating.py`. Root-only B(0.908908,
1.234522), GTR+G4, clock=2, burnin=100, sampfreq=2, nsample=100; independent
seeds 43 and 45. These short chains are deliberately **not converged**.

`tree.tre` is PAML's aliased input. `chain1.tsv`/`chain2.tsv` are complete raw
MCMC files, including PAML's additional Gen=1 row. `chain*-summary.txt` retain
PAML's numbered topology, posterior mean tree and summary table (the lengthy
input sequence echo has been omitted). The tests independently compare clades
and means with those numbered PAML outputs and account for the extra Gen=1
sample in PAML's summary. No synthetic traces are presented as real results.

Source run directories:
`~/work/benchmarks/p1/feat-dating-mcmctree/trial/` and `trial-chain2/`.
