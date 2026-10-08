# Example data

One real genome, used by the examples in the top-level README.

| File | What it is |
|---|---|
| `arabidopsis_mitochondrion.fasta` | *Arabidopsis thaliana* mitochondrion, sequence only |
| `arabidopsis_mitochondrion.gb` | the same genome with its annotation (GenBank) |

Source: NCBI RefSeq [NC_037304.1](https://www.ncbi.nlm.nih.gov/nuccore/NC_037304.1), 367,808 bp,
circular. NCBI sequence records are in the public domain.

```python
import organelleverse as ov

genome = ov.read("examples/data/arabidopsis_mitochondrion.fasta",
                 organelle="mitochondrion", species="Arabidopsis thaliana")
annotation = ov.annotation.annotate(genome)
ov.write(annotation, "results/annotation")
```
