# Assembly results into a pangenome

`ov.pan.from_assemblies` freezes explicit biological samples and groups from
canonical assembly Results. OVASM already emits an `organelle.gfa` with sample
identity, all representative molecules and alternative configurations. Other
assemblers must first produce an assessed OV-GFA and supply its ArtifactRef as
`unified_graph`.

```python
from organelleverse.pangenome import AssemblySample, from_assemblies, build_exact_pan

cohort = from_assemblies([
    AssemblySample(sample="A", species="Species one", group="group1", result=a),
    AssemblySample(sample="B", species="Species two", group="group1", result=b),
])
pan = build_exact_pan(cohort, group="group1", k=63)
```

The sample identifier must match the OV-GFA header and paths. One assembly per
biological sample and organelle is accepted; multiple libraries must be combined
in the assembly input, not relabelled as independent samples. A cohort cannot
mix mitochondria and plastids. Failed and excluded members remain in its manifest.

The exact-pan entry uses OVASM's existing exact k-mer graph union. It preserves
all source segments, links and paths, including off-path sequence and alternative
configurations. It additionally reconstructs every source segment from the
published segment mapping and writes sparse `node / sample / observed membership`
records. These records do **not** infer absence, orthology, transfer direction or
physical chromosome structure. Cross-species groups should initially be treated
as diagnostic sequence collections; comparative homology requires separate
validation and appropriate alignment methods. A small graph is not evidence of
correct homology.

Outputs use the existing managed run store and ArtifactRef integrity checks:
`pan.gfa`, native validation JSON, segment/link source mappings, source membership,
the complete cohort and a reusable run record. Changing input bytes is rejected;
changing membership creates a different managed result. Reusing a result checks
all published artifact bytes. Construction currently loads graphs in memory;
this interface does not establish thousand-sample capacity.

`assembly.export_representatives` writes **all** configuration-1 molecules to one
FASTA per biological sample. `representatives.json` maps each FASTA record back
to its source graph, path and molecule. The destination must be new. Configuration
numbers describe alternative solutions, not validated biological haplotypes.

If OVASM completes successfully with a valid graph and no resolved sequence, OV
returns a warning Result with `assembly_graph_only` and no primary genome.
Record count and representative sequence bases are zero; graph sequences remain
available for graph union. Process failures and missing/corrupt outputs remain
failures. A graph-only result never supplies evidence of biological absence.

Batch scheduling belongs to PhytoScience. OV does not open SSH connections or
depend on the host. The initial batch host integration is documented in
PhytoScience's `docs/assembly-pangenome-batches.md`.
