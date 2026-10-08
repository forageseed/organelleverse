# GetOrganelle managed installation

`assemble(..., method="getorganelle", environment_source="managed")` installs
GetOrganelle 1.7.7.1 from the committed Linux x86-64 explicit Conda lock,
probes its version, registers the provider, and prepares the requested database
target and its upstream-required companion before assembly. Plant plastid and
mitochondrial modes require both `embplant_pt` and `embplant_mt`; animal and
fungal mitochondrial modes require only their respective target.
`check_backend("getorganelle")` discovers executables
on PATH, in the installation registry, and in Conda environments. It does not
install anything or certify database readiness. Before the registry discovery
fix, a working managed prefix could incorrectly appear uninstalled.

For an isolated installation, set all three roots before calling the API:

```bash
export ORGANELLEVERSE_HOME=/absolute/path/to/validation/home
export ORGANELLEVERSE_CACHE_ROOT=$ORGANELLEVERSE_HOME/cache
export ORGANELLEVERSE_TOOL_ROOT=$ORGANELLEVERSE_HOME/tools
```

`ORGANELLEVERSE_HOME` alone does not redirect the assembly environment cache or
tool registry. The managed installer needs micromamba or Conda on PATH. An
isolated `MAMBA_ROOT_PREFIX` can also keep the package cache within the
validation directory.

The database is downloaded from the official immutable commit archive:

- Commit: `8610b6e67d4d9269c85de91e0f58266c31f72388`.
- URL: <https://github.com/Kinggerm/GetOrganelleDB/archive/8610b6e67d4d9269c85de91e0f58266c31f72388.tar.gz>.
- Size: 43,367,830 bytes.
- SHA-256: `a522e42f7127d38bb2ce3eb9224213f5997fdd12aae1fa782c57b29ed3754315`.

Preparation selects exactly `GetOrganelleDB-<commit>/0.0.1` and invokes
`get_organelle_config.py -a <targets> --use-local <directory> --config-dir <cache>`.
For plant mitochondria, `<targets>` is `embplant_mt,embplant_pt`; for plant
plastids it is `embplant_pt,embplant_mt`. The configured target list participates
in the existing prepared-resource identity, so a previous single-target cache
does not satisfy this preparation.
An existing completed run with the same request but the old database identity
fails the assembly service's exact-reuse check with `assembly.destination_conflict`.
Use a separate validation cache when comparing the two versions; keep the old
run as evidence.

Default assembly passes `--config-dir` without synthetic `-s` or `--genes`
arguments. GetOrganelle's plant defaults use both label databases in target-first
priority order. Passing only the target label through `--genes` activates the
custom-database branch and changes graph slimming, including its default
extension limit. Explicit user seed, label and exclusion artifacts still map
to their documented command-line options.

The 14 FASTA files (77,376,556 bytes) match the old full release database byte
for byte. The `0.0.1.minima` subtree is not used. The declared commit and
verified downloaded bytes replace the mutable release-asset locator without
changing the database sequences. Changing this database identity creates a new
prepared database cache; it does not reinstall an unchanged Conda environment.

Managed source/database downloads make at most three attempts for transient
network errors, HTTP 408/429 and HTTP 5xx, with a 180-second socket timeout and
two seconds between attempts. Other HTTP errors and checksum mismatches fail
immediately. Exhausted attempts raise `OrganelleDependencyError`; downloaded
bytes are published only after checksum verification.

When comparing plastome candidates, account for circular rotation and SSC
inversion. Use `comparative.normalize_plastome_orientation` or independently
align missing reference intervals against both strands of the original
assembly. Its normalization boundaries follow the existing IR detector and
are not annotation-exact. Keep the original assembly, transformation record,
and independent alignments. A NOVOPlasty `Option` remains of unknown topology
even if its sequence covers the entire reference; GetOrganelle circularity is
supported by its output headers and graph reconstruction log.
