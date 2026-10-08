# PMAT2 managed source build

The tested build uses upstream commit `04534a2adf0c5309cb2e7478fd2bcc98ced3818c`
(PMAT2 2.1.5), with `orientation.patch` applied before compilation. Both the
archive and patch SHA-256 are pinned in `assembly/environment_specs.py`; patch
bytes are verified before `patch --batch --forward --fuzz=0 -p1` is run.

The patch changes the two GFA writers to map a Newbler source 3′ end to `+`
and a target 5′ end to `+`. `path2fa.c` keeps `flag=5` sequences forward and
reverse-complements `flag=3`. It also marks the executable version banner.
Seed selection, main-graph filtering and path search are unchanged. Results
report **PMAT2 2.1.5 + OrganelleVerse orientation patch** in software_versions.
No output graph is repaired after assembly.

The patch is attached only to this immutable source commit. Other explicitly
requested upstream versions do not receive it. The tested provider contract
requires the patch banner, so an older unpatched installation cannot silently
satisfy the tested build. The host needs `patch` as well as Apptainer/Singularity.

PMAT2 2.2.0 was inspected at `cb03121bcf5a45ca45607571e7d7e6968a2cbbca`;
its commit archive SHA-256 is
`350876346e14d33d390ebc2103ddd8b7f14d7202620ce658e6b65db9393cc8e1`.
It retains both direction errors. See `scripts/assembly_benchmarks/PMAT_ORIENTATION.md`
in the source repository for the comparison, real-data validation and limits.
