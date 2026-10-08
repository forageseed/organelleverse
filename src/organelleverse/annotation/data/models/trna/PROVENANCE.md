# tRNA covariance model — provenance & license

- `trna.cm` — Rfam family **RF00005 (tRNA)** covariance model.
- Source: https://rfam.org/family/RF00005/cm (Infernal 1.1 format).
- License: **CC0 1.0 (public domain)** — Rfam data is released to the public
  domain and is freely redistributable, compatible with this package's MIT
  license.
- Rebuild: `python scripts/annotation_benchmarks/build_trna_cm.py --out <path>`.

- `plant_mito_trna.cm` / `plant_mito_trna.hmm` — **plant-mitochondrial-specific**
  tRNA CM and HMM filter (the genome scanner's default). Built by
  `scripts/annotation_benchmarks/build_plant_mito_trna_model.py`:
  mature tRNA sequences extracted from **public NCBI plant-mitochondrial GenBank
  genomes** (benchmark species — Ranunculus, Nicotiana tabacum, Lactuca sativa,
  Pinus taeda — excluded to avoid train/test leakage), `cmalign`-ed to the
  RF00005 CM for structure, then `cmbuild`/`hmmbuild`. Derives from RF00005 (CC0)
  structure plus public sequence data; freely redistributable. It detects
  divergent plant-mito tRNAs (e.g. trnL-UAA, trnS-CGA) that RF00005 misses,
  without regressing mature-tRNA coordinates.

These models are used by the native `organelleverse.annotation.cmsearch` engine.
GPL-licensed models (e.g. tRNAscan-SE's bundled CMs) are **never** shipped; they
are used only as local benchmark references.

- `plastid_trna.cm` / `plastid_trna.hmm` — **plastid (chloroplast)-specific**
  tRNA CM and HMM filter. Built by the same recipe as the plant-mito model
  (`build_plant_mito_trna_model.py` with `--genomes-dir` pointing at chloroplast
  GenBanks) from tRNAs in the packaged chloroplast reference set (RF00005 (CC0)
  structure + public sequence data; freely redistributable). Used by the genome
  scanner for plastid tRNA detection with both-exon intron recovery. On a
  held-out chloroplast it reaches ~0.92 recall vs GenBank tRNAs (0.76 without
  intron recovery).
