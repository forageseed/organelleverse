"""Real-data validation regressions for the ``population`` domain.

Every test here encodes a defect found by validating the capability against an
independent implementation (bcftools/freebayes, scikit-allel, vcftools, BLAST,
minimap2, GEMMA + plink) on real data - see
``docs/reports/population-variation-realdata/README.md``.
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path

import pytest

from organelleverse._losat import resolve_losat

from organelleverse.population import population as pop
from organelleverse.population import population_core
from organelleverse.population import service as pop_service

_HEAD = """\
##fileformat=VCFv4.2
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{cols}
"""


def _vcf(path: Path, rows: list[str], samples: list[str]) -> Path:
    path.write_text(
        _HEAD.format(cols="\t".join(samples)) + "\n".join(rows) + "\n", encoding="utf-8"
    )
    return path


def _row(pos: int, ref: str, alt: str, gts: list[str]) -> str:
    return "\t".join(
        ["chr1", str(pos), ".", ref, alt, ".", "PASS", ".", "GT", *gts]
    )


# ── call_variants ─────────────────────────────────────────────────────────────


def test_call_variants_plans_haploid_calling_by_default() -> None:
    """Organelles are haploid; the planned CLI must say so (defect found vs
    bcftools: with the callers' diploid default, 189/356 mito records and
    17/166 plastid records came back as 0/1 - a genotype no haploid organelle
    call can represent; see docs/reports/population-variation-realdata)."""
    for method, flag in (("bcftools", "--ploidy"), ("gatk", "--sample-ploidy")):
        result = pop_service.call_variants("bams", method=method)
        argv = result.metrics["argv"]
        assert flag in argv, (method, argv)
        assert argv[argv.index(flag) + 1] == "1"
        assert result.metrics["ploidy"] == 1


def test_call_variants_ploidy_parameter_is_honoured() -> None:
    result = pop_service.call_variants("bams", method="bcftools", ploidy=2)
    argv = result.metrics["argv"]
    assert argv[argv.index("--ploidy") + 1] == "2"
    assert result.metrics["ploidy"] == 2


def test_call_variants_reports_failure_instead_of_a_silent_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A raising executor used to be swallowed into ok/call_planned."""

    def boom(argv: list[str]) -> None:
        raise RuntimeError("bcftools exploded")

    monkeypatch.setattr(
        "organelleverse.population.install.check_backend",
        lambda name, scan_envs=False: {"installed": True, "path": "/usr/bin/bcftools",
                                       "env": "test"},
    )
    result = pop_service.call_variants(
        tmp_path, ref_path=tmp_path / "ref.fasta", method="bcftools", executor=boom
    )
    assert result.status == "failed"
    assert result.errors, "a failed result must carry at least one error"
    assert "variants_called" not in result.flags


# ── compute_fst / fst_scan ────────────────────────────────────────────────────


def test_fst_ignores_the_symbolic_upstream_alt_allele(tmp_path: Path) -> None:
    """io_core's length-1 filter admits ALT='*'; an upstream-spanning record is
    not a biallelic SNP and must not enter F_ST (found on real VCFs)."""
    samples = [f"s{i}" for i in range(4)]
    vcf = _vcf(
        tmp_path / "star.vcf",
        [_row(100, "A", "*", ["0", "0", "1", "1"]), _row(200, "A", "T", ["0", "0", "1", "1"])],
        samples,
    )
    core = population_core.compute_fst(str(vcf), {"s0": "pop1", "s1": "pop1",
                                                  "s2": "pop2", "s3": "pop2"})
    assert core["n_snps"] == 1
    assert [w["pos"] for w in core["windows"]] == [200]


def test_fst_scan_top_candidates_reports_zero_when_no_window_qualifies(
    tmp_path: Path,
) -> None:
    """``max(1, 0)`` used to advertise one candidate that does not exist."""
    samples = [f"s{i}" for i in range(4)]
    vcf = _vcf(
        tmp_path / "sparse.vcf",
        [_row(100, "A", "T", ["0", "0", "1", "1"]),
         _row(200, "A", "T", ["0", "0", "1", "1"]),
         _row(20_000, "A", "T", ["0", "0", "1", "1"]),
         _row(20_100, "A", "T", ["0", "0", "1", "1"]),
         _row(20_200, "A", "T", ["0", "0", "1", "1"])],
        samples,
    )
    result = pop.fst_scan(
        str(vcf),
        pop_assignments={"s0": "pop1", "s1": "pop1", "s2": "pop2", "s3": "pop2"},
        window_size=1000,
        step=1000,
    )
    assert result.metrics["windows"] == 0
    assert result.metrics["top_candidates"] == 0
    assert result.flags == ()


def test_fst_scan_flags_population_samples_missing_from_the_vcf(
    tmp_path: Path,
) -> None:
    """A typo in pop_assignments silently shrank a population to one sample."""
    samples = ["s0", "s1", "s2", "s3"]
    vcf = _vcf(
        tmp_path / "typo.vcf",
        [_row(100, "A", "T", ["0", "0", "1", "1"]),
         _row(200, "A", "T", ["0", "1", "1", "0"]),
         _row(300, "A", "T", ["0", "1", "1", "0"])],
        samples,
    )
    pops = {"s0": "pop1", "s1": "pop1", "TYP0_pop2": "pop2", "s3": "pop2"}
    result = pop.fst_scan(str(vcf), pop_assignments=pops, window_size=1000, step=1000)
    assert "population_samples_missing_from_vcf" in result.flags
    # canonical results freeze sequences into tuples
    assert list(result.metrics["samples_missing_from_vcf"]) == ["TYP0_pop2"]

    core = population_core.compute_fst(str(vcf), pops)
    assert core["samples_missing_from_vcf"] == ["TYP0_pop2"]


# ── prepare_gemma_input / cytonuclear_gwas ────────────────────────────────────


def test_gemma_plan_survives_real_organelle_sample_names_and_contigs() -> None:
    """plink exits 3 on '_' in sample IDs and refuses organelle contig names
    unless --double-id/--allow-extra-chr are passed (found on a real plastid
    VCF: the as-planned steps exited 0,3,2)."""
    prep = pop_service.prepare_gemma_input("pop.vcf", phenotype="pheno.txt")
    make_bed = prep.metrics["argv_steps"][1]
    assert "--double-id" in make_bed
    assert "--allow-extra-chr" in make_bed
    recode = prep.metrics["argv_steps"][2]
    assert "--double-id" in recode
    assert "--allow-extra-chr" in recode


def test_gemma_plan_reads_the_file_plink_actually_writes() -> None:
    """plink --recode bimbam writes ``<prefix>.recode.geno.txt``; the plan used
    to point GEMMA at ``<prefix>.geno.txt``, which never exists."""
    prep = pop_service.prepare_gemma_input("pop.vcf", phenotype="pheno.txt")
    assert prep.metrics["bimbam_geno_path"].endswith("data.gemma.geno.txt")


def test_gemma_kinship_path_uses_the_file_gemma_writes() -> None:
    """``gemma -gk 1`` writes ``<o>.cXX.txt``; the planned ``-k`` pointed at
    ``<o>.sXX.txt``, so the LMM step could never run."""
    result = pop_service.cytonuclear_gwas("pop.vcf", phenotype="pheno.txt")
    kinship_argv = result.metrics["argv_steps"][-1]
    k_path = kinship_argv[kinship_argv.index("-k") + 1]
    assert k_path.endswith("cyto_gwas_kinship.cXX.txt")
    assert "-outdir" in kinship_argv


def test_bimbam_to_gemma_produces_gemma_dosage_format(tmp_path: Path) -> None:
    """``plink --recode bimbam`` output is NOT GEMMA-readable (3-line header,
    letter genotypes); GEMMA fails at gemma_io.cpp:699. The package must emit
    real BIMBAM dosages."""
    src = tmp_path / "data_bimbam.recode.geno.txt"
    src.write_text(
        "3\n2\nIND,sA,sB,sC\n"
        "snp1,CC,TT,CT\n"
        "snp2,GG,GT,TT\n",
        encoding="utf-8",
    )
    dst = tmp_path / "data.gemma.geno.txt"
    out = pop._bimbam_to_gemma(src, dst)
    lines = out.read_text().splitlines()
    assert len(lines) == 2
    assert lines[0].split(",")[:3] == ["snp1", "C", "T"]
    assert lines[0].split(",")[3:] == ["0", "2", "1"]
    assert lines[1].split(",")[:3] == ["snp2", "G", "T"]
    assert lines[1].split(",")[3:] == ["0", "1", "2"]
    for line in lines:  # GEMMA dosage rows: no 3-line header, numeric cells
        for cell in line.split(",")[3:]:
            assert cell in {"0", "1", "2", "NA"}


def test_prepare_gemma_input_runs_the_conversion_and_reports_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With an executor, the BIMBAM file GEMMA needs must exist and be named."""

    def fake_check(name: str, scan_envs: bool = False) -> dict:
        return {"installed": True, "path": f"/usr/bin/{name}", "env": "test"}

    monkeypatch.setattr("organelleverse.population.install.check_backend", fake_check)

    # a fake plink that writes exactly what the real one writes
    def executor(argv: list[str]) -> None:
        if "--recode" in argv:
            Path(str(argv[-1]) + ".recode.geno.txt").write_text(
                "2\n1\nIND,sA,sB\nsnp1,CC,TT\n", encoding="utf-8"
            )

    out = tmp_path / "prep"
    result = pop.prepare_gemma_input("pop.vcf", phenotype="pheno.txt",
                                     out_dir=out, executor=executor)
    assert result.status == "ok"
    assert result.metrics["bimbam_geno_path"] == str(out / "data.gemma.geno.txt")
    assert Path(result.metrics["bimbam_geno_path"]).is_file()


def test_prepare_gemma_input_reports_failure_when_a_step_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Steps 0-2 used to run, then a failing step silently became 'planned'."""

    def fake_check(name: str, scan_envs: bool = False) -> dict:
        return {"installed": True, "path": f"/usr/bin/{name}", "env": "test"}

    monkeypatch.setattr("organelleverse.population.install.check_backend", fake_check)

    def executor(argv: list[str]) -> None:
        if "--make-bed" in argv:
            raise RuntimeError("plink: Multiple instances of '_' in sample ID")

    result = pop.prepare_gemma_input("pop.vcf", phenotype="pheno.txt",
                                     out_dir=tmp_path / "prep", executor=executor)
    assert result.status == "failed"
    assert result.errors
    assert "preparation_planned" not in result.flags


def test_cytonuclear_gwas_writes_the_dosage_file_before_gemma_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The GEMMA steps read the dosage file, so the rewrite must run between
    the plink steps and the GEMMA steps — doing it last left GEMMA a missing
    input (found by running the plan verbatim on a real plastid VCF)."""

    def fake_check(name: str, scan_envs: bool = False) -> dict:
        return {"installed": True, "path": f"/usr/bin/{name}", "env": "test"}

    monkeypatch.setattr("organelleverse.population.install.check_backend", fake_check)

    out = tmp_path / "gwas"
    geno_at_gemma_time = []

    def executor(argv: list[str]) -> None:
        if "--recode" in argv:
            Path(str(argv[-1]) + ".recode.geno.txt").write_text(
                "1\n2\nIND,sA,sB\nsnp1,CC,TT\n", encoding="utf-8"
            )
        if argv[0] == "/usr/bin/gemma":
            geno_at_gemma_time.append(Path(out / "data.gemma.geno.txt").is_file())

    result = pop.cytonuclear_gwas("pop.vcf", phenotype="pheno.txt",
                                  out_dir=out, executor=executor)
    assert result.status == "ok"
    assert "gwas_run" in result.flags
    assert geno_at_gemma_time == [True, True]


# ── detect_numt ───────────────────────────────────────────────────────────────


@pytest.fixture
def no_aligner(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neither LOSAT nor blastn available: detect_numt falls back to exact k-mers."""
    from organelleverse.transfer import transfer_core

    monkeypatch.setenv("ORG_VERSE_LOSAT_BIN", "none")
    monkeypatch.setattr(transfer_core.shutil, "which", lambda name: None)


needs_aligner = pytest.mark.skipif(
    resolve_losat() is None and shutil.which("blastn") is None,
    reason="needs LOSAT or NCBI blastn",
)


def _dna(length: int, seed: int) -> str:
    rng = random.Random(seed)
    return "".join(rng.choice("ACGT") for _ in range(length))


def _diverge(seq: str, rate: float, seed: int = 7) -> str:
    rng = random.Random(seed)
    swap = {"A": "C", "C": "G", "G": "T", "T": "A"}
    return "".join(swap[b] if rng.random() < rate else b for b in seq)


@pytest.mark.parametrize("aligner", ["alignment", "kmer_fallback"])
def test_detect_numt_reports_per_record_coordinates(
    tmp_path: Path, aligner: str, request: pytest.FixtureRequest
) -> None:
    """Every fragment names its record and uses coordinates inside it; the old
    scan returned offsets into the concatenation with no sequence name (found
    against BLAST), unusable for masking a real genome."""
    if aligner == "kmer_fallback":
        request.getfixturevalue("no_aligner")
    elif resolve_losat() is None and shutil.which("blastn") is None:
        pytest.skip("needs LOSAT or NCBI blastn")
    mito_seq = _dna(300, seed=1)
    chr1 = _dna(200, seed=2) + mito_seq[:150] + _dna(200, seed=3)
    chr2 = _dna(400, seed=4) + mito_seq[150:] + _dna(100, seed=5)
    nuclear = tmp_path / "nuclear.fasta"
    nuclear.write_text(f">chr1\n{chr1}\n>chr2 description here\n{chr2}\n", encoding="utf-8")
    mito = tmp_path / "mito.fasta"
    mito.write_text(f">mt\n{mito_seq}\n", encoding="utf-8")

    result = pop.detect_numt(str(nuclear), str(mito), k=15, min_len=60)

    assert result.status == "ok"
    fragments = [dict(f) for f in result.metrics["fragments"]]
    lengths = {"chr1": len(chr1), "chr2": len(chr2)}
    assert {f["seqid"] for f in fragments} == {"chr1", "chr2"}, fragments
    for f in fragments:
        assert 1 <= f["start"] <= f["end"] <= lengths[f["seqid"]], f
    chr1_fragment = next(f for f in fragments if f["seqid"] == "chr1")
    assert abs(chr1_fragment["start"] - 201) <= 3 and abs(chr1_fragment["end"] - 350) <= 3
    assert result.metrics["reference_records_scanned"] == 2
    assert ("kmer_fallback" in result.flags) == (aligner == "kmer_fallback")


@needs_aligner
def test_detect_numt_finds_diverged_copies_the_exact_kmer_scan_misses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On the rice mitochondrion the k-mer scan found 45% of the bases BLAST finds
    (k made no difference): substitutions break exact runs. A ~92% identical copy
    is found by alignment and missed by the fallback."""
    mito_seq = _dna(1200, seed=11)
    nuclear_seq = _dna(2000, seed=12) + _diverge(mito_seq[100:900], 0.08) + _dna(1500, seed=13)
    nuclear = tmp_path / "nuclear.fasta"
    nuclear.write_text(f">chr1\n{nuclear_seq}\n", encoding="utf-8")
    mito = tmp_path / "mito.fasta"
    mito.write_text(f">mt\n{mito_seq}\n", encoding="utf-8")

    found = pop.detect_numt(str(nuclear), str(mito))
    assert found.metrics["fragment_count"] == 1
    fragment = dict(found.metrics["fragments"][0])
    assert fragment["seqid"] == "chr1"
    assert abs(fragment["start"] - 2001) <= 30 and abs(fragment["end"] - 2800) <= 30
    assert 88 <= fragment["identity"] <= 96
    assert "kmer_fallback" not in found.flags

    from organelleverse.transfer import transfer_core

    monkeypatch.setenv("ORG_VERSE_LOSAT_BIN", "none")
    monkeypatch.setattr(transfer_core.shutil, "which", lambda name: None)
    fallback = pop.detect_numt(str(nuclear), str(mito))
    assert "kmer_fallback" in fallback.flags
    assert fallback.metrics["fragment_count"] == 0


def test_detect_numt_flags_fragments_spanning_a_record_boundary(
    tmp_path: Path, no_aligner: None
) -> None:
    """A k-mer fragment crossing the concatenation seam is an artefact; it must be
    clipped at the record boundary (and flagged) rather than reported as one
    impossible interval. (Alignment never concatenates records.)"""
    nuclear = tmp_path / "nuclear.fasta"
    nuclear.write_text(">chr1\n" + "ACGT" * 50 + "\n>chr2\n" + "TTTT" * 50 + "\n",
                       encoding="utf-8")
    mito = tmp_path / "mito.fasta"
    # the query ends ACGT->TTTT, so its 8-mers straddling that junction
    # ('ACGTTTTT'...) only exist in the reference at the chr1|chr2 seam
    mito.write_text(">mt\n" + "ACGT" * 24 + "TTTT" * 8 + "\n", encoding="utf-8")
    result = pop.detect_numt(str(nuclear), str(mito), k=8, min_len=16)
    assert "fragment_clipped_at_reference_record_boundary" in result.flags
    frags = [dict(f) for f in result.metrics["fragments"]]
    assert {f["seqid"] for f in frags} >= {"chr1", "chr2"}, frags
    for f in frags:  # every fragment lives inside exactly one record
        assert 1 <= f["start"] <= f["end"] <= 200, f  # chr1 and chr2 are 200 bp


# ── service facade signature parity (contract) ────────────────────────────────


def test_service_facade_forwards_the_core_call_variants_parameters() -> None:
    """The facade is the public entry point: it must expose (and forward) the
    same parameters the core accepts, ``ploidy`` included."""
    import inspect

    import organelleverse.population as population_pkg

    core_sig = inspect.signature(pop.call_variants)
    assert "ploidy" in core_sig.parameters
    assert core_sig.parameters["ploidy"].default == 1

    facade_sig = inspect.signature(pop_service.call_variants)
    assert "ploidy" in facade_sig.parameters
    for name, param in facade_sig.parameters.items():
        assert name in core_sig.parameters, name
        assert param.default == core_sig.parameters[name].default, name

    # the package export (what capabilities and users import) is the facade
    assert population_pkg.call_variants is pop_service.call_variants
    assert "ploidy" in inspect.signature(population_pkg.call_variants).parameters


def test_call_variants_plan_smoke() -> None:
    result = pop_service.call_variants("bams", method="bcftools")
    assert result.status == "ok"
    assert "call_planned" in result.flags


# ── review follow-ups ─────────────────────────────────────────────────────────


def test_deepvariant_plan_carries_no_unverified_ploidy_flag() -> None:
    """DeepVariant is diploid-only (no ploidy switch); a made-up ``--sample_ploidy``
    would make the default method's plan fail on real installs."""
    result = pop_service.call_variants("bams", method="deepvariant")
    argv = result.metrics["argv"]
    assert not any(token.startswith("--sample_ploidy") for token in argv)
    assert result.metrics["ploidy_applied"] is False
    assert "deepvariant_diploid_only" in result.flags

    diploid = pop_service.call_variants("bams", method="deepvariant", ploidy=2)
    assert "deepvariant_diploid_only" not in diploid.flags
    assert pop_service.call_variants("bams", method="bcftools").metrics["ploidy_applied"] is True


def test_fst_scan_keeps_the_top_window_when_few_windows_exist(tmp_path: Path) -> None:
    """int(n_windows * 0.05) is 0 below 20 windows; the best window must still be reported."""
    samples = [f"s{i}" for i in range(4)]
    rows = [_row(100 * i, "A", "T", ["0", "0", "1", "1"]) for i in range(1, 10)]
    vcf = _vcf(tmp_path / "few.vcf", rows, samples)
    result = pop.fst_scan(
        str(vcf),
        pop_assignments={"s0": "pop1", "s1": "pop1", "s2": "pop2", "s3": "pop2"},
        window_size=500,
        step=300,
    )
    assert 0 < result.metrics["windows"] < 20
    assert result.metrics["top_candidates"] == 1
    assert "selection_candidates" in result.flags


def test_bimbam_to_gemma_leaves_no_partial_file_when_a_row_is_unusable(tmp_path: Path) -> None:
    src = tmp_path / "bad.recode.geno.txt"
    src.write_text("2\n2\nIND,sA,sB\nsnp1,CC,TT\nsnp2,NN,NN\n", encoding="utf-8")
    dst = tmp_path / "out.geno.txt"
    with pytest.raises(ValueError, match="snp2"):
        pop._bimbam_to_gemma(src, dst)
    assert not dst.exists()
