"""Partitioned supermatrix, IQ-TREE partition scheme, MrBayes and RF tree comparison."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from organelleverse.core.errors import OrganelleDependencyError, OrganelleInputError
from organelleverse.phylogeny import mrbayes as mb_mod
from organelleverse.phylogeny.mrbayes import (
    effective_sample_size,
    psrf,
    recompute_asdsf,
    resolve_mrbayes,
    run_mrbayes,
    write_mrbayes_nexus,
)
from organelleverse.phylogeny.partition import (
    _charset_sites,
    build_partitioned_supermatrix,
    iqtree_to_mrbayes_model,
    parse_nexus_sets,
    resolve_iqtree,
    select_partition_scheme,
)
from organelleverse.phylogeny.treecompare import (
    compare_trees,
    read_tree,
    robinson_foulds,
    tree_splits,
)

# --------------------------------------------------------------------------- model mapping


@pytest.mark.parametrize(
    ("model", "nst", "rates", "freq", "exact"),
    [
        ("JC", 1, "equal", "fixed(equal)", True),
        ("F81+F", 1, "equal", "dirichlet(1,1,1,1)", True),
        ("F81", 1, "equal", "dirichlet(1,1,1,1)", True),
        ("K2P+G4", 2, "gamma", "fixed(equal)", True),
        ("HKY+I", 2, "propinv", "dirichlet(1,1,1,1)", True),
        ("HKY+F+I+G4", 2, "invgamma", "dirichlet(1,1,1,1)", True),
        ("SYM+I+G4", 6, "invgamma", "fixed(equal)", True),
        ("GTR+F+G4", 6, "gamma", "dirichlet(1,1,1,1)", True),
        ("TIM2+F+I+G4", 6, "invgamma", "dirichlet(1,1,1,1)", False),
        ("TNe+G4", 6, "gamma", "fixed(equal)", False),
        ("K3Pu+F+R3", 6, "gamma", "dirichlet(1,1,1,1)", False),
        ("TVMe+I+R2", 6, "invgamma", "fixed(equal)", False),
        ("HKY+FQ", 2, "equal", "fixed(equal)", True),
    ],
)
def test_iqtree_to_mrbayes_mapping(model, nst, rates, freq, exact) -> None:
    mapped = iqtree_to_mrbayes_model(model)
    assert (mapped["nst"], mapped["rates"], mapped["statefreqpr"], mapped["exact"]) == (
        nst,
        rates,
        freq,
        exact,
    )


def test_gamma_categories_and_unsupported_models() -> None:
    assert iqtree_to_mrbayes_model("GTR+G8")["ngammacat"] == 8
    assert iqtree_to_mrbayes_model("K3Pu+F+R3")["freerate_approximated"] is True
    for bad in ("LG+G4", "GTR+ASC", "GTR+H4", ""):
        with pytest.raises(OrganelleInputError):
            iqtree_to_mrbayes_model(bad)


# --------------------------------------------------------------------------- NEXUS sets


def test_charset_expansion_and_nexus_parsing(tmp_path: Path) -> None:
    assert _charset_sites("1-9\\3") == [1, 4, 7]
    assert _charset_sites("2-6\\3 10 12-13") == [2, 5, 10, 12, 13]
    assert _charset_sites("5-.", nchar=7) == [5, 6, 7]
    nex = tmp_path / "scheme.nex"
    nex.write_text(
        "#nexus\nbegin sets;\n  charset a_pos1+b_pos1 = 1-6\\3  7-12\\3;\n"
        "  charset rest = 2-6\\3 3-6\\3 8-12\\3 9-12\\3;\n"
        "  charpartition mymodels =\n    GTR+F+G4: a_pos1+b_pos1,\n    HKY+I: rest;\nend;\n"
    )
    parsed = parse_nexus_sets(nex)
    assert [name for name, _ in parsed["charsets"]] == ["a_pos1+b_pos1", "rest"]
    assert parsed["models"] == {"a_pos1+b_pos1": "GTR+F+G4", "rest": "HKY+I"}


# --------------------------------------------------------------------------- supermatrix

_CODONS = ["GCT", "GAA", "AAA", "CTG", "TTC", "GGC", "ACC", "TGG", "CAT", "ATG"]


def _gene(rng: random.Random, n_codons: int) -> list[str]:
    return ["ATG"] + [rng.choice(_CODONS) for _ in range(n_codons - 1)]


def _write_genbank(path: Path, name: str, genes: dict[str, str]) -> None:
    from Bio import SeqIO
    from Bio.Seq import Seq
    from Bio.SeqFeature import FeatureLocation, SeqFeature
    from Bio.SeqRecord import SeqRecord

    spacer = "TTTTTTTTTT"
    seq = spacer
    features = []
    for gene, cds in genes.items():
        start = len(seq)
        seq += cds
        features.append(
            SeqFeature(
                FeatureLocation(start, len(seq), strand=1), type="CDS", qualifiers={"gene": [gene]}
            )
        )
        seq += spacer
    record = SeqRecord(Seq(seq), id=name, name=name[:16], description=name, features=features)
    record.annotations["molecule_type"] = "DNA"
    SeqIO.write(record, str(path), "genbank")


def _toy_genomes(tmp_path: Path, n_taxa: int = 5) -> list[Path]:
    rng = random.Random(7)
    base = {"geneA": _gene(rng, 40), "geneB": _gene(rng, 35), "tiny": _gene(rng, 5)}
    paths = []
    for index in range(n_taxa):
        genes = {}
        for gene, codons in base.items():
            mutated = list(codons)
            for pos in rng.sample(range(1, len(mutated)), k=min(3, len(mutated) - 1)):
                mutated[pos] = rng.choice(_CODONS)
            genes[gene] = "".join(mutated) + "TAA"
        if index == 0:
            genes["geneC"] = "".join(_gene(rng, 40)) + "TAA"
        path = tmp_path / f"taxon{index}.gb"
        _write_genbank(path, f"taxon{index}", genes)
        paths.append(path)
    return paths


def test_supermatrix_codon_partitions(tmp_path: Path) -> None:
    paths = _toy_genomes(tmp_path)
    result = build_partitioned_supermatrix(paths, output_dir=tmp_path / "out")
    assert result.status == "ok", result.summary_text
    metrics = result.metrics
    assert list(metrics["genes"]) == ["geneA", "geneB"]
    # "tiny" (< 30 codons) is reported; taxon-specific "geneC" is never a candidate.
    assert set(metrics["skipped_genes"]) == {"tiny"}
    # stop codons removed: 40 + 35 codons.
    assert metrics["alignment_length"] == 3 * 75
    parsed = parse_nexus_sets(metrics["partitions_nexus"])
    names = [name for name, _ in parsed["charsets"]]
    assert names == [f"gene{g}_pos{p}" for g in "AB" for p in (1, 2, 3)]
    covered = sorted(s for _, spec in parsed["charsets"] for s in _charset_sites(spec))
    assert covered == list(range(1, 3 * 75 + 1))
    fasta = Path(metrics["supermatrix_fasta"]).read_text().split()
    assert fasta[0] == ">taxon0" and len(fasta[1]) == 225


def test_supermatrix_length_outlier_and_gene_mode(tmp_path: Path) -> None:
    paths = _toy_genomes(tmp_path)
    _write_genbank(
        paths[1],
        "taxon1",
        {"geneA": "ATG" + "GCT" * 29 + "TAA", "geneB": "ATG" + "GAA" * 34 + "TAA"},
    )
    result = build_partitioned_supermatrix(
        paths, output_dir=tmp_path / "out", codon_positions=False
    )
    assert result.status == "ok"
    assert list(result.metrics["genes"]) == ["geneB"]
    assert result.metrics["skipped_genes"]["geneA"] == "length_outlier:taxon1"
    assert result.metrics["n_partitions"] == 1


def test_supermatrix_rejects_duplicate_taxa(tmp_path: Path) -> None:
    paths = _toy_genomes(tmp_path)
    with pytest.raises(OrganelleInputError):
        build_partitioned_supermatrix(
            paths, output_dir=tmp_path / "o", taxon_names=["a", "a", "b", "c", "d"]
        )


# --------------------------------------------------------------------------- MrBayes NEXUS


def _alignment(tmp_path: Path, nchar: int = 12) -> Path:
    rng = random.Random(3)
    fasta = tmp_path / "aln.fa"
    fasta.write_text(
        "".join(f">t{i}\n{''.join(rng.choice('ACGT') for _ in range(nchar))}\n" for i in range(5))
    )
    return fasta


def test_write_mrbayes_nexus_partitions_and_models(tmp_path: Path) -> None:
    aln = _alignment(tmp_path)
    scheme = tmp_path / "scheme.nex"
    scheme.write_text(
        "#nexus\nbegin sets;\n  charset a_pos1+b_pos1 = 1-12\\3;\n"
        "  charset rest = 2-12\\3 3-12\\3;\n"
        "  charpartition mymodels = K2P+G4: a_pos1+b_pos1, GTR+F+I+G4: rest;\nend;\n"
    )
    setup = write_mrbayes_nexus(aln, scheme, tmp_path / "mb.nex", ngen=5000, seed=7)
    text = (tmp_path / "mb.nex").read_text()
    assert setup["n_partitions"] == 2 and setup["nchar"] == 12
    assert "charset P1 = 1-12\\3;" in text
    assert "partition ovscheme = 2: P1, P2;" in text
    assert "lset applyto=(1) nst=2 rates=gamma ngammacat=4;" in text
    assert "prset applyto=(1) statefreqpr=fixed(equal);" in text
    assert "lset applyto=(2) nst=6 rates=invgamma ngammacat=4;" in text
    assert "unlink statefreq=(all) revmat=(all) tratio=(all) shape=(all) pinvar=(all);" in text
    assert "prset applyto=(all) ratepr=variable;" in text
    assert "set autoclose=yes nowarn=yes seed=7 swapseed=7;" in text
    assert "ngen=5000 nruns=2 nchains=4" in text
    assert "sumt contype=halfcompat conformat=simple;" in text


def test_write_mrbayes_nexus_rejects_incomplete_or_overlapping(tmp_path: Path) -> None:
    aln = _alignment(tmp_path)
    gap = tmp_path / "gap.nex"
    gap.write_text("#nexus\nbegin sets;\n charset a = 1-6;\nend;\n")
    with pytest.raises(OrganelleInputError, match="not in any charset"):
        write_mrbayes_nexus(aln, gap, tmp_path / "x.nex")
    overlap = tmp_path / "overlap.nex"
    overlap.write_text("#nexus\nbegin sets;\n charset a = 1-8;\n charset b = 8-12;\nend;\n")
    with pytest.raises(OrganelleInputError, match="both"):
        write_mrbayes_nexus(aln, overlap, tmp_path / "x.nex")
    single = tmp_path / "single.nex"
    single.write_text("#nexus\nbegin sets;\n charset all = 1-12;\nend;\n")
    write_mrbayes_nexus(aln, single, tmp_path / "s.nex")
    text = (tmp_path / "s.nex").read_text()
    assert "partition" not in text.split("begin mrbayes;")[1]
    assert "lset nst=6 rates=invgamma ngammacat=4;" in text  # default GTR+F+I+G4


# --------------------------------------------------------------------------- RF distance


def test_robinson_foulds_unrooted_and_nni() -> None:
    a = "((A,B),(C,D),(E,F));"
    rooted_same = "(((A,B),(C,D)),(E,F));"
    nni = "((A,C),(B,D),(E,F));"
    assert robinson_foulds(a, rooted_same)["rf"] == 0
    stats = robinson_foulds(a, nni)
    assert stats["rf"] == 4 and stats["max_rf"] == 6
    assert stats["n_shared_splits"] == 1
    with pytest.raises(OrganelleInputError):
        robinson_foulds(a, "((A,B),(C,D),(E,G));")


def test_tree_parsing_labels_lengths_and_nexus(tmp_path: Path) -> None:
    taxa, splits = tree_splits("((A:0.1,B:0.2)0.97:0.01,('C d':1,D)1.000:2e-3,E);")
    assert taxa == frozenset({"A", "B", "C d", "D", "E"})
    # Canonical split = the side without the alphabetically first taxon ("A").
    assert splits == {frozenset({"C d", "D", "E"}): "0.97", frozenset({"C d", "D"}): "1.000"}
    con = tmp_path / "t.con.tre"
    con.write_text(
        "#NEXUS\nbegin trees;\n translate\n  1 A,\n  2 B,\n  3 C,\n  4 D;\n"
        "  [Note]\n  tree con_50_majrule = (1:0.1,2:0.1,(3:0.2,4:0.2)0.850:0.05);\nend;\n"
    )
    assert read_tree(con) == "(A:0.1,B:0.1,(C:0.2,D:0.2)0.850:0.05);"
    newick = tmp_path / "t.nwk"
    newick.write_text("((A,B),C,D);\n")
    result = compare_trees(con, newick)
    assert result.status == "ok" and result.metrics["rf"] == 0
    assert "identical_topology" in result.flags


# --------------------------------------------------------------------------- diagnostics


def _t_file(path: Path, trees: list[str]) -> None:
    path.write_text(
        "#NEXUS\nbegin trees;\n   translate\n       1 A,\n       2 B,\n       3 C,\n"
        "       4 D,\n       5 E;\n"
        + "".join(f"   tree gen.{i * 100} = [&U] {t}\n" for i, t in enumerate(trees))
        + "end;\n"
    )


def test_recompute_asdsf_matches_hand_calculation(tmp_path: Path) -> None:
    t1 = "((1,2),3,(4,5));"
    t2 = "((1,3),2,(4,5));"
    # run1 after 25% burnin of 8 trees (2 dropped): 6x t1 -> {AB: 1.0, DE: 1.0}
    _t_file(tmp_path / "r1.t", [t2, t2] + [t1] * 6)
    # run2: 3x t1 + 3x t2 -> {AB: .5, AC: .5, DE: 1.0}
    _t_file(tmp_path / "r2.t", [t1, t1] + [t1, t2] * 3)
    stats = recompute_asdsf([tmp_path / "r1.t", tmp_path / "r2.t"], burninfrac=0.25)
    assert stats["samples_per_run"] == [6, 6]
    # sd(1.0, .5) = sd(0, .5) = .353553; sd(1,1) = 0 -> mean over 3 splits
    assert stats["asdsf"] == pytest.approx((2 * 0.3535533906) / 3, rel=1e-6)
    assert stats["n_splits"] == 3


def test_psrf_and_ess() -> None:
    rng = random.Random(1)
    chains = [[rng.gauss(0, 1) for _ in range(2000)] for _ in range(2)]
    assert psrf(chains) == pytest.approx(1.0, abs=0.01)
    assert psrf([[x + 5 for x in chains[0]], chains[1]]) > 2
    assert effective_sample_size(chains[0]) > 1000
    ar = [0.0]
    for _ in range(3999):
        ar.append(0.95 * ar[-1] + rng.gauss(0, 1))
    assert effective_sample_size(ar) < 300  # strongly autocorrelated


# --------------------------------------------------------------------------- runners


def test_run_mrbayes_plans_and_reports_missing_binary(tmp_path, monkeypatch) -> None:
    aln = _alignment(tmp_path)
    scheme = tmp_path / "s.nex"
    scheme.write_text("#nexus\nbegin sets;\n charset all = 1-12;\nend;\n")
    monkeypatch.setattr(mb_mod, "resolve_mrbayes", lambda *a, **k: None)
    planned = run_mrbayes(aln, scheme, output_dir=tmp_path / "mb", dry_run=True)
    assert planned.status == "ok" and "backend_missing" in planned.flags
    assert not (tmp_path / "mb").exists()
    with pytest.raises(OrganelleDependencyError, match="micromamba"):
        run_mrbayes(aln, scheme, output_dir=tmp_path / "mb")


def test_select_partition_scheme_plan_and_missing_binary(tmp_path, monkeypatch) -> None:
    from organelleverse.phylogeny import partition as part_mod

    aln = _alignment(tmp_path)
    parts = tmp_path / "p.nex"
    parts.write_text("#nexus\nbegin sets;\n charset all = 1-12;\nend;\n")
    monkeypatch.setattr(part_mod, "resolve_iqtree", lambda *a, **k: None)
    planned = select_partition_scheme(aln, parts, output_dir=tmp_path / "iq", dry_run=True)
    argv = list(planned.metrics["argv"])
    assert argv[argv.index("-m") + 1] == "MFP+MERGE"
    assert argv[argv.index("-mset") + 1] == "mrbayes"
    assert argv[argv.index("-mrate") + 1] == "E,I,G,I+G"
    assert "backend_missing" in planned.flags
    with pytest.raises(OrganelleDependencyError):
        select_partition_scheme(aln, parts, output_dir=tmp_path / "iq")


def _small_real_alignment(tmp_path: Path) -> tuple[Path, Path]:
    rng = random.Random(11)
    root = [rng.choice("ACGT") for _ in range(300)]
    seqs = {}
    for i in range(6):
        seq = list(root)
        for _ in range(10 + 10 * (i % 3)):
            seq[rng.randrange(300)] = rng.choice("ACGT")
        seqs[f"t{i}"] = "".join(seq)
    aln = tmp_path / "aln.fa"
    aln.write_text("".join(f">{k}\n{v}\n" for k, v in seqs.items()))
    parts = tmp_path / "parts.nex"
    parts.write_text(
        "#nexus\nbegin sets;\n charset g1_pos1 = 1-150\\3;\n charset g1_pos2 = 2-150\\3;\n"
        " charset g1_pos3 = 3-150\\3;\n charset g2 = 151-300;\nend;\n"
    )
    return aln, parts


@pytest.mark.integration
@pytest.mark.skipif(resolve_iqtree() is None, reason="IQ-TREE not installed")
def test_real_iqtree_merge_scheme(tmp_path: Path) -> None:
    aln, parts = _small_real_alignment(tmp_path)
    result = select_partition_scheme(aln, parts, output_dir=tmp_path / "iq", threads=1)
    assert result.status == "ok"
    rows = result.metrics["partitions"]
    assert sum(r["n_sites"] for r in rows) == 300
    assert result.metrics["all_mappings_exact"] is True
    assert Path(result.metrics["treefile"]).is_file()


@pytest.mark.integration
@pytest.mark.skipif(resolve_mrbayes() is None, reason="MrBayes not installed")
def test_real_mrbayes_short_run_reports_diagnostics(tmp_path: Path) -> None:
    aln, parts = _small_real_alignment(tmp_path)
    result = run_mrbayes(
        aln,
        parts,
        output_dir=tmp_path / "mb",
        ngen=4000,
        samplefreq=20,
        printfreq=1000,
        diagnfreq=1000,
        seed=3,
    )
    assert result.status in {"ok", "warning"}
    conv = result.metrics["convergence"]
    assert conv["asdsf"] is not None and conv["min_ess"] is not None
    assert ("mcmc_converged" in result.flags) is conv["converged"]
    recomputed = result.metrics["recomputed_diagnostics"]["split_frequencies"]
    # Independent recomputation reproduces MrBayes' own ASDSF.
    assert recomputed["asdsf"] == pytest.approx(conv["asdsf"], abs=1e-5)
    assert Path(result.metrics["consensus_newick"]).is_file()
