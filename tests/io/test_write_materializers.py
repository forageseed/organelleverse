from __future__ import annotations

import warnings
from pathlib import Path

import pytest

import organelleverse as ov
from organelleverse.core.errors import OrganelleInputError
from organelleverse.core.result import OrganelleResult
from organelleverse.writer import materializing_suites

ALIGNED = (
    ("s1", "ACGTACGTACGTACGTACGT"),
    ("s2", "ACGTACGTACGAACGTACGT"),
    ("s3", "ACGTTCGTACGAACGTACGA"),
)
CDS = (("c1", "ATGGCTAAAGGTCCA"), ("c2", "ATGGCCAAAGGACCA"), ("c3", "ATGGCTAAGGGTCCT"))


def _fasta(path: Path, records: tuple[tuple[str, str], ...]) -> Path:
    path.write_text("".join(f">{n}\n{s}\n" for n, s in records), encoding="utf-8")
    return path


def _written(result: OrganelleResult, suite: str) -> list[Path]:
    assert result.operation_id == f"{suite}.write"
    assert result.status == "ok"
    paths = [Path(a.uri) for a in result.artifacts]
    assert paths and all(p.is_file() for p in paths)
    assert result.metrics["file_count"] == len(paths)
    return paths


def test_the_suites_that_publish_a_materializer_are_listed() -> None:
    assert {
        "diversity",
        "variation",
        "composition",
        "codon_composition",
        "comparative",
        "pangenome",
        "rna_editing",
        "transfer",
    } <= set(materializing_suites())


@pytest.mark.parametrize(
    ("suite", "make"),
    [
        ("diversity", lambda aln, cds: ov.diversity.nucleotide_diversity(aln)),
        ("diversity", lambda aln, cds: ov.diversity.neutral_tests(aln)),
        ("variation", lambda aln, cds: ov.variation.snp(aln)),
        ("variation", lambda aln, cds: ov.variation.snp_density(aln)),
        ("composition", lambda aln, cds: ov.composition.gc_content(aln)),
        ("codon_composition", lambda aln, cds: ov.codon_composition.codon_usage(cds)),
        ("codon_composition", lambda aln, cds: ov.codon_composition.amino_acid(cds)),
    ],
)
def test_ov_write_publishes_what_the_suite_writes(tmp_path: Path, suite: str, make) -> None:
    aln = _fasta(tmp_path / "aln.fasta", ALIGNED)
    cds = _fasta(tmp_path / "cds.fasta", CDS)
    result = make(aln, cds)

    written = ov.write(result, tmp_path / "out")

    _written(written, suite)


@pytest.mark.parametrize(
    ("operation", "metrics"),
    [
        ("comparative.gene_table", {"matrix": {"rbcL": [1, 1]}, "accessions": ["a", "b"]}),
        ("comparative.synteny", {"blocks": [["rbcL", "atpB"]]}),
        ("comparative.compare_genes", {"genes": {}}),
        ("comparative.compare_genomes", {"genomes": [{"genome": "a", "length": 1}]}),
        ("pangenome.gene_pav", {"matrix": {"rbcL": [1, 0]}, "accessions": ["a", "b"]}),
        ("rna_editing.predict_edits", {"sites": []}),
        ("transfer.detect_mtpt", {"fragments": [{"start": 1, "end": 9}]}),
    ],
)
def test_ov_write_publishes_results_whose_metrics_are_the_table(
    tmp_path: Path, operation: str, metrics: dict
) -> None:
    template = ov.diversity.nucleotide_diversity(_fasta(tmp_path / "a.fasta", ALIGNED))
    result = template.model_copy(update={"operation_id": operation, "metrics": metrics})

    written = ov.write(result, tmp_path / "out")

    _written(written, operation.split(".")[0])


def test_a_result_of_a_suite_without_a_materializer_is_still_refused(tmp_path: Path) -> None:
    template = ov.diversity.nucleotide_diversity(_fasta(tmp_path / "a.fasta", ALIGNED))
    result = template.model_copy(update={"operation_id": "trans_splicing.detect_trans_splicing"})

    with pytest.raises(OrganelleInputError) as caught:
        ov.write(result, tmp_path / "out")

    assert caught.value.code == "output.unsupported_operation"


def test_a_failed_result_is_not_written(tmp_path: Path) -> None:
    failed = ov.diversity.nucleotide_diversity(_fasta(tmp_path / "one.fasta", ALIGNED[:1]))
    assert failed.status == "failed"

    with pytest.raises(OrganelleInputError) as caught:
        ov.write(failed, tmp_path / "out")

    assert caught.value.code == "output.failed_result"


def test_save_is_a_deprecated_spelling_of_write(tmp_path: Path) -> None:
    cds = _fasta(tmp_path / "cds.fasta", CDS)
    result = ov.selection.kaks(cds)

    with pytest.warns(DeprecationWarning, match="save is deprecated"):
        saved = ov.selection.save(result, tmp_path / "saved")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        written = ov.selection.write(result, tmp_path / "written")
    assert saved.operation_id == written.operation_id == "selection.write"
    assert ov.selection.save.__name__ == "save"


def test_a_written_result_reads_back(tmp_path: Path) -> None:
    aln = _fasta(tmp_path / "aln.fasta", ALIGNED)
    result = ov.diversity.nucleotide_diversity(aln)

    written = ov.write(result, tmp_path / "out")

    manifests = [Path(a.uri) for a in written.artifacts if a.uri.endswith(".result.json")]
    assert len(manifests) == 1
    loaded = ov.read(manifests[0])
    assert isinstance(loaded, OrganelleResult)
    assert loaded.operation_id == result.operation_id
    assert loaded.object_id == result.object_id
    assert loaded.metrics == result.metrics


def test_a_json_file_that_is_not_a_manifest_is_refused(tmp_path: Path) -> None:
    other = tmp_path / "other.json"
    other.write_text('{"hello": 1}', encoding="utf-8")

    with pytest.raises(OrganelleInputError) as caught:
        ov.read(other)

    assert caught.value.code == "input.not_a_contract_manifest"


def test_suites_without_their_own_writer_publish_plain_tables(tmp_path: Path) -> None:
    assert {"barcode", "hgt", "localization", "population", "morphology", "phenotype"} <= set(
        materializing_suites()
    )
    aln = _fasta(tmp_path / "aln.fasta", ALIGNED)
    proteins = tmp_path / "prot.fasta"
    proteins.write_text(">p1\nMKKLLPTAAAGLLLLAALPAAQP\n>p2\nMASSFRRLLSGLTPAPAAA\n", encoding="utf-8")

    barcode = ov.write(
        ov.barcode.design_barcode(aln, min_window=8, min_p_variability=0.01), tmp_path / "bc"
    )
    names = {p.name for p in _written(barcode, "barcode")}
    assert {"design_barcode.metrics.tsv", "design_barcode.findings.tsv"} <= names

    localization = ov.write(ov.localization.predict(proteins), tmp_path / "loc")
    names = {p.name for p in _written(localization, "localization")}
    assert "predict.per_protein_predictions.tsv" in names
    table = (tmp_path / "loc" / "predict.per_protein_predictions.tsv").read_text(encoding="utf-8")
    header, first, second, *_ = table.split("\n")
    assert "id" in header.split("\t") and first.split("\t")[3] == "p1"


def test_a_written_result_can_be_moved_and_read_again(tmp_path: Path) -> None:
    from organelleverse.core.artifacts import ArtifactRef

    data = tmp_path / "elsewhere" / "calls.vcf"
    data.parent.mkdir()
    data.write_text("##fileformat=VCFv4.2\n", encoding="utf-8")
    template = ov.barcode.design_barcode(_fasta(tmp_path / "aln.fasta", ALIGNED))
    ref = ArtifactRef.from_path(data, kind="variants", format="vcf")
    result = template.model_copy(update={"artifacts": (ref,)})

    written = ov.write(result, tmp_path / "out")

    assert (tmp_path / "out" / "calls.vcf").is_file()
    manifest = next(Path(a.uri) for a in written.artifacts if a.uri.endswith(".result.json"))
    assert '"uri": "calls.vcf"' in manifest.read_text(encoding="utf-8")

    moved = tmp_path / "moved"
    (tmp_path / "out").rename(moved)
    loaded = ov.read(moved / manifest.name)
    assert loaded.object_id == result.object_id
    assert Path(loaded.artifacts[0].uri) == (moved / "calls.vcf").resolve()
    assert Path(loaded.artifacts[0].uri).is_file()
