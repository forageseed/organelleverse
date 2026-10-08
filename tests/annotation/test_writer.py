from pathlib import Path

import pytest

from organelleverse.annotation import writer as writer_module
from organelleverse.annotation.genbank import parse_genbank
from organelleverse.annotation.models import (
    AnnotationDocument,
    AnnotationFeature,
    AnnotationRecord,
    FeatureQualifier,
    LocationPart,
)
from organelleverse.annotation.writer import (
    load_document_json,
    materialize_annotation,
    materialize_result,
    write_document_json,
)
from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.frozen import FrozenMap
from organelleverse.core.result import OrganelleResult


def roundtrip_document() -> AnnotationDocument:
    cds = AnnotationFeature(
        feature_id="r1:0:CDS",
        seqid="r1",
        type="CDS",
        operator="join",
        parts=(
            LocationPart(start=0, end=6, strand=1),
            LocationPart(start=12, end=18, strand=1),
        ),
        qualifiers=(
            FeatureQualifier(name="gene", values=("nad1",)),
            FeatureQualifier(name="product", values=("NADH dehydrogenase subunit 1",)),
            FeatureQualifier(name="translation", values=("MPFK",)),
        ),
        parents=(),
    )
    trna = AnnotationFeature(
        feature_id="r1:1:tRNA",
        seqid="r1",
        type="tRNA",
        operator="single",
        parts=(LocationPart(start=18, end=24, strand=-1),),
        qualifiers=(
            FeatureQualifier(name="gene", values=("trnM-CAU",)),
            FeatureQualifier(name="product", values=("tRNA-Met",)),
        ),
        parents=(),
    )
    rrna = AnnotationFeature(
        feature_id="r1:2:rRNA",
        seqid="r1",
        type="rRNA",
        operator="single",
        parts=(LocationPart(start=24, end=30, strand=1),),
        qualifiers=(FeatureQualifier(name="gene", values=("rrn5",)),),
        parents=(),
    )
    return AnnotationDocument(
        backend="fixture",
        requested_stages=("pcg", "trna", "rrna"),
        completed_stages=("pcg", "trna", "rrna"),
        records=(
            AnnotationRecord(
                seqid="r1",
                name="r1",
                description="writer fixture",
                sequence="ATGCCCGGGGGGTTTAAATTTATGCCCGGG",
                features=(cds, trna, rrna),
            ),
        ),
        source_metadata=FrozenMap({"topology": "linear"}),
    )


def _normalized_features(document: AnnotationDocument) -> set[tuple[object, ...]]:
    return {
        (
            feature.type,
            tuple((part.start, part.end, part.strand) for part in feature.parts),
            feature.qualifier_values("gene"),
            feature.qualifier_values("product"),
        )
        for record in document.records
        for feature in record.features
    }


def multi_contig_document() -> AnnotationDocument:
    """A second, identical record under its own seqid joins the fixture record."""
    base = roundtrip_document()
    first = base.records[0]
    second = first.evolve(
        seqid="r2",
        name="r2",
        features=tuple(
            feature.evolve(feature_id=f"r2:{index}:{feature.type}", seqid="r2")
            for index, feature in enumerate(first.features)
        ),
    )
    return base.evolve(records=(first, second))


def _tree_bytes(path: Path) -> dict[str, bytes]:
    return {
        item.relative_to(path).as_posix(): item.read_bytes()
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }


def test_document_json_is_deterministic_and_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "annotation.json"
    document = roundtrip_document()

    write_document_json(document, path)

    assert path.read_bytes().endswith(b"\n")
    assert load_document_json(path) == document


def test_materialize_annotation_writes_and_reparses_every_release_format(
    tmp_path: Path,
) -> None:
    document = roundtrip_document()

    paths = materialize_annotation(document, tmp_path / "published")

    assert {
        "json",
        "genbank",
        "gff3",
        "tbl",
        "unedited_tbl",
        "fsa",
        "cds_fasta",
        "protein_fasta",
        "trna_fasta",
        "rrna_fasta",
        "manifest",
    } <= set(paths)
    # The table2asn report is optional: it exists only where the binary is
    # provisioned, so materialization stays deterministic without it.
    assert set(paths) <= {
        "json",
        "genbank",
        "gff3",
        "tbl",
        "unedited_tbl",
        "fsa",
        "cds_fasta",
        "protein_fasta",
        "trna_fasta",
        "rrna_fasta",
        "manifest",
        "table2asn_validation",
    }
    assert all(path.is_file() for path in paths.values())
    assert load_document_json(paths["json"]) == document
    reparsed = parse_genbank(paths["genbank"])
    assert _normalized_features(reparsed) == _normalized_features(document)
    assert sum("\tCDS\t" in line for line in paths["gff3"].read_text().splitlines()) == 2
    assert paths["cds_fasta"].read_text().startswith(">nad1")
    assert paths["protein_fasta"].read_text().startswith(">nad1\nMPFK")
    assert paths["trna_fasta"].read_text().startswith(">trnM-CAU")
    assert paths["rrna_fasta"].read_text().startswith(">rrn5")


def test_multi_contig_materialization_adds_a_concatenated_view(tmp_path: Path) -> None:
    import json

    from Bio import SeqIO

    document = multi_contig_document()

    paths = materialize_annotation(document, tmp_path / "published")

    # The release view stays per-contig; only the extra directory concatenates.
    concat = tmp_path / "published" / "concatenated"
    assert {path.name for path in concat.iterdir()} == {
        "annotation.gb",
        "annotation.gff3",
        "annotation.tbl",
        "annotation.unedited.tbl",
        "genome.fsa",
        "merged_stat.txt",
    }
    assert {record.id for record in SeqIO.parse(paths["genbank"], "genbank")} == {"r1", "r2"}
    (merged,) = SeqIO.parse(concat / "annotation.gb", "genbank")
    assert merged.id == "merged_2_contigs"
    assert len(merged.seq) == 2 * 30 + 200
    assert str(merged.seq)[30:230] == "N" * 200
    stat = (concat / "merged_stat.txt").read_text().splitlines()
    assert stat == [
        "contig_id\tlength\tmerged_start\tmerged_end",
        "r1\t30\t1\t30",
        f"r2\t30\t{30 + 200 + 1}\t{2 * 30 + 200}",
    ]
    manifest = json.loads(paths["manifest"].read_text())
    assert "concatenated/annotation.gb" in {file["name"] for file in manifest["files"].values()}
    assert {key for key in paths if key.startswith("concat_")} == {
        "concat_genbank",
        "concat_gff3",
        "concat_tbl",
        "concat_unedited_tbl",
        "concat_fasta",
        "concat_stat",
    }

    # A single-record document keeps its flat release layout.
    single = materialize_annotation(roundtrip_document(), tmp_path / "single")
    assert not (tmp_path / "single" / "concatenated").exists()
    assert not any(key.startswith("concat_") for key in single)


def test_invalid_materialization_preserves_existing_destination(tmp_path: Path) -> None:
    output = tmp_path / "published"
    output.mkdir()
    marker = output / "existing.txt"
    marker.write_bytes(b"keep-me")
    document = roundtrip_document()
    bad_part = LocationPart(start=0, end=31, strand=1)
    bad_feature = (
        document.records[0]
        .features[0]
        .evolve(
            operator="single",
            parts=(bad_part,),
        )
    )
    bad_record = document.records[0].evolve(
        features=(bad_feature, *document.records[0].features[1:])
    )
    invalid = document.evolve(records=(bad_record,))

    with pytest.raises(ValueError, match="annotation_validation_failed"):
        materialize_annotation(invalid, output)

    assert marker.read_bytes() == b"keep-me"
    assert sorted(path.name for path in output.iterdir()) == ["existing.txt"]
    assert not list(tmp_path.glob(".published.tmp-*"))


def test_failed_install_after_backup_restores_annotation_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "published"
    document = roundtrip_document()
    materialize_annotation(document, output)
    before = _tree_bytes(output)
    real_replace = writer_module.os.replace
    destination_moved = False

    def fail_install(source: str | Path, destination: str | Path) -> None:
        nonlocal destination_moved
        source_path = Path(source)
        destination_path = Path(destination)
        if source_path == output:
            destination_moved = True
        elif destination_moved and ".tmp-" in source_path.name and destination_path == output:
            raise RuntimeError("injected post-backup install failure")
        real_replace(source, destination)

    monkeypatch.setattr(writer_module.os, "replace", fail_install)

    with pytest.raises(RuntimeError, match="post-backup install failure"):
        materialize_annotation(document, output)

    assert destination_moved is True
    assert _tree_bytes(output) == before
    assert not list(tmp_path.glob(".published.tmp-*"))
    assert not list(tmp_path.glob(".published.backup-*"))


def test_failed_install_after_backup_restores_extraction_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "cds.fasta"
    source.write_text(">nad1\nATG\n")
    extracted = OrganelleResult(
        operation_id="annotation.extract",
        scope="mitochondrion",
        status="ok",
        artifacts=(ArtifactRef.from_path(source, kind="annotation_cds", format="fasta"),),
    )
    output = tmp_path / "published-extraction"
    materialize_result(extracted, output)
    before = _tree_bytes(output)
    real_replace = writer_module.os.replace
    destination_moved = False

    def fail_install(source_path: str | Path, destination_path: str | Path) -> None:
        nonlocal destination_moved
        source_value = Path(source_path)
        destination_value = Path(destination_path)
        if source_value == output:
            destination_moved = True
        elif destination_moved and ".tmp-" in source_value.name and destination_value == output:
            raise RuntimeError("injected extraction install failure")
        real_replace(source_path, destination_path)

    monkeypatch.setattr(writer_module.os, "replace", fail_install)

    with pytest.raises(RuntimeError, match="extraction install failure"):
        materialize_result(extracted, output)

    assert destination_moved is True
    assert _tree_bytes(output) == before
    assert not list(tmp_path.glob(".published-extraction.tmp-*"))
    assert not list(tmp_path.glob(".published-extraction.backup-*"))


def test_materialize_result_resolves_canonical_artifact_without_recomputation(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.json"
    write_document_json(roundtrip_document(), source)
    artifact = ArtifactRef.from_path(
        source,
        kind="annotation",
        format="json",
        media_type="application/json",
    )
    result = OrganelleResult(
        operation_id="annotation.annotate",
        scope="mitochondrion",
        status="ok",
        metrics=FrozenMap({"feature_count": 3}),
        artifacts=(artifact,),
    )

    written = materialize_result(result, tmp_path / "result-output")

    assert written.operation_id == "annotation.write"
    assert written.status == "ok"
    # 11 release formats; 12 where a provisioned table2asn adds its report.
    from organelleverse.annotation.table2asn import find_table2asn

    assert len(written.artifacts) == 11 + (1 if find_table2asn() else 0)
    assert all(item.resolve().is_file() for item in written.artifacts)


def test_materialize_result_preserves_source_warning_status(tmp_path: Path) -> None:
    source = tmp_path / "source-warning.json"
    write_document_json(roundtrip_document(), source)
    result = OrganelleResult(
        operation_id="annotation.annotate",
        scope="mitochondrion",
        status="warning",
        metrics=FrozenMap({"rejected_cds_count": 1}),
        flags=("cds_candidates_rejected",),
        artifacts=(ArtifactRef.from_path(source, kind="annotation", format="json"),),
    )

    written = materialize_result(result, tmp_path / "warning-output")

    assert written.status == "warning"
    assert written.metrics["source_result_status"] == "warning"
    assert written.flags == ("annotation_materialized", "cds_candidates_rejected")
    # 11 release formats; 12 where a provisioned table2asn adds its report.
    from organelleverse.annotation.table2asn import find_table2asn

    assert len(written.artifacts) == 11 + (1 if find_table2asn() else 0)


def _partial_pseudo_document() -> AnnotationDocument:
    partial_cds = AnnotationFeature(
        feature_id="r2:0:CDS",
        seqid="r2",
        type="CDS",
        operator="single",
        parts=(LocationPart(start=0, end=90, strand=1, start_status="before"),),
        qualifiers=(
            FeatureQualifier(name="gene", values=("cox2",)),
            FeatureQualifier(name="product", values=("cytochrome c oxidase subunit 2",)),
            FeatureQualifier(name="transl_table", values=("4",)),
            FeatureQualifier(name="exception", values=("RNA editing",)),
            FeatureQualifier(name="translation", values=("M" * 30,)),
        ),
        parents=(),
    )
    minus_partial_cds = AnnotationFeature(
        feature_id="r2:1:CDS",
        seqid="r2",
        type="CDS",
        operator="single",
        parts=(LocationPart(start=99, end=198, strand=-1, end_status="after"),),
        qualifiers=(
            FeatureQualifier(name="gene", values=("atp9",)),
            FeatureQualifier(name="product", values=("ATP synthase subunit 9",)),
            FeatureQualifier(name="translation", values=("H" * 33,)),
        ),
        parents=(),
    )
    pseudo_cds = AnnotationFeature(
        feature_id="r2:2:CDS",
        seqid="r2",
        type="CDS",
        operator="single",
        parts=(LocationPart(start=210, end=300, strand=1),),
        qualifiers=(
            FeatureQualifier(name="gene", values=("rps19",)),
            FeatureQualifier(name="product", values=("ribosomal protein S19",)),
            FeatureQualifier(name="pseudo", values=()),
            FeatureQualifier(name="translation", values=("M" * 30,)),
        ),
        parents=(),
    )
    return AnnotationDocument(
        backend="fixture",
        requested_stages=("pcg",),
        completed_stages=("pcg",),
        records=(
            AnnotationRecord(
                seqid="r2",
                name="r2",
                description="tbl fixture",
                sequence="ATG" * 100,
                features=(partial_cds, minus_partial_cds, pseudo_cds),
            ),
        ),
    )


def test_materialized_tbl_matches_expected_five_column_table(tmp_path: Path) -> None:
    paths = materialize_annotation(roundtrip_document(), tmp_path / "published")

    assert paths["tbl"].read_text() == (
        ">Feature r1\n"
        "1\t6\tgene\n"
        "13\t18\n"
        "\t\t\tgene\tnad1\n"
        "1\t6\tCDS\n"
        "13\t18\n"
        "\t\t\tgene\tnad1\n"
        "\t\t\tproduct\tNADH dehydrogenase subunit 1\n"
        "\t\t\ttransl_table\t1\n"
        "24\t19\tgene\n"
        "\t\t\tgene\ttrnM-CAU\n"
        "24\t19\ttRNA\n"
        "\t\t\tgene\ttrnM-CAU\n"
        "\t\t\tproduct\ttRNA-Met\n"
        "25\t30\tgene\n"
        "\t\t\tgene\trrn5\n"
        "25\t30\trRNA\n"
        "\t\t\tgene\trrn5\n"
        "\t\t\tproduct\trrn5\n"
    )


def test_materialized_tbl_marks_partial_ends_and_pseudo(tmp_path: Path) -> None:
    paths = materialize_annotation(_partial_pseudo_document(), tmp_path / "published")
    text = paths["tbl"].read_text()

    # plus-strand 5' partial: BeforePosition on the left coordinate
    assert "<1\t90\tgene" in text
    # minus-strand 5' partial is the genome-high coordinate, written first
    assert "<198\t100\tgene" in text
    # transl_table survives the document round-trip
    assert "\t\t\ttransl_table\t4\n" in text
    # pseudogene declares /pseudo and carries no product qualifier
    pseudo_block = text.split("211\t300\tCDS", 1)[1]
    assert "\t\t\tpseudo\n" in pseudo_block
    assert "ribosomal protein S19" not in pseudo_block


def test_unedited_tbl_drops_editing_exceptions(tmp_path: Path) -> None:
    paths = materialize_annotation(_partial_pseudo_document(), tmp_path / "published")

    assert "\t\t\texception\tRNA editing\n" in paths["tbl"].read_text()
    assert "exception" not in paths["unedited_tbl"].read_text()
    # every other line is identical between the pair
    edited = paths["tbl"].read_text().splitlines()
    unedited = paths["unedited_tbl"].read_text().splitlines()
    assert [line for line in edited if "exception" not in line] == unedited


def test_materialize_result_reports_tbl_artifacts(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    write_document_json(roundtrip_document(), source)
    result = OrganelleResult(
        operation_id="annotation.annotate",
        scope="mitochondrion",
        status="ok",
        artifacts=(ArtifactRef.from_path(source, kind="annotation", format="json"),),
    )

    written = materialize_result(result, tmp_path / "result-output")

    kinds = {artifact.kind for artifact in written.artifacts}
    assert "annotation_tbl" in kinds
    assert "annotation_tbl_unedited" in kinds
    tbl = next(a for a in written.artifacts if a.kind == "annotation_tbl")
    assert tbl.resolve().is_file()
