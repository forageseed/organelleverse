from types import SimpleNamespace

import pytest
from Bio.Data import CodonTable
from Bio.Seq import Seq

from organelleverse.annotation.mitochondrion.solver_rescue import rescue_trans_spliced


PROTEIN = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQANLPGIK"
CODONS = {aa: next(c for c, a in CodonTable.unambiguous_dna_by_id[1].forward_table.items() if a == aa)
          for aa in set(PROTEIN)}
CODING = "".join(CODONS[aa] for aa in PROTEIN)


def references(tmp_path):
    (tmp_path / "nad1.Protein.fasta").write_text(f">ref\n{PROTEIN}\n", encoding="utf8")
    return SimpleNamespace(blast_ref_dir=tmp_path)


@pytest.mark.parametrize("strand", [1, -1])
def test_actual_solver_recovers_complete_reference_and_original_coordinates(tmp_path, strand):
    coding = CODING if strand == 1 else str(Seq(CODING).reverse_complement())
    sequence = "N" * 17 + coding + "N" * 23
    models = rescue_trans_spliced(SimpleNamespace(sequence=sequence), references(tmp_path), ["NAD1", "nad1"])
    assert len(models) == 1
    model = models[0]
    assert len(model.exons) == 1
    exon = model.exons[0]
    assert (exon.start, exon.end, int(exon.strand)) == (18, 17 + len(CODING), strand)
    assert model.score > 30 and model.confidence == 0


def test_actual_solver_assembles_opposite_strands_in_protein_order(tmp_path):
    split = 20 * 3
    a, b = CODING[:split], CODING[split:]
    sequence = "N" * 11 + str(Seq(b).reverse_complement()) + "N" * 101 + a + "N" * 19
    models = rescue_trans_spliced(SimpleNamespace(sequence=sequence), references(tmp_path), ["nad1"])
    assert len(models) == 1
    model = models[0]
    assert [int(e.strand) for e in model.exons] == [1, -1]
    assert [e.number for e in model.exons] == [1, 2]
    assert model.exons[0].start > model.exons[1].start
    parts = [sequence[e.start - 1:e.end] for e in model.exons]
    peptide = str(Seq(parts[0] + str(Seq(parts[1]).reverse_complement())).translate())
    assert peptide == PROTEIN


@pytest.mark.parametrize("sequence", ["N" * 150, CODING[:60], "GCT" * len(PROTEIN)])
def test_incomplete_or_unrelated_sequences_are_not_published(tmp_path, sequence):
    assert rescue_trans_spliced(SimpleNamespace(sequence=sequence), references(tmp_path), ["nad1"]) == []


def test_missing_references_and_unrequested_genes_do_not_run_solver(tmp_path):
    genome = SimpleNamespace(sequence=CODING)
    assert rescue_trans_spliced(genome, SimpleNamespace(blast_ref_dir=None), ["nad1"]) == []
    assert rescue_trans_spliced(genome, references(tmp_path), ["cox1", "nad5"]) == []
