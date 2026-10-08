"""Phase repair must preserve valid models and trim the biological 3' end."""
import pytest
from Bio.Seq import Seq
from Bio.SeqFeature import CompoundLocation, FeatureLocation, SeqFeature

from organelleverse.annotation.plastome.cds_refine import refine_cds_features


def model(sequence, exons, strand):
    if strand == -1:
        sequence = str(Seq(sequence).reverse_complement())
        exons = [(len(sequence)-e+1, len(sequence)-s+1) for s,e in exons]
    feature = SeqFeature(CompoundLocation([FeatureLocation(s-1,e,strand=strand) for s,e in exons]), type='CDS')
    return sequence, feature, exons


@pytest.mark.parametrize('strand', [1,-1])
def test_terminal_start_shift_keeps_valid_preterminal_phase(strand):
    bases = list('C'*60)
    bases[9:18] = 'CATGCCCCC'
    sequence, feature, original = model(''.join(bases), [(10,18),(30,38)], strand)
    expected = str(feature.extract(Seq(sequence)))
    refine_cds_features(sequence,[feature],window=1,multi_ext=0)
    assert [(int(p.start)+1,int(p.end)) for p in feature.location.parts] == original
    assert str(feature.extract(Seq(sequence))) == expected
    assert 'partial' not in feature.qualifiers


@pytest.mark.parametrize('strand', [1,-1])
def test_unresolved_phase_trim_preserves_internal_join(strand):
    sequence, feature, _original = model('C'*60,[(10,18),(30,37)],strand)
    expected = [(10,18),(30,35)]
    if strand == -1:
        expected = [(len(sequence)-e+1,len(sequence)-s+1) for s,e in expected]
    refine_cds_features(sequence,[feature],window=0,multi_ext=0)
    assert [(int(p.start)+1,int(p.end)) for p in feature.location.parts] == expected
    assert feature.qualifiers['partial'] == ['true']
    assert len(feature.extract(Seq(sequence))) == 15
