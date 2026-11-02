"""Transit-peptide model: known plastid and mitochondrial targeting peptides, and few calls on random sequences."""

from __future__ import annotations

import random

from organelleverse.localization.localize_core import predict_heuristic, score_transit_peptides

# Arabidopsis RbcS 1A (chloroplast transit peptide)
_RBCS = (
    "MASSMLSSATMVASPAQATMVAPFNGLKSSAAFPATRKANNDITSITSNGGRVNCMKVWPPIGKKKFETLSYLPDLTDSELAKEVDYLIRNKWIPCVEFELEH"
    "GFVYREHGNSPGYYDGRYWTMWKLPLFGCTDSAQVLKEVEECKKEYPNAFIRIIGFDNTRQVQCISFIAYKPPSFT"
)
# yeast COX4 (classical mitochondrial presequence)
_COX4 = (
    "MLSLRQSIRFFKPATRTLCSSRYLLQQKPVVKTAQNLAEVNGPETLIGPGAKEGTVPTDLDQETGLARLELLGKLEGIDVFDTKPLDSSRKGTMKDPIIIES"
    "YDDYRYVGCTGSPAGSHTIMWLKPTVNEVARCWECGSVYKLNPVGVPNDDHHH"
)
_ACTIN = (
    "MADAEDIQPLVCDNGTGMVKAGFAGDDAPRAVFPSIVGRPRHTGVMVGMGQKDSYVGDEAQSKRGILTLKYPIEHGIVSNWDDMEKIWHHTFYNELRVAPEEH"
    "PVLLTEAPLNPKANREKMTQIMFETFNVPAMYVAIQAVLSLYASGRTTGIVLDSGDGVTHTVPIYEGYALPHAIIRLDLAGRDLTDYLMKILTERGYSFTTT"
)


def test_known_targeting_peptides_are_called() -> None:
    assert score_transit_peptides(_RBCS)["predicted"] == "Plastid"
    assert score_transit_peptides(_COX4)["predicted"] == "Mitochondrion"
    assert score_transit_peptides(_ACTIN)["predicted"] is None
    # a call reaches the 0.5 reporting level of the combined predictor
    assert "Mitochondrion" in predict_heuristic(_COX4).predicted
    assert "Plastid" in predict_heuristic(_RBCS).predicted


def test_random_sequences_are_rarely_called_mitochondrial() -> None:
    rng = random.Random(7)
    seqs = ["M" + "".join(rng.choice("ACDEFGHIKLMNPQRSTVWY") for _ in range(399)) for _ in range(500)]
    rate = sum(score_transit_peptides(s)["predicted"] == "Mitochondrion" for s in seqs) / len(seqs)
    assert rate < 0.03  # the former composition heuristic called about 18%
