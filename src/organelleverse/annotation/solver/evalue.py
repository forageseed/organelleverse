"""Calibrate a profile against the genome it is searching, so a score means something.

The solver returns a best-scoring locus for every profile it is given, always.
Without a null model there is no way to say whether that locus is a gene or the
best of a genome full of noise, and the exhaustive genome-wide scan the design
insists on makes this worse rather than better: more positions searched means a
higher best-of-noise score.

Measured, the separation is wide. Scanning the whole 368 kb Arabidopsis genome on
both strands, the true locus beat the nearest *independent* candidate -- one at
least 20 kb away, rather than the same locus shifted -- by 67.8% for cox2 and
96.5% for nad3. So the threshold has a lot of room; it needs to exist, not to be
delicate.

Calibration is per profile and against the actual genome, not a generic
background: a profile is only ever asked "is this locus better than the rest of
*this* sequence", and organelle genomes have composition of their own. Scores of
random windows are fit to a Gumbel, which is the extreme-value distribution for a
maximum of many local alignments and is what lets a tail probability be quoted
from a few hundred samples instead of measured directly.

This is the ``E-value decides homology, tiers decide function`` split: kept apart,
completeness and precision stop trading against each other, and the threshold here
can stay loose because it is not also being asked to reject pseudogenes.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .dp import Profile, translate


@dataclass(frozen=True)
class Calibration:
    """Gumbel parameters for one profile against one genome."""

    mu: float
    lambda_: float
    n_samples: int
    genome_length: int

    def pvalue(self, score: float) -> float:
        """P(random window scores at least this well)."""
        z = self.lambda_ * (score - self.mu)
        if z > 30:                       # exp(-exp(-30)) underflows to 1.0
            return math.exp(-z)
        return 1.0 - math.exp(-math.exp(-z))

    def evalue(self, score: float) -> float:
        """Expected number of such hits by chance in a genome-wide scan.

        Both strands, and one trial per position: the search really does look
        everywhere, so the multiple-testing correction has to say so.
        """
        return self.pvalue(score) * 2.0 * self.genome_length


def shuffle_preserving_composition(genome: str, rng: random.Random) -> str:
    """Shuffle the genome, keeping its base composition."""
    bases = list(genome)
    rng.shuffle(bases)
    return "".join(bases)


def calibrate(
    profile: Profile,
    genome: str,
    *,
    n_samples: int = 200,
    seed: int = 20260804,
    n_shuffles: int = 2,
) -> Calibration:
    """Fit a Gumbel to what this profile scores on sequence that holds no gene.

    The null is scored **the same way the search is** -- the full gapped
    recursion, over shuffled genome. A first version sampled ungapped windows
    instead, on the grounds that it would be cheaper and would only make the
    E-values conservative. It did the opposite: the gapped optimum is far higher
    than any ungapped window, so every locus cleared the threshold and random
    regions came back at 1e-19. A null model has to be scored by the same
    function as the thing it is calibrating, or it is not a null model.
    """
    from .fast import scan_fast

    genome = genome.upper()
    span = 3 * len(profile)
    if len(genome) <= span + 3 or len(profile) == 0:
        return Calibration(mu=0.0, lambda_=1.0, n_samples=0, genome_length=len(genome))

    rng = random.Random(seed)
    scores: list[float] = []
    for _ in range(n_shuffles):
        decoy = shuffle_preserving_composition(genome, rng)
        hits = scan_fast(profile, decoy, top_n=n_samples // n_shuffles,
                         min_separation=3 * len(profile))
        scores.extend(s for s, _ in hits)
    if len(scores) < 8:
        return Calibration(mu=0.0, lambda_=1.0, n_samples=len(scores),
                           genome_length=len(genome))

    mean = sum(scores) / len(scores)
    var = sum((s - mean) ** 2 for s in scores) / max(len(scores) - 1, 1)
    sd = math.sqrt(var) if var > 0 else 1.0
    # Method of moments for a Gumbel: sd = pi / (lambda * sqrt(6)),
    # mean = mu + gamma / lambda, with gamma the Euler-Mascheroni constant.
    lambda_ = math.pi / (sd * math.sqrt(6.0))
    mu = mean - 0.5772156649 / lambda_
    return Calibration(mu=mu, lambda_=lambda_, n_samples=len(scores),
                       genome_length=len(genome))


def is_present(score: float, calibration: Calibration, *, max_evalue: float = 1e-3) -> bool:
    """Whether a score is better than chance -- homology only, not function.

    Deliberately permissive. Deciding whether a gene is functional is the tiers'
    job, and merging the two questions into one threshold is what produced the
    per-gene score table this solver was built to remove.
    """
    return calibration.evalue(score) <= max_evalue
