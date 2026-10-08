"""Deterministic synthetic SV truth, generated independently of PGGB outputs."""

import random
from collections import defaultdict

COMPLEMENT = str.maketrans("ACGT", "TGCA")


def case(seed, case_id):
    rng = random.Random(seed)
    blocks = ["".join(rng.choice("ACGT") for _ in range(1200)) for _ in range(10)]
    orders = {
        "a#1#1": [(i, False) for i in range(10)],
        "b#1#1": [
            (0, False),
            (1, False),
            (4, True),
            (3, True),
            (2, True),
            (5, False),
            (6, False),
            (7, False),
            (8, False),
            (9, False),
        ],
        "c#1#1": [
            (0, False),
            (7, False),
            (1, False),
            (2, False),
            (4, False),
            (5, False),
            (6, False),
            (8, False),
            (9, False),
            (4, False),
        ],
    }
    sequences, origins = {}, {}
    for sample_index, (name, order) in enumerate(orders.items()):
        sequence, lineage = [], []
        for block_id, reverse in order:
            block = blocks[block_id]
            positions = list(range(len(block)))
            if reverse:
                block, positions = block.translate(COMPLEMENT)[::-1], positions[::-1]
            for base, ancestral in zip(block, positions, strict=True):
                sequence.append(base)
                lineage.append((block_id, ancestral))
        # Mutations have their own derived identity; they are excluded from
        # conserved-positive anchors, not mislabeled as graph sharing failures.
        if sample_index:
            for position in range(41 * sample_index, len(sequence), 113):
                sequence[position] = rng.choice(
                    [base for base in "ACGT" if base != sequence[position]]
                )
                lineage[position] = (f"derived-{sample_index}", position)
        sequences[name], origins[name] = "".join(sequence), lineage
    reference_name = "a#1#1"
    positives = []
    for name in list(orders)[1:]:
        by_origin = defaultdict(list)
        for position, origin in enumerate(origins[name]):
            by_origin[origin].append(position)
        for position, origin in enumerate(origins[reference_name]):
            for other in by_origin[origin]:
                positives.append((reference_name, position, name, other))
    rng.shuffle(positives)
    anchors = []
    used = set()
    for first, position, second, other in positives[:800]:
        used.add((first, position, second, other))
        anchors.append(
            {
                "first": {"path": first, "position": position},
                "second": {"path": second, "position": other},
                "homologous": True,
            }
        )
    # Negative anchors deliberately have equal bases, so base composition alone
    # cannot solve the classification. Their ancestral positions are distinct.
    while len(anchors) < 1600:
        second = rng.choice(list(orders)[1:])
        first_position = rng.randrange(len(sequences[reference_name]))
        second_position = rng.randrange(len(sequences[second]))
        key = (reference_name, first_position, second, second_position)
        if (
            key in used
            or origins[reference_name][first_position] == origins[second][second_position]
        ):
            continue
        if sequences[reference_name][first_position] != sequences[second][second_position]:
            continue
        used.add(key)
        anchors.append(
            {
                "first": {"path": reference_name, "position": first_position},
                "second": {"path": second, "position": second_position},
                "homologous": False,
            }
        )
    return {
        "schema_version": "organelleverse.pangenome.graph-truth.v1",
        "benchmark_id": "pangenome.synthetic_conserved_homology",
        "benchmark_version": "1.0.0",
        "case_id": case_id,
        "source_description": f"Controlled synthetic sequence ancestry, seed {seed}; 10 independently generated 1200-bp blocks, a three-block inversion, transposition, deletion, duplication, and deterministic substitutions. Not a biological validation cohort.",
        "sequences": sequences,
        "anchors": anchors,
    }
