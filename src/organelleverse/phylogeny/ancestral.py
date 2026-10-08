"""Equal-rates Mk ancestral reconstruction by pruning and outside messages.

Lewis (2001), Systematic Biology 50:913-925, doi:10.1080/106351501753462876.
Q has off-diagonal rate/(k-1), diagonal -rate and a uniform root prior.
No ascertainment correction: characters must not be selected to be variable.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from Bio import Phylo, SeqIO
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

from ..core.errors import OrganelleInputError
from ..core.result import OrganelleResult
from ._results import provenance


def _error(message):
    return OrganelleInputError(code="phylogeny.ancestral.input", message=message)


def _transition(length, rate, k):
    decay = math.exp(-rate * length * k / (k - 1))
    off = -math.expm1(-rate * length * k / (k - 1)) / k
    matrix = np.full((k, k), off)
    np.fill_diagonal(matrix, off + decay)
    with np.errstate(divide="ignore"):
        return np.log(matrix)


def _messages(tree, observations, states, rate, posterior=True):
    """Exact log-space sum-product on a tree (including multifurcations)."""
    k = len(states)
    index = {s: i for i, s in enumerate(states)}
    inside, messages, transitions = {}, {}, {}
    for node in tree.find_clades(order="postorder"):
        value = np.zeros(k)
        if node.is_terminal() and observations[node.name] is not None:
            value[:] = -np.inf
            value[index[observations[node.name]]] = 0
        for child in node.clades:
            transitions[child] = _transition(child.branch_length, rate, k)
            messages[child] = logsumexp(transitions[child] + inside[child][None, :], axis=1)
            value += messages[child]
        inside[node] = value
    likelihood = float(logsumexp(inside[tree.root]) - math.log(k))
    if not posterior or not math.isfinite(likelihood):
        return likelihood, {}
    outside = {tree.root: np.full(k, -math.log(k))}
    probabilities = {}
    for node in tree.find_clades(order="preorder"):
        value = inside[node] + outside[node]
        probabilities[node] = np.exp(value - logsumexp(value))
        for child in node.clades:
            # Exclude the child's message without subtracting infinities at
            # zero-length branches or rate=0. Summing only sibling messages
            # also handles genuine zero transition probabilities.
            siblings = np.zeros(k)
            for sibling in node.clades:
                if sibling is not child:
                    siblings += messages[sibling]
            outside[child] = logsumexp(
                (outside[node] + siblings)[:, None] + transitions[child], axis=0
            )
    return likelihood, probabilities


def reconstruct_ancestral_states(
    tree_newick: str | Path,
    traits: dict[str, dict[str, str | int | None]],
    *,
    state_space: dict[str, list[str]] | None = None,
    rate: float | None = None,
    rate_bounds: list[float] | None = None,
) -> OrganelleResult:
    """Reconstruct discrete traits on the supplied rooted Newick tree with Mk.

    ``traits`` maps every tip name to the same trait columns. ``None`` denotes
    unknown; it is never interpreted as absence. The first Newick node is the
    user-supplied root: this operation does not infer a root or reroot a tree.
    Every nonroot edge needs an explicit nonnegative finite branch length.

    Each trait is independent, with uniform equilibrium/root frequencies and
    equal exchange rates. Set ``rate`` to fix the expected changes per unit
    branch length; otherwise maximize likelihood over log(rate) within
    ``rate_bounds`` (default 1e-6..1e6) using bounded scalar optimization.
    A boundary solution is reported as warning, not an identified rate.
    Constant traits require a declared state_space with at least two states.
    Estimating a rate requires at least two observed tips separated by a
    positive tree distance; otherwise the likelihood is constant in rate.

    Changes connect different singleton marginal MAP states, including tips.
    They are not joint-MAP histories, stochastic mappings, expected numbers
    of changes, or branch-change posterior probabilities. Exact posterior
    ties remain sets and do not produce arbitrarily polarized changes.
    """
    bounds = rate_bounds if rate_bounds is not None else [1e-6, 1e6]
    if (
        len(bounds) != 2
        or not all(math.isfinite(x) for x in bounds)
        or not 0 < bounds[0] < bounds[1]
    ):
        raise _error("rate_bounds must contain two increasing positive finite numbers.")
    if rate is not None and (not math.isfinite(rate) or rate < 0):
        raise _error("rate must be nonnegative and finite.")
    try:
        tree = Phylo.read(tree_newick, "newick")
    except ValueError as exc:
        raise _error(f"Invalid Newick tree: {exc}") from exc
    tips = tree.get_terminals()
    names = [n.name for n in tips]
    if len(tips) < 2 or any(not name for name in names) or len(set(names)) != len(names):
        raise _error("A tree needs at least two uniquely named tips.")
    if set(traits) != set(names):
        raise _error(
            "Trait sample names must exactly match tree tips; use None for missing states."
        )
    columns = sorted(traits[names[0]])
    if not columns or any(set(row) != set(columns) for row in traits.values()):
        raise _error("Every sample must have the same nonempty trait columns.")
    nodes = list(tree.find_clades(order="preorder"))
    for node in nodes[1:]:
        if (
            node.branch_length is None
            or not math.isfinite(node.branch_length)
            or node.branch_length < 0
        ):
            raise _error("Each nonroot branch needs an explicit finite nonnegative length.")
    ids = {node: f"node_{i}" for i, node in enumerate(nodes)}
    inferred, changes, fitted, flags = {}, [], {}, []
    for column in columns:
        observations = {
            name: None if traits[name][column] is None else str(traits[name][column])
            for name in names
        }
        observed = {v for v in observations.values() if v is not None}
        states = (
            sorted(state_space[column])
            if state_space and column in state_space
            else sorted(observed)
        )
        if (
            len(states) < 2
            or len(set(states)) != len(states)
            or not observed
            or not observed <= set(states)
        ):
            raise _error(
                f"Trait {column!r} needs observations and >=2 distinct allowed states; supply state_space for constant traits."
            )
        chosen = rate
        at_boundary = False
        if chosen is None:
            known = [tip for tip in tips if observations[tip.name] is not None]
            if not any(tree.distance(known[0], tip) > 0 for tip in known[1:]):
                raise _error(
                    f"Rate is not identifiable for {column!r}: supply rate or observations "
                    "on at least two tips separated by positive tree distance."
                )

            def objective(x, observations=observations, states=states):
                return -_messages(tree, observations, states, math.exp(x), False)[0]

            fit = minimize_scalar(
                objective, bounds=tuple(math.log(x) for x in bounds), method="bounded"
            )
            if not fit.success:
                raise _error(f"Rate optimization failed for {column}: {fit.message}")
            # Include both declared endpoints in the bounded optimization.
            candidates = [(float(fit.fun), math.exp(float(fit.x)), False)]
            candidates += [(objective(math.log(x)), x, True) for x in bounds]
            _, chosen, at_boundary = min(candidates, key=lambda item: item[0])
            if at_boundary:
                flags.append(f"rate_at_boundary:{column}")
        likelihood, probs = _messages(tree, observations, states, chosen)
        if not math.isfinite(likelihood):
            raise _error(f"Trait {column!r} is impossible under the supplied branch lengths/rate.")
        fitted[column] = {
            "states": states,
            "rate": chosen,
            "rate_estimated": rate is None,
            "rate_at_boundary": at_boundary,
            "log_likelihood": likelihood,
        }
        map_states = {
            node: [s for s, p in zip(states, probs[node], strict=True) if p == max(probs[node])]
            for node in nodes
        }
        inferred[column] = [
            {
                "node": ids[n],
                "name": n.name,
                "is_tip": n.is_terminal(),
                "descendant_tips": sorted(t.name for t in n.get_terminals()),
                "posterior": dict(zip(states, map(float, probs[n]), strict=True)),
                "most_likely_states": map_states[n],
                "most_likely_state": map_states[n][0] if len(map_states[n]) == 1 else None,
            }
            for n in nodes
        ]
        for parent in nodes:
            for child in parent.clades:
                a, b = map_states[parent], map_states[child]
                if len(a) == len(b) == 1 and a != b:
                    changes.append(
                        {
                            "trait": column,
                            "parent": ids[parent],
                            "child": ids[child],
                            "from_state": a[0],
                            "to_state": b[0],
                            "parent_state_probability": float(probs[parent][states.index(a[0])]),
                            "child_state_probability": float(probs[child][states.index(b[0])]),
                            "descendant_tips": sorted(t.name for t in child.get_terminals()),
                        }
                    )
    return OrganelleResult(
        operation_id="phylogeny.reconstruct_ancestral_states",
        scope="none",
        status="warning" if flags else "ok",
        flags=tuple(flags),
        summary_text=f"Reconstructed {len(columns)} traits on {len(tips)} tips with the equal-rates Mk model.",
        metrics={
            "model": "Mk_equal_rates",
            "root_prior": "uniform",
            "root": ids[tree.root],
            "root_policy": "supplied_Newick_root",
            "traits": inferred,
            "models": fitted,
            "changes": changes,
            "rate_bounds": bounds if rate is None else None,
            "change_definition": "different singleton marginal MAP endpoint states",
        },
        provenance=provenance(
            "reconstruct_ancestral_states",
            method="Mk_log_pruning",
            parameters={
                "tree_newick": str(tree_newick),
                "traits": traits,
                "state_space": state_space,
                "rate": rate,
                "rate_bounds": bounds,
            },
        ),
    )


def gene_presence_traits(
    genbank_paths: list[str | Path],
    genes: list[str],
    *,
    sample_names: list[str] | None = None,
    include_pseudogenes: bool = False,
) -> OrganelleResult:
    """Build gene presence/absence traits from package-exported GenBank CDS.

    One sample per file, including multipart assemblies. Names default to
    each file's stem. Presence means an annotated CDS with a case-insensitive
    gene qualifier. Pseudogenes are excluded unless explicitly requested.
    Zero means 'not annotated', not experimentally established gene loss.
    """
    names = sample_names if sample_names is not None else [Path(p).stem for p in genbank_paths]
    if not genes or len({g.casefold() for g in genes}) != len(genes):
        raise _error("genes must be a nonempty list of distinct names.")
    if not genbank_paths or len(names) != len(genbank_paths) or len(set(names)) != len(names):
        raise _error("Provide one unique sample name per GenBank file.")
    traits, evidence = {}, {}
    for name, path in zip(names, genbank_paths, strict=True):
        records = list(SeqIO.parse(path, "genbank"))
        if not records or not any(r.features for r in records):
            raise _error(f"No annotation features in {path}.")
        present = {
            g.casefold()
            for rec in records
            for f in rec.features
            if f.type == "CDS"
            and (include_pseudogenes or not {"pseudo", "pseudogene"} & set(f.qualifiers))
            for g in f.qualifiers.get("gene", [])
        }
        traits[name] = {g: int(g.casefold() in present) for g in genes}
        evidence[name] = {"file_name": Path(path).name, "records": len(records)}
    return OrganelleResult(
        operation_id="phylogeny.gene_presence_traits",
        scope="none",
        status="ok",
        summary_text=f"Built {len(genes)} annotation-presence traits for {len(names)} samples.",
        metrics={
            "traits": traits,
            "state_space": {g: ["0", "1"] for g in genes},
            "absence_definition": "not annotated as CDS",
            "evidence": evidence,
            "include_pseudogenes": include_pseudogenes,
        },
        provenance=provenance(
            "gene_presence_traits",
            method="GenBank_CDS",
            parameters={
                "genbank_paths": [str(p) for p in genbank_paths],
                "genes": genes,
                "sample_names": names,
                "include_pseudogenes": include_pseudogenes,
            },
        ),
    )
