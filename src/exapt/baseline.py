from __future__ import annotations

import math
from collections import Counter, defaultdict

from .benchmark import DiseaseSplit, Therapeutics, indication_partition, validate_split
from .evaluation import RankingResult, evaluate_ranking
from .graph import Graph


def drug_popularity(
    facts: Therapeutics, split: DiseaseSplit, *, k: int = 20,
) -> tuple[RankingResult, tuple[int, ...], tuple[int, ...]]:
    validate_split(facts, split)

    drugs = tuple(sorted(facts.drugs))
    diseases = tuple(sorted(split.development))
    counts = Counter(
        drug for drug, disease in facts.unambiguous_indications
        if disease in split.train
    )
    scores = tuple(counts[drug] for drug in drugs)
    result = evaluate_ranking(
        [scores] * len(diseases),
        diseases=diseases,
        drugs=drugs,
        positives=indication_partition(facts, split.development),
        k=k,
    )
    return result, drugs, scores


def target_protein_scores(
    graph: Graph,
    *,
    diseases: tuple[int, ...],
    drugs: tuple[int, ...],
    damping: float = 0.5,
) -> list[list[float]]:
    if not math.isfinite(damping) or damping < 0:
        raise ValueError("baseline: invalid path damping")

    disease_neighbors: dict[int, set[int]] = defaultdict(set)
    drug_neighbors: dict[int, set[int]] = defaultdict(set)

    for relation, edges in zip(graph.relations, graph.edges, strict=True):
        if relation.name == "disease_protein" and relation.display == "associated with":
            neighbors = disease_neighbors
            endpoint_type = "disease"
        elif relation.name == "drug_protein" and relation.display == "target":
            neighbors = drug_neighbors
            endpoint_type = "drug"
        else:
            continue

        if (relation.source_type, relation.target_type) == (endpoint_type, "gene/protein"):
            pairs = zip(edges.source, edges.target, strict=True)
        elif (relation.source_type, relation.target_type) == ("gene/protein", endpoint_type):
            pairs = zip(edges.target, edges.source, strict=True)
        else:
            raise ValueError(f"baseline: unexpected endpoints for {relation.name}")

        for entity, protein in pairs:
            neighbors[int(protein)].add(int(entity))

    disease_rows = {disease: row for row, disease in enumerate(diseases)}
    drug_columns = {drug: column for column, drug in enumerate(drugs)}
    scores = [[0.0] * len(drugs) for _ in diseases]

    for protein in sorted(disease_neighbors.keys() & drug_neighbors.keys()):
        associated = disease_neighbors[protein]
        targets = drug_neighbors[protein]
        weight = (len(associated) * len(targets)) ** -damping

        for disease in sorted(associated & disease_rows.keys()):
            row = scores[disease_rows[disease]]
            for drug in sorted(targets & drug_columns.keys()):
                row[drug_columns[drug]] += weight

    return scores


def protein_target_paths(
    facts: Therapeutics,
    split: DiseaseSplit,
    graph: Graph,
    *,
    k: int = 20,
    damping: float = 0.5,
) -> tuple[RankingResult, tuple[int, ...]]:
    validate_split(facts, split)
    diseases = tuple(sorted(split.development))
    drugs = tuple(sorted(facts.drugs))
    scores = target_protein_scores(
        graph, diseases=diseases, drugs=drugs, damping=damping,
    )
    result = evaluate_ranking(
        scores, diseases=diseases, drugs=drugs,
        positives=indication_partition(facts, split.development), k=k,
    )
    covered = tuple(
        disease for disease, row in zip(diseases, scores, strict=True)
        if any(score > 0 for score in row)
    )
    return result, covered
