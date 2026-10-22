import math
from collections import Counter, defaultdict
from dataclasses import dataclass

import torch

from .graph import Graph


PROFILE_RELATIONS = (
    "disease_protein", "disease_disease", "disease_phenotype_positive",
)


@dataclass
class Prototypes:
    # q = supported query diseases, k = donor slots
    diseases: torch.Tensor  # (q,)
    donors: torch.Tensor    # (q, k)
    weights: torch.Tensor   # (q, k)
    evidence: torch.Tensor  # (q, 3) similarity, agreement, availability


def build_prototypes(
    graph: Graph, train_diseases: set[int], query_diseases: set[int],
    k: int = 3,
) -> Prototypes:
    if k < 1:
        raise ValueError("k must be positive")

    relevant = train_diseases | query_diseases
    donor_diseases = sorted(train_diseases)

    # profiles[view][disease] = neighboring biological node indices
    profiles: dict[str, dict[int, set[int]]] = {
        name: {} for name in PROFILE_RELATIONS
    }
    for relation, matrix in zip(graph.relations, graph.adjacency):
        source_type, name, target_type = relation
        if name not in profiles:
            continue

        # A_r[target, source]; collect disease neighbors in either direction
        targets, sources = matrix.indices().tolist()
        view = profiles[name]
        for source, target in zip(sources, targets):
            if source == target:
                continue
            if source_type == "disease" and source in relevant:
                view.setdefault(source, set()).add(target)
            if target_type == "disease" and target in relevant:
                view.setdefault(target, set()).add(source)

    # inverted[view][neighbor] = training diseases sharing that neighbor
    inverted: dict[str, dict[int, list[int]]] = {}
    for name, view in profiles.items():
        neighbors: dict[int, list[int]] = defaultdict(list)
        for disease in donor_diseases:
            for neighbor in view.get(disease, ()):
                neighbors[neighbor].append(disease)
        inverted[name] = neighbors

    # (query disease, donor indices, normalized weights, evidence)
    rows: list[tuple[int, list[int], list[float], list[float]]] = []
    for disease in sorted(query_diseases):
        available_views = [
            name for name in PROFILE_RELATIONS if profiles[name].get(disease)
        ]
        if not available_views:
            continue

        # sum cosine similarities; count views with shared neighbors
        similarity: dict[int, float] = defaultdict(float)
        supporting_views: Counter[int] = Counter()
        for name in available_views:
            query_neighbors = profiles[name][disease]
            overlaps: Counter[int] = Counter()
            for neighbor in query_neighbors:
                overlaps.update(inverted[name].get(neighbor, ()))

            for donor, overlap in overlaps.items():
                if donor == disease:
                    continue

                donor_size = len(profiles[name][donor])
                norm = math.sqrt(len(query_neighbors) * donor_size)
                similarity[donor] += overlap / norm
                supporting_views[donor] += 1

        # mean over query-available views; absent donor overlap counts as zero
        view_count = len(available_views)
        candidates = (
            (score / view_count, donor)
            for donor, score in similarity.items()
        )
        nearest = sorted(candidates, key=lambda pair: (-pair[0], pair[1]))[:k]
        if not nearest:
            continue

        total = sum(score for score, _ in nearest)
        donors = [donor for _, donor in nearest]
        weights = [score / total for score, _ in nearest]

        # evidence belongs to the best donor, not the weighted prototype
        best_score, best_donor = nearest[0]
        evidence = [
            best_score, supporting_views[best_donor] / view_count,
            view_count / len(PROFILE_RELATIONS),
        ]

        # zero-weight padding keeps prototype unchanged
        missing = k - len(nearest)
        donors.extend([donors[0]] * missing)
        weights.extend([0.0] * missing)
        rows.append((disease, donors, weights, evidence))

    # retain tensor ranks when no query has eligible donors
    diseases = torch.tensor([row[0] for row in rows], dtype=torch.long)
    donors = torch.tensor(
        [row[1] for row in rows], dtype=torch.long
    ).reshape(-1, k)
    weights = torch.tensor(
        [row[2] for row in rows], dtype=torch.float
    ).reshape(-1, k)
    evidence = torch.tensor(
        [row[3] for row in rows], dtype=torch.float
    ).reshape(-1, 3)
    return Prototypes(diseases, donors, weights, evidence)
