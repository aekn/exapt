from __future__ import annotations

import heapq
import math
from collections import defaultdict
from collections.abc import Collection, Sequence
from itertools import accumulate, pairwise
from typing import NamedTuple


Pair = tuple[int, int]  # (drug index, disease index)


class DiseaseRanking(NamedTuple):
    disease: int
    positives: int
    recall: float
    ndcg: float
    reciprocal_rank: float


class RankingResult(NamedTuple):
    k: int
    per_disease: tuple[DiseaseRanking, ...]

    @property
    def recall(self) -> float:
        return sum(row.recall for row in self.per_disease) / len(self.per_disease)

    @property
    def ndcg(self) -> float:
        return sum(row.ndcg for row in self.per_disease) / len(self.per_disease)

    @property
    def reciprocal_rank(self) -> float:
        return sum(row.reciprocal_rank for row in self.per_disease) / len(self.per_disease)


def _ordered_unique(values: Sequence[int], name: str) -> None:
    if any(a >= b for a, b in pairwise(values)):
        raise ValueError(f"evaluation: {name} must be strictly increasing")


def evaluate_ranking(
    scores: Sequence[Sequence[float]],
    *,
    diseases: Sequence[int],
    drugs: Sequence[int],
    positives: Collection[Pair],
    k: int,
) -> RankingResult:
    if k < 1:
        raise ValueError("evaluation: k must be positive")
    if not diseases or not drugs:
        raise ValueError("evaluation: diseases and drugs must be nonempty")

    _ordered_unique(diseases, "diseases")
    _ordered_unique(drugs, "drugs")

    if len(scores) != len(diseases):
        raise ValueError("evaluation: score row count does not match diseases")

    disease_set, drug_set = set(diseases), set(drugs)
    labels: dict[int, set[int]] = defaultdict(set)

    for drug, disease in positives:
        if disease not in disease_set or drug not in drug_set:
            raise ValueError("evaluation: positive is outside the evaluation population")
        labels[disease].add(drug)

    limit = min(k, len(drugs))
    discounts = [1.0 / math.log2(rank + 1) for rank in range(1, limit + 1)]
    ideal = tuple(accumulate(discounts, initial=0.0))

    results: list[DiseaseRanking] = []

    for disease, row in zip(diseases, scores, strict=True):
        relevant = labels[disease]

        if not relevant:
            raise ValueError(f"evaluation: disease {disease} has no recorded positives")
        if len(row) != len(drugs):
            raise ValueError(f"evaluation: disease {disease} has the wrong score count")
        if any(not math.isfinite(score) for score in row):
            raise ValueError(f"evaluation: disease {disease} has a nonfinite score")

        top = heapq.nsmallest(limit, range(len(drugs)), key=lambda i: (-row[i], drugs[i]))

        hits = 0
        discounted = 0.0
        first_rank = 0

        for rank, index in enumerate(top, start=1):
            if drugs[index] in relevant:
                hits += 1
                discounted += discounts[rank - 1]

                if not first_rank:
                    first_rank = rank

        if not first_rank:
            best = min(
                (-row[i], drugs[i]) for i in range(len(drugs)) if drugs[i] in relevant
            )
            first_rank = 1 + sum(
                (-score, drug) < best for score, drug in zip(row, drugs, strict=True)
            )

        results.append(
            DiseaseRanking(
                disease=disease,
                positives=len(relevant),
                recall=hits / len(relevant),
                ndcg=discounted / ideal[min(len(relevant), limit)],
                reciprocal_rank=1.0 / first_rank,
            )
        )

    return RankingResult(k, tuple(results))
