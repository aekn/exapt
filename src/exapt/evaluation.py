import math
from collections.abc import Mapping, Sequence


def rank_drugs(scores: Mapping[int, float], candidates: Sequence[int]) -> list[int]:
    if len(scores) != len(candidates) or set(scores) != set(candidates):
        raise ValueError("Scores must cover the candidate drug universe")
    if not all(math.isfinite(score) for score in scores.values()):
        raise ValueError("Drug scores must be finite")

    return sorted(candidates, key=lambda drug: (-scores[drug], drug))


def disease_metrics(ranking: Sequence[int], positives: set[int]) -> dict[str, float]:
    if not positives:
        raise ValueError("A disease needs at least one observed indication")

    hits = 0
    precision_sum = 0.0
    dcg_at_20 = 0.0
    first_hit_rank = 0
    hits_at_k = {5: 0, 10: 0, 20: 0, 50: 0}

    for rank, drug in enumerate(ranking, 1):
        if drug not in positives:
            continue

        hits += 1
        precision_sum += hits / rank
        if first_hit_rank == 0:
            first_hit_rank = rank
        if rank <= 20:
            dcg_at_20 += 1 / math.log2(rank + 1)
        for k in hits_at_k:
            if rank <= k:
                hits_at_k[k] += 1

    # normalize by all observed indications, including unretrieved positives
    ideal_dcg_at_20 = sum(
        1 / math.log2(rank + 1)
        for rank in range(1, min(len(positives), 20) + 1)
    )
    result = {
        f"recall@{k}": count / len(positives)
        for k, count in hits_at_k.items()
    }
    result.update({
        "map": precision_sum / len(positives),
        "ndcg@20": dcg_at_20 / ideal_dcg_at_20,
        "first_hit_rr": 1 / first_hit_rank if first_hit_rank else 0.0,
        "hit@20": float(hits_at_k[20] > 0),
        "observed_precision@20": hits_at_k[20] / 20,
    })
    return result


def macro_metrics(
    rankings: Mapping[int, Sequence[int]],
    positives: Mapping[int, set[int]],
) -> dict[str, float]:
    if not positives or set(rankings) != set(positives):
        raise ValueError("Rankings must cover every evaluated disease")

    results = [
        disease_metrics(rankings[disease], labels)
        for disease, labels in positives.items()
    ]
    return {
        key: sum(result[key] for result in results) / len(results)
        for key in results[0]
    }


def macro_recall_at_k(
    rankings: Mapping[int, Sequence[int]],
    positives: Mapping[int, set[int]],
    k: int = 20,
) -> float:
    if k <= 0 or not positives or set(rankings) != set(positives):
        raise ValueError("Invalid recall evaluation inputs")
    if any(not targets for targets in positives.values()):
        raise ValueError("Every evaluated disease needs an observed positive")

    return sum(
        len(set(rankings[disease][:k]) & labels) / len(labels)
        for disease, labels in positives.items()
    ) / len(positives)
