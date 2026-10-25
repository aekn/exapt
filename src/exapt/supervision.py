from __future__ import annotations

import random
from typing import NamedTuple

from .benchmark import DiseaseSplit, Therapeutics, validate_split


class TrainingSupervision(NamedTuple):
    diseases: tuple[int, ...]
    drugs: tuple[int, ...]
    positives: tuple[tuple[int, ...], ...]
    excluded: tuple[frozenset[int], ...]


def training_supervision(
    facts: Therapeutics,
    split: DiseaseSplit,
    *,
    query_diseases: frozenset[int],
) -> TrainingSupervision:
    validate_split(facts, split)
    if not query_diseases or not query_diseases <= split.train:
        raise ValueError("supervision: query diseases must be nonempty and in training")

    diseases = tuple(sorted(query_diseases))
    drugs = tuple(sorted(facts.drugs))
    positives: dict[int, set[int]] = {disease: set() for disease in diseases}
    excluded: dict[int, set[int]] = {disease: set() for disease in diseases}

    for drug, disease in facts.unambiguous_indications:
        if disease in query_diseases:
            positives[disease].add(drug)

    for pairs in (facts.indication, facts.contraindication, facts.off_label):
        for drug, disease in pairs:
            if disease in query_diseases:
                excluded[disease].add(drug)

    positive_rows: list[tuple[int, ...]] = []
    excluded_rows: list[frozenset[int]] = []

    for disease in diseases:
        positive_drugs = positives[disease]
        excluded_drugs = excluded[disease]

        if not positive_drugs:
            raise ValueError(f"supervision: disease {disease} has no training indication")
        if not excluded_drugs <= facts.drugs:
            raise ValueError(f"supervision: disease {disease} references an unknown drug")
        if len(excluded_drugs) == len(drugs):
            raise ValueError(f"supervision: disease {disease} has no unjudged candidates")

        positive_rows.append(tuple(sorted(positive_drugs)))
        excluded_rows.append(frozenset(excluded_drugs))

    return TrainingSupervision(
        diseases, drugs,
        tuple(positive_rows), tuple(excluded_rows),
    )


def sample_unjudged(
    supervision: TrainingSupervision,
    *,
    count: int,
    generator: random.Random,
) -> tuple[tuple[int, ...], ...]:
    if count <= 0:
        raise ValueError("supervision: sample count must be positive")

    sampled: list[tuple[int, ...]] = []
    for disease, excluded in zip(
        supervision.diseases, supervision.excluded, strict=True,
    ):
        eligible = [drug for drug in supervision.drugs if drug not in excluded]
        if count > len(eligible):
            raise ValueError(f"supervision: disease {disease} has too few unjudged drugs")
        sampled.append(tuple(generator.sample(eligible, count)))

    return tuple(sampled)
