from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Iterator, NamedTuple, Sequence

from .primekg import COLUMNS, THERAPY, verify_primekg

if TYPE_CHECKING:
    from .graph import Relation


Pair = tuple[int, int]  # (drug index, disease index)


class Therapeutics(NamedTuple):
    drugs: frozenset[int]
    diseases: dict[int, tuple[str, str]]  # index -> (source, identifier)
    indication: frozenset[Pair]
    contraindication: frozenset[Pair]
    off_label: frozenset[Pair]

    @property
    def conflicts(self) -> frozenset[Pair]:
        return self.indication & self.contraindication

    @property
    def unambiguous_indications(self) -> frozenset[Pair]:
        return self.indication - self.contraindication


class DiseaseSplit(NamedTuple):
    train: frozenset[int]
    development: frozenset[int]
    test: frozenset[int]

    @property
    def withheld(self) -> frozenset[int]:
        return self.development | self.test


def _indices(x_index: str, y_index: str, line: int) -> tuple[int, int]:
    try:
        x, y = int(x_index), int(y_index)
    except ValueError as error:
        raise ValueError(f"primekg: row {line}: invalid node index") from error
    if x < 0 or y < 0:
        raise ValueError(f"primekg: row {line}: negative node index")
    return x, y


def _therapeutic_pair(
    x: int, y: int, x_type: str, y_type: str, line: int,
) -> Pair:
    if (x_type, y_type) == ("drug", "disease"):
        return x, y
    if (x_type, y_type) == ("disease", "drug"):
        return y, x
    raise ValueError(f"primekg: row {line}: invalid therapeutic endpoints ({x_type}, {y_type})")


def read_therapeutics(rows: Iterable[Sequence[str]]) -> Therapeutics:
    drugs: set[int] = set()
    diseases: dict[int, tuple[str, str]] = {}
    pairs: dict[str, set[Pair]] = {relation: set() for relation in THERAPY}

    for line, row in enumerate(rows, start=2):
        if len(row) != len(COLUMNS):
            raise ValueError(f"primekg: row {line}: expected {len(COLUMNS)} fields, got {len(row)}")

        (
            relation, _,
            x_index, x_id, x_type, _, x_source,
            y_index, y_id, y_type, _, y_source,
        ) = row
        x, y = _indices(x_index, y_index, line)
        if not (relation and x_id and x_type and y_id and y_type):
            raise ValueError(f"primekg: row {line}: missing required field")

        for index, kind, source, identifier in (
            (x, x_type, x_source, x_id),
            (y, y_type, y_source, y_id),
        ):
            if kind == "drug":
                drugs.add(index)
            elif kind == "disease":
                identity = (source, identifier)
                previous = diseases.setdefault(index, identity)
                if previous != identity:
                    raise ValueError(f"primekg: row {line}: inconsistent disease identity")

        if relation in pairs:
            pairs[relation].add(_therapeutic_pair(x, y, x_type, y_type, line))

    if drugs & diseases.keys():
        raise ValueError("primekg: drug and disease indices overlap")

    return Therapeutics(
        frozenset(drugs), diseases,
        frozenset(pairs["indication"]),
        frozenset(pairs["contraindication"]),
        frozenset(pairs["off-label use"]),
    )


def load_therapeutics(root: Path) -> Therapeutics:
    path = verify_primekg(root)
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        if tuple(next(reader, ())) != COLUMNS:
            raise ValueError("primekg: unexpected CSV header")
        return read_therapeutics(reader)


def _indication_stratum(count: int) -> int:
    if count <= 2:
        return count
    return 3 if count < 10 else 4


def _hash_order(diseases: list[int], prefix: str) -> list[int]:
    return sorted(
        diseases,
        key=lambda disease: hashlib.sha256(
            f"{prefix}{disease}".encode("ascii")
        ).digest(),
    )


def split_diseases(
    facts: Therapeutics,
    *,
    seed: int,
    development_fraction: float,
    test_fraction: float,
) -> DiseaseSplit:
    if not (0 < development_fraction < 1 and 0 < test_fraction < 1):
        raise ValueError("benchmark: development and test fractions must be between 0 and 1")
    if development_fraction + test_fraction >= 1:
        raise ValueError("benchmark: fractions leave no training population")

    counts = Counter(disease for _, disease in facts.unambiguous_indications)
    strata: dict[int, list[int]] = defaultdict(list)
    for disease, count in counts.items():
        strata[_indication_stratum(count)].append(disease)

    development: set[int] = set()
    test: set[int] = set()

    for stratum, diseases in sorted(strata.items()):
        ordered = _hash_order(diseases, f"{seed}:")
        n_dev = round(len(ordered) * development_fraction)
        n_test = round(len(ordered) * test_fraction)
        if n_dev + n_test >= len(ordered):
            raise ValueError(f"benchmark: stratum {stratum} has no training diseases")

        development.update(ordered[:n_dev])
        test.update(ordered[n_dev:n_dev + n_test])

    if not development or not test:
        raise ValueError("benchmark: development and test populations must be nonempty")

    return DiseaseSplit(
        frozenset(facts.diseases.keys() - development - test),
        frozenset(development),
        frozenset(test),
    )


def indication_partition(
    facts: Therapeutics, diseases: frozenset[int],
) -> frozenset[Pair]:
    return frozenset(
        (drug, disease)
        for drug, disease in facts.unambiguous_indications
        if disease in diseases
    )


def validate_split(facts: Therapeutics, split: DiseaseSplit) -> None:
    train, development, test = split
    if train & development or train & test or development & test:
        raise ValueError("benchmark: disease partitions overlap")
    if train | development | test != facts.diseases.keys():
        raise ValueError("benchmark: disease partitions do not cover source diseases")
    if not development or not test:
        raise ValueError("benchmark: evaluation partitions must be nonempty")

    eligible = {disease for _, disease in facts.unambiguous_indications}
    if not development <= eligible or not test <= eligible:
        raise ValueError("benchmark: evaluation disease has no unambiguous indication")
    if not indication_partition(facts, train):
        raise ValueError("benchmark: training has no unambiguous indications")
    if any(drug not in facts.drugs for drug, _ in facts.indication):
        raise ValueError("benchmark: indication references an unknown drug")


def training_folds(
    facts: Therapeutics,
    split: DiseaseSplit,
    *,
    folds: int,
    seed: int,
) -> tuple[frozenset[int], ...]:
    validate_split(facts, split)
    if folds < 2:
        raise ValueError("benchmark: at least two training folds required")

    counts = Counter(
        disease for _, disease in facts.unambiguous_indications
        if disease in split.train
    )
    if folds > len(counts):
        raise ValueError("benchmark: more folds than supervised training diseases")

    strata: dict[int, list[int]] = defaultdict(list)
    for disease, count in counts.items():
        strata[_indication_stratum(count)].append(disease)

    partitions: list[set[int]] = [set() for _ in range(folds)]

    # carry the round robin position across strata to balance fold sizes
    offset = 0

    for _, diseases in sorted(strata.items()):
        ordered = _hash_order(diseases, f"{seed}:train-fold:")
        for i, disease in enumerate(ordered):
            partitions[(offset + i) % folds].add(disease)
        offset += len(ordered)

    if any(not partition for partition in partitions):
        raise ValueError("benchmark: empty training fold")

    return tuple(frozenset(partition) for partition in partitions)


def benchmark_record(
    split: DiseaseSplit,
    folds: tuple[frozenset[int], ...],
    relations: tuple[Relation, ...],
    drugs: frozenset[int],
    source_sha256: str,
) -> dict[str, Any]:
    record = {
        "source_sha256": source_sha256,
        "train": sorted(split.train),
        "development": sorted(split.development),
        "test": sorted(split.test),
        "folds": [sorted(fold) for fold in folds],
        "drugs": sorted(drugs),
        "relations": [list(relation) for relation in relations],
    }
    payload = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    record["sha256"] = hashlib.sha256(payload).hexdigest()
    return record


def visible_rows(
    rows: Iterable[Sequence[str]],
    *,
    withheld_diseases: frozenset[int],
    masked_pairs: frozenset[Pair],
) -> Iterator[Sequence[str]]:
    for line, row in enumerate(rows, start=2):
        if len(row) != len(COLUMNS):
            raise ValueError(f"primekg: row {line}: expected {len(COLUMNS)} fields, got {len(row)}")

        if row[0] in THERAPY:
            x, y = _indices(row[2], row[7], line)
            pair = _therapeutic_pair(x, y, row[4], row[9], line)
            if pair[1] in withheld_diseases or pair in masked_pairs:
                continue

        yield row
