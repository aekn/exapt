from __future__ import annotations

import csv
from array import array
from pathlib import Path
from typing import Iterable, NamedTuple, Sequence

from .primekg import COLUMNS, THERAPY, verify_primekg


Pair = tuple[int, int]  # (drug index, disease index)


class Relation(NamedTuple):
    source_type: str
    name: str
    display: str
    target_type: str


class Edges(NamedTuple):
    source: memoryview
    target: memoryview

    def __len__(self) -> int:
        return len(self.source)


class Graph(NamedTuple):
    node_types: tuple[str, ...]
    relations: tuple[Relation, ...]
    edges: tuple[Edges, ...]

    @property
    def node_count(self) -> int:
        return len(self.node_types)

    @property
    def edge_count(self) -> int:
        return sum(len(edges) for edges in self.edges)


def _edge_views(source: array, target: array) -> Edges:
    return Edges(memoryview(source).toreadonly(), memoryview(target).toreadonly())


def read_graph(rows: Iterable[Sequence[str]]) -> Graph:
    if array("I").itemsize != 4:
        raise RuntimeError("graph: expected 32-bit unsigned array storage")

    node_types: list[str] = []
    buckets: dict[Relation, tuple[array, array]] = {}

    for line, row in enumerate(rows, start=2):
        if len(row) != len(COLUMNS):
            raise ValueError(f"primekg: row {line}: expected {len(COLUMNS)} fields, got {len(row)}")

        (
            name, display,
            source_index, source_id, source_type, _, _,
            target_index, target_id, target_type, _, _,
        ) = row

        try:
            source, target = int(source_index), int(target_index)
        except ValueError as error:
            raise ValueError(f"primekg: row {line}: invalid node index") from error
        if source < 0 or target < 0:
            raise ValueError(f"primekg: row {line}: negative node index")
        if not (name and display and source_id and source_type and target_id and target_type):
            raise ValueError(f"primekg: row {line}: missing required field")
        if source >= 2**32 or target >= 2**32:
            raise ValueError(f"primekg: row {line}: node index exceeds 32-bit storage")
        if name in THERAPY and (source_type, target_type) not in (
            ("drug", "disease"), ("disease", "drug"),
        ):
            raise ValueError(f"primekg: row {line}: invalid therapeutic endpoints")

        needed = max(source, target) + 1
        if needed > len(node_types):
            node_types.extend([""] * (needed - len(node_types)))

        for index, kind in ((source, source_type), (target, target_type)):
            previous = node_types[index]
            if previous and previous != kind:
                raise ValueError(f"primekg: row {line}: inconsistent node type for index {index}")
            node_types[index] = kind

        relation = Relation(source_type, name, display, target_type)
        bucket = buckets.get(relation)
        if bucket is None:
            bucket = (array("I"), array("I"))
            buckets[relation] = bucket

        sources, targets = bucket
        sources.append(source)
        targets.append(target)

    relations = tuple(sorted(buckets))
    return Graph(
        tuple(node_types),
        relations,
        tuple(_edge_views(*buckets[relation]) for relation in relations),
    )


def load_graph(
    root: Path,
    *,
    withheld_diseases: frozenset[int],
    masked_pairs: frozenset[Pair],
) -> Graph:
    path = verify_primekg(root)
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        if tuple(next(reader, ())) != COLUMNS:
            raise ValueError("primekg: unexpected CSV header")
        graph = read_graph(reader)

    return message_graph(
        graph,
        withheld_diseases=withheld_diseases,
        masked_pairs=masked_pairs,
    )


def message_graph(
    graph: Graph,
    *,
    withheld_diseases: frozenset[int],
    masked_pairs: frozenset[Pair],
) -> Graph:
    for disease in withheld_diseases:
        if not (0 <= disease < graph.node_count and graph.node_types[disease] == "disease"):
            raise ValueError(f"graph: invalid withheld disease {disease}")

    for drug, disease in masked_pairs:
        if not (0 <= drug < graph.node_count and graph.node_types[drug] == "drug"):
            raise ValueError(f"graph: invalid masked drug {drug}")
        if not (0 <= disease < graph.node_count and graph.node_types[disease] == "disease"):
            raise ValueError(f"graph: invalid masked disease {disease}")

    visible: list[Edges] = []

    for relation, edges in zip(graph.relations, graph.edges, strict=True):
        if relation.name not in THERAPY:
            visible.append(edges)
            continue

        endpoints = (relation.source_type, relation.target_type)
        if endpoints == ("drug", "disease"):
            drug_first = True
        elif endpoints == ("disease", "drug"):
            drug_first = False
        else:
            raise ValueError(f"graph: invalid therapeutic relation {relation}")

        if not withheld_diseases and not masked_pairs:
            visible.append(edges)
            continue

        sources, targets = array("I"), array("I")
        for source, target in zip(edges.source, edges.target, strict=True):
            drug, disease = (source, target) if drug_first else (target, source)
            if disease in withheld_diseases or (drug, disease) in masked_pairs:
                continue
            sources.append(source)
            targets.append(target)

        visible.append(_edge_views(sources, targets))

    return Graph(graph.node_types, graph.relations, tuple(visible))


def mask_disease_protein_context(
    graph: Graph, diseases: frozenset[int], *, fraction: float, seed: int,
) -> Graph:
    """Remove the same sampled disease-protein pairs in both arc directions."""
    import random

    if not 0 <= fraction <= 1:
        raise ValueError("graph: invalid context mask fraction")
    if any(not (0 <= d < graph.node_count and graph.node_types[d] == "disease") for d in diseases):
        raise ValueError("graph: invalid context disease")

    neighborhoods: dict[int, set[int]] = {d: set() for d in diseases}
    for relation, edges in zip(graph.relations, graph.edges, strict=True):
        if relation.name != "disease_protein":
            continue
        if (relation.source_type, relation.target_type) == ("disease", "gene/protein"):
            orientation = True
        elif (relation.source_type, relation.target_type) == ("gene/protein", "disease"):
            orientation = False
        else:
            raise ValueError("graph: unexpected disease_protein endpoints")
        for x, y in zip(edges.source, edges.target, strict=True):
            disease, protein = (x, y) if orientation else (y, x)
            if disease in neighborhoods:
                neighborhoods[disease].add(protein)

    generator = random.Random(seed)
    removed: set[tuple[int, int]] = set()
    for disease, proteins in sorted(neighborhoods.items()):
        ordered = sorted(proteins)
        removed.update((disease, protein) for protein in generator.sample(
            ordered, int(len(ordered) * fraction),
        ))

    visible: list[Edges] = []
    for relation, edges in zip(graph.relations, graph.edges, strict=True):
        if relation.name != "disease_protein" or not removed:
            visible.append(edges)
            continue
        forward = relation.source_type == "disease"
        sources, targets = array("I"), array("I")
        for source, target in zip(edges.source, edges.target, strict=True):
            disease, protein = (source, target) if forward else (target, source)
            if (disease, protein) not in removed:
                sources.append(source)
                targets.append(target)
        visible.append(_edge_views(sources, targets))
    return Graph(graph.node_types, graph.relations, tuple(visible))
