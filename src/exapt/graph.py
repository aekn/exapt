import csv
from array import array
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch

from .data import THERAPEUTIC_RELATIONS, sha256_file


type Relation = tuple[str, str, str]


@dataclass
class Graph:
    nodes: dict[int, int]
    relations: tuple[Relation, ...]
    adjacency: tuple[torch.Tensor, ...]

    def to(self, device: torch.device) -> "Graph":
        return Graph(
            self.nodes,
            self.relations,
            tuple(matrix.to(device) for matrix in self.adjacency),
        )


def _mean_adjacency(
    targets: torch.Tensor, sources: torch.Tensor, node_count: int
) -> torch.Tensor:
    # A_r[dst, src] = 1 / degree_r(dst)
    # A_r @ H
    # each node gets mean of incoming r neighbors
    indices = torch.stack((targets, sources))
    degree = torch.bincount(targets, minlength=node_count)
    weights = degree[targets].reciprocal()

    return torch.sparse_coo_tensor(
        indices, weights, (node_count, node_count), check_invariants=True
    ).coalesce()


def load_graph(directory: Path) -> Graph:
    if directory.is_file():
        saved = torch.load(directory, map_location="cpu", weights_only=True)
        return Graph(
            {index: index for index in range(saved["nodes"])},
            tuple(tuple(relation) for relation in saved["relations"]),
            tuple(saved["adjacency"]),
        )

    with (directory / "nodes.csv").open(newline="") as stream:
        nodes = {
            int(row["index"]): index
            for index, row in enumerate(csv.DictReader(stream))
        }

    edges: dict[Relation, list[tuple[int, int]]] = defaultdict(list)
    with (directory / "graph.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["relation"] in THERAPEUTIC_RELATIONS:
                raise ValueError("Therapeutic edges are not message edges")

            relation = (row["x_type"], row["relation"], row["y_type"])
            source = nodes[int(row["x_index"])]
            target = nodes[int(row["y_index"])]
            edges[relation].append((target, source))

    relations = tuple(sorted(edges))
    adjacency: list[torch.Tensor] = []

    for relation in relations:
        pairs = torch.tensor(edges[relation], dtype=torch.long).T
        adjacency.append(_mean_adjacency(pairs[0], pairs[1], len(nodes)))

    return Graph(nodes, relations, tuple(adjacency))


def prepare_graph(source: Path, output: Path) -> dict[str, object]:
    edges: dict[Relation, tuple[array, array]] = {}
    nodes: set[int] = set()

    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        relation_col = header.index("relation")
        source_col = header.index("x_index")
        source_type_col = header.index("x_type")
        target_col = header.index("y_index")
        target_type_col = header.index("y_type")

        for row in reader:
            src, dst = int(row[source_col]), int(row[target_col])
            nodes.update((src, dst))

            # remove therapeutic edges from graph
            if row[relation_col] in THERAPEUTIC_RELATIONS:
                continue

            relation = (
                row[source_type_col], row[relation_col], row[target_type_col]
            )
            if relation not in edges:
                edges[relation] = (array("i"), array("i"))

            targets, sources = edges[relation]
            targets.append(dst)
            sources.append(src)

    # embedding rows indexed by raw IDs
    if nodes != set(range(len(nodes))):
        raise ValueError("Expected contiguous source node indices")

    relations = tuple(sorted(edges))
    adjacency: list[torch.Tensor] = []

    for relation in relations:
        targets, sources = edges[relation]
        target_index = torch.tensor(targets, dtype=torch.long)
        source_index = torch.tensor(sources, dtype=torch.long)
        adjacency.append(
            _mean_adjacency(target_index, source_index, len(nodes))
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"nodes": len(nodes), "relations": relations, "adjacency": adjacency},
        output,
    )

    return {
        "source_sha256": sha256_file(source),
        "nodes": len(nodes),
        "directed_edges": sum(len(targets) for targets, _ in edges.values()),
        "unique_edges": sum(matrix._nnz() for matrix in adjacency),
        "relations": len(relations),
        "file": output.name,
        "sha256": sha256_file(output),
    }
