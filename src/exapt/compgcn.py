from __future__ import annotations

from typing import Literal, NamedTuple

import torch
from torch import Tensor, nn

from .graph import Graph, Relation


Direction = Literal["in", "out", "symmetric"]


class CompositionalAdjacency(NamedTuple):
    relations: tuple[Relation, ...]
    node_count: int
    directions: tuple[Direction, ...]
    matrices: tuple[Tensor | None, ...]


def _indices(values: memoryview) -> Tensor:
    return torch.frombuffer(values.obj, dtype=torch.uint32).to(torch.int64)


def _direction(relation: Relation) -> Direction:
    if relation.source_type == relation.target_type:
        return "symmetric"
    return "in" if relation.source_type < relation.target_type else "out"


def compgcn_adjacency(
    graph: Graph,
    *,
    device: torch.device | str = "cpu",
    layout: Literal["coo", "csr"] = "csr",
) -> CompositionalAdjacency:
    if layout not in ("coo", "csr"):
        raise ValueError(f"compgcn: invalid sparse layout {layout!r}")
    if len(graph.relations) != len(graph.edges):
        raise ValueError("compgcn: relation and edge counts differ")

    node_count = graph.node_count
    directions: tuple[Direction, ...] = tuple(
        _direction(relation) for relation in graph.relations
    )
    in_degree = {
        direction: torch.zeros(node_count, dtype=torch.int64)
        for direction in ("in", "out", "symmetric")
    }
    out_degree = {
        direction: torch.zeros(node_count, dtype=torch.int64)
        for direction in ("in", "out", "symmetric")
    }

    for direction, edges in zip(directions, graph.edges, strict=True):
        if not len(edges):
            continue
        out_degree[direction] += torch.bincount(_indices(edges.source), minlength=node_count)
        in_degree[direction] += torch.bincount(_indices(edges.target), minlength=node_count)

    matrices: list[Tensor | None] = []
    for direction, edges in zip(directions, graph.edges, strict=True):
        if not len(edges):
            matrices.append(None)
            continue

        source = _indices(edges.source)
        target = _indices(edges.target)

        # A_r[target, source] / sqrt(deg_in[target] * deg_out[source])
        degrees = (
            in_degree[direction][target].to(torch.float32)
            * out_degree[direction][source].to(torch.float32)
        )
        values = degrees.rsqrt()
        matrix = torch.sparse_coo_tensor(
            torch.stack((target, source)), values,
            (node_count, node_count), check_invariants=True,
        ).coalesce()

        if layout == "csr":
            matrix = matrix.to_sparse_csr()
        matrices.append(matrix.to(device))

    return CompositionalAdjacency(graph.relations, node_count, directions, tuple(matrices))


class CompGCNLayer(nn.Module):
    def __init__(
        self,
        relations: tuple[Relation, ...],
        in_features: int,
        out_features: int,
        *,
        update_relations: bool = True,
    ) -> None:
        super().__init__()
        if not relations or in_features <= 0 or out_features <= 0:
            raise ValueError("compgcn: invalid relation vocabulary or dimensions")

        self.relations = relations
        self.directions = tuple(_direction(relation) for relation in relations)

        self.incoming = nn.Parameter(torch.empty(in_features, out_features))
        self.outgoing = nn.Parameter(torch.empty(in_features, out_features))
        self.root = nn.Parameter(torch.empty(in_features, out_features))
        self.relation_weight = (
            nn.Parameter(torch.empty(in_features, out_features)) if update_relations else None
        )
        self.loop_relation = nn.Parameter(torch.empty(in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.normalization = nn.BatchNorm1d(out_features)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for weight in (self.incoming, self.outgoing, self.root):
            nn.init.xavier_uniform_(weight)
        if self.relation_weight is not None:
            nn.init.xavier_uniform_(self.relation_weight)
        nn.init.normal_(self.loop_relation, std=self.loop_relation.numel() ** -0.5)
        nn.init.zeros_(self.bias)
        self.normalization.reset_parameters()

    def forward(
        self, features: Tensor, relations: Tensor, graph: CompositionalAdjacency,
    ) -> tuple[Tensor, Tensor | None]:
        if features.shape != (graph.node_count, self.root.shape[0]):
            raise ValueError("compgcn: node feature shape does not match graph")
        if relations.shape != (len(self.relations), self.root.shape[0]):
            raise ValueError("compgcn: relation embedding shape does not match graph")
        if graph.relations != self.relations:
            raise ValueError("compgcn: relation vocabulary does not match graph")
        if graph.directions != self.directions:
            raise ValueError("compgcn: graph direction mapping does not match relations")
        if len(graph.matrices) != len(self.relations):
            raise ValueError("compgcn: adjacency count does not match relations")

        # phi(h, r) = h * r; project once per direction after aggregation
        messages: dict[Direction, Tensor] = {}
        for index, (direction, matrix) in enumerate(
            zip(graph.directions, graph.matrices, strict=True)
        ):
            if matrix is None:
                continue

            composed = torch.sparse.mm(matrix, features) * relations[index]
            if direction in messages:
                messages[direction] = messages[direction] + composed
            else:
                messages[direction] = composed

        output = (features * self.loop_relation) @ self.root
        weights = {
            "in": self.incoming,
            "out": self.outgoing,
            "symmetric": (self.incoming + self.outgoing) * 0.5,
        }
        for direction, total in messages.items():
            output = output + total @ weights[direction]

        output = self.normalization(output / 3.0 + self.bias)
        transformed = relations @ self.relation_weight if self.relation_weight is not None else None
        return output, transformed


class CompGCNEncoder(nn.Module):
    def __init__(
        self,
        relations: tuple[Relation, ...],
        in_features: int,
        hidden_features: int,
        out_features: int,
    ) -> None:
        super().__init__()
        if not relations or min(in_features, hidden_features, out_features) <= 0:
            raise ValueError("compgcn: invalid relation vocabulary or dimensions")

        self.relations = relations
        self.relation_embeddings = nn.Parameter(torch.empty(len(relations), in_features))
        self.first = CompGCNLayer(relations, in_features, hidden_features)
        self.second = CompGCNLayer(
            relations, hidden_features, out_features, update_relations=False,
        )
        nn.init.normal_(
            self.relation_embeddings, std=self.relation_embeddings.shape[1] ** -0.5,
        )

    def reset_parameters(self) -> None:
        nn.init.normal_(
            self.relation_embeddings, std=self.relation_embeddings.shape[1] ** -0.5,
        )
        self.first.reset_parameters()
        self.second.reset_parameters()

    def forward(self, features: Tensor, graph: CompositionalAdjacency) -> Tensor:
        hidden, relations = self.first(features, self.relation_embeddings, graph)
        hidden = torch.tanh(hidden)

        assert relations is not None
        encoded, _ = self.second(hidden, relations, graph)
        return torch.tanh(encoded)
