from __future__ import annotations

from typing import Literal, NamedTuple

import torch
from torch import Tensor, nn

from .graph import Graph, Relation


class RelationalAdjacency(NamedTuple):
    relations: tuple[Relation, ...]
    node_count: int
    normalization: Literal["relation", "total"]
    matrices: tuple[Tensor | None, ...]


def _indices(buffer: memoryview) -> Tensor:
    return torch.frombuffer(buffer.obj, dtype=torch.uint32).to(torch.int64)


def rgcn_adjacency(
    graph: Graph,
    *,
    normalization: Literal["relation", "total"],
    device: torch.device | str = "cpu",
    layout: Literal["coo", "csr"] = "csr",
) -> RelationalAdjacency:
    if normalization not in ("relation", "total"):
        raise ValueError(f"rgcn: invalid normalization {normalization!r}")
    if layout not in ("coo", "csr"):
        raise ValueError(f"rgcn: invalid sparse layout {layout!r}")
    if len(graph.relations) != len(graph.edges):
        raise ValueError("rgcn: relation and edge counts differ")

    n = graph.node_count
    total_degree = None
    if normalization == "total":
        total_degree = torch.zeros(n, dtype=torch.int64)
        for edges in graph.edges:
            if len(edges):
                total_degree += torch.bincount(_indices(edges.target), minlength=n)

    matrices: list[Tensor | None] = []
    for edges in graph.edges:
        if not len(edges):
            matrices.append(None)
            continue

        source = _indices(edges.source)
        target = _indices(edges.target)
        degree = total_degree
        if degree is None:
            degree = torch.bincount(target, minlength=n)

        values = 1.0 / degree[target].to(torch.float32)
        indices = torch.stack((target, source))
        matrix = torch.sparse_coo_tensor(
            indices, values, (n, n), check_invariants=True,
        ).coalesce()

        if layout == "csr":
            matrix = matrix.to_sparse_csr()
        matrices.append(matrix.to(device))

    return RelationalAdjacency(graph.relations, n, normalization, tuple(matrices))


class RGCNLayer(nn.Module):
    def __init__(
        self, relations: tuple[Relation, ...], in_features: int, out_features: int,
    ) -> None:
        super().__init__()
        if not relations:
            raise ValueError("rgcn: at least one relation required")
        if in_features <= 0 or out_features <= 0:
            raise ValueError("rgcn: feature dimensions must be positive")

        self.relations = relations
        self.weight = nn.Parameter(torch.empty(len(relations), in_features, out_features))
        self.root = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # each relation is a (in_features, out_features) matrix
        for weight in self.weight:
            nn.init.xavier_uniform_(weight)
        nn.init.xavier_uniform_(self.root)
        nn.init.zeros_(self.bias)

    def forward(self, features: Tensor, graph: RelationalAdjacency) -> Tensor:
        if features.ndim != 2 or features.shape != (graph.node_count, self.root.shape[0]):
            raise ValueError("rgcn: node feature shape does not match graph")
        if graph.relations != self.relations:
            raise ValueError("rgcn: relation vocabulary does not match graph")

        output = features @ self.root
        for matrix, weight in zip(graph.matrices, self.weight, strict=True):
            if matrix is not None:
                output = output + torch.sparse.mm(matrix, features @ weight)
        return output + self.bias


class RGCNEncoder(nn.Module):
    def __init__(
        self, relations: tuple[Relation, ...], in_features: int,
        hidden_features: int, out_features: int,
    ) -> None:
        super().__init__()
        self.first = RGCNLayer(relations, in_features, hidden_features)
        self.second = RGCNLayer(relations, hidden_features, out_features)

    def forward(self, features: Tensor, graph: RelationalAdjacency) -> Tensor:
        hidden = torch.relu(self.first(features, graph))
        return self.second(hidden, graph)
