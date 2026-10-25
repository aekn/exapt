from __future__ import annotations

import torch
from torch import Tensor, nn


class DistMultDecoder(nn.Module):

    def __init__(self, features: int) -> None:
        super().__init__()
        if features <= 0:
            raise ValueError("model: feature dimension must be positive")

        self.relation = nn.Parameter(torch.empty(features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.relation)

    def forward(
        self, embeddings: Tensor, diseases: Tensor, drugs: Tensor,
    ) -> Tensor:
        if embeddings.ndim != 2 or embeddings.shape[1] != self.relation.numel():
            raise ValueError("model: embedding dimension does not match decoder")
        if diseases.ndim != 1 or drugs.ndim != 1:
            raise ValueError("model: disease and drug indices must be vectors")
        if diseases.dtype != torch.long or drugs.dtype != torch.long:
            raise ValueError("model: disease and drug indices must be int64")

        # scores[q, a] = sum_j h[d_q, j] * r[j] * h[a, j]
        queries = embeddings.index_select(0, diseases) * self.relation
        candidates = embeddings.index_select(0, drugs)
        return queries @ candidates.T
