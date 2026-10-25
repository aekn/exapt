from __future__ import annotations

import torch
from torch import Tensor, nn

from .compgcn import CompositionalAdjacency
from .distmult import DistMultDecoder
from .rgcn import RelationalAdjacency


class NodeEmbeddings(nn.Module):
    def __init__(
        self,
        node_count: int,
        features: int,
        *,
        trainable: bool,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        if node_count <= 0 or features <= 0:
            raise ValueError("model: embedding dimensions must be positive")

        self.weight = nn.Parameter(
            torch.empty(node_count, features), requires_grad=trainable,
        )
        self.reset_parameters(generator=generator)

    def reset_parameters(self, *, generator: torch.Generator | None = None) -> None:
        # E[||h_i||^2] = 1 for h_ij ~ N(0, 1 / features)
        nn.init.normal_(
            self.weight, std=self.weight.shape[1] ** -0.5, generator=generator,
        )

    def forward(self) -> Tensor:
        return self.weight


class IndicationModel(nn.Module):
    def __init__(
        self,
        embeddings: NodeEmbeddings,
        encoder: nn.Module,
        decoder: DistMultDecoder,
    ) -> None:
        super().__init__()
        self.embeddings = embeddings
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self,
        graph: RelationalAdjacency | CompositionalAdjacency,
        diseases: Tensor,
        drugs: Tensor,
    ) -> Tensor:
        features = self.embeddings()
        encoded = self.encoder(features, graph)
        return self.decoder(encoded, diseases, drugs)
