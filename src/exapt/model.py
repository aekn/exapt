import torch
from torch import nn
from torch.nn import functional as F

from .graph import Graph
from .prototypes import Prototypes


class RelationalLayer(nn.Module):
    def __init__(self, width: int, relations: int):
        super().__init__()
        self.self_linear = nn.Linear(width, width)
        self.rel_linear = nn.ModuleList(
            nn.Linear(width, width, bias=False) for _ in range(relations)
        )

    def forward(
        self, features: torch.Tensor, adjacency: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        # H' = H @ W_self.T + b + sum_r (A_r @ H) @ W_r.T
        updated = self.self_linear(features)

        for matrix, linear in zip(adjacency, self.rel_linear):
            # A_r[dst, src] = 1 / deg_r(dst)
            messages = torch.sparse.mm(matrix, features)
            updated = updated + linear(messages)

        return updated


class RelationalEncoder(nn.Module):
    def __init__(self, nodes: int, relations: int, width: int = 32):
        super().__init__()
        self.embeddings = nn.Embedding(nodes, width)
        nn.init.normal_(self.embeddings.weight, std=width ** -0.5)

        self.first = RelationalLayer(width, relations)
        self.second = RelationalLayer(width, relations)

    def forward(self, graph: Graph) -> torch.Tensor:
        hidden = self.first(self.embeddings.weight, graph.adjacency)
        hidden = F.relu(hidden)
        return self.second(hidden, graph.adjacency)


class DistMultDecoder(nn.Module):
    def __init__(self, width: int, relations: int = 3):
        super().__init__()
        # all heads initially reduce to the dot product
        self.weight = nn.Parameter(torch.ones(relations, width))

    def score(
        self, drugs: torch.Tensor, diseases: torch.Tensor,
        relation: int = 0,
    ) -> torch.Tensor:
        # vector inputs; s_r(a, d) = h_a @ diag(w_r) @ h_d
        return (drugs * self.weight[relation] * diseases).sum(dim=-1)

    def forward(
        self, embeddings: torch.Tensor, drugs: torch.Tensor,
        diseases: torch.Tensor, relation: int = 0,
    ) -> torch.Tensor:
        return self.score(embeddings[drugs], embeddings[diseases], relation)


class RelationalGNN(nn.Module):
    def __init__(self, nodes: int, relations: int, width: int = 32):
        super().__init__()
        self.encoder = RelationalEncoder(nodes, relations, width)
        self.decoder = DistMultDecoder(width)

    @property
    def indication(self) -> torch.Tensor:
        return self.decoder.weight[0]

    def forward(self, graph: Graph) -> torch.Tensor:
        return self.encoder(graph)

    def disease_vectors(
        self, embeddings: torch.Tensor, diseases: torch.Tensor
    ) -> torch.Tensor:
        return embeddings[diseases]

    def score(
        self, embeddings: torch.Tensor, drugs: torch.Tensor,
        diseases: torch.Tensor, relation: int = 0,
    ) -> torch.Tensor:
        vectors = self.disease_vectors(embeddings, diseases)
        return self.decoder.score(embeddings[drugs], vectors, relation)


class ExaptGNN(nn.Module):
    # n = nodes, q = supported diseases, k = donors, f = embedding width
    index: torch.Tensor     # (n,) prototype row, -1 if unsupported
    diseases: torch.Tensor  # (q,)
    donors: torch.Tensor    # (q, k)
    weights: torch.Tensor   # (q, k)
    evidence: torch.Tensor  # (q, 3) similarity, agreement, availability

    def __init__(
        self, base: RelationalGNN, prototypes: Prototypes, width: int
    ):
        super().__init__()
        self.base = base
        self.gate = nn.Linear(2 * width + 3, 1)
        # zero weights give alpha = sigmoid(b) = 0.2 initially
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -1.38629436112)

        # replaceable donor tables remain registered buffers
        self.register_buffer("index", torch.empty(0, dtype=torch.long))
        self.register_buffer("diseases", torch.empty(0, dtype=torch.long))
        self.register_buffer("donors", torch.empty((0, 0), dtype=torch.long))
        self.register_buffer("weights", torch.empty((0, 0)))
        self.register_buffer("evidence", torch.empty((0, 3)))
        self.set_prototypes(prototypes)

    def set_prototypes(self, prototypes: Prototypes) -> None:
        device = self.base.encoder.embeddings.weight.device
        nodes = self.base.encoder.embeddings.num_embeddings

        index = torch.full((nodes,), -1, dtype=torch.long, device=device)
        diseases = prototypes.diseases.to(device)
        index[diseases] = torch.arange(len(diseases), device=device)

        self.index = index
        self.diseases = diseases
        self.donors = prototypes.donors.to(device)
        self.weights = prototypes.weights.to(device)
        self.evidence = prototypes.evidence.to(device)

    @property
    def indication(self) -> torch.Tensor:
        return self.base.indication

    def forward(self, graph: Graph) -> torch.Tensor:
        # augmentation is applied to requested diseases during scoring
        return self.base(graph)

    def disease_vectors(
        self, embeddings: torch.Tensor, diseases: torch.Tensor
    ) -> torch.Tensor:
        if diseases.numel() == 0 or self.diseases.numel() == 0:
            return embeddings[diseases]

        # aggregate donors once per distinct disease in this scoring call
        unique, inverse = diseases.reshape(-1).unique(return_inverse=True)
        rows = self.index[unique]
        safe_rows = rows.clamp_min(0)

        # unsupported queries gather row 0, then receive alpha=0
        query = embeddings[unique]                    # (u, f)
        donors = embeddings[self.donors[safe_rows]]   # (u, k, f)
        weights = self.weights[safe_rows]             # (u, k)
        prototype = (donors * weights.unsqueeze(-1)).sum(dim=1)
        evidence = self.evidence[safe_rows]           # (u, 3)

        # alpha_q = sigmoid(W_g [h_q; p_q; e_q] + b_g)
        gate_input = torch.cat((query, prototype, evidence), dim=-1)
        alpha = torch.sigmoid(self.gate(gate_input))
        alpha = alpha * (rows >= 0).unsqueeze(-1)

        # h'_q = h_q + alpha_q (p_q - h_q)
        augmented = query + alpha * (prototype - query)
        return augmented[inverse].reshape(*diseases.shape, -1)

    def score(
        self, embeddings: torch.Tensor, drugs: torch.Tensor,
        diseases: torch.Tensor, relation: int = 0,
    ) -> torch.Tensor:
        vectors = self.disease_vectors(embeddings, diseases)
        return self.base.decoder.score(embeddings[drugs], vectors, relation)
