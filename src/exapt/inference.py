from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Sequence

import torch
from torch import Tensor

from .compgcn import CompGCNEncoder, compgcn_adjacency
from .distmult import DistMultDecoder
from .graph import Graph
from .model import IndicationModel, NodeEmbeddings
from .rgcn import RGCNEncoder, rgcn_adjacency


def load_selected(
    path: Path, graph: Graph, *, device: torch.device | str
) -> tuple[IndicationModel, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint["metadata"]["config"]
    width = config["width"]
    embeddings = NodeEmbeddings(
        graph.node_count, width, trainable=config["trainable_features"],
    )
    encoder = (
        CompGCNEncoder(graph.relations, width, width, width)
        if config["encoder"] == "compgcn"
        else RGCNEncoder(graph.relations, width, width, width)
    )
    model = IndicationModel(embeddings, encoder, DistMultDecoder(width)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint["metadata"]


def score_catalog(
    model: IndicationModel,
    graph: Graph,
    *,
    diseases: Sequence[int],
    drugs: Sequence[int],
    encoder: Literal["rgcn", "compgcn"],
    normalization: Literal["relation", "total", "directional"],
    layout: Literal["coo", "csr"],
    device: torch.device | str,
) -> list[list[float]]:
    if encoder == "compgcn":
        adjacency = compgcn_adjacency(graph, device=device, layout=layout)
    elif encoder == "rgcn" and normalization in ("total", "relation"):
        adjacency = rgcn_adjacency(graph, normalization=normalization, device=device, layout=layout)
    else:
        raise ValueError("inference: invalid graph configuration")
    with torch.inference_mode():
        scores: Tensor = model(
            adjacency,
            torch.tensor(diseases, dtype=torch.long, device=device),
            torch.tensor(drugs, dtype=torch.long, device=device),
        )
    return scores.cpu().tolist()
