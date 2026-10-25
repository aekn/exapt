from __future__ import annotations

import hashlib
import heapq
import json
import math
import os
import re
import platform
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal, NamedTuple

import torch
from torch import Tensor

from .benchmark import (
    DiseaseSplit, Therapeutics, benchmark_record, indication_partition, split_diseases,
    training_folds,
)
from .compgcn import CompositionalAdjacency, CompGCNEncoder, compgcn_adjacency
from .distmult import DistMultDecoder
from .evaluation import RankingResult, evaluate_ranking
from .graph import Graph, Relation, message_graph
from .model import IndicationModel, NodeEmbeddings
from .objective import sampled_bce_loss, sampled_bpr_loss, sampled_lsep_loss
from .rgcn import RGCNEncoder, RelationalAdjacency, rgcn_adjacency
from .supervision import TrainingSupervision, sample_unjudged, training_supervision


def code_fingerprint() -> str:
    root = Path(__file__).resolve().parent
    sources = (
        "benchmark.py", "primekg.py", "graph.py", "supervision.py",
        "rgcn.py", "compgcn.py", "model.py", "distmult.py",
        "objective.py", "evaluation.py", "train.py",
    )
    digest = hashlib.sha256()
    for name in sources:
        payload = (root / name).read_bytes()
        digest.update(name.encode("ascii") + b"\0")
        digest.update(len(payload).to_bytes(8, "big") + payload)
    return digest.hexdigest()


@dataclass(frozen=True)
class FitConfig:
    seed: int
    benchmark_seed: int
    epochs: int
    folds: int
    width: int
    negatives: int
    negative_weight: float
    learning_rate: float
    weight_decay: float
    development_fraction: float
    test_fraction: float
    normalization: Literal["relation", "total", "directional"]
    layout: Literal["coo", "csr"]
    trainable_features: bool
    k: int = 20
    encoder: Literal["rgcn", "compgcn"] = "rgcn"
    objective: Literal["bce", "bpr", "lsep"] = "bce"

    def validate(self) -> None:
        if self.epochs < 1 or self.folds < 2 or self.width < 1 or self.negatives < 1:
            raise ValueError("train: epochs, folds, width, and negatives must be positive")
        if self.objective not in ("bce", "bpr", "lsep"):
            raise ValueError("train: invalid objective")
        if self.objective != "bce" and self.negative_weight != 1.0:
            raise ValueError("train: negative weight applies only to BCE")
        if self.encoder not in ("rgcn", "compgcn"):
            raise ValueError("train: invalid encoder")
        if self.normalization not in ("relation", "total", "directional"):
            raise ValueError("train: invalid normalization")
        if self.encoder == "compgcn" and self.normalization != "directional":
            raise ValueError("train: CompGCN requires directional normalization")
        if self.encoder == "rgcn" and self.normalization == "directional":
            raise ValueError("train: R-GCN requires relation or total normalization")
        if self.layout not in ("coo", "csr"):
            raise ValueError("train: invalid sparse layout")
        if self.k < 1:
            raise ValueError("train: k must be positive")
        if not (0 < self.development_fraction < 1 and 0 < self.test_fraction < 1):
            raise ValueError("train: split fractions must be between zero and one")
        if self.development_fraction + self.test_fraction >= 1:
            raise ValueError("train: split fractions leave no training population")
        if (
            not math.isfinite(self.learning_rate) or self.learning_rate <= 0
            or not math.isfinite(self.weight_decay) or self.weight_decay < 0
            or not math.isfinite(self.negative_weight) or self.negative_weight < 0
        ):
            raise ValueError("train: invalid optimization parameter")


class _FoldContext(NamedTuple):
    supervision: TrainingSupervision
    adjacency: RelationalAdjacency | CompositionalAdjacency
    diseases: Tensor
    positive_rows: Tensor
    positive_columns: Tensor


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _save_checkpoint(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _write_history(path: Path, history: list[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in history:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _selected_path(output: Path, epoch: int) -> Path:
    return output / "selected" / f"e{epoch:04d}.pt"


def _indices(
    supervision: TrainingSupervision,
    drug_columns: dict[int, int],
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    rows = [row for row, positives in enumerate(supervision.positives) for _ in positives]
    columns = [drug_columns[drug] for positives in supervision.positives for drug in positives]

    return (
        torch.tensor(rows, dtype=torch.long, device=device),
        torch.tensor(columns, dtype=torch.long, device=device),
    )


def _sample_columns(
    supervision: TrainingSupervision,
    drug_columns: dict[int, int],
    generator: random.Random,
    count: int,
    device: torch.device,
) -> Tensor:
    samples = sample_unjudged(supervision, count=count, generator=generator)
    return torch.tensor(
        [[drug_columns[drug] for drug in row] for row in samples],
        dtype=torch.long, device=device,
    )


def _sampled_scores(
    model: IndicationModel,
    context: _FoldContext,
    sampled: Tensor,
    drugs: Tensor,
) -> tuple[Tensor, Tensor]:
    encoded = model.encoder(model.embeddings(), context.adjacency)
    queries = encoded[context.diseases] * model.decoder.relation

    positive_queries = queries[context.positive_rows]
    positive_drugs = encoded[drugs[context.positive_columns]]
    positive_scores = (positive_queries * positive_drugs).sum(dim=-1)

    unjudged_drugs = encoded[drugs[sampled]]
    unjudged_scores = (queries[:, None, :] * unjudged_drugs).sum(dim=-1)
    return positive_scores, unjudged_scores


def _development(
    model: IndicationModel,
    adjacency: RelationalAdjacency | CompositionalAdjacency,
    facts: Therapeutics,
    split: DiseaseSplit,
    *,
    drugs: tuple[int, ...],
    k: int,
    device: torch.device,
) -> tuple[RankingResult, list[list[float]]]:
    diseases = tuple(sorted(split.development))
    disease_ids = torch.tensor(diseases, dtype=torch.long, device=device)
    drug_ids = torch.tensor(drugs, dtype=torch.long, device=device)

    model.eval()
    with torch.inference_mode():
        scores = model(adjacency, disease_ids, drug_ids).cpu().tolist()

    result = evaluate_ranking(
        scores,
        diseases=diseases,
        drugs=drugs,
        positives=indication_partition(facts, split.development),
        k=k,
    )
    return result, scores


def _development_record(
    result: RankingResult, scores: list[list[float]], drugs: tuple[int, ...],
) -> dict[str, Any]:
    top = min(result.k, len(drugs))
    records: list[dict[str, Any]] = []

    for row, values in zip(result.per_disease, scores, strict=True):
        ordered = heapq.nsmallest(
            top, range(len(drugs)), key=lambda index: (-values[index], drugs[index]),
        )
        records.append({
            "disease": row.disease,
            "positives": row.positives,
            "recall": row.recall,
            "ndcg": row.ndcg,
            "reciprocal_rank": row.reciprocal_rank,
            "top_drugs": [drugs[index] for index in ordered],
            "top_scores": [values[index] for index in ordered],
        })

    return {
        "k": result.k,
        "recall": result.recall,
        "ndcg": result.ndcg,
        "reciprocal_rank": result.reciprocal_rank,
        "per_disease": records,
    }


def fit(
    facts: Therapeutics,
    source_graph: Graph,
    *,
    config: FitConfig,
    source_sha256: str,
    output: Path,
    device: torch.device | str,
    code_sha256: str | None = None,
    git_sha: str | None = None,
    dataset_revision: str | None = None,
    purpose: str = "",
    resume: bool = False,
    on_epoch: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Train disease-masked folds; select by development Recall@k, then nDCG@k."""
    config.validate()
    device = torch.device(device)
    if len(source_sha256) != 64 or any(c not in "0123456789abcdef" for c in source_sha256):
        raise ValueError("train: expected source SHA-256")
    if code_sha256 is not None and (
        len(code_sha256) != 64 or any(c not in "0123456789abcdef" for c in code_sha256)
    ):
        raise ValueError("train: expected code archive SHA-256")
    for name, value, width in (("git", git_sha, 40), ("dataset", dataset_revision, 40)):
        if value is not None and re.fullmatch(rf"[0-9a-f]{{{width}}}", value) is None:
            raise ValueError(f"train: invalid {name} commit SHA")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("train: CUDA unavailable")

    split = split_diseases(
        facts, seed=config.benchmark_seed,
        development_fraction=config.development_fraction,
        test_fraction=config.test_fraction,
    )
    folds = training_folds(facts, split, folds=config.folds, seed=config.benchmark_seed)
    benchmark = benchmark_record(
        split, folds, source_graph.relations, facts.drugs, source_sha256,
    )
    scientific_config = asdict(config)
    del scientific_config["epochs"]
    metadata = {
        "config": scientific_config,
        "score_computation": "indexed-distmult-v1",
        "source_sha256": source_sha256,
        "code_sha256": code_sha256,
        "git_sha": git_sha,
        "dataset_revision": dataset_revision,
        "purpose": purpose,
        "benchmark_sha256": benchmark["sha256"],
        "training_protocol": f"disease-masked-sampled-{config.objective}-v1",
        "node_count": source_graph.node_count,
        "edge_count": source_graph.edge_count,
        "drug_count": len(facts.drugs),
        "training_diseases": len(split.train),
        "development_diseases": len(split.development),
        "test_diseases": len(split.test),
        "training_indications": len(indication_partition(facts, split.train)),
        "development_indications": len(indication_partition(facts, split.development)),
        "fold_diseases": [len(fold) for fold in folds],
    }

    metadata["sha256"] = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    output = Path(output)
    if resume:
        if not (output / "resume.pt").is_file():
            raise ValueError("train: resume checkpoint does not exist")
        if json.loads((output / "config.json").read_text()) != metadata:
            raise ValueError("train: configuration or source does not match checkpoint")
        if json.loads((output / "benchmark.json").read_text()) != benchmark:
            raise ValueError("train: benchmark partitions or relations have changed")
        state = torch.load(output / "resume.pt", map_location="cpu", weights_only=True)
        if state["metadata"] != metadata:
            raise ValueError("train: checkpoint identity mismatch")
    else:
        output.mkdir(parents=True, exist_ok=False)
        _write_json(output / "config.json", metadata)
        _write_json(output / "benchmark.json", benchmark)
        state = None

    session = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "target_epochs": config.epochs,
        "torch": str(torch.__version__),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "device": str(device),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
    sessions_path = output / "sessions.json"
    sessions = json.loads(sessions_path.read_text()) if sessions_path.is_file() else []
    sessions.append(session)
    _write_json(sessions_path, sessions)

    drugs = tuple(sorted(facts.drugs))
    drug_columns = {drug: index for index, drug in enumerate(drugs)}
    drug_ids = torch.tensor(drugs, dtype=torch.long, device=device)

    def adjacency_builder(graph: Graph) -> RelationalAdjacency | CompositionalAdjacency:
        if config.encoder == "compgcn":
            return compgcn_adjacency(graph, device=device, layout=config.layout)
        normalization = config.normalization
        if normalization == "directional":
            raise ValueError("train: R-GCN requires relation or total normalization")
        return rgcn_adjacency(
            graph, normalization=normalization, device=device, layout=config.layout,
        )

    contexts: list[_FoldContext] = []
    for queries in folds:
        supervision = training_supervision(facts, split, query_diseases=queries)
        visible = message_graph(
            source_graph,
            withheld_diseases=split.withheld | queries,
            masked_pairs=frozenset(),
        )
        adjacency = adjacency_builder(visible)
        rows, columns = _indices(supervision, drug_columns, device)
        disease_ids = torch.tensor(supervision.diseases, dtype=torch.long, device=device)
        contexts.append(_FoldContext(supervision, adjacency, disease_ids, rows, columns))

    visible = message_graph(
        source_graph, withheld_diseases=split.withheld, masked_pairs=frozenset(),
    )
    development_graph = adjacency_builder(visible)

    torch.manual_seed(config.seed)
    random_generator = random.Random(config.seed)

    embeddings = NodeEmbeddings(
        source_graph.node_count, config.width, trainable=config.trainable_features,
        generator=torch.Generator().manual_seed(config.seed),
    )
    encoder = (
        CompGCNEncoder(source_graph.relations, config.width, config.width, config.width)
        if config.encoder == "compgcn"
        else RGCNEncoder(source_graph.relations, config.width, config.width, config.width)
    )
    model = IndicationModel(embeddings, encoder, DistMultDecoder(config.width)).to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate, weight_decay=config.weight_decay,
    )

    start_epoch = 0
    best: tuple[float, float] | None = None
    history: list[dict[str, Any]] = []

    if state is not None:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        random_generator.setstate(state["sampling_rng"])
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        elif device.type == "mps" and state.get("mps_rng") is not None:
            torch.mps.set_rng_state(state["mps_rng"])

        start_epoch = state["epoch"]
        history = state["history"]
        best = tuple(state["best"]) if state["best"] is not None else None
        if config.epochs < start_epoch:
            raise ValueError("train: target epochs precede completed checkpoint")
        if start_epoch < 1 or len(history) != start_epoch or best is None:
            raise ValueError("train: inconsistent resume history")
        best_record = max(
            history,
            key=lambda record: (record["development_recall"], record["development_ndcg"]),
        )
        if best != (best_record["development_recall"], best_record["development_ndcg"]):
            raise ValueError("train: inconsistent best development metrics")

        selected_path = _selected_path(output, best_record["epoch"])
        if not selected_path.is_file():
            raise ValueError("train: selected checkpoint missing from resumed run")
        selected = torch.load(selected_path, map_location="cpu", weights_only=True)
        if selected["metadata"] != metadata:
            raise ValueError("train: selected checkpoint identity mismatch")
        if selected["epoch"] != best_record["epoch"] or selected["development"] != best_record:
            raise ValueError("train: selected checkpoint does not match history")
        _write_json(output / "development.json", selected["rankings"])
        _write_history(output / "epochs.jsonl", history)

    for epoch in range(start_epoch, config.epochs):
        model.train()
        fold_losses: list[float] = []
        order = list(range(len(contexts)))
        random_generator.shuffle(order)

        for index in order:
            context = contexts[index]
            sampled = _sample_columns(
                context.supervision, drug_columns, random_generator,
                config.negatives, device,
            )
            positive_scores, unjudged_scores = _sampled_scores(
                model, context, sampled, drug_ids,
            )
            if config.objective == "bce":
                loss = sampled_bce_loss(
                    positive_scores, context.positive_rows, unjudged_scores,
                    negative_weight=config.negative_weight,
                )
            elif config.objective == "bpr":
                loss = sampled_bpr_loss(
                    positive_scores, context.positive_rows, unjudged_scores,
                )
            else:
                loss = sampled_lsep_loss(
                    positive_scores, context.positive_rows, unjudged_scores,
                )
            if not torch.isfinite(loss).item():
                raise RuntimeError(f"train: nonfinite loss at epoch {epoch + 1}")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.requires_grad and (
                    parameter.grad is None or not torch.isfinite(parameter.grad).all().item()
                ):
                    raise RuntimeError(f"train: invalid gradient for {name}")
            optimizer.step()
            fold_losses.append(float(loss.detach()))

        result, scores = _development(
            model, development_graph, facts, split,
            drugs=drugs, k=config.k, device=device,
        )
        record = {
            "epoch": epoch + 1,
            "train_loss": sum(fold_losses) / len(fold_losses),
            "development_recall": result.recall,
            "development_ndcg": result.ndcg,
            "development_mrr": result.reciprocal_rank,
        }
        history.append(record)

        selection = (result.recall, result.ndcg)
        improved = best is None or selection > best
        if improved:
            best = selection
            _save_checkpoint(_selected_path(output, epoch + 1), {
                "epoch": epoch + 1,
                "metadata": metadata,
                "model": model.state_dict(),
                "development": record,
                "rankings": _development_record(result, scores, drugs),
            })

        _save_checkpoint(output / "resume.pt", {
            "epoch": epoch + 1,
            "metadata": metadata,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "sampling_rng": random_generator.getstate(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            "mps_rng": torch.mps.get_rng_state() if device.type == "mps" else None,
            "best": best,
            "history": history,
        })
        if improved:
            _write_json(output / "development.json", _development_record(result, scores, drugs))
        _write_history(output / "epochs.jsonl", history)
        if improved:
            for older in (output / "selected").glob("e*.pt"):
                if older.name != f"e{epoch + 1:04d}.pt":
                    older.unlink()
        if on_epoch is not None:
            on_epoch(record.copy())

    best_epoch = max(history, key=lambda row: (
        row["development_recall"], row["development_ndcg"],
    ))["epoch"]
    selected = torch.load(_selected_path(output, best_epoch), map_location="cpu", weights_only=True)
    return {
        "best_epoch": selected["epoch"],
        "best_development": selected["development"],
        "completed_epochs": config.epochs,
        "output": str(output),
    }
