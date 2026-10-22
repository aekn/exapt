import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import torch

from .checkpoint import sha256
from .data import THERAPEUTIC_RELATIONS
from .graph import load_graph
from .model import ExaptGNN, RelationalGNN
from .pretrain import pretrain_model
from .prototypes import build_prototypes
from .train import disease_bpr, load_pairs, load_therapy, train_model


def overfit(
    dataset: Path, *, width: int = 32, steps: int = 300,
    learning_rate: float = 0.003, device: str = "cpu",
    prototype_k: int | None = None, seed: int = 163,
) -> dict[str, Any]:
    if (
        steps < 1 or width < 1 or not math.isfinite(learning_rate)
        or learning_rate <= 0
    ):
        raise ValueError("Invalid overfit configuration")
    if prototype_k is not None and prototype_k < 1:
        raise ValueError("prototype_k must be positive")

    torch.manual_seed(seed)
    graph = load_graph(dataset)
    if len(graph.nodes) > 10_000:
        raise ValueError("Overfit check requires a reduced smoke graph")

    therapy = load_therapy(dataset / "therapeutic_train.csv", graph.nodes)
    training = load_pairs(dataset / "train.csv", graph.nodes)
    with (dataset / "drugs.csv").open(newline="") as stream:
        drugs = sorted(
            graph.nodes[int(row["drug_index"])]
            for row in csv.DictReader(stream)
        )

    base = RelationalGNN(len(graph.nodes), len(graph.relations), width)
    model: RelationalGNN | ExaptGNN = base
    if prototype_k is not None:
        donors = {disease for _, disease in training}
        prototypes = build_prototypes(graph, donors, donors, prototype_k)
        model = ExaptGNN(base, prototypes, width)

    # fixed training pairs and negatives; development labels are never read
    supported = (
        set(model.diseases.tolist()) if isinstance(model, ExaptGNN)
        else set()
    )
    selected: list[tuple[int, int, int, list[int]]] = []
    for relation, name in enumerate(THERAPEUTIC_RELATIONS):
        pairs = therapy[name]
        if not pairs:
            continue
        if supported:
            pairs = sorted(pairs, key=lambda pair: pair[1] not in supported)

        drug, disease = pairs[0]
        known = {drug for drug, d in therapy[name] if d == disease}
        negatives = [candidate for candidate in drugs if candidate not in known]
        negatives = negatives[:4]
        if not negatives:
            raise ValueError(f"No legal comparison drugs for {name}")
        selected.append((relation, drug, disease, negatives))

    if not selected or selected[0][0] != 0:
        raise ValueError("Overfit check requires training indication labels")

    target = torch.device(device)
    graph = graph.to(target)
    model = model.to(target)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.0,
        foreach=False,
    )
    tasks = [
        (
            relation,
            torch.tensor([drug], device=target),
            torch.tensor([disease], device=target),
            torch.tensor([negatives], device=target),
        )
        for relation, drug, disease, negatives in selected
    ]
    coefficients = (1.0, 0.25, 0.25)
    weights = torch.ones(1, device=target)

    def objective() -> torch.Tensor:
        embeddings = model(graph)
        losses = []
        for relation, positive, disease, negatives in tasks:
            positive_scores = model.score(
                embeddings, positive, disease, relation,
            )
            negative_scores = model.score(
                embeddings, negatives, disease[:, None], relation,
            )
            losses.append(coefficients[relation] * disease_bpr(
                positive_scores, negative_scores, weights,
            ))
        return torch.stack(losses).sum()

    initial = 0.0
    for step in range(steps):
        optimizer.zero_grad()
        loss = objective()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite overfit loss")
        if step == 0:
            initial = float(loss.detach())
        loss.backward()
        if not all(
            bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
            if parameter.grad is not None
        ):
            raise FloatingPointError("Nonfinite overfit gradient")
        optimizer.step()

    with torch.no_grad():
        final = float(objective())
    if not math.isfinite(final):
        raise FloatingPointError("Nonfinite final overfit loss")

    result = {
        "check": "train_only_overfit", "device": device,
        "width": width, "steps": steps, "prototype_k": prototype_k,
        "relations": [THERAPEUTIC_RELATIONS[item[0]] for item in selected],
        "initial_loss": initial, "final_loss": final,
        "reduction": 1 - final / initial,
        "passed": final < initial * 0.5,
    }
    if not result["passed"]:
        raise RuntimeError(f"Fixed training-pair overfit failed: {result}")
    return result


def profile(
    stage: str, data: Path, output: Path, *, dataset: Path | None = None,
    width: int = 128, device: str = "cuda", prototype_k: int | None = None,
) -> dict[str, Any]:
    if stage not in ("fit", "pretrain"):
        raise ValueError("Unknown profiling stage")
    if stage == "fit" and dataset is None:
        raise ValueError("Fit profiling requires the training split")
    if width < 1:
        raise ValueError("Profile width must be positive")
    if output.exists() and (
        not output.is_dir() or any(output.iterdir())
    ):
        raise FileExistsError(f"Profiling directory is not empty: {output}")
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        torch.cuda.reset_peak_memory_stats(device)

    started = time.perf_counter()
    if stage == "fit":
        assert dataset is not None
        result = train_model(
            data, dataset, output, epochs=1, width=width,
            eval_every=1, min_epochs=0, device=device,
            prototype_k=prototype_k, run_id="profile",
        )
    else:
        result = pretrain_model(
            data, output, steps=1, width=width, batch_size=8192,
            eval_every=1, checkpoint_every=1, device=device,
            run_id="profile",
        )
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)

    snapshots = list((output / "checkpoints").glob("*.pt"))
    if len(snapshots) != 1:
        raise RuntimeError("Profiling expected exactly one checkpoint")

    report = {
        "stage": stage, "device": device,
        "torch": str(torch.__version__), "width": width,
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint_bytes": snapshots[0].stat().st_size,
        "checkpoint_sha256": sha256(snapshots[0]),
        "cuda_peak_allocated_bytes": (
            torch.cuda.max_memory_allocated(device)
            if device.startswith("cuda") else None
        ),
        "cuda_peak_reserved_bytes": (
            torch.cuda.max_memory_reserved(device)
            if device.startswith("cuda") else None
        ),
        "status": result.get("status"),
    }
    (output / "preflight.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
