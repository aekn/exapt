import csv
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from .checkpoint import (
    append_event, cpu_state, load_snapshot, restore_rng, rng_state,
    save_checkpoint, save_snapshot, write_events,
)
from .data import THERAPEUTIC_RELATIONS, sha256_file
from .evaluation import macro_metrics
from .graph import load_graph
from .model import ExaptGNN, RelationalGNN
from .prototypes import build_prototypes
from .stopping import ranking_key, ranking_progress


def load_pairs(path: Path, nodes: dict[int, int]) -> list[tuple[int, int]]:
    with path.open(newline="") as source:
        return [
            (nodes[int(row["drug_index"])], nodes[int(row["disease_index"])])
            for row in csv.DictReader(source)
        ]


def load_therapy(
    path: Path, nodes: dict[int, int]
) -> dict[str, list[tuple[int, int]]]:
    pairs: dict[str, list[tuple[int, int]]] = {
        name: [] for name in THERAPEUTIC_RELATIONS
    }
    with path.open(newline="") as source:
        for row in csv.DictReader(source):
            pairs[row["relation"]].append((
                nodes[int(row["drug_index"])],
                nodes[int(row["disease_index"])],
            ))
    return pairs


def disease_bpr(
    positive: torch.Tensor, negative: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    # (m,), (m, k), (m,) positive, negative, pair weights
    # L = sum_i w_i mean_j softplus(s_neg_ij - s_pos_i)
    return (F.softplus(negative - positive[:, None]).mean(1) * weights).sum()


def train_model(
    graph_path: Path, dataset: Path, output: Path, *,
    epochs: int = 500, width: int = 32, seed: int = 163,
    learning_rate: float = 5e-4, eval_every: int = 5,
    weight_decay: float = 1e-4, patience: int = 15,
    min_epochs: int = 50, map_min_delta: float = 1e-4,
    device: str = "cpu", pretrained: Path | None = None,
    prototype_k: int | None = None, negatives: int = 8,
    resume: Path | None = None, stop_after: int | None = None,
    run_id: str = "local", bucket: str | None = None,
    allow_public_bucket: bool = False,
    source_commit: str | None = None,
    dataset_revision: str | None = None,
    manifest_sha256: str | None = None,
    remote_seconds: float = 900,
) -> dict[str, Any]:
    if min(epochs, eval_every, negatives, patience) < 1:
        raise ValueError("epochs, eval_every and negatives must be positive")
    if min_epochs < 0 or weight_decay < 0 or map_min_delta < 0:
        raise ValueError("Invalid stopping or weight decay configuration")
    if stop_after is not None and not 1 <= stop_after <= epochs:
        raise ValueError("stop_after must be within the planned epochs")

    bucket_store = None
    if bucket is not None:
        from .hub import BucketStore

        if not all((source_commit, dataset_revision, manifest_sha256)):
            raise ValueError(
                "Remote training requires source commit, dataset revision "
                "and manifest SHA-256"
            )
        bucket_store = BucketStore(
            bucket, run_id, "fit", allow_public=allow_public_bucket,
            sync_seconds=remote_seconds,
        )

    torch.manual_seed(seed)
    sampling_rng = random.Random(seed)

    graph = load_graph(graph_path)
    train_pairs = load_pairs(dataset / "train.csv", graph.nodes)
    dev_pairs = load_pairs(dataset / "development.csv", graph.nodes)
    therapeutic_pairs = load_therapy(
        dataset / "therapeutic_train.csv", graph.nodes
    )

    with (dataset / "drugs.csv").open(newline="") as source:
        candidates = sorted(
            graph.nodes[int(row["drug_index"])]
            for row in csv.DictReader(source)
        )

    # indication labels, separate from contraindication and off-label heads
    train_positives: dict[int, set[int]] = defaultdict(set)
    for drug, disease in train_pairs:
        train_positives[disease].add(drug)

    dev_labels: dict[int, set[int]] = defaultdict(set)
    for drug, disease in dev_pairs:
        dev_labels[disease].add(drug)

    if graph_path.is_file():
        graph_files = [graph_path]
    else:
        graph_files = [graph_path / "graph.csv", graph_path / "nodes.csv"]
    graph_inputs = {path.name: sha256_file(path) for path in graph_files}

    base_model = RelationalGNN(len(graph.nodes), len(graph.relations), width)
    if pretrained is not None:
        pretrained_state = torch.load(
            pretrained, map_location="cpu", weights_only=True
        )
        if (
            pretrained_state.get("graph_inputs") != graph_inputs
            or pretrained_state["nodes"] != len(graph.nodes)
            or pretrained_state["relations"] != graph.relations
            or pretrained_state["width"] != width
        ):
            raise ValueError("Pretrained encoder does not match this graph")

        # initialize the graph encoder, not the therapeutic decoder
        encoder_state = {
            name: value for name, value in pretrained_state["encoder"].items()
            if name.startswith(("embeddings.", "first.", "second."))
        }
        base_model.encoder.load_state_dict(encoder_state)

    model: RelationalGNN | ExaptGNN = base_model
    coverage: dict[str, int] | None = None
    donor_ids = sorted(train_positives) if prototype_k is not None else []
    if prototype_k is not None:
        # training diseases donate, while development diseases may borrow
        donor_diseases = set(donor_ids)
        query_diseases = donor_diseases | set(dev_labels)
        prototypes = build_prototypes(
            graph, donor_diseases, query_diseases, prototype_k
        )

        supported = set(prototypes.diseases.tolist())
        coverage = {
            "train": len(supported & donor_diseases),
            "development": len(supported & dev_labels.keys()),
        }
        model = ExaptGNN(base_model, prototypes, width)

    graph = graph.to(torch.device(device))
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay,
        foreach=False,
    )

    # (relation, coefficient, drugs, diseases, pair weights, known positives)
    tasks = []
    task_weights = (1.0, 0.25, 0.25)
    for relation, (name, task_weight) in enumerate(
        zip(THERAPEUTIC_RELATIONS, task_weights)
    ):
        pairs = therapeutic_pairs[name]
        if not pairs:
            continue

        observed_drugs: dict[int, set[int]] = defaultdict(set)
        for drug, disease in pairs:
            observed_drugs[disease].add(drug)

        drugs, diseases = torch.tensor(pairs, dtype=torch.long, device=device).T

        # w_i = 1 / (|D_r| * |P_r(d_i)|), equal total weight per disease
        pair_weights = torch.tensor(
            [
                1 / (len(observed_drugs) * len(observed_drugs[d]))
                for _, d in pairs
            ],
            dtype=torch.float, device=device,
        )
        tasks.append((
            relation, task_weight, drugs, diseases,
            pair_weights, observed_drugs,
        ))

    if not tasks or tasks[0][0] != 0:
        raise ValueError("Training needs indication supervision")

    candidate_nodes = torch.tensor(candidates, dtype=torch.long, device=device)
    dev_diseases = sorted(dev_labels)
    dev_nodes = torch.tensor(dev_diseases, dtype=torch.long, device=device)

    def evaluate() -> tuple[dict[str, float], torch.Tensor, torch.Tensor]:
        model.eval()
        with torch.no_grad():
            embeddings = model(graph)
            drug_vectors = embeddings[candidate_nodes] * model.indication
            disease_vectors = model.disease_vectors(embeddings, dev_nodes)

            # S = ((H_drug * w) @ H_disease.T).T  (diseases, drugs)
            scores = (drug_vectors @ disease_vectors.T).T
            if not bool(torch.isfinite(scores).all()):
                raise FloatingPointError("Nonfinite development prediction")
            # stable ties keep ascending candidate node order
            order = torch.argsort(scores, dim=1, descending=True, stable=True)

        rankings = {
            disease: [candidates[i] for i in indices]
            for disease, indices in zip(dev_diseases, order.cpu().tolist())
        }
        return macro_metrics(rankings, dev_labels), scores, order

    output.mkdir(parents=True, exist_ok=True)

    inputs = {
        name: sha256_file(dataset / name)
        for name in (
            "train.csv", "development.csv", "drugs.csv", "therapeutic_train.csv"
        )
    }
    inputs.update(graph_inputs)

    config = {
        "epochs": epochs,
        "width": width,
        "seed": seed,
        "learning_rate": learning_rate,
        "optimizer": "AdamW",
        "weight_decay": weight_decay,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "foreach": False,
        "patience": patience,
        "min_epochs": min_epochs,
        "map_min_delta": map_min_delta,
        "negatives": negatives,
        "eval_every": eval_every,
        "prototype_k": prototype_k,
        "pretrained": sha256_file(pretrained) if pretrained else None,
    }
    if bucket_store is not None:
        from .hub import verify_training_inputs

        verify_training_inputs(
            source_commit or "", dataset.parent, manifest_sha256 or "",
        )
        program = (
            "rgcn-distmult" if pretrained is None and prototype_k is None
            else "exaptgnn-v1"
        )
        bucket_store.prepare_stage(
            program=program,
            seed=seed, config=config, inputs=inputs,
            source_commit=source_commit or "",
            dataset_revision=dataset_revision or "",
            manifest_sha256=manifest_sha256 or "",
        )

    history: list[tuple[int, float, float | None]] = []
    events: list[dict[str, Any]] = []
    # select by recall@20, MAP, then NDCG@20; patience ignores NDCG
    best_ranking_key = (-1.0, -1.0, -1.0)
    best_metrics: dict[str, float] | None = None
    best_model_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    progress_anchor = (-1.0, -1.0, -1.0)
    stale_checks = 0
    start_epoch = 1
    stage = (
        "baseline" if pretrained is None and prototype_k is None
        else "finetune"
    )
    model_metadata = {
        "graph_inputs": graph_inputs,
        "nodes": graph.nodes,
        "relations": graph.relations,
        "therapeutic_relations": THERAPEUTIC_RELATIONS,
        "width": width,
        "prototype_k": prototype_k,
        "donor_diseases": donor_ids,
    }

    if resume is not None:
        checkpoint = load_snapshot(resume)
        if (
            checkpoint["stage"] != stage
            or checkpoint["run_id"] != run_id
            or checkpoint["config"] != config
            or checkpoint["inputs"] != inputs
        ):
            raise ValueError("Resume stage, configuration or data differs")
        if checkpoint["position"] >= epochs:
            raise ValueError("Checkpoint already reached the planned epochs")

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        restore_rng(checkpoint["rng"])
        sampling_rng.setstate(checkpoint["sampling_rng"])
        history = checkpoint["history"]
        events = checkpoint["events"]
        best_epoch = checkpoint["best_epoch"]
        best_ranking_key = tuple(checkpoint["best_key"])
        best_metrics = checkpoint["best_metrics"]
        best_model_state = checkpoint["best_state"]
        progress_anchor = tuple(checkpoint["progress_anchor"])
        stale_checks = checkpoint["stale_checks"]
        start_epoch = checkpoint["position"] + 1
        if best_model_state is not None:
            save_checkpoint(output / "model.pt", {
                "model": best_model_state, **model_metadata,
            })

    # restore the persisted event stream from the committed checkpoint
    if resume is None and (output / "latest.json").exists():
        raise FileExistsError("Output contains an existing training run")
    write_events(output, events)
    (output / "run.json").write_text(json.dumps({
        "run_id": run_id, "stage": stage, "status": "running",
        "config": config, "inputs": inputs,
    }, indent=2) + "\n")
    last_epoch = stop_after if stop_after is not None else epochs
    termination = "manual_stop" if last_epoch < epochs else "max_epochs"
    completed_epoch = start_epoch - 1
    print(
        f"train     device {device}  nodes {len(graph.nodes)}  "
        f"width {width}  pairs "
        f"{sum(len(v) for v in therapeutic_pairs.values())}",
        flush=True,
    )
    for epoch in range(start_epoch, last_epoch + 1):
        started = time.perf_counter()

        # k comparison drugs per positive, excluding this head's positives
        sampled = []
        for (
            relation, task_weight, drugs, diseases, pair_weights, observed_drugs
        ) in tasks:
            pairs = therapeutic_pairs[THERAPEUTIC_RELATIONS[relation]]
            negative_rows: list[list[int]] = []
            for _, disease in pairs:
                row: list[int] = []
                while len(row) < negatives:
                    drug = sampling_rng.choice(candidates)
                    if drug not in observed_drugs[disease]:
                        row.append(drug)
                negative_rows.append(row)

            negative_drugs = torch.tensor(
                negative_rows, dtype=torch.long, device=device
            )
            sampled.append((
                relation, task_weight, drugs, diseases,
                pair_weights, negative_drugs,
            ))

        model.train()
        optimizer.zero_grad()
        embeddings = model(graph)

        losses = []
        head_losses: dict[str, float] = {}
        for (
            relation, task_weight, drugs, diseases, pair_weights, negative_drugs
        ) in sampled:
            positive_scores = model.score(embeddings, drugs, diseases, relation)
            negative_scores = model.score(
                embeddings, negative_drugs, diseases[:, None], relation,
            )
            head_loss = disease_bpr(
                positive_scores, negative_scores, pair_weights,
            )
            head_losses[THERAPEUTIC_RELATIONS[relation]] = float(
                head_loss.detach()
            )
            losses.append(task_weight * head_loss)

        loss = torch.stack(losses).sum()
        if not bool(torch.isfinite(loss)):
            (output / "run.json").write_text(json.dumps({
                "run_id": run_id, "stage": stage, "status": "numerical_failure",
                "epoch": epoch, "config": config, "inputs": inputs,
            }, indent=2) + "\n")
            raise FloatingPointError(f"Nonfinite loss at epoch {epoch}")
        loss.backward()
        if not all(
            bool(torch.isfinite(param.grad).all())
            for param in model.parameters() if param.grad is not None
        ):
            (output / "run.json").write_text(json.dumps({
                "run_id": run_id, "stage": stage, "status": "numerical_failure",
                "epoch": epoch, "config": config, "inputs": inputs,
            }, indent=2) + "\n")
            raise FloatingPointError(f"Nonfinite gradient at epoch {epoch}")
        optimizer.step()

        metrics = None
        best_changed = False
        if epoch % eval_every == 0 or epoch == epochs:
            try:
                metrics, _, _ = evaluate()
            except FloatingPointError:
                (output / "run.json").write_text(json.dumps({
                    "run_id": run_id, "stage": stage,
                    "status": "numerical_failure", "epoch": epoch,
                    "config": config, "inputs": inputs,
                }, indent=2) + "\n")
                raise
            current_key = ranking_key(metrics)
            if ranking_progress(current_key, progress_anchor, map_min_delta):
                progress_anchor = current_key
                stale_checks = 0
            else:
                stale_checks += 1
            if current_key > best_ranking_key:
                best_epoch, best_ranking_key = epoch, current_key
                best_metrics = metrics
                best_changed = True
                best_model_state = cpu_state(model)
                save_checkpoint(output / "model.pt", {
                    "model": best_model_state, **model_metadata,
                })

        loss_value = float(loss.detach())
        history.append((
            epoch, loss_value, metrics["recall@20"] if metrics else None,
        ))
        event = {
            "run_id": run_id, "stage": stage, "event": "train",
            "epoch": epoch, "loss": loss_value, "heads": head_losses,
        }
        events.append(event)
        append_event(output, event)
        if metrics is not None:
            event = {
                "run_id": run_id, "stage": stage, "event": "eval",
                "epoch": epoch, "best_epoch": best_epoch,
                "stale_checks": stale_checks, **metrics,
            }
            events.append(event)
            append_event(output, event)

        if bucket_store is not None and best_changed:
            bucket_store.publish_selected(output)
        if epoch % 25 == 0:
            milestone = output / "milestones" / f"e{epoch:04d}.pt"
            save_checkpoint(milestone, {
                "model": cpu_state(model), **model_metadata,
            })
            if bucket_store is not None:
                bucket_store.publish_milestone(milestone)

        # every supervised update is recoverable
        save_snapshot(output, stage, epoch, {
            "format": 1,
            "stage": stage,
            "run_id": run_id,
            "position": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "rng": rng_state(),
            "sampling_rng": sampling_rng.getstate(),
            "history": history,
            "events": events,
            "best_epoch": best_epoch,
            "best_key": best_ranking_key,
            "best_metrics": best_metrics,
            "best_state": best_model_state,
            "progress_anchor": progress_anchor,
            "stale_checks": stale_checks,
            "config": config,
            "inputs": inputs,
            "model_meta": model_metadata,
        })

        should_stop = (
            metrics is not None and stale_checks >= patience
            and epoch >= min_epochs
        )
        if should_stop:
            termination = "early_stopping"
        if bucket_store is not None:
            bucket_store.publish_snapshot(
                output, force=(epoch == last_epoch or should_stop),
            )

        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        head_summary = "  ".join(
            f"{label} {head_losses[name]:.4f}"
            for label, name in zip(
                ("ind", "contra", "off"), THERAPEUTIC_RELATIONS
            )
            if name in head_losses
        )
        print(
            f"epoch  {epoch:>5}/{epochs:<5} loss {loss_value:.5f}  "
            f"{head_summary}  {elapsed:.2f}s", flush=True,
        )
        if metrics is not None:
            print(
                f"eval   {epoch:>5}/{epochs:<5} recall@20 "
                f"{metrics['recall@20']:.6f}  "
                f"best {best_ranking_key[0]:.6f} ({best_epoch})", flush=True,
            )
        completed_epoch = epoch
        if should_stop:
            break

    # interrupted runs have recovery state but no final predictions
    if termination == "manual_stop":
        (output / "run.json").write_text(json.dumps({
            "run_id": run_id, "stage": stage, "status": termination,
            "epoch": completed_epoch, "best_epoch": best_epoch,
            "config": config, "inputs": inputs,
        }, indent=2) + "\n")
        return {
            "epoch": completed_epoch,
            "best_epoch": best_epoch,
            "status": termination,
        }

    if best_model_state is None:
        raise RuntimeError("Training completed without a selected checkpoint")
    model.load_state_dict(best_model_state)
    metrics, scores, order = evaluate()
    recall = metrics["recall@20"]

    original_ids = {local: source for source, local in graph.nodes.items()}
    with (output / "loss.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("epoch", "loss", "development_recall_at_20"))
        writer.writerows(history)

    with (output / "predictions.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("disease_index", "drug_index", "score", "rank"))
        for disease, values, indices in zip(
            dev_diseases, scores.cpu().tolist(), order.cpu().tolist()
        ):
            for rank, i in enumerate(indices, 1):
                writer.writerow((
                    original_ids[disease], original_ids[candidates[i]],
                    values[i], rank,
                ))

    result: dict[str, Any] = {
        "run_id": run_id,
        "stage": stage,
        "status": "completed",
        "termination": termination,
        "completed_epochs": completed_epoch,
        "stale_checks": stale_checks,
        "config": config,
        "seed": seed,
        "epochs": epochs,
        "width": width,
        "device": device,
        "torch": torch.__version__,
        "inputs": inputs,
        "loss": "BPR",
        "negative_ratio": negatives,
        "therapeutic_relations": THERAPEUTIC_RELATIONS,
        "therapeutic_pairs": {
            name: len(therapeutic_pairs[name]) for name in THERAPEUTIC_RELATIONS
        },
        "learning_rate": learning_rate,
        "nodes": len(graph.nodes),
        "relations": len(graph.relations),
        "train_pairs": len(train_pairs),
        "development_pairs": len(dev_pairs),
        "candidate_drugs": len(candidates),
        "best_epoch": best_epoch,
        "development_metrics": metrics,
        "initial_loss": history[0][1],
        "final_loss": history[-1][1],
        "development_recall_at_20": recall,
    }
    if pretrained is not None:
        result["pretrained_sha256"] = sha256_file(pretrained)
    if prototype_k is not None:
        result["model"] = "exaptgnn"
        result["prototype_k"] = prototype_k
        result["prototype_covered"] = coverage

    (output / "run.json").write_text(json.dumps(result, indent=2) + "\n")
    if bucket_store is not None:
        bucket_store.publish_final(output)
    return result


def train_smoke(
    directory: Path, output: Path, epochs: int = 20,
    width: int = 32, seed: int = 163, device: str = "cpu",
) -> dict[str, Any]:
    return train_model(
        directory, directory, output, epochs=epochs, width=width,
        seed=seed, device=device, eval_every=epochs,
    )
