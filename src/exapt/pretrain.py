import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .checkpoint import (
    append_event, cpu_state, load_snapshot, restore_rng, rng_state,
    save_checkpoint, save_snapshot, write_events,
)
from .data import sha256_file
from .graph import Graph, load_graph
from .model import DistMultDecoder, RelationalEncoder
from .stopping import loss_progress


def prepare_pretrain(
    graph_path: Path, output: Path, seed: int = 163
) -> dict[str, Any]:
    graph = load_graph(graph_path)
    node_count = len(graph.nodes)

    # unordered pair key: N * min(u, v) + max(u, v)
    # holdout assignments are shared across directions and relations
    pair_keys: list[torch.Tensor] = []
    for matrix in graph.adjacency:
        targets, sources = matrix.indices()
        pair_keys.append(
            torch.minimum(sources, targets) * node_count
            + torch.maximum(sources, targets)
        )

    unique_pairs = torch.unique(torch.cat(pair_keys))
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(unique_pairs), generator=generator)
    n_train = round(len(unique_pairs) * 0.18)
    n_validation = round(len(unique_pairs) * 0.02)
    train_keys = unique_pairs[order[:n_train]]
    validation_keys = unique_pairs[order[n_train:n_train + n_validation]]

    adjacency: list[torch.Tensor] = []
    positives: list[torch.Tensor] = []   # (m_r, 2) directed training pairs
    validation: list[torch.Tensor] = []  # (v_r, 2) directed validation pairs
    full_edges: list[torch.Tensor] = []  # sorted directed edge keys
    nodes_by_type: dict[str, list[torch.Tensor]] = {}

    for relation, matrix, keys in zip(
        graph.relations, graph.adjacency, pair_keys, strict=True
    ):
        source_type, _, target_type = relation
        targets, sources = matrix.indices()
        nodes_by_type.setdefault(source_type, []).append(sources)
        nodes_by_type.setdefault(target_type, []).append(targets)

        train_mask = torch.isin(keys, train_keys)
        validation_mask = torch.isin(keys, validation_keys)
        masked = train_mask | validation_mask
        positives.append(torch.stack(
            (sources[train_mask], targets[train_mask]), dim=1,
        ))
        validation.append(torch.stack(
            (sources[validation_mask], targets[validation_mask]), dim=1,
        ))

        indices = matrix.indices()[:, ~masked]
        values = matrix.values()[~masked]

        # renormalize A_r rows after removing held-out edges
        row_sum = torch.zeros(node_count, dtype=values.dtype)
        row_sum.index_add_(0, indices[0], values)
        weights = values / row_sum[indices[0]]
        adjacency.append(torch.sparse_coo_tensor(
            indices, weights, matrix.shape, check_invariants=True,
        ).coalesce())

        # negatives cannot be any observed edge
        full_edges.append((sources * node_count + targets).sort().values)

    # endpoint pools include nodes from held-out edges
    pools = {
        name: torch.unique(torch.cat(indices))
        for name, indices in nodes_by_type.items()
    }
    if graph_path.is_file():
        source_files = [graph_path]
    else:
        source_files = [graph_path / "graph.csv", graph_path / "nodes.csv"]
    graph_inputs = {file.name: sha256_file(file) for file in source_files}

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "graph_inputs": graph_inputs,
        "nodes": node_count,
        "relations": graph.relations,
        "adjacency": adjacency,
        "positives": positives,
        "validation": validation,
        "pools": pools,
        "full_edges": full_edges,
    }, output)

    return {
        "inputs": graph_inputs,
        "sha256": sha256_file(output),
        "nodes": node_count,
        "relations": len(graph.relations),
        "prediction_edges": sum(len(edges) for edges in positives),
        "validation_edges": sum(len(edges) for edges in validation),
        "message_edges": sum(matrix._nnz() for matrix in adjacency),
        "seed": seed,
        "train_fraction": 0.18,
        "validation_fraction": 0.02,
    }


def _sample_negatives(
    sources: torch.Tensor, pool: torch.Tensor,
    known: torch.Tensor, node_count: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    # known contains sorted directed edges, including both holdouts
    negatives = pool[torch.randint(
        len(pool), (len(sources),), generator=generator,
    )]
    while True:
        # directed pair key: N * source + target
        keys = sources * node_count + negatives
        positions = torch.searchsorted(known, keys)

        # insertion position may equal len(known)
        collision = (positions < len(known)) & (
            known[positions.clamp(max=len(known) - 1)] == keys
        )
        if not collision.any():
            return negatives

        count = int(collision.sum())
        negatives[collision] = pool[torch.randint(
            len(pool), (count,), generator=generator,
        )]


def pretrain_model(
    prepared: Path, output: Path, *, steps: int = 2000,
    width: int = 32, seed: int = 163, learning_rate: float = 0.001,
    batch_size: int = 8192, eval_every: int = 50,
    weight_decay: float = 1e-4, patience: int = 8,
    loss_min_delta: float = 1e-4,
    device: str = "cpu", resume: Path | None = None,
    checkpoint_every: int = 25, stop_after: int | None = None,
    run_id: str = "local", bucket: str | None = None,
    allow_public_bucket: bool = False,
    source_commit: str | None = None,
    dataset_revision: str | None = None,
    manifest_sha256: str | None = None,
    remote_seconds: float = 900,
) -> dict[str, Any]:
    if min(steps, batch_size, eval_every, checkpoint_every, patience) < 1:
        raise ValueError(
            "steps, batch_size, eval_every and checkpoint_every "
            "must be positive"
        )
    if weight_decay < 0 or loss_min_delta < 0:
        raise ValueError("Invalid stopping or weight decay configuration")
    if stop_after is not None and not 1 <= stop_after <= steps:
        raise ValueError("stop_after must be within the planned steps")

    bucket_store = None
    if bucket is not None:
        from .hub import BucketStore

        if not all((source_commit, dataset_revision, manifest_sha256)):
            raise ValueError(
                "Remote training requires source commit, dataset revision "
                "and manifest SHA-256"
            )
        bucket_store = BucketStore(
            bucket, run_id, "pretrain", allow_public=allow_public_bucket,
            sync_seconds=remote_seconds,
        )

    torch.manual_seed(seed)
    prepared_data = torch.load(prepared, map_location="cpu", weights_only=True)
    graph = Graph(
        {index: index for index in range(prepared_data["nodes"])},
        tuple(tuple(relation) for relation in prepared_data["relations"]),
        tuple(prepared_data["adjacency"]),
    ).to(torch.device(device))

    encoder = RelationalEncoder(len(graph.nodes), len(graph.relations), width)
    encoder.to(device)
    decoder = DistMultDecoder(width, len(graph.relations)).to(device)
    nn.init.normal_(decoder.weight, std=width ** -0.5)

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=learning_rate, weight_decay=weight_decay, foreach=False,
    )
    # P(r) proportional to sqrt(m_r), tempering frequent relations
    relation_counts = torch.tensor(
        [len(pairs) for pairs in prepared_data["positives"]], dtype=torch.float,
    )
    relation_weights = relation_counts.sqrt()

    # fixed validation negatives use a separate RNG stream
    validation_rng = torch.Generator().manual_seed(seed + 1)
    validation_examples: list[tuple[int, torch.Tensor, torch.Tensor]] = []
    for relation, pairs in enumerate(prepared_data["validation"]):
        if len(pairs) == 0:
            continue

        indices = torch.randperm(len(pairs), generator=validation_rng)[:128]
        selected = pairs[indices]
        pool = prepared_data["pools"][graph.relations[relation][2]]
        negatives = _sample_negatives(
            selected[:, 0], pool, prepared_data["full_edges"][relation],
            len(graph.nodes), validation_rng,
        )
        validation_examples.append((relation, selected, negatives))

    if not validation_examples:
        raise ValueError("No reconstruction validation pairs")

    # constant weights across validation steps, on the model's device
    validation_weights = torch.tensor(
        [relation_weights[index] for index, _, _ in validation_examples],
        device=device,
    )

    history: list[tuple[int, float, float | None]] = []
    validation_history: list[tuple[int, str, str, str, int, float]] = []
    best_validation_loss = float("inf")
    best_step = 0
    progress_loss = float("inf")
    stale_checks = 0
    start_step = 1

    output.mkdir(parents=True, exist_ok=True)
    input_hash = sha256_file(prepared)
    config = {
        "steps": steps,
        "checkpoint_every": checkpoint_every,
        "width": width,
        "seed": seed,
        "learning_rate": learning_rate,
        "optimizer": "AdamW",
        "weight_decay": weight_decay,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "foreach": False,
        "patience": patience,
        "loss_min_delta": loss_min_delta,
        "batch_size": batch_size,
        "eval_every": eval_every,
    }
    if bucket_store is not None:
        from .hub import verify_training_inputs

        verify_training_inputs(
            source_commit or "", prepared.parent, manifest_sha256 or "",
        )
        bucket_store.prepare_stage(
            program="exaptgnn-v1", seed=seed, config=config,
            inputs={"pretrain.pt": input_hash},
            source_commit=source_commit or "",
            dataset_revision=dataset_revision or "",
            manifest_sha256=manifest_sha256 or "",
        )

    events: list[dict[str, Any]] = []
    best_encoder_state: dict[str, torch.Tensor] | None = None
    if resume is not None:
        checkpoint = load_snapshot(resume)
        if (
            checkpoint["stage"] != "pretrain"
            or checkpoint["run_id"] != run_id
            or checkpoint["config"] != config
            or checkpoint["input_sha256"] != input_hash
        ):
            raise ValueError("Resume stage, configuration or data differs")
        if checkpoint["position"] >= steps:
            raise ValueError("Checkpoint already reached the planned steps")

        encoder.load_state_dict(checkpoint["encoder"])
        decoder.load_state_dict(checkpoint["decoder"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        restore_rng(checkpoint["rng"])

        history = checkpoint["history"]
        validation_history = checkpoint["validation_history"]
        events = checkpoint["events"]
        best_step = checkpoint["best_step"]
        best_validation_loss = checkpoint["best_validation_loss"]
        best_encoder_state = checkpoint["best_state"]
        progress_loss = checkpoint["progress_loss"]
        stale_checks = checkpoint["stale_checks"]
        start_step = checkpoint["position"] + 1

        if best_encoder_state is not None:
            save_checkpoint(output / "model.pt", {
                "encoder": best_encoder_state,
                "nodes": len(graph.nodes),
                "relations": graph.relations,
                "width": width,
                "graph_inputs": prepared_data["graph_inputs"],
            })

    if resume is None and (output / "latest.json").exists():
        raise FileExistsError("Output contains an existing training run")

    write_events(output, events)
    (output / "run.json").write_text(json.dumps({
        "run_id": run_id, "stage": "pretrain", "status": "running",
        "config": config, "input_sha256": input_hash,
        "graph_inputs": prepared_data["graph_inputs"],
    }, indent=2) + "\n")
    last_step = stop_after if stop_after is not None else steps
    termination = "manual_stop" if last_step < steps else "max_steps"
    completed_step = start_step - 1
    last_published_best_step = -1

    print(
        f"pretrain device {device}  nodes {len(graph.nodes)}  "
        f"width {width}  batch {batch_size}", flush=True,
    )

    for step in range(start_step, last_step + 1):
        started = time.perf_counter()

        # sample relation by sqrt(m_r), then positive edges uniformly
        sampled_relations = torch.multinomial(
            relation_weights, batch_size, replacement=True
        )

        # (relation index, positive source-target pairs, negative targets)
        examples: list[tuple[int, torch.Tensor, torch.Tensor]] = []
        for relation in sampled_relations.unique().tolist():
            size = int((sampled_relations == relation).sum())
            positives = prepared_data["positives"][relation]
            pairs = positives[torch.randint(len(positives), (size,))]
            pool = prepared_data["pools"][graph.relations[relation][2]]
            negatives = _sample_negatives(
                pairs[:, 0], pool, prepared_data["full_edges"][relation],
                len(graph.nodes),
            )
            examples.append((relation, pairs, negatives))

        encoder.train()
        decoder.train()
        optimizer.zero_grad()
        embeddings = encoder(graph)

        losses = []
        for relation, pairs, negatives in examples:
            sources = pairs[:, 0].to(device)
            targets = pairs[:, 1].to(device)
            negative_targets = negatives.to(device)

            # s_r(u, v) = h_u @ diag(w_r) @ h_v
            pos_scores = decoder.score(
                embeddings[sources], embeddings[targets], relation
            )
            neg_scores = decoder.score(
                embeddings[sources], embeddings[negative_targets], relation
            )

            # L = mean_i [softplus(-s_pos_i) + softplus(s_neg_i)]
            losses.append(
                (F.softplus(-pos_scores) + F.softplus(neg_scores)).sum()
            )

        loss = torch.stack(losses).sum() / batch_size
        if not bool(torch.isfinite(loss)):
            (output / "run.json").write_text(json.dumps({
                "run_id": run_id, "stage": "pretrain",
                "status": "numerical_failure", "step": step,
                "config": config, "input_sha256": input_hash,
            }, indent=2) + "\n")
            raise FloatingPointError(f"Nonfinite pretraining loss at {step}")
        loss.backward()
        if not all(
            bool(torch.isfinite(param.grad).all())
            for param in (*encoder.parameters(), *decoder.parameters())
            if param.grad is not None
        ):
            (output / "run.json").write_text(json.dumps({
                "run_id": run_id, "stage": "pretrain",
                "status": "numerical_failure", "step": step,
                "config": config, "input_sha256": input_hash,
            }, indent=2) + "\n")
            raise FloatingPointError(
                f"Nonfinite pretraining gradient at {step}"
            )
        optimizer.step()

        validation_loss = None
        if step % eval_every == 0 or step == steps:
            encoder.eval()
            decoder.eval()
            with torch.no_grad():
                embeddings = encoder(graph)
                relation_losses = []
                for relation, pairs, negatives in validation_examples:
                    sources = pairs[:, 0].to(device)
                    targets = pairs[:, 1].to(device)
                    negative_targets = negatives.to(device)

                    positive_scores = decoder.score(
                        embeddings[sources], embeddings[targets], relation
                    )
                    negative_scores = decoder.score(
                        embeddings[sources], embeddings[negative_targets],
                        relation,
                    )
                    relation_loss = (
                        F.softplus(-positive_scores)
                        + F.softplus(negative_scores)
                    ).mean()
                    relation_losses.append(relation_loss)
                    validation_history.append((
                        step, *graph.relations[relation], len(pairs),
                        float(relation_loss),
                    ))

                if relation_losses:
                    # mean relation loss, weighted by sqrt(training edges)
                    weighted_losses = torch.stack(relation_losses)
                    validation_loss = float(
                        (weighted_losses * validation_weights).sum()
                        / validation_weights.sum()
                    )

            if (
                validation_loss is not None
                and not math.isfinite(validation_loss)
            ):
                (output / "run.json").write_text(json.dumps({
                    "run_id": run_id, "stage": "pretrain",
                    "status": "numerical_failure", "step": step,
                    "config": config, "input_sha256": input_hash,
                }, indent=2) + "\n")
                raise FloatingPointError(
                    f"Nonfinite reconstruction validation loss at {step}"
                )
            if validation_loss is not None:
                if loss_progress(
                    validation_loss, progress_loss, loss_min_delta
                ):
                    progress_loss = validation_loss
                    stale_checks = 0
                else:
                    stale_checks += 1
            if (
                validation_loss is not None
                and validation_loss < best_validation_loss
            ):
                best_validation_loss, best_step = validation_loss, step
                best_encoder_state = cpu_state(encoder)
                save_checkpoint(output / "model.pt", {
                    "encoder": best_encoder_state,
                    "nodes": len(graph.nodes),
                    "relations": graph.relations,
                    "width": width,
                    "graph_inputs": prepared_data["graph_inputs"],
                })

        loss_value = float(loss.detach())
        history.append((step, loss_value, validation_loss))
        event = {
            "run_id": run_id, "stage": "pretrain", "event": "train",
            "step": step, "loss": loss_value,
        }
        events.append(event)
        append_event(output, event)
        if validation_loss is not None:
            event = {
                "run_id": run_id, "stage": "pretrain", "event": "eval",
                "step": step, "loss": validation_loss,
                "best_step": best_step, "stale_checks": stale_checks,
            }
            events.append(event)
            append_event(output, event)

        if bucket_store is not None and best_step > last_published_best_step:
            if best_step == step:
                bucket_store.publish_selected(output)
            last_published_best_step = best_step
        if step % 500 == 0:
            milestone = output / "milestones" / f"s{step:04d}.pt"
            save_checkpoint(milestone, {
                "encoder": cpu_state(encoder),
                "nodes": len(graph.nodes),
                "relations": graph.relations,
                "width": width,
                "graph_inputs": prepared_data["graph_inputs"],
            })
            if bucket_store is not None:
                bucket_store.publish_milestone(milestone)

        should_stop = (
            validation_loss is not None and stale_checks >= patience
        )
        if should_stop:
            termination = "early_stopping"
        if step % checkpoint_every == 0 or step == last_step or should_stop:
            save_snapshot(output, "pretrain", step, {
                "format": 1,
                "stage": "pretrain",
                "run_id": run_id,
                "position": step,
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "rng": rng_state(),
                "history": history,
                "validation_history": validation_history,
                "events": events,
                "best_step": best_step,
                "best_validation_loss": best_validation_loss,
                "best_state": best_encoder_state,
                "progress_loss": progress_loss,
                "stale_checks": stale_checks,
                "config": config,
                "input_sha256": input_hash,
                "graph_inputs": prepared_data["graph_inputs"],
            })
            if bucket_store is not None:
                bucket_store.publish_snapshot(
                    output, force=(step == last_step or should_stop),
                )

        if step % eval_every == 0 or step == 1 or step == steps:
            if device.startswith("cuda"):
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            print(
                f"step   {step:>5}/{steps:<5} loss {loss_value:.5f}  "
                f"time {elapsed:.2f}s", flush=True,
            )

        if validation_loss is not None:
            print(
                f"eval   {step:>5}/{steps:<5} loss {validation_loss:.5f}  "
                f"best {best_validation_loss:.5f} ({best_step})", flush=True,
            )
        completed_step = step
        if should_stop:
            break

    if termination == "manual_stop":
        (output / "run.json").write_text(json.dumps({
            "run_id": run_id, "stage": "pretrain", "status": termination,
            "step": completed_step, "best_step": best_step,
            "config": config, "input_sha256": input_hash,
        }, indent=2) + "\n")
        return {
            "step": completed_step,
            "best_step": best_step,
            "status": termination,
        }

    if best_encoder_state is None:
        raise RuntimeError("Pretraining completed without a selected encoder")

    with (output / "loss.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("step", "loss", "validation_loss"))
        writer.writerows(history)

    with (output / "validation.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow((
            "step", "source_type", "relation", "target_type", "pairs", "loss"
        ))
        writer.writerows(validation_history)

    result = {
        "run_id": run_id,
        "stage": "pretrain",
        "status": "completed",
        "termination": termination,
        "completed_steps": completed_step,
        "stale_checks": stale_checks,
        "config": config,
        "graph_inputs": prepared_data["graph_inputs"],
        "seed": seed,
        "steps": steps,
        "width": width,
        "learning_rate": learning_rate,
        "optimizer": "AdamW",
        "weight_decay": weight_decay,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "foreach": False,
        "patience": patience,
        "loss_min_delta": loss_min_delta,
        "batch_size": batch_size,
        "relation_sampling_power": 0.5,
        "device": device,
        "torch": str(torch.__version__),
        "loss": "binary_logistic",
        "negative_ratio": 1,
        "input_sha256": input_hash,
        "initial_loss": history[0][1],
        "final_loss": history[-1][1],
        "best_step": best_step,
        "best_validation_loss": best_validation_loss if best_step else None,
    }
    (output / "run.json").write_text(json.dumps(result, indent=2) + "\n")
    if bucket_store is not None:
        bucket_store.publish_final(output)
    return result
