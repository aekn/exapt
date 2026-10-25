import argparse
import json
import random
import sys
from pathlib import Path
from time import perf_counter


def _log(stage: str, message: str) -> None:
    print(f"{stage:<9} {message}", file=sys.stderr, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure one disease-masked training-fold update on verified PrimeKG.",
        allow_abbrev=False,
    )
    parser.add_argument("root", type=Path, help="verified PrimeKG source directory")

    benchmark = parser.add_argument_group("benchmark")
    benchmark.add_argument("--benchmark-seed", required=True, type=int)
    benchmark.add_argument("--folds", required=True, type=int)
    benchmark.add_argument("--fold-index", type=int, default=0)
    benchmark.add_argument("--development-fraction", required=True, type=float)
    benchmark.add_argument("--test-fraction", required=True, type=float)

    model = parser.add_argument_group("model and graph")
    model.add_argument("--encoder", required=True, choices=("rgcn", "compgcn"))
    model.add_argument("--width", required=True, type=int)
    model.add_argument(
        "--normalization", required=True, choices=("relation", "total", "directional"),
    )
    model.add_argument("--layout", required=True, choices=("coo", "csr"))
    model.add_argument("--features", required=True, choices=("trainable", "fixed"))

    optimization = parser.add_argument_group("optimization")
    optimization.add_argument("--seed", required=True, type=int)
    optimization.add_argument("--negatives", required=True, type=int)
    optimization.add_argument("--negative-weight", required=True, type=float)
    optimization.add_argument("--learning-rate", required=True, type=float)
    optimization.add_argument("--weight-decay", required=True, type=float)

    execution = parser.add_argument_group("execution")
    execution.add_argument("--device", required=True, help="PyTorch device: cpu, cuda[:N], or mps")
    execution.add_argument(
        "--full-residency", action="store_true",
        help="keep all training-fold and development adjacencies resident during the update",
    )
    args = parser.parse_args()

    # --help is available without loading PyTorch or project modules
    import torch

    from exapt.benchmark import load_therapeutics, split_diseases, training_folds
    from exapt.compgcn import CompositionalAdjacency, CompGCNEncoder, compgcn_adjacency
    from exapt.distmult import DistMultDecoder
    from exapt.graph import Graph, load_graph, message_graph
    from exapt.model import IndicationModel, NodeEmbeddings
    from exapt.objective import sampled_bce_loss
    from exapt.rgcn import RGCNEncoder, RelationalAdjacency, rgcn_adjacency
    from exapt.supervision import training_supervision
    from exapt.train import FitConfig, _FoldContext, _indices, _sample_columns, _sampled_scores

    config = FitConfig(
        seed=args.seed, benchmark_seed=args.benchmark_seed,
        epochs=1, folds=args.folds,
        width=args.width, negatives=args.negatives,
        negative_weight=args.negative_weight,
        learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        development_fraction=args.development_fraction,
        test_fraction=args.test_fraction,
        encoder=args.encoder, normalization=args.normalization,
        layout=args.layout, trainable_features=args.features == "trainable",
    )
    try:
        config.validate()
    except ValueError as error:
        parser.error(str(error))
    if not 0 <= args.fold_index < config.folds:
        parser.error("--fold-index must be between 0 and --folds - 1")

    try:
        device = torch.device(args.device)
    except (RuntimeError, ValueError) as error:
        parser.error(f"invalid device: {error}")

    if device.type == "cuda":
        if not torch.cuda.is_available():
            parser.error("CUDA unavailable")
        if device.index is not None and device.index >= torch.cuda.device_count():
            parser.error(f"CUDA device {device.index} unavailable")
    elif device.type == "mps":
        if not torch.backends.mps.is_available():
            parser.error("MPS unavailable")
        if device.index is not None and device.index >= torch.mps.device_count():
            parser.error(f"MPS device {device.index} unavailable")
    elif device.type == "cpu":
        if device.index not in (None, 0):
            parser.error(f"unsupported CPU device index {device.index}")
    else:
        parser.error(f"unsupported device: {device} (expected cpu, cuda, or mps)")

    def sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            torch.mps.synchronize()

    _log(
        "preflight",
        f"{config.encoder}  {device}  {config.layout}/{config.normalization}  "
        f"width={config.width}  {args.features}  fold={args.fold_index + 1}/{config.folds}",
    )

    started = perf_counter()
    facts = load_therapeutics(args.root)
    split = split_diseases(
        facts, seed=config.benchmark_seed,
        development_fraction=config.development_fraction,
        test_fraction=config.test_fraction,
    )
    folds = training_folds(facts, split, folds=config.folds, seed=config.benchmark_seed)
    queries = folds[args.fold_index]
    supervision = training_supervision(facts, split, query_diseases=queries)
    source_seconds = perf_counter() - started
    _log(
        "source",
        f"{len(facts.diseases):,} diseases  {len(queries):,} queries  {source_seconds:.2f}s",
    )

    started = perf_counter()
    graph = load_graph(
        args.root, withheld_diseases=frozenset(), masked_pairs=frozenset(),
    )
    graph_seconds = perf_counter() - started
    _log(
        "graph",
        f"{graph.node_count:,} nodes  {graph.edge_count:,} arcs  "
        f"{len(graph.relations):,} relations  {graph_seconds:.2f}s",
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def adjacency_builder(visible: Graph) -> RelationalAdjacency | CompositionalAdjacency:
        if config.encoder == "compgcn":
            return compgcn_adjacency(visible, layout=config.layout, device=device)
        normalization = config.normalization
        if normalization == "directional":
            raise ValueError("preflight: R-GCN requires relation or total normalization")
        return rgcn_adjacency(
            visible, normalization=normalization,
            layout=config.layout, device=device,
        )

    started = perf_counter()
    drugs = tuple(sorted(facts.drugs))
    columns = {drug: index for index, drug in enumerate(drugs)}
    drug_ids = torch.tensor(drugs, dtype=torch.long, device=device)

    contexts: list[_FoldContext] = []
    visible_counts: list[int] = []
    indices = range(config.folds) if args.full_residency else (args.fold_index,)
    for fold_index in indices:
        fold_queries = folds[fold_index]
        fold_supervision = (
            supervision if fold_index == args.fold_index
            else training_supervision(facts, split, query_diseases=fold_queries)
        )
        visible = message_graph(
            graph, withheld_diseases=split.withheld | fold_queries,
            masked_pairs=frozenset(),
        )
        adjacency = adjacency_builder(visible)
        positive_rows, positive_columns = _indices(fold_supervision, columns, device)
        disease_ids = torch.tensor(fold_supervision.diseases, dtype=torch.long, device=device)
        contexts.append(_FoldContext(
            fold_supervision, adjacency, disease_ids, positive_rows, positive_columns,
        ))
        visible_counts.append(visible.edge_count)

    if args.full_residency:
        development_graph = adjacency_builder(message_graph(
            graph, withheld_diseases=split.withheld, masked_pairs=frozenset(),
        ))
        context = contexts[args.fold_index]
        visible_edges = visible_counts[args.fold_index]
    else:
        development_graph = None
        context = contexts[0]
        visible_edges = visible_counts[0]

    torch.manual_seed(config.seed)
    embeddings = NodeEmbeddings(
        graph.node_count, config.width, trainable=config.trainable_features,
        generator=torch.Generator().manual_seed(config.seed),
    )
    encoder = (
        CompGCNEncoder(graph.relations, config.width, config.width, config.width)
        if config.encoder == "compgcn"
        else RGCNEncoder(graph.relations, config.width, config.width, config.width)
    )
    model = IndicationModel(embeddings, encoder, DistMultDecoder(config.width)).to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate, weight_decay=config.weight_decay,
    )
    parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]

    sync()
    setup_seconds = perf_counter() - started
    _log("setup", f"{setup_seconds:.2f}s  {visible_edges:,} visible arcs  "
         f"{len(contexts) + (development_graph is not None)} resident graphs")
    allocated_after_setup = (
        torch.cuda.memory_allocated(device) if device.type == "cuda" else None
    )

    started = perf_counter()
    sampled = _sample_columns(
        supervision, columns, random.Random(config.seed), config.negatives, device,
    )
    sync()
    sampling_seconds = perf_counter() - started

    model.train()
    started = perf_counter()
    positive_scores, unjudged_scores = _sampled_scores(model, context, sampled, drug_ids)
    loss = sampled_bce_loss(
        positive_scores, context.positive_rows, unjudged_scores,
        negative_weight=config.negative_weight,
    )
    sync()
    forward_seconds = perf_counter() - started
    if not torch.isfinite(loss).item():
        raise RuntimeError("preflight: nonfinite loss")
    loss_value = loss.item()

    started = perf_counter()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    sync()
    backward_seconds = perf_counter() - started

    started = perf_counter()
    for name, parameter in parameters:
        if parameter.grad is None or not torch.isfinite(parameter.grad).all().item():
            raise RuntimeError(f"preflight: invalid gradient for {name}")
    gradient_check_seconds = perf_counter() - started

    started = perf_counter()
    optimizer.step()
    sync()
    optimizer_seconds = perf_counter() - started

    development_seconds = None
    if development_graph is not None:
        model.eval()
        started = perf_counter()
        with torch.inference_mode():
            development_ids = torch.tensor(
                sorted(split.development), dtype=torch.long, device=device,
            )
            development_scores = model(development_graph, development_ids, drug_ids)
            if not torch.isfinite(development_scores).all().item():
                raise RuntimeError("preflight: nonfinite development scores")
        sync()
        development_seconds = perf_counter() - started
        _log("evaluate", f"{development_seconds:.2f}s  full drug catalog")

    _log(
        "update",
        f"loss {loss_value:.4f}  sample {sampling_seconds:.2f}s  "
        f"forward {forward_seconds:.2f}s  backward {backward_seconds:.2f}s  "
        f"checks {gradient_check_seconds:.2f}s  optimizer {optimizer_seconds:.2f}s",
    )

    manifest = json.loads((args.root / "manifest.json").read_text(encoding="utf-8"))
    result = {
        "scope": (
            "all training folds plus development; one masked training update; "
            "excludes checkpoints" if args.full_residency else
            "one masked training fold; excludes other folds, development, and checkpoints"
        ),
        "full_residency": args.full_residency,
        "resident_graphs": len(contexts) + (development_graph is not None),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "source_sha256": manifest["sha256"],
        "torch": str(torch.__version__),
        "encoder": config.encoder,
        "normalization": config.normalization,
        "benchmark_seed": config.benchmark_seed,
        "layout": config.layout,
        "features": args.features,
        "width": config.width,
        "seed": config.seed,
        "folds": config.folds,
        "fold_index": args.fold_index,
        "development_fraction": config.development_fraction,
        "test_fraction": config.test_fraction,
        "negatives": config.negatives,
        "negative_weight": config.negative_weight,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "query_diseases": len(supervision.diseases),
        "positive_pairs": sum(map(len, supervision.positives)),
        "sampled_pairs": len(supervision.diseases) * config.negatives,
        "nodes": graph.node_count,
        "edges": graph.edge_count,
        "visible_edges": visible_edges,
        "relations": len(graph.relations),
        "source_seconds": round(source_seconds, 3),
        "graph_seconds": round(graph_seconds, 3),
        "setup_seconds": round(setup_seconds, 3),
        "sampling_seconds": round(sampling_seconds, 3),
        "forward_seconds": round(forward_seconds, 3),
        "backward_seconds": round(backward_seconds, 3),
        "gradient_check_seconds": round(gradient_check_seconds, 3),
        "optimizer_seconds": round(optimizer_seconds, 3),
        "development_seconds": (
            round(development_seconds, 3) if development_seconds is not None else None
        ),
        "train_loss": loss_value,
        "cuda_allocated_after_setup_bytes": allocated_after_setup,
        "cuda_peak_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
        "cuda_peak_reserved_bytes": (
            torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
        ),
        "mps_allocated_after_step_bytes": (
            torch.mps.current_allocated_memory() if device.type == "mps" else None
        ),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
