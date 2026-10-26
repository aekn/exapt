import argparse
import json
import os
import re
import subprocess
import sys
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any


def _log(event: str, detail: str) -> None:
    print(f"{event:<8} {detail}", file=sys.stderr, flush=True)


def _clock(seconds: float) -> str:
    hours, remaining = divmod(round(seconds), 3600)
    minutes, seconds = divmod(remaining, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit a disease-masked relational encoder on verified PrimeKG.", allow_abbrev=False)
    parser.add_argument("source", type=Path, help="verified PrimeKG source directory")
    parser.add_argument("output", type=Path, help="experiment directory (new unless --resume)")

    benchmark = parser.add_argument_group("benchmark")
    benchmark.add_argument("--benchmark-seed", required=True, type=int)
    benchmark.add_argument("--folds", required=True, type=int)
    benchmark.add_argument("--development-fraction", required=True, type=float)
    benchmark.add_argument("--test-fraction", required=True, type=float)
    benchmark.add_argument("--k", type=int, default=20)

    model = parser.add_argument_group("model and graph")
    model.add_argument("--encoder", required=True, choices=("rgcn", "compgcn"))
    model.add_argument("--width", required=True, type=int)
    model.add_argument("--normalization", required=True, choices=("relation", "total", "directional"))
    model.add_argument("--layout", required=True, choices=("coo", "csr"))
    model.add_argument("--features", required=True, choices=("trainable", "fixed"))
    model.add_argument("--dropout", type=float, default=0.0)

    optimization = parser.add_argument_group("optimization")
    optimization.add_argument("--objective", choices=("bce", "bpr", "lsep", "lsep_mean"), default="bce")
    optimization.add_argument("--candidate-pool", choices=("sampled", "full"), default="sampled")
    optimization.add_argument("--scheduler", choices=("constant", "cosine"), default="constant")
    optimization.add_argument("--schedule-epochs", type=int)
    optimization.add_argument("--min-learning-rate", type=float, default=0.0)
    optimization.add_argument("--seed", required=True, type=int, help="initialization and sampling seed")
    optimization.add_argument("--epochs", required=True, type=int)
    optimization.add_argument("--negatives", required=True, type=int)
    optimization.add_argument("--negative-weight", required=True, type=float)
    optimization.add_argument("--learning-rate", required=True, type=float)
    optimization.add_argument("--weight-decay", required=True, type=float)

    execution = parser.add_argument_group("execution")
    execution.add_argument("--device", required=True, help="PyTorch device: cpu, cuda[:N], or mps")
    execution.add_argument("--resume", action="store_true")
    execution.add_argument("--bucket-run", help="publish recoverable epochs to HF Bucket")
    execution.add_argument("--dataset-revision", help="pinned Hugging Face dataset commit")
    execution.add_argument("--purpose", default="", help="brief experiment purpose")
    args = parser.parse_args()

    warnings.filterwarnings(
        "ignore",
        message=r"^Failed to initialize NumPy: No module named 'numpy'",
        category=UserWarning,
        module=r"^torch\._subclasses\.functional_tensor$",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"^Sparse CSR tensor support is in beta state\.",
        category=UserWarning,
        module=r"^exapt\.rgcn$",
    )

    import torch

    from exapt.benchmark import load_therapeutics
    from exapt.graph import load_graph
    from exapt.train import FitConfig, code_fingerprint, fit

    config = FitConfig(
        seed=args.seed, benchmark_seed=args.benchmark_seed,
        epochs=args.epochs, folds=args.folds,
        width=args.width, negatives=args.negatives,
        negative_weight=args.negative_weight,
        learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        development_fraction=args.development_fraction,
        test_fraction=args.test_fraction, k=args.k,
        encoder=args.encoder, normalization=args.normalization,
        layout=args.layout, trainable_features=args.features == "trainable",
        objective=args.objective, candidate_pool=args.candidate_pool,
        scheduler=args.scheduler, schedule_epochs=args.schedule_epochs,
        min_learning_rate=args.min_learning_rate, dropout=args.dropout,
    )
    try:
        config.validate()
    except ValueError as error:
        parser.error(str(error))

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

    try:
        manifest = json.loads((args.source / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        parser.error(f"cannot read source manifest: {error}")

    source_sha256 = manifest.get("sha256") if isinstance(manifest, dict) else None
    if not isinstance(source_sha256, str) or len(source_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in source_sha256
    ):
        parser.error("source manifest has invalid SHA-256")

    bucket = None
    token = None
    recovered_epoch = None
    if args.bucket_run is not None:
        from exapt.storage import BucketRun

        token = os.environ.get("HF_TOKEN")
        if not token:
            parser.error("HF_TOKEN required for Bucket persistence")
        try:
            bucket = BucketRun(args.bucket_run)
            if not args.output.exists():
                previous_epoch = bucket.restore(args.output, token=token)
                if previous_epoch is not None:
                    args.resume = True
                    recovered_epoch = previous_epoch
        except (OSError, RuntimeError, ValueError, KeyError) as error:
            parser.error(f"cannot restore Bucket run: {error}")

    if args.resume:
        if not (args.output / "resume.pt").is_file():
            parser.error("resume checkpoint does not exist")
        try:
            previous = json.loads((args.output / "config.json").read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            parser.error(f"cannot read existing experiment configuration: {error}")
        scientific_config = asdict(config)
        del scientific_config["epochs"]
        if not isinstance(previous, dict) or (
            previous.get("config") != scientific_config
            or previous.get("source_sha256") != source_sha256
        ):
            parser.error("resume configuration or source does not match existing experiment")
    elif args.output.exists():
        parser.error(f"output already exists: {args.output} (use --resume to continue)")

    git_result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, check=False,
    )
    git_sha = git_result.stdout.strip() if git_result.returncode == 0 else None
    if git_sha is not None and re.fullmatch(r"[0-9a-f]{40}", git_sha) is None:
        parser.error("invalid Git revision")
    if args.dataset_revision is not None and re.fullmatch(r"[0-9a-f]{40}", args.dataset_revision) is None:
        parser.error("--dataset-revision must be a full commit SHA")
    run = args.bucket_run or args.output.name
    location = f"exapt-lab/hangar/runs/{args.bucket_run}" if bucket is not None else None
    device_name = str(device)
    if device.type == "cuda":
        device_name += f" / {torch.cuda.get_device_name(device)}"

    print(f"exapt / {run}\n", file=sys.stderr, flush=True)
    _log("model", f"{config.encoder} / distmult")
    _log("train", f"{config.objective} / {config.normalization} / n{config.negatives} / w{config.width}")
    _log("seed", str(config.seed))
    _log("folds", str(config.folds))
    _log("budget", f"{config.epochs} epochs")
    _log("device", device_name)
    _log("local", str(args.output))
    if location is not None:
        _log("bucket", location)
    if recovered_epoch is not None:
        _log("resume", f"epoch {recovered_epoch} / target {config.epochs}")

    print(file=sys.stderr, flush=True)
    _log("source", "load therapeutics")
    facts = load_therapeutics(args.source)
    _log("source", f"ready / {len(facts.diseases):,} diseases")
    _log("graph", "load source arcs")
    graph = load_graph(
        args.source, withheld_diseases=frozenset(), masked_pairs=frozenset(),
    )
    _log("graph", f"ready / {graph.edge_count:,} arcs")
    _log("folds", "prepare masked contexts")

    best_score: tuple[float, float] | None = None
    if args.resume:
        previous_history = torch.load(
            args.output / "resume.pt", map_location="cpu", weights_only=True,
        )["history"]
        best_score = max(
            ((item["development_recall"], item["development_ndcg"]) for item in previous_history),
            default=None,
        )

    fit_total = 0.0
    development_total = 0.0
    header_printed = False
    columns = "{:>5}   {:>8}   {:>8}   {:>7}   {:>7}   {:>8}"

    def report(record: dict[str, Any], fit_seconds: float, development_seconds: float) -> None:
        nonlocal best_score, fit_total, development_total, header_printed
        fit_total += fit_seconds
        development_total += development_seconds
        score = (record["development_recall"], record["development_ndcg"])
        improved = best_score is None or score > best_score
        if improved:
            best_score = score

        if not header_printed:
            print(file=sys.stderr, flush=True)
            print(
                columns.format("epoch", "loss", "lr", f"R@{config.k}", f"nDCG@{config.k}", "wall"),
                file=sys.stderr, flush=True,
            )
            header_printed = True

        print(
            columns.format(
                f"{record['epoch']:04d}",
                f"{record['train_loss']:.4f}",
                f"{record['end_learning_rate']:.2e}",
                f"{record['development_recall']:.4f}",
                f"{record['development_ndcg']:.4f}",
                _clock(fit_seconds + development_seconds),
            ) + ("   *" if improved else ""),
            file=sys.stderr, flush=True,
        )
        if bucket is not None:
            assert token is not None
            bucket.publish(args.output, token=token)

    result = fit(
        facts, graph, config=config, source_sha256=source_sha256,
        output=args.output, device=device, resume=args.resume, on_epoch=report,
        code_sha256=code_fingerprint(), git_sha=git_sha,
        dataset_revision=args.dataset_revision, purpose=args.purpose,
    )
    best = result["best_development"]
    print(file=sys.stderr, flush=True)
    _log(
        "done", f"{result['completed_epochs']} epochs / "
        f"{result['completed_epochs'] * config.folds} updates",
    )
    _log(
        "best", f"e{result['best_epoch']:04d} / "
        f"R@{config.k} {best['development_recall']:.4f} / "
        f"nDCG@{config.k} {best['development_ndcg']:.4f}",
    )
    _log(
        "time", f"fit {_clock(fit_total)} / dev {_clock(development_total)} / "
        f"session {_clock(fit_total + development_total)}",
    )
    if location is not None:
        _log("bucket", location)
    else:
        _log("local", str(args.output))


if __name__ == "__main__":
    main()
