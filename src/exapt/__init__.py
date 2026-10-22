import argparse
import json
from pathlib import Path

from .data import audit_graph, prepare_indications
from .split import split_indications, split_therapy


def main() -> None:
    parser = argparse.ArgumentParser(prog="exapt")
    commands = parser.add_subparsers(dest="command", required=True)

    audit = commands.add_parser("audit")
    audit.add_argument("graph", type=Path)

    prepare = commands.add_parser("prepare")
    prepare.add_argument("graph", type=Path)
    prepare.add_argument("output", type=Path)

    split = commands.add_parser("split")
    split.add_argument("indications", type=Path)
    split.add_argument("graph", type=Path)
    split.add_argument("output", type=Path)

    therapy = commands.add_parser("therapy")
    therapy.add_argument("labels", type=Path)
    therapy.add_argument("split", type=Path)

    graph = commands.add_parser("graph")
    graph.add_argument("source", type=Path)
    graph.add_argument("output", type=Path)

    smoke = commands.add_parser("smoke")
    smoke.add_argument("dataset", type=Path)
    smoke.add_argument("output", type=Path)
    smoke.add_argument("--epochs", type=int, default=20)
    smoke.add_argument("--device", default="cpu")

    train = commands.add_parser("train")
    train.add_argument("graph", type=Path)
    train.add_argument("dataset", type=Path)
    train.add_argument("output", type=Path)
    train.add_argument("--epochs", type=int, default=500)
    train.add_argument("--width", type=int, default=32)
    train.add_argument("--learning-rate", type=float, default=5e-4)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--patience", type=int, default=15)
    train.add_argument("--min-epochs", type=int, default=50)
    train.add_argument("--map-min-delta", type=float, default=1e-4)
    train.add_argument("--negatives", type=int, default=8)
    train.add_argument("--eval-every", type=int, default=5)
    train.add_argument("--device", default="cpu")

    train.add_argument("--pretrained", type=Path)
    train.add_argument("--prototype-k", type=int)

    train.add_argument("--resume", type=Path)
    train.add_argument("--stop-after", type=int)
    train.add_argument("--run-id", default="local")
    train.add_argument("--bucket")
    train.add_argument("--allow-public-bucket", action="store_true")
    train.add_argument("--source-commit")
    train.add_argument("--dataset-revision")
    train.add_argument("--manifest-sha256")
    train.add_argument("--remote-seconds", type=float, default=900)

    pretrain_data = commands.add_parser("pretrain-data")
    pretrain_data.add_argument("graph", type=Path)
    pretrain_data.add_argument("output", type=Path)

    pretrain = commands.add_parser("pretrain")
    pretrain.add_argument("data", type=Path)
    pretrain.add_argument("output", type=Path)
    pretrain.add_argument("--steps", type=int, default=2000)
    pretrain.add_argument("--width", type=int, default=32)
    pretrain.add_argument("--batch-size", type=int, default=8192)
    pretrain.add_argument("--eval-every", type=int, default=50)
    pretrain.add_argument("--weight-decay", type=float, default=1e-4)
    pretrain.add_argument("--patience", type=int, default=8)
    pretrain.add_argument("--loss-min-delta", type=float, default=1e-4)
    pretrain.add_argument("--device", default="cpu")

    pretrain.add_argument("--resume", type=Path)
    pretrain.add_argument("--checkpoint-every", type=int, default=25)
    pretrain.add_argument("--stop-after", type=int)
    pretrain.add_argument("--run-id", default="local")
    pretrain.add_argument("--bucket")
    pretrain.add_argument("--allow-public-bucket", action="store_true")
    pretrain.add_argument("--source-commit")
    pretrain.add_argument("--dataset-revision")
    pretrain.add_argument("--manifest-sha256")
    pretrain.add_argument("--remote-seconds", type=float, default=900)

    preflight = commands.add_parser("preflight")
    preflight_commands = preflight.add_subparsers(
        dest="check", required=True,
    )

    overfit = preflight_commands.add_parser("overfit")
    overfit.add_argument("dataset", type=Path)
    overfit.add_argument("--width", type=int, default=32)
    overfit.add_argument("--steps", type=int, default=300)
    overfit.add_argument("--learning-rate", type=float, default=0.003)
    overfit.add_argument("--device", default="cpu")
    overfit.add_argument("--prototype-k", type=int)

    profile_fit = preflight_commands.add_parser("profile-fit")
    profile_fit.add_argument("graph", type=Path)
    profile_fit.add_argument("dataset", type=Path)
    profile_fit.add_argument("output", type=Path)
    profile_fit.add_argument("--width", type=int, default=128)
    profile_fit.add_argument("--device", default="cuda")
    profile_fit.add_argument("--prototype-k", type=int)

    profile_pretrain = preflight_commands.add_parser("profile-pretrain")
    profile_pretrain.add_argument("data", type=Path)
    profile_pretrain.add_argument("output", type=Path)
    profile_pretrain.add_argument("--width", type=int, default=128)
    profile_pretrain.add_argument("--device", default="cuda")

    hub = commands.add_parser("hub")
    hub_commands = hub.add_subparsers(dest="action", required=True)

    manifest = hub_commands.add_parser("manifest")
    manifest.add_argument("directory", type=Path)

    download = hub_commands.add_parser("download")
    download.add_argument("repo_id")
    download.add_argument("revision")
    download.add_argument("directory", type=Path)

    recover = hub_commands.add_parser("recover")
    recover.add_argument("bucket")
    recover.add_argument("run_id")
    recover.add_argument("stage", choices=("pretrain", "fit"))
    recover.add_argument("directory", type=Path)
    recover.add_argument("--allow-public-bucket", action="store_true")

    selected = hub_commands.add_parser("selected")
    selected.add_argument("bucket")
    selected.add_argument("run_id")
    selected.add_argument("stage", choices=("pretrain", "fit"))
    selected.add_argument("output", type=Path)
    selected.add_argument("--allow-public-bucket", action="store_true")

    args = parser.parse_args()

    try:
        match args.command:
            case "audit":
                result = audit_graph(args.graph)
            case "prepare":
                result = prepare_indications(args.graph, args.output)
            case "split":
                result = split_indications(
                    args.indications, args.graph, args.output
                )
            case "therapy":
                result = split_therapy(args.labels, args.split)
            case "graph":
                from .graph import prepare_graph

                result = prepare_graph(args.source, args.output)
            case "smoke":
                from .train import train_smoke

                result = train_smoke(
                    args.dataset, args.output,
                    epochs=args.epochs, device=args.device,
                )
            case "train":
                from .train import train_model

                result = train_model(
                    args.graph, args.dataset, args.output,
                    epochs=args.epochs, width=args.width,
                    learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay, patience=args.patience,
                    min_epochs=args.min_epochs,
                    map_min_delta=args.map_min_delta,
                    negatives=args.negatives, eval_every=args.eval_every,
                    device=args.device,
                    pretrained=args.pretrained, prototype_k=args.prototype_k,
                    resume=args.resume, stop_after=args.stop_after,
                    run_id=args.run_id, bucket=args.bucket,
                    allow_public_bucket=args.allow_public_bucket,
                    source_commit=args.source_commit,
                    dataset_revision=args.dataset_revision,
                    manifest_sha256=args.manifest_sha256,
                    remote_seconds=args.remote_seconds,
                )
            case "pretrain-data":
                from .pretrain import prepare_pretrain

                result = prepare_pretrain(args.graph, args.output)
            case "pretrain":
                from .pretrain import pretrain_model

                result = pretrain_model(
                    args.data, args.output,
                    steps=args.steps, width=args.width,
                    batch_size=args.batch_size, eval_every=args.eval_every,
                    weight_decay=args.weight_decay, patience=args.patience,
                    loss_min_delta=args.loss_min_delta, device=args.device,
                    resume=args.resume, checkpoint_every=args.checkpoint_every,
                    stop_after=args.stop_after, run_id=args.run_id,
                    bucket=args.bucket,
                    allow_public_bucket=args.allow_public_bucket,
                    source_commit=args.source_commit,
                    dataset_revision=args.dataset_revision,
                    manifest_sha256=args.manifest_sha256,
                    remote_seconds=args.remote_seconds,
                )
            case "preflight":
                from .preflight import overfit, profile

                match args.check:
                    case "overfit":
                        result = overfit(
                            args.dataset, width=args.width,
                            steps=args.steps,
                            learning_rate=args.learning_rate,
                            device=args.device,
                            prototype_k=args.prototype_k,
                        )
                    case "profile-fit":
                        result = profile(
                            "fit", args.graph, args.output,
                            dataset=args.dataset, width=args.width,
                            device=args.device, prototype_k=args.prototype_k,
                        )
                    case "profile-pretrain":
                        result = profile(
                            "pretrain", args.data, args.output,
                            width=args.width, device=args.device,
                        )
                    case _:
                        parser.error(f"Unknown preflight: {args.check}")
            case "hub":
                from .hub import (
                    BucketStore, download_dataset, verify_manifest,
                    write_manifest,
                )

                match args.action:
                    case "manifest":
                        path = write_manifest(args.directory)
                        result = {
                            "manifest": str(path),
                            "files": verify_manifest(args.directory),
                        }
                    case "download":
                        result = download_dataset(
                            args.repo_id, args.revision, args.directory,
                        )
                    case "recover":
                        store = BucketStore(
                            args.bucket, args.run_id, args.stage,
                            allow_public=args.allow_public_bucket,
                        )
                        path = store.recover(args.directory)
                        result = {"resume": str(path)}
                    case "selected":
                        store = BucketStore(
                            args.bucket, args.run_id, args.stage,
                            allow_public=args.allow_public_bucket,
                        )
                        path = store.download_selected(args.output)
                        result = {"selected": str(path)}
                    case _:
                        parser.error(f"Unknown hub action: {args.action}")
            case _:
                parser.error(f"Unknown command: {args.command}")

    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    print(json.dumps(result, indent=2))
