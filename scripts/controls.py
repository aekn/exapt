import argparse
import json
from pathlib import Path

from exapt.baseline import drug_popularity, protein_target_paths
from exapt.benchmark import (
    benchmark_record, load_therapeutics, split_diseases, training_folds,
)
from exapt.graph import load_graph, message_graph


def _rows(result, covered=None):
    return [
        {
            "disease": row.disease,
            "positives": row.positives,
            "recall": row.recall,
            "ndcg": row.ndcg,
            "reciprocal_rank": row.reciprocal_rank,
            **({"has_nonzero_path": row.disease in covered} if covered is not None else {}),
        }
        for row in result.per_disease
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate training-only popularity and biological path controls.",
        allow_abbrev=False,
    )
    parser.add_argument("source", type=Path, help="verified PrimeKG source directory")
    parser.add_argument("--benchmark-seed", required=True, type=int)
    parser.add_argument("--folds", required=True, type=int)
    parser.add_argument("--development-fraction", required=True, type=float)
    parser.add_argument("--test-fraction", required=True, type=float)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--damping", type=float, default=0.5)
    args = parser.parse_args()

    try:
        facts = load_therapeutics(args.source)
        split = split_diseases(
            facts, seed=args.benchmark_seed,
            development_fraction=args.development_fraction,
            test_fraction=args.test_fraction,
        )
        folds = training_folds(facts, split, folds=args.folds, seed=args.benchmark_seed)
        graph = load_graph(
            args.source, withheld_diseases=frozenset(), masked_pairs=frozenset(),
        )
        manifest = json.loads((args.source / "manifest.json").read_text())
        benchmark = benchmark_record(
            split, folds, graph.relations, facts.drugs, manifest["sha256"],
        )
        popularity, drugs, counts = drug_popularity(facts, split, k=args.k)

        # remove every therapeutic edge, not just development and test edges
        nontherapeutic = message_graph(
            graph,
            withheld_diseases=frozenset(
                index for index, kind in enumerate(graph.node_types) if kind == "disease"
            ),
            masked_pairs=frozenset(),
        )
        paths, covered = protein_target_paths(
            facts, split, nontherapeutic, k=args.k, damping=args.damping,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))

    def metrics(result):
        return {
            "macro_recall": result.recall,
            "macro_ndcg": result.ndcg,
            "macro_reciprocal_rank": result.reciprocal_rank,
        }

    record = {
        "benchmark": benchmark,
        "setting": {
            "benchmark_seed": args.benchmark_seed,
            "development_fraction": args.development_fraction,
            "test_fraction": args.test_fraction,
            "folds": args.folds,
            "k": args.k,
        },
        "data": {
            "drugs": len(drugs),
            "training_diseases": len(split.train),
            "supervised_training_diseases": len({
                disease for drug, disease in facts.unambiguous_indications
                if disease in split.train
            }),
            "development_diseases": len(split.development),
            "test_diseases": len(split.test),
            "training_indications": sum(counts),
            "development_indications": sum(row.positives for row in popularity.per_disease),
        },
        "popularity": {
            "method": "training-indication-frequency",
            **metrics(popularity),
            "per_disease": _rows(popularity),
        },
        "protein_paths": {
            "method": "disease-protein-drug-target",
            "damping": args.damping,
            "diseases_with_nonzero_paths": len(covered),
            **metrics(paths),
            "per_disease": _rows(paths, covered),
        },
    }
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
