import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .data import THERAPEUTIC_RELATIONS, sha256_file


SEED = 163
SMOKE_TRAIN_DISEASES = 36
SMOKE_DEV_DISEASES = 12
SMOKE_DRUGS = 512
SMOKE_NEIGHBORS = 12

PARTITIONS = ("train", "development", "final")
PAIR_FIELDS = (
    "drug_index", "disease_index", "drug_id", "disease_id",
    "contraindication", "off_label",
)
GRAPH_FIELDS = (
    "relation", "x_index", "x_type", "x_id",
    "y_index", "y_type", "y_id",
)


def _stratum(indications: int, protein: bool) -> tuple[int, bool]:
    if indications == 1:
        frequency = 1
    elif indications < 5:
        frequency = 2
    else:
        frequency = 3
    return frequency, protein


def _partition(
    diseases: set[int], counts: Counter[int], protein: set[int], seed: int
) -> dict[int, str]:
    # (indication frequency, protein coverage) -> disease indices
    groups: dict[tuple[int, bool], list[int]] = defaultdict(list)
    for disease in diseases:
        groups[_stratum(counts[disease], disease in protein)].append(disease)

    rng = random.Random(seed)
    assignment: dict[int, str] = {}
    for stratum in sorted(groups):
        members = sorted(groups[stratum])
        rng.shuffle(members)

        # round within each stratum, not across all diseases
        n_dev = round(len(members) * 0.1)
        n_final = round(len(members) * 0.1)
        for index, disease in enumerate(members):
            if index < n_dev:
                assignment[disease] = "development"
            elif index < n_dev + n_final:
                assignment[disease] = "final"
            else:
                assignment[disease] = "train"

    return assignment


def _write_csv(path: Path, header: tuple[str, ...], rows: list[tuple]) -> str:
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.writer(target, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)
    return sha256_file(path)


def _smoke_diseases(
    assignment: dict[int, str], counts: Counter[int],
    protein: set[int], seed: int,
) -> dict[str, set[int]]:
    rng = random.Random(seed + 1)
    chosen: dict[str, set[int]] = {}

    for name, limit in (
        ("train", SMOKE_TRAIN_DISEASES),
        ("development", SMOKE_DEV_DISEASES),
    ):
        groups: dict[tuple[int, bool], list[int]] = defaultdict(list)
        for disease, partition in assignment.items():
            if partition == name:
                stratum = _stratum(counts[disease], disease in protein)
                groups[stratum].append(disease)

        for members in groups.values():
            members.sort()
            rng.shuffle(members)

        # sample strata in round-robin order to preserve coverage
        selected: set[int] = set()
        strata = sorted(groups)
        while len(selected) < limit and any(groups.values()):
            for stratum in strata:
                if groups[stratum] and len(selected) < limit:
                    selected.add(groups[stratum].pop())
        chosen[name] = selected

    return chosen


def _smoke_graph(
    graph: Path, output: Path, diseases: dict[str, set[int]],
    positives: list[tuple[int, int]], drugs: dict[int, str],
    disease_ids: dict[int, str], seed: int,
) -> dict[str, Any]:
    selected_diseases = diseases["train"] | diseases["development"]
    selected_drugs = {
        drug for drug, disease in positives if disease in selected_diseases
    }
    candidates = sorted(set(drugs) - selected_drugs)
    rng = random.Random(seed + 2)
    rng.shuffle(candidates)
    remaining = max(0, SMOKE_DRUGS - len(selected_drugs))
    selected_drugs.update(candidates[:remaining])
    anchors = selected_diseases | selected_drugs

    # reservoir per (anchor, relation), counting edge occurrences
    neighbors: dict[tuple[int, str], list[int]] = defaultdict(list)
    seen_edges: Counter[tuple[int, str]] = Counter()

    with graph.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.reader(source)
        header = next(reader)
        (
            rel_col, src_col, src_type_col, src_id_col,
            dst_col, dst_type_col, dst_id_col,
        ) = (header.index(field) for field in GRAPH_FIELDS)

        for row in reader:
            relation = row[rel_col]
            if relation in THERAPEUTIC_RELATIONS:
                continue

            src, dst = int(row[src_col]), int(row[dst_col])
            for anchor, other in ((src, dst), (dst, src)):
                if anchor not in anchors:
                    continue

                key = (anchor, relation)
                seen_edges[key] += 1
                sample = neighbors[key]
                if len(sample) < SMOKE_NEIGHBORS:
                    sample.append(other)
                else:
                    index = rng.randrange(seen_edges[key])
                    if index < SMOKE_NEIGHBORS:
                        sample[index] = other

    chosen_neighbors = {key: set(values) for key, values in neighbors.items()}
    output.mkdir(parents=True, exist_ok=True)
    edges = output / "graph.csv"
    nodes: dict[int, tuple[str, str]] = {}
    edge_counts: Counter[str] = Counter()

    # preserve source row order; either endpoint may select an edge
    with (
        graph.open("r", encoding="utf-8-sig", newline="") as source,
        edges.open("w", encoding="utf-8", newline="") as target,
    ):
        reader = csv.reader(source)
        header = next(reader)
        (
            rel_col, src_col, src_type_col, src_id_col,
            dst_col, dst_type_col, dst_id_col,
        ) = (header.index(field) for field in GRAPH_FIELDS)

        writer = csv.writer(target, lineterminator="\n")
        writer.writerow(("x_index", "x_type", "relation", "y_index", "y_type"))
        for row in reader:
            relation = row[rel_col]
            if relation in THERAPEUTIC_RELATIONS:
                continue

            src, dst = int(row[src_col]), int(row[dst_col])
            # keep row if either endpoint sampled the other
            from_source = dst in chosen_neighbors.get((src, relation), ())
            from_target = src in chosen_neighbors.get((dst, relation), ())
            if not (from_source or from_target):
                continue

            writer.writerow((
                src, row[src_type_col], relation, dst, row[dst_type_col],
            ))
            edge_counts[relation] += 1
            nodes[src] = row[src_type_col], row[src_id_col]
            nodes[dst] = row[dst_type_col], row[dst_id_col]

    for disease in selected_diseases:
        nodes.setdefault(disease, ("disease", disease_ids[disease]))
    for drug in selected_drugs:
        nodes.setdefault(drug, ("drug", drugs[drug]))

    node_hash = _write_csv(
        output / "nodes.csv", ("index", "type", "id"),
        [(node, *nodes[node]) for node in sorted(nodes)],
    )
    drug_hash = _write_csv(
        output / "drugs.csv", ("drug_index", "drug_id"),
        [(drug, drugs[drug]) for drug in sorted(selected_drugs)],
    )

    return {
        "diseases": {name: len(values) for name, values in diseases.items()},
        "candidate_drugs": len(selected_drugs),
        "nodes": len(nodes),
        "directed_edges": sum(edge_counts.values()),
        "relations": dict(sorted(edge_counts.items())),
        "graph_sha256": sha256_file(edges),
        "nodes_sha256": node_hash,
        "drugs_sha256": drug_hash,
        "selection": {
            "seed": seed,
            "max_neighbors_per_anchor_relation": SMOKE_NEIGHBORS,
        },
    }


def split_indications(
    indications: Path, graph: Path, output: Path, seed: int = SEED
) -> dict[str, Any]:
    with indications.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError("No indications to split")

    positive_pairs = [
        (int(row["drug_index"]), int(row["disease_index"])) for row in rows
    ]
    counts = Counter(disease for _, disease in positive_pairs)
    disease_ids = {int(row["disease_index"]): row["disease_id"] for row in rows}

    # distinct diseases cannot share components of a source identifier
    owners: dict[str, int] = {}
    for disease, identifier in disease_ids.items():
        for component in identifier.split("_"):
            previous = owners.setdefault(component, disease)
            if previous != disease:
                raise ValueError("Indication diseases share source ID components")

    protein_diseases: set[int] = set()
    drugs: dict[int, str] = {}
    with graph.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.reader(source)
        header = next(reader)
        (
            rel_col, src_col, src_type_col, src_id_col,
            dst_col, dst_type_col, dst_id_col,
        ) = (header.index(field) for field in GRAPH_FIELDS)

        for row in reader:
            relation = row[rel_col]
            if row[src_type_col] == "drug":
                drugs[int(row[src_col])] = row[src_id_col]
            if row[dst_type_col] == "drug":
                drugs[int(row[dst_col])] = row[dst_id_col]
            if relation == "disease_protein":
                if row[src_type_col] == "disease":
                    protein_diseases.add(int(row[src_col]))
                if row[dst_type_col] == "disease":
                    protein_diseases.add(int(row[dst_col]))

    assignment = _partition(set(counts), counts, protein_diseases, seed)
    output.mkdir(parents=True, exist_ok=True)

    # keep input row order within disease partition
    tables: dict[str, list[tuple]] = {name: [] for name in PARTITIONS}
    for row in rows:
        disease = int(row["disease_index"])
        record = tuple(row[field] for field in PAIR_FIELDS)
        tables[assignment[disease]].append(record)

    hashes = {}
    for name, records in tables.items():
        hashes[name] = _write_csv(output / f"{name}.csv", PAIR_FIELDS, records)
    hashes["drugs"] = _write_csv(
        output / "drugs.csv", ("drug_index", "drug_id"),
        [(drug, drugs[drug]) for drug in sorted(drugs)],
    )

    smoke_diseases = _smoke_diseases(assignment, counts, protein_diseases, seed)
    smoke_dir = output / "smoke"
    smoke_dir.mkdir(exist_ok=True)
    for name, selected in smoke_diseases.items():
        records = [
            tuple(row[key] for key in PAIR_FIELDS)
            for row in rows if int(row["disease_index"]) in selected
        ]
        hashes[f"smoke/{name}"] = _write_csv(
            smoke_dir / f"{name}.csv", PAIR_FIELDS, records,
        )

    smoke_report = _smoke_graph(
        graph, smoke_dir, smoke_diseases,
        positive_pairs, drugs, disease_ids, seed,
    )
    strata = Counter(
        f"{_stratum(counts[disease], disease in protein_diseases)[0]}/"
        f"{'protein' if disease in protein_diseases else 'no_protein'}/"
        f"{assignment[disease]}"
        for disease in assignment
    )

    report: dict[str, Any] = {
        "seed": seed,
        "source_sha256": sha256_file(graph),
        "indications_sha256": sha256_file(indications),
        "partition": {
            name: {
                "diseases": sum(value == name for value in assignment.values()),
                "indications": len(tables[name]),
            }
            for name in tables
        },
        "candidate_drugs": len(drugs),
        "biology_strata": dict(sorted(strata.items())),
        "hashes": hashes,
        "smoke": smoke_report,
    }
    (output / "split.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def split_therapy(therapy: Path, split: Path) -> dict[str, Any]:
    def read_indices(path: Path, field: str) -> set[int]:
        with path.open(newline="") as stream:
            return {int(row[field]) for row in csv.DictReader(stream)}

    train_diseases = read_indices(split / "train.csv", "disease_index")
    dev_diseases = read_indices(split / "development.csv", "disease_index")
    final_diseases = read_indices(split / "final.csv", "disease_index")
    if (
        (train_diseases & dev_diseases)
        or (train_diseases & final_diseases)
        or (dev_diseases & final_diseases)
    ):
        raise ValueError("Disease partitions overlap")

    smoke_dir = split / "smoke"
    smoke_train_diseases = read_indices(smoke_dir / "train.csv", "disease_index")
    smoke_drugs = read_indices(smoke_dir / "drugs.csv", "drug_index")
    fields = (
        "drug_index", "disease_index", "relation", "drug_id", "disease_id"
    )
    train_rows: list[tuple[str, ...]] = []
    dev_rows: list[tuple[str, ...]] = []
    smoke_rows: list[tuple[str, ...]] = []

    # final diseases excluded from therapeutic label exports
    with therapy.open(newline="") as stream:
        for row in csv.DictReader(stream):
            disease = int(row["disease_index"])
            if disease not in train_diseases and disease not in dev_diseases:
                continue
            record = tuple(row[field] for field in fields)
            if disease in dev_diseases:
                dev_rows.append(record)
                continue
            train_rows.append(record)
            if (
                disease in smoke_train_diseases
                and int(row["drug_index"]) in smoke_drugs
            ):
                smoke_rows.append(record)

    train_counts = Counter(row[2] for row in train_rows)
    with (split / "train.csv").open(newline="") as stream:
        frozen_indications = {
            (row["drug_index"], row["disease_index"])
            for row in csv.DictReader(stream)
        }
    therapy_indications = {
        (row[0], row[1]) for row in train_rows if row[2] == "indication"
    }
    if therapy_indications != frozen_indications:
        raise ValueError("Indication labels differ from the frozen split")

    hashes = {
        "train": _write_csv(split / "therapeutic_train.csv", fields, train_rows),
        "development": _write_csv(
            split / "therapeutic_development.csv", fields, dev_rows,
        ),
        "smoke": _write_csv(
            smoke_dir / "therapeutic_train.csv", fields, smoke_rows,
        ),
    }
    report = {
        "source_sha256": sha256_file(therapy),
        "split_sha256": {
            name: sha256_file(split / f"{name}.csv")
            for name in PARTITIONS
        },
        "train": dict(sorted(train_counts.items())),
        "development": dict(sorted(Counter(row[2] for row in dev_rows).items())),
        "smoke": dict(sorted(Counter(row[2] for row in smoke_rows).items())),
        "hashes": hashes,
    }
    (split / "therapy.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
