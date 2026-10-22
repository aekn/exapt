import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any


REQUIRED_COLUMNS = ("x_type", "x_id", "relation", "y_type", "y_id")
THERAPEUTIC_RELATIONS = ("indication", "contraindication", "off-label use")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_graph(path: Path) -> dict[str, Any]:
    before = path.stat()
    digest = sha256_file(path)

    relations: Counter[str] = Counter()
    edge_types: Counter[tuple[str, str, str]] = Counter()
    missing: Counter[str] = Counter()

    drug_ids: set[str] = set()
    disease_ids: set[str] = set()
    protein_diseases: set[str] = set()
    phenotype_diseases: set[str] = set()

    # relation: [drug-to-disease, disease-to-drug] row counts
    therapeutic_rows = {name: [0, 0] for name in THERAPEUTIC_RELATIONS}

    # relation: canonical (drug, disease) pairs for each edge direction
    therapeutic_pairs: dict[
        str, tuple[set[tuple[str, str]], set[tuple[str, str]]]
    ] = {name: (set(), set()) for name in THERAPEUTIC_RELATIONS}
    row_count = 0

    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.reader(source)
        columns = next(reader, None)
        if not columns or len(columns) != len(set(columns)):
            raise ValueError("Missing or duplicate CSV header")

        absent = set(REQUIRED_COLUMNS) - set(columns)
        if absent:
            raise ValueError(f"Missing required columns: {', '.join(sorted(absent))}")

        indices = tuple(columns.index(name) for name in REQUIRED_COLUMNS)
        src_type_col, src_id_col, rel_col, dst_type_col, dst_id_col = indices

        for values in reader:
            row_count += 1
            if len(values) != len(columns):
                raise ValueError(f"Malformed CSV record {row_count}")

            src_type, src_id = values[src_type_col], values[src_id_col]
            relation = values[rel_col]
            dst_type, dst_id = values[dst_type_col], values[dst_id_col]

            relations[relation] += 1
            edge_types[(src_type, relation, dst_type)] += 1

            # missing fields count toward raw rows, not entities or pairs
            if not (
                src_type.strip() and src_id.strip() and relation.strip()
                and dst_type.strip() and dst_id.strip()
            ):
                for name, index in zip(REQUIRED_COLUMNS, indices):
                    if not values[index].strip():
                        missing[name] += 1
                continue

            if src_type == "drug":
                drug_ids.add(src_id)
            if dst_type == "drug":
                drug_ids.add(dst_id)
            if src_type == "disease":
                disease_ids.add(src_id)
            if dst_type == "disease":
                disease_ids.add(dst_id)

            if relation in therapeutic_pairs:
                if (src_type, dst_type) == ("drug", "disease"):
                    direction, pair = 0, (src_id, dst_id)
                elif (src_type, dst_type) == ("disease", "drug"):
                    direction, pair = 1, (dst_id, src_id)
                else:
                    raise ValueError(f"Invalid {relation} endpoints at row {row_count}")

                therapeutic_rows[relation][direction] += 1
                therapeutic_pairs[relation][direction].add(pair)

            elif relation == "disease_protein":
                if src_type == "disease":
                    protein_diseases.add(src_id)
                if dst_type == "disease":
                    protein_diseases.add(dst_id)

            elif relation == "disease_phenotype_positive":
                if src_type == "disease":
                    phenotype_diseases.add(src_id)
                if dst_type == "disease":
                    phenotype_diseases.add(dst_id)

    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("The source file changed during the audit")

    # merge edge directions keeping statistics
    pairs = {
        name: forward | reverse
        for name, (forward, reverse) in therapeutic_pairs.items()
    }
    indications = pairs["indication"]
    per_disease = Counter(disease for _, disease in indications)
    disease_counts = list(per_disease.values())
    median_indications = median(disease_counts) if disease_counts else None

    treatment = {}
    for name, (forward, reverse) in therapeutic_pairs.items():
        forward_rows, reverse_rows = therapeutic_rows[name]
        duplicates = forward_rows + reverse_rows - len(forward) - len(reverse)
        treatment[name] = {
            "pairs": len(pairs[name]),
            "forward_rows": forward_rows,
            "reverse_rows": reverse_rows,
            "forward_only": len(forward - reverse),
            "reverse_only": len(reverse - forward),
            "duplicate_rows": duplicates,
        }

    report: dict[str, Any] = {
        "file": path.name,
        "bytes": before.st_size,
        "sha256": digest,
        "rows": row_count,
        "columns": columns,
        "missing": dict(sorted(missing.items())),
        "relations": dict(sorted(relations.items())),
        "edge_types": [
            {"source": src, "relation": rel, "target": dst, "rows": count}
            for (src, rel, dst), count in sorted(edge_types.items())
        ],
        "entities": {"drugs": len(drug_ids), "diseases": len(disease_ids)},
        "therapeutic_relations": treatment,
        "indications": {
            "pairs": len(indications),
            "drugs": len({drug for drug, _ in indications}),
            "diseases": len(per_disease),
            "median_per_disease": median_indications,
            "diseases_with_at_least_2": sum(n >= 2 for n in disease_counts),
            "diseases_with_at_least_5": sum(n >= 5 for n in disease_counts),
            "diseases_with_protein_edges": len(per_disease.keys() & protein_diseases),
            "diseases_with_phenotype_edges": len(
                per_disease.keys() & phenotype_diseases
            ),
            "diseases_without_either": len(
                per_disease.keys() - protein_diseases - phenotype_diseases
            ),
            "also_contraindicated": len(indications & pairs["contraindication"]),
            "also_off_label": len(indications & pairs["off-label use"]),
        },
    }
    return report


def prepare_indications(path: Path, output: Path) -> dict[str, Any]:
    # relation: canonical (drug, disease) indices for both edge directions
    pairs: dict[str, tuple[set[tuple[int, int]], set[tuple[int, int]]]] = {
        name: (set(), set()) for name in THERAPEUTIC_RELATIONS
    }

    # (entity type, node index) -> external entity id
    nodes: dict[tuple[str, int], str] = {}

    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.reader(source)
        columns = next(reader, None)
        needed = (
            "relation", "x_type", "x_index", "x_id",
            "y_type", "y_index", "y_id",
        )
        if not columns or len(columns) != len(set(columns)):
            raise ValueError("Missing or duplicate CSV header")

        absent = set(needed) - set(columns)
        if absent:
            raise ValueError(f"Missing columns: {', '.join(sorted(absent))}")

        (
            rel_col, src_type_col, src_index_col, src_id_col,
            dst_type_col, dst_index_col, dst_id_col,
        ) = (columns.index(name) for name in needed)

        for number, row in enumerate(reader, 1):
            if len(row) != len(columns):
                raise ValueError(f"Malformed CSV record {number}")

            relation = row[rel_col]
            if relation not in pairs:
                continue

            src_type, dst_type = row[src_type_col], row[dst_type_col]
            if (src_type, dst_type) == ("drug", "disease"):
                direction = 0
                drug_index, disease_index = int(row[src_index_col]), int(row[dst_index_col])
                drug_id, disease_id = row[src_id_col], row[dst_id_col]
            elif (src_type, dst_type) == ("disease", "drug"):
                direction = 1
                drug_index, disease_index = int(row[dst_index_col]), int(row[src_index_col])
                drug_id, disease_id = row[dst_id_col], row[src_id_col]
            else:
                raise ValueError(f"Invalid {relation} endpoints at row {number}")

            # each typed node index must resolve to one external id
            for kind, index, identifier in (
                ("drug", drug_index, drug_id),
                ("disease", disease_index, disease_id),
            ):
                if not identifier.strip():
                    raise ValueError(f"Missing {kind} ID at row {number}")
                previous = nodes.setdefault((kind, index), identifier)
                if previous != identifier:
                    raise ValueError(f"Conflicting {kind} index {index}")

            pairs[relation][direction].add((drug_index, disease_index))

    # both directions must be the same
    for name, (forward, reverse) in pairs.items():
        if forward != reverse:
            raise ValueError(f"Forward and reverse {name} edges differ")

    indications = pairs["indication"][0]
    if not indications:
        raise ValueError("No indications found")

    output.mkdir(parents=True, exist_ok=True)
    table = output / "indications.csv"
    with table.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target, lineterminator="\n")
        writer.writerow((
            "drug_index", "disease_index", "drug_id", "disease_id",
            "contraindication", "off_label",
        ))
        for drug, disease in sorted(indications, key=lambda pair: (pair[1], pair[0])):
            pair = (drug, disease)
            writer.writerow((
                drug, disease, nodes[("drug", drug)], nodes[("disease", disease)],
                int(pair in pairs["contraindication"][0]),
                int(pair in pairs["off-label use"][0]),
            ))

    # all three labels, ordered disease, relation, then drug
    therapy = output / "therapeutic.csv"
    therapy_rows = [
        (drug, disease, relation, nodes[("drug", drug)], nodes[("disease", disease)])
        for relation in THERAPEUTIC_RELATIONS
        for drug, disease in pairs[relation][0]
    ]
    therapy_rows.sort(key=lambda row: (row[1], row[2], row[0]))
    with therapy.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target, lineterminator="\n")
        writer.writerow((
            "drug_index", "disease_index", "relation", "drug_id", "disease_id"
        ))
        writer.writerows(therapy_rows)

    report: dict[str, Any] = {
        "source": {
            "file": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        },
        "indications": len(indications),
        "diseases": len({disease for _, disease in indications}),
        "drugs": len({drug for drug, _ in indications}),
        "also_contraindicated": len(indications & pairs["contraindication"][0]),
        "also_off_label": len(indications & pairs["off-label use"][0]),
        "table": {"file": table.name, "sha256": sha256_file(table)},
        "therapeutic": {
            "file": therapy.name,
            "sha256": sha256_file(therapy),
            "counts": {
                name: len(pairs[name][0]) for name in THERAPEUTIC_RELATIONS
            },
        },
    }
    (output / "indications.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report
