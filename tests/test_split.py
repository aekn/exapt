import csv
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from exapt.data import THERAPEUTIC_RELATIONS
from exapt.split import (
    SMOKE_NEIGHBORS, _partition, _smoke_graph, _stratum,
    split_indications, split_therapy,
)


class SplitTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def _write_split_fixture(self) -> tuple[Path, Path, Path]:
        graph = self.root / "kg.csv"
        indications = self.root / "indications.csv"
        therapy = self.root / "therapeutic.csv"

        with graph.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow((
                "relation", "x_type", "x_index", "x_id",
                "y_type", "y_index", "y_id",
            ))
            for i in range(60):
                drug, disease, protein = 100 + i, 200 + i, 400 + i
                writer.writerows((
                    ("indication", "drug", drug, f"DB{i}",
                     "disease", disease, f"D{i}"),
                    ("indication", "disease", disease, f"D{i}",
                     "drug", drug, f"DB{i}"),
                    ("disease_protein", "disease", disease, f"D{i}",
                     "gene/protein", protein, f"P{i}"),
                    ("disease_protein", "gene/protein", protein, f"P{i}",
                     "disease", disease, f"D{i}"),
                ))

        with indications.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow((
                "drug_index", "disease_index", "drug_id", "disease_id",
                "contraindication", "off_label",
            ))
            for i in range(60):
                writer.writerow((100 + i, 200 + i, f"DB{i}", f"D{i}", 0, 0))

        with therapy.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow((
                "drug_index", "disease_index", "relation", "drug_id", "disease_id"
            ))
            for i in range(60):
                for relation in THERAPEUTIC_RELATIONS:
                    writer.writerow((100 + i, 200 + i, relation, f"DB{i}", f"D{i}"))

        return graph, indications, therapy

    def test_stratified_disease_partition_is_deterministic(self):
        counts = Counter({disease: disease % 7 + 1 for disease in range(120)})
        proteins = {disease for disease in counts if disease % 3}

        assignment = _partition(set(counts), counts, proteins, seed=163)
        self.assertEqual(assignment, _partition(set(counts), counts, proteins, 163))
        self.assertEqual(set(assignment), set(counts))

        # development and final are rounded independently within each stratum
        strata: dict[tuple[int, bool], list[int]] = {}
        for disease in counts:
            stratum = _stratum(counts[disease], disease in proteins)
            strata.setdefault(stratum, []).append(disease)
        for members in strata.values():
            partitions = Counter(assignment[disease] for disease in members)
            self.assertEqual(partitions["development"], round(len(members) * 0.1))
            self.assertEqual(partitions["final"], round(len(members) * 0.1))
            self.assertEqual(sum(partitions.values()), len(members))

    def test_smoke_graph_and_therapy_respect_frozen_partitions(self):
        graph, indications, therapy = self._write_split_fixture()
        output = self.root / "split"
        report = split_indications(indications, graph, output)
        self.assertEqual(report["partition"]["train"]["diseases"], 48)
        self.assertEqual(report["partition"]["development"]["diseases"], 6)
        self.assertEqual(report["partition"]["final"]["diseases"], 6)

        partitions: dict[str, set[int]] = {}
        for name in ("train", "development", "final"):
            with (output / f"{name}.csv").open(newline="") as stream:
                partitions[name] = {
                    int(row["disease_index"]) for row in csv.DictReader(stream)
                }
        self.assertEqual(set.union(*partitions.values()), set(range(200, 260)))
        self.assertFalse(partitions["train"] & partitions["development"])
        self.assertFalse(partitions["train"] & partitions["final"])
        self.assertFalse(partitions["development"] & partitions["final"])

        with (output / "smoke" / "graph.csv").open(newline="") as stream:
            edges = list(csv.DictReader(stream))
        self.assertTrue(edges)
        self.assertTrue(all(
            row["relation"] not in THERAPEUTIC_RELATIONS for row in edges
        ))

        with (output / "smoke" / "development.csv").open(newline="") as stream:
            smoke_dev = {
                int(row["disease_index"]) for row in csv.DictReader(stream)
            }
        self.assertTrue(smoke_dev <= partitions["development"]

        )

        projection = split_therapy(therapy, output)
        self.assertEqual(
            projection["train"], dict.fromkeys(THERAPEUTIC_RELATIONS, 48)
        )
        self.assertEqual(
            projection["development"], dict.fromkeys(THERAPEUTIC_RELATIONS, 6)
        )

        with (output / "therapeutic_train.csv").open(newline="") as stream:
            train_labels = list(csv.DictReader(stream))
        with (output / "therapeutic_development.csv").open(newline="") as stream:
            dev_labels = list(csv.DictReader(stream))
        self.assertEqual(len(train_labels), 3 * 48)
        self.assertEqual(len(dev_labels), 3 * 6)
        self.assertEqual(
            {int(row["disease_index"]) for row in train_labels},
            partitions["train"],
        )
        self.assertEqual(
            {int(row["disease_index"]) for row in dev_labels},
            partitions["development"],
        )
        # final labels never appear in either export
        self.assertTrue(all(
            int(row["disease_index"]) not in partitions["final"]
            for row in train_labels + dev_labels
        ))

        with (output / "smoke" / "train.csv").open(newline="") as stream:
            smoke_train = {
                int(row["disease_index"]) for row in csv.DictReader(stream)
            }
        with (output / "smoke" / "drugs.csv").open(newline="") as stream:
            smoke_drugs = {int(row["drug_index"]) for row in csv.DictReader(stream)}
        with (output / "smoke" / "therapeutic_train.csv").open(newline="") as stream:
            smoke_labels = list(csv.DictReader(stream))
        self.assertTrue(smoke_labels)
        self.assertTrue(all(
            int(row["disease_index"]) in smoke_train
            and int(row["drug_index"]) in smoke_drugs
            for row in smoke_labels
        ))

    def test_split_artifacts_are_byte_for_byte_deterministic(self):
        graph, indications, therapy = self._write_split_fixture()
        first = self.root / "first"
        second = self.root / "second"
        self.assertEqual(
            split_indications(indications, graph, first, seed=163),
            split_indications(indications, graph, second, seed=163),
        )
        self.assertEqual(split_therapy(therapy, first), split_therapy(therapy, second))

        files = (
            "train.csv", "development.csv", "final.csv", "drugs.csv",
            "split.json", "therapeutic_train.csv", "therapeutic_development.csv",
            "therapy.json", "smoke/train.csv", "smoke/development.csv",
            "smoke/graph.csv", "smoke/nodes.csv", "smoke/drugs.csv",
            "smoke/therapeutic_train.csv",
        )
        for name in files:
            with self.subTest(file=name):
                self.assertEqual(
                    (first / name).read_bytes(), (second / name).read_bytes()
                )

    def test_smoke_reservoir_bounds_neighbors_per_anchor_relation(self):
        graph = self.root / "kg.csv"
        with graph.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow((
                "relation", "x_index", "x_type", "x_id",
                "y_index", "y_type", "y_id",
            ))
            for protein in range(100, 140):
                writer.writerow((
                    "disease_protein", 1, "disease", "D1",
                    protein, "gene/protein", f"P{protein}",
                ))

        diseases = {"train": {1}, "development": set()}
        positives = [(2, 1)]
        drugs = {2: "DB2"}
        disease_ids = {1: "D1"}
        first = self.root / "first"
        second = self.root / "second"
        result = _smoke_graph(
            graph, first, diseases, positives, drugs, disease_ids, seed=163
        )
        repeated = _smoke_graph(
            graph, second, diseases, positives, drugs, disease_ids, seed=163
        )
        self.assertEqual(result, repeated)
        self.assertEqual(result["directed_edges"], SMOKE_NEIGHBORS)
        self.assertEqual(result["relations"], {"disease_protein": SMOKE_NEIGHBORS})

        with (first / "graph.csv").open(newline="") as stream:
            edges = list(csv.DictReader(stream))
        self.assertEqual(len(edges), SMOKE_NEIGHBORS)
        self.assertEqual({int(row["x_index"]) for row in edges}, {1})
        self.assertEqual(
            len({int(row["y_index"]) for row in edges}), SMOKE_NEIGHBORS
        )
        for name in ("graph.csv", "nodes.csv", "drugs.csv"):
            self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())

    def test_rejects_shared_disease_identity(self):
        indications = self.root / "indications.csv"
        indications.write_text(
            "drug_index,disease_index,drug_id,disease_id,"
            "contraindication,off_label\n"
            "1,10,DB1,D1_D2,0,0\n"
            "2,20,DB2,D2,0,0\n"
        )
        with self.assertRaisesRegex(ValueError, "share source ID"):
            split_indications(
                indications, self.root / "missing.csv", self.root / "output"
            )

    def test_therapy_must_match_training_labels_and_disjoint_partitions(self):
        split = self.root / "split"
        smoke = split / "smoke"
        smoke.mkdir(parents=True)
        header = "drug_index,disease_index\n"
        (split / "train.csv").write_text(header + "1,10\n")
        (split / "development.csv").write_text(header + "2,20\n")
        (split / "final.csv").write_text(header + "3,30\n")
        (smoke / "train.csv").write_text(header + "1,10\n")
        (smoke / "drugs.csv").write_text("drug_index\n1\n")

        therapy = self.root / "therapeutic.csv"
        therapy.write_text(
            "drug_index,disease_index,relation,drug_id,disease_id\n"
            "2,20,indication,DB2,D20\n"
        )
        with self.assertRaisesRegex(ValueError, "frozen split"):
            split_therapy(therapy, split)

        # partition overlap is invalid even if indication labels differ
        (split / "development.csv").write_text(header + "2,10\n")
        with self.assertRaisesRegex(ValueError, "overlap"):
            split_therapy(therapy, split)
