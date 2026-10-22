import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from exapt.data import audit_graph, prepare_indications


class AuditGraphTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.graph = Path(directory.name) / "kg.csv"

    def test_counts_missing_values_and_source_hash(self):
        raw = (
            "x_type,x_id,relation,y_type,y_id,extra\n"
            "drug,1,indication,disease,2,first\n"
            "disease,2,disease_protein,gene/protein,3,second\n"
            "drug,4,indication,disease,,third\n"
        ).encode()
        self.graph.write_bytes(raw)

        report = audit_graph(self.graph)
        self.assertEqual(report["rows"], 3)
        self.assertEqual(report["bytes"], len(raw))
        self.assertEqual(report["sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(report["missing"], {"y_id": 1})
        self.assertEqual(report["indications"]["pairs"], 1)
        self.assertEqual(report["relations"], {
            "disease_protein": 1, "indication": 2,
        })
        self.assertEqual(report["edge_types"][1], {
            "source": "drug", "relation": "indication",
            "target": "disease", "rows": 2,
        })

    def test_therapeutic_pairs_and_disease_coverage(self):
        self.graph.write_text(
            "x_type,x_id,relation,y_type,y_id\n"
            "drug,DB1,indication,disease,D1\n"
            "disease,D1,indication,drug,DB1\n"
            "drug,DB1,indication,disease,D1\n"
            "disease,D2,indication,drug,DB2\n"
            "drug,DB1,contraindication,disease,D1\n"
            "drug,DB1,off-label use,disease,D1\n"
            "disease,D1,disease_protein,gene/protein,P1\n"
            "effect/phenotype,E1,disease_phenotype_positive,disease,D2\n"
        )

        report = audit_graph(self.graph)
        self.assertEqual(report["therapeutic_relations"]["indication"], {
            "pairs": 2, "forward_rows": 2, "reverse_rows": 2,
            "forward_only": 0, "reverse_only": 1, "duplicate_rows": 1,
        })
        self.assertEqual(report["indications"], {
            "pairs": 2, "drugs": 2, "diseases": 2,
            "median_per_disease": 1,
            "diseases_with_at_least_2": 0,
            "diseases_with_at_least_5": 0,
            "diseases_with_protein_edges": 1,
            "diseases_with_phenotype_edges": 1,
            "diseases_without_either": 0,
            "also_contraindicated": 1,
            "also_off_label": 1,
        })

    def test_rejects_invalid_therapeutic_endpoints(self):
        self.graph.write_text(
            "x_type,x_id,relation,y_type,y_id\n"
            "drug,D1,indication,gene/protein,P1\n"
        )
        with self.assertRaisesRegex(ValueError, "Invalid indication endpoints"):
            audit_graph(self.graph)

    def test_rejects_missing_columns(self):
        self.graph.write_text("x_type,x_id,relation,y_type\n")
        with self.assertRaisesRegex(ValueError, "y_id"):
            audit_graph(self.graph)

    def test_rejects_duplicate_columns(self):
        self.graph.write_text("x_type,x_id,relation,y_type,y_id,y_id\n")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            audit_graph(self.graph)

    def test_rejects_malformed_records(self):
        self.graph.write_text(
            "x_type,x_id,relation,y_type,y_id\n"
            "drug,1,indication,disease,2,unexpected\n"
        )
        with self.assertRaisesRegex(ValueError, "record 1"):
            audit_graph(self.graph)


class PrepareIndicationsTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.graph = Path(directory.name) / "kg.csv"
        self.output = Path(directory.name) / "processed"

    def test_deduplicates_reverse_edges_and_flags_overlaps(self):
        self.graph.write_text(
            "relation,x_type,x_index,x_id,y_type,y_index,y_id\n"
            "indication,disease,20,D1,drug,10,DB1\n"
            "indication,drug,10,DB1,disease,20,D1\n"
            "indication,drug,11,DB2,disease,19,D2\n"
            "indication,disease,19,D2,drug,11,DB2\n"
            "indication,drug,10,DB1,disease,20,D1\n"
            "contraindication,drug,10,DB1,disease,20,D1\n"
            "contraindication,disease,20,D1,drug,10,DB1\n"
            "off-label use,drug,11,DB2,disease,19,D2\n"
            "off-label use,disease,19,D2,drug,11,DB2\n"
            "drug_protein,drug,10,DB1,gene/protein,30,P1\n"
        )

        report = prepare_indications(self.graph, self.output)
        with (self.output / "indications.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))

        self.assertEqual(rows, [
            {"drug_index": "11", "disease_index": "19", "drug_id": "DB2",
             "disease_id": "D2", "contraindication": "0", "off_label": "1"},
            {"drug_index": "10", "disease_index": "20", "drug_id": "DB1",
             "disease_id": "D1", "contraindication": "1", "off_label": "0"},
        ])
        self.assertEqual(report["indications"], 2)
        with (self.output / "therapeutic.csv").open(newline="") as stream:
            therapy = list(csv.DictReader(stream))
        self.assertEqual(therapy, [
            {"drug_index": "11", "disease_index": "19",
             "relation": "indication", "drug_id": "DB2", "disease_id": "D2"},
            {"drug_index": "11", "disease_index": "19",
             "relation": "off-label use", "drug_id": "DB2", "disease_id": "D2"},
            {"drug_index": "10", "disease_index": "20",
             "relation": "contraindication", "drug_id": "DB1", "disease_id": "D1"},
            {"drug_index": "10", "disease_index": "20",
             "relation": "indication", "drug_id": "DB1", "disease_id": "D1"},
        ])
        self.assertEqual(report["therapeutic"]["counts"], {
            "indication": 2, "contraindication": 1, "off-label use": 1,
        })
        self.assertEqual(report["also_contraindicated"], 1)
        self.assertEqual(report["also_off_label"], 1)
        self.assertEqual(
            report["therapeutic"]["sha256"],
            hashlib.sha256((self.output / "therapeutic.csv").read_bytes()).hexdigest(),
        )
        self.assertEqual(
            report["source"]["sha256"],
            hashlib.sha256(self.graph.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            json.loads((self.output / "indications.json").read_text()), report
        )

    def test_rejects_missing_reverse_edges_and_conflicting_node_ids(self):
        header = "relation,x_type,x_index,x_id,y_type,y_index,y_id\n"
        forward = "indication,drug,10,DB1,disease,20,D1\n"
        reverse = "indication,disease,20,D1,drug,10,DB1\n"

        self.graph.write_text(header + forward)
        with self.assertRaisesRegex(ValueError, "Forward and reverse"):
            prepare_indications(self.graph, self.output)

        self.graph.write_text(
            header + forward + reverse
            + "contraindication,drug,10,DB1,disease,20,D1\n"
        )
        with self.assertRaisesRegex(ValueError, "Forward and reverse"):
            prepare_indications(self.graph, self.output)

        self.graph.write_text(
            header + forward + reverse
            + "indication,drug,10,DIFFERENT,disease,20,D1\n"
        )
        with self.assertRaisesRegex(ValueError, "Conflicting drug index"):
            prepare_indications(self.graph, self.output)
