import tempfile
import unittest
from pathlib import Path

import torch

from exapt.graph import load_graph, prepare_graph


class GraphTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def test_prepared_graph_excludes_therapeutic_edges(self):
        source = self.root / "kg.csv"
        source.write_text(
            "relation,x_index,x_type,y_index,y_type\n"
            "drug_protein,0,drug,1,gene/protein\n"
            "drug_protein,1,gene/protein,0,drug\n"
            "indication,0,drug,2,disease\n"
            "indication,2,disease,0,drug\n"
            "disease_protein,2,disease,1,gene/protein\n"
            "disease_protein,1,gene/protein,2,disease\n"
        )

        output = self.root / "graph.pt"
        report = prepare_graph(source, output)
        graph = load_graph(output)

        self.assertEqual(graph.nodes, {0: 0, 1: 1, 2: 2})
        self.assertEqual(
            (report["nodes"], report["directed_edges"], report["relations"]),
            (3, 4, 4),
        )
        self.assertEqual(report["unique_edges"], 4)
        self.assertEqual(sum(A._nnz() for A in graph.adjacency), 4)
        self.assertTrue(all(rel[1] != "indication" for rel in graph.relations))

        features = torch.ones(3, 1, requires_grad=True)
        total = torch.stack([
            torch.sparse.mm(matrix, features).sum()
            for matrix in graph.adjacency
        ]).sum()
        total.backward()
        self.assertIsNotNone(features.grad)

    def test_csv_loader_rejects_therapeutic_message_edges(self):
        (self.root / "nodes.csv").write_text(
            "index,type,id\n0,drug,D0\n1,disease,C1\n"
        )
        (self.root / "graph.csv").write_text(
            "x_index,x_type,relation,y_index,y_type\n"
            "0,drug,indication,1,disease\n"
        )

        with self.assertRaisesRegex(ValueError, "Therapeutic"):
            load_graph(self.root)
