import csv
import math
import tempfile
import unittest
from pathlib import Path

import torch

from exapt.graph import Graph, load_graph
from exapt.model import ExaptGNN, RelationalGNN
from exapt.prototypes import Prototypes, build_prototypes
from exapt.train import load_pairs, train_model


class PrototypeTests(unittest.TestCase):
    def test_cosine_donors_padding_and_missing_profiles(self):
        # diseases 0-4, proteins 5-6
        # N(0)={5}, N(1)={5,6}, N(2)={5,6}, N(3)={}, N(4)={6}
        edges = ((0, 5), (1, 5), (1, 6), (2, 5), (2, 6), (4, 6))
        indices = torch.tensor([
            [protein for _, protein in edges],
            [disease for disease, _ in edges],
        ], dtype=torch.long)
        adjacency = torch.sparse_coo_tensor(
            indices, torch.ones(len(edges)), (7, 7), check_invariants=True
        ).coalesce()
        graph = Graph(
            {node: node for node in range(7)},
            (("disease", "disease_protein", "gene/protein"),),
            (adjacency,),
        )

        prototypes = build_prototypes(graph, {0, 1}, {2, 3, 4}, k=2)
        torch.testing.assert_close(prototypes.diseases, torch.tensor([2, 4]))
        torch.testing.assert_close(
            prototypes.donors, torch.tensor([[1, 0], [1, 1]])
        )

        # cos(2,1)=1, cos(2,0)=1/sqrt(2)
        similarity = 1 / math.sqrt(2)
        torch.testing.assert_close(prototypes.weights, torch.tensor([
            [1 / (1 + similarity), similarity / (1 + similarity)],
            [1.0, 0.0],
        ]))
        torch.testing.assert_close(prototypes.weights.sum(1), torch.ones(2))
        torch.testing.assert_close(prototypes.evidence, torch.tensor([
            [1.0, 1.0, 1 / 3], [similarity, 1.0, 1 / 3],
        ]))

        # a training disease must not use its own representation as a donor
        self_donor = build_prototypes(graph, {0, 1}, {0}, k=2)
        torch.testing.assert_close(self_donor.donors, torch.tensor([[1, 1]]))
        torch.testing.assert_close(
            self_donor.weights, torch.tensor([[1.0, 0.0]])
        )

        empty = build_prototypes(graph, {0, 1}, {3}, k=3)
        self.assertEqual(tuple(empty.diseases.shape), (0,))
        self.assertEqual(tuple(empty.donors.shape), (0, 3))
        self.assertEqual(tuple(empty.weights.shape), (0, 3))
        self.assertEqual(tuple(empty.evidence.shape), (0, 3))
        with self.assertRaisesRegex(ValueError, "positive"):
            build_prototypes(graph, {0, 1}, {2}, k=0)

    def test_disease_neighbors_use_both_edge_orientations(self):
        # train diseases 0,1; query 2; shared disease neighbor 3
        for indices in (
            torch.tensor([[3, 3, 4, 2], [0, 2, 1, 2]], dtype=torch.long),
            torch.tensor([[0, 2, 1, 2], [3, 3, 4, 2]], dtype=torch.long),
        ):
            with self.subTest(indices=indices.tolist()):
                adjacency = torch.sparse_coo_tensor(
                    indices, torch.ones(4), (5, 5), check_invariants=True
                ).coalesce()
                graph = Graph(
                    {node: node for node in range(5)},
                    (("disease", "disease_disease", "disease"),),
                    (adjacency,),
                )
                prototypes = build_prototypes(graph, {0, 1}, {2}, k=2)
                torch.testing.assert_close(
                    prototypes.diseases, torch.tensor([2])
                )
                torch.testing.assert_close(
                    prototypes.donors, torch.tensor([[0, 0]])
                )
                torch.testing.assert_close(
                    prototypes.weights, torch.tensor([[1., 0.]])
                )
                torch.testing.assert_close(
                    prototypes.evidence, torch.tensor([[1., 1., 1 / 3]])
                )

    def test_separate_profiles_and_new_query_diseases(self):
        # train diseases 0,1; queries 2,3; protein 4, phenotypes 5,6
        graph = Graph(
            {node: node for node in range(7)},
            (
                ("disease", "disease_protein", "gene/protein"),
                ("disease", "disease_phenotype_positive", "effect/phenotype"),
            ),
            (
                torch.sparse_coo_tensor(
                    torch.tensor([[4, 4, 4], [0, 1, 2]], dtype=torch.long),
                    torch.ones(3), (7, 7), check_invariants=True,
                ).coalesce(),
                torch.sparse_coo_tensor(
                    torch.tensor([[5, 5, 6], [0, 3, 2]], dtype=torch.long),
                    torch.ones(3), (7, 7), check_invariants=True,
                ).coalesce(),
            ),
        )
        first = build_prototypes(graph, {0, 1}, {2}, k=2)
        torch.testing.assert_close(first.diseases, torch.tensor([2]))
        torch.testing.assert_close(first.weights, torch.tensor([[0.5, 0.5]]))
        torch.testing.assert_close(
            first.evidence, torch.tensor([[0.5, 0.5, 2 / 3]])
        )

        model = ExaptGNN(RelationalGNN(7, 2, 8), first, 8)
        embeddings = model(graph)
        torch.testing.assert_close(
            model.disease_vectors(embeddings, torch.tensor([3])),
            embeddings[3:4],
        )

        # reuse trained parameters with a different set of query diseases
        second = build_prototypes(graph, {0, 1}, {3}, k=2)
        model.set_prototypes(second)
        torch.testing.assert_close(model.diseases, torch.tensor([3]))
        torch.testing.assert_close(
            second.evidence, torch.tensor([[1., 1., 1 / 3]])
        )
        torch.testing.assert_close(
            model.disease_vectors(embeddings, torch.tensor([2])),
            embeddings[2:3],
        )
        expected = 0.8 * embeddings[3] + 0.2 * embeddings[0]
        torch.testing.assert_close(
            model.disease_vectors(embeddings, torch.tensor([3]))[0], expected
        )

    def test_batched_scoring_uses_evidence_and_unique_queries(self):
        prototypes = Prototypes(
            diseases=torch.tensor([2]),
            donors=torch.tensor([[0, 1]]),
            weights=torch.tensor([[0.25, 0.75]]),
            evidence=torch.tensor([[0.6, 0.5, 1 / 3]]),
        )
        model = ExaptGNN(RelationalGNN(4, 0, width=2), prototypes, width=2)
        with torch.no_grad():
            model.gate.weight.zero_()
            model.gate.weight[0, -3:].copy_(torch.tensor([2., -1., 0.5]))
            model.gate.bias.zero_()
            model.base.decoder.weight[1].copy_(torch.tensor([2., 3.]))

        embeddings = torch.tensor([
            [1., 2.], [3., 4.], [5., 6.], [7., 8.],
        ])
        query = torch.tensor([[2, 3], [2, 2]])
        result = model.disease_vectors(embeddings, query)

        # p_2 = 0.25*h_0 + 0.75*h_1
        # alpha = sigmoid(2*similarity - agreement + 0.5*availability)
        prototype = 0.25 * embeddings[0] + 0.75 * embeddings[1]
        alpha = torch.sigmoid(torch.tensor(1.2 - 0.5 + 0.5 / 3))
        mixed = embeddings[2] + alpha * (prototype - embeddings[2])
        expected = torch.stack(
            (mixed, embeddings[3], mixed, mixed)
        ).reshape(2, 2, 2)
        torch.testing.assert_close(result, expected)
        self.assertEqual(model.index.tolist(), [-1, -1, 0, -1])

        empty = torch.empty((0, 2), dtype=torch.long)
        self.assertEqual(
            tuple(model.disease_vectors(embeddings, empty).shape), (0, 2, 2)
        )

        drugs = torch.tensor([[0, 1], [1, 0]])
        scores = model.score(embeddings, drugs, query, relation=1)
        relation_weights = torch.tensor([2., 3.])
        manual = (embeddings[drugs] * relation_weights * expected).sum(-1)
        torch.testing.assert_close(scores, manual)

    def test_gated_augmentation_and_gradients(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "nodes.csv").write_text(
                "index,type,id\n"
                "1,disease,C1\n2,disease,C2\n3,disease,C3\n"
                "4,disease,C4\n5,gene/protein,P5\n6,gene/protein,P6\n"
            )
            (root / "graph.csv").write_text(
                "x_index,x_type,relation,y_index,y_type\n"
                "1,disease,disease_protein,5,gene/protein\n"
                "5,gene/protein,disease_protein,1,disease\n"
                "2,disease,disease_protein,5,gene/protein\n"
                "5,gene/protein,disease_protein,2,disease\n"
                "2,disease,disease_protein,6,gene/protein\n"
                "6,gene/protein,disease_protein,2,disease\n"
                "3,disease,disease_protein,6,gene/protein\n"
                "6,gene/protein,disease_protein,3,disease\n"
            )
            graph = load_graph(root)

        donor = graph.nodes[2]
        disease = graph.nodes[3]
        unsupported = graph.nodes[4]
        prototypes = build_prototypes(
            graph, {graph.nodes[1], donor}, {disease, unsupported}, k=2
        )
        torch.testing.assert_close(
            prototypes.donors, torch.tensor([[donor, donor]])
        )

        torch.manual_seed(163)
        base = RelationalGNN(len(graph.nodes), len(graph.relations), width=8)
        model = ExaptGNN(base, prototypes, width=8)
        original = base(graph)
        queries = torch.tensor([disease, unsupported])
        augmented = model.disease_vectors(model(graph), queries)

        # zero gate weights give alpha=0.2 for supported diseases
        torch.testing.assert_close(
            augmented[0], original[disease] * 0.8 + original[donor] * 0.2
        )
        torch.testing.assert_close(augmented[1], original[unsupported])
        augmented[0].sum().backward()

        gate_grad = model.gate.weight.grad
        embedding_grad = base.encoder.embeddings.weight.grad
        assert gate_grad is not None
        assert embedding_grad is not None
        self.assertGreater(gate_grad.abs().sum().item(), 0)
        self.assertGreater(embedding_grad[donor].abs().sum().item(), 0)

        state = model.state_dict()
        for name in ("index", "diseases", "donors", "weights", "evidence"):
            self.assertIn(name, state)
        self.assertNotIn("donors", dict(model.named_parameters()))

    def test_prototype_checkpoint_reproduces_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "nodes.csv").write_text(
                "index,type,id\n"
                "1,drug,D1\n2,drug,D2\n3,drug,D3\n"
                "4,disease,C4\n5,disease,C5\n6,gene/protein,P6\n"
            )
            (root / "graph.csv").write_text(
                "x_index,x_type,relation,y_index,y_type\n"
                "1,drug,drug_protein,6,gene/protein\n"
                "6,gene/protein,drug_protein,1,drug\n"
                "4,disease,disease_protein,6,gene/protein\n"
                "6,gene/protein,disease_protein,4,disease\n"
                "5,disease,disease_protein,6,gene/protein\n"
                "6,gene/protein,disease_protein,5,disease\n"
            )
            (root / "drugs.csv").write_text(
                "drug_index,drug_id\n1,D1\n2,D2\n3,D3\n"
            )
            header = (
                "drug_index,disease_index,drug_id,disease_id,"
                "contraindication,off_label\n"
            )
            (root / "train.csv").write_text(header + "1,4,D1,C4,0,0\n")
            (root / "development.csv").write_text(header + "2,5,D2,C5,0,0\n")
            (root / "therapeutic_train.csv").write_text(
                "drug_index,disease_index,relation,drug_id,disease_id\n"
                "1,4,indication,D1,C4\n"
            )

            output = root / "run"
            result = train_model(
                root, root, output, epochs=2, eval_every=1,
                width=8, prototype_k=2,
            )
            self.assertEqual(result["model"], "exaptgnn")
            self.assertEqual(result["prototype_covered"], {
                "train": 0, "development": 1,
            })

            saved = torch.load(output / "model.pt", weights_only=True)
            self.assertEqual(saved["prototype_k"], 2)
            self.assertIn("gate.weight", saved["model"])

            graph = load_graph(root)
            train_pairs = load_pairs(root / "train.csv", graph.nodes)
            dev_pairs = load_pairs(root / "development.csv", graph.nodes)
            train_diseases = {disease for _, disease in train_pairs}
            self.assertEqual(saved["donor_diseases"], sorted(train_diseases))
            query_diseases = train_diseases | {
                disease for _, disease in dev_pairs
            }
            prototypes = build_prototypes(
                graph, train_diseases, query_diseases, k=2
            )
            restored = ExaptGNN(
                RelationalGNN(len(graph.nodes), len(graph.relations), 8),
                prototypes, 8,
            )
            restored.load_state_dict(saved["model"])
            restored.eval()

            with torch.no_grad():
                embeddings = restored(graph)
                drugs = torch.tensor([graph.nodes[d] for d in (1, 2, 3)])
                diseases = torch.full_like(drugs, graph.nodes[5])
                scores = restored.score(embeddings, drugs, diseases).tolist()

            with (output / "predictions.csv").open(newline="") as stream:
                predictions = list(csv.DictReader(stream))
            by_drug = {
                int(row["drug_index"]): float(row["score"])
                for row in predictions
            }
            self.assertEqual(set(by_drug), {1, 2, 3})
            for drug, score in zip((1, 2, 3), scores):
                self.assertAlmostEqual(score, by_drug[drug], places=5)
