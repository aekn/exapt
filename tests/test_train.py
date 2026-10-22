import csv
import tempfile
import unittest
from pathlib import Path
from typing import NotRequired, TypedDict

import torch
from torch.nn import functional as F

from exapt.checkpoint import load_snapshot
from exapt.graph import load_graph, prepare_graph
from exapt.model import RelationalGNN
from exapt.train import disease_bpr, train_model, train_smoke


class TrainingSettings(TypedDict):
    epochs: int
    width: int
    eval_every: int
    negatives: NotRequired[int]


class DiseaseBalancedLossTests(unittest.TestCase):
    def test_equal_disease_contribution_and_gradients(self):
        # two positives for disease a, one for b; two negatives per positive
        positive = torch.tensor([1., 2., 3.], requires_grad=True)
        negative = torch.tensor([
            [0., 0.5], [1., 1.5], [0., 1.],
        ], requires_grad=True)
        weights = torch.tensor([0.25, 0.25, 0.5])

        loss = disease_bpr(positive, negative, weights)
        pair_loss = F.softplus(negative - positive[:, None]).mean(dim=1)
        expected = (pair_loss[:2].mean() + pair_loss[2]) / 2
        torch.testing.assert_close(loss, expected)
        self.assertNotAlmostEqual(loss.item(), pair_loss.mean().item())

        loss.backward()
        positive_grad = positive.grad
        negative_grad = negative.grad
        assert positive_grad is not None
        assert negative_grad is not None
        self.assertTrue(torch.all(positive_grad < 0))
        self.assertTrue(torch.all(negative_grad > 0))


class TrainingTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def test_three_head_training_checkpoint_reproduces_predictions(self):
        dataset = self.root / "smoke"
        dataset.mkdir()
        (dataset / "nodes.csv").write_text(
            "index,type,id\n"
            "1,drug,D1\n2,drug,D2\n3,drug,D3\n"
            "4,disease,C4\n5,disease,C5\n6,gene/protein,P6\n"
        )
        (dataset / "graph.csv").write_text(
            "x_index,x_type,relation,y_index,y_type\n"
            "1,drug,drug_protein,6,gene/protein\n"
            "6,gene/protein,drug_protein,1,drug\n"
            "4,disease,disease_protein,6,gene/protein\n"
            "6,gene/protein,disease_protein,4,disease\n"
        )
        (dataset / "drugs.csv").write_text(
            "drug_index,drug_id\n1,D1\n2,D2\n3,D3\n"
        )
        header = (
            "drug_index,disease_index,drug_id,disease_id,"
            "contraindication,off_label\n"
        )
        (dataset / "train.csv").write_text(header + "1,4,D1,C4,0,0\n")
        (dataset / "development.csv").write_text(header + "2,5,D2,C5,0,0\n")
        (dataset / "therapeutic_train.csv").write_text(
            "drug_index,disease_index,relation,drug_id,disease_id\n"
            "1,4,indication,D1,C4\n"
            "2,4,contraindication,D2,C4\n"
            "3,4,off-label use,D3,C4\n"
        )

        output = self.root / "run"
        record = train_smoke(dataset, output, epochs=6, width=8)
        self.assertEqual(record["train_pairs"], 1)
        self.assertEqual(record["development_pairs"], 1)
        self.assertEqual(record["negative_ratio"], 8)
        self.assertEqual(record["therapeutic_pairs"], {
            "indication": 1, "contraindication": 1, "off-label use": 1,
        })

        checkpoint = torch.load(output / "model.pt", weights_only=True)
        self.assertEqual(
            tuple(checkpoint["model"]["decoder.weight"].shape), (3, 8)
        )

        graph = load_graph(dataset)
        model = RelationalGNN(len(graph.nodes), len(graph.relations), width=8)
        model.load_state_dict(checkpoint["model"])
        model.eval()

        with torch.no_grad():
            embeddings = model(graph)
            drugs = torch.tensor([graph.nodes[d] for d in (1, 2, 3)])
            diseases = torch.full_like(drugs, graph.nodes[5])
            scores = model.score(embeddings, drugs, diseases).tolist()

        with (output / "predictions.csv").open(newline="") as stream:
            predictions = list(csv.DictReader(stream))
        self.assertEqual(len(predictions), 3)
        saved_scores = {
            int(row["drug_index"]): float(row["score"])
            for row in predictions
        }
        for drug, score in zip((1, 2, 3), scores):
            self.assertAlmostEqual(score, saved_scores[drug], places=5)

    def test_best_checkpoint_and_candidate_universe(self):
        source = self.root / "kg.csv"
        source.write_text(
            "relation,x_index,x_type,y_index,y_type\n"
            "drug_protein,0,drug,1,gene/protein\n"
            "drug_protein,1,gene/protein,0,drug\n"
            "disease_protein,2,disease,1,gene/protein\n"
            "disease_protein,1,gene/protein,2,disease\n"
            "drug_protein,3,drug,1,gene/protein\n"
            "drug_protein,1,gene/protein,3,drug\n"
            "disease_protein,4,disease,1,gene/protein\n"
            "disease_protein,1,gene/protein,4,disease\n"
            "indication,0,drug,2,disease\n"
            "indication,3,drug,4,disease\n"
        )
        graph = self.root / "graph.pt"
        prepare_graph(source, graph)

        dataset = self.root / "split"
        dataset.mkdir()
        header = (
            "drug_index,disease_index,drug_id,disease_id,"
            "contraindication,off_label\n"
        )
        (dataset / "train.csv").write_text(header + "0,2,D0,C2,0,0\n")
        (dataset / "development.csv").write_text(header + "3,4,D3,C4,0,0\n")
        (dataset / "drugs.csv").write_text("drug_index,drug_id\n0,D0\n3,D3\n")
        (dataset / "therapeutic_train.csv").write_text(
            "drug_index,disease_index,relation,drug_id,disease_id\n"
            "0,2,indication,D0,C2\n"
        )

        output = self.root / "run"
        record = train_model(
            graph, dataset, output, epochs=2, width=8, eval_every=1
        )
        self.assertEqual(record["best_epoch"], 1)
        self.assertEqual(record["development_recall_at_20"], 1.0)

        with (output / "predictions.csv").open(newline="") as stream:
            predictions = list(csv.DictReader(stream))
        self.assertEqual(len(predictions), 2)
        self.assertEqual(
            {int(row["drug_index"]) for row in predictions}, {0, 3}
        )
        self.assertEqual({int(row["rank"]) for row in predictions}, {1, 2})

    def test_rejects_invalid_negative_count(self):
        with self.assertRaisesRegex(ValueError, "negatives"):
            train_model(self.root, self.root, self.root, negatives=0)

    def _write_resume_dataset(self) -> Path:
        root = self.root / "resume_data"
        root.mkdir()
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
            "2,4,contraindication,D2,C4\n"
            "3,4,off-label use,D3,C4\n"
        )
        return root

    def test_recovery_matches_uninterrupted_training(self):
        data = self._write_resume_dataset()
        reference = self.root / "reference"
        partial = self.root / "partial"
        resumed = self.root / "resumed"
        settings: TrainingSettings = {
            "epochs": 6, "width": 8, "eval_every": 2, "negatives": 3,
        }

        train_model(data, data, reference, **settings)
        train_model(data, data, partial, stop_after=3, **settings)
        checkpoint = load_snapshot(partial / "latest.json")
        self.assertEqual(checkpoint["position"], 3)
        self.assertEqual(checkpoint["best_epoch"], 2)
        self.assertEqual(
            sorted(
                path.name for path in (partial / "checkpoints").glob("*.pt")
            ),
            ["epoch_000001.pt", "epoch_000002.pt", "epoch_000003.pt"],
        )
        (partial / "model.pt").unlink()
        train_model(
            data, data, resumed, resume=partial / "latest.json", **settings
        )

        for name in (
            "loss.csv", "predictions.csv", "run.json", "metrics.jsonl"
        ):
            with self.subTest(artifact=name):
                self.assertEqual(
                    (reference / name).read_bytes(),
                    (resumed / name).read_bytes(),
                )

        expected = load_snapshot(reference / "latest.json")
        actual = load_snapshot(resumed / "latest.json")
        self.assertEqual(expected.keys(), actual.keys())
        for name in (
            "format", "stage", "run_id", "position", "history", "events",
            "best_epoch", "best_key", "best_metrics", "config", "inputs",
            "model_meta", "sampling_rng",
        ):
            self.assertEqual(expected[name], actual[name])
        for name in ("model", "optimizer", "rng", "best_state"):
            with self.subTest(state=name):
                torch.testing.assert_close(
                    expected[name], actual[name], rtol=0, atol=0
                )

        selected = torch.load(reference / "model.pt", weights_only=True)
        restored = torch.load(resumed / "model.pt", weights_only=True)
        self.assertEqual(selected.keys(), restored.keys())
        torch.testing.assert_close(
            selected["model"], restored["model"], rtol=0, atol=0
        )
        self.assertEqual(
            len(list((reference / "checkpoints").glob("*.pt"))), 6
        )

    def test_recovery_before_first_evaluation_and_invalid_source(self):
        data = self._write_resume_dataset()
        partial = self.root / "partial"
        settings: TrainingSettings = {"epochs": 4, "width": 8, "eval_every": 2}
        train_model(data, data, partial, stop_after=1, **settings)
        checkpoint = load_snapshot(partial / "latest.json")
        self.assertEqual(checkpoint["position"], 1)
        self.assertIsNone(checkpoint["best_state"])
        self.assertFalse((partial / "model.pt").exists())
        train_model(
            data, data, self.root / "resumed",
            resume=partial / "latest.json", **settings,
        )

        training_path = data / "train.csv"
        training_path.write_text(training_path.read_text() + "2,4,D2,C4,0,0\n")
        with self.assertRaisesRegex(ValueError, "configuration or data"):
            train_model(
                data, data, self.root / "invalid",
                resume=partial / "latest.json", **settings,
            )

    def test_checkpoint_pointer_detects_corruption(self):
        data = self._write_resume_dataset()
        output = self.root / "partial"
        train_model(
            data, data, output, epochs=3, width=8,
            eval_every=2, stop_after=1,
        )
        checkpoint = output / "checkpoints" / "epoch_000001.pt"
        with checkpoint.open("ab") as stream:
            stream.write(b"corruption")
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_snapshot(output / "latest.json")
