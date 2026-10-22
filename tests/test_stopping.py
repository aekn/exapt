import tempfile
import unittest
from pathlib import Path
from typing import TypedDict

from exapt.checkpoint import load_snapshot, save_checkpoint
from exapt.pretrain import prepare_pretrain, pretrain_model
from exapt.stopping import loss_progress, ranking_key, ranking_progress
from exapt.train import train_model


class TrainingSettings(TypedDict):
    epochs: int
    width: int
    eval_every: int
    patience: int
    min_epochs: int
    learning_rate: float


class PretrainingSettings(TypedDict):
    steps: int
    width: int
    eval_every: int
    patience: int
    checkpoint_every: int
    learning_rate: float
    batch_size: int


class StoppingTests(unittest.TestCase):
    def test_ranking_selection_ties_and_minimum_progress(self):
        anchor = (0.25, 0.1, 0.2)
        self.assertTrue(ranking_progress((0.26, 0, 0), anchor, 1e-4))
        self.assertFalse(ranking_progress((0.25, 0.10009, 1), anchor, 1e-4))
        self.assertTrue(ranking_progress((0.25, 0.10011, 0), anchor, 1e-4))
        self.assertFalse(ranking_progress((0.24, 1, 1), anchor, 1e-4))
        self.assertFalse(ranking_progress((0.25, 0.1, 0.99), anchor, 1e-4))
        self.assertGreater((0.25, 0.1, 0.3), anchor)
        self.assertGreater((0.25, 0.11, 0), anchor)
        self.assertEqual(ranking_key({
            "recall@20": 0.25, "map": 0.1, "ndcg@20": 0.2,
        }), anchor)

    def test_pretraining_progress_minimum_delta(self):
        self.assertFalse(loss_progress(0.99995, 1., 1e-4))
        self.assertTrue(loss_progress(0.99989, 1., 1e-4))
        self.assertFalse(loss_progress(1.001, 1., 1e-4))

    def test_supervised_stops_and_resumes_with_same_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "nodes.csv").write_text(
                "index,type,id\n1,drug,D1\n2,drug,D2\n3,drug,D3\n"
                "4,disease,C4\n5,disease,C5\n6,gene/protein,P6\n"
            )
            (root / "graph.csv").write_text(
                "x_index,x_type,relation,y_index,y_type\n"
                "1,drug,drug_protein,6,gene/protein\n"
                "6,gene/protein,drug_protein,1,drug\n"
            )
            (root / "drugs.csv").write_text(
                "drug_index,drug_id\n1,D1\n2,D2\n3,D3\n"
            )
            header = (
                "drug_index,disease_index,drug_id,disease_id,"
                "contraindication,off_label\n"
            )
            (root / "train.csv").write_text(
                header + "1,4,D1,C4,0,0\n"
            )
            (root / "development.csv").write_text(
                header + "2,5,D2,C5,0,0\n"
            )
            (root / "therapeutic_train.csv").write_text(
                "drug_index,disease_index,relation,drug_id,disease_id\n"
                "1,4,indication,D1,C4\n"
            )
            settings: TrainingSettings = {
                "epochs": 10, "width": 8, "eval_every": 1, "patience": 2,
                "min_epochs": 0, "learning_rate": 0.,
            }
            reference = root / "reference"
            partial = root / "partial"
            resumed = root / "resumed"
            expected = train_model(root, root, reference, **settings)
            paused = train_model(
                root, root, partial, stop_after=1, **settings,
            )
            self.assertEqual(paused["status"], "manual_stop")
            actual = train_model(
                root, root, resumed,
                resume=partial / "latest.json", **settings,
            )
            self.assertEqual(expected["termination"], "early_stopping")
            self.assertEqual(expected["completed_epochs"], 3)
            self.assertEqual(expected["best_epoch"], 1)
            self.assertEqual(actual, expected)
            self.assertEqual(
                (reference / "metrics.jsonl").read_bytes(),
                (resumed / "metrics.jsonl").read_bytes(),
            )
            self.assertEqual(
                (reference / "predictions.csv").read_bytes(),
                (resumed / "predictions.csv").read_bytes(),
            )
            cp = load_snapshot(resumed / "latest.json")
            self.assertEqual(cp["stale_checks"], 2)
            self.assertEqual(cp["position"], 3)
            self.assertEqual(cp["config"]["optimizer"], "AdamW")
            self.assertEqual(
                cp["optimizer"]["param_groups"][0]["weight_decay"], 1e-4
            )

            old = root / "old-adam.pt"
            cp["config"] = {**cp["config"], "optimizer": "Adam"}
            save_checkpoint(old, cp)
            with self.assertRaisesRegex(ValueError, "configuration or data"):
                train_model(
                    root, root, root / "old-resume",
                    resume=old, **settings,
                )

    def test_map_progress_prevents_early_stop(self):
        metrics = [
            (0.1, 0.01000), (0.1, 0.01006), (0.1, 0.01012),
            (0.1, 0.01018), (0.1, 0.01024),
        ]
        anchor = (-1., -1., -1.)
        selected = (-1., -1., -1.)
        stale = 0
        best_epoch = 0
        for epoch, (recall, ap) in enumerate(metrics, 1):
            current = recall, ap, 0.2
            if ranking_progress(current, anchor, 1e-4):
                anchor = current
                stale = 0
            else:
                stale += 1
            if current > selected:
                selected = current
                best_epoch = epoch
            self.assertLess(stale, 2)
        self.assertEqual(best_epoch, 5)
        self.assertEqual(anchor[1], 0.01024)
        self.assertEqual(stale, 0)

    def test_pretraining_stops_and_resumes_on_validation_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "nodes.csv").write_text(
                "index,type,id\n"
                + "".join(f"{i},drug,D{i}\n" for i in range(120))
            )
            rows = ["x_index,x_type,relation,y_index,y_type\n"]
            for i in range(119):
                rows += [
                    f"{i},drug,drug_drug,{i+1},drug\n",
                    f"{i+1},drug,drug_drug,{i},drug\n",
                ]
            (root / "graph.csv").write_text("".join(rows))
            prepared = root / "pretrain.pt"
            prepare_pretrain(root, prepared)
            settings: PretrainingSettings = {
                "steps": 10, "width": 4, "eval_every": 1, "patience": 2,
                "checkpoint_every": 2, "learning_rate": 0.,
                "batch_size": 32,
            }
            expected = pretrain_model(prepared, root / "reference", **settings)
            paused = pretrain_model(
                prepared, root / "partial", stop_after=1, **settings,
            )
            actual = pretrain_model(
                prepared, root / "resumed",
                resume=root / "partial/latest.json", **settings,
            )
            self.assertEqual(paused["status"], "manual_stop")
            self.assertEqual(expected["termination"], "early_stopping")
            self.assertEqual(expected["completed_steps"], 3)
            self.assertEqual(actual, expected)
            self.assertEqual(
                (root / "reference/metrics.jsonl").read_bytes(),
                (root / "resumed/metrics.jsonl").read_bytes(),
            )
            cp = load_snapshot(root / "resumed/latest.json")
            self.assertEqual(cp["position"], 3)
            self.assertEqual(cp["stale_checks"], 2)
            self.assertEqual(cp["config"]["optimizer"], "AdamW")
            self.assertEqual(
                cp["optimizer"]["param_groups"][0]["weight_decay"], 1e-4
            )
