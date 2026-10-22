import csv
import tempfile
import unittest
from pathlib import Path
from typing import TypedDict

import torch

from exapt.checkpoint import load_snapshot
from exapt.graph import load_graph
from exapt.pretrain import _sample_negatives, prepare_pretrain, pretrain_model


def _write_chain(root: Path, nodes: int, relations: tuple[str, ...]) -> None:
    (root / "nodes.csv").write_text(
        "index,type,id\n"
        + "".join(f"{node},drug,D{node}\n" for node in range(nodes))
    )

    rows = ["x_index,x_type,relation,y_index,y_type\n"]
    for relation in relations:
        for source in range(nodes - 1):
            target = source + 1
            rows.extend((
                f"{source},drug,{relation},{target},drug\n",
                f"{target},drug,{relation},{source},drug\n",
            ))
    (root / "graph.csv").write_text("".join(rows))


class PretrainSettings(TypedDict):
    steps: int
    width: int
    eval_every: int
    batch_size: int
    checkpoint_every: int


class PretrainingTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def test_global_holdout_and_best_encoder_checkpoint(self):
        root = self.root
        _write_chain(root, 120, ("drug_drug",))

        graph = load_graph(root)
        prepared = root / "pretrain.pt"
        report = prepare_pretrain(root, prepared)
        saved = torch.load(prepared, weights_only=True)

        # 119 undirected pairs, each represented by two directed edges
        training_edges = saved["positives"][0]
        validation_edges = saved["validation"][0]
        retained = saved["adjacency"][0]
        self.assertEqual((len(training_edges), len(validation_edges)), (42, 4))
        self.assertEqual(
            graph.adjacency[0]._nnz(),
            retained._nnz() + len(training_edges) + len(validation_edges),
        )
        self.assertEqual(report["prediction_edges"], len(training_edges))
        self.assertEqual(report["validation_edges"], len(validation_edges))

        visible = {(src, dst) for dst, src in retained.indices().T.tolist()}
        held = {
            tuple(pair)
            for pair in torch.cat((training_edges, validation_edges)).tolist()
        }
        self.assertTrue(visible.isdisjoint(held))
        self.assertTrue(visible.isdisjoint({(dst, src) for src, dst in held}))
        train_pairs = {tuple(pair) for pair in training_edges.tolist()}
        validation_pairs = {tuple(pair) for pair in validation_edges.tolist()}
        self.assertTrue(train_pairs.isdisjoint(validation_pairs))
        original = {
            (src, dst) for dst, src in graph.adjacency[0].indices().T.tolist()
        }
        self.assertEqual(original, visible | held)

        row_sum = torch.sparse.sum(retained, dim=1).to_dense()
        nonempty = row_sum[row_sum > 0]
        torch.testing.assert_close(nonempty, torch.ones_like(nonempty))

        first = root / "first"
        second = root / "second"
        result = pretrain_model(
            prepared, first, steps=3, width=4, eval_every=1, batch_size=64
        )
        repeat = pretrain_model(
            prepared, second, steps=3, width=4, eval_every=1, batch_size=64
        )
        self.assertEqual(result, repeat)
        for name in ("loss.csv", "validation.csv", "run.json"):
            self.assertEqual(
                (first / name).read_bytes(), (second / name).read_bytes()
            )

        with (first / "loss.csv").open(newline="") as stream:
            history = list(csv.DictReader(stream))
        losses = [float(row["validation_loss"]) for row in history]
        best = min(losses)
        self.assertEqual(result["best_validation_loss"], best)
        best_step = next(
            step for step, value in enumerate(losses, 1) if value == best
        )
        self.assertEqual(result["best_step"], best_step)

        selected = root / "selected"
        pretrain_model(
            prepared, selected, steps=best_step, width=4,
            eval_every=1, batch_size=64,
        )
        checkpoint = torch.load(first / "model.pt", weights_only=True)
        repeated = torch.load(second / "model.pt", weights_only=True)
        stopped = torch.load(selected / "model.pt", weights_only=True)
        self.assertEqual(checkpoint["width"], 4)
        self.assertEqual(
            checkpoint["encoder"].keys(), repeated["encoder"].keys()
        )
        for name, value in checkpoint["encoder"].items():
            torch.testing.assert_close(
                value, repeated["encoder"][name], rtol=0, atol=0
            )
            torch.testing.assert_close(
                value, stopped["encoder"][name], rtol=0, atol=0
            )

    def test_same_node_pair_has_one_assignment_across_relations(self):
        root = self.root
        _write_chain(root, 120, ("drug_drug", "other_drug_relation"))
        prepare_pretrain(root, root / "pretrain.pt")
        saved = torch.load(root / "pretrain.pt", weights_only=True)

        for partition in ("positives", "validation"):
            first_relation = {
                tuple(pair) for pair in saved[partition][0].tolist()
            }
            second_relation = {
                tuple(pair) for pair in saved[partition][1].tolist()
            }
            self.assertEqual(first_relation, second_relation)

        held = {
            tuple(pair)
            for partition in ("positives", "validation")
            for pairs in saved[partition]
            for pair in pairs.tolist()
        }
        for matrix in saved["adjacency"]:
            visible = {
                (src, dst) for dst, src in matrix.indices().T.tolist()
            }
            self.assertTrue(visible.isdisjoint(held))

    def test_negative_sampling_excludes_observed_edges(self):
        sources = torch.zeros(200, dtype=torch.long)
        pool = torch.arange(8)
        known = torch.arange(4)

        first = _sample_negatives(
            sources, pool, known, 8, torch.Generator().manual_seed(163)
        )
        repeated = _sample_negatives(
            sources, pool, known, 8, torch.Generator().manual_seed(163)
        )
        torch.testing.assert_close(first, repeated, rtol=0, atol=0)
        self.assertTrue(torch.all((first >= 4) & (first < 8)))

    def test_pretraining_requires_validation_pairs(self):
        root = self.root
        _write_chain(root, 11, ("drug_drug",))
        prepared = root / "pretrain.pt"
        prepare_pretrain(root, prepared)
        with self.assertRaisesRegex(ValueError, "No reconstruction validation"):
            pretrain_model(prepared, root / "run", steps=1, width=4)

    def test_resume_matches_uninterrupted_pretraining(self):
        root = self.root
        _write_chain(root, 120, ("drug_drug",))
        prepared = root / "pretrain.pt"
        prepare_pretrain(root, prepared)
        settings: PretrainSettings = {
            "steps": 6, "width": 4, "eval_every": 2,
            "batch_size": 32, "checkpoint_every": 2,
        }
        reference = root / "reference"
        partial = root / "partial"
        resumed = root / "resumed"

        pretrain_model(prepared, reference, **settings)
        pretrain_model(prepared, partial, stop_after=3, **settings)
        checkpoint = load_snapshot(partial / "latest.json")
        self.assertEqual(checkpoint["position"], 3)
        self.assertEqual(checkpoint["best_step"], 2)
        self.assertEqual(checkpoint["config"], settings | {
            "seed": 163, "learning_rate": 0.001,
            "optimizer": "AdamW", "weight_decay": 1e-4,
            "betas": (0.9, 0.999), "eps": 1e-8, "foreach": False,
            "patience": 8, "loss_min_delta": 1e-4,
        })
        self.assertEqual(
            sorted(
                path.name for path in (partial / "checkpoints").glob("*.pt")
            ),
            ["step_000002.pt", "step_000003.pt"],
        )
        (partial / "model.pt").unlink()
        pretrain_model(
            prepared, resumed, resume=partial / "latest.json", **settings
        )

        for name in ("loss.csv", "validation.csv", "run.json", "metrics.jsonl"):
            with self.subTest(artifact=name):
                self.assertEqual(
                    (reference / name).read_bytes(),
                    (resumed / name).read_bytes(),
                )

        expected = load_snapshot(reference / "latest.json")
        actual = load_snapshot(resumed / "latest.json")
        self.assertEqual(expected.keys(), actual.keys())
        for name in (
            "format", "stage", "run_id", "position", "config", "input_sha256",
            "graph_inputs", "history", "validation_history", "events",
            "best_step", "best_validation_loss",
        ):
            self.assertEqual(expected[name], actual[name])
        for name in ("encoder", "decoder", "optimizer", "rng", "best_state"):
            with self.subTest(state=name):
                torch.testing.assert_close(
                    expected[name], actual[name], rtol=0, atol=0
                )

        selected = torch.load(reference / "model.pt", weights_only=True)
        restored = torch.load(resumed / "model.pt", weights_only=True)
        self.assertEqual(selected.keys(), restored.keys())
        torch.testing.assert_close(
            selected["encoder"], restored["encoder"], rtol=0, atol=0
        )

    def test_resume_before_validation_and_bad_input(self):
        root = self.root
        _write_chain(root, 120, ("drug_drug",))
        prepared = root / "pretrain.pt"
        prepare_pretrain(root, prepared)
        settings: PretrainSettings = {
            "steps": 4, "width": 4, "eval_every": 2,
            "batch_size": 32, "checkpoint_every": 2,
        }
        partial = root / "partial"
        pretrain_model(prepared, partial, stop_after=1, **settings)
        checkpoint = load_snapshot(partial / "latest.json")
        self.assertEqual(checkpoint["best_step"], 0)
        self.assertIsNone(checkpoint["best_state"])
        pretrain_model(
            prepared, root / "resumed", resume=partial / "latest.json",
            **settings,
        )
        prepare_pretrain(root, prepared, seed=7)
        with self.assertRaisesRegex(ValueError, "configuration or data"):
            pretrain_model(
                prepared, root / "changed", resume=partial / "latest.json",
                **settings,
            )
