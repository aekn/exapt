import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from exapt import main
from exapt.checkpoint import load_snapshot, sha256
from exapt.preflight import overfit, profile
from exapt.pretrain import prepare_pretrain


class PreflightTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        data = self.root / "smoke"
        data.mkdir()
        (data / "nodes.csv").write_text(
            "index,type,id\n"
            "1,drug,D1\n2,drug,D2\n3,drug,D3\n"
            "4,disease,C4\n5,disease,C5\n6,gene/protein,P6\n"
        )
        (data / "graph.csv").write_text(
            "x_index,x_type,relation,y_index,y_type\n"
            "1,drug,drug_protein,6,gene/protein\n"
            "6,gene/protein,drug_protein,1,drug\n"
            "4,disease,disease_protein,6,gene/protein\n"
            "6,gene/protein,disease_protein,4,disease\n"
        )
        (data / "drugs.csv").write_text(
            "drug_index,drug_id\n1,D1\n2,D2\n3,D3\n"
        )
        header = (
            "drug_index,disease_index,drug_id,disease_id,"
            "contraindication,off_label\n"
        )
        (data / "train.csv").write_text(
            header + "1,4,D1,C4,0,0\n"
        )
        (data / "development.csv").write_text(
            header + "2,5,D2,C5,0,0\n"
        )
        (data / "therapeutic_train.csv").write_text(
            "drug_index,disease_index,relation,drug_id,disease_id\n"
            "1,4,indication,D1,C4\n"
            "2,4,contraindication,D2,C4\n"
            "3,4,off-label use,D3,C4\n"
        )
        self.dataset = data

    def test_fixed_training_only_overfit(self):
        result = overfit(
            self.dataset, width=8, steps=120, learning_rate=0.02,
        )
        self.assertTrue(result["passed"])
        self.assertEqual(len(result["relations"]), 3)
        self.assertGreater(result["initial_loss"], result["final_loss"])
        self.assertGreater(result["reduction"], 0.5)

    def test_overfit_does_not_read_development_labels(self):
        (self.dataset / "development.csv").unlink()
        result = overfit(
            self.dataset, width=8, steps=120, learning_rate=0.02,
        )
        self.assertTrue(result["passed"])

    def test_one_update_uses_production_trainer(self):
        output = self.root / "profile"
        result = profile(
            "fit", self.dataset, output, dataset=self.dataset,
            width=8, device="cpu",
        )
        self.assertEqual(result["status"], "completed")
        self.assertGreater(result["checkpoint_bytes"], 0)
        self.assertIsNone(result["cuda_peak_allocated_bytes"])
        self.assertIsNone(result["cuda_peak_reserved_bytes"])
        self.assertEqual(json.loads(
            (output / "preflight.json").read_text(),
        ), result)

        checkpoint = load_snapshot(output / "latest.json")
        self.assertEqual(checkpoint["position"], 1)
        self.assertEqual(checkpoint["config"]["optimizer"], "AdamW")
        snapshots = list((output / "checkpoints").glob("*.pt"))
        self.assertEqual(len(snapshots), 1)
        self.assertEqual(result["checkpoint_sha256"], sha256(snapshots[0]))

        with self.assertRaises(FileExistsError):
            profile(
                "fit", self.dataset, output, dataset=self.dataset,
                width=8, device="cpu",
            )

    def test_pretraining_profiles_one_production_update(self):
        graph = self.root / "chain"
        graph.mkdir()
        graph.joinpath("nodes.csv").write_text(
            "index,type,id\n"
            + "".join(f"{node},drug,D{node}\n" for node in range(120))
        )
        rows = ["x_index,x_type,relation,y_index,y_type\n"]
        for source in range(119):
            rows.extend((
                f"{source},drug,drug_drug,{source + 1},drug\n",
                f"{source + 1},drug,drug_drug,{source},drug\n",
            ))
        graph.joinpath("graph.csv").write_text("".join(rows))

        prepared = self.root / "pretrain.pt"
        prepare_pretrain(graph, prepared)
        output = self.root / "pretrain-profile"
        result = profile("pretrain", prepared, output, width=4, device="cpu")

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["stage"], "pretrain")
        self.assertGreater(result["checkpoint_bytes"], 0)
        self.assertEqual(load_snapshot(output / "latest.json")["position"], 1)

    def test_cli_preflight_dispatch(self):
        cases = (
            (
                ["overfit", str(self.dataset), "--width", "8"],
                "overfit", None, {"width": 8, "steps": 300},
            ),
            (
                [
                    "profile-fit", str(self.dataset), str(self.dataset),
                    str(self.root / "fit"), "--device", "cpu",
                ],
                "profile", "fit", {"device": "cpu", "width": 128},
            ),
            (
                [
                    "profile-pretrain", str(self.root / "pretrain.pt"),
                    str(self.root / "pretrain"), "--width", "4",
                ],
                "profile", "pretrain", {"device": "cuda", "width": 4},
            ),
        )
        for arguments, function, stage, expected in cases:
            with self.subTest(arguments=arguments):
                with (
                    patch(f"exapt.preflight.{function}",
                          return_value={"check": "mocked"}) as mocked,
                    patch("sys.argv", ["exapt", "preflight", *arguments]),
                    redirect_stdout(io.StringIO()) as output,
                ):
                    main()
                self.assertEqual(json.loads(output.getvalue()), {
                    "check": "mocked",
                })
                args, kwargs = mocked.call_args
                if stage is not None:
                    self.assertEqual(args[0], stage)
                for key, value in expected.items():
                    self.assertEqual(kwargs[key], value)

    def test_invalid_preflight_configuration(self):
        with self.assertRaises(ValueError):
            overfit(self.dataset, steps=0)
        with self.assertRaises(ValueError):
            overfit(self.dataset, width=0)
        with self.assertRaises(ValueError):
            overfit(self.dataset, learning_rate=0.0)
        with self.assertRaises(ValueError):
            overfit(self.dataset, learning_rate=float("nan"))
        with self.assertRaises(ValueError):
            overfit(self.dataset, prototype_k=0)

        with self.assertRaises(ValueError):
            profile("fit", self.dataset, self.root / "empty")
        with self.assertRaisesRegex(ValueError, "Unknown profiling stage"):
            profile("other", self.dataset, self.root / "empty")
        with self.assertRaisesRegex(ValueError, "width"):
            profile("pretrain", self.dataset, self.root / "empty", width=0)
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA"):
                profile("pretrain", self.dataset, self.root / "empty")
        existing = self.root / "existing.txt"
        existing.write_text("content")
        with self.assertRaises(FileExistsError):
            profile("pretrain", self.dataset, existing, device="cpu")


if __name__ == "__main__":
    unittest.main()
