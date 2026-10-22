import json
import tempfile
import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TypedDict
from unittest.mock import patch

from exapt.checkpoint import load_snapshot, save_snapshot
from exapt.hub import (
    BucketStore, download_dataset, validate_run_id,
    verify_manifest, write_manifest,
)


BUCKET = "exapt-lab/hangar"
DATASET = "exapt-lab/data"
HASH = "a" * 40


class StageSettings(TypedDict):
    program: str
    seed: int
    config: dict[str, Any]
    inputs: dict[str, str]
    source_commit: str
    dataset_revision: str
    manifest_sha256: str


class FakeBucket:
    def __init__(self):
        self.objects: dict[str, bytes] = {}
        self.calls: list[tuple[str, str]] = []
        self.fail: str | None = None

    def upload(self, bucket, *, add=None, delete=None):
        assert bucket == BUCKET
        for source, key in add or []:
            if key == self.fail:
                raise OSError("upload interrupted")
            self.objects[key] = Path(source).read_bytes()
            self.calls.append(("upload", key))
        for key in delete or []:
            self.objects.pop(key)
            self.calls.append(("delete", key))

    def listing(self, bucket, prefix=None, recursive=False):
        assert bucket == BUCKET
        return [
            SimpleNamespace(type="file", path=k, size=len(v))
            for k, v in self.objects.items()
            if k.startswith(prefix or "")
        ]

    def download(self, bucket, *, files, raise_on_missing_files=False):
        assert bucket == BUCKET
        for key, destination in files:
            if key not in self.objects:
                raise OSError(f"missing {key}")
            path = Path(destination)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(self.objects[key])

    @contextmanager
    def mocked(self):
        with ExitStack() as stack:
            for context in (
                patch("huggingface_hub.bucket_info", create=True,
                      return_value=SimpleNamespace(private=True)),
                patch("huggingface_hub.batch_bucket_files", create=True,
                      side_effect=self.upload),
                patch("huggingface_hub.list_bucket_tree", create=True,
                      side_effect=self.listing),
                patch("huggingface_hub.download_bucket_files", create=True,
                      side_effect=self.download),
            ):
                stack.enter_context(context)
            yield


def snapshot(directory: Path, position: int) -> None:
    save_snapshot(directory, "baseline", position, {
        "format": 1, "run_id": "001", "stage": "baseline",
        "position": position, "model": {"weight": position},
    })


class HubTests(unittest.TestCase):
    def test_dataset_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "graph.pt").write_bytes(b"graph")
            (root / "split").mkdir()
            (root / "split" / "train.csv").write_text("drug,disease\n")
            write_manifest(root)
            self.assertEqual(verify_manifest(root), 2)
            (root / "unlisted.csv").write_text("stale")
            with self.assertRaisesRegex(ValueError, "membership"):
                verify_manifest(root)
            (root / "unlisted.csv").unlink()
            (root / "split" / "train.csv").write_text("changed")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                verify_manifest(root)

    def test_dataset_download_ignores_unlisted_repo_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            remote = root / "remote"
            remote.mkdir()
            (remote / "graph.pt").write_bytes(b"graph")
            (remote / "split").mkdir()
            (remote / "split" / "train.csv").write_text("drug,disease\n")
            write_manifest(remote)
            (remote / "README.md").write_text("dataset card")

            output = root / "download"
            revision = "a" * 40

            def manifest_download(*, filename, local_dir, **kwargs):
                self.assertEqual(filename, "manifest.json")
                self.assertEqual(kwargs["revision"], revision)
                local_dir.mkdir(parents=True, exist_ok=True)
                destination = local_dir / filename
                destination.write_bytes((remote / filename).read_bytes())
                return str(destination)

            def snapshot(*, local_dir, allow_patterns, **kwargs):
                self.assertEqual(kwargs["revision"], revision)
                self.assertEqual(set(allow_patterns), {
                    "manifest.json", "graph.pt", "split/train.csv",
                })
                for name in allow_patterns:
                    source = remote / name
                    destination = local_dir / name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(source.read_bytes())

            with (
                patch("huggingface_hub.hf_hub_download",
                      side_effect=manifest_download),
                patch("huggingface_hub.snapshot_download",
                      side_effect=snapshot),
            ):
                result = download_dataset(DATASET, revision, output)
            self.assertEqual(result["files"], 2)
            self.assertFalse((output / "README.md").exists())
            self.assertEqual(verify_manifest(output), 2)

    def test_dataset_requires_pinned_revision(self):
        with self.assertRaisesRegex(ValueError, "commit SHA"):
            download_dataset("exapt-lab/data", "main", Path("data"))

    def test_numeric_program_ids(self):
        for value in ("001", "002", "105"):
            validate_run_id(value)
        for value in ("local", "baseline-0001", "../001", "1", "001/fit"):
            with self.assertRaises(ValueError):
                validate_run_id(value)

    def test_identity_is_immutable_and_program_scoped(self):
        fake = FakeBucket()
        with fake.mocked():
            store = BucketStore(BUCKET, "002", "pretrain")
            settings: StageSettings = {
                "program": "exaptgnn-v1",
                "seed": 163,
                "config": {"steps": 100},
                "inputs": {"pretrain.pt": "filehash"},
                "source_commit": HASH,
                "dataset_revision": HASH,
                "manifest_sha256": "b" * 64,
            }
            store.prepare_stage(**settings)
            store.prepare_stage(**settings)

            changed_config: StageSettings = {
                **settings, "config": {"steps": 200},
            }
            with self.assertRaisesRegex(ValueError, "identity/config"):
                store.prepare_stage(**changed_config)

            changed_source: StageSettings = {
                **settings, "source_commit": "main",
            }
            with self.assertRaisesRegex(ValueError, "full hexadecimal SHA"):
                store.prepare_stage(**changed_source)

            fit = BucketStore(BUCKET, "002", "fit")
            fit_settings: StageSettings = {
                **settings, "config": {"epochs": 500},
            }
            fit.prepare_stage(**fit_settings)
            self.assertIn("002/run.json", fake.objects)
            self.assertIn("002/pretrain/config.json", fake.objects)
            self.assertIn("002/fit/config.json", fake.objects)

            with self.assertRaisesRegex(ValueError, "Run ID"):
                BucketStore(BUCKET, "001", "fit").prepare_stage(**settings)

    def test_immutable_config_accepts_json_roundtrip(self):
        fake = FakeBucket()
        settings: StageSettings = {
            "program": "exaptgnn-v1",
            "seed": 163,
            "config": {
                "steps": 3, "optimizer": "AdamW",
                "betas": (0.9, 0.999),
            },
            "inputs": {"pretrain.pt": "inputhash"},
            "source_commit": HASH,
            "dataset_revision": HASH,
            "manifest_sha256": "b" * 64,
        }
        with fake.mocked():
            first = BucketStore(BUCKET, "002", "pretrain")
            first.prepare_stage(**settings)
            saved = json.loads(fake.objects["002/pretrain/config.json"])
            self.assertEqual(saved["config"]["betas"], [0.9, 0.999])

            resumed = BucketStore(BUCKET, "002", "pretrain")
            resumed.prepare_stage(**settings)

            changed: StageSettings = {
                **settings,
                "config": {**settings["config"], "betas": (0.8, 0.999)},
            }
            with self.assertRaisesRegex(ValueError, "identity/config"):
                resumed.prepare_stage(**changed)

    def test_fresh_runtime_pretrain_restores_optimizer(self):
        from exapt.pretrain import prepare_pretrain, pretrain_model

        fake = FakeBucket()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            (data / "nodes.csv").write_text(
                "index,type,id\n" + "".join(
                    f"{i},drug,D{i}\n" for i in range(120)
                )
            )
            (data / "graph.csv").write_text(
                "x_index,x_type,relation,y_index,y_type\n" + "".join(
                    f"{i},drug,drug_drug,{i + 1},drug\n"
                    f"{i + 1},drug,drug_drug,{i},drug\n"
                    for i in range(119)
                )
            )
            prepared = root / "pretrain.pt"
            prepare_pretrain(data, prepared)
            settings = {
                "steps": 3, "width": 4, "batch_size": 32,
                "eval_every": 1, "checkpoint_every": 1,
                "run_id": "998", "bucket": BUCKET,
                "source_commit": HASH, "dataset_revision": HASH,
                "manifest_sha256": "b" * 64,
            }
            with fake.mocked(), patch(
                "exapt.hub.verify_training_inputs", return_value=None,
            ):
                pretrain_model(
                    prepared, root / "initial", stop_after=1, **settings,
                )
                restored = root / "fresh"
                pointer = BucketStore(
                    BUCKET, "998", "pretrain",
                ).recover(restored)
                self.assertEqual(load_snapshot(pointer)["position"], 1)
                result = pretrain_model(
                    prepared, restored, resume=pointer, **settings,
                )
                self.assertEqual(result["completed_steps"], 3)
                latest = json.loads(
                    fake.objects["998/pretrain/latest.json"]
                )
                self.assertEqual(latest["current"]["step"], 3)

    def test_two_recovery_snapshots_and_fallback(self):
        fake = FakeBucket()
        with fake.mocked(), tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "run"
            output.mkdir()
            (output / "metrics.jsonl").write_text("test\n")
            store = BucketStore(BUCKET, "001", "fit", sync_seconds=900)
            for epoch in (1, 2, 3):
                snapshot(output, epoch)
                store.publish_snapshot(output, force=True)

            prefix = "001/fit/"
            self.assertIn(prefix + "resume/e0002.pt", fake.objects)
            self.assertIn(prefix + "resume/e0003.pt", fake.objects)
            self.assertNotIn(prefix + "resume/e0001.pt", fake.objects)
            pointer = json.loads(fake.objects[prefix + "latest.json"])
            self.assertEqual(pointer["current"]["epoch"], 3)
            self.assertEqual(pointer["previous"]["epoch"], 2)
            self.assertEqual(
                len(list((output / "checkpoints").glob("*.pt"))), 2
            )
            before = len(fake.calls)
            self.assertFalse(store.publish_snapshot(output, force=False))
            self.assertEqual(before, len(fake.calls))

            # A fresh Colab directory recovers the selected run, not a new ID.
            recovered = store.recover(root / "fresh")
            self.assertEqual(load_snapshot(recovered)["model"]["weight"], 3)
            self.assertEqual(recovered.parent.name, "fresh")

            # Corruption of the newest object falls back to the verified prior.
            fake.objects[prefix + "resume/e0003.pt"] = b"corrupted"
            fallback = store.recover(root / "fallback")
            self.assertEqual(load_snapshot(fallback)["model"]["weight"], 2)

    def test_fallback_replay_uses_new_immutable_name(self):
        fake = FakeBucket()
        with fake.mocked(), tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "run"
            run.mkdir()
            store = BucketStore(BUCKET, "001", "fit")
            for epoch in (1, 2, 3):
                snapshot(run, epoch)
                store.publish_snapshot(run, force=True)
            prefix = "001/fit/"
            fake.objects[prefix + "resume/e0003.pt"] = b"damaged"
            recovered = root / "recovered"
            pointer = store.recover(recovered)
            self.assertEqual(load_snapshot(pointer)["position"], 2)
            snapshot(recovered, 3)
            store.publish_snapshot(recovered, force=True)
            self.assertIn(prefix + "resume/e0003-r01.pt", fake.objects)
            latest = json.loads(fake.objects[prefix + "latest.json"])
            self.assertEqual(latest["current"]["path"],
                             "resume/e0003-r01.pt")
            self.assertEqual(latest["previous"]["path"],
                             "resume/e0002.pt")
            self.assertNotIn(prefix + "resume/e0003.pt", fake.objects)

    def test_upload_failure_does_not_advance_pointer(self):
        fake = FakeBucket()
        with fake.mocked(), tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "metrics.jsonl").write_text("test\n")
            store = BucketStore(BUCKET, "001", "fit")
            snapshot(output, 1)
            store.publish_snapshot(output, force=True)
            remote = "001/fit/latest.json"
            original = fake.objects[remote]
            snapshot(output, 2)
            fake.fail = "001/fit/resume/e0002.pt"
            with self.assertRaisesRegex(OSError, "upload interrupted"):
                store.publish_snapshot(output, force=True)
            self.assertEqual(fake.objects[remote], original)

    def test_selected_and_milestone_are_not_optimizer_snapshots(self):
        fake = FakeBucket()
        with fake.mocked(), tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "model.pt").write_bytes(b"selected")
            milestone = output / "milestones" / "e0025.pt"
            milestone.parent.mkdir()
            milestone.write_bytes(b"weights")
            store = BucketStore(BUCKET, "001", "fit")
            store.publish_selected(output)
            store.publish_milestone(milestone)
            self.assertEqual(
                fake.objects["001/fit/selected/model.pt"], b"selected"
            )
            self.assertEqual(
                fake.objects["001/fit/milestones/e0025.pt"], b"weights"
            )

    def test_public_bucket_requires_explicit_permission(self):
        with patch("huggingface_hub.bucket_info", create=True,
                   return_value=SimpleNamespace(private=False)):
            with self.assertRaisesRegex(ValueError, "public"):
                BucketStore(BUCKET, "001", "fit")
            BucketStore(BUCKET, "001", "fit", allow_public=True)


if __name__ == "__main__":
    unittest.main()
