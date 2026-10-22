import json
import re
import subprocess
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from .checkpoint import sha256


DATASET = "exapt-lab/data"
HANGAR = "exapt-lab/hangar"
_SOURCE = "aekn/exapt"
_REVISION = re.compile(r"[0-9a-f]{40}")
_GIT_ID = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_RUN = re.compile(r"[0-9]{3,}")


def validate_run_id(run_id: str) -> None:
    if not _RUN.fullmatch(run_id):
        raise ValueError("Run ID must be sequential digits, e.g. 001 or 002")


def write_manifest(directory: Path) -> Path:
    files = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        relative = path.relative_to(directory)
        if any(part.startswith(".") for part in relative.parts):
            continue
        files[relative.as_posix()] = {
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
        }
    if not files:
        raise ValueError("No prepared data files found")
    manifest = directory / "manifest.json"
    manifest.write_text(json.dumps({
        "format": 1, "files": files,
    }, indent=2) + "\n")
    return manifest


def verify_manifest(directory: Path) -> int:
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("format") != 1:
        raise ValueError("Unsupported dataset manifest")
    expected = set(manifest["files"])
    present = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file()
        and path.name != "manifest.json"
        and not any(
            part.startswith(".") for part in path.relative_to(directory).parts
        )
    }
    if expected != present:
        raise ValueError("Dataset files differ from manifest membership")
    for relative, record in manifest["files"].items():
        name = Path(relative)
        if name.is_absolute() or ".." in name.parts:
            raise ValueError(f"Invalid dataset path: {relative}")
        path = directory / name
        if (
            path.stat().st_size != record["size_bytes"]
            or sha256(path) != record["sha256"]
        ):
            raise ValueError(f"Data checksum mismatch: {relative}")
    return len(manifest["files"])


def download_dataset(
    repo_id: str, revision: str, directory: Path,
) -> dict[str, Any]:
    if not _REVISION.fullmatch(revision):
        raise ValueError("Dataset revision must be a full commit SHA")
    from huggingface_hub import hf_hub_download, snapshot_download

    manifest = hf_hub_download(
        repo_id=repo_id, repo_type="dataset", revision=revision,
        filename="manifest.json", local_dir=directory,
    )
    files = json.loads(Path(manifest).read_text())["files"]
    for relative in files:
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Invalid dataset path: {relative}")

    snapshot_download(
        repo_id=repo_id, repo_type="dataset", revision=revision,
        local_dir=directory, allow_patterns=["manifest.json", *files],
    )
    count = verify_manifest(directory)
    return {"repo": repo_id, "revision": revision, "files": count}


def verify_training_inputs(
    source_commit: str, dataset_root: Path, manifest_sha256: str,
) -> None:
    if not _GIT_ID.fullmatch(source_commit):
        raise ValueError("Source commit must be a full Git object ID")
    source_root = Path(__file__).resolve().parents[2]
    algorithm = subprocess.run(
        ["git", "rev-parse", "--show-object-format"], cwd=source_root,
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    if len(source_commit) != {"sha1": 40, "sha256": 64}[algorithm]:
        raise ValueError("Source commit does not match Git object format")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source_root,
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    if head != source_commit:
        raise ValueError("Running source differs from pinned Git commit")
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=source_root,
        capture_output=True, text=True, check=True,
    ).stdout
    if dirty:
        raise ValueError("Remote training requires a clean source tree")
    manifest = dataset_root / "manifest.json"
    if sha256(manifest) != manifest_sha256:
        raise ValueError("Local prepared dataset manifest differs")
    verify_manifest(dataset_root)


class BucketStore:
    def __init__(
        self, bucket: str, run_id: str, stage: str, *,
        allow_public: bool = False, sync_seconds: float = 900,
    ) -> None:
        validate_run_id(run_id)
        if stage not in ("pretrain", "fit"):
            raise ValueError("Bucket stage must be pretrain or fit")
        if sync_seconds <= 0:
            raise ValueError("Sync interval must be positive")
        from huggingface_hub import bucket_info

        info = bucket_info(bucket)
        if not info.private and not allow_public:
            raise ValueError("Bucket is public; explicit permission required")
        self.bucket = bucket
        self.run_id = run_id
        self.stage = stage
        self.sync_seconds = sync_seconds
        self.last_sync: float | None = None
        self.uploads = 0

    def _remote(self, relative: str) -> str:
        return f"{self.run_id}/{self.stage}/{relative}"

    def _upload(self, path: Path, remote: str) -> None:
        from huggingface_hub import batch_bucket_files

        batch_bucket_files(self.bucket, add=[(path, remote)])

    def _download(self, remote: str, destination: Path) -> None:
        from huggingface_hub import download_bucket_files

        destination.parent.mkdir(parents=True, exist_ok=True)
        download_bucket_files(
            self.bucket, files=[(remote, destination)],
            raise_on_missing_files=True,
        )

    def _listing(self, prefix: str) -> list[Any]:
        from huggingface_hub import list_bucket_tree

        return [
            item
            for item in list_bucket_tree(
                self.bucket, prefix=prefix, recursive=True,
            )
            if item.type == "file"
        ]

    def _verify_size(self, remote: str, size: int) -> None:
        if not any(
            item.path == remote and item.size == size
            for item in self._listing(remote)
        ):
            raise OSError(f"Bucket file not visible: {remote}")

    def _immutable(self, path: Path, remote: str) -> None:
        digest = sha256(path)
        if any(item.path == remote for item in self._listing(remote)):
            with TemporaryDirectory() as temporary:
                previous = Path(temporary) / "existing"
                self._download(remote, previous)
                if sha256(previous) != digest:
                    raise FileExistsError(
                        f"Immutable Bucket file differs: {remote}"
                    )
        else:
            self._upload(path, remote)
        self._verify_size(remote, path.stat().st_size)

    def _read_json(self, remote: str) -> dict[str, Any] | None:
        if not any(item.path == remote for item in self._listing(remote)):
            return None
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "value.json"
            self._download(remote, path)
            return json.loads(path.read_text())

    def _publish_json(self, value: dict[str, Any], remote: str) -> None:
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "value.json"
            path.write_text(json.dumps(value, indent=2) + "\n")
            self._upload(path, remote)
            self._verify_size(remote, path.stat().st_size)
            # verify remote metadata bytes, not only size
            downloaded = path.with_name("confirmed.json")
            self._download(remote, downloaded)
            if sha256(path) != sha256(downloaded):
                raise OSError(f"Bucket metadata mismatch: {remote}")

    def prepare_stage(
        self, *, program: str, seed: int, config: dict[str, Any],
        inputs: dict[str, str], source_commit: str,
        dataset_revision: str, manifest_sha256: str,
    ) -> None:
        if not _GIT_ID.fullmatch(source_commit):
            raise ValueError("Source commit must be a full hexadecimal SHA")
        if not _REVISION.fullmatch(dataset_revision):
            raise ValueError("Dataset revision must be a full hexadecimal SHA")
        if not _SHA256.fullmatch(manifest_sha256):
            raise ValueError("Dataset manifest must be a SHA-256 digest")
        if program not in ("rgcn-distmult", "exaptgnn-v1"):
            raise ValueError("Unknown experiment program")
        if self.run_id == "001" and (
            program != "rgcn-distmult" or self.stage != "fit"
        ):
            raise ValueError("Run ID and experiment program disagree")
        if self.run_id == "002" and program != "exaptgnn-v1":
            raise ValueError("Run ID and experiment program disagree")

        stages = (
            {"fit": "fit/config.json"}
            if program == "rgcn-distmult"
            else {
                "pretrain": "pretrain/config.json",
                "fit": "fit/config.json",
            }
        )
        run = {
            "schema_version": 1,
            "id": self.run_id,
            "model": {"name": program},
            "seed": seed,
            "source": {"repository": _SOURCE, "commit": source_commit},
            "data": {
                "repository": DATASET,
                "revision": dataset_revision,
                "manifest_sha256": manifest_sha256,
            },
            "stages": stages,
        }
        # normalize JSON arrays before comparing remote stage metadata
        frozen = json.loads(json.dumps({
            "schema_version": 1, "config": config, "inputs": inputs,
        }))

        for remote, value in (
            (f"{self.run_id}/run.json", run),
            (self._remote("config.json"), frozen),
        ):
            existing = self._read_json(remote)
            if existing is not None and existing != value:
                raise ValueError(
                    f"Remote run identity/config differs: {remote}"
                )
            if existing is None:
                with TemporaryDirectory() as temporary:
                    path = Path(temporary) / "immutable.json"
                    path.write_text(json.dumps(value, indent=2) + "\n")
                    self._immutable(path, remote)

    def publish_selected(self, directory: Path) -> None:
        model = directory / "model.pt"
        if model.is_file():
            remote = self._remote("selected/model.pt")
            self._upload(model, remote)
            self._verify_size(remote, model.stat().st_size)

    def download_selected(self, destination: Path) -> Path:
        remote = self._remote("selected/model.pt")
        entries = [
            item for item in self._listing(remote) if item.path == remote
        ]
        if len(entries) != 1:
            raise FileNotFoundError(f"Selected model not found: {remote}")
        self._download(remote, destination)
        if destination.stat().st_size != entries[0].size:
            raise ValueError("Selected model download size mismatch")
        return destination

    def publish_milestone(self, path: Path) -> None:
        remote = self._remote(f"milestones/{path.name}")
        self._immutable(path, remote)

    def _checkpoint_entry(self, pointer: dict[str, Any]) -> dict[str, Any]:
        position = int(pointer["position"])
        name = (
            f"s{position:04d}.pt" if self.stage == "pretrain"
            else f"e{position:04d}.pt"
        )
        key = "step" if self.stage == "pretrain" else "epoch"
        return {
            "path": f"resume/{name}", key: position,
            "bytes": pointer["size_bytes"], "sha256": pointer["sha256"],
        }

    def _verify_entry(self, entry: dict[str, Any], directory: Path) -> Path:
        name = Path(entry["path"])
        if (
            name.is_absolute() or ".." in name.parts
            or len(name.parts) != 2 or name.parts[0] != "resume"
            or not re.fullmatch(r"[es][0-9]{4,}(?:-r[0-9]{2})?\.pt", name.name)
        ):
            raise ValueError("Invalid remote recovery path")
        target = directory / name.name
        self._download(self._remote(name.as_posix()), target)
        if target.stat().st_size != entry["bytes"] or (
            sha256(target) != entry["sha256"]
        ):
            raise ValueError("Remote recovery checksum mismatch")
        return target

    def publish_snapshot(self, directory: Path, *, force: bool = False) -> bool:
        now = time.monotonic()
        if (
            not force and self.last_sync is not None
            and now - self.last_sync < self.sync_seconds
        ):
            return False

        local_pointer = json.loads((directory / "latest.json").read_text())
        if local_pointer["run_id"] != self.run_id:
            raise ValueError("Checkpoint run ID differs from Hangar identity")
        expected = "pretrain" if self.stage == "pretrain" else (
            "baseline" if self.run_id == "001" else "finetune"
        )
        if local_pointer["stage"] != expected:
            raise ValueError("Checkpoint stage differs from Hangar stage")
        snapshot = directory / local_pointer["path"]
        if (
            snapshot.stat().st_size != local_pointer["size_bytes"]
            or sha256(snapshot) != local_pointer["sha256"]
        ):
            raise ValueError("Local checkpoint checksum mismatch")

        entry = self._checkpoint_entry(local_pointer)
        current_pointer = self._read_json(self._remote("latest.json"))
        stem = Path(entry["path"]).stem
        for attempt in range(100):
            try:
                self._immutable(snapshot, self._remote(entry["path"]))
                break
            except FileExistsError:
                # fallback replay must not overwrite an existing snapshot
                entry["path"] = f"resume/{stem}-r{attempt + 1:02d}.pt"
        else:
            raise FileExistsError("Too many checkpoint replay generations")

        self.uploads += 1
        if self.uploads == 1 or self.uploads % 10 == 0:
            with TemporaryDirectory() as temporary:
                checked = Path(temporary) / "verified.pt"
                self._download(self._remote(entry["path"]), checked)
                if sha256(checked) != entry["sha256"]:
                    raise OSError("Remote checkpoint integrity mismatch")

        previous = None
        if current_pointer is not None:
            old = current_pointer["current"]
            old_position = old.get("epoch", old.get("step"))
            new_position = entry.get("epoch", entry.get("step"))
            if old_position >= new_position:
                # a newer checkpoint may be damaged or stale
                previous = current_pointer.get("previous")
            elif old["path"] != entry["path"]:
                previous = old
            else:
                previous = current_pointer.get("previous")

        pointer = {"current": entry, "previous": previous}
        self._publish_json(pointer, self._remote("latest.json"))

        metrics = directory / "metrics.jsonl"
        if metrics.is_file():
            self._upload(metrics, self._remote("metrics.jsonl"))
        self.last_sync = now
        self._prune_remote(pointer)
        self._prune_local(directory)
        return True

    def _prune_remote(self, pointer: dict[str, Any]) -> None:
        from huggingface_hub import batch_bucket_files

        keep = {
            self._remote(entry["path"])
            for entry in (pointer["current"], pointer["previous"])
            if entry is not None
        }
        prefix = self._remote("resume/")
        old = [
            item.path for item in self._listing(prefix)
            if item.path not in keep and re.fullmatch(
                re.escape(prefix) + r"[es][0-9]{4,}(?:-r[0-9]{2})?\.pt",
                item.path,
            )
        ]
        if old:
            batch_bucket_files(self.bucket, delete=old)

    @staticmethod
    def _prune_local(directory: Path) -> None:
        snapshots = sorted((directory / "checkpoints").glob("*.pt"))
        for path in snapshots[:-2]:
            path.unlink()

    def publish_final(self, directory: Path) -> None:
        self.publish_snapshot(directory, force=True)
        self.publish_selected(directory)
        for name in (
            "metrics.jsonl", "loss.csv", "validation.csv", "predictions.csv"
        ):
            path = directory / name
            if path.is_file():
                self._upload(path, self._remote(name))

    def recover(self, directory: Path) -> Path:
        if (directory / "latest.json").exists():
            raise FileExistsError(f"Recovery directory not empty: {directory}")
        pointer = self._read_json(self._remote("latest.json"))
        if pointer is None:
            raise FileNotFoundError("Hangar stage has no latest.json")
        failure: Exception | None = None
        for entry in (pointer["current"], pointer.get("previous")):
            if entry is None:
                continue
            try:
                snapshot = self._verify_entry(entry, directory)
                break
            except (OSError, ValueError, KeyError) as exc:
                failure = exc
        else:
            raise OSError("No verified remote recovery checkpoint") from failure

        stage = (
            "pretrain" if self.stage == "pretrain"
            else "baseline" if self.run_id == "001" else "finetune"
        )
        local = {
            "format": 1,
            "stage": stage,
            "run_id": self.run_id,
            "position": entry.get("epoch", entry.get("step")),
            "path": f"checkpoints/{snapshot.name}",
            "sha256": entry["sha256"], "size_bytes": entry["bytes"],
        }
        target = directory / "checkpoints" / snapshot.name
        target.parent.mkdir(parents=True, exist_ok=True)
        snapshot.replace(target)
        result = directory / "latest.json"
        result.write_text(json.dumps(local, indent=2) + "\n")
        return result
