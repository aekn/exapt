from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import warnings
from pathlib import Path
from typing import Any

import torch


BUCKET = "exapt-lab/hangar"
_METADATA = ("config.json", "benchmark.json", "sessions.json")


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _entry(path: Path, remote: str) -> dict[str, str]:
    return {"path": remote, "sha256": _digest(path)}


def _log(history: list[dict[str, Any]]) -> bytes:
    return ("".join(json.dumps(row, sort_keys=True) + "\n" for row in history)).encode()


class BucketRun:
    def __init__(self, run_id: str):
        if re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", run_id) is None:
            raise ValueError("storage: invalid run id")
        self.prefix = f"runs/{run_id}"

    def _state(self, token: str) -> dict[str, Any] | None:
        from huggingface_hub import HfFileSystem, download_bucket_files

        remote = f"{self.prefix}/state.json"
        if not HfFileSystem(token=token).exists(f"buckets/{BUCKET}/{remote}"):
            return None
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            download_bucket_files(
                BUCKET, files=[(remote, path)], token=token,
                raise_on_missing_files=True,
            )
            state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("version") != 2 or state.get("prefix") != self.prefix:
            raise ValueError("storage: invalid remote run state")
        return state

    def publish(self, output: Path, *, token: str) -> dict[str, Any]:
        from huggingface_hub import batch_bucket_files

        output = Path(output)
        resume = torch.load(output / "resume.pt", map_location="cpu", weights_only=True)
        best_epoch = max(resume["history"], key=lambda item: (
            item["development_recall"], item["development_ndcg"],
        ))["epoch"]
        local_selected = output / "selected" / f"e{best_epoch:04d}.pt"
        selected = torch.load(local_selected, map_location="cpu", weights_only=True)
        epoch = resume["epoch"]
        fingerprint = resume["metadata"]["sha256"]
        if epoch < 1 or selected["epoch"] != best_epoch or selected["metadata"] != resume["metadata"]:
            raise ValueError("storage: inconsistent local checkpoints")

        previous_state = self._state(token)
        if previous_state is not None:
            if previous_state["fingerprint"] != fingerprint:
                raise ValueError("storage: run id belongs to another experiment")
            old = previous_state["current"]
            if old["epoch"] > epoch:
                raise ValueError("storage: remote run is ahead of local checkpoint")
            if old["epoch"] == epoch:
                if _digest(output / "resume.pt") != old["resume"]["sha256"]:
                    raise ValueError("storage: remote checkpoint differs from local")
                return previous_state

        current = {
            "epoch": epoch,
            "resume": _entry(output / "resume.pt", f"{self.prefix}/resume/e{epoch:04d}.pt"),
            "selected": {**_entry(
                local_selected, f"{self.prefix}/selected/e{selected['epoch']:04d}.pt",
            ), "epoch": selected["epoch"]},
        }
        artifacts = {
            name: _entry(output / name, f"{self.prefix}/{name}")
            for name in _METADATA
        }
        artifacts["development.json"] = _entry(
            output / "development.json", f"{self.prefix}/development.json",
        )
        artifacts["epochs.jsonl"] = _entry(
            output / "epochs.jsonl", f"{self.prefix}/epochs.jsonl",
        )
        add: list[tuple[str | Path | bytes, str]] = [
            (output / name, item["path"]) for name, item in artifacts.items()
        ]
        add.append((output / "resume.pt", current["resume"]["path"]))
        selected_path = current["selected"]["path"]
        if previous_state is None or selected_path != previous_state["current"]["selected"]["path"]:
            add.append((local_selected, selected_path))

        batch_bucket_files(BUCKET, add=add, token=token)
        state = {
            "version": 2,
            "prefix": self.prefix,
            "fingerprint": fingerprint,
            "artifacts": artifacts,
            "current": current,
            "previous": previous_state["current"] if previous_state is not None else None,
        }
        batch_bucket_files(
            BUCKET,
            add=[(json.dumps(state, sort_keys=True).encode(), f"{self.prefix}/state.json")],
            token=token,
        )

        if previous_state is not None:
            keep = {current["resume"]["path"], current["selected"]["path"]}
            if state["previous"] is not None:
                keep.update(item["path"] for item in (
                    state["previous"]["resume"], state["previous"]["selected"],
                ))
            stale = set()
            for generation in (previous_state["current"], previous_state["previous"]):
                if generation is not None:
                    stale.update(item["path"] for item in (
                        generation["resume"], generation["selected"],
                    ) if item["path"] not in keep)
            if stale:
                try:
                    batch_bucket_files(BUCKET, delete=sorted(stale), token=token)
                except Exception as error:
                    warnings.warn(f"storage: stale checkpoint cleanup failed: {error}", stacklevel=2)
        return state

    def restore(self, output: Path, *, token: str) -> int | None:
        from huggingface_hub import batch_bucket_files, download_bucket_files

        output = Path(output)
        if output.exists():
            raise ValueError("storage: restore destination already exists")
        state = self._state(token)
        if state is None:
            return None

        output.parent.mkdir(parents=True, exist_ok=True)
        for generation in (state["current"], state["previous"]):
            if generation is None:
                continue
            try:
                with tempfile.TemporaryDirectory(dir=output.parent) as directory:
                    root = Path(directory) / "stage"
                    root.mkdir()
                    (root / "selected").mkdir()
                    entries = {
                        "resume.pt": generation["resume"],
                        f"selected/e{generation['selected']['epoch']:04d}.pt": generation["selected"],
                        **{name: state["artifacts"][name] for name in _METADATA[:2]},
                    }
                    download_bucket_files(
                        BUCKET,
                        files=[(item["path"], root / name) for name, item in entries.items()],
                        token=token, raise_on_missing_files=True,
                    )
                    if any(_digest(root / name) != item["sha256"] for name, item in entries.items()):
                        continue
                    resume = torch.load(root / "resume.pt", map_location="cpu", weights_only=True)
                    selected = torch.load(
                        root / f"selected/e{generation['selected']['epoch']:04d}.pt",
                        map_location="cpu", weights_only=True,
                    )
                    if (
                        resume["epoch"] != generation["epoch"]
                        or resume["metadata"]["sha256"] != state["fingerprint"]
                        or selected["metadata"] != resume["metadata"]
                        or selected["epoch"] > resume["epoch"]
                    ):
                        continue
                    if len(resume["history"]) != generation["epoch"]:
                        continue
                    if (selected["development"]["development_recall"],
                        selected["development"]["development_ndcg"]) != tuple(resume["best"]):
                        continue
                    (root / "epochs.jsonl").write_bytes(_log(resume["history"]))
                    (root / "development.json").write_text(
                        json.dumps(selected["rankings"], indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    if "sessions.json" in state["artifacts"]:
                        try:
                            download_bucket_files(
                                BUCKET,
                                files=[(state["artifacts"]["sessions.json"]["path"], root / "sessions.json")],
                                token=token, raise_on_missing_files=True,
                            )
                        except (OSError, RuntimeError, ValueError):
                            pass
                    os.replace(root, output)
            except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                continue

            if generation is state["previous"]:
                repaired = {**state, "current": generation, "previous": None}
                batch_bucket_files(
                    BUCKET,
                    add=[(json.dumps(repaired, sort_keys=True).encode(), f"{self.prefix}/state.json")],
                    token=token,
                )
                stale = [state["current"]["resume"]["path"]]
                if state["current"]["selected"]["path"] != generation["selected"]["path"]:
                    stale.append(state["current"]["selected"]["path"])
                try:
                    batch_bucket_files(BUCKET, delete=stale, token=token)
                except Exception as error:
                    warnings.warn(f"storage: stale checkpoint cleanup failed: {error}", stacklevel=2)
            return generation["epoch"]
        raise RuntimeError("storage: no intact remote resume checkpoint")
