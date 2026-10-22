import hashlib
import json
from pathlib import Path
from typing import Any

import torch


def save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_snapshot(
    output: Path, stage: str, position: int, state: dict[str, Any],
) -> Path:
    unit = "step" if stage == "pretrain" else "epoch"
    name = f"{unit}_{position:06d}.pt"
    path = output / "checkpoints" / name
    if path.exists():
        raise FileExistsError(f"Checkpoint already exists: {path}")
    save_checkpoint(path, state)
    pointer = {
        "format": 1,
        "stage": stage,
        "run_id": state["run_id"],
        "position": position,
        "path": f"checkpoints/{name}",
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }
    temporary = output / "latest.json.tmp"
    temporary.write_text(json.dumps(pointer, indent=2) + "\n")
    temporary.replace(output / "latest.json")
    return path


def load_snapshot(path: Path) -> dict[str, Any]:
    if path.suffix == ".json":
        pointer = json.loads(path.read_text())
        checkpoint = path.parent / pointer["path"]
        if (
            checkpoint.stat().st_size != pointer["size_bytes"]
            or sha256(checkpoint) != pointer["sha256"]
        ):
            raise ValueError("Checkpoint checksum does not match latest.json")
        path = checkpoint
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("format") != 1:
        raise ValueError("Unsupported checkpoint format")
    return state


def cpu_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in module.state_dict().items()
    }


def write_events(output: Path, events: list[dict[str, Any]]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / "metrics.jsonl.tmp"
    with temporary.open("w") as stream:
        for event in events:
            stream.write(json.dumps(event, separators=(",", ":")) + "\n")
    temporary.replace(output / "metrics.jsonl")


def append_event(output: Path, event: dict[str, Any]) -> None:
    with (output / "metrics.jsonl").open("a") as stream:
        stream.write(json.dumps(event, separators=(",", ":")) + "\n")
        stream.flush()


def rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {"torch": torch.get_rng_state()}
    if torch.cuda.is_initialized():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state: dict[str, Any]) -> None:
    torch.set_rng_state(state["torch"])
    if "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])
