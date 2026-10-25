from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    import polars as pl


REPO = "exapt-lab/data"
SOURCE = "sources/primekg-2.1"
COLUMNS = (
    "relation", "display_relation",
    "x_index", "x_id", "x_type", "x_name", "x_source",
    "y_index", "y_id", "y_type", "y_name", "y_source",
)
THERAPY = ("indication", "contraindication", "off-label use")


class _Node(NamedTuple):
    identifier: str
    kind: str
    source: str
    name: str


def _verify_source(root: Path) -> tuple[Path, int]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("primekg: manifest must be a JSON object")

    identity = (manifest.get("source"), manifest.get("version"), manifest.get("file_id"))
    if identity != ("doi:10.7910/DVN/IXA7BM", "2.1", 6180620):
        raise ValueError("primekg: unexpected source identity")
    if manifest.get("file") != "kg.csv":
        raise ValueError("primekg: unexpected source filename")

    size = manifest.get("bytes")
    rows = manifest.get("rows")
    digest = manifest.get("sha256")

    if type(size) is not int or size <= 0:
        raise ValueError("primekg: invalid manifest bytes")
    if type(rows) is not int or rows < 0:
        raise ValueError("primekg: invalid manifest rows")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("primekg: invalid manifest sha256")

    path = root / "kg.csv"
    actual_size = path.stat().st_size
    if actual_size != size:
        raise ValueError(f"primekg: kg.csv size mismatch ({actual_size} != {size})")

    with path.open("rb") as stream:
        actual_digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual_digest != digest:
        raise ValueError("primekg: kg.csv SHA-256 mismatch")

    with path.open("r", encoding="utf-8", newline="") as stream:
        if tuple(next(csv.reader(stream, strict=True), ())) != COLUMNS:
            raise ValueError("primekg: unexpected CSV header")

    return path, rows


def verify_primekg(root: Path) -> Path:
    return _verify_source(Path(root))[0]


def fetch_primekg(revision: str, token: str | None = None) -> Path:
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("primekg: revision must be a full commit SHA")

    from huggingface_hub import hf_hub_download

    paths = [
        Path(hf_hub_download(
            repo_id=REPO,
            repo_type="dataset",
            filename=f"{SOURCE}/{name}",
            revision=revision,
            token=token,
        ))
        for name in ("manifest.json", "kg.csv")
    ]
    if paths[0].parent != paths[1].parent:
        raise RuntimeError("primekg: source files have different cache directories")

    root = paths[0].parent
    verify_primekg(root)
    return root


def open_primekg(root: Path) -> pl.LazyFrame:
    path = verify_primekg(root)

    import polars as pl

    schema = {
        name: pl.Int64 if name in ("x_index", "y_index") else pl.String
        for name in COLUMNS
    }
    return pl.scan_csv(path, schema=schema)


def audit_primekg(root: Path) -> dict[str, Any]:
    path, expected_rows = _verify_source(Path(root))

    nodes: dict[int, _Node] = {}
    typed_indices: dict[tuple[str, str, str], int] = {}
    index_conflicts: dict[int, tuple[_Node, _Node]] = {}
    name_conflicts: dict[int, tuple[str, str]] = {}
    aliases: dict[tuple[str, str, str], tuple[int, int]] = {}

    relation_counts: Counter[str] = Counter()
    signature_counts: Counter[tuple[str, str, str]] = Counter()
    display_labels: defaultdict[str, set[str]] = defaultdict(set)
    therapy_pairs: dict[str, set[tuple[int, int]]] = { name: set() for name in THERAPY }
    therapy_arcs: dict[str, Counter[tuple[int, int]]] = { name: Counter() for name in THERAPY }


    def record_node(index: int, node: _Node) -> None:
        previous = nodes.get(index)
        if previous is not None:
            if (
                previous.identifier != node.identifier
                or previous.kind != node.kind
                or previous.source != node.source
            ):
                index_conflicts.setdefault(index, (previous, node))
            if previous.name != node.name:
                name_conflicts.setdefault(index, (previous.name, node.name))
            return

        nodes[index] = node
        key = (node.kind, node.source, node.identifier)
        first_index = typed_indices.setdefault(key, index)
        if first_index != index:
            aliases.setdefault(key, (first_index, index))

    rows = 0
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        if tuple(next(reader, ())) != COLUMNS:
            raise ValueError("primekg: unexpected CSV header")

        for rows, row in enumerate(reader, start=1):
            if len(row) != len(COLUMNS):
                raise ValueError(
                    f"primekg: line {reader.line_num}: expected "
                    f"{len(COLUMNS)} fields, got {len(row)}"
                )

            (
                relation, display,
                x_index, x_id, x_type, x_name, x_source,
                y_index, y_id, y_type, y_name, y_source,
            ) = row

            try:
                x, y = int(x_index), int(y_index)
            except ValueError as error:
                raise ValueError(f"primekg: line {reader.line_num}: invalid node index") from error
            if x < 0 or y < 0:
                raise ValueError(f"primekg: line {reader.line_num}: negative node index")
            if not (relation and x_id and x_type and y_id and y_type):
                raise ValueError(f"primekg: line {reader.line_num}: missing required field")

            record_node(x, _Node(x_id, x_type, x_source, x_name))
            record_node(y, _Node(y_id, y_type, y_source, y_name))

            relation_counts[relation] += 1
            signature_counts[(relation, x_type, y_type)] += 1
            display_labels[relation].add(display)

            if relation in therapy_pairs:
                if (x_type, y_type) == ("drug", "disease"):
                    pair = (x, y)
                elif (x_type, y_type) == ("disease", "drug"):
                    pair = (y, x)
                else:
                    raise ValueError(
                        f"primekg: line {reader.line_num}: invalid therapy endpoints "
                        f"({x_type}, {y_type})"
                    )

                therapy_pairs[relation].add(pair)
                therapy_arcs[relation][(x, y)] += 1

    if rows != expected_rows:
        raise ValueError(f"primekg: row count mismatch ({rows} != {expected_rows})")

    index_range = [min(nodes), max(nodes)] if nodes else []
    node_types = Counter(node.kind for node in nodes.values())
    alias_types = Counter(kind for kind, _, _ in aliases)
    indication = therapy_pairs["indication"]
    contraindication = therapy_pairs["contraindication"]
    off_label = therapy_pairs["off-label use"]

    therapy_summary = {
        relation: {
            "pairs": len(pairs),
            "diseases": len({disease for _, disease in pairs}),
        }
        for relation, pairs in therapy_pairs.items()
    }
    reverse_mismatches = {
        relation: sum(
            count != arcs.get((y, x), 0)
            for (x, y), count in arcs.items()
        )
        for relation, arcs in therapy_arcs.items()
    }
    alias_examples = [
        {
            "type": kind,
            "source": source,
            "id": identifier,
            "first_index": first,
            "other_index": other,
            "first_name": nodes[first].name,
            "other_name": nodes[other].name,
        }
        for (kind, source, identifier), (first, other)
        in sorted(aliases.items())[:10]
    ]

    return {
        "rows": rows,
        "nodes": len(nodes),
        "node_types": dict(sorted(node_types.items())),
        "index_range": index_range,
        "contiguous_indices": bool(
            index_range and index_range[1] - index_range[0] + 1 == len(nodes)
        ),
        "index_identity_conflicts": len(index_conflicts),
        "index_name_conflicts": len(name_conflicts),
        "typed_identifier_aliases": len(aliases),
        "typed_aliases_by_type": dict(sorted(alias_types.items())),
        "typed_aliases_with_different_names": sum(
            nodes[first].name != nodes[other].name
            for first, other in aliases.values()
        ),
        "index_identity_examples": [
            {"index": index, "first": first, "other": other}
            for index, (first, other) in sorted(index_conflicts.items())[:10]
        ],
        "index_name_examples": [
            {"index": index, "first": first, "other": other}
            for index, (first, other) in sorted(name_conflicts.items())[:10]
        ],
        "typed_alias_examples": alias_examples,
        "relations": dict(sorted(relation_counts.items())),
        "typed_signatures": [
            {
                "relation": relation,
                "source_type": x_type,
                "target_type": y_type,
                "rows": count,
            }
            for (relation, x_type, y_type), count in sorted(signature_counts.items())
        ],
        "display_labels": {
            relation: sorted(labels) for relation, labels in sorted(display_labels.items())
        },
        "therapy": therapy_summary,
        "therapeutic_reverse_mismatches": reverse_mismatches,
        "therapeutic_overlaps": {
            "indication_contraindication": len(indication & contraindication),
            "indication_off_label": len(indication & off_label),
            "contraindication_off_label": len(contraindication & off_label),
        },
    }
