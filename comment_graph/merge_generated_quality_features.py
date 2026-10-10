"""Join label-blind image/news/parent scores and candidate-rank features.

This produces features for a future human-supervised quality gate. It does not
convert reranker disagreement into a quality label or change graph edges.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path


CANDIDATE_COLUMNS = (
    "reranker_chosen_vs_best_alternative_margin",
    "reranker_chosen_parent_rank_fraction",
)


def read_unique(path: Path, split: str) -> dict[str, dict]:
    rows = {}
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = str(row["annotation_id"])
            if key in rows or row["official_split"] != split:
                raise ValueError(f"Duplicate ID or split mismatch: {path}:{number}/{key}")
            rows[key] = row
    if not rows:
        raise ValueError(f"Empty feature file: {path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-jsonl", type=Path, required=True)
    parser.add_argument("--base-metadata-json", type=Path, required=True)
    parser.add_argument("--candidate-jsonl", type=Path, required=True)
    parser.add_argument("--candidate-metadata-json", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validate", "test"), required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-metadata-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_jsonl.exists() or args.output_metadata_json.exists():
        raise FileExistsError("Refusing to overwrite merged feature outputs")
    base = read_unique(args.base_jsonl, args.split)
    candidate = read_unique(args.candidate_jsonl, args.split)
    base_meta = json.loads(args.base_metadata_json.read_text(encoding="utf-8"))
    candidate_meta = json.loads(args.candidate_metadata_json.read_text(encoding="utf-8"))
    if (set(base) != set(candidate) or base_meta["rows"] != len(base) or
            candidate_meta["rows"] != len(candidate) or
            base_meta["split"] != args.split or candidate_meta["split"] != args.split):
        raise ValueError("Base/candidate feature coverage or split mismatch")
    if (base_meta["reranker_model"] != candidate_meta["reranker_model"] or
            base_meta["reranker_snapshot"] != candidate_meta["reranker_snapshot"]):
        raise ValueError("Frozen text reranker differs across feature sources")
    rows = []
    for key, row in base.items():
        extra = candidate[key]
        if row["news_id"] != extra["news_id"]:
            raise ValueError(f"Feature/news mismatch: {key}")
        if any(column in row for column in CANDIDATE_COLUMNS):
            raise ValueError(f"Already contains candidate feature: {key}")
        rows.append({**row, **{column: extra[column] for column in CANDIDATE_COLUMNS}})
    metadata = {
        "rows": len(rows), "split": args.split,
        "queue_sha256": base_meta["queue_sha256"],
        "reranker_model": base_meta["reranker_model"],
        "reranker_snapshot": base_meta["reranker_snapshot"],
        "clip_model": base_meta["clip_model"],
        "clip_snapshot": base_meta["clip_snapshot"],
        "candidate_max_length": candidate_meta["max_length"],
        "candidate_source_graph_sha256": candidate_meta["source_graph_sha256"],
        "base_feature_sha256": hashlib.sha256(args.base_jsonl.read_bytes()).hexdigest(),
        "candidate_feature_sha256": hashlib.sha256(args.candidate_jsonl.read_bytes()).hexdigest(),
        "warning": "Model proxy scores only. Candidate rank is not a generated-parent correctness label.",
    }
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_metadata_json.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n",
                                     dir=args.output_jsonl.parent,
                                     prefix=args.output_jsonl.stem + ".", suffix=".tmp",
                                     delete=False) as handle:
        temporary = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    try:
        temporary.replace(args.output_jsonl)
        args.output_metadata_json.write_text(json.dumps(metadata, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
