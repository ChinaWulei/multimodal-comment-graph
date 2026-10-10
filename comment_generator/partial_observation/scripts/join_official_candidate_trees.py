from __future__ import annotations

"""Admit cached comment trees only after their official image decodes correctly."""

import argparse
import json
from pathlib import Path

import pandas as pd

from common import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Join official candidate reply trees to decoded-image manifest rows.")
    parser.add_argument("--manifest-csv", type=Path, default=root / "data" / "official_subset" / "manifest.csv")
    parser.add_argument("--candidate-trees-jsonl", type=Path, default=root / "data" / "official_structure_candidate_comment_trees.jsonl")
    parser.add_argument("--output-jsonl", type=Path, default=root / "data" / "official_subset" / "raw_comment_trees.jsonl")
    parser.add_argument("--stats-json", type=Path, default=root / "data" / "official_subset" / "comment_tree_join_stats.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest_csv.resolve(), dtype=str, keep_default_na=False)
    eligible = manifest[
        manifest["image_valid"].str.lower().eq("true")
        & manifest["news_text"].str.strip().ne("")
        & manifest["split"].isin(["train", "validate", "test"])
    ].copy()
    if eligible["news_id"].duplicated().any():
        raise ValueError("Eligible manifest contains duplicate news IDs")
    manifest_by_id = {str(row["news_id"]): row for row in eligible.to_dict("records")}
    candidates = read_jsonl(args.candidate_trees_jsonl.resolve())
    candidate_by_id = {str(row["news_id"]): row for row in candidates}
    if len(candidate_by_id) != len(candidates):
        raise ValueError("Candidate tree cache contains duplicate news IDs")

    joined, missing_tree, split_mismatch = [], [], []
    for news_id, row in sorted(manifest_by_id.items()):
        tree = candidate_by_id.get(news_id)
        if tree is None:
            missing_tree.append(news_id)
            continue
        if str(tree["split"]) != str(row["split"]):
            split_mismatch.append(news_id)
            continue
        joined.append({
            "news_id": news_id, "split": str(row["split"]),
            "image_path": str(row["image_path"]), "news_text": str(row["news_text"]),
            "roots": tree["roots"], "orphans": tree["orphans"], "tree_stats": tree["tree_stats"],
        })
    if split_mismatch:
        raise ValueError(f"Split mismatch between image manifest and trees: {split_mismatch[:5]}")
    args.output_jsonl.resolve().parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_jsonl.resolve(), joined)
    stats = {
        "decoded_image_manifest_rows": len(eligible), "joined_news": len(joined),
        "joined_by_split": {split: sum(row["split"] == split for row in joined) for split in ("train", "validate", "test")},
        "missing_candidate_tree_count": len(missing_tree), "missing_candidate_tree_examples": missing_tree[:20],
        "image_policy": "every output row comes from image_valid=true in the decoded-image manifest",
        "output": str(args.output_jsonl.resolve()),
    }
    args.stats_json.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.stats_json.resolve().write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if not joined:
        raise RuntimeError("No decoded-image rows available; download and validate images before joining trees")


if __name__ == "__main__":
    main()
