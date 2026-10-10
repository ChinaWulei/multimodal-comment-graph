"""Build label-blind inference contexts for news lacking generated graphs.

This is an exploratory hard-context cohort: unlike structure SFT, it does not
require a teacher-approved relevant thread. No veracity label or target comment is
read into the model input, and no result from this cohort should tune a model.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from PIL import Image

from build_structured_sft import flatten_parent_first
from common import read_jsonl, write_jsonl
from structure_utils import validate_structure_input


def args_from_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-csv", type=Path, required=True)
    parser.add_argument("--real-trees-jsonl", type=Path, required=True)
    parser.add_argument("--existing-generated-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--stats-json", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "validate", "test"], default="test")
    parser.add_argument("--max-context-nodes", type=int, default=12)
    parser.add_argument("--max-comment-characters", type=int, default=400)
    return parser.parse_args()


def candidate_context(root: dict, max_nodes: int, max_chars: int) -> list[dict]:
    """Use a parent-first prefix so every included parent is visible."""
    nodes = []
    for node in flatten_parent_first(root)[:max_nodes]:
        body = str(node.get("body", "")).strip()
        if not body or len(body) > max_chars:
            break
        nodes.append(node)
    return nodes


def main() -> None:
    args = args_from_cli()
    if args.max_context_nodes < 1 or args.max_comment_characters < 32:
        raise ValueError("Invalid context limits")
    with args.manifest_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    if len(manifest) != len({str(row["news_id"]) for row in manifest}):
        raise ValueError("Duplicate news IDs in manifest")
    existing_rows = read_jsonl(args.existing_generated_jsonl)
    if any(row.get("official_split") != args.split for row in existing_rows):
        raise ValueError("Existing generated graphs contain another official split")
    existing = {str(row["news_id"]) for row in existing_rows}
    if len(existing) != len(existing_rows):
        raise ValueError("Duplicate IDs in existing generated graphs")
    tree_rows = read_jsonl(args.real_trees_jsonl)
    trees = {str(row["news_id"]): row for row in tree_rows}
    if len(trees) != len(tree_rows):
        raise ValueError("Duplicate IDs in real trees")

    records, counts = [], Counter()
    for row in manifest:
        if row["split"] != args.split:
            continue
        counts[f"{args.split}_manifest"] += 1
        if str(row["image_valid"]).lower() != "true":
            counts["image_invalid"] += 1
            continue
        news_id = str(row["news_id"])
        if news_id in existing:
            counts["already_generated"] += 1
            continue
        news_text = str(row["news_text"]).strip()
        image_path = Path(row["image_path"])
        if not news_text or not image_path.is_file():
            counts["missing_news_or_image"] += 1
            continue
        with Image.open(image_path) as image:
            image.verify()
        tree = trees.get(news_id)
        if not tree or tree.get("split") != args.split:
            counts["missing_split_tree"] += 1
            continue
        candidates = []
        for root in tree.get("roots", []):
            nodes = candidate_context(root, args.max_context_nodes, args.max_comment_characters)
            if nodes:
                candidates.append((len(nodes), str(root["comment_id"]), nodes))
        if not candidates:
            counts["no_eligible_context"] += 1
            continue
        # Same cardinality priority as the existing --one-per-news inference
        # option. The tie break is deterministic and does not involve labels.
        _, root_id, selected = max(candidates, key=lambda item: (item[0], item[1]))
        local = {str(node["comment_id"]): index for index, node in enumerate(selected, start=1)}
        observed = []
        for index, node in enumerate(selected, start=1):
            parent_id = str(node.get("parent_id", ""))
            parent = local.get(parent_id, 0)
            if parent >= index:
                raise ValueError(f"Non-parent-first context: {news_id}/{root_id}")
            observed.append({"node_index": index, "parent_node_index": parent,
                             "content": str(node["body"]).strip()})
        model_input = {"image_path": str(image_path), "news_text": news_text,
                       "observed_nodes": observed}
        validate_structure_input(model_input)
        records.append({
            "sample_id": f"{news_id}:{root_id}:label_blind_inference",
            "metadata": {
                "news_id": news_id, "official_split": args.split, "thread_root_id": root_id,
                "selection_policy": "largest_eligible_parent_first_prefix_no_teacher_threshold",
                "observed_node_comment_ids": {
                    str(index): str(node["comment_id"])
                    for index, node in enumerate(selected, start=1)
                },
            },
            "model_input": model_input,
        })
        counts["contexts_built"] += 1
    records.sort(key=lambda record: record["metadata"]["news_id"])
    if not records:
        raise RuntimeError(f"No unseen {args.split} contexts were built")
    write_jsonl(args.output_jsonl, records)
    stats = {
        "source_split": args.split, "news_count": len(records),
        "counts": dict(counts), "news_ids": [record["metadata"]["news_id"] for record in records],
        "observed_node_count_distribution": dict(sorted(Counter(
            len(record["model_input"]["observed_nodes"]) for record in records).items())),
        "label_policy": "No label field read or included; inference only.",
        "cohort_caveat": "Previously excluded by relevance-filtered generation; exploratory hard-context cohort, not a random sample.",
    }
    args.stats_json.parent.mkdir(parents=True, exist_ok=True)
    args.stats_json.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
