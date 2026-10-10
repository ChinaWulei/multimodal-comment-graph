from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from common import load_config, read_jsonl, resolve_path, write_jsonl


def parse_args() -> argparse.Namespace:
    default = Path(__file__).resolve().parents[1] / "config" / "generator_structure.yaml"
    parser = argparse.ArgumentParser(
        description="Build label-free, parent-supervised SFT records from real Fakeddit reply trees."
    )
    parser.add_argument("--config", type=Path, default=default)
    parser.add_argument("--limit", type=int, default=0, help="Optional smoke-test record limit.")
    parser.add_argument(
        "--split", choices=["train", "validate", "test"], default="train",
        help="Official source split. Defaults to train to prevent supervised leakage.",
    )
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument("--stats-json", type=Path, default=None)
    return parser.parse_args()


def flatten_parent_first(root: dict[str, Any]) -> list[dict[str, Any]]:
    """Return a deterministic parent-before-child traversal of one reply tree."""
    ordered: list[dict[str, Any]] = []
    seen: set[str] = set()
    stack = [root]
    while stack:
        node = stack.pop()
        node_id = str(node["comment_id"])
        if node_id in seen:
            continue
        seen.add(node_id)
        ordered.append(node)
        children = list(node.get("children", []))
        # Trees were already deterministically sorted upstream. Reverse for stack traversal.
        stack.extend(reversed(children))
    return ordered


def main() -> None:
    args = parse_args()
    config, config_path = load_config(args.config)
    data = config["data"]
    protocol = config["structure_aware"]
    data_dir = resolve_path(config_path, data["output_dir"])
    trees_path = data_dir / data["raw_comment_trees_jsonl"]
    relevance_path = data_dir / data["thread_distillation_results_jsonl"]
    if not trees_path.is_file() or not relevance_path.is_file():
        raise RuntimeError(
            "Raw trees or relevance results are missing. Run build_comment_trees.py and "
            "distill_comment_trees.py before structure-aware dataset construction."
        )

    min_score = float(protocol["min_relevance_score"])
    max_context = int(protocol["max_context_nodes"])
    max_comment_characters = int(protocol["max_comment_characters"])
    if max_context < 1:
        raise ValueError("structure_aware.max_context_nodes must be at least 1")
    if max_comment_characters < 32:
        raise ValueError("structure_aware.max_comment_characters must be at least 32")

    relevant_roots: set[tuple[str, str]] = set()
    relevance_counts = Counter()
    for record in read_jsonl(relevance_path):
        if record.get("status") != "ok":
            relevance_counts["invalid_teacher_record"] += 1
            continue
        if not bool(record.get("relevant")):
            relevance_counts["teacher_irrelevant"] += 1
            continue
        if float(record.get("relevance_score", 0.0)) < min_score:
            relevance_counts["below_relevance_threshold"] += 1
            continue
        relevant_roots.add((str(record["news_id"]), str(record["thread_root_id"])))
    if not relevant_roots:
        raise RuntimeError("No relevant reply trees are available at the configured threshold")

    records: list[dict[str, Any]] = []
    stats = Counter()
    tree_sizes = Counter()
    for news in read_jsonl(trees_path):
        if str(news["split"]) != args.split:
            continue
        news_id = str(news["news_id"])
        for root in news.get("roots", []):
            root_id = str(root["comment_id"])
            if (news_id, root_id) not in relevant_roots:
                continue
            ordered = flatten_parent_first(root)
            if len(ordered) < 2:
                stats["single_node_relevant_thread"] += 1
                continue
            tree_sizes[len(ordered)] += 1
            index_by_comment_id = {
                str(node["comment_id"]): index + 1 for index, node in enumerate(ordered)
            }
            # A prefix is used deliberately: every visible non-root parent is visible too.
            # This preserves the real reply edges without inventing a time ordering.
            usable_targets = min(len(ordered), max_context + 1)
            for target_position in range(usable_targets):
                target = ordered[target_position]
                observed_raw = ordered[:target_position]
                # Keep original comment text intact. Samples that cannot fit the VLM
                # context budget are excluded rather than silently truncated.
                if len(str(target["body"])) > max_comment_characters:
                    stats["target_too_long_for_context"] += 1
                    continue
                if any(len(str(node["body"])) > max_comment_characters for node in observed_raw):
                    stats["visible_context_contains_long_comment"] += 1
                    continue
                observed_nodes = []
                for node in observed_raw:
                    node_id = str(node["comment_id"])
                    raw_parent = str(node.get("parent_id", ""))
                    observed_nodes.append(
                        {
                            "node_index": index_by_comment_id[node_id],
                            "parent_node_index": index_by_comment_id.get(raw_parent, 0),
                            "content": str(node["body"]),
                        }
                    )
                raw_parent = str(target.get("parent_id", ""))
                target_parent = index_by_comment_id.get(raw_parent, 0)
                if target_parent and target_parent > target_position:
                    raise RuntimeError(
                        f"Parent occurs after its child in {news_id}:{target['comment_id']}"
                    )
                target_object = {
                    "parent_node_index": target_parent,
                    "content": str(target["body"]),
                }
                records.append(
                    {
                        "sample_id": f"{news_id}:{root_id}:{target['comment_id']}",
                        "metadata": {
                            "news_id": news_id,
                            "official_split": str(news["split"]),
                            "thread_root_id": root_id,
                            "target_comment_id": str(target["comment_id"]),
                            "target_raw_parent_id": raw_parent,
                            "target_is_root_reply": target_parent == 0,
                            "reply_supervision": "ground_truth_parent_id",
                            "node_order": "deterministic_parent_first_not_timestamp",
                            "tree_node_count": len(ordered),
                            "visible_context_node_count": len(observed_nodes),
                            # Bookkeeping only: raw IDs never enter model_input or
                            # target_json, but later let a generated edge be grafted
                            # back onto the exact observed comment node.
                            "observed_node_comment_ids": {
                                str(node["node_index"]): str(node_raw["comment_id"])
                                for node, node_raw in zip(observed_nodes, observed_raw)
                            },
                        },
                        "model_input": {
                            "image_path": str(news["image_path"]),
                            "news_text": str(news["news_text"]),
                            "observed_nodes": observed_nodes,
                        },
                        "target": target_object,
                        "target_json": json.dumps(target_object, ensure_ascii=False, separators=(",", ":")),
                    }
                )
                stats["root_reply_targets" if target_parent == 0 else "reply_targets"] += 1
    records.sort(key=lambda item: item["sample_id"])
    if args.limit > 0:
        records = records[: args.limit]
        stats["limited_for_smoke"] = len(records)
    if not records:
        raise RuntimeError("No structure-aware SFT records were constructed")

    split_output_key = {
        "train": "structure_train_jsonl",
        "validate": "structure_validate_jsonl",
        "test": "structure_test_jsonl",
    }[args.split]
    # The default must follow --split.  Otherwise sequential train/validation/
    # test construction silently overwrites the train file with a later split.
    output_path = args.output_jsonl.resolve() if args.output_jsonl else data_dir / data[split_output_key]
    if args.stats_json:
        stats_path = args.stats_json.resolve()
    elif args.split == "train":
        stats_path = data_dir / data["structure_stats_json"]
    else:
        # Keep validation/test audit evidence separate instead of overwriting the
        # train summary when this utility is invoked for each official split.
        stats_path = data_dir / f"structure_stats_{args.split}.json"
    write_jsonl(output_path, records)
    summary = {
        "output": str(output_path),
        "records": len(records),
        "source_split": args.split,
        "unique_news": len({item["metadata"]["news_id"] for item in records}),
        "unique_threads": len(
            {(item["metadata"]["news_id"], item["metadata"]["thread_root_id"]) for item in records}
        ),
        "official_split_counts": dict(
            Counter(item["metadata"]["official_split"] for item in records)
        ),
        "target_counts": dict(stats),
        "eligible_relevant_roots": len(relevant_roots),
        "relevance_rejections": dict(relevance_counts),
        "tree_size_distribution": dict(sorted(tree_sizes.items())),
        "max_context_nodes": max_context,
        "max_comment_characters": max_comment_characters,
        "supervision": "ground_truth_parent_id from Fakeddit comment trees",
        "time_claim": "none; parent-first traversal is a deterministic structural order, not timestamp order",
        "label_policy": "No truthfulness label is present in model_input or target_json.",
    }
    stats_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
