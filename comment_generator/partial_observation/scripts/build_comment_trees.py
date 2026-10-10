from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from common import (
    INVALID_COMMENT_BODIES,
    is_obvious_automated_comment,
    iter_comment_records,
    load_config,
    resolve_path,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    default = Path(__file__).resolve().parents[1] / "config" / "generator_tree.yaml"
    parser = argparse.ArgumentParser(description="Build reply trees for image-backed Fakeddit news.")
    parser.add_argument("--config", type=Path, default=default)
    parser.add_argument("--progress-every", type=int, default=500_000)
    parser.add_argument("--manifest-path", type=Path, default=None, help="Optional explicit manifest CSV.")
    parser.add_argument("--output-jsonl", type=Path, default=None, help="Optional explicit reply-tree JSONL path.")
    parser.add_argument("--stats-path", type=Path, default=None, help="Optional explicit statistics JSON path.")
    return parser.parse_args()


def public_node(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "comment_id": record["comment_id"],
        "body": record["body"],
        "author": record["author"],
        "ups": record["ups"],
        "parent_id": record["parent_id"],
        "submission_id": record["submission_id"],
        "is_top_level": record["is_top_level"],
        "children": [],
    }


def count_tree(root: dict[str, Any]) -> tuple[int, int]:
    count = 0
    max_depth = 0
    seen: set[str] = set()
    stack = [(root, 1)]
    while stack:
        node, depth = stack.pop()
        comment_id = str(node["comment_id"])
        if comment_id in seen:
            continue
        seen.add(comment_id)
        count += 1
        max_depth = max(max_depth, depth)
        stack.extend((child, depth + 1) for child in node["children"])
    return count, max_depth


def main() -> None:
    args = parse_args()
    config, config_path = load_config(args.config)
    data = config["data"]
    output_dir = resolve_path(config_path, data["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest_path.resolve() if args.manifest_path else output_dir / data["manifest_csv"]
    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    candidates = manifest[
        manifest["image_valid"].str.lower().eq("true")
        & manifest["news_text"].str.strip().ne("")
        & manifest["split"].str.strip().ne("")
    ].copy()
    news_rows = {row["news_id"]: row for row in candidates.to_dict("records")}
    target_ids = set(news_rows)
    comments: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    duplicate_ids = Counter()
    invalid = automated = malformed = parsed = retained = 0
    comments_path = resolve_path(config_path, config["source"]["comments_tsv"])

    for record, _ in iter_comment_records(comments_path):
        parsed += 1
        if record is None:
            malformed += 1
            continue
        news_id = record["submission_id"]
        if news_id not in target_ids:
            if args.progress_every and parsed % args.progress_every == 0:
                print(f"records={parsed:,} retained={retained:,}")
            continue
        body = record["body"]
        if not body or body.casefold() in INVALID_COMMENT_BODIES:
            invalid += 1
            continue
        if bool(config["partial_observation"].get("exclude_obvious_automated_comments", True)) and is_obvious_automated_comment(
            record["author"], body
        ):
            automated += 1
            continue
        comment_id = record["comment_id"]
        if not comment_id:
            invalid += 1
            continue
        if comment_id in comments[news_id]:
            duplicate_ids[news_id] += 1
            continue
        comments[news_id][comment_id] = record
        retained += 1
        if args.progress_every and parsed % args.progress_every == 0:
            print(f"records={parsed:,} retained={retained:,}")

    results = []
    totals = Counter()
    for news_id in sorted(news_rows):
        row = news_rows[news_id]
        records = comments.get(news_id, {})
        nodes = {comment_id: public_node(record) for comment_id, record in records.items()}
        root_ids: list[str] = []
        orphan_ids: list[str] = []
        for comment_id, record in records.items():
            parent_id = record["parent_id"]
            if record["is_top_level"] or parent_id == news_id:
                root_ids.append(comment_id)
            elif parent_id in nodes and parent_id != comment_id:
                nodes[parent_id]["children"].append(nodes[comment_id])
            else:
                orphan_ids.append(comment_id)

        for node in nodes.values():
            node["children"].sort(
                key=lambda child: (-(child["ups"] or 0), child["comment_id"])
            )
        roots = [nodes[comment_id] for comment_id in root_ids]
        orphans = [nodes[comment_id] for comment_id in orphan_ids]
        roots.sort(key=lambda node: (-(node["ups"] or 0), node["comment_id"]))
        orphans.sort(key=lambda node: node["comment_id"])
        thread_sizes = []
        thread_depths = []
        for root in roots:
            size, depth = count_tree(root)
            thread_sizes.append(size)
            thread_depths.append(depth)
        orphan_sizes = []
        orphan_depths = []
        for root in orphans:
            size, depth = count_tree(root)
            orphan_sizes.append(size)
            orphan_depths.append(depth)
        totals["news"] += 1
        totals["comments"] += len(nodes)
        totals["roots"] += len(roots)
        totals["orphan_roots"] += len(orphans)
        totals["orphan_comments"] += sum(orphan_sizes)
        totals["replies"] += sum(max(0, size - 1) for size in thread_sizes)
        results.append(
            {
                "news_id": news_id,
                "split": row["split"],
                "image_path": row["image_path"],
                "news_text": row["news_text"],
                "roots": roots,
                "orphans": orphans,
                "tree_stats": {
                    "valid_comments": len(nodes),
                    "root_threads": len(roots),
                    "reply_comments_in_roots": sum(max(0, size - 1) for size in thread_sizes),
                    "orphan_root_threads": len(orphans),
                    "orphan_comments_total": sum(orphan_sizes),
                    "max_thread_size": max(thread_sizes, default=0),
                    "max_thread_depth": max(thread_depths, default=0),
                },
            }
        )

    output_path = args.output_jsonl.resolve() if args.output_jsonl else output_dir / data["raw_comment_trees_jsonl"]
    write_jsonl(output_path, results)
    stats = {
        "source_records_parsed": parsed,
        "malformed_records": malformed,
        "invalid_or_deleted_records_for_candidate_news": invalid,
        "automated_records_excluded": automated,
        "duplicate_comment_ids_excluded": int(sum(duplicate_ids.values())),
        "candidate_news": len(target_ids),
        "news_written": totals["news"],
        "valid_comments_retained": totals["comments"],
        "root_threads": totals["roots"],
        "reply_comments_attached": totals["replies"],
        "orphan_root_threads": totals["orphan_roots"],
        "orphan_comments_total": totals["orphan_comments"],
        "output": str(output_path),
    }
    stats_path = args.stats_path.resolve() if args.stats_path else output_dir / "comment_tree_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
