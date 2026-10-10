from __future__ import annotations

"""Cache genuine Fakeddit reply trees for the preselected official subset.

The cache intentionally has no ``image_valid`` flag and must *not* be used as
training input on its own.  It separates the expensive one-pass comment scan
from image acquisition; a later manifest join admits only decoded images.
"""

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
    default = Path(__file__).resolve().parents[1] / "config" / "generator_structure.yaml"
    parser = argparse.ArgumentParser(description="Cache reply trees for the planned official Fakeddit subset.")
    parser.add_argument("--config", type=Path, default=default)
    parser.add_argument("--progress-every", type=int, default=1_000_000)
    return parser.parse_args()


def public_node(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "comment_id": record["comment_id"], "body": record["body"],
        "author": record["author"], "ups": record["ups"],
        "parent_id": record["parent_id"], "submission_id": record["submission_id"],
        "is_top_level": record["is_top_level"], "children": [],
    }


def summarize(root: dict[str, Any]) -> tuple[int, int]:
    seen, count, depth = set(), 0, 0
    stack = [(root, 1)]
    while stack:
        node, current_depth = stack.pop()
        if node["comment_id"] in seen:
            continue
        seen.add(node["comment_id"])
        count += 1
        depth = max(depth, current_depth)
        stack.extend((child, current_depth + 1) for child in node["children"])
    return count, depth


def main() -> None:
    args = parse_args()
    config, config_path = load_config(args.config)
    data_root = resolve_path(config_path, config["data"]["output_dir"])
    selected_rows: dict[str, dict[str, Any]] = {}
    for split in ("train", "validate", "test"):
        path = data_root / f"official_structure_subset_{split}.csv"
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        for row in frame.to_dict("records"):
            news_id = str(row["news_id"]).strip().lower()
            if news_id in selected_rows:
                raise RuntimeError(f"Selected news occurs more than once: {news_id}")
            selected_rows[news_id] = {**row, "split": split, "news_id": news_id}

    by_news: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    parsed = retained = deleted = automated = duplicate = malformed = 0
    comments_path = resolve_path(config_path, config["source"]["comments_tsv"])
    for record, _ in iter_comment_records(comments_path):
        parsed += 1
        if record is None:
            malformed += 1
            continue
        news_id = record["submission_id"]
        if news_id not in selected_rows:
            continue
        body = record["body"]
        if not body or body.casefold() in INVALID_COMMENT_BODIES:
            deleted += 1
            continue
        if is_obvious_automated_comment(record["author"], body):
            automated += 1
            continue
        comment_id = record["comment_id"]
        if not comment_id or comment_id in by_news[news_id]:
            duplicate += 1
            continue
        by_news[news_id][comment_id] = record
        retained += 1
        if args.progress_every and parsed % args.progress_every == 0:
            print(f"records={parsed:,} retained={retained:,}", flush=True)

    output, per_split = [], Counter()
    totals = Counter()
    for news_id, row in sorted(selected_rows.items()):
        records = by_news.get(news_id, {})
        nodes = {comment_id: public_node(record) for comment_id, record in records.items()}
        roots, orphans = [], []
        for comment_id, record in records.items():
            parent_id = record["parent_id"]
            if record["is_top_level"] or parent_id == news_id:
                roots.append(nodes[comment_id])
            elif parent_id in nodes and parent_id != comment_id:
                nodes[parent_id]["children"].append(nodes[comment_id])
            else:
                orphans.append(nodes[comment_id])
        for node in nodes.values():
            node["children"].sort(key=lambda child: (-(child["ups"] or 0), child["comment_id"]))
        roots.sort(key=lambda node: (-(node["ups"] or 0), node["comment_id"]))
        orphans.sort(key=lambda node: node["comment_id"])
        root_sizes = [summarize(root) for root in roots]
        output.append({
            "news_id": news_id, "split": row["split"], "news_text": row["news_text"],
            "image_url": row["image_url"], "image_status": "not_verified_local",
            "roots": roots, "orphans": orphans,
            "tree_stats": {
                "valid_comments": len(nodes), "root_threads": len(roots),
                "reply_comments_in_roots": sum(max(0, size - 1) for size, _ in root_sizes),
                "orphan_root_threads": len(orphans),
                "max_thread_size": max((size for size, _ in root_sizes), default=0),
                "max_thread_depth": max((depth for _, depth in root_sizes), default=0),
            },
        })
        per_split[row["split"]] += 1
        totals["valid_comments"] += len(nodes)
        totals["root_threads"] += len(roots)
        totals["reply_comments"] += sum(max(0, size - 1) for size, _ in root_sizes)

    output_path = data_root / "official_structure_candidate_comment_trees.jsonl"
    write_jsonl(output_path, output)
    stats = {
        "planned_news": len(selected_rows), "news_written": len(output),
        "planned_by_split": dict(per_split), "comments_scanned": parsed,
        "retained_comments": retained, "deleted_or_empty_excluded": deleted,
        "automated_excluded": automated, "duplicate_or_missing_id_excluded": duplicate,
        "malformed_records": malformed, **dict(totals),
        "image_policy": "candidate cache only; join to a decoded-image manifest before any model input",
        "output": str(output_path),
    }
    stats_path = data_root / "official_structure_candidate_tree_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
