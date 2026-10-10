from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common import load_config, read_jsonl, resolve_path


def parse_args() -> argparse.Namespace:
    default = Path(__file__).resolve().parents[1] / "config" / "generator_tree.yaml"
    parser = argparse.ArgumentParser(description="Validate saved comment reply trees.")
    parser.add_argument("--config", type=Path, default=default)
    parser.add_argument("--input-jsonl", type=Path, default=None, help="Optional explicit reply-tree JSONL.")
    parser.add_argument("--report-path", type=Path, default=None, help="Optional explicit validation report path.")
    return parser.parse_args()


def walk(root: dict[str, Any], news_id: str) -> tuple[set[str], int, list[str]]:
    seen: set[str] = set()
    active: set[str] = set()
    errors: list[str] = []
    max_depth = 0

    def visit(node: dict[str, Any], depth: int) -> None:
        nonlocal max_depth
        comment_id = str(node["comment_id"])
        if comment_id in active:
            errors.append(f"cycle:{news_id}:{comment_id}")
            return
        if comment_id in seen:
            errors.append(f"duplicate_node:{news_id}:{comment_id}")
            return
        if node["submission_id"] != news_id:
            errors.append(f"wrong_submission:{news_id}:{comment_id}")
        seen.add(comment_id)
        active.add(comment_id)
        max_depth = max(max_depth, depth)
        for child in node.get("children", []):
            if child["parent_id"] != comment_id:
                errors.append(f"wrong_parent:{news_id}:{child['comment_id']}")
            visit(child, depth + 1)
        active.remove(comment_id)

    visit(root, 1)
    return seen, max_depth, errors


def main() -> None:
    args = parse_args()
    config, config_path = load_config(args.config)
    data_dir = resolve_path(config_path, config["data"]["output_dir"])
    path = args.input_jsonl.resolve() if args.input_jsonl else data_dir / config["data"]["raw_comment_trees_jsonl"]
    records = read_jsonl(path)
    global_ids: set[str] = set()
    errors: list[str] = []
    roots = orphan_roots = nodes = max_depth = 0
    for record in records:
        news_ids: set[str] = set()
        for root in record["roots"]:
            found, depth, found_errors = walk(root, record["news_id"])
            duplicate = news_ids & found
            if duplicate:
                errors.append(f"duplicate_across_roots:{record['news_id']}:{sorted(duplicate)[:3]}")
            news_ids |= found
            errors.extend(found_errors)
            max_depth = max(max_depth, depth)
            roots += 1
        for root in record["orphans"]:
            found, depth, found_errors = walk(root, record["news_id"])
            duplicate = news_ids & found
            if duplicate:
                errors.append(f"duplicate_across_components:{record['news_id']}:{sorted(duplicate)[:3]}")
            news_ids |= found
            errors.extend(found_errors)
            max_depth = max(max_depth, depth)
            orphan_roots += 1
        repeated_global = global_ids & news_ids
        if repeated_global:
            errors.append(f"comment_id_cross_news:{sorted(repeated_global)[:3]}")
        global_ids |= news_ids
        nodes += len(news_ids)
        expected = int(record["tree_stats"]["valid_comments"])
        if len(news_ids) != expected:
            errors.append(
                f"node_count_mismatch:{record['news_id']}:expected={expected}:found={len(news_ids)}"
            )
    result = {
        "news": len(records),
        "root_threads": roots,
        "orphan_root_threads": orphan_roots,
        "unique_comment_nodes": nodes,
        "max_depth": max_depth,
        "errors": len(errors),
        "error_examples": errors[:20],
    }
    report = args.report_path.resolve() if args.report_path else config_path.parent.parent / "reports" / "comment_tree_validation.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if errors:
        raise RuntimeError(f"Comment tree validation failed with {len(errors)} errors")


if __name__ == "__main__":
    main()
