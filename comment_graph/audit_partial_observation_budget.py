"""Verify a smaller real-comment budget preserves generator-visible context."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from comment_graph.common import generated_coverage_ids, load_config, make_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-real-comments", type=int, default=16)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    if args.max_real_comments < 1:
        raise ValueError("max-real-comments must be positive")
    config, config_path = load_config(args.config)
    config.data.max_real_comments = args.max_real_comments
    result = {"max_real_comments": args.max_real_comments, "splits": {}}
    for split in ("train", "validate", "test"):
        allowed = generated_coverage_ids(config, config_path, split)
        dataset = make_dataset(config, config_path, split, "real_generated", None, None,
                               allowed_news_ids=allowed)
        counts = []
        generated_counts = []
        for row in dataset.records:
            news_id = str(row["news_id"])
            texts, types, edges = dataset._nodes_and_edges(news_id)
            real = sum(node_type == 1 for node_type in types)
            generated = sum(node_type == 2 for node_type in types)
            if real > args.max_real_comments or generated > int(config.data.max_generated_comments):
                raise ValueError(f"Comment budget exceeded: {news_id}")
            if any(not 0 <= source < target <= len(texts) for source, target, _ in edges):
                raise ValueError(f"Invalid directed parent edge: {news_id}")
            counts.append(real)
            generated_counts.append(generated)
        result["splits"][split] = {
            "news": len(dataset), "real_nodes_min": min(counts), "real_nodes_max": max(counts),
            "real_nodes_mean": sum(counts) / len(counts),
            "generated_nodes_mean": sum(generated_counts) / len(generated_counts),
        }
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
