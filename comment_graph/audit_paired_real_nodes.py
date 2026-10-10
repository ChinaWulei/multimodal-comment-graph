"""Assert generated-comment ablations keep identical observed real discussion."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from comment_graph.common import (generated_coverage_ids, load_config, make_dataset,
                                  make_tokenizer_and_transform)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--generated-jsonl", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "validate", "test"], required=True)
    parser.add_argument("--max-real-comments", type=int, default=16)
    parser.add_argument("--max-generated-comments", type=int, default=5)
    parser.add_argument("--graph-structure", choices=["observed_tree", "generated_root"],
                        default="observed_tree")
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    config, config_path = load_config(args.config)
    config.data.generated_comments[args.split] = str(args.generated_jsonl.resolve())
    config.data.max_real_comments = args.max_real_comments
    config.data.max_generated_comments = args.max_generated_comments
    config.graph.structure_mode = args.graph_structure
    tokenizer, transform = make_tokenizer_and_transform(config)
    allowed = generated_coverage_ids(config, config_path, args.split)
    real = make_dataset(config, config_path, args.split, "real_only", tokenizer, transform,
                        allowed_news_ids=allowed)
    augmented = make_dataset(config, config_path, args.split, "real_generated", tokenizer, transform,
                             allowed_news_ids=allowed)
    if len(real) != len(augmented) or len(real) != len(allowed):
        raise ValueError("Split graph coverage differs across paired arms")
    zero_generated = generated_count = 0
    for index in range(len(real)):
        left, right = real[index], augmented[index]
        if left.news_id != right.news_id or int(left.y) != int(right.y):
            raise ValueError(f"News/label mismatch at {index}")
        n = int(left.num_real_comments)
        m = int(right.num_real_comments)
        if n != m or not torch.equal(left.node_type, right.node_type[: n + 1]):
            raise ValueError(f"Real-node counts/types differ for {left.news_id}")
        if not torch.equal(left.node_input_ids, right.node_input_ids[: n + 1]):
            raise ValueError(f"Real node text differs for {left.news_id}")
        left_real_edges = left.edge_index[:, left.edge_type == 0]
        right_real_edges = right.edge_index[:, right.edge_type == 0]
        if not torch.equal(left_real_edges, right_real_edges):
            raise ValueError(f"Observed real edges differ for {left.news_id}")
        k = int(right.num_generated_comments)
        zero_generated += k == 0
        generated_count += k
    result = {"split": args.split, "graph_structure": args.graph_structure,
              "paired_news": len(real), "real_node_budget": args.max_real_comments,
              "generated_node_budget": args.max_generated_comments,
              "generated_nodes": generated_count, "news_with_zero_generated": zero_generated,
              "news_ids_identical": True, "real_text_and_edges_identical": True}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
