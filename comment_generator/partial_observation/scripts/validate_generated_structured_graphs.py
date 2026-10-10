"""Validate persisted structure-aware generated comment graphs before GNN use."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from common import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate generated comment graph JSONL.")
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--expected-split", choices=("train", "validate", "test"), required=True)
    parser.add_argument(
        "--expected-context-jsonl",
        type=Path,
        default=None,
        help="Optional structure_* file; require exactly one generated graph for every news ID in it.",
    )
    parser.add_argument(
        "--min-observed-nodes",
        type=int,
        default=0,
        help="When checking expected contexts, use only records with at least this many observed real nodes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.input_jsonl.resolve())
    if not records:
        raise RuntimeError("No generated graphs to validate")
    news_ids: set[str] = set()
    generated_counts: Counter[int] = Counter()
    edge_counts: Counter[int] = Counter()
    depth_policies: set[str] = set()
    context_policies: set[str] = set()
    failures = 0
    for record in records:
        required = {
            "sample_id", "news_id", "official_split", "image_path", "graph_nodes", "edges",
            "generated_nodes", "generation_failures", "adapter_dir", "generation_config",
            "parent_selection_policy", "label_used_for_generation", "time_interpretation",
        }
        if set(record) != required:
            raise ValueError(f"Unexpected output schema for {record.get('news_id')}")
        news_id = str(record["news_id"])
        if not news_id or news_id in news_ids:
            raise ValueError(f"Missing or duplicate news_id: {news_id!r}")
        news_ids.add(news_id)
        if record["official_split"] != args.expected_split:
            raise ValueError(f"Split mismatch for {news_id}")
        if record["label_used_for_generation"] is not False:
            raise ValueError(f"Label use was incorrectly recorded for {news_id}")
        depth_reference = record["generation_config"].get("depth_reference", "news_root")
        depth_policies.add(depth_reference)
        policies = {"news_root": "model_predicted_with_depth_bounded_candidates",
                    "observed_boundary": "model_predicted_with_generated_depth_bounded_candidates"}
        context_policy = record["generation_config"].get("context_policy", "iterative")
        context_policies.add(context_policy)
        expected_policy = ("model_predicted_news_or_real_only" if context_policy == "observed_only"
                           else policies.get(depth_reference))
        if (context_policy not in {"iterative", "observed_only"} or depth_reference not in policies
                or record["parent_selection_policy"] != expected_policy):
            raise ValueError(f"Unexpected parent selection policy for {news_id}")
        nodes = record["graph_nodes"]
        if not nodes or nodes[0] != {"node_index": 0, "node_type": "news", "content": nodes[0].get("content", "")}:
            raise ValueError(f"Invalid news root node for {news_id}")
        if not str(nodes[0]["content"]).strip():
            raise ValueError(f"Blank news root text for {news_id}")
        indices = [int(node.get("node_index", -1)) for node in nodes]
        if indices != list(range(len(nodes))):
            raise ValueError(f"Non-contiguous node indices for {news_id}")
        types = [node.get("node_type") for node in nodes]
        if types[0] != "news" or any(item not in {"news", "real_comment", "generated_comment"} for item in types):
            raise ValueError(f"Invalid node type for {news_id}")
        if any(not str(node.get("content", "")).strip() for node in nodes):
            raise ValueError(f"Blank node content for {news_id}")
        generated = [node for node in nodes if node["node_type"] == "generated_comment"]
        if len(generated) != len(record["generated_nodes"]):
            raise ValueError(f"Generated-node count mismatch for {news_id}")
        generated_texts = [str(node["content"]).casefold() for node in generated]
        if len(generated_texts) != len(set(generated_texts)):
            raise ValueError(f"Duplicate generated text for {news_id}")
        parents: dict[int, int] = {}
        for edge in record["edges"]:
            if set(edge) != {"source", "target", "edge_type"}:
                raise ValueError(f"Invalid edge schema for {news_id}")
            source, target, edge_type = int(edge["source"]), int(edge["target"]), int(edge["edge_type"])
            if not (0 <= source < len(nodes) and 0 < target < len(nodes) and source != target):
                raise ValueError(f"Invalid edge endpoints for {news_id}")
            expected = 1 if nodes[target]["node_type"] == "generated_comment" else 0
            if edge_type != expected:
                raise ValueError(f"Incorrect edge relation for {news_id}:{source}->{target}")
            if context_policy == "observed_only" and nodes[source]["node_type"] == "generated_comment":
                raise ValueError(f"Synthetic parent forbidden by observed_only: {news_id}:{source}->{target}")
            if target in parents:
                raise ValueError(f"Multiple parent edges for {news_id}:{target}")
            # The generator appends nodes only after their parents exist.
            # This also rejects cycles without relying on a depth limit.
            if source >= target:
                raise ValueError(f"Parent must precede child for {news_id}:{source}->{target}")
            parents[target] = source
            edge_counts[edge_type] += 1
        if set(parents) != set(range(1, len(nodes))):
            raise ValueError(f"Disconnected nodes in generated tree for {news_id}")
        generated_by_index = {int(node["node_index"]): node for node in record["generated_nodes"]}
        if set(generated_by_index) != {int(node["node_index"]) for node in generated}:
            raise ValueError(f"Generated-node identity mismatch for {news_id}")
        for node in generated:
            index = int(node["node_index"])
            detail = generated_by_index[index]
            parent = parents[index]
            expected_parent_id = ("news" if parent == 0 else
                                  nodes[parent].get("source_comment_id") or nodes[parent].get("generated_node_id"))
            if (int(detail["parent_node_index"]) != parent or
                    detail["content"] != node["content"] or
                    not expected_parent_id or
                    detail.get("parent_comment_id") != expected_parent_id or
                    node.get("parent_comment_id") != expected_parent_id or
                    detail.get("generated_node_id") != node.get("generated_node_id")):
                raise ValueError(f"Generated edge/raw-ID mapping mismatch for {news_id}:{index}")
            depth, ancestor = 0, index
            while ancestor:
                if depth_reference == "observed_boundary" and nodes[ancestor]["node_type"] == "real_comment":
                    break
                depth += 1
                ancestor = parents[ancestor]
            if depth > int(record["generation_config"]["max_generated_depth"]):
                raise ValueError(f"Generated node exceeds depth limit for {news_id}:{index}")
        generated_counts[len(generated)] += 1
        failures += len(record["generation_failures"])
    if len(depth_policies) != 1:
        raise ValueError(f"Mixed generation depth policies in one graph file: {sorted(depth_policies)}")
    if len(context_policies) != 1:
        raise ValueError("Mixed context policies in one graph file")
    expected_news = None
    if args.expected_context_jsonl:
        contexts = read_jsonl(args.expected_context_jsonl.resolve())
        expected_news = {
            str(row.get("metadata", {}).get("news_id", ""))
            for row in contexts
            if len(row.get("model_input", {}).get("observed_nodes", [])) >= args.min_observed_nodes
        }
        if "" in expected_news or len(expected_news) == 0:
            raise ValueError("Expected context file has a missing or empty news ID")
        if expected_news != news_ids:
            missing, extra = sorted(expected_news - news_ids), sorted(news_ids - expected_news)
            raise ValueError(
                f"Generated coverage mismatch: missing={len(missing)} extra={len(extra)} "
                f"missing_examples={missing[:5]} extra_examples={extra[:5]}"
            )
    print(json.dumps({
        "graphs": len(records), "unique_news": len(news_ids),
        "official_split": args.expected_split,
        "generated_node_count_distribution": dict(sorted(generated_counts.items())),
        "edge_type_counts": dict(sorted(edge_counts.items())),
        "generation_failure_attempts": failures,
        "depth_reference": next(iter(depth_policies)),
        "expected_news_coverage": len(expected_news) if expected_news is not None else None,
        "expected_min_observed_nodes": args.min_observed_nodes,
        "generation_label_use_flag_check": "passed; provenance flag only, not a full leakage audit",
        "schema_and_edge_check": "passed",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
