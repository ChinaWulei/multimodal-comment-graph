"""Verify a gated graph differs only by dropping generated subtrees.

This is stronger than checking graph validity: it proves that news, observed
real discussion, surviving generated texts, and all surviving parent IDs are
identical to the source graph. It does not assess comment quality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def read_unique(path: Path, split: str) -> dict[str, dict]:
    result = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        key = str(row["news_id"])
        if key in result or row["official_split"] != split:
            raise ValueError(f"Duplicate news or wrong split at {path}:{number}: {key}")
        result[key] = row
    if not result:
        raise ValueError(f"Empty graph file: {path}")
    return result


def audit(original: dict, gated: dict) -> Counter:
    news_id = str(original["news_id"])
    if any(original[key] != gated[key] for key in
           ("sample_id", "news_id", "official_split", "image_path", "adapter_dir",
            "generation_failures",
            "label_used_for_generation", "time_interpretation", "parent_selection_policy")):
        raise ValueError(f"News/source provenance changed: {news_id}")
    original_config = original["generation_config"]
    gated_config = {key: value for key, value in gated["generation_config"].items()
                    if key != "posthoc_quality_gate"}
    if original_config != gated_config:
        raise ValueError(f"Generator configuration changed: {news_id}")
    original_real = [node for node in original["graph_nodes"]
                     if node["node_type"] != "generated_comment"]
    gated_real = [node for node in gated["graph_nodes"]
                  if node["node_type"] != "generated_comment"]
    if original_real != gated_real:
        raise ValueError(f"News or observed real nodes changed: {news_id}")
    original_real_edges = [edge for edge in original["edges"] if edge["edge_type"] == 0]
    gated_real_edges = [edge for edge in gated["edges"] if edge["edge_type"] == 0]
    if original_real_edges != gated_real_edges:
        raise ValueError(f"Observed real edges changed: {news_id}")
    original_generated = {str(item["generated_node_id"]): item
                          for item in original["generated_nodes"]}
    gated_generated = {str(item["generated_node_id"]): item
                       for item in gated["generated_nodes"]}
    if len(original_generated) != len(original["generated_nodes"]) or len(gated_generated) != len(gated["generated_nodes"]):
        raise ValueError(f"Duplicate generated ID: {news_id}")
    if not set(gated_generated) <= set(original_generated):
        raise ValueError(f"New generated comment appeared: {news_id}")
    original_order = [str(item["generated_node_id"]) for item in original["generated_nodes"]]
    gated_order = [str(item["generated_node_id"]) for item in gated["generated_nodes"]]
    if gated_order != [key for key in original_order if key in gated_generated]:
        raise ValueError(f"Generated node order changed: {news_id}")
    original_node_by_id = {str(node["generated_node_id"]): node for node in original["graph_nodes"]
                           if node["node_type"] == "generated_comment"}
    gated_node_by_id = {str(node["generated_node_id"]): node for node in gated["graph_nodes"]
                        if node["node_type"] == "generated_comment"}
    real_ids = {"news", *(str(node["source_comment_id"]) for node in original_real[1:])}
    for key, item in gated_generated.items():
        source = original_generated[key]
        omitted = {"node_index", "parent_node_index", "original_node_index",
                   "quality_gate_content_probability", "quality_gate_parent_probability"}
        if {field: value for field, value in item.items() if field not in omitted} != {
                field: value for field, value in source.items() if field not in omitted}:
            raise ValueError(f"Generated content/parent/provenance changed: {news_id}/{key}")
        if {field: value for field, value in gated_node_by_id[key].items() if field != "node_index"} != {
                field: value for field, value in original_node_by_id[key].items() if field != "node_index"}:
            raise ValueError(f"Generated graph-node metadata changed: {news_id}/{key}")
        parent_id = str(item["parent_comment_id"])
        if parent_id not in real_ids and parent_id not in gated_generated:
            raise ValueError(f"Kept child of a rejected generated parent: {news_id}/{key}")
        if item.get("original_node_index", source["node_index"]) != source["node_index"]:
            raise ValueError(f"Original index provenance changed: {news_id}/{key}")
    if set(gated_node_by_id) != set(gated_generated):
        raise ValueError(f"Generated detail/node mismatch: {news_id}")
    return Counter({"news": 1, "real_nodes": len(original_real),
                    "real_edges": len(original_real_edges),
                    "generated_before": len(original_generated),
                    "generated_after": len(gated_generated),
                    "generated_dropped": len(original_generated) - len(gated_generated),
                    "news_with_zero_generated": int(not gated_generated)})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-jsonl", type=Path, required=True)
    parser.add_argument("--gated-jsonl", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validate", "test"), required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError(args.output_json)
    original = read_unique(args.original_jsonl, args.split)
    gated = read_unique(args.gated_jsonl, args.split)
    if set(original) != set(gated):
        raise ValueError("Original and gated news coverage differ")
    counts: Counter[str] = Counter()
    for key in original:
        counts.update(audit(original[key], gated[key]))
    result = {"split": args.split, **dict(counts),
              "original_sha256": hashlib.sha256(args.original_jsonl.read_bytes()).hexdigest(),
              "gated_sha256": hashlib.sha256(args.gated_jsonl.read_bytes()).hexdigest(),
              "news_real_nodes_edges_and_survivor_parents_identical": True,
              "comment_quality_assessed": False}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
