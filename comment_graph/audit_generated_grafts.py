"""Read-only audit of generated parent links after actual dataset truncation.

Uses the same dataset construction methods as training, without loading any
encoder, tokenizing text, training, or modifying the selected news/edges.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from comment_graph.common import load_config, make_dataset
from comment_graph.fakeddit_structure_dataset import root_generated_edges


def audit_dataset(dataset):
    totals = Counter()
    details, issues = [], []
    real_only = copy.copy(dataset)
    real_only.mode = "real_only"
    for row in dataset.records:
        news_id = row["news_id"]
        real, real_edges = dataset._real_nodes_and_edges(news_id)
        texts, types, edges = dataset._nodes_and_edges(news_id)
        control_texts, control_types, control_edges = real_only._nodes_and_edges(news_id)
        if (control_texts != texts[:len(real)] or control_types != types[:len(real)] or
                control_edges != [e for e in edges if e[2] == 0]):
            issues.append({"news_id": news_id, "reason": "paired_real_only_arm_differs"})
        generated = dataset.generated_graphs[news_id]["generated_nodes"][:dataset.max_generated_comments]
        if texts[:len(real)] != [str(n["body"]).strip() for n in real]:
            issues.append({"news_id": news_id, "reason": "real_texts_changed"})
        if [e for e in edges if e[2] == 0] != real_edges:
            issues.append({"news_id": news_id, "reason": "real_edges_changed"})
        if len(texts) != len(real) + len(generated):
            issues.append({"news_id": news_id, "reason": "node_count_changed"})
        if types != [1] * len(real) + [2] * len(generated):
            issues.append({"news_id": news_id, "reason": "node_types_changed"})
        local = {str(n["comment_id"]): i + 1 for i, n in enumerate(real)}
        generated_ids = set()
        expected_edges = []
        counts = Counter()
        for i, node in enumerate(generated, start=1):
            if "parent_comment_id" not in node or "generated_node_id" not in node:
                issues.append({"news_id": news_id, "reason": "missing_raw_parent_identity"})
                continue
            parent = str(node["parent_comment_id"])
            if parent == "news":
                kind = "predicted_news_root"
            elif parent in local:
                kind = "predicted_generated_parent" if parent in generated_ids else "predicted_real_parent"
            else:
                kind = "missing_parent_fallback_to_root"
                details.append({"news_id": news_id, "generated_node_id": node["generated_node_id"],
                                "missing_parent_comment_id": parent})
            source, target = local.get(parent, 0), len(real) + i
            expected_edges.append((source, target, 1))
            if source >= target:
                issues.append({"news_id": news_id, "reason": "self_or_forward_parent"})
            if texts[target - 1] != node["content"].strip():
                issues.append({"news_id": news_id, "reason": "generated_text_changed"})
            counts[kind] += 1
            local[str(node["generated_node_id"])] = target
            generated_ids.add(str(node["generated_node_id"]))
        if [e for e in edges if e[2] == 1] != expected_edges:
            issues.append({"news_id": news_id, "reason": "generated_edge_mapping_mismatch"})
        rooted = root_generated_edges(edges)
        if [e for e in rooted if e[2] == 0] != real_edges:
            issues.append({"news_id": news_id, "reason": "root_control_changes_real_edges"})
        totals.update(counts)
        totals["news"] += 1
        totals["real_nodes"] += len(real)
        totals["generated_nodes"] += len(generated)
        totals["news_with_nonroot_generated_parent"] += any(e[0] != 0 for e in expected_edges)
        totals["news_with_parent_fallback"] += counts["missing_parent_fallback_to_root"] > 0
    return {"counts": dict(totals), "parent_fallbacks": details, "errors": issues,
            "all_predicted_parents_preserved": not details}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validate", "test"), action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    config, config_path = load_config(args.config)
    if config.data.dataset_name != "fakeddit_structure":
        raise ValueError("This audit requires the Fakeddit structure adapter")
    result = {"config": str(config_path), "max_real_comments": int(config.data.max_real_comments),
              "max_generated_comments": int(config.data.max_generated_comments),
              "real_node_selection": str(config.data.get("real_node_selection", "prefix")),
              "scope": "Currently persisted graphs only; this does not certify complete split coverage.",
              "splits": {}}
    for split in dict.fromkeys(args.split):
        dataset = make_dataset(config, config_path, split, "real_generated", None, None)
        if {r["news_id"] for r in dataset.records} != set(dataset.generated_graphs):
            raise ValueError(f"Persisted graphs excluded by dataset eligibility in {split}")
        result["splits"][split] = audit_dataset(dataset)
    result["passed"] = not any(r["errors"] for r in result["splits"].values())
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
