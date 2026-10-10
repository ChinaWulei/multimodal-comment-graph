"""Synthetic mutation checks for the source-versus-pruned graph auditor."""

from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from comment_graph.audit_pruned_graph_pair import audit, read_unique  # noqa: E402
from comment_graph.filter_generated_by_quality import filter_graph  # noqa: E402


def expect_rejected(original: dict, modified: dict) -> None:
    try:
        audit(original, modified)
    except ValueError:
        return
    raise AssertionError("Graph mutation was not detected")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs-jsonl", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validate"), required=True)
    args = parser.parse_args()
    graphs = read_unique(args.graphs_jsonl, args.split)
    first_pair = None
    for graph in graphs.values():
        ids = [str(item["generated_node_id"]) for item in graph["generated_nodes"]]
        probs = {key: ((1.0, 1.0) if i % 2 == 0 else (0.0, 0.0))
                 for i, key in enumerate(ids)}
        filtered, _ = filter_graph(graph, probs, 0.5, 0.5)
        assert audit(graph, filtered)["generated_after"] <= len(ids)
        if first_pair is None:
            first_pair = graph, filtered
    graph, filtered = first_pair
    changed_real = deepcopy(filtered)
    changed_real["graph_nodes"][0]["content"] += " MUTATED"
    expect_rejected(graph, changed_real)
    if filtered["generated_nodes"]:
        changed_parent = deepcopy(filtered)
        changed_parent["generated_nodes"][0]["parent_comment_id"] = "invented_parent"
        expect_rejected(graph, changed_parent)
    print(f"Paired-graph auditor accepted pruning on {len(graphs)} {args.split} news "
          "and rejected real-text/parent mutations.")


if __name__ == "__main__":
    main()
