"""Fast invariant checks for the Fakeddit reply-tree adapter (no model/data download)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from comment_graph.fakeddit_structure_dataset import bidirectional, flatten_real_nodes, root_generated_edges, select_real_nodes


def node(comment_id, parent_id, children=None):
    return {"comment_id": comment_id, "parent_id": parent_id, "body": comment_id, "children": children or []}


def main() -> None:
    tree = {
        "roots": [
            node("r1", "news", [node("c1", "r1", [node("g1", "c1")])]),
            node("r2", "news"),
        ]
    }
    selected, edges = flatten_real_nodes(tree, limit=3)
    assert [item["comment_id"] for item in selected] == ["r1", "c1", "g1"]
    assert edges == [(0, 1, 0), (1, 2, 0), (2, 3, 0)]
    edge_index, edge_type = bidirectional(edges, {0, 1, 2, 3})
    pairs = {tuple(pair) for pair in edge_index.t().tolist()}
    assert pairs == {(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)}
    assert edge_type.tolist() == [0] * 6
    augmented = [*edges, (2, 4, 1), (4, 5, 1)]
    rooted = root_generated_edges(augmented)
    assert rooted[:len(edges)] == edges
    assert rooted[len(edges):] == [(0, 4, 1), (0, 5, 1)]
    assert [target for _, target, _ in rooted] == [target for _, target, _ in augmented]
    kept, kept_edges = select_real_nodes(tree, 3, ["r2"])
    assert [n["comment_id"] for n in kept] == ["r1", "c1", "r2"]
    assert kept_edges == [(0, 1, 0), (1, 2, 0), (0, 3, 0)]
    assert select_real_nodes(tree, 3, ["g1"])[0] == selected
    for required, limit in [(["g1"], 2), (["missing"], 4)]:
        try:
            select_real_nodes(tree, limit, required)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid required context was not rejected")
    print(json.dumps({
        "selected_comment_ids": [item["comment_id"] for item in selected],
        "directed_edge_count": int(edge_index.size(1)),
        "checks": ["parent_first", "truncation_preserves_parent", "bidirectional_edges",
                   "generated_root_preserves_real_tree_and_node_set",
                   "context_and_ancestors_preserved_with_fixed_budget", "missing_or_oversized_context_rejected"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
