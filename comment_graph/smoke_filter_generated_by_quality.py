"""Synthetic branch-pruning invariant smoke; not a model-quality result."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from comment_graph.filter_generated_by_quality import filter_graph, read_unique  # noqa: E402


def smoke_real_graphs(path: Path, split: str) -> None:
    """Exercise pruning invariants on actual generated graphs, without a gate."""
    graphs = read_unique(path, "news_id")
    if not graphs or any(graph["official_split"] != split for graph in graphs.values()):
        raise ValueError("Empty or wrong-split graph input")
    validator = (PROJECT_ROOT / "comment_generator" / "partial_observation" /
                 "scripts" / "validate_generated_structured_graphs.py")
    totals = {"news": 0, "original_generated": 0, "partial_kept": 0}
    with tempfile.TemporaryDirectory(prefix="generated_prune_smoke_") as directory:
        for variant in ("keep_all", "reject_all", "alternating"):
            output = Path(directory) / f"{variant}.jsonl"
            with output.open("w", encoding="utf-8", newline="\n") as handle:
                for graph in graphs.values():
                    ids = [str(item["generated_node_id"]) for item in graph["generated_nodes"]]
                    if variant == "keep_all":
                        probabilities = {key: (1.0, 1.0) for key in ids}
                    elif variant == "reject_all":
                        probabilities = {key: (0.0, 0.0) for key in ids}
                    else:
                        probabilities = {key: ((1.0, 1.0) if i % 2 == 0 else (0.0, 0.0))
                                         for i, key in enumerate(ids)}
                    filtered, _ = filter_graph(graph, probabilities, 0.5, 0.5)
                    original_real_nodes = [node for node in graph["graph_nodes"]
                                           if node["node_type"] != "generated_comment"]
                    original_real_edges = [edge for edge in graph["edges"] if edge["edge_type"] == 0]
                    assert filtered["graph_nodes"][:len(original_real_nodes)] == original_real_nodes
                    assert [edge for edge in filtered["edges"] if edge["edge_type"] == 0] == original_real_edges
                    if variant == "keep_all":
                        assert len(filtered["generated_nodes"]) == len(ids)
                        assert filtered["edges"] == graph["edges"]
                    elif variant == "reject_all":
                        assert not filtered["generated_nodes"]
                    else:
                        totals["partial_kept"] += len(filtered["generated_nodes"])
                    handle.write(json.dumps(filtered, ensure_ascii=False) + "\n")
                    if variant == "keep_all":
                        totals["news"] += 1
                        totals["original_generated"] += len(ids)
            subprocess.run([sys.executable, str(validator), "--input-jsonl", str(output),
                            "--expected-split", split], check=True, stdout=subprocess.DEVNULL)
    print(json.dumps({"split": split, **totals, "variants_validated": 3,
                      "quality_model_tested": False}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs-jsonl", type=Path,
                        help="Optional actual generated graphs for keep/drop/partial invariant checks")
    parser.add_argument("--split", choices=("train", "validate", "test"))
    args = parser.parse_args()
    if bool(args.graphs_jsonl) != bool(args.split):
        parser.error("--graphs-jsonl and --split must be supplied together")
    graph = {
        "news_id": "fixture_news",
        "graph_nodes": [
            {"node_index": 0, "node_type": "news", "content": "news title"},
            {"node_index": 1, "node_type": "real_comment", "source_comment_id": "r1", "content": "real"},
            {"node_index": 2, "node_type": "generated_comment", "generated_node_id": "g2", "parent_comment_id": "r1", "content": "bad"},
            {"node_index": 3, "node_type": "generated_comment", "generated_node_id": "g3", "parent_comment_id": "g2", "content": "descendant"},
            {"node_index": 4, "node_type": "generated_comment", "generated_node_id": "g4", "parent_comment_id": "r1", "content": "good"},
        ],
        "edges": [
            {"source": 0, "target": 1, "edge_type": 0},
            {"source": 1, "target": 2, "edge_type": 1},
            {"source": 2, "target": 3, "edge_type": 1},
            {"source": 1, "target": 4, "edge_type": 1},
        ],
        "generated_nodes": [
            {"node_index": 2, "parent_node_index": 1, "generated_node_id": "g2", "parent_comment_id": "r1", "content": "bad"},
            {"node_index": 3, "parent_node_index": 2, "generated_node_id": "g3", "parent_comment_id": "g2", "content": "descendant"},
            {"node_index": 4, "parent_node_index": 1, "generated_node_id": "g4", "parent_comment_id": "r1", "content": "good"},
        ],
        "generation_config": {"depth_reference": "observed_boundary", "max_generated_depth": 6},
    }
    scores = {"g2": (0.1, 0.9), "g3": (0.9, 0.9), "g4": (0.9, 0.9)}
    filtered, reasons = filter_graph(graph, scores, 0.5, 0.5)
    assert reasons == {"content_rejected": 1, "ancestor_rejected": 1, "accepted": 1}
    assert [node["node_index"] for node in filtered["graph_nodes"]] == [0, 1, 2]
    assert filtered["graph_nodes"][:2] == graph["graph_nodes"][:2]
    assert filtered["edges"] == [graph["edges"][0], {"source": 1, "target": 2, "edge_type": 1}]
    assert filtered["generated_nodes"][0]["generated_node_id"] == "g4"
    assert filtered["generated_nodes"][0]["original_node_index"] == 4
    assert filtered["generated_nodes"][0]["parent_node_index"] == 1
    assert len(graph["generated_nodes"]) == 3  # Input was not mutated.
    all_kept, counts = filter_graph(graph, {key: (0.9, 0.9) for key in scores}, 0.5, 0.5)
    assert counts == {"accepted": 3}
    assert all_kept["edges"] == graph["edges"]
    none_kept, counts = filter_graph(graph, {key: (0.0, 0.0) for key in scores}, 0.5, 0.5)
    assert not none_kept["generated_nodes"]
    assert none_kept["graph_nodes"] == graph["graph_nodes"][:2]
    print("Synthetic branch-pruning smoke passed; no real gate was trained.")
    if args.graphs_jsonl:
        smoke_real_graphs(args.graphs_jsonl, args.split)


if __name__ == "__main__":
    main()
