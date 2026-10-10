"""Synthetic-model integration smoke on real features/graphs; no quality claim."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from comment_graph.filter_generated_by_quality import filter_graph, infer_probs, read_unique  # noqa: E402
from comment_graph.train_generated_quality_gate import CONTENT_FEATURES, PARENT_FEATURES  # noqa: E402


def mock_model(width: int):
    # Synthetic separable vectors test interface shape only; never fit on
    # news or generated-comment quality labels.
    x = np.vstack([np.full(width, -1.0), np.full(width, 1.0)])
    return make_pipeline(StandardScaler(), LogisticRegression()).fit(x, [0, 1])


def check(graph_path: Path, feature_path: Path, split: str) -> dict:
    graphs = read_unique(graph_path, "news_id")
    features = read_unique(feature_path, "annotation_id")
    package = {"content_model": mock_model(len(CONTENT_FEATURES)),
               "parent_model": mock_model(len(PARENT_FEATURES)),
               "content_features": CONTENT_FEATURES, "parent_features": PARENT_FEATURES,
               "content_threshold": 0.5, "parent_threshold": 0.5}
    probs = infer_probs(package, features, split)
    if set(probs) != {str(item["generated_node_id"])
                      for graph in graphs.values() for item in graph["generated_nodes"]}:
        raise ValueError("Synthetic inference coverage mismatch")
    validator = (PROJECT_ROOT / "comment_generator" / "partial_observation" /
                 "scripts" / "validate_generated_structured_graphs.py")
    kept = 0
    with tempfile.TemporaryDirectory(prefix="quality_inference_smoke_") as directory:
        output = Path(directory) / "graphs.jsonl"
        with output.open("w", encoding="utf-8", newline="\n") as handle:
            for graph in graphs.values():
                filtered, _ = filter_graph(graph, probs, 0.5, 0.5)
                original_real = [node for node in graph["graph_nodes"]
                                 if node["node_type"] != "generated_comment"]
                original_edges = [edge for edge in graph["edges"] if edge["edge_type"] == 0]
                assert filtered["graph_nodes"][:len(original_real)] == original_real
                assert [edge for edge in filtered["edges"] if edge["edge_type"] == 0] == original_edges
                kept += len(filtered["generated_nodes"])
                handle.write(json.dumps(filtered, ensure_ascii=False) + "\n")
        subprocess.run([sys.executable, str(validator), "--input-jsonl", str(output),
                        "--expected-split", split], check=True, stdout=subprocess.DEVNULL)
    return {"split": split, "news": len(graphs), "generated_before": len(probs),
            "generated_after_arbitrary_synthetic_model": kept,
            "structured_graph_valid": True, "human_quality_model_used": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-graphs", type=Path, required=True)
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--validate-graphs", type=Path, required=True)
    parser.add_argument("--validate-features", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps({"train": check(args.train_graphs, args.train_features, "train"),
                      "validate": check(args.validate_graphs, args.validate_features, "validate")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
