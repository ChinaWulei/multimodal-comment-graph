"""Check merged quality/candidate features against immutable graph IDs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from comment_graph.filter_generated_by_quality import read_unique
from comment_graph.train_generated_quality_gate import CONTENT_FEATURES, PARENT_FEATURES


def check(graph_path: Path, features_path: Path, metadata_path: Path, split: str) -> dict:
    graphs = read_unique(graph_path, "news_id")
    features = read_unique(features_path, "annotation_id")
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    wanted = {str(item["generated_node_id"]): (news_id, int(item["parent_node_index"]))
              for news_id, graph in graphs.items() for item in graph["generated_nodes"]}
    if len(wanted) != sum(len(graph["generated_nodes"]) for graph in graphs.values()):
        raise ValueError("Duplicate generated node ID in source graphs")
    if (set(wanted) != set(features) or meta["rows"] != len(features) or
            meta["split"] != split or
            meta["candidate_source_graph_sha256"] != hashlib.sha256(graph_path.read_bytes()).hexdigest()):
        raise ValueError(f"Graph/feature/metadata coverage or provenance mismatch: {split}")
    if any(graph["official_split"] != split for graph in graphs.values()):
        raise ValueError("Wrong graph split")
    names = set(CONTENT_FEATURES) | set(PARENT_FEATURES)
    for key, row in features.items():
        if row["news_id"] != wanted[key][0] or row["official_split"] != split:
            raise ValueError(f"News/split mismatch: {key}")
        values = np.asarray([float(row[name]) for name in names])
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite quality features: {key}")
        if not 0 <= float(row["reranker_chosen_parent_rank_fraction"]) <= 1:
            raise ValueError(f"Invalid rank fraction: {key}")
    return {"split": split, "news": len(graphs), "generated_features": len(features),
            "source_models": {name: meta[name] for name in
                              ("reranker_snapshot", "clip_snapshot", "candidate_max_length")},
            "ids_and_numeric_features_valid": True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-graphs", type=Path, required=True)
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--train-metadata", type=Path, required=True)
    parser.add_argument("--validate-graphs", type=Path, required=True)
    parser.add_argument("--validate-features", type=Path, required=True)
    parser.add_argument("--validate-metadata", type=Path, required=True)
    args = parser.parse_args()
    train = check(args.train_graphs, args.train_features, args.train_metadata, "train")
    validate = check(args.validate_graphs, args.validate_features, args.validate_metadata,
                     "validate")
    if train["source_models"] != validate["source_models"]:
        raise ValueError("Train/validation frozen models differ")
    if set(read_unique(args.train_features, "annotation_id")) & set(
            read_unique(args.validate_features, "annotation_id")):
        raise ValueError("Train/validation generated IDs overlap")
    print(json.dumps({"train": train, "validate": validate,
                      "train_validate_id_overlap": 0}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
