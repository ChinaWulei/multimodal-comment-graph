"""Conservatively prune unsafe generated branches before classifier loading.

The gate is trained on human-reviewed *training* comments. This postprocessor
never reads news truth labels and never invents replacement reply parents.
If a generated parent is rejected, every descendant on that branch is removed.
Original news and real-comment nodes/edges are preserved exactly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from collections import Counter
from copy import deepcopy
from pathlib import Path

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_unique(path: Path, key: str) -> dict[str, dict]:
    result = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        identity = str(row[key])
        if identity in result:
            raise ValueError(f"Duplicate {key}: {path}:{number}/{identity}")
        result[identity] = row
    return result


def infer_probs(package: dict, features: dict[str, dict], split: str) -> dict[str, tuple[float, float]]:
    required = ("content_model", "parent_model", "content_features", "parent_features",
                "content_threshold", "parent_threshold")
    if any(key not in package for key in required):
        raise ValueError("Incomplete trained quality-gate package")
    if not features or any(row["official_split"] != split for row in features.values()):
        raise ValueError("Feature file split mismatch")
    ids = list(features)
    result = {}
    for task in ("content", "parent"):
        names = package[f"{task}_features"]
        x = np.asarray([[float(features[key][name]) for name in names] for key in ids], dtype=float)
        if not np.isfinite(x).all():
            raise ValueError(f"Nonfinite {task} features")
        p = package[f"{task}_model"].predict_proba(x)[:, 1]
        if task == "content":
            result = {key: [float(value), 0.0] for key, value in zip(ids, p)}
        else:
            for key, value in zip(ids, p):
                result[key][1] = float(value)
    return {key: tuple(value) for key, value in result.items()}


def filter_graph(graph: dict, probabilities: dict[str, tuple[float, float]],
                 content_threshold: float, parent_threshold: float) -> tuple[dict, Counter]:
    nodes = graph["graph_nodes"]
    generated = graph["generated_nodes"]
    if [int(node["node_index"]) for node in nodes] != list(range(len(nodes))):
        raise ValueError(f"Noncontiguous input graph: {graph['news_id']}")
    first_generated = next((i for i, node in enumerate(nodes)
                            if node["node_type"] == "generated_comment"), len(nodes))
    if any(node["node_type"] == "generated_comment" for node in nodes[:first_generated]) or any(
            node["node_type"] != "generated_comment" for node in nodes[first_generated:]):
        raise ValueError(f"Generated nodes must follow all observed nodes: {graph['news_id']}")
    if len(nodes) - first_generated != len(generated):
        raise ValueError(f"Generated detail count mismatch: {graph['news_id']}")
    if {int(item["node_index"]) for item in generated} != set(range(first_generated, len(nodes))):
        raise ValueError(f"Generated detail indices mismatch: {graph['news_id']}")
    new_nodes = deepcopy(nodes[:first_generated])
    new_edges = [deepcopy(edge) for edge in graph["edges"] if int(edge["edge_type"]) == 0]
    new_generated = []
    remap = {i: i for i in range(first_generated)}
    reasons: Counter[str] = Counter()
    for item in generated:
        old_child = int(item["node_index"])
        old_parent = int(item["parent_node_index"])
        uid = str(item["generated_node_id"])
        if old_child < first_generated or not (0 <= old_parent < old_child):
            raise ValueError(f"Invalid generated parent: {graph['news_id']}/{uid}")
        if uid not in probabilities:
            raise ValueError(f"Missing gate features: {graph['news_id']}/{uid}")
        content_p, parent_p = probabilities[uid]
        if not (0 <= content_p <= 1 and 0 <= parent_p <= 1):
            raise ValueError(f"Invalid gate probabilities: {graph['news_id']}/{uid}")
        if old_parent not in remap:
            reasons["ancestor_rejected"] += 1
            continue
        if content_p < content_threshold:
            reasons["content_rejected"] += 1
            continue
        if parent_p < parent_threshold:
            reasons["parent_rejected"] += 1
            continue
        new_child = len(new_nodes)
        new_parent = remap[old_parent]
        new_node = deepcopy(nodes[old_child])
        new_node["node_index"] = new_child
        new_item = deepcopy(item)
        new_item["original_node_index"] = old_child
        new_item["node_index"] = new_child
        new_item["parent_node_index"] = new_parent
        new_item["quality_gate_content_probability"] = content_p
        new_item["quality_gate_parent_probability"] = parent_p
        new_nodes.append(new_node)
        new_generated.append(new_item)
        new_edges.append({"source": new_parent, "target": new_child, "edge_type": 1})
        remap[old_child] = new_child
        reasons["accepted"] += 1
    output = deepcopy(graph)
    output["graph_nodes"] = new_nodes
    output["edges"] = new_edges
    output["generated_nodes"] = new_generated
    output["generation_config"] = deepcopy(graph["generation_config"])
    output["generation_config"]["posthoc_quality_gate"] = {
        "content_threshold": content_threshold,
        "parent_threshold": parent_threshold,
        "policy": "reject_unsafe_node_and_all_generated_descendants_no_rewiring",
    }
    return output, reasons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs-jsonl", type=Path, required=True)
    parser.add_argument("--features-jsonl", type=Path, required=True)
    parser.add_argument("--features-metadata-json", type=Path, required=True)
    parser.add_argument("--gate-model", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "validate", "test"], required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_jsonl.exists() or args.report_json.exists():
        raise FileExistsError("Refusing to overwrite an existing gated graph/report")
    graphs = read_unique(args.graphs_jsonl, "news_id")
    features = read_unique(args.features_jsonl, "annotation_id")
    if any(graph["official_split"] != args.split for graph in graphs.values()):
        raise ValueError("Graph split mismatch")
    # Only load locally trained, trusted joblib files. Pickle is not a safe
    # interchange format for untrusted downloads.
    package = joblib.load(args.gate_model)
    feature_meta = json.loads(args.features_metadata_json.read_text(encoding="utf-8"))
    source_models = {name: feature_meta[name] for name in
                     ("reranker_model", "reranker_snapshot", "clip_model", "clip_snapshot",
                      "candidate_max_length")}
    graph_hash = hashlib.sha256(args.graphs_jsonl.read_bytes()).hexdigest()
    if (feature_meta["split"] != args.split or feature_meta["rows"] != len(features) or
            package.get("source_models") != source_models or
            feature_meta.get("candidate_source_graph_sha256") != graph_hash):
        raise ValueError("Feature split, count, graph or frozen-model snapshots differ from gate training")
    feature_hash = hashlib.sha256(args.features_jsonl.read_bytes()).hexdigest()
    if args.split == "train" and package.get("feature_sha256") != feature_hash:
        raise ValueError("Training feature file differs from the gate fit source")
    if args.split == "train" and package.get("training_graph_sha256") != graph_hash:
        raise ValueError("Training graph file differs from the gate fit source")
    content_threshold = float(package["content_threshold"])
    parent_threshold = float(package["parent_threshold"])
    probabilities = infer_probs(package, features, args.split)
    wanted = {str(item["generated_node_id"])
              for graph in graphs.values() for item in graph["generated_nodes"]}
    if wanted != set(probabilities):
        raise ValueError(f"Graph/features generated-ID coverage mismatch: missing={len(wanted-set(probabilities))}, "
                         f"extra={len(set(probabilities)-wanted)}")
    output = []
    totals: Counter[str] = Counter()
    zero_generated_news = 0
    for graph in graphs.values():
        filtered, reasons = filter_graph(graph, probabilities, content_threshold, parent_threshold)
        output.append(filtered)
        totals.update(reasons)
        zero_generated_news += len(filtered["generated_nodes"]) == 0
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    validator = PROJECT_ROOT / "comment_generator" / "partial_observation" / "scripts" / "validate_generated_structured_graphs.py"
    report = {"split": args.split, "news": len(output), "generated_before": len(wanted),
              "generated_after": totals["accepted"], "news_with_zero_generated": zero_generated_news,
              "pruning_reasons": dict(totals),
              "source_graph_sha256": graph_hash,
              "feature_sha256": feature_hash,
              "source_models": source_models,
              "gate_model_sha256": hashlib.sha256(args.gate_model.read_bytes()).hexdigest(),
              "news_labels_used": False,
              "caveat": "Learned acceptance of synthetic edges; not ground-truth reply recovery."}
    # Validate the complete graph file before publishing either output. A
    # failed validator must not leave behind a plausible-looking result.
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n",
                                     dir=args.output_jsonl.parent,
                                     prefix=args.output_jsonl.stem + ".",
                                     suffix=".jsonl", delete=False) as handle:
        temporary_graph = Path(handle.name)
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    try:
        subprocess.run([sys.executable, str(validator), "--input-jsonl", str(temporary_graph),
                        "--expected-split", args.split], check=True)
        # Report creation is also staged, then published only after the graph
        # validator succeeds. The two replaces are best-effort, not a database
        # transaction; a write failure is still surfaced to the caller.
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n",
                                         dir=args.report_json.parent,
                                         prefix=args.report_json.stem + ".",
                                         suffix=".json", delete=False) as handle:
            temporary_report = Path(handle.name)
            json.dump(report, handle, ensure_ascii=False, indent=2)
        try:
            temporary_graph.replace(args.output_jsonl)
            temporary_report.replace(args.report_json)
        finally:
            temporary_report.unlink(missing_ok=True)
    finally:
        temporary_graph.unlink(missing_ok=True)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
