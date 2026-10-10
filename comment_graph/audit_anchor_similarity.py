"""Measure train-only news anchoring scores before choosing a graph gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from comment_graph.common import load_config
from comment_graph.train_news_comment_relevance import encode_texts, read_graphs


def percentiles(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    result = {"count": len(ordered), "mean": sum(ordered) / len(ordered)}
    for q in (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95):
        result[f"p{int(q * 100):02d}"] = ordered[round(q * (len(ordered) - 1))]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--graphs-jsonl", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "validate"], required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    graphs = read_graphs(args.graphs_jsonl, args.split)
    config, _ = load_config(args.config)
    reference = str(config.encoder.text_model_name)
    tokenizer = AutoTokenizer.from_pretrained(reference, local_files_only=True, use_fast=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = AutoModel.from_pretrained(reference, local_files_only=True).to(device).eval()
    texts = sorted({str(node["content"]).strip() for graph in graphs for node in graph["graph_nodes"]})
    index = {text: idx for idx, text in enumerate(texts)}
    features = F.normalize(encode_texts(texts, tokenizer, encoder, device, 32), dim=-1)
    groups = {"top_level_real": [], "real_reply": [], "generated_chain_head": [],
              "generated_descendant": []}
    for graph in graphs:
        nodes = {int(node["node_index"]): node for node in graph["graph_nodes"]}
        parent = {int(edge["target"]): int(edge["source"]) for edge in graph["edges"]}
        news = features[index[str(nodes[0]["content"]).strip()]]
        for node_id, node in nodes.items():
            if node_id == 0:
                continue
            score = float(((news * features[index[str(node["content"]).strip()]]).sum() + 1) / 2)
            is_generated = node["node_type"] == "generated_comment"
            parent_generated = nodes[parent[node_id]]["node_type"] == "generated_comment"
            if is_generated:
                key = "generated_descendant" if parent_generated else "generated_chain_head"
            else:
                key = "top_level_real" if parent[node_id] == 0 else "real_reply"
            groups[key].append(score)
    result = {"split": args.split, "news_count": len(graphs),
              "score_definition": "(1 + cosine(frozen BERT mean-pooled news, comment)) / 2",
              "text_model": reference, "groups": {name: percentiles(values) for name, values in groups.items()},
              "caveat": "No verified generated-comment quality labels; train-only distributions are calibration references, not success metrics."}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
