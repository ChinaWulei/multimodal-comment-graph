"""Probe frozen text reranking for news-comment relevance, without labels.

Observed in-thread comments are weak positives; same-split cross-news swaps
are weak negatives. This is a diagnostic, not verified comment-quality truth.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from sklearn.metrics import average_precision_score, roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="BAAI/bge-reranker-v2-m3")
    args = parser.parse_args()
    graphs = read_jsonl(args.graphs_jsonl)
    if len(graphs) != len({str(graph["news_id"]) for graph in graphs}):
        raise ValueError("Duplicate news IDs")
    headlines = {str(graph["news_id"]): str(graph["graph_nodes"][0]["content"])
                 for graph in graphs}
    ids = sorted(headlines)
    if len(ids) < 2:
        raise ValueError("Need at least two news for swap negatives")
    rng = random.Random(args.seed)
    cases = []
    for graph in graphs:
        news_id = str(graph["news_id"])
        real_parents = {int(edge["target"]): int(edge["source"])
                        for edge in graph["edges"] if int(edge["edge_type"]) == 0}
        real_nodes = [node for node in graph["graph_nodes"] if node["node_type"] == "real_comment"]
        generated_nodes = [node for node in graph["graph_nodes"] if node["node_type"] == "generated_comment"]
        for node in real_nodes:
            comment = str(node["content"])
            node_index = int(node["node_index"])
            parent_index = real_parents[node_index]
            donor_id = rng.choice([other for other in ids if other != news_id])
            cases.append({"news_id": news_id, "kind": "real_positive", "comment": comment,
                          "headline": headlines[news_id], "node_index": node_index,
                          "real_parent_index": parent_index})
            cases.append({"news_id": news_id, "kind": "cross_news_negative", "comment": comment,
                          "headline": headlines[donor_id], "node_index": node_index,
                          "real_parent_index": parent_index})
        for node in generated_nodes:
            cases.append({"news_id": news_id, "kind": "generated_unlabeled", "comment": str(node["content"]),
                          "headline": headlines[news_id]})
    snapshot = snapshot_download(repo_id=args.model, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(
        snapshot, local_files_only=True).to(device).eval()
    for start in range(0, len(cases), 8):
        subset = cases[start:start + 8]
        pairs = [(f"News: {case['headline']}", case["comment"]) for case in subset]
        tokens = tokenizer(pairs, padding=True, truncation=True,
                           max_length=256, return_tensors="pt").to(device)
        with torch.inference_mode():
            scores = model(**tokens).logits.float().view(-1).cpu().tolist()
        for case, score in zip(subset, scores):
            case["reranker_score"] = score
    labeled = [case for case in cases if case["kind"] != "generated_unlabeled"]
    y = [int(case["kind"] == "real_positive") for case in labeled]
    scores = [case["reranker_score"] for case in labeled]
    positive = [case["reranker_score"] for case in cases if case["kind"] == "real_positive"]
    negative = [case["reranker_score"] for case in cases if case["kind"] == "cross_news_negative"]
    generated = [case["reranker_score"] for case in cases if case["kind"] == "generated_unlabeled"]
    strata = {}
    for name, is_top_level in (("top_level", True), ("reply", False)):
        subset = [case for case in labeled if (case["real_parent_index"] == 0) == is_top_level]
        weak_labels = [int(case["kind"] == "real_positive") for case in subset]
        strata[name] = {"paired_cases": len(subset) // 2,
                        "roc_auc": roc_auc_score(weak_labels,
                                                 [case["reranker_score"] for case in subset])}
    result = {
        "model": args.model, "model_location": snapshot,
        "split": graphs[0]["official_split"], "news_count": len(graphs),
        "real_positive_count": len(positive), "cross_news_negative_count": len(negative),
        "generated_unlabeled_count": len(generated),
        "weak_label_roc_auc": roc_auc_score(y, scores),
        "weak_label_average_precision": average_precision_score(y, scores),
        "by_observed_position": strata,
        "mean_real_positive_score": sum(positive) / len(positive),
        "mean_cross_news_negative_score": sum(negative) / len(negative),
        "mean_generated_unlabeled_score": sum(generated) / len(generated),
        "caveat": "Real in-thread and cross-news swapped are weak labels; scores do not verify individual generated comment relevance or image use.",
        "cases": cases,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "cases"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
