"""Probe an already-cached reranker against the cosine-placement pilot.

This only compares candidate parent relevance. It does not claim that a
retrieval reranker is trained to recover Reddit reply relationships.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from comment_graph.common import load_config, make_dataset, make_tokenizer_and_transform


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pilot-summary", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--max-edges", type=int, default=32)
    parser.add_argument("--model", default="BAAI/bge-reranker-v2-m3")
    args = parser.parse_args()
    config, config_path = load_config(args.config)
    tokenizer, transform = make_tokenizer_and_transform(config)
    dataset = make_dataset(config, config_path, "validate", "real_generated", tokenizer, transform)
    manifest = {row["news_id"]: row for row in dataset.records}
    summary = json.loads(args.pilot_summary.read_text(encoding="utf-8"))
    cases = []
    for prediction in summary["validation"]["prediction_records"]:
        news_id = prediction["news_id"]
        comments, _, _ = dataset._nodes_and_edges(news_id)
        contents = [manifest[news_id]["news_text"], *comments]
        for edge in prediction.get("generated_reliability", []):
            if not edge["rewired"]:
                continue
            cases.append({
                "news_id": news_id,
                "news_text": contents[0],
                "generated": contents[edge["node_index"]],
                "original_parent": contents[edge["original_parent"]],
                "cosine_parent": contents[edge["selected_parent"]],
                "original_parent_similarity": edge["original_parent_similarity"],
                "cosine_parent_similarity": edge["selected_parent_similarity"],
            })
            if len(cases) >= args.max_edges:
                break
        if len(cases) >= args.max_edges:
            break
    # snapshot_download is local-only: no network or model download.
    snapshot = snapshot_download(repo_id=args.model, local_files_only=True)
    reranker_tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reranker = AutoModelForSequenceClassification.from_pretrained(
        snapshot, local_files_only=True,
    ).to(device).eval()
    for start in range(0, len(cases), 8):
        subset = cases[start:start + 8]
        pairs = []
        for case in subset:
            pairs.extend(((case["generated"], case["original_parent"]),
                          (case["generated"], case["cosine_parent"]),
                          (case["generated"], f"News: {case['news_text']} Existing comment: {case['original_parent']}"),
                          (case["generated"], f"News: {case['news_text']} Existing comment: {case['cosine_parent']}")))
        tokens = reranker_tokenizer(pairs, padding=True, truncation=True,
                                    max_length=256, return_tensors="pt").to(device)
        with torch.inference_mode():
            scores = reranker(**tokens).logits.float().view(-1).cpu().tolist()
        for i, case in enumerate(subset):
            case["reranker_original_score"] = scores[4 * i]
            case["reranker_cosine_score"] = scores[4 * i + 1]
            case["reranker_prefers_original"] = scores[4 * i] > scores[4 * i + 1]
            case["news_context_original_score"] = scores[4 * i + 2]
            case["news_context_cosine_score"] = scores[4 * i + 3]
            case["news_context_prefers_original"] = scores[4 * i + 2] > scores[4 * i + 3]
    result = {
        "model": args.model, "model_location": snapshot,
        "split": "validate", "n_rewired_cases": len(cases),
        "reranker_prefers_original": sum(case["reranker_prefers_original"] for case in cases),
        "news_context_prefers_original": sum(case["news_context_prefers_original"] for case in cases),
        "warning": "Scores are relevance proxies, not verified reply-parent labels.",
        "cases": cases,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "cases"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
