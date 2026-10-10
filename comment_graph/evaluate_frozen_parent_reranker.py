"""Label-blind frozen reranker audit on observed Fakeddit reply edges.

This is a parent-retrieval diagnostic, not a generated-placement ground truth
or a classification test. Candidate order follows parent-first traversal,
because this Fakeddit comment file has no timestamp.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from comment_graph.common import generated_coverage_ids, load_config, make_dataset  # noqa: E402
from comment_graph.train_reply_parent_ranker import encode_graphs  # noqa: E402


def evaluate(config_path: Path, model_name: str, batch_size: int,
             max_length: int, limit_news: int | None, split: str,
             sample_seed: int, include_edge_details: bool) -> dict:
    config, config_path = load_config(config_path)
    if str(config.data.dataset_name) != "fakeddit_structure":
        raise ValueError("Observed parent evaluation requires Fakeddit real trees")
    allowed = generated_coverage_ids(config, config_path, split)
    dataset = make_dataset(config, config_path, split, "real_only", None, None,
                           allowed_news_ids=allowed)
    if limit_news is not None:
        dataset.records = random.Random(sample_seed).sample(
            dataset.records, min(limit_news, len(dataset.records)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # The same frozen BERT embeddings as the earlier learned-parent audit give
    # a paired cosine baseline for these exact news and observed edges.
    encoded = encode_graphs(config, config_path, split, device)
    embedding_by_news = {graph["news_id"]: graph for graph in encoded}
    snapshot = snapshot_download(repo_id=model_name, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    reranker = AutoModelForSequenceClassification.from_pretrained(
        snapshot, local_files_only=True).to(device).eval()
    counts = {"edges": 0, "root_edges": 0, "nonroot_edges": 0,
              "reranker_correct": 0, "reranker_root_correct": 0,
              "reranker_nonroot_correct": 0, "cosine_correct": 0,
              "cosine_root_correct": 0, "cosine_nonroot_correct": 0}
    reciprocal = {"reranker": 0.0, "cosine": 0.0}
    per_news, per_edge = [], []
    for record in dataset.records:
        news_id = str(record["news_id"])
        nodes, edges = dataset._real_nodes_and_edges(news_id)
        texts = [str(record["news_text"]), *(str(node["body"]) for node in nodes)]
        parent_by_child = {int(child): int(parent) for parent, child, relation in edges
                           if relation == 0}
        if set(parent_by_child) != set(range(1, len(texts))):
            raise ValueError(f"Incomplete observed parent mapping: {news_id}")
        features = F.normalize(embedding_by_news[news_id]["features"].float(), dim=-1)
        if len(features) != len(texts):
            raise ValueError(f"BERT/text node mismatch: {news_id}")
        graph_counts = {"edges": 0, "reranker_correct": 0, "cosine_correct": 0}
        for child in range(1, len(texts)):
            true_parent = parent_by_child[child]
            if not 0 <= true_parent < child:
                raise ValueError(f"Parent must precede child: {news_id}/{child}")
            pairs = [(texts[candidate], texts[child]) for candidate in range(child)]
            values = []
            for start in range(0, len(pairs), batch_size):
                batch = tokenizer(pairs[start:start + batch_size], padding=True,
                                  truncation=True, max_length=max_length,
                                  return_tensors="pt").to(device)
                with torch.inference_mode():
                    values.extend(reranker(**batch).logits.float().view(-1).cpu().tolist())
            scores = torch.tensor(values)
            cosine = features[:child] @ features[child]
            reranker_pred = int(scores.argmax())
            cosine_pred = int(cosine.argmax())
            if include_edge_details:
                ranked = sorted(map(float, scores), reverse=True)
                per_edge.append({
                    "news_id": news_id, "child_node_index": child,
                    "observed_parent_node_index": true_parent,
                    "reranker_top_node_index": reranker_pred,
                    "reranker_top_correct": reranker_pred == true_parent,
                    "reranker_top_score": ranked[0],
                    "reranker_top_margin": ranked[0] - ranked[1] if len(ranked) > 1 else 0.0,
                    "candidate_count": child,
                })
            is_root = true_parent == 0
            counts["edges"] += 1
            counts["root_edges" if is_root else "nonroot_edges"] += 1
            for name, predicted, ranked in (("reranker", reranker_pred, scores),
                                            ("cosine", cosine_pred, cosine)):
                correct = int(predicted == true_parent)
                counts[f"{name}_correct"] += correct
                counts[f"{name}_{'root' if is_root else 'nonroot'}_correct"] += correct
                reciprocal[name] += 1 / (1 + int((ranked > ranked[true_parent]).sum()))
                graph_counts[f"{name}_correct"] += correct
            graph_counts["edges"] += 1
        per_news.append({"news_id": news_id, **graph_counts})
        print(f"news={len(per_news)}/{len(dataset.records)} edges={counts['edges']}", flush=True)
    total = counts["edges"]
    if not total:
        raise ValueError("No observed reply edges")
    return {
        "scope": f"official Fakeddit {split}, observed real replies only",
        "split": split, "sample_seed": sample_seed if limit_news is not None else None,
        "news": len(per_news), "edges": total,
        "root_edges": counts["root_edges"], "nonroot_edges": counts["nonroot_edges"],
        "reranker_model": model_name, "reranker_snapshot": str(snapshot),
        "reranker_top1": counts["reranker_correct"] / total,
        "cosine_top1": counts["cosine_correct"] / total,
        "reranker_root_recall": counts["reranker_root_correct"] / max(counts["root_edges"], 1),
        "reranker_nonroot_recall": counts["reranker_nonroot_correct"] / max(counts["nonroot_edges"], 1),
        "cosine_root_recall": counts["cosine_root_correct"] / max(counts["root_edges"], 1),
        "cosine_nonroot_recall": counts["cosine_nonroot_correct"] / max(counts["nonroot_edges"], 1),
        "reranker_mrr": reciprocal["reranker"] / total,
        "cosine_mrr": reciprocal["cosine"] / total,
        "counts": counts, "per_news": per_news,
        "per_edge": per_edge if include_edge_details else None,
        "news_veracity_labels_used": False,
        "caveat": "No timestamps: candidate availability is parent-first traversal, not chronological exposure. Validation has already been used in earlier experiments; diagnostic only."
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name(
        "config_fakeddit_structure_expanded_generateddepth.yaml"))
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--split", choices=("train", "validate"), default="validate")
    parser.add_argument("--limit-news", type=int)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--include-edge-details", action="store_true")
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_length < 8 or (args.limit_news is not None and args.limit_news < 1):
        raise ValueError("Invalid batch/length/news limit")
    if args.output_json.exists():
        raise FileExistsError(args.output_json)
    result = evaluate(args.config, args.reranker_model, args.batch_size,
                      args.max_length, args.limit_news, args.split,
                      args.sample_seed, args.include_edge_details)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in {"counts", "per_news", "per_edge"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
