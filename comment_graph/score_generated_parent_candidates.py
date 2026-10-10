"""Audit selected generated parents against the exact legal alternatives.

Uses a frozen local text reranker. Scores are proxies, not human judgments or
ground-truth generated-reply edges. Nothing is rewired or filtered here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def parent_depth(index: int, parents: dict[int, int], boundary: set[int] | None) -> int:
    depth, seen = 0, set()
    while index:
        if boundary is not None and index in boundary:
            break
        if index in seen or index not in parents:
            raise ValueError("Cyclic or incomplete parent map")
        seen.add(index)
        depth += 1
        index = parents[index]
    return depth


def contexts(path: Path, split: str) -> list[tuple[dict, list[dict]]]:
    result = []
    news_ids, node_ids = set(), set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            graph = json.loads(line)
            news_id = str(graph["news_id"])
            if news_id in news_ids or graph["official_split"] != split:
                raise ValueError(f"Duplicate news or split mismatch: {news_id}")
            news_ids.add(news_id)
            nodes = graph["graph_nodes"]
            if [int(node["node_index"]) for node in nodes] != list(range(len(nodes))):
                raise ValueError(f"Non-contiguous graph nodes: {news_id}")
            parents = {int(edge["target"]): int(edge["source"]) for edge in graph["edges"]}
            if set(parents) != set(range(1, len(nodes))):
                raise ValueError(f"Incomplete parent edges: {news_id}")
            policy = graph["generation_config"]["depth_reference"]
            if policy not in {"news_root", "observed_boundary"}:
                raise ValueError(f"Unknown depth policy: {news_id}/{policy}")
            boundary = ({i for i, node in enumerate(nodes)
                         if node["node_type"] == "real_comment"}
                        if policy == "observed_boundary" else None)
            max_depth = int(graph["generation_config"]["max_generated_depth"])
            entries = []
            for detail in graph["generated_nodes"]:
                child = int(detail["node_index"])
                chosen = int(detail["parent_node_index"])
                uid = str(detail["generated_node_id"])
                if uid in node_ids or nodes[child]["node_type"] != "generated_comment":
                    raise ValueError(f"Duplicate or invalid generated node: {news_id}/{uid}")
                node_ids.add(uid)
                allowed = [0, *(i for i in range(1, child)
                                if parent_depth(i, parents, boundary) + 1 <= max_depth)]
                if chosen not in allowed or parents[child] != chosen:
                    raise ValueError(f"Recorded parent outside generation candidates: {news_id}/{uid}")
                entries.append({"annotation_id": uid, "news_id": news_id,
                                "official_split": split, "child": child,
                                "chosen": chosen, "candidates": allowed,
                                "texts": [str(node["content"]) for node in nodes]})
            result.append((graph, entries))
    if not result or not node_ids:
        raise ValueError("No generated graphs/nodes")
    return result


def score(groups: list[tuple[dict, list[dict]]], snapshot: str,
          batch_size: int, max_length: int) -> list[dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        snapshot, local_files_only=True).to(device).eval()
    result = []
    for graph_number, (graph, entries) in enumerate(groups, 1):
        for entry in entries:
            pairs = [(entry["texts"][candidate], entry["texts"][entry["child"]])
                     for candidate in entry["candidates"]]
            values = []
            for start in range(0, len(pairs), batch_size):
                tokens = tokenizer(pairs[start:start + batch_size], padding=True,
                                   truncation=True, max_length=max_length,
                                   return_tensors="pt").to(device)
                with torch.inference_mode():
                    values.extend(model(**tokens).logits.float().view(-1).cpu().tolist())
            if len(values) != len(entry["candidates"]):
                raise RuntimeError("Candidate score count mismatch")
            chosen_position = entry["candidates"].index(entry["chosen"])
            chosen_score = values[chosen_position]
            alternatives = [value for i, value in enumerate(values) if i != chosen_position]
            best_alternative = max(alternatives) if alternatives else chosen_score
            rank = sum(value > chosen_score for value in values)
            best_position = max(range(len(values)), key=values.__getitem__)
            softmax_probability = float(torch.softmax(torch.tensor(values), dim=0)[chosen_position])
            result.append({
                "annotation_id": entry["annotation_id"], "news_id": entry["news_id"],
                "official_split": entry["official_split"],
                "chosen_parent_node_index": entry["chosen"],
                "candidate_count": len(values),
                "reranker_chosen_parent_score": chosen_score,
                "reranker_best_alternative_score": best_alternative,
                "reranker_chosen_vs_best_alternative_margin": chosen_score - best_alternative,
                "reranker_chosen_parent_rank": rank,
                "reranker_chosen_parent_rank_fraction": rank / max(len(values) - 1, 1),
                "reranker_chosen_parent_softmax_probability": softmax_probability,
                "reranker_best_parent_node_index": entry["candidates"][best_position],
                "reranker_best_parent_node_type": graph["graph_nodes"][entry["candidates"][best_position]]["node_type"],
                "reranker_agrees_with_generated_parent": best_position == chosen_position,
            })
        if graph_number % 25 == 0 or graph_number == len(groups):
            print(f"graphs={graph_number}/{len(groups)} scored_generated={len(result)}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs-jsonl", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validate", "test"), required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--metadata-json", type=Path, required=True)
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_length < 8:
        raise ValueError("Invalid batch size or max length")
    if args.output_jsonl.exists() or args.metadata_json.exists():
        raise FileExistsError("Refusing to overwrite candidate scores")
    groups = contexts(args.graphs_jsonl, args.split)
    snapshot = snapshot_download(repo_id=args.reranker_model, local_files_only=True)
    result = score(groups, snapshot, args.batch_size, args.max_length)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_json.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n",
                                     dir=args.output_jsonl.parent,
                                     prefix=args.output_jsonl.stem + ".", suffix=".tmp",
                                     delete=False) as handle:
        temporary = Path(handle.name)
        for row in result:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    metadata = {
        "rows": len(result), "news": len(groups), "split": args.split,
        "source_graph_sha256": hashlib.sha256(args.graphs_jsonl.read_bytes()).hexdigest(),
        "reranker_model": args.reranker_model, "reranker_snapshot": str(snapshot),
        "batch_size": args.batch_size, "max_length": args.max_length,
        "news_veracity_labels_used": False,
        "warning": "Comparative proxy only; no generated-parent ground truth and no automatic edge replacement.",
    }
    try:
        temporary.replace(args.output_jsonl)
        args.metadata_json.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
