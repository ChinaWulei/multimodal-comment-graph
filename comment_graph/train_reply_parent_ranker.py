"""Train a parent ranker on Fakeddit's observed real-comment reply edges.

No generated edge, veracity label, or test-split comment is used for fitting.
The frozen BERT node encoding is identical to the classifier's node encoding.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from comment_graph.common import generated_coverage_ids, load_config, make_dataset, seed_everything
from comment_graph.reply_parent_ranker import ReplyParentRanker


def encode_graphs(config, config_path: Path, split: str, device: torch.device):
    allowed = generated_coverage_ids(config, config_path, split)
    dataset = make_dataset(config, config_path, split, "real_only", None, None,
                           allowed_news_ids=allowed)
    tokenizer = AutoTokenizer.from_pretrained(str(config.encoder.text_model_name),
                                               local_files_only=True, use_fast=True)
    bert = AutoModel.from_pretrained(str(config.encoder.text_model_name),
                                     local_files_only=True).to(device).eval()
    for parameter in bert.parameters():
        parameter.requires_grad_(False)
    graphs = []
    with torch.inference_mode():
        for row in dataset.records:
            news_id = str(row["news_id"])
            nodes, edges = dataset._real_nodes_and_edges(news_id)
            texts = [str(row["news_text"]), *(str(node["body"]) for node in nodes)]
            pieces = []
            for start in range(0, len(texts), int(config.encoder.node_encode_batch_size)):
                tokens = tokenizer(texts[start:start + int(config.encoder.node_encode_batch_size)],
                                   padding="max_length", truncation=True,
                                   max_length=int(config.data.max_comment_length),
                                   return_tensors="pt").to(device)
                hidden = bert(**tokens).last_hidden_state
                mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
                pieces.append(pooled.cpu())
            features = torch.cat(pieces)
            parent_by_child = {int(child): int(parent) for parent, child, relation in edges if relation == 0}
            if set(parent_by_child) != set(range(1, len(texts))):
                raise ValueError(f"Missing or duplicate observed parent in {news_id}")
            if any(not 0 <= parent_by_child[i] < i for i in parent_by_child):
                raise ValueError(f"Non-ancestral observed parent in {news_id}")
            graphs.append({"news_id": news_id, "features": features,
                           "parents": torch.tensor([parent_by_child[i] for i in range(1, len(texts))])})
    del bert
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return graphs


def score_split(model, graphs, device):
    model.eval()
    correct = root_correct = nonroot_correct = 0
    root_total = nonroot_total = 0
    reciprocal_rank = 0.0
    total_loss = 0.0
    cosine_correct = cosine_nonroot_correct = 0
    with torch.inference_mode():
        for graph in graphs:
            features = graph["features"].to(device)
            parents = graph["parents"].to(device)
            scores = model.real_parent_scores(features)
            predicted = scores.argmax(-1)
            cosine = F.normalize(features.float(), dim=-1)
            cosine_scores = cosine[1:] @ cosine.T
            children = torch.arange(1, len(features), device=device)
            cosine_scores = cosine_scores.masked_fill(
                torch.arange(len(features), device=device)[None, :] >= children[:, None], -1e9)
            cosine_predicted = cosine_scores.argmax(-1)
            root = parents == 0
            nonroot = ~root
            correct += int((predicted == parents).sum())
            root_correct += int(((predicted == parents) & root).sum())
            nonroot_correct += int(((predicted == parents) & nonroot).sum())
            cosine_correct += int((cosine_predicted == parents).sum())
            cosine_nonroot_correct += int(((cosine_predicted == parents) & nonroot).sum())
            root_total += int(root.sum())
            nonroot_total += int(nonroot.sum())
            reciprocal_rank += float((1.0 / (1 + (scores > scores.gather(1, parents[:, None])).sum(1))).sum())
            total_loss += float(F.cross_entropy(scores, parents, reduction="sum"))
    total = root_total + nonroot_total
    root_recall = root_correct / max(root_total, 1)
    nonroot_recall = nonroot_correct / max(nonroot_total, 1)
    return {
        "news": len(graphs), "real_reply_edges": total,
        "root_edges": root_total, "nonroot_edges": nonroot_total,
        "top1": correct / total, "root_recall": root_recall,
        "nonroot_recall": nonroot_recall,
        "parent_macro_recall": (root_recall + nonroot_recall) / 2,
        "mrr": reciprocal_rank / total, "loss": total_loss / total,
        "always_root_top1": root_total / total,
        "cosine_top1": cosine_correct / total,
        "cosine_nonroot_recall": cosine_nonroot_correct / max(nonroot_total, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config_fakeddit_structure_expanded_generateddepth.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1 or args.learning_rate <= 0:
        raise ValueError("Invalid training schedule")
    seed_everything(args.seed)
    config, config_path = load_config(args.config)
    if str(config.data.dataset_name) != "fakeddit_structure":
        raise ValueError("Parent supervision requires Fakeddit's real reply trees")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_graphs = encode_graphs(config, config_path, "train", device)
    val_graphs = encode_graphs(config, config_path, "validate", device)
    if {graph["news_id"] for graph in train_graphs} & {graph["news_id"] for graph in val_graphs}:
        raise ValueError("Train/validation news ID leakage in parent supervision")
    model = ReplyParentRanker(input_dim=train_graphs[0]["features"].size(1)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "reply_parent_ranker.pt"
    history = []
    best, stale = -1.0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = list(range(len(train_graphs)))
        random.Random(args.seed + epoch).shuffle(order)
        loss_sum, edges = 0.0, 0
        for index in order:
            graph = train_graphs[index]
            features = graph["features"].to(device)
            parents = graph["parents"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model.real_parent_scores(features)
            loss = F.cross_entropy(logits, parents)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(parents)
            edges += len(parents)
        validation = score_split(model, val_graphs, device)
        entry = {"epoch": epoch, "train_loss": loss_sum / edges, "validation": validation}
        history.append(entry)
        print(f"epoch={epoch} train_loss={entry['train_loss']:.5f} "
              f"val_top1={validation['top1']:.4f} val_nonroot={validation['nonroot_recall']:.4f} "
              f"val_parent_macro_recall={validation['parent_macro_recall']:.4f}", flush=True)
        if validation["parent_macro_recall"] > best:
            best, stale = validation["parent_macro_recall"], 0
            torch.save({"model": model.state_dict(), "seed": args.seed,
                        "input_dim": train_graphs[0]["features"].size(1),
                        "config": {"hidden_dim": 128, "dropout": 0.1},
                        "selected_epoch": epoch, "validation": validation,
                        "text_model": str(config.encoder.text_model_name)}, checkpoint_path)
        else:
            stale += 1
            if stale >= args.patience:
                break
    selected = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(selected["model"])
    report = {"seed": args.seed, "train_news": len(train_graphs),
              "validate_news": len(val_graphs),
              "train_real_reply_edges": sum(len(graph["parents"]) for graph in train_graphs),
              "selected_epoch": selected["selected_epoch"],
              "validation": selected["validation"],
              "checkpoint": str(checkpoint_path.resolve()), "history": history,
              "test_used": False, "news_labels_used": False,
              "generated_edges_used_for_training": False}
    report_path = args.output_dir / "summary.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "history"}, indent=2))


if __name__ == "__main__":
    main()
