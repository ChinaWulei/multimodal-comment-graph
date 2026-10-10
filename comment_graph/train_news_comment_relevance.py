"""Train a weakly supervised, label-free relevance head on real discussion.

Positive: an observed real comment paired with its own news headline.
Negatives: the same comment paired with one random and one nearest-headline
other news from the same official split. No veracity labels or test split are
read. The validation AUC measures only these weak pairs, not human relevance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from transformers import AutoModel, AutoTokenizer

from comment_graph.common import load_config
from comment_graph.news_comment_relevance import NewsCommentRelevance


def read_graphs(path: Path, split: str) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [str(row["news_id"]) for row in rows]
    if not rows or len(ids) != len(set(ids)) or any(row["official_split"] != split for row in rows):
        raise ValueError(f"Invalid {split} generated-graph source")
    return rows


def encode_texts(texts: list[str], tokenizer, encoder, device, batch_size: int) -> torch.Tensor:
    outputs = []
    encoder.eval()
    for start in range(0, len(texts), batch_size):
        tokens = tokenizer(texts[start:start + batch_size], padding="max_length",
                           truncation=True, max_length=96, return_tensors="pt").to(device)
        with torch.inference_mode():
            hidden = encoder(**tokens).last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        outputs.append(pooled.float().cpu())
    return torch.cat(outputs, dim=0)


def build_examples(graphs: list[dict], text_index: dict[str, int], embeddings: torch.Tensor,
                   seed: int) -> list[tuple[int, int, int, str]]:
    rng = random.Random(seed)
    news_ids = [str(row["news_id"]) for row in graphs]
    news_texts = [str(row["graph_nodes"][0]["content"]).strip() for row in graphs]
    news_features = F.normalize(embeddings[[text_index[text] for text in news_texts]], dim=-1)
    similarity = news_features @ news_features.T
    similarity.fill_diagonal_(-float("inf"))
    hard_other = similarity.argmax(dim=1).tolist()
    result = []
    for news_index, row in enumerate(graphs):
        news_embedding_index = text_index[news_texts[news_index]]
        comments = [str(node["content"]).strip() for node in row["graph_nodes"]
                    if node["node_type"] == "real_comment"]
        for comment in comments:
            if not comment:
                continue
            comment_index = text_index[comment]
            random_index = rng.choice([index for index in range(len(news_ids)) if index != news_index])
            result.extend((
                (news_embedding_index, comment_index, 1, "positive"),
                (text_index[news_texts[random_index]], comment_index, 0, "random_negative"),
                (text_index[news_texts[hard_other[news_index]]], comment_index, 0, "hard_negative"),
            ))
    return result


def score_examples(model, embeddings: torch.Tensor, rows, device):
    model.eval()
    scores = []
    with torch.inference_mode():
        for start in range(0, len(rows), 256):
            batch = rows[start:start + 256]
            news = embeddings[[row[0] for row in batch]].to(device)
            comments = embeddings[[row[1] for row in batch]].to(device)
            scores.extend(model(news, comments).float().cpu().tolist())
    return scores


def metrics_for(rows, scores) -> dict:
    labels = [row[2] for row in rows]
    result = {"roc_auc": roc_auc_score(labels, scores),
              "average_precision": average_precision_score(labels, scores)}
    positives = [score for row, score in zip(rows, scores) if row[3] == "positive"]
    for kind in ("random_negative", "hard_negative"):
        negatives = [score for row, score in zip(rows, scores) if row[3] == kind]
        result[f"{kind}_roc_auc"] = roc_auc_score(
            [1] * len(positives) + [0] * len(negatives), positives + negatives)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-graphs", type=Path, required=True)
    parser.add_argument("--validate-graphs", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        raise ValueError("Invalid training hyperparameters")
    train = read_graphs(args.train_graphs, "train")
    validate = read_graphs(args.validate_graphs, "validate")
    if {str(row["news_id"]) for row in train} & {str(row["news_id"]) for row in validate}:
        raise ValueError("News leakage between official train/validation splits")
    config, _ = load_config(args.config)
    model_ref = str(config.encoder.text_model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_ref, local_files_only=True, use_fast=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = AutoModel.from_pretrained(model_ref, local_files_only=True).to(device).eval()
    texts = sorted({str(node["content"]).strip() for row in [*train, *validate]
                    for node in row["graph_nodes"] if node["node_type"] in {"news", "real_comment"}})
    if not texts or any(not text for text in texts):
        raise ValueError("Empty news or real comment in scorer source")
    index = {value: i for i, value in enumerate(texts)}
    embeddings = encode_texts(texts, tokenizer, encoder, device, 32)
    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    train_rows = build_examples(train, index, embeddings, args.seed)
    val_rows = build_examples(validate, index, embeddings, args.seed)
    baseline = metrics_for(val_rows, [float(F.cosine_similarity(
        embeddings[row[0]][None, :], embeddings[row[1]][None, :]).item()) for row in val_rows])
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    model = NewsCommentRelevance(input_dim=embeddings.size(1)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    best_auc, best_epoch, stale = -1.0, 0, 0
    history = []
    rng = random.Random(args.seed)
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = list(range(len(train_rows)))
        rng.shuffle(order)
        losses = []
        for start in range(0, len(order), args.batch_size):
            batch = [train_rows[index] for index in order[start:start + args.batch_size]]
            news = embeddings[[row[0] for row in batch]].to(device)
            comments = embeddings[[row[1] for row in batch]].to(device)
            labels = torch.tensor([row[2] for row in batch], dtype=torch.float32, device=device)
            logits = model(news, comments)
            loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=logits.new_tensor(2.0))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        val_metrics = metrics_for(val_rows, score_examples(model, embeddings, val_rows, device))
        history.append({"epoch": epoch, "train_loss": sum(losses) / len(losses), **val_metrics})
        print(f"epoch={epoch} train_loss={history[-1]['train_loss']:.4f} "
              f"val_auc={val_metrics['roc_auc']:.4f} hard_auc={val_metrics['hard_negative_roc_auc']:.4f}", flush=True)
        if val_metrics["roc_auc"] > best_auc + 1e-5:
            best_auc, best_epoch, stale = val_metrics["roc_auc"], epoch, 0
            args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model": model.state_dict(), "input_dim": embeddings.size(1), "hidden_dim": 128,
                "seed": args.seed, "best_epoch": epoch, "validation_weak_auc": best_auc,
                "text_model_name": str(config.encoder.text_model_name),
                "train_graph_sha256": hashlib.sha256(args.train_graphs.read_bytes()).hexdigest(),
                "validation_graph_sha256": hashlib.sha256(args.validate_graphs.read_bytes()).hexdigest(),
                "supervision": "own-news real comment vs random/hard other-news headline; no veracity label",
            }, args.output_checkpoint)
        else:
            stale += 1
            if stale >= 5:
                break
    report = {
        "train_news": len(train), "validation_news": len(validate),
        "train_examples": len(train_rows), "validation_examples": len(val_rows),
        "baseline_cosine_validation": baseline,
        "best_validation_weak_auc": best_auc, "best_epoch": best_epoch,
        "history": history, "checkpoint": str(args.output_checkpoint.resolve()),
        "warning": "Weak synthetic pairing benchmark only; not a verified generated-comment quality label or attachment metric.",
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "history"}, indent=2))


if __name__ == "__main__":
    main()
