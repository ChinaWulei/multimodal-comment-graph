"""Precompute label-blind text/parent/image proxy scores for review cases.

These scores are model features, never ground-truth comment-quality labels.
Models must already exist in the local Hugging Face cache; no download occurs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from PIL import Image, ImageOps
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer, CLIPModel, CLIPProcessor,
)


def read_queue(path: Path, expected_split: str) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [str(row["annotation_id"]) for row in rows]
    if len(ids) != len(set(ids)) or any(row["official_split"] != expected_split for row in rows):
        raise ValueError("Feature contexts must have unique IDs and the expected official split")
    if any(row.get("content_relevance") is not None or row.get("parent_fit") is not None for row in rows):
        raise ValueError("Score only the unannotated reference queue; keep human labels separate")
    return rows


def score_reranker(rows: list[dict], model_name: str, device: torch.device, batch_size: int) -> tuple[list[dict], str]:
    snapshot = snapshot_download(repo_id=model_name, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(snapshot, local_files_only=True).to(device).eval()
    pairs = []
    for row in rows:
        comment = row["generated_comment"]
        pairs.extend([
            (f"News: {row['headline']}", comment),
            (row["parent_content"], comment),
            (f"News: {row['headline']} Parent: {row['parent_content']}", comment),
        ])
    scores: list[float] = []
    for start in range(0, len(pairs), batch_size):
        tokens = tokenizer(pairs[start:start + batch_size], padding=True, truncation=True,
                           max_length=256, return_tensors="pt").to(device)
        with torch.inference_mode():
            scores.extend(model(**tokens).logits.float().view(-1).cpu().tolist())
    if len(scores) != len(rows) * 3:
        raise RuntimeError("Reranker output length mismatch")
    features = [{"reranker_news_comment": scores[i * 3],
                 "reranker_parent_comment": scores[i * 3 + 1],
                 "reranker_news_parent_comment": scores[i * 3 + 2]}
                for i in range(len(rows))]
    del model, tokenizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return features, str(snapshot)


def score_clip(rows: list[dict], model_name: str, device: torch.device,
               batch_size: int) -> tuple[list[dict], str]:
    snapshot = snapshot_download(repo_id=model_name, local_files_only=True)
    processor = CLIPProcessor.from_pretrained(snapshot, local_files_only=True)
    model = CLIPModel.from_pretrained(snapshot, local_files_only=True).to(device).eval()
    image_paths = list(dict.fromkeys(str(row["image_path"]) for row in rows))
    image_indices = {path: i for i, path in enumerate(image_paths)}
    own_indices = torch.tensor([image_indices[str(row["image_path"])] for row in rows], dtype=torch.long)
    image_vectors = []
    for start in range(0, len(image_paths), batch_size):
        images = []
        for path in image_paths[start:start + batch_size]:
            with Image.open(path) as image:
                images.append(ImageOps.exif_transpose(image).convert("RGB").copy())
        pixels = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
        with torch.inference_mode():
            image_vectors.append(F.normalize(model.get_image_features(pixel_values=pixels).float(), dim=-1).cpu())
    image_vectors = torch.cat(image_vectors)

    def encode_texts(texts: list[str]) -> torch.Tensor:
        vectors = []
        for start in range(0, len(texts), batch_size):
            tokens = processor(text=texts[start:start + batch_size], padding=True,
                               truncation=True, return_tensors="pt")
            with torch.inference_mode():
                vectors.append(F.normalize(model.get_text_features(
                    input_ids=tokens["input_ids"].to(device),
                    attention_mask=tokens["attention_mask"].to(device)).float(), dim=-1).cpu())
        return torch.cat(vectors)

    comment_vectors = encode_texts([str(row["generated_comment"]) for row in rows])
    title_vectors = encode_texts([str(row["headline"]) for row in rows])
    all_image_scores = comment_vectors @ image_vectors.T
    own = all_image_scores[torch.arange(len(rows)), own_indices]
    if len(image_paths) > 1:
        other_mean = (all_image_scores.sum(dim=1) - own) / (len(image_paths) - 1)
        percentile = ((all_image_scores < own[:, None]).sum(dim=1).float() / (len(image_paths) - 1))
    else:
        other_mean = torch.zeros_like(own)
        percentile = torch.ones_like(own)
    title_scores = (title_vectors * image_vectors[own_indices]).sum(dim=1)
    features = [{"clip_image_comment_cosine": float(own[i]),
                 "clip_image_title_cosine": float(title_scores[i]),
                 "clip_image_comment_vs_other_mean": float(own[i] - other_mean[i]),
                 "clip_image_comment_rank_percentile": float(percentile[i])}
                for i in range(len(rows))]
    del model, processor
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return features, str(snapshot)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--metadata-json", type=Path, required=True)
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--split", choices=["train", "validate"], default="train")
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    rows = read_queue(args.queue_jsonl, args.split)
    if not rows:
        raise ValueError("Empty annotation queue")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reranker, reranker_snapshot = score_reranker(rows, args.reranker_model, device, args.batch_size)
    visual, clip_snapshot = score_clip(rows, args.clip_model, device, args.batch_size)
    result = [{"annotation_id": row["annotation_id"], "news_id": row["news_id"],
               "official_split": args.split, **reranker[i], **visual[i]}
              for i, row in enumerate(rows)]
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in result) + "\n",
                                 encoding="utf-8")
    metadata = {"rows": len(result), "split": args.split,
                "queue_sha256": hashlib.sha256(args.queue_jsonl.read_bytes()).hexdigest(),
                "reranker_model": args.reranker_model, "reranker_snapshot": reranker_snapshot,
                "clip_model": args.clip_model, "clip_snapshot": clip_snapshot,
                "warning": "Model proxy scores only; no human relevance/parent-fit labels, no image fact verification."}
    args.metadata_json.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"features": str(args.output_jsonl), "metadata": str(args.metadata_json),
                      **metadata}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
