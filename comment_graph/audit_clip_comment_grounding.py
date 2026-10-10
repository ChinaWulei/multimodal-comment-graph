"""Check whether cached CLIP sees an image signal in generated comments.

This is a label-free diagnostic. CLIP similarity is not a factuality or
comment-quality ground truth, and image swapping is not a causal proof.
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
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from PIL import Image, ImageOps
from transformers import CLIPModel, CLIPProcessor

from comment_graph.common import load_config, make_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "validate"], default="validate")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--model", default="openai/clip-vit-base-patch32")
    args = parser.parse_args()
    config, config_path = load_config(args.config)
    dataset = make_dataset(config, config_path, args.split, "real_generated", None, None)
    image_paths = [row["image_path"] for row in dataset.records]
    images = []
    for path in image_paths:
        with Image.open(path) as image:
            images.append(ImageOps.exif_transpose(image).convert("RGB"))
    comment_texts = []
    headlines = []
    comment_news_index = []
    for index, row in enumerate(dataset.records):
        comments, types, _ = dataset._nodes_and_edges(str(row["news_id"]))
        headlines.append(str(row["news_text"]))
        for content, node_type in zip(comments, types):
            if node_type == 2:
                comment_texts.append(content)
                comment_news_index.append(index)
    snapshot = snapshot_download(repo_id=args.model, local_files_only=True)
    processor = CLIPProcessor.from_pretrained(snapshot, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained(snapshot, local_files_only=True).to(device).eval()
    with torch.inference_mode():
        image_parts = []
        for start in range(0, len(images), 16):
            pixels = processor(images=images[start:start + 16], return_tensors="pt")["pixel_values"].to(device)
            image_parts.append(F.normalize(model.get_image_features(pixel_values=pixels).float(), dim=-1).cpu())
        image_vectors = torch.cat(image_parts)

        def encode_texts(texts):
            parts = []
            for start in range(0, len(texts), 32):
                tokens = processor(text=texts[start:start + 32], padding=True, truncation=True,
                                   return_tensors="pt")
                ids = tokens["input_ids"].to(device)
                mask = tokens["attention_mask"].to(device)
                parts.append(F.normalize(model.get_text_features(input_ids=ids,
                                                                  attention_mask=mask).float(), dim=-1).cpu())
            return torch.cat(parts)

        comments = encode_texts(comment_texts)
        titles = encode_texts(headlines)
    records = []
    for i, text_vector in enumerate(comments):
        news = comment_news_index[i]
        wrong = (news + 1) % len(dataset)
        records.append({"news_id": str(dataset.records[news]["news_id"]),
                        "comment": comment_texts[i],
                        "matched_image_similarity": float(text_vector @ image_vectors[news]),
                        "swapped_image_similarity": float(text_vector @ image_vectors[wrong])})
    title_records = []
    for i, vector in enumerate(titles):
        wrong = (i + 1) % len(dataset)
        title_records.append((float(vector @ image_vectors[i]), float(vector @ image_vectors[wrong])))
    result = {
        "split": args.split, "news": len(dataset), "generated_comments": len(records),
        "comment_mean_clean_minus_swapped": sum(r["matched_image_similarity"] - r["swapped_image_similarity"] for r in records) / len(records),
        "comment_clean_above_swapped_fraction": sum(r["matched_image_similarity"] > r["swapped_image_similarity"] for r in records) / len(records),
        "headline_mean_clean_minus_swapped": sum(clean - wrong for clean, wrong in title_records) / len(title_records),
        "headline_clean_above_swapped_fraction": sum(clean > wrong for clean, wrong in title_records) / len(title_records),
        "model": args.model, "model_location": snapshot,
        "warning": "CLIP is a proxy for image-text alignment, not proof of visual grounding or comment correctness.",
        "records": records,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
