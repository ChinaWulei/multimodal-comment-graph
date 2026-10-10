"""Label-blind cross-news comment swap check for a frozen reliability model.

This is a diagnostic, not a replacement for human quality/attachment labels.
Only generated comment token sequences are swapped within the validation split;
news, images, real comments, observed edges, and labels are untouched.
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
from torch_geometric.loader import DataLoader

from comment_graph.common import load_config, make_dataset, make_tokenizer_and_transform
from comment_graph.model import CommentGraphMFCL
from comment_graph.isolated_fusion import select_nodes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    config, config_path = load_config(args.config)
    config.graph.structure_mode = "observed_tree"
    config.graph.message_passing = "typed_rgcn"
    config.graph.pooling_mode = "mean"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    mode = str(checkpoint.get("generated_reliability", "none"))
    decision_mode = str(checkpoint.get("decision_mode", "concat"))
    if mode not in {"none", "quality", "confidence", "quality_confidence", "placement", "both"}:
        raise ValueError("Unknown generated reliability mode in checkpoint")
    saved_config = checkpoint.get("config", {})
    saved_data = saved_config.get("data", {})
    saved_graph = saved_config.get("graph", {})
    if "max_real_comments" in saved_data:
        config.data.max_real_comments = int(saved_data["max_real_comments"])
    if "max_generated_comments" in saved_data:
        config.data.max_generated_comments = int(saved_data["max_generated_comments"])
    if "real_node_selection" in saved_data:
        config.data.real_node_selection = str(saved_data["real_node_selection"])
    if "generated_reliability" in saved_graph:
        config.graph.generated_reliability = saved_graph["generated_reliability"]
    config.graph.generated_reliability["mode"] = mode
    config.decision_mode = decision_mode
    if "reliability_residual" in saved_config:
        config["reliability_residual"] = saved_config["reliability_residual"]
    tokenizer, transform = make_tokenizer_and_transform(config)
    dataset = make_dataset(config, config_path, "validate", "real_generated", tokenizer, transform)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    model = CommentGraphMFCL(config, "real_generated", decision_mode=decision_mode).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    records = []
    with torch.no_grad():
        for batch in loader:
            if len(batch.news_id) < 2:
                continue
            batch = batch.to(device)
            generated_by_graph = [torch.nonzero((batch.batch == i) & (batch.node_type == 2),
                                                 as_tuple=False).flatten()
                                  for i in range(len(batch.news_id))]
            if any(len(indices) != len(generated_by_graph[0]) for indices in generated_by_graph):
                raise ValueError("Cross-news swap requires equal generated-comment counts")
            clean = model(batch)
            swapped_batch = batch.clone()
            original_ids = batch.node_input_ids.clone()
            original_mask = batch.node_attention_mask.clone()
            for i, target in enumerate(generated_by_graph):
                donor = generated_by_graph[(i + 1) % len(generated_by_graph)]
                swapped_batch.node_input_ids[target] = original_ids[donor]
                swapped_batch.node_attention_mask[target] = original_mask[donor]
            swapped = model(swapped_batch)
            image_swapped_batch = batch.clone()
            image_swapped_batch.pixel_values = batch.pixel_values.roll(shifts=1, dims=0)
            image_swapped = model(image_swapped_batch)
            without_generated_batch, _ = select_nodes(batch, batch.node_type != 2)
            without_generated = model(without_generated_batch)
            for i, news_id in enumerate(batch.news_id):
                indices = generated_by_graph[i]
                clean_prob = F.softmax(clean["logits"][i].float(), dim=-1)
                swapped_prob = F.softmax(swapped["logits"][i].float(), dim=-1)
                image_swapped_prob = F.softmax(image_swapped["logits"][i].float(), dim=-1)
                without_generated_prob = F.softmax(without_generated["logits"][i].float(), dim=-1)
                records.append({
                    "news_id": str(news_id),
                    "donor_news_id": str(batch.news_id[(i + 1) % len(batch.news_id)]),
                    "clean_mean_reliability": float(clean["node_quality"][indices].mean()),
                    "swapped_mean_reliability": float(swapped["node_quality"][indices].mean()),
                    "image_swapped_mean_reliability": float(image_swapped["node_quality"][indices].mean()),
                    "clean_prediction": int(clean_prob.argmax()),
                    "swapped_prediction": int(swapped_prob.argmax()),
                    "prediction_probability_mae": float((clean_prob - swapped_prob).abs().mean()),
                    "image_swap_probability_mae": float((clean_prob - image_swapped_prob).abs().mean()),
                    "without_generated_prediction": int(without_generated_prob.argmax()),
                    "without_generated_probability_mae": float((clean_prob - without_generated_prob).abs().mean()),
                    "label": int(batch.y.view(-1)[i]),
                })
    result = {
        "split": "validate", "mode": mode, "decision_mode": decision_mode,
        "max_real_comments": int(config.data.max_real_comments),
        "n_news": len(records),
        "clean_mean_reliability": sum(r["clean_mean_reliability"] for r in records) / len(records),
        "swapped_mean_reliability": sum(r["swapped_mean_reliability"] for r in records) / len(records),
        "image_swapped_mean_reliability": sum(r["image_swapped_mean_reliability"] for r in records) / len(records),
        "mean_generated_comment_swap_probability_mae": sum(r["prediction_probability_mae"] for r in records) / len(records),
        "mean_image_swap_probability_mae": sum(r["image_swap_probability_mae"] for r in records) / len(records),
        "mean_remove_generated_probability_mae": sum(r["without_generated_probability_mae"] for r in records) / len(records),
        "remove_generated_prediction_flip_count": sum(
            r["clean_prediction"] != r["without_generated_prediction"] for r in records
        ),
        "generated_helped_validation_news": sum(
            r["clean_prediction"] == r["label"] and r["without_generated_prediction"] != r["label"]
            for r in records
        ),
        "generated_hurt_validation_news": sum(
            r["clean_prediction"] != r["label"] and r["without_generated_prediction"] == r["label"]
            for r in records
        ),
        "news_with_lower_swapped_reliability": sum(
            r["swapped_mean_reliability"] < r["clean_mean_reliability"] for r in records
        ),
        "prediction_flip_count": sum(r["clean_prediction"] != r["swapped_prediction"] for r in records),
        "records": records,
    }
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
