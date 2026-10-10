"""Compare per-news predictions from two comment-graph checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.amp import autocast
from torch_geometric.loader import DataLoader

from comment_graph.common import load_config, make_dataset, make_tokenizer_and_transform
from comment_graph.model import CommentGraphMFCL


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--left-checkpoint", type=Path, required=True)
    parser.add_argument("--left-mode", required=True)
    parser.add_argument("--left-pooling", required=True)
    parser.add_argument("--right-checkpoint", type=Path, required=True)
    parser.add_argument("--right-mode", required=True)
    parser.add_argument("--right-pooling", required=True)
    parser.add_argument("--graph-structure", default="dynamic_cross_modal")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def predict(config, config_path, tokenizer, transform, args, side, device):
    checkpoint_path = getattr(args, f"{side}_checkpoint").resolve()
    mode = getattr(args, f"{side}_mode")
    pooling = getattr(args, f"{side}_pooling")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["mode"] != mode or checkpoint["pooling_mode"] != pooling:
        raise ValueError("Requested mode/pooling does not match checkpoint metadata")
    dataset = make_dataset(
        config, config_path, args.split, mode, tokenizer, transform
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config.training.batch_size),
        shuffle=False,
        num_workers=0,
    )
    model = CommentGraphMFCL(
        config,
        mode=mode,
        graph_structure=args.graph_structure,
        pooling_mode=pooling,
        decision_mode=checkpoint.get("decision_mode", "concat"),
    ).to(device)
    incompatible = model.load_state_dict(checkpoint["model"], strict=False)
    unexpected = list(incompatible.unexpected_keys)
    missing = list(incompatible.missing_keys)
    allowed_missing = all(key.startswith(("quality_gate.", "quality_attention.")) for key in missing)
    if unexpected or (missing and not allowed_missing):
        raise RuntimeError(
            f"Unexpected checkpoint mismatch for {checkpoint_path}: "
            f"missing={missing}, unexpected={unexpected}"
        )
    model.eval()
    use_amp = bool(config.training.use_amp and device.type == "cuda")
    records = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            with autocast(device_type=device.type, enabled=use_amp):
                output = model(batch)
            probabilities = output["logits"].softmax(dim=-1)
            for graph_index, news_id in enumerate(batch.news_id):
                graph_mask = batch.batch == graph_index
                generated_mask = graph_mask & (batch.node_type == 2)
                quality = output["node_quality"][generated_mask]
                records.append(
                    {
                        "news_id": str(news_id),
                        "label": int(batch.y.view(-1)[graph_index].cpu()),
                        "prediction": int(probabilities[graph_index].argmax().cpu()),
                        "fake_probability": float(probabilities[graph_index, 1].cpu()),
                        "generated_quality": (
                            {
                                "min": float(quality.min().cpu()),
                                "mean": float(quality.mean().cpu()),
                                "max": float(quality.max().cpu()),
                            }
                            if quality.numel()
                            else None
                        ),
                    }
                )
            alpha_real = float(output["quality_alpha_real"].cpu())
            alpha_generated = float(output["quality_alpha_generated"].cpu())
    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_metadata": {
            "epoch": int(checkpoint["epoch"]),
            "mode": checkpoint["mode"],
            "graph_structure": checkpoint["graph_structure"],
            "pooling_mode": checkpoint["pooling_mode"],
            "decision_mode": checkpoint.get("decision_mode", "concat"),
        },
        "missing_new_unused_keys": missing,
        "alpha_real": alpha_real,
        "alpha_generated": alpha_generated,
        "records": records,
    }


def main():
    args = parse_args()
    config, config_path = load_config(args.config)
    tokenizer, transform = make_tokenizer_and_transform(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    left = predict(config, config_path, tokenizer, transform, args, "left", device)
    right = predict(config, config_path, tokenizer, transform, args, "right", device)
    left_by_id = {record["news_id"]: record for record in left["records"]}
    right_by_id = {record["news_id"]: record for record in right["records"]}
    if set(left_by_id) != set(right_by_id):
        raise RuntimeError("The two evaluated datasets do not contain identical news IDs")

    comparisons = []
    for news_id in sorted(left_by_id):
        left_record = left_by_id[news_id]
        right_record = right_by_id[news_id]
        comparisons.append(
            {
                "news_id": news_id,
                "label": left_record["label"],
                "left_prediction": left_record["prediction"],
                "right_prediction": right_record["prediction"],
                "left_fake_probability": left_record["fake_probability"],
                "right_fake_probability": right_record["fake_probability"],
                "absolute_probability_difference": abs(
                    left_record["fake_probability"] - right_record["fake_probability"]
                ),
                "right_generated_quality": right_record["generated_quality"],
            }
        )

    quality_means = [
        row["right_generated_quality"]["mean"]
        for row in comparisons
        if row["right_generated_quality"] is not None
    ]
    result = {
        "device": str(device),
        "sample_count": len(comparisons),
        "left": {key: value for key, value in left.items() if key != "records"},
        "right": {key: value for key, value in right.items() if key != "records"},
        "same_hard_prediction_count": sum(
            row["left_prediction"] == row["right_prediction"] for row in comparisons
        ),
        "different_hard_prediction_count": sum(
            row["left_prediction"] != row["right_prediction"] for row in comparisons
        ),
        "left_error_ids": [
            row["news_id"] for row in comparisons if row["left_prediction"] != row["label"]
        ],
        "right_error_ids": [
            row["news_id"] for row in comparisons if row["right_prediction"] != row["label"]
        ],
        "mean_absolute_probability_difference": sum(
            row["absolute_probability_difference"] for row in comparisons
        ) / len(comparisons),
        "max_absolute_probability_difference": max(
            row["absolute_probability_difference"] for row in comparisons
        ),
        "right_generated_quality_mean_across_news": (
            sum(quality_means) / len(quality_means) if quality_means else None
        ),
        "largest_probability_differences": sorted(
            comparisons,
            key=lambda row: row["absolute_probability_difference"],
            reverse=True,
        )[:10],
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output_json:
        output_path = args.output_json.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
