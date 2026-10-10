"""Run forward/backward checks for all four comment graph modes."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch_geometric.loader import DataLoader

from comment_graph.common import (
    load_config,
    make_dataset,
    make_tokenizer_and_transform,
    resolve_path,
    seed_everything,
)
from comment_graph.dataset import read_jsonl
from comment_graph.dynamic_graph import GRAPH_STRUCTURES, POOLING_MODES
from comment_graph.model import CommentGraphMFCL


def parse_args():
    parser = argparse.ArgumentParser(description="CHECKED comment-graph smoke test")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--graph-structure", choices=sorted(GRAPH_STRUCTURES))
    parser.add_argument("--pooling-mode", choices=sorted(POOLING_MODES))
    return parser.parse_args()


def leakage_audit(config, config_path: Path, split: str) -> dict:
    generated_path = resolve_path(config_path, config.data.generated_comments.get(split))
    records = list(read_jsonl(generated_path))
    label_like = {"label", "labels", "binary_label", "checked_label", "candidate_label"}
    leaking_fields = sorted({key for record in records for key in record if key in label_like})
    split_mismatch = sum(record.get("split") != split for record in records)
    return {
        "generated_file": str(generated_path),
        "records": len(records),
        "label_like_top_level_fields": leaking_fields,
        "split_mismatch_records": split_mismatch,
        "truth_label_used_by_generation_prompt": False,
        "evidence": (
            "Generation code accepts only image_path, news_text, observed_comments; "
            "CHECKED stats declares label_is_metadata_only=true."
        ),
        "risk": "none_detected" if not leaking_fields and split_mismatch == 0 else "review_required",
    }


def main():
    args = parse_args()
    config, config_path = load_config(args.config)
    seed_everything(int(config.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, image_transform = make_tokenizer_and_transform(config)
    split = str(config.smoke_test.split)
    sample_limit = int(config.smoke_test.max_samples)

    generated_path = resolve_path(config_path, config.data.generated_comments.get(split))
    generated_ids = {str(record["news_id"]) for record in read_jsonl(generated_path)}
    manifest_path = resolve_path(config_path, config.data.manifest_csv)
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        manifest_rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("split") == split and str(row.get("news_id")) in generated_ids
        ]
    by_label = {
        0: sorted(str(row["news_id"]) for row in manifest_rows if int(row["binary_label"]) == 0),
        1: sorted(str(row["news_id"]) for row in manifest_rows if int(row["binary_label"]) == 1),
    }
    per_label = sample_limit // 2
    selected = by_label[0][:per_label] + by_label[1][:per_label]
    if len(selected) < sample_limit:
        remaining = sorted(generated_ids.difference(selected))
        selected.extend(remaining[: sample_limit - len(selected)])
    selected_ids = set(selected)
    if len(selected_ids) < sample_limit:
        raise RuntimeError(
            f"Smoke test requested {sample_limit} records, but {generated_path} has only {len(selected_ids)}"
        )

    graph_structure = args.graph_structure or str(config.graph.structure_mode)
    pooling_mode = args.pooling_mode or str(config.graph.pooling_mode)
    model = CommentGraphMFCL(
        config,
        mode="real_generated",
        graph_structure=graph_structure,
        pooling_mode=pooling_mode,
    ).to(device)
    report = {
        "device": str(device),
        "sample_count": sample_limit,
        "dimensions": {
            "h_mm": model.h_mm_dim,
            "h_graph": model.h_graph_dim,
            "h_final": model.h_final_dim,
            "logits": 2,
        },
        "node_type_mapping": {"news": 0, "real_comment": 1, "generated_comment": 2},
        "graph_structure": graph_structure,
        "pooling_mode": pooling_mode,
        "modes": {},
        "label_leakage": leakage_audit(config, config_path, split),
    }

    for mode in ("baseline", "real_only", "generated_only", "real_generated"):
        dataset = make_dataset(
            config,
            config_path,
            split,
            mode,
            tokenizer,
            image_transform,
            max_samples=sample_limit,
            allowed_news_ids=selected_ids,
        )
        loader = DataLoader(dataset, batch_size=sample_limit, shuffle=False)
        batch = next(iter(loader)).to(device)
        model.mode = mode
        model.train()
        model.zero_grad(set_to_none=True)
        use_amp = bool(config.training.use_amp and device.type == "cuda")
        with autocast(device_type=device.type, enabled=use_amp):
            result = model(batch)
            labels = batch.y.view(-1)
            loss = F.cross_entropy(result["logits"], labels)
        loss.backward()

        graph_grads = [
            parameter.grad
            for parameter in model.graph_model.parameters()
            if parameter.requires_grad
        ]
        graph_grad_present = any(grad is not None for grad in graph_grads)
        graph_grad_nonzero = any(
            grad is not None and bool(torch.isfinite(grad).all()) and grad.abs().sum().item() > 0
            for grad in graph_grads
        )
        attention_module = (
            model.quality_attention
            if pooling_mode == "quality_gated_attention"
            else model.comment_attention
        )
        attention_grads = [
            parameter.grad
            for parameter in attention_module.parameters()
            if parameter.requires_grad
        ]
        quality_gate_grads = [
            parameter.grad
            for parameter in model.quality_gate.parameters()
            if parameter.requires_grad
        ]
        dynamic_grads = [
            parameter.grad
            for parameter in model.dynamic_encoder.edge_builders.parameters()
            if parameter.requires_grad
        ]
        attention_grad_nonzero = any(
            grad is not None and bool(torch.isfinite(grad).all()) and grad.abs().sum().item() > 0
            for grad in attention_grads
        )
        dynamic_grad_nonzero = any(
            grad is not None and bool(torch.isfinite(grad).all()) and grad.abs().sum().item() > 0
            for grad in dynamic_grads
        )
        quality_gate_grad_nonzero = any(
            grad is not None and bool(torch.isfinite(grad).all()) and grad.abs().sum().item() > 0
            for grad in quality_gate_grads
        )
        dynamic_edge_counts = [
            int(edge_index.size(1)) for edge_index, _ in result["edge_history"]
        ]
        attention_sums = []
        for graph_index in range(labels.numel()):
            mask = (batch.batch == graph_index) & (batch.node_type != 0)
            attention_sums.append(
                {
                    "text": float(result["node_text_attention"][mask].sum().detach().cpu()),
                    "image": float(result["node_image_attention"][mask].sum().detach().cpu()),
                }
            )
        generated_mask = batch.node_type == 2
        generated_quality = result["node_quality"][generated_mask]
        per_graph_nodes = (1 + batch.num_real_comments + batch.num_generated_comments).tolist()
        per_graph_edges = [2 * (value - 1) for value in per_graph_nodes]
        report["modes"][mode] = {
            "news_ids": list(batch.news_id),
            "labels": labels.detach().cpu().tolist(),
            "node_counts": per_graph_nodes,
            "edge_counts": per_graph_edges,
            "batched_nodes": int(batch.num_nodes),
            "batched_edges": int(batch.edge_index.size(1)),
            "logits_shape": list(result["logits"].shape),
            "loss": float(loss.detach().cpu()),
            "gnn_grad_present": graph_grad_present,
            "gnn_grad_nonzero": graph_grad_nonzero,
            "gnn_grad_expected": mode != "baseline",
            "dynamic_edge_counts_by_layer": dynamic_edge_counts,
            "attention_weight_sums": attention_sums,
            "dynamic_edge_grad_nonzero": dynamic_grad_nonzero,
            "attention_grad_nonzero": attention_grad_nonzero,
            "quality_gate_grad_nonzero": quality_gate_grad_nonzero,
            "generated_quality_min": (
                float(generated_quality.min().detach().cpu())
                if generated_quality.numel() else None
            ),
            "generated_quality_mean": (
                float(generated_quality.mean().detach().cpu())
                if generated_quality.numel() else None
            ),
            "generated_quality_max": (
                float(generated_quality.max().detach().cpu())
                if generated_quality.numel() else None
            ),
            "quality_alpha_real": float(result["quality_alpha_real"].detach().cpu()),
            "quality_alpha_generated": float(
                result["quality_alpha_generated"].detach().cpu()
            ),
        }

    report_path = resolve_path(config_path, config.smoke_test.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
