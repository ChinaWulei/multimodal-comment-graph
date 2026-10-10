"""Evaluate one validation-selected comment-graph checkpoint without retraining.

This keeps model selection (validation) and final reporting (test) separate:
the exact saved model weights are loaded before a single test pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from comment_graph.common import (
    generated_coverage_ids,
    load_config,
    make_dataset,
    make_tokenizer_and_transform,
    seed_everything,
)
from comment_graph.model import CommentGraphMFCL
from comment_graph.train import class_weights, evaluate, make_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an already-selected graph checkpoint on the test split.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--generated-test-jsonl", type=Path,
        help="Override only the generated test graphs, e.g. a predeclared unseen-news cohort.",
    )
    parser.add_argument(
        "--expected-context-jsonl", type=Path,
        help="Require test dataset IDs to equal the label-blind generation-context IDs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config, config_path = load_config(args.config)
    checkpoint = torch.load(args.checkpoint.resolve(), map_location="cpu", weights_only=False)
    required = {"mode", "graph_structure", "pooling_mode", "message_passing", "decision_mode", "model", "validation_metrics"}
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(f"Checkpoint misses required metadata: {sorted(missing)}")

    mode = str(checkpoint["mode"])
    config.comment_graph_mode = mode
    config.graph.structure_mode = str(checkpoint["graph_structure"])
    config.graph.pooling_mode = str(checkpoint["pooling_mode"])
    config.graph.message_passing = str(checkpoint["message_passing"])
    config.decision_mode = str(checkpoint["decision_mode"])
    config.seed = int(checkpoint.get("seed", config.seed))

    # The checkpoint is the authoritative experiment record.  Restore the
    # dataset/model switches which alter graph construction or the forward
    # pass; otherwise a later edited YAML could silently evaluate different
    # inputs than the ones used to select this checkpoint on validation.
    saved_config = checkpoint.get("config", {})
    saved_data = saved_config.get("data", {})
    saved_graph = saved_config.get("graph", {})
    if "local_correction" in saved_config:
        # Temperature and dropout are not recoverable from the state dict.
        # Restore them too, so an edited YAML cannot change the selected head.
        config["local_correction"] = saved_config["local_correction"]
    if "reliability_residual" in saved_config:
        config["reliability_residual"] = saved_config["reliability_residual"]
    if "max_generated_comments" in saved_data:
        config.data.max_generated_comments = int(saved_data["max_generated_comments"])
    config.data.real_node_selection = str(saved_data.get("real_node_selection", "prefix"))
    if "max_real_comments" in saved_data:
        config.data.max_real_comments = int(saved_data["max_real_comments"])
    if "semantic_parent_gate" in saved_graph:
        config.graph.semantic_parent_gate = bool(saved_graph["semantic_parent_gate"])
    if "semantic_parent_gate_floor" in saved_graph:
        config.graph.semantic_parent_gate_floor = float(saved_graph["semantic_parent_gate_floor"])
    if "generated_reliability" in saved_graph:
        config.graph.generated_reliability = saved_graph["generated_reliability"]
    elif "generated_reliability" in checkpoint:
        config.graph.generated_reliability = {"mode": checkpoint["generated_reliability"]}
    generated_test_path = None
    if args.generated_test_jsonl is not None:
        generated_test_path = args.generated_test_jsonl.resolve()
        if not generated_test_path.is_file():
            raise FileNotFoundError(generated_test_path)
        config.data.generated_comments.test = str(generated_test_path)
    seed_everything(int(config.seed))

    tokenizer, image_transform = make_tokenizer_and_transform(config)
    allowed = None
    if bool(config.data.get("restrict_to_generated_coverage", False)):
        allowed = generated_coverage_ids(config, config_path, "test")
    test_set = make_dataset(
        config, config_path, "test", mode, tokenizer, image_transform,
        allowed_news_ids=allowed,
    )
    evaluated_ids = [str(row["news_id"]) for row in test_set.records]
    if len(evaluated_ids) != len(set(evaluated_ids)):
        raise ValueError("Duplicate news IDs in frozen test cohort")
    expected_context_path = None
    if args.expected_context_jsonl is not None:
        expected_context_path = args.expected_context_jsonl.resolve()
        if not expected_context_path.is_file():
            raise FileNotFoundError(expected_context_path)
        expected_rows = [json.loads(line) for line in expected_context_path.read_text(encoding="utf-8").splitlines()
                         if line.strip()]
        expected_ids = [str(row["metadata"]["news_id"]) for row in expected_rows]
        if len(expected_ids) != len(set(expected_ids)) or set(expected_ids) != set(evaluated_ids):
            raise ValueError("Frozen test cohort does not exactly match expected generation-context IDs")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CommentGraphMFCL(
        config,
        mode,
        graph_structure=str(config.graph.structure_mode),
        pooling_mode=str(config.graph.pooling_mode),
        decision_mode=str(config.decision_mode),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)

    # Class weighting affects only the reported loss.  Match the training
    # configuration when it was enabled; F1/accuracy use no label weights.
    weights = None
    if bool(config.training.class_weighted_loss):
        train_allowed = generated_coverage_ids(config, config_path, "train") if allowed is not None else None
        train_set = make_dataset(
            config, config_path, "train", mode, tokenizer, image_transform,
            allowed_news_ids=train_allowed,
        )
        weights = class_weights(train_set, device)
    metrics = evaluate(
        model,
        make_loader(test_set, config, shuffle=False),
        device,
        bool(config.training.use_amp and device.type == "cuda"),
        weights,
    )
    result = {
        "protocol": "frozen_validation_checkpoint_single_test_evaluation",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.resolve().read_bytes()).hexdigest(),
        "mode": mode,
        "graph_structure": str(config.graph.structure_mode),
        "pooling_mode": str(config.graph.pooling_mode),
        "message_passing": str(config.graph.message_passing),
        "decision_mode": str(config.decision_mode),
        "max_generated_comments": int(config.data.get("max_generated_comments", 0)),
        "max_real_comments": int(config.data.get("max_real_comments", 0)),
        "real_node_selection": str(config.data.real_node_selection),
        "semantic_parent_gate": bool(config.graph.get("semantic_parent_gate", False)),
        "semantic_parent_gate_floor": float(config.graph.get("semantic_parent_gate_floor", 0.0)),
        "generated_reliability": str(config.graph.get("generated_reliability", {}).get("mode", "none")),
        "seed": int(config.seed),
        "test_size": len(test_set),
        "test_news_ids": evaluated_ids,
        "generated_test_jsonl": str(generated_test_path) if generated_test_path else None,
        "generated_test_sha256": hashlib.sha256(generated_test_path.read_bytes()).hexdigest()
        if generated_test_path else None,
        "expected_context_jsonl": str(expected_context_path) if expected_context_path else None,
        "paired_generated_coverage": allowed is not None,
        "stored_validation": checkpoint["validation_metrics"],
        "test": metrics,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "protocol": result["protocol"], "checkpoint": result["checkpoint"],
        "test_size": result["test_size"], "test_macro_f1": metrics["macro_f1"],
        "test_accuracy": metrics["accuracy"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
