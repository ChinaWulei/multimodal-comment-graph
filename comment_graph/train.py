"""Train/evaluate MFCL with CHECKED real/generated comment star graphs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics
from torch.amp import GradScaler, autocast
from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader

from comment_graph.common import (
    generated_coverage_ids, load_config, make_dataset, make_tokenizer_and_transform,
    resolve_path, seed_everything,
)
from comment_graph.dataset import GRAPH_MODES
from comment_graph.dynamic_graph import GRAPH_STRUCTURES, POOLING_MODES
from comment_graph.model import CommentGraphMFCL
from comment_graph.local_correction import DECISION_MODES
from comment_graph.isolated_fusion import classification_objective
from comment_graph.robustness import ROBUSTNESS_MODES, add_mismatch_loss, mismatch_forward
from comment_graph.evidence_fusion import EVIDENCE_MODES
from comment_graph.checkpoint_io import portable_checkpoint_path, save_checkpoint_atomic
from comment_graph.generated_reliability import RELIABILITY_MODES


def parse_args():
    parser = argparse.ArgumentParser(description="Train CHECKED MFCL comment-graph classifier")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--mode", choices=sorted(GRAPH_MODES))
    parser.add_argument("--graph-structure", choices=sorted(GRAPH_STRUCTURES))
    parser.add_argument("--pooling-mode", choices=sorted(POOLING_MODES))
    parser.add_argument("--message-passing", choices=["legacy_gcn", "typed_rgcn"])
    parser.add_argument("--decision-mode", choices=sorted(DECISION_MODES))
    parser.add_argument("--robustness-mode", choices=sorted(ROBUSTNESS_MODES))
    parser.add_argument("--correction-reference", choices=["none", "null_generated"])
    parser.add_argument("--mm-fusion", choices=["mfcl", "mts", "mimoe"])
    parser.add_argument("--evidence-mode", choices=sorted(EVIDENCE_MODES))
    parser.add_argument("--audit-mismatch", action="store_true", help="Report label-blind comment mismatch stress metrics")
    parser.add_argument("--sampling", choices=["shuffle", "balanced"], help="Training only; balanced disables inverse-frequency training CE weights")
    parser.add_argument(
        "--max-generated-comments",
        type=int,
        help="Optional ablation cap for generated nodes per news; does not alter the persisted graph JSONL.",
    )
    parser.add_argument(
        "--max-real-comments", type=int,
        help="Optional partial-observation cap; selected generator context and ancestors must still fit.",
    )
    parser.add_argument("--generated-train-jsonl", type=Path,
                        help="Use an explicitly verified train-split generated graph file.")
    parser.add_argument("--generated-validate-jsonl", type=Path,
                        help="Use an explicitly verified validation-split generated graph file.")
    parser.add_argument(
        "--semantic-parent-gate",
        action="store_true",
        help="Apply label-free generated-node gates from parent/comment semantic consistency.",
    )
    parser.add_argument(
        "--generated-reliability", choices=sorted(RELIABILITY_MODES),
        help="Ablate generated-comment quality control and parent reattachment.",
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--checkpoint-dir", type=Path, help="Separate output directory for this experiment")
    parser.add_argument("--seed", type=int, help="Optional reproducible seed and checkpoint suffix")
    parser.add_argument(
        "--summary-json",
        type=Path,
        help="Optional path for a machine-readable validation/test summary.",
    )
    parser.add_argument(
        "--allow-formal-training",
        action="store_true",
        help="Required guard: formal training is not started accidentally.",
    )
    parser.add_argument("--run-test", action="store_true", help="Evaluate test only after model selection is frozen")
    return parser.parse_args()


def make_loader(dataset, config, shuffle, drop_last=False):
    # Keep data order independent of how many parameters an ablation initializes.
    generator = torch.Generator().manual_seed(int(config.seed))
    return DataLoader(
        dataset,
        batch_size=int(config.training.batch_size),
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=int(config.training.num_workers),
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def class_weights(dataset, device):
    counts = Counter(dataset.labels())
    total = sum(counts.values())
    num_classes = int(dataset.num_classes)
    values = [total / (num_classes * max(counts.get(label, 0), 1)) for label in range(num_classes)]
    return torch.tensor(values, dtype=torch.float32, device=device)


def make_training_loader(dataset, config, sampling):
    if sampling == "shuffle":
        return make_loader(dataset, config, shuffle=True, drop_last=True)
    if sampling != "balanced":
        raise ValueError(f"Unsupported sampling: {sampling}")
    labels = dataset.labels()
    counts = Counter(labels)
    expected = set(range(int(dataset.num_classes)))
    if set(counts) != expected:
        raise ValueError(f"Balanced sampling requires every class: expected {sorted(expected)}, got {sorted(counts)}")
    sampler = WeightedRandomSampler(
        [1.0 / counts[y] for y in labels], len(labels), replacement=True,
        generator=torch.Generator().manual_seed(int(config.seed)),
    )
    return DataLoader(dataset, batch_size=int(config.training.batch_size), sampler=sampler,
                      drop_last=True, num_workers=int(config.training.num_workers),
                      pin_memory=torch.cuda.is_available(),
                      generator=torch.Generator().manual_seed(int(config.seed)))


def evaluate(model, loader, device, use_amp, weights, audit_mismatch=False, audit_seed=20261004):
    model.eval()
    if audit_mismatch and model.decision_mode != "source_isolated":
        raise ValueError("Mismatch audit requires source_isolated")
    audit_rng = torch.Generator().manual_seed(audit_seed)
    audit_records = []
    predictions, labels, probabilities, records, global_predictions = [], [], [], [], []
    generated_diagnostics = []
    num_classes = int(model.num_classes)
    class_labels = list(range(num_classes))
    loss_sum, loss_denominator, base_loss_sum = 0.0, 0.0, 0.0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            with autocast(device_type=device.type, enabled=use_amp):
                output = model(batch, return_generated_context=audit_mismatch)
                logits = output["logits"]
                y = batch.y.view(-1)
                loss = F.cross_entropy(logits, y, weight=weights, reduction="sum")
                base_loss = F.cross_entropy(output["global_logits"], y, weight=weights, reduction="sum")
            loss_sum += loss.item()
            base_loss_sum += base_loss.item()
            loss_denominator += weights[y].sum().item() if weights is not None else y.numel()
            predictions.extend(logits.argmax(dim=-1).cpu().tolist())
            labels.extend(y.cpu().tolist())
            class_probabilities = logits.float().softmax(-1).cpu().tolist()
            if num_classes == 2:
                probabilities.extend(row[1] for row in class_probabilities)
            global_predictions.extend(output["global_logits"].argmax(-1).cpu().tolist())
            correction = output["local_correction"]
            diagnostics_by_graph = {}
            for item in output.get("generated_reliability_diagnostics", []):
                graph_index = int(batch.batch[item["node_index"]])
                first_node = int(batch.ptr[graph_index])
                local_item = dict(item)
                for key in ("node_index", "original_parent", "selected_parent"):
                    local_item[key] -= first_node
                diagnostics_by_graph.setdefault(graph_index, []).append(local_item)
                generated_diagnostics.append(local_item)
            noisy = mismatch_forward(model, output, audit_rng) if audit_mismatch else None
            if noisy is not None:
                if num_classes != 2:
                    raise ValueError("Mismatch audit is supported only for binary classifiers")
                clean_prob = logits.detach().float().softmax(-1)[:, 1]
                base_prob = output["global_logits"].detach().float().softmax(-1)[:, 1]
                noisy_prob = noisy["logits"].softmax(-1)[:, 1]
                for i in noisy["valid"].nonzero().flatten().tolist():
                    audit_records.append({
                        "news_id": str(batch.news_id[i]),
                        "donor_news_id": str(batch.news_id[noisy["donors"][i]]),
                        "label": int(y[i]), "clean_prediction": int(logits[i].argmax()),
                        "base_prediction": int(output["global_logits"][i].argmax()),
                        "mismatch_prediction": int(noisy["logits"][i].argmax()),
                        "clean_fake_probability": float(clean_prob[i]),
                        "base_fake_probability": float(base_prob[i]),
                        "mismatch_fake_probability": float(noisy_prob[i]),
                        "clean_gate": float(correction["correction_gate"][i, 0]),
                        "mismatch_gate": float(noisy["correction"]["correction_gate"][i, 0]),
                    })
            for i, news_id in enumerate(batch.news_id):
                row = {"news_id": str(news_id), "label": int(y[i]),
                       "prediction": int(logits[i].argmax()), "class_probabilities": class_probabilities[i]}
                if i in diagnostics_by_graph:
                    row["generated_reliability"] = diagnostics_by_graph[i]
                if num_classes == 2:
                    row["fake_probability"] = class_probabilities[i][1]
                if correction:
                    row["correction_gate"] = float(correction["correction_gate"][i, 0])
                    # Retain the readable binary fields for existing CHECKED
                    # result readers, while storing the complete residual for
                    # Fakeddit's 3-way task.
                    if num_classes == 2:
                        row.update({
                            "global_fake_probability": float(output["global_logits"][i].float().softmax(-1)[1]),
                            "fake_logit_delta": float(correction["logit_delta"][i, 1]),
                            "real_logit_delta": float(correction["logit_delta"][i, 0]),
                        })
                    else:
                        row["global_class_probabilities"] = (
                            output["global_logits"][i].float().softmax(-1).cpu().tolist()
                        )
                        row["class_logit_delta"] = correction["logit_delta"][i].float().cpu().tolist()
                records.append(row)
    result = {
        "loss": loss_sum / max(loss_denominator, 1e-8),
        "base_loss": base_loss_sum / max(loss_denominator, 1e-8),
        "loss_aggregation": "sum_weighted_ce / sum_target_weights",
        "accuracy": metrics.accuracy_score(labels, predictions),
        "macro_f1": metrics.f1_score(labels, predictions, average="macro", zero_division=0),
        "weighted_f1": metrics.f1_score(labels, predictions, average="weighted", zero_division=0),
        "roc_auc": metrics.roc_auc_score(labels, probabilities) if num_classes == 2 and len(set(labels)) == 2 else None,
        "confusion_matrix": metrics.confusion_matrix(labels, predictions, labels=class_labels).tolist(),
        "global_macro_f1": metrics.f1_score(labels, global_predictions, average="macro", zero_division=0),
        "global_accuracy": metrics.accuracy_score(labels, global_predictions),
        "global_confusion_matrix": metrics.confusion_matrix(labels, global_predictions, labels=class_labels).tolist(),
        "prediction_records": records,
        "report": metrics.classification_report(labels, predictions, labels=class_labels, digits=4, zero_division=0),
        "per_class_metrics": metrics.classification_report(labels, predictions, labels=class_labels,
                                                              output_dict=True, zero_division=0),
    }
    if generated_diagnostics:
        result["generated_reliability_summary"] = {
            "generated_nodes": len(generated_diagnostics),
            "rewired": sum(bool(item["rewired"]) for item in generated_diagnostics),
            "mean_text_reliability": float(np.mean([item["text_reliability"] for item in generated_diagnostics])),
            "mean_fused_reliability": float(np.mean([
                item.get("fused_reliability", item["text_reliability"])
                for item in generated_diagnostics
            ])),
        }
    if audit_mismatch:
        audit = {"evaluated_news": len(audit_records), "skipped_news": len(labels) - len(audit_records),
                 "seed": audit_seed, "scope": "same-split, within-batch; no labels used for donor selection",
                 "records": audit_records}
        if audit_records:
            ys = [r["label"] for r in audit_records]
            for condition in ("clean", "base", "mismatch"):
                pred = [r[f"{condition}_prediction"] for r in audit_records]
                audit[f"{condition}_macro_f1"] = metrics.f1_score(ys, pred, average="macro", zero_division=0)
                audit[f"{condition}_confusion_matrix"] = metrics.confusion_matrix(ys, pred, labels=[0, 1]).tolist()
            audit.update({
                "clean_gate_mean": float(np.mean([r["clean_gate"] for r in audit_records])),
                "mismatch_gate_mean": float(np.mean([r["mismatch_gate"] for r in audit_records])),
                "mismatch_vs_base_probability_mae": float(np.mean([abs(r["mismatch_fake_probability"] - r["base_fake_probability"]) for r in audit_records])),
                "clean_vs_base_probability_mae": float(np.mean([abs(r["clean_fake_probability"] - r["base_fake_probability"]) for r in audit_records])),
                "clean_vs_mismatch_probability_mae": float(np.mean([abs(r["clean_fake_probability"] - r["mismatch_fake_probability"]) for r in audit_records])),
                "clean_vs_mismatch_prediction_flip_rate": float(np.mean([r["clean_prediction"] != r["mismatch_prediction"] for r in audit_records])),
                "mismatch_flip_rate_from_base": float(np.mean([r["mismatch_prediction"] != r["base_prediction"] for r in audit_records])),
                "clean_corrected_base_errors": sum(r["clean_prediction"] == r["label"] and r["base_prediction"] != r["label"] for r in audit_records),
                "clean_introduced_errors": sum(r["clean_prediction"] != r["label"] and r["base_prediction"] == r["label"] for r in audit_records),
            })
        result["mismatch_audit"] = audit
    return result


def main():
    args = parse_args()
    if not args.allow_formal_training:
        raise SystemExit("Formal training guard active. Run smoke_test.py first, then pass --allow-formal-training.")
    config, config_path = load_config(args.config)
    if args.run_test and (args.generated_train_jsonl or args.generated_validate_jsonl):
        raise ValueError("Custom train/validation generated coverage cannot use --run-test in this entry")
    for split, override in (("train", args.generated_train_jsonl),
                            ("validate", args.generated_validate_jsonl)):
        if override is not None:
            path = override.resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            config.data.generated_comments[split] = str(path)
    if "mm_fusion" not in config:
        config.mm_fusion = {"mode": "mfcl", "hidden_dim": 768, "layers": 8}
    if args.mm_fusion:
        config.mm_fusion["mode"] = args.mm_fusion
    if "entity_evidence" not in config:
        config.entity_evidence = {"mode": "none"}
    if args.evidence_mode:
        config.entity_evidence["mode"] = args.evidence_mode
    mm_fusion_mode = config.mm_fusion.get("mode", "mfcl")
    evidence_mode = str(config.entity_evidence.get("mode", "none"))
    if evidence_mode not in EVIDENCE_MODES:
        raise ValueError("Unknown entity evidence mode")
    if args.seed is not None:
        config.seed = args.seed
    for key in ("epochs", "patience"):
        value = getattr(args, key)
        if value is not None:
            if value < 1:
                raise ValueError(f"{key} must be positive")
            config.training[key] = value
    if args.checkpoint_dir:
        config.training.checkpoint_dir = str(args.checkpoint_dir.resolve())
    if args.max_generated_comments is not None:
        if args.max_generated_comments < 0:
            raise ValueError("max-generated-comments must be non-negative")
        config.data.max_generated_comments = int(args.max_generated_comments)
    if args.max_real_comments is not None:
        if args.max_real_comments < 1:
            raise ValueError("max-real-comments must be positive")
        config.data.max_real_comments = int(args.max_real_comments)
    if args.semantic_parent_gate:
        config.graph.semantic_parent_gate = True
    if "generated_reliability" not in config.graph:
        config.graph.generated_reliability = {}
    if args.generated_reliability:
        config.graph.generated_reliability["mode"] = args.generated_reliability
    generated_reliability = str(config.graph.generated_reliability.get("mode", "none"))
    sampling = args.sampling or config.training.get("sampling", "shuffle")
    config.training.sampling = sampling
    mode = args.mode or str(config.comment_graph_mode)
    graph_structure = args.graph_structure or str(config.graph.structure_mode)
    pooling_mode = args.pooling_mode or str(config.graph.pooling_mode)
    message_passing = args.message_passing or str(config.graph.get("message_passing", "legacy_gcn"))
    decision_mode = args.decision_mode or config.get("decision_mode", "concat")
    config.comment_graph_mode = mode
    config.graph.structure_mode = graph_structure
    config.graph.pooling_mode = pooling_mode
    config.graph.message_passing = message_passing
    config.decision_mode = decision_mode
    num_classes = int(config.get("classification", {}).get("num_classes", 2))
    if num_classes != 2 and decision_mode == "source_isolated":
        raise ValueError("source_isolated is currently implemented only for binary classification")
    reference = args.correction_reference or config.get("isolated_fusion", {}).get("correction_reference", "none")
    if reference not in {"none", "null_generated"}:
        raise ValueError("Unknown correction reference")
    if reference != "none" and (decision_mode != "source_isolated" or mode not in {"real_generated", "generated_only"}):
        raise ValueError("Null reference requires source_isolated and generated comments")
    if "isolated_fusion" not in config:
        config.isolated_fusion = {}
    config.isolated_fusion["correction_reference"] = reference
    robust_mode = args.robustness_mode or config.get("robustness", {}).get("mode", "none")
    mismatch_weight = (float(config.get("robustness", {}).get("mismatch_weight", .5))
                       if robust_mode == "mismatch_fallback" else 0.0)
    if robust_mode not in ROBUSTNESS_MODES or mismatch_weight < 0:
        raise ValueError("Invalid robustness mode or weight")
    if (robust_mode != "none" or args.audit_mismatch) and (
            decision_mode != "source_isolated" or mode not in {"real_generated", "generated_only"}):
        raise ValueError("Robustness requires source_isolated and a mode with generated comments")
    audit_seed = int(config.get("robustness", {}).get("audit_seed", 20261004))
    config.robustness = {"mode": robust_mode, "mismatch_weight": mismatch_weight,
                         "audit_seed": audit_seed}
    mismatch_rng = torch.Generator().manual_seed(int(config.seed) + 10000)
    if decision_mode == "source_isolated":
        base_loss_weight = float(config.get("isolated_fusion", {}).get("base_loss_weight", .5))
    elif decision_mode == "reliability_residual":
        base_loss_weight = float(config.get("reliability_residual", {}).get("base_loss_weight", .5))
    else:
        base_loss_weight = 0.0
    if base_loss_weight < 0:
        raise ValueError("base_loss_weight must be nonnegative")
    seed_everything(int(config.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, image_transform = make_tokenizer_and_transform(config)

    # For a fair generated-comment ablation, every arm receives exactly the
    # same news IDs: those for which a generated graph is available.  Without
    # this restriction real_only could win merely by retaining extra news.
    paired_coverage = bool(config.data.get("restrict_to_generated_coverage", False))
    allowed = {}
    if paired_coverage:
        allowed = {
            split: generated_coverage_ids(config, config_path, split)
            for split in (("train", "validate", "test") if args.run_test else ("train", "validate"))
        }
    # Generated modes deliberately fail here until train/validate generated JSONLs exist.
    train_set = make_dataset(
        config, config_path, "train", mode, tokenizer, image_transform,
        allowed_news_ids=allowed.get("train"),
    )
    val_set = make_dataset(
        config, config_path, "validate", mode, tokenizer, image_transform,
        allowed_news_ids=allowed.get("validate"),
    )
    test_set = (
        make_dataset(
            config, config_path, "test", mode, tokenizer, image_transform,
            allowed_news_ids=allowed.get("test"),
        )
        if args.run_test
        else None
    )
    if {r['news_id'] for r in train_set.records} & {r['news_id'] for r in val_set.records}:
        raise ValueError("Train/validation news ID leakage")
    train_loader = make_training_loader(train_set, config, sampling)
    val_loader = make_loader(val_set, config, shuffle=False)

    model = CommentGraphMFCL(
        config,
        mode,
        graph_structure=graph_structure,
        pooling_mode=pooling_mode,
        decision_mode=decision_mode,
    ).to(device)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [p for p in model.feature_encoder.parameters() if p.requires_grad],
                "lr": float(config.encoder.learning_rate),
            },
            {
                "params": [
                    p
                    for name, p in model.named_parameters()
                    if p.requires_grad and not name.startswith("feature_encoder.")
                ],
                "lr": float(config.training.learning_rate),
            },
        ],
        weight_decay=float(config.training.weight_decay),
    )
    weights = class_weights(train_set, device) if bool(config.training.class_weighted_loss) else None
    # Balanced sampling already changes the training prior to approximately 1:1.
    # Do not compensate again with inverse-frequency CE weights. Evaluation uses
    # the same full-training-derived weights across arms for comparable losses.
    training_weights = None if sampling == "balanced" else weights
    use_amp = bool(config.training.use_amp and device.type == "cuda")
    amp_init_scale = (float(config.get("isolated_fusion", {}).get("amp_init_scale", 1024.0))
                      if decision_mode == "source_isolated" else
                      float(config.get("reliability_residual", {}).get("amp_init_scale", 1024.0))
                      if decision_mode == "reliability_residual" else 65536.0)
    scaler = GradScaler(device.type, enabled=use_amp, init_scale=amp_init_scale)
    checkpoint_dir = resolve_path(config_path, config.training.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{decision_mode}" if decision_mode != "concat" else ""
    if message_passing != "legacy_gcn":
        suffix += f"_{message_passing}"
    if generated_reliability != "none":
        suffix += f"_reliability_{generated_reliability}"
    if mm_fusion_mode != "mfcl":
        suffix += f"_{mm_fusion_mode}"
    if evidence_mode != "none":
        suffix += f"_evidence_{evidence_mode}"
    if robust_mode != "none":
        suffix += f"_{robust_mode}"
    if sampling != "shuffle":
        suffix += f"_{sampling}"
    if reference != "none":
        suffix += f"_{reference}"
    if args.max_real_comments is not None:
        suffix += f"_real{config.data.max_real_comments}"
    if args.seed is not None:
        suffix += f"_seed{config.seed}"
    dataset_name = str(config.data.get("dataset_name", "checked"))
    checkpoint_path = portable_checkpoint_path(
        checkpoint_dir, f"{dataset_name}_mfcl_{mode}_{graph_structure}_{pooling_mode}{suffix}.pt"
    )
    print(f"mode={mode} graph={graph_structure} pooling={pooling_mode} message_passing={message_passing} decision={decision_mode} "
          f"seed={config.seed} train={len(train_set)} validate={len(val_set)} "
          f"real_node_selection={config.data.get('real_node_selection', 'prefix')} "
          f"max_generated_comments={config.data.max_generated_comments} "
          f"semantic_parent_gate={bool(config.graph.get('semantic_parent_gate', False))} "
          f"generated_reliability={generated_reliability} "
          f"base_loss_weight={base_loss_weight} robustness={robust_mode} sampling={sampling} "
          f"reference={reference} mm_fusion={mm_fusion_mode} evidence={evidence_mode}", flush=True)
    best_f1, stale = -1.0, 0
    history = []

    for epoch in range(1, int(config.training.epochs) + 1):
        model.train()
        total_loss, seen, final_loss_sum, base_loss_sum, fusion_aux_sum = 0.0, 0.0, 0.0, 0.0, 0.0
        mismatch_sum, mismatch_count = 0.0, 0
        sampled_counts = Counter()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=use_amp):
                output = model(batch, return_generated_context=mismatch_weight > 0)
                y = batch.y.view(-1)
                loss, final_ce, base_ce = classification_objective(output, y, training_weights, base_loss_weight)
                fusion_aux = loss.new_zeros(())
                if mm_fusion_mode == "mimoe":
                    mimoe = output["fusion_output"]
                    single_modal = .5 * (
                        F.cross_entropy(mimoe["text_unimodal_logits"], y, weight=training_weights) +
                        F.cross_entropy(mimoe["image_unimodal_logits"], y, weight=training_weights)
                    )
                    # Matches the public MIMoE objective: main CE + mean
                    # unimodal CE + beta * interaction routing loss.
                    beta = float(config.get("mimoe", {}).get("routing_beta", .7))
                    fusion_aux = single_modal + beta * mimoe["routing_loss"]
                    loss = loss + fusion_aux
                loss, mismatch_loss, count = add_mismatch_loss(model, output, loss, mismatch_rng, mismatch_weight)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            sampled_counts.update(y.detach().cpu().tolist())
            denominator = training_weights[y].sum().item() if training_weights is not None else y.numel()
            total_loss += loss.detach().item() * denominator
            final_loss_sum += final_ce.detach().item() * denominator
            base_loss_sum += base_ce.detach().item() * denominator
            fusion_aux_sum += fusion_aux.detach().item() * denominator
            seen += denominator
            mismatch_sum += float(mismatch_loss.detach()) * count
            mismatch_count += count
        val_metrics = evaluate(model, val_loader, device, use_amp, weights, args.audit_mismatch, audit_seed)
        history.append({"epoch": epoch, "train_total_loss": total_loss / max(seen, 1),
                        "sampled_training_class_counts": dict(sampled_counts),
                        "train_final_ce": final_loss_sum / max(seen, 1),
                        "train_base_ce": base_loss_sum / max(seen, 1),
                        "train_mimoe_auxiliary": fusion_aux_sum / max(seen, 1),
                        "train_mismatch_kl": mismatch_sum / max(mismatch_count, 1),
                        "validation_loss": val_metrics["loss"],
                        "validation_base_loss": val_metrics["base_loss"],
                        "validation_macro_f1": val_metrics["macro_f1"],
                        "validation_global_macro_f1": val_metrics["global_macro_f1"]})
        print(
            f"epoch={epoch} train_loss={total_loss / max(seen, 1):.6f} "
            f"train_final_ce={final_loss_sum / max(seen, 1):.6f} "
            f"train_base_ce={base_loss_sum / max(seen, 1):.6f} "
            f"mimoe_aux={fusion_aux_sum / max(seen, 1):.6f} "
            f"mismatch_kl={mismatch_sum / max(mismatch_count, 1):.6f} "
            f"val_loss={val_metrics['loss']:.6f} val_macro_f1={val_metrics['macro_f1']:.4f}"
        )
        if val_metrics["macro_f1"] > best_f1:
            best_f1, stale = val_metrics["macro_f1"], 0
            save_checkpoint_atomic(
                {
                    "epoch": epoch,
                    "mode": mode,
                    "graph_structure": graph_structure,
                    "pooling_mode": pooling_mode,
                    "message_passing": message_passing,
                    "generated_reliability": generated_reliability,
                    "decision_mode": decision_mode,
                    "seed": int(config.seed), "sampling": sampling,
                    "base_loss_weight": base_loss_weight,
                    "robustness_mode": robust_mode,
                    "correction_reference": reference,
                    "mm_fusion": mm_fusion_mode,
                    "evidence_mode": evidence_mode,
                    "mismatch_weight": mismatch_weight,
                    "config": config.toDict(),
                    "model": model.state_dict(),
                    "validation_metrics": val_metrics,
                    "label_mapping": dict(config.get("classification", {}).get("label_mapping", {"real": 0, "fake": 1})),
                },
                checkpoint_path,
            )
        else:
            stale += 1
            if stale >= int(config.training.patience):
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    summary = {
        "mode": mode,
        "dataset": dataset_name,
        "num_classes": num_classes,
        "graph_structure": graph_structure,
        "pooling_mode": pooling_mode,
        "message_passing": message_passing,
        "generated_reliability": generated_reliability,
        "decision_mode": decision_mode,
        "seed": int(config.seed),
        "sampling": sampling,
        "train_size": len(train_set), "validate_size": len(val_set),
        "paired_generated_coverage": paired_coverage,
        "coverage_sizes": {split: len(ids) for split, ids in allowed.items()},
        "generated_sources": ({split: {
            "path": str(resolve_path(config_path, config.data.generated_comments[split])),
            "sha256": hashlib.sha256(resolve_path(config_path, config.data.generated_comments[split]).read_bytes()).hexdigest(),
        } for split in ("train", "validate")}
            if dataset_name == "fakeddit_structure" else None),
        "training_class_counts": dict(Counter(train_set.labels())),
        "real_node_selection": str(config.data.get("real_node_selection", "prefix")),
        "max_real_comments": int(config.data.max_real_comments),
        "max_generated_comments": int(config.data.max_generated_comments),
        "validation_class_counts": dict(Counter(val_set.labels())),
        "training_ce_weights": training_weights.cpu().tolist() if training_weights is not None else None,
        "evaluation_ce_weights": weights.cpu().tolist() if weights is not None else None,
        "base_loss_weight": base_loss_weight,
        "robustness_mode": robust_mode,
        "correction_reference": reference,
        "mm_fusion": mm_fusion_mode,
        "evidence_mode": evidence_mode,
        "mismatch_weight": mismatch_weight,
        "training_history": history,
        "best_validation_macro_f1": best_f1,
        "best_epoch": int(checkpoint["epoch"]),
        "validation": checkpoint["validation_metrics"],
        "checkpoint": str(checkpoint_path),
    }
    if args.run_test:
        model.load_state_dict(checkpoint["model"])
        summary["test"] = evaluate(model, make_loader(test_set, config, False), device, use_amp, weights, args.audit_mismatch, audit_seed)
    if args.summary_json:
        summary_path = args.summary_json.resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    console_summary = dict(summary)
    console_summary.pop("training_history", None)
    for split in ("validation", "test"):
        if split in console_summary:
            console_summary[split] = {
                key: value for key, value in console_summary[split].items()
                if key != "prediction_records"
            }
            if "mismatch_audit" in console_summary[split]:
                console_summary[split]["mismatch_audit"] = {
                    k: v for k, v in console_summary[split]["mismatch_audit"].items() if k != "records"
                }
    print(json.dumps(console_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
