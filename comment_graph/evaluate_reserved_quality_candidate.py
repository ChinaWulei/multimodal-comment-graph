"""One-shot reserved AI-quality audit of a frozen candidate; never refits thresholds."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from comment_graph.train_quality_lora_cv import SYSTEM, content_target, request, sha
from comment_graph.train_generated_quality_gate import load_data


def summarize(rows, task, criteria):
    from sklearn.metrics import average_precision_score, roc_auc_score
    known = [r for r in rows if r[task + "_target"] is not None]
    y = [r[task + "_target"] for r in known]
    p = [r[task + "_probability"] for r in known]
    accepted = [r for r in known if r[task + "_accepted"]]
    positive = sum(y)
    correct = sum(r[task + "_target"] for r in accepted)
    precision = correct/len(accepted) if accepted else None
    recall = correct/positive if positive else None
    auc = float(roc_auc_score(y, p)) if len(set(y)) == 2 else None
    checks = {"auc": auc is not None and auc > criteria["auc_strictly_above"],
              "precision": precision is not None and precision >= criteria["precision_at_least"],
              "recall": recall is not None and recall >= criteria["recall_at_least"],
              "coverage": len(accepted) >= criteria["accepted_at_least"]}
    return {"known_targets": len(known), "positive": positive, "negative": len(known)-positive,
            "ambiguous_targets": len(rows)-len(known),
            "accepted_total": sum(r[task + "_accepted"] for r in rows), "accepted_known": len(accepted),
            "accepted_ambiguous": sum(r[task + "_accepted"] for r in rows if r[task + "_target"] is None),
            "true_positive": correct, "false_positive": len(accepted)-correct,
            "false_negative": positive-correct, "precision": precision, "recall": recall,
            "roc_auc": auc, "average_precision": float(average_precision_score(y, p)) if positive else None,
            "checks": checks, "passed": all(checks.values()),
            "coverage_inconclusive": not checks["coverage"]}


def main():
    import joblib
    import numpy as np
    import torch
    from PIL import Image, ImageOps
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, default=ROOT / "comment_graph/outputs/quality_candidate_frozen_train140")
    parser.add_argument("--queue", type=Path, default=ROOT / "comment_graph/reports/quality_expansion_20261008/reserved_audit.jsonl")
    parser.add_argument("--annotations", type=Path, default=ROOT / "comment_graph/reports/quality_expansion_20261008/reserved_audit_ai.jsonl")
    parser.add_argument("--features", type=Path, default=ROOT / "comment_graph/reports/generated_quality_parent_merged_train.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("One-shot audit; preserve prior scores and do not refit on them")
    candidate_path = args.candidate_dir / "candidate_manifest.json"
    frozen = json.loads(candidate_path.read_text(encoding="utf-8"))
    if frozen["status"] != "frozen_pending_reserved_audit" or sha(args.queue) != frozen["reserved_queue_sha256"]:
        raise ValueError("Audit queue or candidate status differs from the frozen plan")
    adapter = Path(frozen["content_adapter"])
    for name, checksum in frozen["content_adapter_file_hashes"].items():
        if sha(adapter / name) != checksum:
            raise ValueError(f"Frozen adapter changed: {name}")
    parent_path = args.candidate_dir / "parent_model.joblib"
    if sha(parent_path) != frozen["parent_model_sha256"]:
        raise ValueError("Frozen parent model changed")
    final_report_path = adapter.parent / "report.json"
    if sha(final_report_path) != frozen["content_training_report_sha256"]:
        raise ValueError("Final training report changed")
    training = json.loads(final_report_path.read_text(encoding="utf-8"))
    protocol = training["protocol"]
    annotations, features, feature_hash = load_data(args.queue, args.annotations, args.features)
    if (len(annotations) != frozen["reserved_rows"] or
            {r["news_id"] for r in annotations} != set(frozen["reserved_news"]) or
            any(r.get("annotation_source") != "ai" or r.get("human_reviewed") is not False or
                r.get("model_scores_seen") is not False for r in annotations)):
        raise ValueError("Expected the complete score-blinded reserved AI review")
    train_news = {n for f in training["fold_reports"] for n in f["train_news"]}
    if train_news & {r["news_id"] for r in annotations}:
        raise ValueError("Quality training/audit news overlap")
    original = json.loads((ROOT / "comment_graph/outputs/ai_quality_gate_train140/fit_report.json").read_text(encoding="utf-8"))
    if feature_hash != original["feature_sha256"]:
        raise ValueError("Proxy feature cache differs from the fitting cache")
    parent = joblib.load(parent_path)
    if parent["annotation_source"] != "ai" or tuple(parent["features"]) != tuple(frozen["parent_features"]):
        raise ValueError("Parent model feature contract differs")
    x = np.asarray([[f[name] for name in parent["features"]] for f in features], dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite proxy features")
    # Annotation snapshot is fixed on disk before any model scores are produced.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    review_lock = {"annotations_sha256": sha(args.annotations), "candidate_sha256": sha(candidate_path),
                   "queue_sha256": sha(args.queue), "feature_sha256": feature_hash,
                   "rows": len(annotations), "labels_frozen_before_scoring": True,
                   "thresholds_refit": False, "annotation_source": "ai", "human_annotations": 0}
    (args.output_dir / "review_lock.json").write_text(json.dumps(review_lock, indent=2), encoding="utf-8")
    parent_scores = parent["model"].predict_proba(x)[:, 1]
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Requires BF16 CUDA; no CPU fallback")
    model_path = Path(protocol["model_path"])
    if sha(model_path / "config.json") != protocol["model_config_sha256"] or SYSTEM != protocol["system_prompt"]:
        raise ValueError("Base model configuration or prompt changed")
    processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True,
                                             min_pixels=3136, max_pixels=protocol["max_pixels"])
    choice_ids = [processor.tokenizer.encode(word, add_special_tokens=False) for word in ("No", "Yes")]
    if any(len(ids) != 1 for ids in choice_ids):
        raise ValueError("Unexpected decision tokenization")
    token_ids = [ids[0] for ids in choice_ids]
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_path), local_files_only=True, dtype=torch.bfloat16,
        device_map={"": 0}, attn_implementation="sdpa")
    model = PeftModel.from_pretrained(base, str(adapter), is_trainable=False).eval()
    model.requires_grad_(False)
    inner = model.get_base_model()
    predictions = []
    start = time.perf_counter()
    with (args.output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row, parent_score in zip(annotations, parent_scores):
            with Image.open(row["image_path"]) as raw:
                picture = ImageOps.exif_transpose(raw).convert("RGB").copy()
            messages = [{"role": "system", "content": SYSTEM},
                        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": request(row)}]}]
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[prompt], images=[picture], return_tensors="pt").to("cuda")
            if inputs["input_ids"].shape[1] > protocol["max_input_tokens"] or not inputs["pixel_values"].numel():
                raise ValueError("Image missing or input exceeds budget")
            with torch.inference_mode():
                hidden = inner.model(**inputs, use_cache=False, return_dict=True).last_hidden_state[:, -1].float()
                logits = torch.nn.functional.linear(hidden, inner.lm_head.weight[token_ids].float())
                content_score = float(torch.softmax(logits, -1)[0, 1])
            if not np.isfinite([content_score, parent_score]).all():
                raise ValueError("Nonfinite quality score")
            c_target = content_target(row)
            p_target = 1 if row["parent_fit"] == "good" else 0 if row["parent_fit"] == "wrong" else None
            combined_target = (0 if c_target == 0 or p_target == 0 else
                               1 if c_target == 1 and p_target == 1 else None)
            c_accept = content_score >= frozen["content_threshold"]
            p_accept = float(parent_score) >= frozen["parent_threshold"]
            prediction = {"annotation_id": row["annotation_id"], "news_id": row["news_id"],
                          "content_probability": content_score, "parent_probability": float(parent_score),
                          "content_accepted": c_accept, "parent_accepted": p_accept,
                          "combined_accepted": c_accept and p_accept,
                          "combined_probability": content_score * float(parent_score),
                          "content_target": c_target, "parent_target": p_target, "combined_target": combined_target,
                          "image_sha256": sha(row["image_path"])}
            handle.write(json.dumps(prediction) + "\n")
            handle.flush()
            predictions.append(prediction)
    if sha(args.annotations) != review_lock["annotations_sha256"]:
        raise ValueError("Annotations changed during scoring")
    criteria = frozen["audit_protocol"]["per_head"]
    result = {"protocol": review_lock, "content_threshold": frozen["content_threshold"],
              "parent_threshold": frozen["parent_threshold"], "criteria": criteria,
              **{task: summarize(predictions, task, criteria) for task in ("content", "parent", "combined")},
              "runtime_seconds": time.perf_counter()-start,
              "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated()/1024**3,
              "warning": "Held-out from quality fitting, but AI weak labels, small sample, same reviewer and training-domain news. Not human ground truth or fake-news test performance."}
    result["per_head_audit_passed"] = result["content"]["passed"] and result["parent"]["passed"]
    result["graph_filter_enabled"] = False
    result["status"] = "passed_needs_downstream_validation" if result["per_head_audit_passed"] else "not_qualified"
    (args.output_dir / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
