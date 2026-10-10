"""Descriptive comparison of training-only quality experiments, not detector results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve prior summaries")
    outputs = ROOT / "comment_graph/outputs"
    reports = {"proxy_logistic": read(outputs / "ai_quality_gate_train140/fit_report.json")["content"],
               "frozen_context_logistic": read(outputs / "ai_quality_context_train140/fit_report.json")["content"],
               "quality_lora": read(outputs / "ai_quality_lora_cv_train140/report.json")}
    by_arm = {name: {p["annotation_id"]: p for p in report["oof_predictions"]}
              for name, report in reports.items()}
    keys = list(by_arm["proxy_logistic"])
    if any(set(data) != set(keys) for data in by_arm.values()):
        raise ValueError("Comparison samples differ")
    y = np.asarray([by_arm["proxy_logistic"][key]["target"] for key in keys])
    scores = {name: np.asarray([data[k]["probability"] for k in keys]) for name, data in by_arm.items()}
    if any(not np.array_equal(y, [data[k]["target"] for k in keys]) for data in by_arm.values()):
        raise ValueError("Comparison targets differ")
    summaries = {}
    for name, report in reports.items():
        options = []
        for threshold in np.unique(scores[name]):
            accepted = scores[name] >= threshold
            correct = int(y[accepted].sum())
            if accepted.sum() >= 5 and correct/y.sum() >= .25:
                options.append({"threshold": float(threshold), "accepted": int(accepted.sum()),
                                "correct": correct, "precision": correct/int(accepted.sum()),
                                "recall": correct/int(y.sum())})
        summaries[name] = {"oof_roc_auc": report["oof_roc_auc"],
                           "oof_average_precision": report["oof_average_precision"],
                           "passed": report["passed"],
                           "best_precision_given_original_coverage": max(options, key=lambda x: (x["precision"], x["recall"]))}
    # Paired, class-stratified resampling of these same news; descriptive only.
    # Shared fold models induce dependence, and development-set reuse remains.
    rng = np.random.default_rng(42)
    positives, negatives = np.flatnonzero(y), np.flatnonzero(1-y)
    delta = []
    for _ in range(2000):
        ids = np.concatenate([rng.choice(positives, len(positives), replace=True),
                              rng.choice(negatives, len(negatives), replace=True)])
        delta.append(roc_auc_score(y[ids], scores["quality_lora"][ids]) -
                     roc_auc_score(y[ids], scores["proxy_logistic"][ids]))
    annotations = {r["annotation_id"]: r for r in map(json.loads,
                   (outputs / "ai_quality_gate_train140/annotations_ai.jsonl").read_text(encoding="utf-8").splitlines())}
    false_positive_order = sorted((k for k in keys if by_arm["quality_lora"][k]["target"] == 0),
                                  key=lambda k: by_arm["quality_lora"][k]["probability"], reverse=True)[:10]
    errors = [{"annotation_id": k, "probability": by_arm["quality_lora"][k]["probability"],
               **{field: annotations[k][field] for field in ("headline", "parent_content", "generated_comment",
                  "content_relevance", "unsupported_specific_claim", "review_notes")}}
              for k in false_positive_order]
    result = {"scope": "training-only AI weak-label content quality development", "n": len(keys),
              "positive": int(y.sum()), "negative": int((1-y).sum()), "arms": summaries,
              "auc_delta_lora_minus_proxy": summaries["quality_lora"]["oof_roc_auc"]-summaries["proxy_logistic"]["oof_roc_auc"],
              "descriptive_paired_bootstrap_95_percent_interval": [float(v) for v in np.quantile(delta, [.025, .975])],
              "bootstrap_repetitions": 2000, "highest_scored_ai_negative_cases": errors,
              "reserved_audit_used": False, "news_detector_metrics": None,
              "warning": "Not confirmatory: correlated fold predictions, adaptive development-set reuse, AI-label noise and enriched fitting sample. No independent quality improvement or fake-news classification improvement is established."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "highest_scored_ai_negative_cases"}, indent=2))


if __name__ == "__main__":
    main()
