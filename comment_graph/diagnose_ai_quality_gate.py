"""Post-hoc fitting-set error analysis, not new threshold selection or evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-dir", type=Path, required=True)
    args = parser.parse_args()
    report_path = args.fit_dir / "fit_report.json"
    annotation_path = args.fit_dir / "annotations_ai.jsonl"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    annotations = {r["annotation_id"]: r for r in
                   map(json.loads, annotation_path.read_text(encoding="utf-8").splitlines())}
    groups = {}
    predictions = report["content"]["oof_predictions"]
    for prediction in predictions:
        row = annotations[prediction["annotation_id"]]
        group = ("clean_relevant" if prediction["target"] == 1 else
                 "unsupported_claim" if row["unsupported_specific_claim"] is True else "off_topic")
        groups.setdefault(group, []).append(prediction["probability"])
    result = {"post_hoc_diagnostic_only": True, "threshold_changed": False,
              "reserved_audit_used": False, "official_validation_test_used": False,
              "label_source": "AI weak labels, not human ground truth",
              "fit_report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
              "groups": {name: {"n": len(values), "mean_oof_score": float(np.mean(values)),
                                 "median_oof_score": float(np.median(values))}
                         for name, values in groups.items()},
              "pairwise_auc": {}}
    positive = groups.get("clean_relevant", [])
    for name in ("unsupported_claim", "off_topic"):
        negative = groups.get(name, [])
        if positive and negative:
            result["pairwise_auc"]["clean_relevant_vs_" + name] = float(roc_auc_score(
                [1] * len(positive) + [0] * len(negative), positive + negative))
    path = args.fit_dir / "content_error_analysis.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
