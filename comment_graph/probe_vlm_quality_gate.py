"""Fixed train-only comparison of proxy and frozen-VLM features against AI labels.

Exploratory re-use of the 60-case development set, not independent evaluation.
No classifier graph is modified and no deployable gate is published here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from comment_graph.train_generated_quality_gate import CONTENT_FEATURES, PARENT_FEATURES, fit_head, load_data
from comment_graph.validate_generated_quality_annotations import load_unique

VLM_CONTENT = ("vlm_relevance_log_odds", "vlm_unsupported_log_odds")
VLM_PARENT = ("vlm_parent_fit_log_odds",)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    reports = ROOT / "comment_graph/reports"
    parser.add_argument("--reference-jsonl", type=Path, default=reports / "generated_quality_annotation_train60.jsonl")
    parser.add_argument("--annotations-jsonl", type=Path, default=ROOT / "comment_graph/outputs/ai_quality_gate/annotations_ai.jsonl")
    parser.add_argument("--features-jsonl", type=Path, default=reports / "generated_quality_parent_merged_train.jsonl")
    parser.add_argument("--vlm-jsonl", type=Path, required=True)
    parser.add_argument("--vlm-metadata-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError("Preserve existing probe results")
    annotations, features, feature_hash = load_data(args.reference_jsonl, args.annotations_jsonl, args.features_jsonl)
    if any(row.get("annotation_source") != "ai" for row in annotations):
        raise ValueError("This experiment is explicitly AI-supervised")
    vlm = load_unique(args.vlm_jsonl)
    meta = json.loads(args.vlm_metadata_json.read_text(encoding="utf-8"))
    if (not meta["complete"] or meta["protocol"]["split"] != "train" or
            meta["protocol"]["queue_sha256"] != hashlib.sha256(args.reference_jsonl.read_bytes()).hexdigest() or
            set(vlm) != {row["annotation_id"] for row in annotations}):
        raise ValueError("VLM scores must cover exactly the original training review queue")
    merged = []
    for row, feature in zip(annotations, features):
        score = vlm[row["annotation_id"]]
        if score["news_id"] != row["news_id"] or score["official_split"] != "train":
            raise ValueError("VLM feature split/news mismatch")
        merged.append({**feature, **score})
    arms = {"proxy_only": (CONTENT_FEATURES, PARENT_FEATURES),
            "vlm_only": (VLM_CONTENT, VLM_PARENT),
            "proxy_plus_vlm": (CONTENT_FEATURES + VLM_CONTENT, PARENT_FEATURES + VLM_PARENT)}
    report = {"split": "official_train", "annotation_source": "ai", "seed": args.seed,
              "reference_sha256": hashlib.sha256(args.reference_jsonl.read_bytes()).hexdigest(),
              "annotation_sha256": hashlib.sha256(args.annotations_jsonl.read_bytes()).hexdigest(),
              "feature_sha256": feature_hash, "vlm_sha256": hashlib.sha256(args.vlm_jsonl.read_bytes()).hexdigest(),
              "vlm_protocol": meta["protocol"], "arms": {},
              "warning": "Development-set reuse with AI weak labels. CV/threshold selection is not independent validation and says nothing yet about news-classification gain."}
    for arm, fields_by_task in arms.items():
        results = {}
        for task, fields in zip(("content", "parent"), fields_by_task):
            diagnostics = {}
            try:
                fit_head(annotations, merged, task, fields, args.seed, label_source="ai", diagnostics=diagnostics)
            except ValueError as error:
                diagnostics["failure_reason"] = str(error)
            results[task] = diagnostics
        results["both_passed"] = all(results[task]["passed"] for task in ("content", "parent"))
        report["arms"][arm] = results
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {arm: {task: {key: value for key, value in results[task].items()
                           if key not in {"oof_predictions", "features"}}
                     for task in ("content", "parent")} for arm, results in report["arms"].items()}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
