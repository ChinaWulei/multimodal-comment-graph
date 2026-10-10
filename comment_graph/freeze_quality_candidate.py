"""Freeze the two candidate heads and thresholds before reading reserved cases."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from comment_graph.train_quality_lora_cv import sha
from comment_graph.train_generated_quality_gate import PARENT_FEATURES, fit_head, load_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Do not replace a frozen candidate")
    reports = ROOT / "comment_graph/reports"
    outputs = ROOT / "comment_graph/outputs"
    fitting = outputs / "ai_quality_gate_train140"
    final_dir = outputs / "ai_quality_lora_final_train140"
    selection_path = reports / "quality_expansion_20261008/lora_precision_audit.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    training = json.loads((final_dir / "report.json").read_text(encoding="utf-8"))
    previous = json.loads((fitting / "fit_report.json").read_text(encoding="utf-8"))
    if (not selection["fp32"]["passed"] or training["status"] != "final_trained_needs_reserved_audit" or
            training["protocol"]["selection_report_sha256"] != sha(selection_path)):
        raise ValueError("Content candidate has not completed the selected training")
    annotations, features, _ = load_data(fitting / "reference_fit.jsonl", fitting / "annotations_ai.jsonl",
                                        reports / "generated_quality_parent_merged_train.jsonl")
    model, parent_report = fit_head(annotations, features, "parent", PARENT_FEATURES, 42, label_source="ai")
    if parent_report["threshold"] != previous["parent"]["threshold"]:
        raise ValueError("Parent threshold changed from the original selection")
    manifest = json.loads((reports / "quality_expansion_20261008/manifest.json").read_text(encoding="utf-8"))
    if {r["news_id"] for r in annotations} & set(manifest["reserved_audit_news_ids"]):
        raise ValueError("Audit news leaked into fitting")
    adapter = final_dir / "final_adapter"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": PARENT_FEATURES, "annotation_source": "ai"},
                args.output_dir / "parent_model.joblib")
    frozen = {"status": "frozen_pending_reserved_audit", "annotation_source": "ai", "human_annotations": 0,
              "content_adapter": str(adapter), "content_inference": "FP32 projection of No/Yes vocabulary rows",
              "content_training_report_sha256": sha(final_dir / "report.json"),
              "content_adapter_file_hashes": {p.name: sha(p) for p in sorted(adapter.iterdir()) if p.is_file()},
              "parent_model_sha256": sha(args.output_dir / "parent_model.joblib"),
              "content_threshold": selection["fp32"]["threshold"],
              "parent_threshold": parent_report["threshold"], "parent_features": PARENT_FEATURES,
              "annotation_sha256": sha(fitting / "annotations_ai.jsonl"),
              "selection_sha256": sha(selection_path),
              "reserved_queue_sha256": manifest["queue_sha256"]["reserved_audit"],
              "reserved_news": manifest["reserved_audit_news_ids"],
              "reserved_rows": manifest["reserved_audit_news"], "reserved_contents_used": False,
              "audit_protocol": {"thresholds_refit": False, "weak_labels": "AI review blinded to model scores",
                                 "per_head": {"auc_strictly_above": .6, "precision_at_least": .8,
                                              "recall_at_least": .25, "accepted_at_least": 5},
                                 "uncertain_annotations": "exclude ambiguous targets, report their acceptance separately",
                                 "insufficient_coverage": "inconclusive; do not claim success from a tiny accepted set"},
              "warning": "This is not an enabled graph filter. Candidate must pass reserved quality audit, then downstream paired detector validation. AI review is not independent human ground truth."}
    (args.output_dir / "candidate_manifest.json").write_text(json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "parent_fit_report.json").write_text(json.dumps(parent_report, indent=2), encoding="utf-8")
    print(json.dumps({"frozen": True, "content_threshold": frozen["content_threshold"],
                      "parent_threshold": frozen["parent_threshold"], "audit_rows": frozen["reserved_rows"],
                      "graph_filter_enabled": False, "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
