"""User-adopted AI-label pilot; shares the human pilot's fixed CV criteria.

The 60 judgments come from an AI review, not independent human annotators.
No quality criterion is relaxed if these weak labels fail to support a gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from comment_graph.train_generated_quality_gate import (
    CONTENT_FEATURES, PARENT_FEATURES, fit_head, load_data,
)
from comment_graph.validate_generated_quality_annotations import load_unique

# Binds news-ID-only AI judgments to the actual material reviewed on 2026-10-08.
REVIEWED_REFERENCE_SHA256 = "dca70cc79a3c8ee8ba996afd68fd21b2c832bbf5b515d1dae4ff3b866fc72109"
REVIEWED_EXPANSION_SHA256 = "2ffaa80f2eb9c1ff786f92a0e727b30bbbcba81ab417bb9ef4b3f7b323a47977"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_expansion(directory: Path, previous: list[dict], feature_path: Path):
    """Bind the extra AI judgments to reviewed material, never read audit cases."""
    manifest_path = directory / "manifest.json"
    queue_path = directory / "fit.jsonl"
    draft_path = directory / "fit_ai_draft.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (sha256(queue_path) != REVIEWED_EXPANSION_SHA256 or
            manifest["queue_sha256"]["fit"] != REVIEWED_EXPANSION_SHA256 or
            manifest["source_sha256"]["previous_queue"] != REVIEWED_REFERENCE_SHA256 or
            manifest["source_sha256"]["features"] != sha256(feature_path)):
        raise ValueError("Expansion source hashes differ from the reviewed experiment")
    reference, drafts = load_unique(queue_path), load_unique(draft_path)
    if len(reference) != 80 or set(reference) != set(drafts):
        raise ValueError("Expected all 80 reviewed fitting cases, each exactly once")
    old_news = {row["news_id"] for row in previous}
    new_news = {row["news_id"] for row in reference.values()}
    audit_news = set(manifest["reserved_audit_news_ids"])
    if (len(new_news) != 80 or new_news != set(manifest["fit_news_ids"]) or
            old_news & new_news or (old_news | new_news) & audit_news):
        raise ValueError("Fitting/previous/audit news are not disjoint")
    converted = []
    for key, source in reference.items():
        ai = drafts[key]
        if (source["official_split"] != "train" or source["quality_review_role"] != "fit" or
                ai["news_id"] != source["news_id"] or ai.get("annotation_source") != "ai" or
                ai.get("human_reviewed") is not False):
            raise ValueError(f"Invalid expansion provenance: {key}")
        row = dict(source)
        row.update({field: ai[field + "_ai"] for field in
                    ("content_relevance", "parent_fit", "unsupported_specific_claim")})
        row.update(annotation_source="ai", human_reviewed=False, user_adopted=True,
                   review_notes=ai["review_notes_ai"], needs_human_review=ai["needs_human_review"])
        converted.append(row)
    provenance = {"manifest_sha256": sha256(manifest_path),
                  "fit_queue_sha256": sha256(queue_path), "ai_draft_sha256": sha256(draft_path),
                  "new_ai_annotations": len(converted), "reserved_audit_news": len(audit_news),
                  "reserved_audit_content_read": False, "sampling": manifest["fit_sampling"],
                  "warning": "Fit enrichment favors news relevance; CV is not a population quality estimate."}
    return list(reference.values()), converted, provenance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    reports = ROOT / "comment_graph" / "reports"
    parser.add_argument("--reference-jsonl", type=Path, default=reports / "generated_quality_annotation_train60.jsonl")
    parser.add_argument("--ai-jsonl", type=Path, default=reports / "generated_quality_annotation_train60_ai_draft.jsonl")
    parser.add_argument("--features-jsonl", type=Path, default=reports / "generated_quality_parent_merged_train.jsonl")
    parser.add_argument("--features-metadata-json", type=Path, default=reports / "generated_quality_parent_merged_train_meta.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "comment_graph" / "outputs" / "ai_quality_gate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expansion-dir", type=Path,
                        help="Optional reviewed 80-case AI fitting expansion; excludes reserved audit")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Choose an empty output directory; existing experiments are preserved")
    if sha256(args.reference_jsonl) != REVIEWED_REFERENCE_SHA256:
        raise ValueError("Reference queue changed since the AI visual review")
    reference = load_unique(args.reference_jsonl)
    drafts = [json.loads(line) for line in args.ai_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(drafts) != len({row["news_id"] for row in drafts}):
        raise ValueError("Duplicate AI news IDs")
    draft_by_news = {row["news_id"]: row for row in drafts}
    if len(reference) != 60 or len({row["news_id"] for row in reference.values()}) != 60:
        raise ValueError("Expected the reviewed 60 distinct training news")
    if set(draft_by_news) != {row["news_id"] for row in reference.values()}:
        raise ValueError("AI review and source queue have different news IDs")
    converted = []
    for source in reference.values():
        ai = draft_by_news[source["news_id"]]
        row = dict(source)
        row.update({field: ai[field + "_ai"] for field in
                    ("content_relevance", "parent_fit", "unsupported_specific_claim")})
        row.update({"annotation_source": "ai", "human_reviewed": False,
                    "user_adopted": True, "review_notes": ai["review_notes_ai"],
                    "needs_human_review": ai["needs_human_review"]})
        converted.append(row)
    expansion = None
    combined_reference = list(reference.values())
    if args.expansion_dir is not None:
        extra_reference, extra_annotations, expansion = load_expansion(
            args.expansion_dir, converted, args.features_jsonl)
        combined_reference.extend(extra_reference)
        converted.extend(extra_annotations)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference_path = args.output_dir / "reference_fit.jsonl"
    reference_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                     for row in combined_reference), encoding="utf-8")
    converted_path = args.output_dir / "annotations_ai.jsonl"
    converted_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in converted), encoding="utf-8")
    annotations, features, feature_hash = load_data(reference_path, converted_path, args.features_jsonl)
    meta = json.loads(args.features_metadata_json.read_text(encoding="utf-8"))
    if meta["split"] != "train" or meta["rows"] != len(load_unique(args.features_jsonl)):
        raise ValueError("Feature metadata must describe the complete training feature file")
    if not meta.get("candidate_source_graph_sha256"):
        raise ValueError("Merged candidate-score provenance is missing")
    models = {name: meta[name] for name in
              ("reranker_model", "reranker_snapshot", "clip_model", "clip_snapshot", "candidate_max_length")}
    report = {
        "status": "checking", "split": "official_train", "quality_labels": "AI only (user adopted)",
        "human_annotations": 0, "ai_annotations": len(annotations), "news_labels_used": False,
        "seed": args.seed, "news_disjoint_cross_validation": True,
        "source_reference_sha256": sha256(args.reference_jsonl), "ai_annotation_sha256": sha256(args.ai_jsonl),
        "combined_reference_sha256": sha256(reference_path),
        "combined_annotation_sha256": sha256(converted_path), "expansion": expansion,
        "feature_sha256": feature_hash, "source_models": models,
        "training_graph_sha256": meta["candidate_source_graph_sha256"],
        "needs_human_review": sum(row["needs_human_review"] for row in annotations),
        "label_counts": {name: dict(Counter(str(row[name]) for row in annotations)) for name in
                         ("content_relevance", "parent_fit", "unsupported_specific_claim")},
        "criteria": {"minimum_examples": 20, "minimum_per_class": 5, "auc_strictly_above": .6,
                     "precision_at_least": .8, "recall_at_least": .25, "accepted_at_least": 5},
        "warning": "Agreement with AI weak labels only. OOF thresholds are selected on OOF predictions, not independently evaluated. No human-quality accuracy or news-classification improvement is established.",
    }
    fitted = {}
    for name, fields in (("content", CONTENT_FEATURES), ("parent", PARENT_FEATURES)):
        diagnostics = {}
        try:
            fitted[name], _ = fit_head(annotations, features, name, fields, args.seed,
                                       label_source="ai", diagnostics=diagnostics)
        except ValueError as error:
            diagnostics["failure_reason"] = str(error)
        report[name] = diagnostics
    passed = all(report[name].get("passed") for name in ("content", "parent"))
    report["status"] = "passed" if passed else "failed"
    report["gate_saved"] = passed
    if passed:
        package = {"content_model": fitted["content"], "parent_model": fitted["parent"],
                   "content_features": CONTENT_FEATURES, "parent_features": PARENT_FEATURES,
                   "content_threshold": report["content"]["threshold"], "parent_threshold": report["parent"]["threshold"],
                   "feature_sha256": feature_hash, "source_models": models,
                   "training_graph_sha256": meta["candidate_source_graph_sha256"], "seed": args.seed,
                   "quality_labels": report["quality_labels"], "annotation_source": "ai",
                   "ai_annotation_sha256": report["combined_annotation_sha256"],
                   "expansion": expansion}
        joblib.dump(package, args.output_dir / "gate.joblib")
    (args.output_dir / "fit_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"status": report["status"], "quality_labels": report["quality_labels"],
               "gate_saved": passed, "report": str(args.output_dir / "fit_report.json")}
    for name in ("content", "parent"):
        summary[name] = {key: value for key, value in report[name].items()
                         if key not in {"oof_predictions", "features"}}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
