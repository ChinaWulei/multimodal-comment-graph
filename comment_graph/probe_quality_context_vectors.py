"""Predeclared frozen-context + proxy pilot; all preprocessing fits inside folds.

PCA dimension 16, C=.2, balanced logistic regression, five folds seed42.
Uses AI weak labels and no reserved audit / detector-validation/test labels.
Only saves a candidate if both original quality gates pass; audit is still needed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from comment_graph.score_comment_claims import digest, input_digest
from comment_graph.train_generated_quality_gate import CONTENT_FEATURES, PARENT_FEATURES, fit_head, load_data
from comment_graph.validate_generated_quality_annotations import load_unique


def make_context_model(seed=42):
    # PCA and all scalers are fit on each fold's training rows, never held-out rows.
    preprocess = ColumnTransformer([
        ("proxy", StandardScaler(), slice(0, len(CONTENT_FEATURES))),
        ("context", make_pipeline(Normalizer(), PCA(n_components=16, svd_solver="full"),
                                   StandardScaler()), slice(len(CONTENT_FEATURES), None)),
    ])
    return make_pipeline(preprocess, LogisticRegression(C=.2, class_weight="balanced",
                                                        max_iter=1000, random_state=seed))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-dir", type=Path, required=True)
    parser.add_argument("--vectors-jsonl", type=Path, required=True)
    parser.add_argument("--vectors-metadata", type=Path, required=True)
    parser.add_argument("--features-jsonl", type=Path, default=ROOT / "comment_graph/reports/generated_quality_parent_merged_train.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Preserve previous results")
    reference_path = args.fit_dir / "reference_fit.jsonl"
    annotation_path = args.fit_dir / "annotations_ai.jsonl"
    annotations, features, feature_hash = load_data(reference_path, annotation_path, args.features_jsonl)
    if any(r.get("annotation_source") != "ai" or r.get("human_reviewed") is not False for r in annotations):
        raise ValueError("This entry is explicitly AI-supervised")
    vectors = load_unique(args.vectors_jsonl)
    meta = json.loads(args.vectors_metadata.read_text(encoding="utf-8"))
    if (not meta["complete"] or meta["protocol"]["task"] != "context_vector" or
            meta["protocol"]["queue_sha256"] != digest(reference_path) or
            set(vectors) != {r["annotation_id"] for r in annotations}):
        raise ValueError("Incomplete or mismatched context vectors")
    fields = CONTENT_FEATURES + tuple(f"context_{i}" for i in range(2048))
    merged = []
    for row, feature in zip(annotations, features):
        vector = vectors[row["annotation_id"]]
        if (vector["news_id"] != row["news_id"] or vector["official_split"] != "train" or
                not vector["valid"] or vector["input_sha256"] != input_digest(row) or
                vector["image_sha256"] != digest(Path(row["image_path"]))):
            raise ValueError("Vector input provenance mismatch")
        x = np.asarray(vector["context_vector"], dtype=float)
        if x.shape != (2048,) or not np.isfinite(x).all():
            raise ValueError("Invalid vector dimensions or values")
        merged.append({**feature, **{f"context_{i}": float(value) for i, value in enumerate(x)}})
    report = {"label_source": "AI weak labels", "human_annotations": 0, "ai_annotations": len(annotations),
              "reference_sha256": digest(reference_path), "annotation_sha256": digest(annotation_path),
              "proxy_features_sha256": feature_hash, "vectors_sha256": digest(args.vectors_jsonl),
              "vectors_protocol": meta["protocol"], "seed": 42, "pca_dimension": 16,
              "regularization_C": .2, "preprocessing_within_fold": True,
              "reserved_audit_used": False, "news_labels_used": False,
              "warning": "Exploratory training-set reuse; AI-label agreement, not independent quality or classification improvement."}
    models = {}
    for name, names in (("content", fields), ("parent", PARENT_FEATURES)):
        diagnostics = {}
        try:
            models[name], _ = fit_head(annotations, merged, name, names, 42,
                                      label_source="ai", diagnostics=diagnostics,
                                      model_factory=make_context_model if name == "content" else None)
        except ValueError as error:
            diagnostics["failure_reason"] = str(error)
        diagnostics.pop("features", None)
        report[name] = diagnostics
    passed = all(report[name]["passed"] for name in ("content", "parent"))
    report["status"] = "passed_needs_reserved_audit" if passed else "failed"
    report["candidate_saved"] = passed
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if passed:
        joblib.dump({"content_model": models["content"], "parent_model": models["parent"],
                     "content_features": fields, "parent_features": PARENT_FEATURES,
                     "content_threshold": report["content"]["threshold"],
                     "parent_threshold": report["parent"]["threshold"],
                     "annotation_source": "ai", "requires_reserved_audit": True,
                     "provenance": report}, args.output_dir / "candidate.joblib")
    (args.output_dir / "fit_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], **{name: {k: v for k, v in report[name].items()
                      if k != "oof_predictions"} for name in ("content", "parent")}}, indent=2))


if __name__ == "__main__":
    main()
