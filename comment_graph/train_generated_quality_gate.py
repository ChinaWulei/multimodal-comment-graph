"""Fit conservative comment-content and parent-fit gates from human labels.

No news veracity labels, classifier validation labels, or synthetic/pseudo
quality labels are accepted. The model is not fit unless human annotations
provide enough positive and negative examples for grouped cross-validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from comment_graph.validate_generated_quality_annotations import (  # noqa: E402
    IMMUTABLE, PARENT_FIT, RELEVANCE, load_unique,
)

CONTENT_FEATURES = (
    "reranker_news_comment", "reranker_news_parent_comment",
    "clip_image_comment_cosine", "clip_image_title_cosine",
)
PARENT_FEATURES = (
    "reranker_parent_comment", "reranker_news_parent_comment",
    "reranker_news_comment",
    "reranker_chosen_vs_best_alternative_margin",
    "reranker_chosen_parent_rank_fraction",
)


def load_data(reference_path: Path, annotation_path: Path,
              feature_path: Path) -> tuple[list[dict], list[dict], str]:
    reference = load_unique(reference_path)
    annotations = load_unique(annotation_path)
    features = load_unique(feature_path)
    if set(reference) != set(annotations) or not set(reference) <= set(features):
        raise ValueError("Reference and annotation IDs must match; every annotated ID needs features")
    if len({row["news_id"] for row in reference.values()}) != len(reference):
        raise ValueError("Require one annotated generated comment per news for news-disjoint CV")
    ordered_annotations, ordered_features = [], []
    for key, source in reference.items():
        annotation, feature = annotations[key], features[key]
        if source["official_split"] != "train" or feature["official_split"] != "train":
            raise ValueError(f"Non-training data: {key}")
        for field in IMMUTABLE:
            if annotation.get(field) != source[field]:
                raise ValueError(f"Annotation changed source data: {key}/{field}")
        if feature["news_id"] != source["news_id"]:
            raise ValueError(f"Feature/news mismatch: {key}")
        if annotation.get("content_relevance") not in RELEVANCE | {None}:
            raise ValueError(f"Invalid content label: {key}")
        if annotation.get("parent_fit") not in PARENT_FIT | {None}:
            raise ValueError(f"Invalid parent label: {key}")
        if annotation.get("unsupported_specific_claim") is not None and not isinstance(
                annotation["unsupported_specific_claim"], bool):
            raise ValueError(f"Invalid unsupported-claim label: {key}")
        for name in set(CONTENT_FEATURES) | set(PARENT_FEATURES):
            value = float(feature[name])
            if not np.isfinite(value):
                raise ValueError(f"Nonfinite feature: {key}/{name}")
        ordered_annotations.append(annotation)
        ordered_features.append(feature)
    return ordered_annotations, ordered_features, hashlib.sha256(feature_path.read_bytes()).hexdigest()


def precision_threshold(scores: np.ndarray, labels: np.ndarray,
                        minimum_precision: float = 0.8,
                        minimum_recall: float = 0.25,
                        minimum_accepted: int = 5) -> dict:
    positives = int(labels.sum())
    options = []
    for threshold in sorted(set(map(float, scores))):
        chosen = scores >= threshold
        accepted = int(chosen.sum())
        if accepted < minimum_accepted:
            continue
        true_positive = int(labels[chosen].sum())
        precision = true_positive / accepted
        recall = true_positive / positives
        if precision >= minimum_precision and recall >= minimum_recall:
            options.append((recall, precision, -threshold, threshold, accepted))
    if not options:
        raise ValueError("No out-of-fold threshold meets the predeclared precision/recall/coverage gate")
    recall, precision, _, threshold, accepted = max(options)
    return {"threshold": threshold, "oof_precision": precision,
            "oof_recall": recall, "oof_accepted": accepted,
            "requested_min_precision": minimum_precision,
            "requested_min_recall": minimum_recall,
            "requested_min_accepted": minimum_accepted}


def fit_head(annotations: list[dict], features: list[dict], name: str,
             feature_names: tuple[str, ...], seed: int, *,
             label_source: str = "human", diagnostics: dict | None = None,
             model_factory=None) -> tuple[object, dict]:
    if label_source not in {"human", "ai"}:
        raise ValueError("Unknown quality annotation source")
    if name == "content":
        # A concrete unsupported claim is unusable even when the topic fits.
        # Ambiguous partial relevance and unreviewed claim status are not
        # promoted into clean positives.
        selected = []
        for i, row in enumerate(annotations):
            if row["content_relevance"] == "off_topic" or row["unsupported_specific_claim"] is True:
                selected.append((i, 0))
            elif row["content_relevance"] == "relevant" and row["unsupported_specific_claim"] is False:
                selected.append((i, 1))
    elif name == "parent":
        selected = [(i, int(row["parent_fit"] == "good"))
                    for i, row in enumerate(annotations)
                    if row["parent_fit"] in {"good", "wrong"}]
    else:
        raise ValueError(name)
    labels = np.asarray([label for _, label in selected], dtype=int)
    report = diagnostics if diagnostics is not None else {}
    report.update({"annotation_source": label_source,
                   f"{label_source}_labeled": len(labels),
                   "positive": int(labels.sum()),
                   "negative": int((1 - labels).sum()),
                   "excluded_ambiguous": len(annotations) - len(labels),
                   "features": feature_names, "passed": False})
    if len(labels) < 20 or min(int(labels.sum()), int((1 - labels).sum())) < 5:
        raise ValueError(f"Insufficient {label_source}-labeled examples for {name}: "
                         f"n={len(labels)}, positive={int(labels.sum())}, negative={int((1-labels).sum())}")
    x = np.asarray([[float(features[i][field]) for field in feature_names]
                    for i, _ in selected], dtype=float)
    splits = min(5, int(labels.sum()), int((1 - labels).sum()))
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
    oof = np.empty(len(labels), dtype=float)
    def new_model():
        if model_factory is not None:
            return model_factory()
        return make_pipeline(StandardScaler(), LogisticRegression(
            C=0.2, class_weight="balanced", max_iter=1000, random_state=seed))
    for train_idx, test_idx in cv.split(x, labels):
        model = new_model()
        model.fit(x[train_idx], labels[train_idx])
        oof[test_idx] = model.predict_proba(x[test_idx])[:, 1]
    auc = roc_auc_score(labels, oof)
    ap = average_precision_score(labels, oof)
    report.update({"folds": splits, "oof_roc_auc": auc,
                   "oof_average_precision": ap,
                   "oof_predictions": [
                       {"annotation_id": annotations[i].get("annotation_id"),
                        "news_id": annotations[i].get("news_id"),
                        "target": int(label), "probability": float(probability)}
                       for (i, label), probability in zip(selected, oof)]})
    if auc <= 0.60:
        raise ValueError(f"{name} gate failed news-disjoint CV discrimination: AUC={auc:.4f}")
    threshold = precision_threshold(oof, labels)
    final_model = new_model()
    final_model.fit(x, labels)
    report.update({"passed": True, **threshold})
    return final_model, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-jsonl", type=Path, required=True)
    parser.add_argument("--annotations-jsonl", type=Path, required=True)
    parser.add_argument("--features-jsonl", type=Path, required=True)
    parser.add_argument("--features-metadata-json", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    annotations, features, feature_hash = load_data(
        args.reference_jsonl, args.annotations_jsonl, args.features_jsonl)
    if any(row.get("annotation_source", "human") != "human" for row in annotations):
        raise ValueError("This entry requires human labels; use train_ai_quality_gate.py for AI labels")
    feature_meta = json.loads(args.features_metadata_json.read_text(encoding="utf-8"))
    if (feature_meta["split"] != "train" or feature_meta["rows"] < len(features) or
            feature_meta["rows"] != len(load_unique(args.features_jsonl))):
        raise ValueError("Feature metadata does not describe the complete training feature file")
    source_models = {name: feature_meta[name] for name in
                     ("reranker_model", "reranker_snapshot", "clip_model", "clip_snapshot",
                      "candidate_max_length")}
    if not feature_meta.get("candidate_source_graph_sha256"):
        raise ValueError("Merged candidate-score provenance is required for parent-fit training")
    content_model, content_report = fit_head(annotations, features, "content", CONTENT_FEATURES, args.seed)
    parent_model, parent_report = fit_head(annotations, features, "parent", PARENT_FEATURES, args.seed)
    package = {"content_model": content_model, "parent_model": parent_model,
               "content_features": CONTENT_FEATURES, "parent_features": PARENT_FEATURES,
               "content_threshold": content_report["threshold"],
               "parent_threshold": parent_report["threshold"],
               "feature_sha256": feature_hash, "source_models": source_models,
               "training_graph_sha256": feature_meta["candidate_source_graph_sha256"],
               "seed": args.seed}
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(package, args.output_model)
    report = {"split": "official_train", "news_labels_used": False,
              "quality_labels": "human only", "news_disjoint_cross_validation": True,
              "seed": args.seed, "feature_sha256": feature_hash,
              "source_models": source_models,
              "training_graph_sha256": feature_meta["candidate_source_graph_sha256"],
              "content": content_report, "parent": parent_report,
              "warning": "60-case pilot: CV thresholds are uncertain; no generated placement ground truth or independent classification test is established."}
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"model": str(args.output_model), "report": str(args.output_report),
                      "content_oof_auc": content_report["oof_roc_auc"],
                      "parent_oof_auc": parent_report["oof_roc_auc"]}, indent=2))


if __name__ == "__main__":
    main()
