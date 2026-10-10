"""Reserve disjoint official-train news for AI quality learning and later audit.

This selects unlabeled review material only. It does not generate annotations,
fit a model, expose detector truth labels, or evaluate the reserved audit cases.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    reports = ROOT / "comment_graph/reports"
    parser.add_argument("--contexts", type=Path, default=reports / "generated_quality_all_train_contexts.jsonl")
    parser.add_argument("--previous-queue", type=Path, default=reports / "generated_quality_annotation_train60.jsonl")
    parser.add_argument("--features", type=Path, default=reports / "generated_quality_parent_merged_train.jsonl")
    parser.add_argument("--output-dir", type=Path, default=reports / "quality_expansion_20261008")
    parser.add_argument("--fit-news", type=int, default=80)
    parser.add_argument("--audit-news", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20261008)
    args = parser.parse_args()
    if min(args.fit_news, args.audit_news) < 1:
        raise ValueError("Both cohorts must be nonempty")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Preserve the first allocation; select a fresh directory if explicitly redesigning")
    contexts = read_rows(args.contexts)
    previous = read_rows(args.previous_queue)
    features = read_rows(args.features)
    feature_by_id = {row["annotation_id"]: row for row in features}
    if len(feature_by_id) != len(features) or len({row["annotation_id"] for row in contexts}) != len(contexts):
        raise ValueError("Duplicate candidate or feature IDs")
    if {row["annotation_id"] for row in contexts} != set(feature_by_id):
        raise ValueError("Context/feature coverage differs")
    if any(row["official_split"] != "train" for row in contexts + previous + features):
        raise ValueError("Only official training news may enter this review expansion")
    if any(any(row.get(key) is not None for key in
               ("content_relevance", "parent_fit", "unsupported_specific_claim")) for row in contexts):
        raise ValueError("Selection must use unlabeled contexts")
    previous_news = {row["news_id"] for row in previous}
    if len(previous_news) != len(previous):
        raise ValueError("Previous queue must contain one case per news")
    by_news = defaultdict(list)
    for row in contexts:
        if feature_by_id[row["annotation_id"]]["news_id"] != row["news_id"]:
            raise ValueError("Feature/news mismatch")
        by_news[row["news_id"]].append(row)
    if not previous_news <= set(by_news):
        raise ValueError("Previous queue contains unknown news")
    eligible = sorted(set(by_news) - previous_news)
    if len(eligible) < args.fit_news + args.audit_news:
        raise ValueError("Insufficient disjoint news for requested fit/audit cohorts")
    rng = random.Random(args.seed)
    rng.shuffle(eligible)
    audit_ids = eligible[:args.audit_news]
    fit_ids = eligible[args.audit_news:args.audit_news + args.fit_news]
    audit = []
    for news_id in audit_ids:
        candidates = sorted(by_news[news_id], key=lambda row: row["annotation_id"])
        audit.append(dict(rng.choice(candidates), quality_review_role="reserved_audit",
                          sampling_policy="uniform_news_then_uniform_generated_node"))
    fit = []
    for index, news_id in enumerate(fit_ids):
        candidates = sorted(by_news[news_id], key=lambda row: row["annotation_id"])
        if index < args.fit_news // 2:
            chosen = max(candidates, key=lambda row: feature_by_id[row["annotation_id"]]["reranker_news_comment"])
            policy = "uniform_news_then_highest_frozen_news_relevance_score"
        else:
            chosen = rng.choice(candidates)
            policy = "uniform_news_then_uniform_generated_node"
        fit.append(dict(chosen, quality_review_role="fit", sampling_policy=policy))
    all_selected = fit + audit
    if len({row["news_id"] for row in all_selected}) != len(all_selected):
        raise ValueError("Fit/audit news overlap")
    if previous_news & {row["news_id"] for row in all_selected}:
        raise ValueError("Expansion overlaps the development queue")
    for row in all_selected:
        with Image.open(row["image_path"]) as image:
            image.verify()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("fit", fit), ("reserved_audit", audit)):
        (args.output_dir / f"{name}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    manifest = {
        "official_split": "train", "seed": args.seed, "candidate_news": len(by_news),
        "previous_development_news": len(previous_news), "new_fit_news": len(fit),
        "reserved_audit_news": len(audit), "unselected_news": len(eligible) - len(all_selected),
        "news_disjoint": True, "images_verified": len(all_selected), "news_truth_labels_used": False,
        "source_sha256": {"contexts": sha256(args.contexts), "previous_queue": sha256(args.previous_queue),
                          "features": sha256(args.features)},
        "queue_sha256": {name: sha256(args.output_dir / f"{name}.jsonl") for name in ("fit", "reserved_audit")},
        "fit_sampling": dict(Counter(row["sampling_policy"] for row in fit)),
        "fit_news_ids": fit_ids, "reserved_audit_news_ids": audit_ids,
        "protocol": "Annotate fit cases as AI weak labels. Fix features, targets, regularization and thresholds using previous+fit only. Then annotate/evaluate reserved_audit once without further tuning on its results. Official classifier validation/test are untouched.",
        "caveat": "Training enrichment is not population sampling; the reserved sample is uniform over eligible news, one generated node each. AI labels remain imperfect and correlated across annotators/models.",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ("candidate_news", "previous_development_news",
                     "new_fit_news", "reserved_audit_news", "unselected_news", "news_disjoint", "images_verified")}, indent=2))


if __name__ == "__main__":
    main()
