"""Summarize the predeclared 28-news hard-context test without model selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev


ARMS = ("real_only", "generated_plain", "generated_gated", "generated_residual")
SEEDS = (42, 13, 24)


def paired_changes(reference: list[dict], candidate: list[dict]) -> dict[str, int]:
    corrected = introduced = both_correct = both_wrong = 0
    for old, new in zip(reference, candidate):
        if (old["news_id"], old["label"]) != (new["news_id"], new["label"]):
            raise ValueError("Unpaired hard-context prediction records")
        old_right = old["prediction"] == old["label"]
        new_right = new["prediction"] == new["label"]
        if not old_right and new_right:
            corrected += 1
        elif old_right and not new_right:
            introduced += 1
        elif old_right:
            both_correct += 1
        else:
            both_wrong += 1
    return {"corrected": corrected, "introduced": introduced,
            "both_correct": both_correct, "both_wrong": both_wrong}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = {}
    expected = None
    graph_hash = None
    for seed in SEEDS:
        rows[seed] = {}
        for arm in ARMS:
            path = args.run_dir / f"{arm}_seed{seed}.json"
            row = json.loads(path.read_text(encoding="utf-8"))
            if row["seed"] != seed or row["test_size"] != 28 or row["max_real_comments"] != 16:
                raise ValueError(f"Invalid run metadata: {path}")
            if not row["generated_test_sha256"] or not row["checkpoint_sha256"]:
                raise ValueError(f"Unfrozen graph/checkpoint identity: {path}")
            if graph_hash is None:
                graph_hash = row["generated_test_sha256"]
            elif graph_hash != row["generated_test_sha256"]:
                raise ValueError(f"Generated graph changed across runs: {path}")
            records = row["test"]["prediction_records"]
            ids_labels = [(item["news_id"], item["label"]) for item in records]
            if len(ids_labels) != 28 or len(set(news_id for news_id, _ in ids_labels)) != 28:
                raise ValueError(f"Invalid prediction coverage: {path}")
            if expected is None:
                expected = ids_labels
            elif ids_labels != expected:
                raise ValueError(f"News IDs or labels differ across arms: {path}")
            rows[seed][arm] = row

    aggregate = {}
    for arm in ARMS:
        scores = [float(rows[seed][arm]["test"]["macro_f1"]) for seed in SEEDS]
        accuracies = [float(rows[seed][arm]["test"]["accuracy"]) for seed in SEEDS]
        recalls = []
        for seed in SEEDS:
            cm = rows[seed][arm]["test"]["confusion_matrix"]
            if len(cm) != 3 or any(len(class_row) != 3 for class_row in cm):
                raise ValueError(f"Expected three-class confusion matrix for {arm}/{seed}")
            recalls.append([cm[label][label] / sum(cm[label]) if sum(cm[label]) else 0.0
                            for label in range(3)])
        vs_real = {str(seed): paired_changes(
            rows[seed]["real_only"]["test"]["prediction_records"],
            rows[seed][arm]["test"]["prediction_records"]) for seed in SEEDS}
        vs_plain = {str(seed): paired_changes(
            rows[seed]["generated_plain"]["test"]["prediction_records"],
            rows[seed][arm]["test"]["prediction_records"]) for seed in SEEDS}
        aggregate[arm] = {
            "macro_f1_by_seed": dict(zip(map(str, SEEDS), scores)),
            "macro_f1_mean": mean(scores), "macro_f1_sample_std": stdev(scores),
            "accuracy_mean": mean(accuracies),
            "per_class_recall_mean": {str(label): mean(item[label] for item in recalls)
                                      for label in range(3)},
            "paired_macro_f1_delta_vs_real_by_seed": {
                str(seed): float(rows[seed][arm]["test"]["macro_f1"])
                - float(rows[seed]["real_only"]["test"]["macro_f1"]) for seed in SEEDS},
            "paired_vs_real": vs_real, "paired_vs_plain": vs_plain,
        }
    report = {
        "dataset": "Fakeddit", "cohort": "official_test_hard_context_unseen_generated_subset",
        "news_count": 28, "seeds": SEEDS, "generated_graph_sha256": graph_hash,
        "caveat": "Exploratory non-representative hard cohort; test provenance overlaps prior work. Do not use it to tune or claim confirmatory publication-level generalization.",
        "arms": aggregate,
    }
    json_out = args.run_dir / "aggregate_hard_context.json"
    json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Frozen hard-context exploratory evaluation", "",
             "28 previously ungenerated Fakeddit official-test news, selected by the prior relevance pipeline's exclusion. Not a representative independent holdout.",
             "Three validation-selected checkpoints per arm; no retraining or test-based parameter selection.", "",
             "| Arm | Macro-F1 mean ± seed SD | Accuracy mean | Seed macro-F1 (42, 13, 24) |",
             "| --- | ---: | ---: | --- |"]
    for arm, item in aggregate.items():
        vals = ", ".join(f"{item['macro_f1_by_seed'][str(seed)]:.4f}" for seed in SEEDS)
        lines.append(f"| {arm} | {item['macro_f1_mean']:.4f} ± {item['macro_f1_sample_std']:.4f} | "
                     f"{item['accuracy_mean']:.4f} | {vals} |")
    lines += ["", "Paired changes relative to the same-seed real-only classifier (corrected / introduced errors):", ""]
    for arm in ARMS[1:]:
        changes = aggregate[arm]["paired_vs_real"]
        detail = ", ".join(f"{seed}: {changes[str(seed)]['corrected']}/{changes[str(seed)]['introduced']}"
                           for seed in SEEDS)
        lines.append(f"- {arm}: {detail}")
    lines += ["", "Mean recall by original numeric class (mapping is not inferred here):", "",
              "| Arm | Class 0 | Class 1 | Class 2 |", "| --- | ---: | ---: | ---: |"]
    for arm, item in aggregate.items():
        recall = item["per_class_recall_mean"]
        lines.append(f"| {arm} | {recall['0']:.4f} | {recall['1']:.4f} | {recall['2']:.4f} |")
    lines += ["", "Scores on 28 deliberately difficult news cannot prove generalization, correct generated-parent placement, or visual grounding. A genuinely untouched larger test cohort and human edge/comment review remain necessary.", ""]
    md_out = args.run_dir / "aggregate_hard_context.md"
    md_out.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"markdown": str(md_out), "json": str(json_out),
                      "macro_f1_means": {arm: row["macro_f1_mean"] for arm, row in aggregate.items()}}, indent=2))


if __name__ == "__main__":
    main()
