"""Summarize paired validation runs with at most 16 observed real comments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev


ARMS = {
    "real_only": ("real_only", "concat", "none"),
    "generated_plain": ("real_generated", "concat", "none"),
    "generated_gated": ("real_generated", "concat", "quality_confidence"),
    "generated_residual": ("real_generated", "reliability_residual", "quality_confidence"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 13, 24])
    args = parser.parse_args()
    rows = {}
    reference = None
    for seed in args.seeds:
        rows[seed] = {}
        for arm, (mode, decision, reliability) in ARMS.items():
            path = args.run_dir / f"{arm}_seed{seed}.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            expected = (seed, mode, decision, reliability, 16, 5, 154, 30)
            actual = tuple(data[key] for key in (
                "seed", "mode", "decision_mode", "generated_reliability",
                "max_real_comments", "max_generated_comments", "train_size", "validate_size"))
            if actual != expected or "test" in data:
                raise ValueError(f"Unexpected run configuration or test metrics: {path}: {actual}")
            if data["graph_structure"] != "observed_tree" or data["message_passing"] != "typed_rgcn":
                raise ValueError(f"Graph implementation changed: {path}")
            records = data["validation"]["prediction_records"]
            coverage = [(record["news_id"], record["label"]) for record in records]
            if len(coverage) != len(set(news_id for news_id, _ in coverage)) or len(coverage) != 30:
                raise ValueError(f"Invalid validation coverage: {path}")
            if reference is None:
                reference = coverage
            elif coverage != reference:
                raise ValueError(f"Paired validation IDs/labels differ: {path}")
            rows[seed][arm] = data

    metrics = {}
    for arm in ARMS:
        f1 = [float(rows[seed][arm]["validation"]["macro_f1"]) for seed in args.seeds]
        accuracy = [float(rows[seed][arm]["validation"]["accuracy"]) for seed in args.seeds]
        metrics[arm] = {
            "macro_f1_by_seed": dict(zip(map(str, args.seeds), f1)),
            "macro_f1_mean": mean(f1),
            "macro_f1_sample_std": stdev(f1) if len(f1) > 1 else None,
            "accuracy_mean": mean(accuracy),
            "paired_delta_vs_real_only": {
                str(seed): float(rows[seed][arm]["validation"]["macro_f1"])
                - float(rows[seed]["real_only"]["validation"]["macro_f1"])
                for seed in args.seeds
            },
            "paired_delta_vs_generated_plain": {
                str(seed): float(rows[seed][arm]["validation"]["macro_f1"])
                - float(rows[seed]["generated_plain"]["validation"]["macro_f1"])
                for seed in args.seeds
            },
        }
        metrics[arm]["paired_delta_vs_real_only_mean"] = mean(metrics[arm]["paired_delta_vs_real_only"].values())
        metrics[arm]["paired_delta_vs_generated_plain_mean"] = mean(metrics[arm]["paired_delta_vs_generated_plain"].values())

    result = {
        "dataset": "Fakeddit", "split": "official_validation", "train_news": 154,
        "validation_news": 30, "real_comment_budget": 16, "generated_comment_budget": 5,
        "seeds": args.seeds, "test_evaluated": False,
        "warning": "Small exploratory validation cohort; existing 49-news test was previously reused and is not a fresh holdout.",
        "arms": metrics,
    }
    output = args.run_dir / "aggregate_validation.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Partial-observation generated-comment ablation", "",
             "Fakeddit: 154 training and 30 validation news, at most 16 observed real comments and 5 generated comments per news.",
             "Three seeds, identical validation news IDs and labels. No test evaluation in these runs.", "",
             "| Arm | Validation macro-F1 mean ± SD | Accuracy mean | Δ vs real-only | Δ vs unfiltered generated |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for arm, row in metrics.items():
        lines.append(f"| {arm} | {row['macro_f1_mean']:.4f} ± {row['macro_f1_sample_std']:.4f} | "
                     f"{row['accuracy_mean']:.4f} | {row['paired_delta_vs_real_only_mean']:+.4f} | "
                     f"{row['paired_delta_vs_generated_plain_mean']:+.4f} |")
    lines += ["", "Seed-level validation macro-F1:", ""]
    for arm, row in metrics.items():
        values = ", ".join(f"{seed}: {score:.4f}" for seed, score in row["macro_f1_by_seed"].items())
        lines.append(f"- {arm}: {values}")
    lines += ["", "The validation set has only 30 news (10 per class). These exploratory scores do not establish",
              "correct reply parents, image grounding, generalization, or publication-level improvement. The",
              "existing 49-news test cohort was inspected earlier and cannot be treated as a fresh holdout.", ""]
    markdown = args.run_dir / "aggregate_validation.md"
    markdown.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"json": str(output), "markdown": str(markdown), "macro_f1_means":
                      {arm: row["macro_f1_mean"] for arm, row in metrics.items()}}, indent=2))


if __name__ == "__main__":
    main()
