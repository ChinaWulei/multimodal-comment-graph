"""Aggregate fixed-seed validation-only reliability ablations without test use."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 13, 24])
    parser.add_argument("--modes", nargs="+", default=["none", "quality", "confidence", "quality_confidence"])
    args = parser.parse_args()
    runs = {}
    reference_ids = None
    for seed in args.seeds:
        runs[seed] = {}
        for mode in args.modes:
            path = args.run_dir / f"{mode}_seed{seed}.json"
            if not path.is_file():
                raise FileNotFoundError(f"Expected complete validation run: {path}")
            row = json.loads(path.read_text(encoding="utf-8"))
            if "test" in row:
                raise ValueError(f"This report must not mix in a test result: {path}")
            if int(row["seed"]) != seed or row["generated_reliability"] != mode:
                raise ValueError(f"Mode/seed mismatch: {path}")
            ids = [item["news_id"] for item in row["validation"]["prediction_records"]]
            if len(ids) != len(set(ids)) or len(ids) != row["validate_size"]:
                raise ValueError(f"Invalid validation news coverage: {path}")
            if reference_ids is None:
                reference_ids = ids
            elif reference_ids != ids:
                raise ValueError(f"Paired news IDs/order differ: {path}")
            runs[seed][mode] = row
    aggregate = {}
    for mode in args.modes:
        f1 = [runs[seed][mode]["validation"]["macro_f1"] for seed in args.seeds]
        accuracy = [runs[seed][mode]["validation"]["accuracy"] for seed in args.seeds]
        difference = [runs[seed][mode]["validation"]["macro_f1"] -
                      runs[seed]["none"]["validation"]["macro_f1"] for seed in args.seeds]
        aggregate[mode] = {
            "macro_f1_by_seed": dict(zip(args.seeds, f1)),
            "macro_f1_mean": mean(f1), "macro_f1_sample_std": stdev(f1) if len(f1) > 1 else None,
            "accuracy_mean": mean(accuracy),
            "paired_f1_delta_vs_none_by_seed": dict(zip(args.seeds, difference)),
            "paired_f1_delta_vs_none_mean": mean(difference),
        }
    result = {
        "dataset": "fakeddit_structure", "split": "validate",
        "train_news": runs[args.seeds[0]]["none"]["train_size"],
        "validation_news": len(reference_ids), "seeds": args.seeds,
        "test_evaluated": False,
        "warning": "Exploratory small validation cohort; not a confirmatory paper result.",
        "modes": aggregate,
    }
    out_json = args.run_dir / "aggregate_validation.json"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Generated-comment reliability: validation-only ablation", "",
             f"Official-train news: {result['train_news']}; official-validation news: {len(reference_ids)}; "
             f"seeds: {', '.join(map(str, args.seeds))}. Test is not evaluated here.", "",
             "| Mode | Mean macro-F1 ± sample SD | Mean accuracy | Paired macro-F1 difference vs none |",
             "| --- | ---: | ---: | ---: |"]
    for mode, metrics in aggregate.items():
        sd = f" ± {metrics['macro_f1_sample_std']:.4f}" if metrics["macro_f1_sample_std"] is not None else ""
        lines.append(f"| {mode} | {metrics['macro_f1_mean']:.4f}{sd} | "
                     f"{metrics['accuracy_mean']:.4f} | {metrics['paired_f1_delta_vs_none_mean']:+.4f} |")
    lines.extend(["", "Paired macro-F1 differences by seed:", ""])
    for mode, metrics in aggregate.items():
        if mode != "none":
            values = ", ".join(f"{seed}: {delta:+.4f}" for seed, delta in
                               metrics["paired_f1_delta_vs_none_by_seed"].items())
            lines.append(f"- {mode}: {values}")
    lines.extend(["", "These numbers do not verify comment quality, true reply-parent correctness, "
                  "or image use. The 49-news test cohort was inspected in earlier exploratory work "
                  "and cannot serve as a fresh confirmatory holdout.", ""])
    out_md = args.run_dir / "aggregate_validation.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"json": str(out_json), "markdown": str(out_md),
                      "macro_f1_means": {mode: row["macro_f1_mean"] for mode, row in aggregate.items()}}, indent=2))


if __name__ == "__main__":
    main()
