"""Compare predeclared first-reply ablation with the paired real-only control."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, stdev

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from comment_graph.summarize_all_valid_images_validation import (
    ids_from_graphs, metrics_on, paired_changes,
)

SEEDS = (42, 13, 24)
ARMS = ("real_only", "first_root", "first_parent", "first_parent_gated")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-only-dir", type=Path, required=True)
    parser.add_argument("--first-reply-dir", type=Path, required=True)
    parser.add_argument("--original-graphs", type=Path, required=True)
    parser.add_argument("--additional-graphs", type=Path, required=True)
    parser.add_argument("--generator-validation-sft", type=Path, required=True)
    args = parser.parse_args()
    original = ids_from_graphs(args.original_graphs, "validate")
    additional = ids_from_graphs(args.additional_graphs, "validate")
    if len(original) != 30 or len(additional) != 47 or original & additional:
        raise ValueError("Expected disjoint 30/47 news sources")
    generator_selection = {
        str(row["metadata"]["news_id"])
        for line in args.generator_validation_sft.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for row in [json.loads(line)]
        if row["metadata"]["official_split"] == "validate"
    }
    if len(generator_selection & (original | additional)) != 33:
        raise ValueError("Unexpected generator checkpoint-selection overlap")
    strata = {"all_77": original | additional,
              "hard_context_47": additional,
              "generator_selection_disjoint_44": (original | additional) - generator_selection}
    runs: dict[int, dict[str, dict]] = {}
    reference = None
    source_hashes = None
    for seed in SEEDS:
        runs[seed] = {}
        for arm in ARMS:
            directory = args.real_only_dir if arm == "real_only" else args.first_reply_dir
            row = json.loads((directory / f"{arm}_seed{seed}.json").read_text(encoding="utf-8"))
            expected_generated = 5 if arm == "real_only" else 1
            if (row["seed"] != seed or row["train_size"] != 292 or row["validate_size"] != 77 or
                    row["max_real_comments"] != 16 or row["max_generated_comments"] != expected_generated or
                    "test" in row):
                raise ValueError(f"Run metadata mismatch: {arm}/{seed}")
            hashes = {split: row["generated_sources"][split]["sha256"]
                      for split in ("train", "validate")}
            if source_hashes is None:
                source_hashes = hashes
            elif hashes != source_hashes:
                raise ValueError(f"Generated sources differ: {arm}/{seed}")
            records = row["validation"]["prediction_records"]
            pairs = [(record["news_id"], record["label"]) for record in records]
            if len(pairs) != 77 or {item[0] for item in pairs} != strata["all_77"]:
                raise ValueError(f"Wrong validation cohort: {arm}/{seed}")
            if reference is None:
                reference = pairs
            elif reference != pairs:
                raise ValueError(f"Unpaired news order/labels: {arm}/{seed}")
            runs[seed][arm] = row
    result = {"dataset": "Fakeddit", "split": "official_validation", "test_evaluated": False,
              "seeds": SEEDS, "train_news": 292, "validation_news": 77,
              "source_sha256": source_hashes, "strata": {}}
    lines = ["# First-generated-reply ablation (exploratory validation only)", "",
             "Same 292/77 official split, 16 real comments, and generator outputs as the five-reply comparison. No test evaluation.", "",
             "The real-only control is reused unchanged; first-reply variants retain exactly one generated comment per news.", ""]
    for name, allowed in strata.items():
        arm_scores = {}
        for arm in ARMS:
            scores = [metrics_on(runs[seed][arm]["validation"]["prediction_records"], allowed)["macro_f1"]
                      for seed in SEEDS]
            arm_scores[arm] = {"macro_f1_by_seed": dict(zip(map(str, SEEDS), scores)),
                               "macro_f1_mean": mean(scores), "macro_f1_seed_std": stdev(scores)}
            if arm != "real_only":
                arm_scores[arm]["paired_vs_real"] = {
                    str(seed): paired_changes(
                        runs[seed]["real_only"]["validation"]["prediction_records"],
                        runs[seed][arm]["validation"]["prediction_records"], allowed)
                    for seed in SEEDS}
        result["strata"][name] = {"news": len(allowed), "arms": arm_scores}
        lines += [f"## {name} ({len(allowed)} news)", "",
                  "| Arm | Macro-F1 mean ± seed SD | Seed scores 42 / 13 / 24 |",
                  "| --- | ---: | --- |"]
        for arm, score in arm_scores.items():
            per_seed = " / ".join(f"{score['macro_f1_by_seed'][str(seed)]:.4f}" for seed in SEEDS)
            lines.append(f"| {arm} | {score['macro_f1_mean']:.4f} ± {score['macro_f1_seed_std']:.4f} | {per_seed} |")
        lines += ["", "Paired corrected/introduced errors versus real-only:", ""]
        for arm in ARMS[1:]:
            changes = arm_scores[arm]["paired_vs_real"]
            lines.append(f"- {arm}: " + ", ".join(
                f"{seed}: {changes[str(seed)]['corrected']}/{changes[str(seed)]['introduced']}"
                for seed in SEEDS))
        lines += [""]
    lines += ["Generator checkpoint selection used 33/77 validation news; the disjoint 44-news stratum is still small and not an untouched test. Do not claim publication-level generalization from this ablation.", ""]
    output = args.first_reply_dir
    (output / "aggregate_first_reply.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "aggregate_first_reply.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"report": str(output / "aggregate_first_reply.md"),
                      "all_77": {arm: row["macro_f1_mean"]
                                 for arm, row in result["strata"]["all_77"]["arms"].items()},
                      "disjoint_44": {arm: row["macro_f1_mean"]
                                      for arm, row in result["strata"]["generator_selection_disjoint_44"]["arms"].items()}},
                     indent=2))


if __name__ == "__main__":
    main()
