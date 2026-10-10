"""Paired validation-only report for news-anchored generated reply chains."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev


SEEDS = (42, 13, 24)
CONTROL_ARMS = ("real_only", "generated_plain", "generated_gated")


def compare(reference: list[dict], candidate: list[dict]) -> dict[str, int]:
    corrected = introduced = 0
    for old, new in zip(reference, candidate):
        if (old["news_id"], old["label"]) != (new["news_id"], new["label"]):
            raise ValueError("Validation pairs differ")
        corrected += old["prediction"] != old["label"] and new["prediction"] == new["label"]
        introduced += old["prediction"] == old["label"] and new["prediction"] != new["label"]
    return {"corrected": corrected, "introduced": introduced}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor-dir", type=Path, required=True)
    parser.add_argument("--control-dir", type=Path, required=True)
    args = parser.parse_args()
    runs = {}
    expected = None
    for seed in SEEDS:
        runs[seed] = {}
        for arm in (*CONTROL_ARMS, "ancestry_anchor"):
            directory = args.anchor_dir if arm == "ancestry_anchor" else args.control_dir
            path = directory / f"{arm}_seed{seed}.json"
            row = json.loads(path.read_text(encoding="utf-8"))
            if "test" in row or row["seed"] != seed or row["max_real_comments"] != 16 or row["validate_size"] != 30:
                raise ValueError(f"Unexpected run/test data in {path}")
            if arm == "ancestry_anchor" and row["generated_reliability"] != "ancestry_anchor":
                raise ValueError(f"Not an ancestry-anchor checkpoint: {path}")
            records = row["validation"]["prediction_records"]
            pairs = [(item["news_id"], item["label"]) for item in records]
            if len(pairs) != 30 or len({pair[0] for pair in pairs}) != 30:
                raise ValueError(f"Invalid validation coverage: {path}")
            if expected is None:
                expected = pairs
            elif pairs != expected:
                raise ValueError(f"News IDs or labels differ: {path}")
            runs[seed][arm] = row
    scores = {arm: [float(runs[seed][arm]["validation"]["macro_f1"]) for seed in SEEDS]
              for arm in (*CONTROL_ARMS, "ancestry_anchor")}
    anchors = [item for seed in SEEDS for record in runs[seed]["ancestry_anchor"]["validation"]["prediction_records"]
               for item in record["generated_reliability"]]
    if len(anchors) != 450:
        raise ValueError(f"Expected 450 generated validation nodes, got {len(anchors)}")
    result = {
        "split": "official_validation", "train_news": 154, "validation_news": 30,
        "seeds": SEEDS, "test_evaluated": False,
        "arms": {arm: {"macro_f1_by_seed": dict(zip(map(str, SEEDS), values)),
                        "macro_f1_mean": mean(values), "macro_f1_seed_std": stdev(values)}
                 for arm, values in scores.items()},
        "anchor_diagnostic": {
            "generated_nodes": len(anchors),
            "mean_news_anchor_factor": mean(item["news_anchor_factor"] for item in anchors),
            "inherited_anchor_nodes": sum(item["anchor_source_node"] != item["node_index"] for item in anchors),
            "near_floor_nodes": sum(item["news_anchor_factor"] <= 0.201 for item in anchors),
        },
        "warning": "Small validation cohort; this is exploratory and no test score can confirm a post-test-designed method.",
    }
    for arm in CONTROL_ARMS:
        result["arms"]["ancestry_anchor"][f"paired_delta_vs_{arm}_by_seed"] = {
            str(seed): scores["ancestry_anchor"][i] - scores[arm][i] for i, seed in enumerate(SEEDS)}
        result["arms"]["ancestry_anchor"][f"paired_changes_vs_{arm}"] = {
            str(seed): compare(runs[seed][arm]["validation"]["prediction_records"],
                               runs[seed]["ancestry_anchor"]["validation"]["prediction_records"])
            for seed in SEEDS}
    out_json = args.anchor_dir / "aggregate_validation.json"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Ancestry-anchored generated comments: validation only", "",
             "154 train / 30 validation news; 16 real and up to 5 generated comments; same three seeds and validation IDs. Test not evaluated.", "",
             "| Arm | Macro-F1 mean ± seed SD | Seeds 42 / 13 / 24 |",
             "| --- | ---: | --- |"]
    for arm, row in result["arms"].items():
        vals = " / ".join(f"{row['macro_f1_by_seed'][str(seed)]:.4f}" for seed in SEEDS)
        lines.append(f"| {arm} | {row['macro_f1_mean']:.4f} ± {row['macro_f1_seed_std']:.4f} | {vals} |")
    lines += ["", "Paired ancestry-anchor differences versus controls (macro-F1):", ""]
    for arm in CONTROL_ARMS:
        deltas = result["arms"]["ancestry_anchor"][f"paired_delta_vs_{arm}_by_seed"]
        lines.append(f"- {arm}: " + ", ".join(f"{seed}: {deltas[str(seed)]:+.4f}" for seed in SEEDS))
    diag = result["anchor_diagnostic"]
    lines += ["", f"Anchor factors: mean {diag['mean_news_anchor_factor']:.4f}; "
              f"{diag['inherited_anchor_nodes']}/{diag['generated_nodes']} generated descendants inherit a head anchor; "
              f"{diag['near_floor_nodes']} nodes are near the floor.", "",
              "This is a label-free heuristic calibrated from train-only real-root similarity, not verified parent or comment-quality supervision. The 28-news hard-context test was already inspected and must not be used to select this new mode.", ""]
    out_md = args.anchor_dir / "aggregate_validation.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"markdown": str(out_md), "json": str(out_json),
                      "macro_f1_means": {arm: value["macro_f1_mean"] for arm, value in result["arms"].items()}}, indent=2))


if __name__ == "__main__":
    main()
