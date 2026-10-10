"""Freeze all validation-selected ablations, then evaluate reused test once.

The 49-news Fakeddit test cohort has been inspected in previous experiments;
these results are explicitly exploratory, not a fresh confirmatory test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from statistics import mean, stdev

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 13, 24])
    parser.add_argument("--modes", nargs="+", default=["none", "quality", "confidence", "quality_confidence"])
    args = parser.parse_args()
    aggregate_validation = args.run_dir / "aggregate_validation.json"
    if not aggregate_validation.is_file():
        raise FileNotFoundError("Run summarize_reliability_validation.py first")
    validation = json.loads(aggregate_validation.read_text(encoding="utf-8"))
    if validation["seeds"] != args.seeds or set(validation["modes"]) != set(args.modes):
        raise ValueError("Validation aggregate does not match requested frozen arms")
    test_dir = args.run_dir / "exploratory_reused_test"
    test_dir.mkdir(parents=True, exist_ok=True)
    plan_path = test_dir / "frozen_test_plan.json"
    source_files = [
        args.config.resolve(), Path(__file__).resolve(),
        PROJECT_ROOT / "comment_graph" / "model.py",
        PROJECT_ROOT / "comment_graph" / "typed_graph.py",
        PROJECT_ROOT / "comment_graph" / "generated_reliability.py",
        PROJECT_ROOT / "comment_graph" / "dynamic_graph.py",
        PROJECT_ROOT / "comment_graph" / "isolated_fusion.py",
        PROJECT_ROOT / "comment_graph" / "common.py",
        PROJECT_ROOT / "comment_graph" / "fakeddit_structure_dataset.py",
        PROJECT_ROOT / "Weibo" / "online_encoder.py",
        PROJECT_ROOT / "Weibo" / "propogation.py",
        PROJECT_ROOT / "comment_graph" / "evaluate_frozen_checkpoint.py",
    ]
    current_plan = {
        "protocol": "validation-selected frozen checkpoints; exploratory reused 49-news test",
        "test_is_fresh": False, "seeds": args.seeds, "modes": args.modes,
        "validation_aggregate_sha256": sha256(aggregate_validation),
        "source_sha256": {str(path): sha256(path) for path in source_files},
        "arms": [],
    }
    for seed in args.seeds:
        for mode in args.modes:
            summary_path = args.run_dir / f"{mode}_seed{seed}.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if "test" in summary or int(summary["seed"]) != seed or summary["generated_reliability"] != mode:
                raise ValueError(f"Invalid validation-only arm: {summary_path}")
            checkpoint = Path(summary["checkpoint"])
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            current_plan["arms"].append({
                "seed": seed, "mode": mode, "summary": str(summary_path.resolve()),
                "validation_macro_f1": summary["best_validation_macro_f1"],
                "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256(checkpoint),
            })
    if plan_path.exists():
        existing = json.loads(plan_path.read_text(encoding="utf-8"))
        if existing != current_plan:
            raise ValueError("Frozen plan changed; refusing to mix checkpoints/source changes with test results")
    else:
        plan_path.write_text(json.dumps(current_plan, ensure_ascii=False, indent=2), encoding="utf-8")
    evaluator = PROJECT_ROOT / "comment_graph" / "evaluate_frozen_checkpoint.py"
    results = {}
    expected_news_ids = None
    for arm in current_plan["arms"]:
        seed, mode = arm["seed"], arm["mode"]
        result_path = test_dir / f"{mode}_seed{seed}.json"
        if not result_path.exists():
            subprocess.run([
                sys.executable, str(evaluator), "--config", str(args.config),
                "--checkpoint", arm["checkpoint"], "--output-json", str(result_path),
            ], cwd=PROJECT_ROOT, check=True)
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result["checkpoint"] != arm["checkpoint"] or result["seed"] != seed or \
                result["generated_reliability"] != mode:
            raise ValueError(f"Frozen result mismatch: {result_path}")
        if sha256(Path(arm["checkpoint"])) != arm["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint changed after freezing: {arm['checkpoint']}")
        ids = [row["news_id"] for row in result["test"]["prediction_records"]]
        if len(ids) != len(set(ids)) or len(ids) != result["test_size"]:
            raise ValueError(f"Invalid test IDs: {result_path}")
        if expected_news_ids is None:
            expected_news_ids = ids
        elif ids != expected_news_ids:
            raise ValueError(f"Paired test IDs/order differ: {result_path}")
        results[(seed, mode)] = result
    table = {}
    for mode in args.modes:
        f1 = [results[(seed, mode)]["test"]["macro_f1"] for seed in args.seeds]
        accuracy = [results[(seed, mode)]["test"]["accuracy"] for seed in args.seeds]
        delta = [results[(seed, mode)]["test"]["macro_f1"] -
                 results[(seed, "none")]["test"]["macro_f1"] for seed in args.seeds]
        table[mode] = {
            "macro_f1_by_seed": dict(zip(args.seeds, f1)),
            "macro_f1_mean": mean(f1), "macro_f1_sample_std": stdev(f1),
            "accuracy_mean": mean(accuracy),
            "paired_macro_f1_delta_vs_none_by_seed": dict(zip(args.seeds, delta)),
            "paired_macro_f1_delta_vs_none_mean": mean(delta),
        }
    aggregate = {
        "protocol": current_plan["protocol"], "test_is_fresh": False,
        "test_news": len(expected_news_ids), "seeds": args.seeds, "modes": table,
        "warning": "This reused test cohort has been inspected in previous exploratory work. Do not claim confirmatory significance or publication-level improvement.",
    }
    out_json = test_dir / "aggregate_exploratory_test.json"
    out_json.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"test_news": len(expected_news_ids),
                      "macro_f1_means": {mode: row["macro_f1_mean"] for mode, row in table.items()},
                      "output": str(out_json)}, indent=2))


if __name__ == "__main__":
    main()
