"""Evaluate previously frozen MFCL/MIMoE checkpoints exactly once on Weibo test.

This runner does no training and refuses incomplete validation experiment folders.
It is intended only after the paired validation protocol has been completed.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-runs", type=Path, required=True,
                        help="Directory containing completed mfcl_seed*.json and mimoe_seed*.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = args.validation_runs.resolve()
    destination = args.output_dir.resolve()
    if destination.exists():
        raise SystemExit("Choose a new test output directory; existing reports are immutable")
    summaries = sorted(source.glob("*_seed*.json"))
    if not summaries:
        raise SystemExit("No completed validation summaries found")
    parsed = []
    for path in summaries:
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("dataset") != "weibo" or row.get("mm_fusion") not in {"mfcl", "mimoe"}:
            continue
        if "test" in row:
            raise SystemExit(f"{path} already contains test results; refusing repeated test use")
        parsed.append((path, row))
    keys = {(row["seed"], row["mm_fusion"]) for _, row in parsed}
    seeds = sorted({row["seed"] for _, row in parsed})
    if not seeds or any((seed, arm) not in keys for seed in seeds for arm in ("mfcl", "mimoe")):
        raise SystemExit("Need one completed MFCL and MIMoE validation checkpoint for every seed")
    destination.mkdir(parents=True)
    here = Path(__file__).resolve().parent
    plan = {
        "purpose": "once-only frozen-checkpoint Weibo test evaluation",
        "training_performed": False,
        "selection": "all checkpoints were selected solely by their own validation macro-F1 before this test run",
        "source_validation_runs": str(source), "seeds": seeds, "runs": [],
    }
    for source_json, row in sorted(parsed, key=lambda item: (item[1]["seed"], item[1]["mm_fusion"])):
        arm, seed = row["mm_fusion"], row["seed"]
        output = destination / f"{arm}_seed{seed}_test.json"
        plan["runs"].append({
            "arm": arm, "seed": seed, "validation_summary": str(source_json),
            "checkpoint": row["checkpoint"], "validation_macro_f1": row["validation"]["macro_f1"],
            "output": str(output), "status": "planned",
        })

    def save() -> None:
        (destination / "test_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")

    save()
    for run in plan["runs"]:
        run["status"] = "running"
        save()
        command = [sys.executable, "-u", str(here / "evaluate_checkpoint.py"),
                   "--checkpoint", run["checkpoint"], "--config", str(here / "config_weibo.yaml"),
                   "--split", "test", "--allow-test", "--output", run["output"]]
        with (destination / f"{run['arm']}_seed{run['seed']}.log").open("w", encoding="utf-8") as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True,
                                     encoding="utf-8", errors="replace", creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        run["exit_code"] = process.returncode
        run["status"] = "completed" if process.returncode == 0 else "failed"
        if process.returncode:
            save()
            raise SystemExit(f"Test evaluation failed; see {destination}")
        report = json.loads(Path(run["output"]).read_text(encoding="utf-8"))
        run["test_macro_f1"] = report["metrics"]["macro_f1"]
        run["test_accuracy"] = report["metrics"]["accuracy"]
        save()
        print(f"DONE {run['arm']}_seed{run['seed']} test_macro_f1={run['test_macro_f1']:.6f}", flush=True)
    by_seed = {}
    for run in plan["runs"]:
        by_seed.setdefault(run["seed"], {})[run["arm"]] = run["test_macro_f1"]
    deltas = [values["mimoe"] - values["mfcl"] for _, values in sorted(by_seed.items())]
    plan["aggregate"] = {
        "mfcl_test_macro_f1_mean": statistics.mean(values["mfcl"] for values in by_seed.values()),
        "mimoe_test_macro_f1_mean": statistics.mean(values["mimoe"] for values in by_seed.values()),
        "paired_delta_mean": statistics.mean(deltas),
        "paired_delta_std": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
        "per_seed_delta": {str(seed): values["mimoe"] - values["mfcl"] for seed, values in by_seed.items()},
    }
    save()
    print(json.dumps(plan["aggregate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
