"""One predeclared MFCL vs adapted-MIMoE Weibo validation comparison.

The test split is deliberately not loaded.  Select a configuration on validation
before a separate, once-only checkpoint evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1 or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Use positive epochs/patience and distinct seeds")
    here = Path(__file__).resolve().parent
    destination = (args.output_dir or here / "outputs" / f"mimoe_weibo_{datetime.now():%Y%m%d_%H%M%S}").resolve()
    destination.mkdir(parents=True, exist_ok=False)
    config = here / "config_weibo.yaml"
    plan = {
        "dataset": "weibo", "arms": ["mfcl", "mimoe"], "seeds": args.seeds,
        "epochs": args.epochs, "patience": args.patience,
        "primary_metric": "validation macro-F1", "test_allowed": False,
        "adaptation": "MIMoE interaction/router/expert core on MFCL pooled features; not end-to-end MAE/ChineseCLIP replication",
        "source_commit": "002e747829dffcb2925803bb617cd1f9dd2de20c",
        "source_sha256": {}, "runs": [],
    }
    for path in [config, *here.glob("*.py")]:
        plan["source_sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    for seed in args.seeds:
        for fusion in plan["arms"]:
            name = f"{fusion}_seed{seed}"
            summary = destination / f"{name}.json"
            command = [
                sys.executable, "-u", str(here / "train.py"), "--config", str(config),
                "--mode", "baseline", "--graph-structure", "star", "--pooling-mode", "mean",
                "--decision-mode", "concat", "--mm-fusion", fusion, "--sampling", "shuffle",
                "--seed", str(seed), "--epochs", str(args.epochs), "--patience", str(args.patience),
                "--checkpoint-dir", str(destination / name), "--summary-json", str(summary),
                "--allow-formal-training",
            ]
            plan["runs"].append({"name": name, "fusion": fusion, "seed": seed,
                                 "command": command, "result": str(summary), "status": "planned"})

    def save() -> None:
        (destination / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")

    save()
    print(f"EXPERIMENT DIRECTORY: {destination}", flush=True)
    for run in plan["runs"]:
        run["status"] = "running"
        save()
        print(f"START {run['name']}", flush=True)
        with (destination / f"{run['name']}.log").open("w", encoding="utf-8") as log:
            child = subprocess.Popen(run["command"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, encoding="utf-8", errors="replace",
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            try:
                for line in child.stdout:
                    log.write(line)
                    log.flush()
                    if line.startswith(("mode=", "epoch=")):
                        print(f"[{run['name']}] {line.rstrip()}", flush=True)
                code = child.wait()
            except BaseException:
                child.terminate()
                child.wait()
                run["status"] = "interrupted"
                save()
                raise
        run["exit_code"] = code
        run["status"] = "completed" if code == 0 else "failed"
        if code == 0:
            result = json.loads(Path(run["result"]).read_text(encoding="utf-8"))
            run["best_validation_macro_f1"] = result["validation"]["macro_f1"]
            run["best_epoch"] = result["best_epoch"]
        save()
        if code:
            raise SystemExit(f"{run['name']} failed; inspect {destination}")
        print(f"DONE {run['name']} F1={run['best_validation_macro_f1']:.6f}", flush=True)
    print(f"ALL DONE: {destination}", flush=True)


if __name__ == "__main__":
    main()
