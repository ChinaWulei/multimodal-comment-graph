"""Complete the declared placement control, then test all frozen paired arms.

No test-driven model selection. Requires the upstream three-seed validation
pipeline to have completed successfully. Existing output folders are immutable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARMS = {
    "real_tree_typed_concat": ("real_only", "observed_tree", "concat"),
    "real_tree_typed_global_local": ("real_only", "observed_tree", "global_local_correction"),
    "generated_tree_typed_concat": ("real_generated", "observed_tree", "concat"),
    "generated_tree_typed_global_local": ("real_generated", "observed_tree", "global_local_correction"),
    "generated_root_typed_concat": ("real_generated", "generated_root", "concat"),
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def validate_summary(row, arm, seed):
    mode, graph, decision = ARMS[arm]
    expected = {"mode": mode, "graph_structure": graph, "decision_mode": decision,
                "seed": seed, "dataset": "fakeddit_structure", "num_classes": 3,
                "pooling_mode": "mean", "message_passing": "typed_rgcn",
                "real_node_selection": "observed_context", "max_real_comments": 64,
                "max_generated_comments": 5, "paired_generated_coverage": True}
    for key, value in expected.items():
        if row.get(key) != value:
            raise ValueError(f"{arm}/{seed}: {key}={row.get(key)!r}, expected {value!r}")
    if "test" in row:
        raise ValueError("Validation summary already contains test results")
    if row["best_validation_macro_f1"] != row["validation"]["macro_f1"]:
        raise ValueError("Validation-selected metric does not match saved summary")
    if not Path(row["checkpoint"]).is_file():
        raise FileNotFoundError(row["checkpoint"])


def validate_checkpoint(summary):
    import torch
    checkpoint = torch.load(summary["checkpoint"], map_location="cpu", weights_only=False, mmap=True)
    for key in ("seed", "mode", "graph_structure", "pooling_mode", "message_passing", "decision_mode"):
        if checkpoint.get(key) != summary[key]:
            raise ValueError(f"Checkpoint and validation summary disagree on {key}")
    if checkpoint.get("epoch") != summary["best_epoch"]:
        raise ValueError("Checkpoint and selected epoch disagree")
    if checkpoint["validation_metrics"] != summary["validation"]:
        raise ValueError("Checkpoint and stored validation results disagree")
    data = checkpoint["config"]["data"]
    for key in ("real_node_selection", "max_real_comments", "max_generated_comments"):
        if data.get(key) != summary[key]:
            raise ValueError(f"Checkpoint has a different data protocol: {key}")


def correction_effect(records):
    """Descriptive within-checkpoint ablation; not an independently trained base."""
    if not records or not any("global_class_probabilities" in r for r in records):
        return None
    if not all("global_class_probabilities" in r and "class_logit_delta" in r for r in records):
        raise ValueError("Incomplete local-correction prediction diagnostics")
    changed = corrected = harmed = 0
    magnitudes = []
    for row in records:
        probabilities = row["global_class_probabilities"]
        base = max(range(len(probabilities)), key=lambda i: probabilities[i])
        final, label = row["prediction"], row["label"]
        changed += base != final
        corrected += base != label and final == label
        harmed += base == label and final != label
        magnitudes.extend(abs(value) for value in row["class_logit_delta"])
    return {"scope": "Global logits versus final logits of the SAME jointly trained checkpoint, not a standalone baseline.",
            "news": len(records), "changed_predictions": changed, "corrected_predictions": corrected,
            "harmed_predictions": harmed, "mean_absolute_logit_delta": statistics.mean(magnitudes)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source, destination = args.source_run.resolve(), args.output_dir.resolve()
    if read(source / "status.json")["stage"] != "validation_complete":
        raise RuntimeError("Upstream generation and all validation runs must finish first")
    plan = read(source / "plan.json")
    config = Path(plan["formal_config"])
    amendment = read(source / "context_selection_amendment.json")
    if sha256(config) != amendment["amended_classifier_config_sha256"]:
        raise RuntimeError("Classifier config changed after the declared context-selection amendment")
    seeds = plan["seeds"]
    if seeds != [42, 13, 24]:
        raise ValueError("Unexpected seed protocol")
    upstream = []
    for seed in seeds:
        for arm in list(ARMS)[:4]:
            path = source / f"validation_seed{seed}" / f"{arm}.json"
            summary = read(path)
            validate_summary(summary, arm, seed)
            upstream.append((arm, seed, path, summary))
    destination.mkdir(parents=True, exist_ok=False)
    status_path = destination / "status.json"

    def status(stage, detail):
        write(status_path, {"stage": stage, "detail": detail,
                            "updated_at": datetime.now(timezone.utc).isoformat()})

    def run(script, arguments, log):
        with log.open("w", encoding="utf-8") as handle:
            result = subprocess.run([sys.executable, "-u", str(ROOT / "comment_graph" / script), *map(str, arguments)],
                                    cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if result.returncode:
            raise RuntimeError(f"Command failed ({result.returncode}); inspect {log}")

    try:
        status("graft_audit", "Require all generated parents retained and identical real nodes between arms")
        audit_path = destination / "graft_audit.json"
        run("audit_generated_grafts.py", ["--config", config, "--split", "train", "--split", "validate",
            "--split", "test", "--output-json", audit_path], destination / "graft_audit.log")
        audit = read(audit_path)
        if not audit["passed"] or not all(v["all_predicted_parents_preserved"] for v in audit["splits"].values()):
            raise RuntimeError("Generated parents lost during dataset selection")
        write(destination / "protocol.json", {
            "seeds": seeds, "arms": ARMS, "source_run": str(source), "config_sha256": sha256(config),
            "selection": "Each arm uses its own best validation macro-F1 checkpoint; all arms and seeds reported.",
            "placement_control": "Only generated parent endpoints differ; real node selection and real reply edges fixed.",
            "test_status": "Existing test split previously inspected; these are exploratory, not untouched confirmatory results.",
        })
        entries = list(upstream)
        for seed in seeds:
            arm = "generated_root_typed_concat"
            status("placement_validation", f"{arm}, seed={seed}; no test")
            arm_dir = destination / f"placement_seed{seed}"
            path = arm_dir / "validation.json"
            run("train.py", ["--config", config, "--mode", "real_generated", "--graph-structure", "generated_root",
                "--message-passing", "typed_rgcn", "--pooling-mode", "mean", "--decision-mode", "concat",
                "--seed", seed, "--checkpoint-dir", arm_dir, "--summary-json", path, "--allow-formal-training"],
                destination / f"placement_seed{seed}.log")
            summary = read(path)
            validate_summary(summary, arm, seed)
            entries.append((arm, seed, path, summary))
        validation_ids = None
        frozen = []
        for arm, seed, path, summary in entries:
            validate_checkpoint(summary)
            ids = sorted((r["news_id"], r["label"]) for r in summary["validation"]["prediction_records"])
            if validation_ids is None:
                validation_ids = ids
            if ids != validation_ids:
                raise ValueError("Unpaired validation news coverage")
            frozen.append({"arm": arm, "seed": seed, "validation_summary": str(path),
                           "checkpoint": summary["checkpoint"], "checkpoint_sha256": sha256(summary["checkpoint"]),
                           "validation_macro_f1": summary["validation"]["macro_f1"],
                           "output": str(destination / f"test_{arm}_seed{seed}.json")})
        # Persist all checkpoint identities before the first test is read.
        write(destination / "frozen_test_plan.json", frozen)
        records = []
        test_ids = None
        for entry in frozen:
            status("frozen_test", f"{entry['arm']}, seed={entry['seed']}")
            if sha256(entry["checkpoint"]) != entry["checkpoint_sha256"]:
                raise ValueError("Frozen checkpoint changed")
            run("evaluate_frozen_checkpoint.py", ["--config", config, "--checkpoint", entry["checkpoint"],
                "--output-json", entry["output"]], destination / f"test_{entry['arm']}_seed{entry['seed']}.log")
            result = read(entry["output"])
            ids = sorted((r["news_id"], r["label"]) for r in result["test"]["prediction_records"])
            if test_ids is None:
                test_ids = ids
            if ids != test_ids:
                raise ValueError("Unpaired test news coverage")
            records.append({**entry, "test_size": result["test_size"],
                            "within_checkpoint_correction": correction_effect(result["test"]["prediction_records"]),
                            **{key: result["test"][key] for key in ("accuracy", "macro_f1", "weighted_f1")}})
            write(destination / "test_results.json", records)
        aggregates = {}
        for arm in ARMS:
            rows = [r for r in records if r["arm"] == arm]
            aggregates[arm] = {key: {"mean": statistics.mean(r[key] for r in rows),
                                      "sample_std": statistics.stdev(r[key] for r in rows)}
                               for key in ("accuracy", "macro_f1", "weighted_f1")}
        comparisons = []
        for left, right in [("generated_tree_typed_concat", "real_tree_typed_concat"),
                            ("generated_tree_typed_global_local", "real_tree_typed_global_local"),
                            ("generated_tree_typed_concat", "generated_root_typed_concat"),
                            ("generated_tree_typed_global_local", "generated_tree_typed_concat")]:
            deltas = [next(r["macro_f1"] for r in records if r["arm"] == left and r["seed"] == s) -
                      next(r["macro_f1"] for r in records if r["arm"] == right and r["seed"] == s) for s in seeds]
            comparisons.append({"left": left, "right": right, "seeds": seeds,
                                "paired_macro_f1_deltas": deltas, "mean_delta": statistics.mean(deltas)})
        write(destination / "aggregate_results.json", {"seeds": seeds, "test_size": len(test_ids),
              "per_arm": aggregates, "paired_comparisons": comparisons,
              "caution": "Exploratory small-sample results; standard deviation across three seeds is not a confidence interval."})
        status("complete", "All 15 frozen evaluations saved, including negative results")
    except Exception as exc:
        status("failed", str(exc))
        raise


if __name__ == "__main__":
    main()
