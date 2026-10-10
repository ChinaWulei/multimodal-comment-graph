"""Independently recompute saved metrics and paired test-news uncertainty.

Does not load a model, retrain, select a checkpoint or modify predictions.
Bootstrap intervals condition on the three trained seeds; they do not account
for model-selection uncertainty or create a new untouched test set.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from comment_graph.finalize_fakeddit_structure import ARMS, read, sha256


def macro_f1(labels, predictions):
    matrix = np.bincount(3 * labels + predictions, minlength=9).reshape(3, 3)
    denominator = matrix.sum(0) + matrix.sum(1)
    return float(np.divide(2 * matrix.diagonal(), denominator,
                           out=np.zeros(3, dtype=float), where=denominator != 0).mean())


def paired_bootstrap(labels, left, right, repeats, seed):
    rng = np.random.default_rng(seed)
    strata = [np.flatnonzero(labels == c) for c in range(3)]
    samples = np.concatenate([rng.choice(s, size=(repeats, len(s)), replace=True) for s in strata], axis=1)
    differences = np.empty(repeats)
    for i, indices in enumerate(samples):
        differences[i] = np.mean([macro_f1(labels[indices], l[indices]) - macro_f1(labels[indices], r[indices])
                                  for l, r in zip(left, right)])
    return np.quantile(differences, [0.025, 0.975]).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--overlap-audit", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261006)
    args = parser.parse_args()
    if args.bootstrap_repeats < 1:
        raise ValueError("Bootstrap repeat count must be positive")
    if args.output_json.exists():
        raise FileExistsError("Use a new audit report path to preserve previous results")
    run = args.run_dir.resolve()
    if read(run / "status.json")["stage"] != "complete":
        raise RuntimeError("All frozen tests must finish before aggregate analysis")
    plan = read(run / "frozen_test_plan.json")
    seeds = [42, 13, 24]
    if {(r["arm"], r["seed"]) for r in plan} != {(a, s) for a in ARMS for s in seeds} or len(plan) != 15:
        raise ValueError("Missing or duplicate planned arm/seed")
    by_arm, common_ids, labels, source_hashes = {}, None, None, {}
    recomputed = []
    for entry in plan:
        if sha256(entry["checkpoint"]) != entry["checkpoint_sha256"]:
            raise ValueError("A frozen checkpoint changed after test planning")
        report = read(entry["output"])
        if Path(report["checkpoint"]).resolve() != Path(entry["checkpoint"]).resolve() or report["seed"] != entry["seed"]:
            raise ValueError("Wrong checkpoint evaluated")
        mode, graph, decision = ARMS[entry["arm"]]
        for key, value in {"mode": mode, "graph_structure": graph, "decision_mode": decision,
                           "real_node_selection": "observed_context", "max_real_comments": 64,
                           "max_generated_comments": 5}.items():
            if report[key] != value:
                raise ValueError(f"Wrong test setting: {key}")
        records = report["test"]["prediction_records"]
        indexed = {r["news_id"]: r for r in records}
        if len(indexed) != len(records) or len(records) != report["test_size"]:
            raise ValueError("Duplicate or missing per-news predictions")
        ids = sorted(indexed)
        y = np.asarray([indexed[i]["label"] for i in ids], dtype=int)
        pred = np.asarray([indexed[i]["prediction"] for i in ids], dtype=int)
        if any(v not in (0, 1, 2) for v in np.concatenate([y, pred])):
            raise ValueError("Unexpected class code")
        if common_ids is None:
            common_ids, labels = ids, y
        if ids != common_ids or not np.array_equal(y, labels):
            raise ValueError("Unpaired news or labels across models")
        values = {"accuracy": accuracy_score(y, pred),
                  "macro_f1": f1_score(y, pred, labels=[0, 1, 2], average="macro", zero_division=0),
                  "weighted_f1": f1_score(y, pred, labels=[0, 1, 2], average="weighted", zero_division=0)}
        for key, value in values.items():
            if not math.isclose(value, report["test"][key], abs_tol=1e-12, rel_tol=0):
                raise ValueError(f"Reported metric does not match predictions: {key}")
        if confusion_matrix(y, pred, labels=[0, 1, 2]).tolist() != report["test"]["confusion_matrix"]:
            raise ValueError("Confusion matrix mismatch")
        assert math.isclose(macro_f1(y, pred), values["macro_f1"], abs_tol=1e-12)
        by_arm.setdefault(entry["arm"], {})[entry["seed"]] = pred
        source_hashes[str(Path(entry["output"]).resolve())] = sha256(entry["output"])
        recomputed.append({"arm": entry["arm"], "seed": entry["seed"], **values})
    matrices = {arm: np.stack([rows[s] for s in seeds]) for arm, rows in by_arm.items()}
    pairs = [
        ("generated_tree_typed_concat", "real_tree_typed_concat", True),
        ("generated_tree_typed_global_local", "real_tree_typed_global_local", True),
        ("generated_tree_typed_concat", "generated_root_typed_concat", True),
        ("generated_tree_typed_global_local", "generated_tree_typed_concat", True),
        ("generated_root_typed_concat", "real_tree_typed_concat", False),
        ("real_tree_typed_global_local", "real_tree_typed_concat", False),
    ]
    comparisons = []
    for left, right, predeclared in pairs:
        lm, rm = matrices[left], matrices[right]
        per_seed = []
        for i, seed in enumerate(seeds):
            improved = [common_ids[j] for j in np.flatnonzero((lm[i] == labels) & (rm[i] != labels))]
            worsened = [common_ids[j] for j in np.flatnonzero((lm[i] != labels) & (rm[i] == labels))]
            per_seed.append({"seed": seed, "macro_f1_delta": macro_f1(labels, lm[i]) - macro_f1(labels, rm[i]),
                             "different_prediction_count": int(np.sum(lm[i] != rm[i])),
                             "improved_news_ids": improved, "worsened_news_ids": worsened})
        comparisons.append({"left": left, "right": right, "predeclared_in_finalization_plan": predeclared,
                            "mean_macro_f1_delta": float(np.mean([r["macro_f1_delta"] for r in per_seed])),
                            "conditional_news_bootstrap_95_interval": paired_bootstrap(labels, lm, rm, args.bootstrap_repeats, args.bootstrap_seed),
                            "per_seed": per_seed})
    overlap = read(args.overlap_audit)
    groups = overlap["scopes"]["generation_eligible_news"]["cross_split_groups"]["decoded_pixels_sha256"]
    excluded = sorted({r["news_id"] for g in groups for r in g["members"] if r["split"] == "test"} & set(common_ids))
    keep = np.asarray([i not in excluded for i in common_ids])
    if not keep.any():
        raise ValueError("Overlap sensitivity subset is empty")
    sensitivity = {arm: {"per_seed_macro_f1": [macro_f1(labels[keep], p[keep]) for p in matrix],
                          "mean_macro_f1": float(np.mean([macro_f1(labels[keep], p[keep]) for p in matrix]))}
                   for arm, matrix in matrices.items()}
    output = {"verified_models": len(plan), "test_size": len(common_ids), "test_class_counts": dict(Counter(map(int, labels))),
              "seeds": seeds, "metrics_recomputed_from_predictions": recomputed,
              "frozen_checkpoint_hashes_verified": True, "test_report_sha256": source_hashes,
              "paired_comparisons": comparisons,
              "bootstrap": {"repeats": args.bootstrap_repeats, "seed": args.bootstrap_seed,
                            "method": "Resample news WITHIN label strata, same news indices for all models and all fixed seeds; average three seed differences per replicate.",
                            "limits": "Conditional on these three trained runs. Does not include training/model-selection uncertainty. Intervals are not corrected for multiple comparisons; no new significance or untouched-test claim."},
              "shared_image_sensitivity": {"scope": "Post-hoc descriptive sensitivity; primary results remain all 49 news. Duplicate images were flagged before classification.",
                                            "excluded_test_ids": excluded, "remaining_news": int(keep.sum()), "per_arm": sensitivity},
              "caution": "These are exploratory results on a small, previously inspected test split. Same headline/byte checks do not rule out all dataset shortcuts."}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"verified_models": len(plan), "test_size": len(common_ids),
                      "comparisons": [{k: r[k] for k in ("left", "right", "mean_macro_f1_delta", "conditional_news_bootstrap_95_interval")} for r in comparisons],
                      "excluded_shared_images": excluded, "output": str(args.output_json.resolve())}, indent=2))


if __name__ == "__main__":
    main()
