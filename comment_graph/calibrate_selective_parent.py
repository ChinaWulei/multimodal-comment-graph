"""Train-only selective BGE-parent threshold; evaluate unchanged on validation.

Observed real reply edges are a weak proxy. This does not provide generated
comment quality labels or authorize automatic graph rewriting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def load(path: Path, split: str) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    rows = report.get("per_edge")
    if report.get("split") != split or not isinstance(rows, list) or len(rows) != report["edges"]:
        raise ValueError(f"Missing or wrong-split per-edge audit: {path}")
    if len({(row["news_id"], row["child_node_index"]) for row in rows}) != len(rows):
        raise ValueError("Duplicate observed reply edge")
    return report


def wilson(successes: int, total: int, z: float = 1.96) -> list[float] | None:
    if not total:
        return None
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def statistics(rows: list[dict], threshold: float | None) -> dict:
    eligible = [row for row in rows if int(row["candidate_count"]) >= 2]
    accepted = ([row for row in eligible if float(row["reranker_top_margin"]) >= threshold]
                if threshold is not None else [])
    correct = sum(row["reranker_top_correct"] is True for row in accepted)
    return {"eligible_edges": len(eligible), "accepted_edges": len(accepted),
            "coverage": len(accepted) / len(eligible) if eligible else 0.0,
            "accepted_correct": correct,
            "selective_top1_precision": correct / len(accepted) if accepted else None,
            "precision_wilson95": wilson(correct, len(accepted)),
            "all_eligible_top1": (sum(row["reranker_top_correct"] is True for row in eligible)
                                  / len(eligible) if eligible else None)}


def select_threshold(rows: list[dict], minimum_precision: float,
                     minimum_coverage: float) -> tuple[float | None, dict]:
    eligible = [row for row in rows if int(row["candidate_count"]) >= 2]
    if not eligible:
        raise ValueError("No nontrivial parent-choice edges")
    options = []
    for threshold in sorted({float(row["reranker_top_margin"]) for row in eligible}):
        result = statistics(eligible, threshold)
        precision = result["selective_top1_precision"]
        if (precision is not None and precision >= minimum_precision and
                result["coverage"] >= minimum_coverage):
            options.append((result["coverage"], precision, threshold, result))
    if not options:
        return None, statistics(eligible, None)
    _, _, threshold, result = max(options)
    return threshold, result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-json", type=Path, required=True)
    parser.add_argument("--validation-json", type=Path)
    parser.add_argument("--calibration-json", type=Path,
                        help="Existing train-only result; required with --validation-json")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--minimum-train-precision", type=float, default=0.60)
    parser.add_argument("--minimum-train-coverage", type=float, default=0.10)
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError(args.output_json)
    if not (0 < args.minimum_train_precision <= 1 and 0 < args.minimum_train_coverage <= 1):
        raise ValueError("Precision and coverage targets must be in (0,1]")
    train = load(args.train_json, "train")
    if args.validation_json:
        if not args.calibration_json:
            raise ValueError("Validation requires a pre-existing train-only calibration JSON")
        frozen = json.loads(args.calibration_json.read_text(encoding="utf-8"))
        if (frozen["train_source"] != str(args.train_json.resolve()) or
                frozen["train_sha256"] != hashlib.sha256(args.train_json.read_bytes()).hexdigest() or
                frozen["reranker_snapshot"] != train["reranker_snapshot"] or
                frozen["minimum_train_precision"] != args.minimum_train_precision or
                frozen["minimum_train_coverage"] != args.minimum_train_coverage):
            raise ValueError("Training threshold provenance or declared targets changed")
        validation = load(args.validation_json, "validate")
        if validation["reranker_snapshot"] != train["reranker_snapshot"]:
            raise ValueError("Train/validation reranker snapshots differ")
        if {row["news_id"] for row in train["per_edge"]} & {
                row["news_id"] for row in validation["per_edge"]}:
            raise ValueError("Train/validation news overlap")
        result = {"phase": "validation", "threshold": frozen["threshold"],
                  "train_calibration": str(args.calibration_json.resolve()),
                  "validation_sha256": hashlib.sha256(args.validation_json.read_bytes()).hexdigest(),
                  "train_metrics": frozen["train_metrics"],
                  "validation_metrics": statistics(validation["per_edge"], frozen["threshold"]),
                  "validation_raw_top1": validation["reranker_top1"],
                  "policy_available": frozen["threshold"] is not None,
                  "caveat": "Real-edge proxy only; not generated-parent correctness or classifier benefit."}
    else:
        threshold, metrics = select_threshold(train["per_edge"],
                                              args.minimum_train_precision,
                                              args.minimum_train_coverage)
        result = {"phase": "train_calibration", "train_source": str(args.train_json.resolve()),
                  "train_sha256": hashlib.sha256(args.train_json.read_bytes()).hexdigest(),
                  "reranker_snapshot": train["reranker_snapshot"],
                  "minimum_train_precision": args.minimum_train_precision,
                  "minimum_train_coverage": args.minimum_train_coverage,
                  "threshold": threshold, "train_metrics": metrics,
                  "policy_available": threshold is not None,
                  "caveat": "Threshold chosen only from observed real training edges; generated comments have no true parent."}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
