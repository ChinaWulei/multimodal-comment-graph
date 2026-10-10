"""Paired 292/77 Fakeddit validation report, stratified by context source."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean, stdev

from sklearn.metrics import accuracy_score, f1_score


SEEDS = (42, 13, 24)
ARMS = ("real_only", "generated_root", "generated_plain", "generated_gated")


def ids_from_graphs(path: Path, split: str) -> set[str]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [str(row["news_id"]) for row in rows]
    if len(ids) != len(set(ids)) or any(row["official_split"] != split for row in rows):
        raise ValueError(f"Duplicate IDs or split mismatch in {path}")
    return set(ids)


def metrics_on(records: list[dict], allowed: set[str]) -> dict:
    selected = [record for record in records if record["news_id"] in allowed]
    if len(selected) != len(allowed):
        raise ValueError("Validation stratum coverage is incomplete")
    labels = [record["label"] for record in selected]
    predictions = [record["prediction"] for record in selected]
    return {"news": len(selected), "class_counts": dict(Counter(map(str, labels))),
            "accuracy": accuracy_score(labels, predictions),
            "macro_f1": f1_score(labels, predictions, labels=[0, 1, 2],
                                 average="macro", zero_division=0)}


def paired_changes(old: list[dict], new: list[dict], allowed: set[str]) -> dict:
    corrected = introduced = 0
    for left, right in zip(old, new):
        if (left["news_id"], left["label"]) != (right["news_id"], right["label"]):
            raise ValueError("Unpaired validation records")
        if left["news_id"] not in allowed:
            continue
        corrected += left["prediction"] != left["label"] and right["prediction"] == right["label"]
        introduced += left["prediction"] == left["label"] and right["prediction"] != right["label"]
    return {"corrected": corrected, "introduced": introduced}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--original-graphs", type=Path, required=True)
    parser.add_argument("--additional-graphs", type=Path, required=True)
    parser.add_argument("--overlap-audit", type=Path,
                        help="Optional exact title/image reuse audit for a descriptive sensitivity stratum.")
    parser.add_argument("--generator-validation-sft", type=Path,
                        help="Generator SFT validation samples used for checkpoint selection; report a disjoint stratum.")
    args = parser.parse_args()
    original = ids_from_graphs(args.original_graphs, "validate")
    additional = ids_from_graphs(args.additional_graphs, "validate")
    if len(original) != 30 or len(additional) != 47 or original & additional:
        raise ValueError("Unexpected original/additional split coverage")
    strata = {"all": original | additional, "original_relevance_filtered": original,
              "additional_hard_context": additional}
    generator_validation_overlap = None
    if args.generator_validation_sft:
        rows = [json.loads(line) for line in args.generator_validation_sft.read_text(encoding="utf-8").splitlines()
                if line.strip()]
        if any(row["metadata"]["official_split"] != "validate" for row in rows):
            raise ValueError("Generator checkpoint-selection file contains non-validation news")
        selected_news = {str(row["metadata"]["news_id"]) for row in rows}
        generator_validation_overlap = selected_news & strata["all"]
        strata["not_used_for_generator_checkpoint_selection"] = strata["all"] - generator_validation_overlap
        if not strata["not_used_for_generator_checkpoint_selection"]:
            raise ValueError("No classifier-validation news are disjoint from generator validation")
    if args.overlap_audit:
        audit = json.loads(args.overlap_audit.read_text(encoding="utf-8"))
        if audit["train_news"] != 292 or audit["validation_news"] != 77:
            raise ValueError("Overlap audit is not for the 292/77 cohort")
        overlap_ids = set(audit["validation_exact_title_overlap_ids"])
        overlap_ids.update(row["validation_news_id"] for row in audit["validation_exact_image_overlap"])
        if not overlap_ids <= strata["all"]:
            raise ValueError("Overlap audit contains IDs outside validation")
        strata["without_exact_title_or_image_reuse"] = strata["all"] - overlap_ids
    runs = {}
    reference = None
    source_hashes = None
    for seed in SEEDS:
        runs[seed] = {}
        for arm in ARMS:
            path = args.run_dir / f"{arm}_seed{seed}.json"
            row = json.loads(path.read_text(encoding="utf-8"))
            if ("test" in row or row["seed"] != seed or row["train_size"] != 292 or
                    row["validate_size"] != 77 or row["max_real_comments"] != 16):
                raise ValueError(f"Unexpected run metadata: {path}")
            hashes = {split: row["generated_sources"][split]["sha256"]
                      for split in ("train", "validate")}
            if source_hashes is None:
                source_hashes = hashes
            elif hashes != source_hashes:
                raise ValueError(f"Generated sources changed across runs: {path}")
            records = row["validation"]["prediction_records"]
            pairs = [(record["news_id"], record["label"]) for record in records]
            if len(pairs) != 77 or {news_id for news_id, _ in pairs} != strata["all"]:
                raise ValueError(f"Wrong validation IDs: {path}")
            if reference is None:
                reference = pairs
            elif reference != pairs:
                raise ValueError(f"Unpaired IDs/labels across arms: {path}")
            runs[seed][arm] = row
    aggregate = {}
    for stratum, allowed in strata.items():
        aggregate[stratum] = {"news": len(allowed), "arms": {}}
        for arm in ARMS:
            values = [metrics_on(runs[seed][arm]["validation"]["prediction_records"], allowed)
                      for seed in SEEDS]
            f1 = [value["macro_f1"] for value in values]
            accuracy = [value["accuracy"] for value in values]
            aggregate[stratum]["arms"][arm] = {
                "class_counts": values[0]["class_counts"],
                "macro_f1_by_seed": dict(zip(map(str, SEEDS), f1)),
                "macro_f1_mean": mean(f1), "macro_f1_seed_std": stdev(f1),
                "accuracy_mean": mean(accuracy),
            }
        for arm in ARMS[1:]:
            aggregate[stratum]["arms"][arm]["paired_vs_real"] = {
                str(seed): paired_changes(
                    runs[seed]["real_only"]["validation"]["prediction_records"],
                    runs[seed][arm]["validation"]["prediction_records"], allowed)
                for seed in SEEDS}
        for arm in ("generated_plain", "generated_gated"):
            aggregate[stratum]["arms"][arm]["paired_vs_generated_root"] = {
                str(seed): paired_changes(
                    runs[seed]["generated_root"]["validation"]["prediction_records"],
                    runs[seed][arm]["validation"]["prediction_records"], allowed)
                for seed in SEEDS}
    report = {"dataset": "Fakeddit", "split": "official_validation",
              "train_news": 292, "validation_news": 77, "seeds": SEEDS,
              "test_evaluated": False, "source_sha256": source_hashes,
              "generator_checkpoint_selection_overlap_news": (len(generator_validation_overlap)
                                                              if generator_validation_overlap is not None else None),
              "strata": aggregate,
              "warning": "Exploratory 77-news validation; no fresh independent test claim. Additional contexts are selected failures of a previous relevance filter. Generator checkpoint selection used some official-validation comments; use the disjoint stratum when supplied."}
    json_out = args.run_dir / "aggregate_validation.json"
    json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# All-valid-image Fakeddit graph ablation (validation only)", "",
             "292 official-train / 77 official-validation news; 16 real and up to 5 generated comments, three paired seeds. No test evaluation.", ""]
    if generator_validation_overlap is not None:
        lines += [f"Generator checkpoint selection used real comments from {len(generator_validation_overlap)} of these 77 validation news. The disjoint stratum below contains {len(strata['not_used_for_generator_checkpoint_selection'])} news; it is still exploratory classifier validation, not an independent test.", ""]
    for stratum, group in aggregate.items():
        lines += [f"## {stratum} ({group['news']} news)", "",
                  "| Arm | Macro-F1 mean ± seed SD | Accuracy mean | Seed macro-F1 (42 / 13 / 24) |",
                  "| --- | ---: | ---: | --- |"]
        for arm, item in group["arms"].items():
            seed_scores = " / ".join(f"{item['macro_f1_by_seed'][str(seed)]:.4f}" for seed in SEEDS)
            lines.append(f"| {arm} | {item['macro_f1_mean']:.4f} ± {item['macro_f1_seed_std']:.4f} | "
                         f"{item['accuracy_mean']:.4f} | {seed_scores} |")
        lines += ["", "Paired corrected/introduced errors relative to real-only:", ""]
        for arm in ARMS[1:]:
            changes = group["arms"][arm]["paired_vs_real"]
            lines.append(f"- {arm}: " + ", ".join(
                f"{seed}: {changes[str(seed)]['corrected']}/{changes[str(seed)]['introduced']}"
                for seed in SEEDS))
        lines += ["", "Paired corrected/introduced errors relative to generated-root:", ""]
        for arm in ("generated_plain", "generated_gated"):
            changes = group["arms"][arm]["paired_vs_generated_root"]
            lines.append(f"- {arm}: " + ", ".join(
                f"{seed}: {changes[str(seed)]['corrected']}/{changes[str(seed)]['introduced']}"
                for seed in SEEDS))
        lines += [""]
    lines += ["Additional hard-context news were not randomly sampled; per-stratum metrics are descriptive. A larger untouched test and human content/parent review are required for a publication claim.", ""]
    md_out = args.run_dir / "aggregate_validation.md"
    md_out.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"markdown": str(md_out), "json": str(json_out),
                      "all_macro_f1_means": {arm: row["macro_f1_mean"]
                                             for arm, row in aggregate["all"]["arms"].items()}}, indent=2))


if __name__ == "__main__":
    main()
