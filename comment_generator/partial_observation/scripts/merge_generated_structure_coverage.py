"""Merge original and hard-context graphs without changing either source."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from common import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-csv", type=Path, required=True)
    parser.add_argument("--existing-jsonl", type=Path, required=True)
    parser.add_argument("--missing-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--stats-json", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "validate", "test"], required=True)
    parser.add_argument("--expected-depth-reference", choices=["news_root", "observed_boundary"],
                        required=True)
    args = parser.parse_args()
    with args.manifest_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    expected = {str(row["news_id"]) for row in manifest
                if row["split"] == args.split and str(row["image_valid"]).lower() == "true"
                and str(row["news_text"]).strip() and Path(row["image_path"]).is_file()}
    if not expected:
        raise RuntimeError("No split-valid images/news in manifest")
    sources = (("original_relevance_filtered", read_jsonl(args.existing_jsonl)),
               ("additional_label_blind_context", read_jsonl(args.missing_jsonl)))
    merged = {}
    source_counts = Counter()
    for source_name, rows in sources:
        for row in rows:
            news_id = str(row.get("news_id", ""))
            if row.get("official_split") != args.split or not news_id or news_id in merged:
                raise ValueError(f"Split mismatch, duplicate or blank news ID: {source_name}/{news_id}")
            if row.get("label_used_for_generation") is not False:
                raise ValueError(f"Generation label-use flag invalid: {source_name}/{news_id}")
            generation = row.get("generation_config", {})
            if generation.get("depth_reference") != args.expected_depth_reference:
                raise ValueError(f"Depth-policy mismatch: {source_name}/{news_id}: "
                                 f"{generation.get('depth_reference')}")
            if int(generation.get("nodes_per_graph", -1)) != 5 or int(generation.get("max_generated_depth", -1)) != 6:
                raise ValueError(f"Generation-count/depth mismatch: {source_name}/{news_id}")
            merged[news_id] = row
            source_counts[source_name] += 1
    if set(merged) != expected:
        raise ValueError(f"Coverage mismatch: missing={sorted(expected - set(merged))[:10]}, "
                         f"extra={sorted(set(merged) - expected)[:10]}")
    rows = [merged[news_id] for news_id in sorted(merged)]
    content = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.output_jsonl.exists() and args.output_jsonl.read_text(encoding="utf-8") != content:
        raise FileExistsError(f"Existing merged output differs; refusing overwrite: {args.output_jsonl}")
    args.output_jsonl.write_text(content, encoding="utf-8")
    result = {
        "split": args.split, "all_valid_image_news": len(expected), "merged_graphs": len(rows),
        "source_counts": dict(source_counts),
        "depth_reference": args.expected_depth_reference,
        "generated_node_count_distribution": dict(sorted(Counter(
            len(row["generated_nodes"]) for row in rows).items())),
        "generation_failure_attempts": sum(len(row["generation_failures"]) for row in rows),
        "label_policy": "No news veracity label used for generation or merge.",
        "source_caveat": "Additional contexts were previously excluded by the relevance filter and may be noisy.",
    }
    args.stats_json.parent.mkdir(parents=True, exist_ok=True)
    args.stats_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
