from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import pandas as pd

from common import LABEL_COLUMNS, load_config, read_jsonl, resolve_path, stable_random, write_jsonl


def parse_args() -> argparse.Namespace:
    default = Path(__file__).resolve().parents[1] / "config" / "generator.yaml"
    parser = argparse.ArgumentParser(description="Build label-free Partial Observation SFT data.")
    parser.add_argument("--config", type=Path, default=default)
    parser.add_argument("--pool-jsonl", type=Path)
    parser.add_argument(
        "--output-prefix",
        default="",
        help="Prefix generated JSONL/stat filenames; useful for integration smoke tests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config, config_path = load_config(args.config)
    data_config = config["data"]
    output_dir = resolve_path(config_path, data_config["output_dir"])
    manifest_path = output_dir / data_config["manifest_csv"]
    pool_field = data_config.get("comment_pool_source", "comment_pool_jsonl")
    if pool_field not in data_config:
        raise ValueError(f"Unknown data.comment_pool_source: {pool_field}")
    pool_path = (
        args.pool_jsonl.resolve()
        if args.pool_jsonl
        else output_dir / data_config[pool_field]
    )
    pool_source_name = (
        f"override:{pool_path.name}" if args.pool_jsonl else pool_field
    )
    if not pool_path.is_file() or pool_path.stat().st_size == 0:
        raise RuntimeError(
            f"Required comment pool is missing or empty: {pool_path}. "
            "Run comment-tree distillation before building SFT data."
        )
    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    label_complete = manifest[LABEL_COLUMNS].apply(
        lambda column: column.astype(str).str.strip().ne("")
    ).all(axis=1)
    eligible = manifest[
        manifest["image_valid"].str.lower().eq("true")
        & manifest["news_text"].str.strip().ne("")
        & manifest["split"].str.strip().ne("")
        & label_complete
    ].copy()
    pool = {record["news_id"]: record for record in read_jsonl(pool_path)}

    protocol = config["partial_observation"]
    seed = int(config["seed"])
    ratio = float(protocol["observed_ratio"])
    min_total = int(protocol["min_total_comments"])
    max_observed = int(protocol["max_observed_comments"])
    max_targets = int(protocol.get("max_target_comments_per_news", 0))
    if not 0 < ratio < 1:
        raise ValueError("observed_ratio must be between 0 and 1")
    if min_total < 2 or max_observed < 1:
        raise ValueError("min_total_comments must be >=2 and max_observed_comments >=1")

    samples_by_split: dict[str, list[dict]] = {"train": [], "validate": [], "test": []}
    contexts = []
    gold_targets = []
    observed_counts = Counter()
    target_counts = Counter()
    skipped_news = Counter()

    for row in eligible.sort_values(["split", "news_id"]).to_dict("records"):
        news_id = row["news_id"]
        if news_id not in pool:
            skipped_news["not_in_selected_comment_pool"] += 1
            continue
        comments = list(pool[news_id]["comments"])
        if len(comments) < min_total:
            skipped_news["selected_comments_lt_min"] += 1
            continue
        observed_count = max(1, math.floor(len(comments) * ratio))
        observed_count = min(observed_count, max_observed, len(comments) - 1)
        rng = stable_random(seed, news_id)
        indices = list(range(len(comments)))
        rng.shuffle(indices)
        observed_indices = set(indices[:observed_count])
        target_indices = [index for index in range(len(comments)) if index not in observed_indices]
        if max_targets > 0 and len(target_indices) > max_targets:
            chosen = set(rng.sample(target_indices, max_targets))
            target_indices = [index for index in target_indices if index in chosen]
        observed_records = [
            comments[index] for index in range(len(comments)) if index in observed_indices
        ]
        target_records = [comments[index] for index in target_indices]
        labels = {column: row[column] for column in LABEL_COLUMNS}
        model_input = {
            "image_path": row["image_path"],
            "news_text": row["news_text"],
            "observed_comments": [record["body"] for record in observed_records],
        }
        metadata = {
            "news_id": news_id,
            "split": row["split"],
            "labels": labels,
            "observed_comment_ids": [record["comment_id"] for record in observed_records],
            "target_comment_ids": [record["comment_id"] for record in target_records],
            "total_top_level_comments": len(comments),
            "observed_comment_count": len(observed_records),
            "target_comment_count": len(target_records),
            "selection_method": "seeded_random_without_timestamp",
            "seed": seed,
            "comment_pool_source": pool_source_name,
        }
        contexts.append({"metadata": metadata, "model_input": model_input})
        gold_targets.append(
            {
                "news_id": news_id,
                "split": row["split"],
                "target_comments": target_records,
            }
        )
        split = row["split"]
        if split not in samples_by_split:
            raise ValueError(f"Unknown split {split!r} for {news_id}")
        for record in target_records:
            samples_by_split[split].append(
                {
                    "sample_id": f"{news_id}:{record['comment_id']}",
                    "metadata": {
                        **metadata,
                        "target_comment_id": record["comment_id"],
                        "target_author": record["author"],
                        "target_ups": record["ups"],
                        "target_parent_id": record["parent_id"],
                        "target_source_kind": record.get("source_kind", "raw_comment"),
                        "target_source_comment_ids": record.get(
                            "source_comment_ids", [record["comment_id"]]
                        ),
                        "target_relevance_score": record.get("relevance_score"),
                        "target_comment_type": record.get("comment_type"),
                    },
                    "model_input": model_input,
                    "target_comment": record["body"],
                }
            )
        observed_counts[len(observed_records)] += 1
        target_counts[len(target_records)] += 1

    output_names = {
        "train": data_config["train_jsonl"],
        "validate": data_config["val_jsonl"],
        "test": data_config["test_jsonl"],
    }
    for split, records in samples_by_split.items():
        write_jsonl(output_dir / f"{args.output_prefix}{output_names[split]}", records)
    write_jsonl(
        output_dir / f"{args.output_prefix}{data_config['news_contexts_jsonl']}", contexts
    )
    write_jsonl(
        output_dir / f"{args.output_prefix}{data_config['gold_targets_jsonl']}", gold_targets
    )

    stats = {
        "protocol": {
            "observed_ratio": ratio,
            "min_total_comments": min_total,
            "max_observed_comments": max_observed,
            "max_target_comments_per_news": max_targets,
            "seed": seed,
            "top_level_only": bool(protocol["top_level_only"]),
            "time_interpretation": "none; observed/target is a seeded random partition",
            "comment_pool_source": pool_source_name,
        },
        "eligible_news": len(contexts),
        "news_by_split": dict(Counter(record["metadata"]["split"] for record in contexts)),
        "sft_samples_by_split": {
            split: len(records) for split, records in samples_by_split.items()
        },
        "sft_samples_total": sum(len(records) for records in samples_by_split.values()),
        "observed_count_distribution": dict(sorted(observed_counts.items())),
        "target_count_distribution": dict(sorted(target_counts.items())),
        "label_location": "metadata.labels only",
        "model_input_fields": ["image_path", "news_text", "observed_comments"],
        "target_field": "target_comment",
        "skipped_news": dict(skipped_news),
    }
    (output_dir / f"{args.output_prefix}{data_config['stats_json']}").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
