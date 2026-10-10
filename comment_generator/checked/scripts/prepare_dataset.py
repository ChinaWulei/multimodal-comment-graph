from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
WHITESPACE = re.compile(r"\s+")
MEANINGFUL = re.compile(r"[A-Za-z\u4e00-\u9fff]")
URL = re.compile(r"https?://\S+", re.IGNORECASE)
BOILERPLATE = {
    "图片评论",
    "转发微博",
    "快转微博",
    "查看图片",
    "网页链接",
    "微博视频",
    "抱歉，此微博已被删除",
}


def parse_args() -> argparse.Namespace:
    default = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    parser = argparse.ArgumentParser(description="Prepare CHECKED Qwen-VL SFT data.")
    parser.add_argument("--config", type=Path, default=default)
    return parser.parse_args()


def resolve(config_path: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (config_path.parent / path).resolve()


def stable_rng(seed: int, key: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def normalize_text(value: Any) -> str:
    text = CONTROL_CHARS.sub(" ", str(value or ""))
    text = WHITESPACE.sub(" ", text).strip()
    return text.lstrip("：: ").strip()


def clean_comments(
    comments: Any,
    min_chars: int,
    max_chars: int,
    counters: Counter[str],
) -> list[dict[str, Any]]:
    if not isinstance(comments, list):
        return []
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(comments):
        counters["raw_comments"] += 1
        if not isinstance(item, dict):
            counters["non_object"] += 1
            continue
        body = normalize_text(item.get("text"))
        normalized = WHITESPACE.sub(" ", URL.sub("", body)).strip().casefold()
        if not body:
            counters["empty"] += 1
            continue
        if normalized in {"[deleted]", "[removed]"}:
            counters["deleted_removed"] += 1
            continue
        if normalized in BOILERPLATE:
            counters["boilerplate"] += 1
            continue
        if len(body) < min_chars:
            counters["too_short"] += 1
            continue
        if len(body) > max_chars:
            counters["too_long"] += 1
            continue
        if len(MEANINGFUL.findall(body)) < 2:
            counters["not_meaningful"] += 1
            continue
        if normalized in seen:
            counters["duplicate_within_news"] += 1
            continue
        seen.add(normalized)
        output.append(
            {
                "comment_id": str(item.get("id") or f"comment_{index}"),
                "author": str(item.get("user_id") or ""),
                "body": body,
                "date": str(item.get("date") or ""),
                "parent_id": "",
                "ups": None,
                "source_kind": "checked_direct_comment",
                "normalized_text": normalized,
            }
        )
        counters["clean_before_global_dedup"] += 1
    return output


def load_success_images(status_db: Path) -> dict[str, list[dict[str, Any]]]:
    connection = sqlite3.connect(status_db)
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        """
        SELECT news_id, label, image_index, local_path, width, height, image_format
        FROM downloads
        WHERE status='success'
        ORDER BY news_id, image_index
        """
    ).fetchall()
    connection.close()
    images: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        path = Path(row["local_path"])
        if not path.is_file() or path.stat().st_size <= 0:
            continue
        images[row["news_id"]].append(dict(row))
    return dict(images)


def stratified_split(
    records: list[dict[str, Any]],
    seed: int,
    train_ratio: float,
    validate_ratio: float,
) -> dict[str, str]:
    by_label: dict[str, list[str]] = defaultdict(list)
    for record in records:
        by_label[record["label"]].append(record["news_id"])
    assignments: dict[str, str] = {}
    for label, ids in sorted(by_label.items()):
        ids = sorted(ids)
        stable_rng(seed, f"split:{label}").shuffle(ids)
        total = len(ids)
        val_count = max(1, round(total * validate_ratio))
        test_count = max(1, round(total * (1.0 - train_ratio - validate_ratio)))
        if val_count + test_count >= total:
            raise ValueError(f"Not enough {label} news for three splits: {total}")
        for news_id in ids[:test_count]:
            assignments[news_id] = "test"
        for news_id in ids[test_count : test_count + val_count]:
            assignments[news_id] = "validate"
        for news_id in ids[test_count + val_count :]:
            assignments[news_id] = "train"
    return assignments


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed = int(config["seed"])
    source = config["source"]
    dataset_dir = resolve(config_path, source["dataset_dir"])
    status_db = resolve(config_path, source["image_status_db"])
    output_dir = resolve(config_path, config["data"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    preprocessing = config["preprocessing"]
    counters: Counter[str] = Counter()
    images = load_success_images(status_db)

    candidates: list[dict[str, Any]] = []
    for folder in (dataset_dir / "fake_news", dataset_dir / "real_news"):
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        for json_path in sorted(folder.glob("*.json")):
            counters["news_json"] += 1
            raw = json.loads(json_path.read_text(encoding="utf-8"))
            news_id = str(raw.get("id") or json_path.stem)
            label = str(raw.get("label") or folder.name.removesuffix("_news"))
            news_text = normalize_text(raw.get("text"))
            if not news_text:
                counters["news_missing_text"] += 1
                continue
            if news_id not in images:
                counters["news_missing_downloaded_image"] += 1
                continue
            comments = clean_comments(
                raw.get("comments"),
                int(preprocessing["min_comment_chars"]),
                int(preprocessing["max_comment_chars"]),
                counters,
            )
            if len(comments) < int(preprocessing["min_total_comments"]):
                counters["news_lt_min_clean_comments"] += 1
                continue
            candidates.append(
                {
                    "news_id": news_id,
                    "label": label,
                    "binary_label": 1 if label == "fake" else 0,
                    "news_text": news_text[: int(preprocessing["max_news_chars"])],
                    "news_text_original_chars": len(news_text),
                    "images": images[news_id],
                    "comments": comments,
                    "source_json": str(json_path.resolve()),
                }
            )

    split_cfg = config["split"]
    assignments = stratified_split(
        candidates,
        seed,
        float(split_cfg["train_ratio"]),
        float(split_cfg["validate_ratio"]),
    )

    # Give evaluation splits priority when identical generic comments occur in
    # multiple news items; duplicates are then removed from validation/train.
    priority = {"test": 0, "validate": 1, "train": 2}
    globally_seen: set[str] = set()
    selected_news: list[dict[str, Any]] = []
    max_candidates = int(preprocessing["max_candidate_comments_per_news"])
    for record in sorted(
        candidates,
        key=lambda value: (priority[assignments[value["news_id"]]], value["news_id"]),
    ):
        available = [
            comment
            for comment in record["comments"]
            if comment["normalized_text"] not in globally_seen
        ]
        counters["cross_news_duplicate_comments"] += len(record["comments"]) - len(available)
        rng = stable_rng(seed, f"comments:{record['news_id']}")
        if len(available) > max_candidates:
            chosen_indices = sorted(rng.sample(range(len(available)), max_candidates))
            available = [available[index] for index in chosen_indices]
            counters["comments_capped"] += len(record["comments"]) - max_candidates
        if len(available) < int(preprocessing["min_total_comments"]):
            counters["news_lt_min_after_global_dedup"] += 1
            continue
        for comment in available:
            globally_seen.add(comment["normalized_text"])
            comment.pop("normalized_text", None)
        record["comments"] = available
        record["split"] = assignments[record["news_id"]]
        selected_news.append(record)

    selected_news.sort(key=lambda value: (value["split"], value["news_id"]))
    data_cfg = config["data"]
    pools: list[dict[str, Any]] = []
    contexts: list[dict[str, Any]] = []
    gold_targets: list[dict[str, Any]] = []
    samples: dict[str, list[dict[str, Any]]] = {"train": [], "validate": [], "test": []}
    manifest_rows: list[dict[str, Any]] = []
    observed_distribution: Counter[int] = Counter()
    target_distribution: Counter[int] = Counter()

    for record in selected_news:
        comments = record["comments"]
        rng = stable_rng(seed, f"partition:{record['news_id']}")
        indices = list(range(len(comments)))
        rng.shuffle(indices)
        observed_count = max(1, math.floor(len(comments) * float(preprocessing["observed_ratio"])))
        observed_count = min(
            observed_count,
            int(preprocessing["max_observed_comments"]),
            len(comments) - 1,
        )
        observed_indices = set(indices[:observed_count])
        target_indices = [index for index in indices if index not in observed_indices]
        max_targets = int(preprocessing["max_target_comments_per_news"])
        target_indices = target_indices[:max_targets]
        observed = [comments[index] for index in range(len(comments)) if index in observed_indices]
        targets = [comments[index] for index in target_indices]
        image_paths = [item["local_path"] for item in record["images"]]
        labels = {
            "checked_label": record["label"],
            "binary_label": record["binary_label"],
        }
        model_input = {
            "image_path": image_paths[0],
            "news_text": record["news_text"],
            "observed_comments": [item["body"] for item in observed],
        }
        metadata = {
            "news_id": record["news_id"],
            "split": record["split"],
            "labels": labels,
            "all_image_paths": image_paths,
            "observed_comment_ids": [item["comment_id"] for item in observed],
            "target_comment_ids": [item["comment_id"] for item in targets],
            "selected_comment_count": len(comments),
            "observed_comment_count": len(observed),
            "target_comment_count": len(targets),
            "selection_method": "seeded_random_without_comment_engagement_scores",
            "seed": seed,
            "comment_pool_source": data_cfg["comment_pool_source"],
        }
        pools.append(
            {
                "news_id": record["news_id"],
                "split": record["split"],
                "labels": labels,
                "comments": comments,
            }
        )
        contexts.append({"metadata": metadata, "model_input": model_input})
        gold_targets.append(
            {
                "news_id": record["news_id"],
                "split": record["split"],
                "target_comments": targets,
            }
        )
        for target in targets:
            samples[record["split"]].append(
                {
                    "sample_id": f"{record['news_id']}:{target['comment_id']}",
                    "metadata": {
                        **metadata,
                        "target_comment_id": target["comment_id"],
                        "target_author": target["author"],
                        "target_date": target["date"],
                        "target_ups": None,
                        "target_parent_id": "",
                        "target_source_kind": target["source_kind"],
                    },
                    "model_input": model_input,
                    "target_comment": target["body"],
                }
            )
        observed_distribution[len(observed)] += 1
        target_distribution[len(targets)] += 1
        manifest_rows.append(
            {
                "news_id": record["news_id"],
                "split": record["split"],
                "label": record["label"],
                "binary_label": record["binary_label"],
                "news_text": record["news_text"],
                "primary_image_path": image_paths[0],
                "image_count": len(image_paths),
                "selected_comment_count": len(comments),
                "observed_comment_count": len(observed),
                "target_comment_count": len(targets),
                "source_json": record["source_json"],
            }
        )

    write_jsonl(output_dir / data_cfg["comment_pool_jsonl"], pools)
    write_jsonl(output_dir / data_cfg["news_contexts_jsonl"], contexts)
    write_jsonl(output_dir / data_cfg["gold_targets_jsonl"], gold_targets)
    write_jsonl(output_dir / data_cfg["train_jsonl"], samples["train"])
    write_jsonl(output_dir / data_cfg["val_jsonl"], samples["validate"])
    write_jsonl(output_dir / data_cfg["test_jsonl"], samples["test"])
    with (output_dir / data_cfg["manifest_csv"]).open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    news_by_split = Counter(record["split"] for record in selected_news)
    labels_by_split: dict[str, Counter[str]] = defaultdict(Counter)
    for record in selected_news:
        labels_by_split[record["split"]][record["label"]] += 1
    stats = {
        "protocol": {
            "comment_pool_source": data_cfg["comment_pool_source"],
            "seed": seed,
            "split_unit": "news_id",
            "split_ratios": split_cfg,
            "label_is_metadata_only": True,
            "comment_time_interpretation": "not used for observed/target partition",
            "primary_image_policy": "first image in official pic_url order",
            "comments_are_flat": True,
        },
        "source_news_candidates": len(candidates),
        "selected_news": len(selected_news),
        "news_by_split": dict(news_by_split),
        "labels_by_split": {
            split: dict(counter) for split, counter in sorted(labels_by_split.items())
        },
        "sft_samples_by_split": {split: len(rows) for split, rows in samples.items()},
        "sft_samples_total": sum(len(rows) for rows in samples.values()),
        "observed_count_distribution": dict(sorted(observed_distribution.items())),
        "target_count_distribution": dict(sorted(target_distribution.items())),
        "filter_counters": dict(counters),
        "model_input_fields": ["image_path", "news_text", "observed_comments"],
        "target_field": "target_comment",
    }
    (output_dir / data_cfg["stats_json"]).write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
