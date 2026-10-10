from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image

from common import LABEL_MAPPINGS, label_text, load_config, read_jsonl, resolve_path, write_jsonl


INVALID_COMMENTS = {"[deleted]", "[removed]"}
WHITESPACE = re.compile(r"\s+")


def parse_args() -> argparse.Namespace:
    default_config = Path(__file__).resolve().with_name("config.yaml")
    parser = argparse.ArgumentParser(description="Build Qwen-VL comment SFT JSONL files.")
    parser.add_argument("--config", type=Path, default=default_config)
    return parser.parse_args()


def clean_comment(value: object) -> str:
    return WHITESPACE.sub(" ", str(value)).strip()


def stable_sample(values: list[str], limit: int, seed: int, news_id: str) -> list[str]:
    if len(values) <= limit:
        return values
    digest = hashlib.sha256(f"{seed}:{news_id}".encode("utf-8")).digest()
    local_seed = int.from_bytes(digest[:8], "big")
    return random.Random(local_seed).sample(values, limit)


def image_is_valid(path: Path) -> tuple[bool, str]:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
        return True, ""
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def main() -> None:
    args = parse_args()
    config, config_path = load_config(args.config)
    data_config = config["data"]
    seed = int(config.get("seed", 42))
    label_column = str(data_config["label_column"])
    if label_column not in LABEL_MAPPINGS:
        raise ValueError(f"Unsupported label column: {label_column}")

    samples_path = resolve_path(config_path, data_config["samples_csv"])
    comments_path = resolve_path(config_path, data_config["comments_jsonl"])
    output_dir = resolve_path(config_path, data_config["prepared_dir"])
    assert samples_path is not None and comments_path is not None and output_dir is not None
    if not samples_path.is_file() or not comments_path.is_file():
        raise FileNotFoundError(f"Missing input: {samples_path} or {comments_path}")

    samples = pd.read_csv(samples_path, dtype=str, keep_default_na=False)
    required = {"news_id", "split", "headline", "image_path", label_column}
    missing = required.difference(samples.columns)
    if missing:
        raise ValueError(f"Missing columns in {samples_path}: {sorted(missing)}")
    if samples["news_id"].duplicated().any():
        duplicates = samples.loc[samples["news_id"].duplicated(), "news_id"].tolist()
        raise ValueError(f"Duplicate news IDs in usable samples: {duplicates[:10]}")

    comments_by_news: dict[str, list[str]] = {}
    for record in read_jsonl(comments_path):
        news_id = str(record.get("news_id", "")).strip()
        if not news_id:
            continue
        comments_by_news[news_id] = list(record.get("comments", []))

    max_comments = int(data_config["max_comments_per_news"])
    min_chars = int(data_config["min_comment_chars"])
    max_chars = int(data_config["max_comment_chars"])
    prepared: dict[str, list[dict]] = {"train": [], "validate": [], "test": []}
    failure_counts: Counter[str] = Counter()
    per_news_counts: dict[str, int] = {}

    for row in samples.to_dict("records"):
        news_id = str(row["news_id"]).strip()
        split_value = str(row["split"]).strip().lower()
        split = {"val": "validate", "validation": "validate"}.get(split_value, split_value)
        if split not in prepared:
            failure_counts["unknown_split"] += 1
            continue
        title = str(row["headline"]).strip()
        if not title:
            failure_counts["missing_title"] += 1
            continue
        image_path = Path(str(row["image_path"]))
        if not image_path.is_absolute():
            image_path = (samples_path.parent / image_path).resolve()
        if not image_path.is_file():
            failure_counts["image_missing"] += 1
            continue
        valid_image, _ = image_is_valid(image_path)
        if not valid_image:
            failure_counts["image_invalid"] += 1
            continue

        raw_label = str(row[label_column]).strip()
        try:
            readable_label = label_text(label_column, raw_label)
        except ValueError:
            failure_counts["invalid_label"] += 1
            continue

        seen: set[str] = set()
        comments: list[str] = []
        for value in comments_by_news.get(news_id, []):
            comment = clean_comment(value)
            if (
                not comment
                or comment.lower() in INVALID_COMMENTS
                or len(comment) < min_chars
            ):
                continue
            comment = comment[:max_chars].rstrip()
            key = comment.casefold()
            if key in seen:
                continue
            seen.add(key)
            comments.append(comment)
        if not comments:
            failure_counts["no_valid_comments"] += 1
            continue

        selected = stable_sample(comments, max_comments, seed, news_id)
        per_news_counts[news_id] = len(selected)
        for comment_index, comment in enumerate(selected):
            prepared[split].append(
                {
                    "sample_id": f"{news_id}:{comment_index}",
                    "news_id": news_id,
                    "split": split,
                    "title": title,
                    "image_path": str(image_path.resolve()),
                    "label_column": label_column,
                    "label": int(raw_label),
                    "label_text": readable_label,
                    "target_comment": comment,
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    news_counts = {}
    example_label_counts = {}
    news_label_counts = {}
    for split, records in prepared.items():
        output_path = output_dir / f"{split}.jsonl"
        counts[split] = write_jsonl(output_path, records)
        news_counts[split] = len({record["news_id"] for record in records})
        example_label_counts[split] = dict(
            sorted(Counter(str(record["label"]) for record in records).items())
        )
        news_label_counts[split] = dict(
            sorted(
                Counter(
                    str(label)
                    for _, label in {
                        record["news_id"]: record["label"] for record in records
                    }.items()
                ).items()
            )
        )

    stats = {
        "seed": seed,
        "label_column": label_column,
        "label_mapping": LABEL_MAPPINGS[label_column],
        "source_news_rows": len(samples),
        "output_examples": counts,
        "output_unique_news": news_counts,
        "example_label_counts": example_label_counts,
        "news_label_counts": news_label_counts,
        "max_comments_per_news": max_comments,
        "selected_comments_total": sum(per_news_counts.values()),
        "failure_counts": dict(failure_counts),
        "official_split_preserved": True,
    }
    (output_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
