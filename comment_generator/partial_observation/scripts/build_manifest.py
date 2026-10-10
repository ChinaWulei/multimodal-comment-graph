from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from common import (
    IMAGE_EXTENSIONS,
    INVALID_COMMENT_BODIES,
    LABEL_COLUMNS,
    is_obvious_automated_comment,
    iter_comment_records,
    load_config,
    normalize_reddit_id,
    resolve_path,
    validate_image,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    default = Path(__file__).resolve().parents[1] / "config" / "generator.yaml"
    parser = argparse.ArgumentParser(description="Build image/news/comment manifest.")
    parser.add_argument("--config", type=Path, default=default)
    parser.add_argument("--progress-every", type=int, default=500_000)
    return parser.parse_args()


def read_news(config: dict, config_path: Path) -> tuple[pd.DataFrame, dict]:
    frames = []
    stats = {"rows": {}, "duplicate_id_rows": {}, "overlap_ids": {}}
    columns = ["id", "clean_title", "image_url", *LABEL_COLUMNS]
    for split, value in config["source"]["news_tsv"].items():
        path = resolve_path(config_path, value)
        frame = pd.read_csv(
            path,
            sep="\t",
            usecols=columns,
            dtype=str,
            keep_default_na=False,
            low_memory=False,
        )
        frame["split"] = split
        frame["_news_id"] = frame["id"].map(normalize_reddit_id)
        frames.append(frame)
        stats["rows"][split] = len(frame)
        stats["duplicate_id_rows"][split] = int(
            frame["_news_id"].duplicated(keep=False).sum()
        )
    news = pd.concat(frames, ignore_index=True)
    split_sets = {
        split: set(news.loc[news["split"] == split, "_news_id"])
        for split in config["source"]["news_tsv"]
    }
    names = list(split_sets)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            stats["overlap_ids"][f"{left}-{right}"] = len(
                split_sets[left] & split_sets[right]
            )
    return news, stats


def main() -> None:
    args = parse_args()
    config, config_path = load_config(args.config)
    images_dir = resolve_path(config_path, config["source"]["images_dir"])
    comments_path = resolve_path(config_path, config["source"]["comments_tsv"])
    output_dir = resolve_path(config_path, config["data"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    min_total = int(config["partial_observation"]["min_total_comments"])

    news, news_stats = read_news(config, config_path)
    news_groups = {key: group for key, group in news.groupby("_news_id", sort=False)}
    image_paths = sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    target_ids = {normalize_reddit_id(path.stem) for path in image_paths}

    raw_comment_count: Counter[str] = Counter()
    valid_comment_count: Counter[str] = Counter()
    raw_top_count: Counter[str] = Counter()
    valid_top_count: Counter[str] = Counter()
    valid_reply_count: Counter[str] = Counter()
    automated_comment_count: Counter[str] = Counter()
    top_comments: dict[str, list[dict]] = defaultdict(list)
    seen_bodies: dict[str, set[str]] = defaultdict(set)
    seen_ids: set[str] = set()
    source_records = malformed = 0

    for record, _ in iter_comment_records(comments_path):
        source_records += 1
        if record is None:
            malformed += 1
            continue
        news_id = record["submission_id"]
        if news_id not in target_ids:
            if source_records % args.progress_every == 0:
                print(f"records={source_records:,} target_top={sum(valid_top_count.values()):,}")
            continue
        raw_comment_count[news_id] += 1
        if record["is_top_level"]:
            raw_top_count[news_id] += 1
        body = record["body"]
        if not body or body.casefold() in INVALID_COMMENT_BODIES:
            continue
        valid_comment_count[news_id] += 1
        if not record["is_top_level"]:
            valid_reply_count[news_id] += 1
            continue
        if bool(config["partial_observation"].get("exclude_obvious_automated_comments", True)) and is_obvious_automated_comment(
            record["author"], body
        ):
            automated_comment_count[news_id] += 1
            continue
        body_key = body.casefold()
        if body_key in seen_bodies[news_id]:
            continue
        comment_id = record["comment_id"]
        if comment_id and comment_id in seen_ids:
            continue
        seen_bodies[news_id].add(body_key)
        if comment_id:
            seen_ids.add(comment_id)
        valid_top_count[news_id] += 1
        top_comments[news_id].append(record)
        if source_records % args.progress_every == 0:
            print(f"records={source_records:,} target_top={sum(valid_top_count.values()):,}")

    manifest_rows = []
    comment_pool_rows = []
    for image_path in image_paths:
        news_id = normalize_reddit_id(image_path.stem)
        image_info = validate_image(image_path)
        group = news_groups.get(news_id)
        reasons = []
        if not image_info["image_valid"]:
            reasons.append("image_invalid")
        if group is None or group.empty:
            reasons.append("news_not_found")
            row = None
        elif len(group) != 1:
            reasons.append("duplicate_news_rows")
            row = group.iloc[0]
        else:
            row = group.iloc[0]
        if row is not None and not str(row["clean_title"]).strip():
            reasons.append("missing_text")
        if row is not None and any(not str(row[column]).strip() for column in LABEL_COLUMNS):
            reasons.append("missing_label")
        if valid_top_count[news_id] < min_total:
            reasons.append("top_level_comments_lt_min")
        eligible = not reasons
        manifest_rows.append(
            {
                "news_id": news_id,
                "split": "" if row is None else row["split"],
                "news_text": "" if row is None else row["clean_title"],
                "image_path": str(image_path.resolve()),
                "image_url": "" if row is None else row["image_url"],
                "image_valid": bool(image_info["image_valid"]),
                "image_format": image_info["image_format"],
                "image_width": image_info["image_width"],
                "image_height": image_info["image_height"],
                "2_way_label": "" if row is None else row["2_way_label"],
                "3_way_label": "" if row is None else row["3_way_label"],
                "6_way_label": "" if row is None else row["6_way_label"],
                "raw_comment_count": raw_comment_count[news_id],
                "valid_comment_count": valid_comment_count[news_id],
                "raw_top_level_count": raw_top_count[news_id],
                "valid_top_level_dedup_count": valid_top_count[news_id],
                "valid_reply_count": valid_reply_count[news_id],
                "excluded_automated_top_level_count": automated_comment_count[news_id],
                "eligible": eligible,
                "failure_reason": ";".join(reasons),
            }
        )
        if row is not None:
            comment_pool_rows.append(
                {
                    "news_id": news_id,
                    "split": row["split"],
                    "comments": top_comments.get(news_id, []),
                }
            )

    manifest = pd.DataFrame(manifest_rows).sort_values(["split", "news_id"])
    manifest_path = output_dir / config["data"]["manifest_csv"]
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    pool_path = output_dir / config["data"]["comment_pool_jsonl"]
    write_jsonl(pool_path, comment_pool_rows)

    eligible = manifest[manifest["eligible"]]
    stats = {
        "source_records_parsed": source_records,
        "malformed_comment_records": malformed,
        "images": len(manifest),
        "valid_images": int(manifest["image_valid"].sum()),
        "matched_news": int((manifest["split"] != "").sum()),
        "news_with_any_valid_comments": int((manifest["valid_comment_count"] > 0).sum()),
        "news_with_top_level_comments": int(
            (manifest["valid_top_level_dedup_count"] > 0).sum()
        ),
        "excluded_automated_top_level_comments": int(
            manifest["excluded_automated_top_level_count"].sum()
        ),
        "eligible_news": len(eligible),
        "eligible_by_split": eligible["split"].value_counts().sort_index().to_dict(),
        "eligible_label_distribution": {
            column: eligible[column].value_counts().sort_index().to_dict()
            for column in LABEL_COLUMNS
        },
        "eligible_top_level_comment_mean": float(
            eligible["valid_top_level_dedup_count"].mean()
        )
        if len(eligible)
        else 0.0,
        "eligible_top_level_comment_max": int(
            eligible["valid_top_level_dedup_count"].max()
        )
        if len(eligible)
        else 0,
        "eligible_all_valid_comment_mean": float(eligible["valid_comment_count"].mean())
        if len(eligible)
        else 0.0,
        "eligible_all_valid_comment_max": int(eligible["valid_comment_count"].max())
        if len(eligible)
        else 0,
        "failure_reasons": manifest.loc[~manifest["eligible"], "failure_reason"]
        .value_counts()
        .to_dict(),
        "news_table_checks": news_stats,
        "manifest": str(manifest_path),
        "comment_pool": str(pool_path),
    }
    stats_path = output_dir / "manifest_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
