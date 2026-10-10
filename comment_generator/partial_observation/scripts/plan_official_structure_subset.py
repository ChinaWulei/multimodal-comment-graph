from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from common import (
    INVALID_COMMENT_BODIES,
    LABEL_COLUMNS,
    is_obvious_automated_comment,
    iter_comment_records,
    load_config,
    normalize_reddit_id,
    resolve_path,
)


def parse_args() -> argparse.Namespace:
    default = Path(__file__).resolve().parents[1] / "config" / "generator_structure.yaml"
    parser = argparse.ArgumentParser(
        description="Plan a split-preserving Fakeddit image subset with real reply-tree candidates."
    )
    parser.add_argument("--config", type=Path, default=default)
    parser.add_argument("--progress-every", type=int, default=1_000_000)
    return parser.parse_args()


def allocate(frame: pd.DataFrame, desired: int, label_column: str, mode: str) -> pd.DataFrame:
    """Select high-comment candidates with either source-proportional or balanced quotas."""
    if len(frame) <= desired:
        return frame.copy()
    groups = {label: group.copy() for label, group in frame.groupby(label_column, sort=True)}
    if mode == "source_proportional":
        total = len(frame)
        quotas = {label: int(round(desired * len(group) / total)) for label, group in groups.items()}
    elif mode == "balanced":
        labels = list(groups)
        base, remainder = divmod(desired, len(labels))
        quotas = {
            label: min(base + (index < remainder), len(groups[label]))
            for index, label in enumerate(labels)
        }
    else:
        raise ValueError("official_subset.allocation_mode must be source_proportional or balanced")
    # Correct rounding and cap quotas by what exists.
    for label in quotas:
        quotas[label] = min(quotas[label], len(groups[label]))
    while sum(quotas.values()) < desired:
        candidates = sorted(groups, key=lambda key: (len(groups[key]) - quotas[key], str(key)), reverse=True)
        label = next((key for key in candidates if quotas[key] < len(groups[key])), None)
        if label is None:
            break
        quotas[label] += 1
    selected = []
    for label, group in groups.items():
        selected.append(group.head(quotas[label]))
    return pd.concat(selected, ignore_index=True).sort_values("news_id", kind="stable")


def main() -> None:
    args = parse_args()
    config, config_path = load_config(args.config)
    source = config["source"]
    plan = config["official_subset"]
    output_dir = resolve_path(config_path, config["data"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    label_column = str(plan["stratify_label"])
    allocation_mode = str(plan.get("allocation_mode", "source_proportional"))
    if label_column not in LABEL_COLUMNS:
        raise ValueError(f"official_subset.stratify_label must be one of {LABEL_COLUMNS}")

    columns = ["id", "clean_title", "hasImage", "image_url", "num_comments", *LABEL_COLUMNS]
    frames = []
    for split, relative_path in source["news_tsv"].items():
        path = resolve_path(config_path, relative_path)
        frame = pd.read_csv(path, sep="\t", usecols=columns, dtype=str, keep_default_na=False, low_memory=False)
        frame["news_id"] = frame["id"].map(normalize_reddit_id)
        frame["split"] = split
        frame["has_image"] = frame["hasImage"].str.strip().str.casefold().isin({"true", "1", "yes"})
        frame["news_text"] = frame["clean_title"].str.strip()
        frame["image_url"] = frame["image_url"].str.strip()
        frame["reported_num_comments"] = pd.to_numeric(frame["num_comments"], errors="coerce").fillna(0).astype(int)
        frames.append(frame)
    news = pd.concat(frames, ignore_index=True)
    news = news[
        news["has_image"]
        & news["news_text"].ne("")
        & news["image_url"].ne("")
        & news[label_column].str.strip().ne("")
    ].copy()
    if news["news_id"].duplicated().any():
        raise RuntimeError("News IDs overlap after split loading; aborting to prevent split leakage")
    target_ids = set(news["news_id"])
    total = malformed = 0
    valid = Counter()
    top = Counter()
    reply = Counter()
    comments_path = resolve_path(config_path, source["comments_tsv"])
    for record, _ in iter_comment_records(comments_path):
        total += 1
        if record is None:
            malformed += 1
            continue
        news_id = record["submission_id"]
        if news_id not in target_ids:
            if args.progress_every and total % args.progress_every == 0:
                print(f"records={total:,} candidate_comments={sum(valid.values()):,}")
            continue
        body = record["body"]
        if not body or body.casefold() in INVALID_COMMENT_BODIES:
            continue
        if is_obvious_automated_comment(record["author"], body):
            continue
        valid[news_id] += 1
        if record["is_top_level"] or record["parent_id"] == news_id:
            top[news_id] += 1
        else:
            reply[news_id] += 1
        if args.progress_every and total % args.progress_every == 0:
            print(f"records={total:,} candidate_comments={sum(valid.values()):,}")
    news["valid_comment_count"] = news["news_id"].map(valid).fillna(0).astype(int)
    news["valid_top_level_count"] = news["news_id"].map(top).fillna(0).astype(int)
    news["valid_reply_count"] = news["news_id"].map(reply).fillna(0).astype(int)
    eligible = news[
        (news["valid_top_level_count"] >= int(plan["min_top_level_comments"]))
        & (news["valid_reply_count"] >= int(plan["min_reply_comments"]))
    ].copy()
    eligible = eligible.sort_values(
        ["split", "valid_reply_count", "valid_comment_count", "reported_num_comments", "news_id"],
        ascending=[True, False, False, False, True],
        kind="stable",
    )
    candidate_columns = [
        "news_id", "split", "news_text", "image_url", "reported_num_comments",
        "valid_comment_count", "valid_top_level_count", "valid_reply_count", *LABEL_COLUMNS,
    ]
    candidates_path = output_dir / "official_structure_candidates.csv"
    eligible[candidate_columns].to_csv(candidates_path, index=False, encoding="utf-8-sig")
    selected_stats = {}
    for split, desired in plan["target_news"].items():
        pool = eligible[eligible["split"] == split].copy()
        chosen = allocate(pool, int(desired), label_column, allocation_mode)
        path = output_dir / f"official_structure_subset_{split}.csv"
        chosen[candidate_columns].to_csv(path, index=False, encoding="utf-8-sig")
        selected_stats[split] = {
            "requested": int(desired), "available": len(pool), "selected": len(chosen),
            "path": str(path),
            "label_distribution": chosen[label_column].value_counts().sort_index().to_dict(),
        }
    summary = {
        "comments_scanned": total,
        "malformed_comments": malformed,
        "news_with_text_image_url": len(news),
        "eligible_with_real_reply_candidates": len(eligible),
        "eligible_by_split": eligible["split"].value_counts().sort_index().to_dict(),
        "candidate_csv": str(candidates_path),
        "selected_subsets": selected_stats,
        "selection_policy": {
            "min_top_level_comments": int(plan["min_top_level_comments"]),
            "min_reply_comments": int(plan["min_reply_comments"]),
            "stratify_label": label_column,
            "allocation_mode": allocation_mode,
            "ranking": "valid_reply_count, valid_comment_count, reported_num_comments descending",
        },
        "image_download_started": False,
    }
    (output_dir / "official_structure_subset_plan.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
