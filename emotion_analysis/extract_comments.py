from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterator

import pandas as pd

from common import (
    load_config,
    normalize_reddit_id,
    prefixed_id_type,
    resolve_path,
    write_jsonl,
)


RECORD_START = re.compile(r"^(\d+)\t([A-Za-z0-9]+)\t")
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
WHITESPACE = re.compile(r"\s+")
INVALID_BODIES = {"[deleted]", "[removed]"}


def parse_args() -> argparse.Namespace:
    default_config = Path(__file__).resolve().with_name("config.yaml")
    parser = argparse.ArgumentParser(
        description="Extract Fakeddit comments with parent metadata for usable news."
    )
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--progress-every", type=int, default=500_000)
    return parser.parse_args()


def decode_tsv_field(value: str) -> str:
    if len(value) >= 2 and value.startswith('"') and value.endswith('"'):
        return value[1:-1].replace('""', '"')
    return value


def clean_body(value: str) -> str:
    text = CONTROL_CHARS.sub(" ", decode_tsv_field(value))
    return WHITESPACE.sub(" ", text).strip()


def parse_record(raw: str, source_index: int) -> dict | None:
    text = raw.rstrip("\r\n")
    left = text.split("\t", 3)
    if len(left) != 4:
        return None
    tail = left[3].rsplit("\t", 4)
    if len(tail) != 5:
        return None
    body, is_top_level, parent_id, submission_id, ups = tail
    comment_id = decode_tsv_field(left[1]).strip()
    author = decode_tsv_field(left[2]).strip()
    parent_raw = decode_tsv_field(parent_id).strip()
    submission_raw = decode_tsv_field(submission_id).strip()
    try:
        ups_value = float(decode_tsv_field(ups).strip())
    except ValueError:
        ups_value = None
    return {
        "source_index": source_index,
        "comment_id": normalize_reddit_id(comment_id),
        "author": author,
        "body": clean_body(body),
        "is_top_level": decode_tsv_field(is_top_level).strip().lower()
        in {"true", "1", "yes"},
        "parent_id": normalize_reddit_id(parent_raw),
        "parent_type": prefixed_id_type(parent_raw),
        "submission_id": normalize_reddit_id(submission_raw),
        "ups": ups_value,
    }


def iter_records(path: Path) -> Iterator[tuple[dict | None, int]]:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        handle.readline()
        current_lines: list[str] = []
        current_index: int | None = None
        for line in handle:
            match = RECORD_START.match(line)
            candidate_index = int(match.group(1)) if match else None
            is_boundary = (
                candidate_index is not None
                and (current_index is None or candidate_index > current_index)
            )
            if is_boundary:
                if current_index is not None:
                    yield parse_record("".join(current_lines), current_index), current_index
                current_lines = [line]
                current_index = candidate_index
            elif current_index is not None:
                current_lines.append(line)
        if current_index is not None:
            yield parse_record("".join(current_lines), current_index), current_index


def main() -> None:
    args = parse_args()
    config, config_path = load_config(args.config)
    samples_path = resolve_path(config_path, config["source"]["usable_samples_csv"])
    comments_path = resolve_path(config_path, config["source"]["comments_tsv"])
    data_dir = resolve_path(config_path, config["data"]["directory"])
    output_path = data_dir / config["data"]["extracted_comments"]
    stats_path = data_dir / config["data"]["extraction_stats"]
    if not samples_path.is_file() or not comments_path.is_file():
        raise FileNotFoundError(f"Missing source: {samples_path} or {comments_path}")

    samples = pd.read_csv(samples_path, dtype=str, keep_default_na=False)
    required = {"news_id", "split", "headline", "2_way_label", "3_way_label", "6_way_label"}
    missing = required.difference(samples.columns)
    if missing:
        raise ValueError(f"Missing usable-sample fields: {sorted(missing)}")
    samples["_normalized_id"] = samples["news_id"].map(normalize_reddit_id)
    if samples["_normalized_id"].duplicated().any():
        raise ValueError("Duplicate news IDs in usable_samples.csv")
    target_ids = set(samples["_normalized_id"])

    extracted = []
    total_records = 0
    malformed_records = 0
    target_raw_records = 0
    invalid_body_records = 0
    duplicate_comment_ids = 0
    duplicate_body_records = 0
    seen_comment_ids: set[str] = set()
    seen_bodies: dict[str, set[str]] = defaultdict(set)
    source_first = None
    source_last = None

    for record, source_index in iter_records(comments_path):
        total_records += 1
        source_first = source_index if source_first is None else source_first
        source_last = source_index
        if record is None:
            malformed_records += 1
            continue
        if record["submission_id"] not in target_ids:
            if total_records % args.progress_every == 0:
                print(
                    f"records={total_records:,} extracted={len(extracted):,}", flush=True
                )
            continue
        target_raw_records += 1
        body = record["body"]
        if not body or body.lower() in INVALID_BODIES:
            invalid_body_records += 1
            continue
        comment_id = record["comment_id"]
        if comment_id and comment_id in seen_comment_ids:
            duplicate_comment_ids += 1
            continue
        if comment_id:
            seen_comment_ids.add(comment_id)
        body_key = body.casefold()
        record["duplicate_body"] = body_key in seen_bodies[record["submission_id"]]
        if record["duplicate_body"]:
            duplicate_body_records += 1
        seen_bodies[record["submission_id"]].add(body_key)
        extracted.append(record)
        if total_records % args.progress_every == 0:
            print(f"records={total_records:,} extracted={len(extracted):,}", flush=True)

    data_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, extracted)
    per_news = Counter(record["submission_id"] for record in extracted)
    top_level_count = sum(record["is_top_level"] for record in extracted)
    reply_count = len(extracted) - top_level_count
    known_comment_ids = {record["comment_id"] for record in extracted if record["comment_id"]}
    valid_reply_edges = sum(
        1
        for record in extracted
        if record["parent_type"] == "comment" and record["parent_id"] in known_comment_ids
    )
    missing_parent_edges = sum(
        1
        for record in extracted
        if record["parent_type"] == "comment" and record["parent_id"] not in known_comment_ids
    )
    stats = {
        "target_news": len(target_ids),
        "source_records_parsed": total_records,
        "source_index_first": source_first,
        "source_index_last": source_last,
        "malformed_records": malformed_records,
        "target_raw_records": target_raw_records,
        "invalid_body_records": invalid_body_records,
        "duplicate_comment_ids_skipped": duplicate_comment_ids,
        "valid_comment_records": len(extracted),
        "unique_news_with_comments": len(per_news),
        "top_level_comments": top_level_count,
        "reply_comments": reply_count,
        "duplicate_body_records_preserved": duplicate_body_records,
        "resolved_reply_edges": valid_reply_edges,
        "reply_edges_with_missing_parent": missing_parent_edges,
        "max_comments_for_one_news": max(per_news.values(), default=0),
        "output": str(output_path),
    }
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

