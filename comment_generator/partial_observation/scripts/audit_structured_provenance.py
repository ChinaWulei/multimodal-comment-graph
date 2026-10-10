"""Trace structure-SFT news and comments back to the configured Fakeddit sources.

Read-only with respect to datasets. Writes an audit report; does not train,
download, or use news labels to construct model inputs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from common import (iter_comment_records, load_config, normalize_reddit_id,
                    read_jsonl, resolve_path, validate_image)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    config, config_path = load_config(args.config)
    data_dir = resolve_path(config_path, config["data"]["output_dir"])
    manifest_path = data_dir / "manifest.csv"
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    manifest = {r["news_id"]: r for r in manifest_rows}
    issues = []
    if len(manifest) != len(manifest_rows):
        issues.append({"reason": "duplicate_manifest_news_id"})
    news = {}
    expected_comments = {}
    split_stats = {}
    hashes = {}

    def expect(comment_id, news_id, content, parent, sample_id):
        comment_id = normalize_reddit_id(comment_id)
        value = (news_id, content, normalize_reddit_id(parent))
        if comment_id in expected_comments and expected_comments[comment_id] != value:
            issues.append({"reason": "inconsistent_sft_comment", "comment_id": comment_id,
                           "sample_id": sample_id})
        expected_comments[comment_id] = value

    for split in ("train", "validate", "test"):
        path = data_dir / config["data"][f"structure_{split}_jsonl"]
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        records = read_jsonl(path)
        ids = set()
        for r in records:
            meta, inputs = r["metadata"], r["model_input"]
            news_id = str(meta["news_id"])
            ids.add(news_id)
            value = (split, inputs["news_text"], str(Path(inputs["image_path"]).resolve()))
            if news_id in news and news[news_id] != value:
                issues.append({"reason": "inconsistent_or_cross_split_news", "news_id": news_id})
            news[news_id] = value
            if meta["official_split"] != split:
                issues.append({"reason": "sft_split_mismatch", "sample_id": r["sample_id"]})
            if set(inputs) != {"news_text", "image_path", "observed_nodes"}:
                issues.append({"reason": "unexpected_input_fields", "sample_id": r["sample_id"]})
            mapping = {int(i): str(cid) for i, cid in meta["observed_node_comment_ids"].items()}
            expect(meta["target_comment_id"], news_id, r["target"]["content"],
                   meta["target_raw_parent_id"], r["sample_id"])
            target_parent = int(r["target"]["parent_node_index"])
            mapped_target_parent = news_id if target_parent == 0 else mapping.get(target_parent)
            if normalize_reddit_id(meta["target_raw_parent_id"]) != mapped_target_parent:
                issues.append({"reason": "target_parent_mapping", "sample_id": r["sample_id"]})
            for node in inputs["observed_nodes"]:
                parent_index = int(node["parent_node_index"])
                parent = news_id if parent_index == 0 else mapping.get(parent_index, "MISSING")
                expect(mapping[int(node["node_index"])], news_id, node["content"], parent, r["sample_id"])
        split_stats[split] = {"samples": len(records), "news": len(ids)}

    # Check source title and membership against every official split, including
    # accidental duplicate occurrences outside the declared split.
    news_source_counts = Counter()
    for split, value in config["source"]["news_tsv"].items():
        path = resolve_path(config_path, value)
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if not {"id", "clean_title"}.issubset(reader.fieldnames or []):
                raise ValueError(f"Missing news fields in {path}")
            for row in reader:
                news_id = normalize_reddit_id(row["id"])
                if news_id not in news:
                    continue
                news_source_counts[news_id] += 1
                expected_split, title, _ = news[news_id]
                if split != expected_split or row["clean_title"].strip() != title.strip():
                    issues.append({"reason": "raw_news_split_or_title", "news_id": news_id, "split": split})
    for news_id, (split, title, image_path) in news.items():
        if news_source_counts[news_id] != 1:
            issues.append({"reason": "raw_news_occurrence_count", "news_id": news_id,
                           "count": news_source_counts[news_id]})
        row = manifest.get(news_id, {})
        if (row.get("split") != split or row.get("news_text") != title or
                str(Path(row.get("image_path", "")).resolve()) != image_path):
            issues.append({"reason": "manifest_image_text_link", "news_id": news_id})
        if normalize_reddit_id(Path(image_path).stem) != news_id:
            issues.append({"reason": "image_filename_id", "news_id": news_id})
        result = validate_image(Path(image_path))
        if not result["image_valid"]:
            issues.append({"reason": "image_decode", "news_id": news_id, "error": result["image_error"]})

    matched, occurrences = set(), Counter()
    comments_path = resolve_path(config_path, config["source"]["comments_tsv"])
    total = malformed = 0
    for record, _ in iter_comment_records(comments_path):
        total += 1
        if record is None:
            malformed += 1
            continue
        cid = record["comment_id"]
        if cid in expected_comments:
            occurrences[cid] += 1
            value = (record["submission_id"], record["body"], record["parent_id"])
            if value == expected_comments[cid]:
                matched.add(cid)
            else:
                issues.append({"reason": "raw_comment_mismatch", "comment_id": cid,
                               "fields": [k for k, a, b in zip(
                                   ("submission_id", "body", "parent_id"), value, expected_comments[cid]) if a != b]})
        if total % 1000000 == 0:
            print(f"raw_comments_scanned={total} exact_matches={len(matched)}/{len(expected_comments)}", flush=True)
    for cid in sorted(set(expected_comments) - matched):
        issues.append({"reason": "raw_comment_exact_match_missing", "comment_id": cid})
    report = {
        "status": "passed" if not issues else "failed", "config": str(config_path),
        "source_comments": str(comments_path), "source_comments_bytes": comments_path.stat().st_size,
        "source_news": {s: str(resolve_path(config_path, p)) for s, p in config["source"]["news_tsv"].items()},
        "sft_sha256": hashes, "splits": split_stats, "unique_news_and_images_checked": len(news),
        "unique_target_and_context_comments": len(expected_comments), "exact_raw_comment_matches": len(matched),
        "raw_records_scanned": total, "malformed_raw_records": malformed,
        "duplicate_requested_comment_ids_in_source": sum(n > 1 for n in occurrences.values()),
        "issue_count": len(issues), "issues": issues,
        "scope": "Raw news text/split and raw comment submission/body/parent matching; manifest image association and decoding. Does not prove visual semantic relevance or content truthfulness.",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"issues", "sft_sha256", "source_news"}}, ensure_ascii=False, indent=2))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
