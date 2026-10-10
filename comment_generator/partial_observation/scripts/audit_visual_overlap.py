"""Audit exact cross-split overlap without changing data or training selection.

Checks all structure-SFT news and the subset eligible for graph generation.
Exact matches are diagnostics, not proof of semantic relevance or leakage-free
evaluation. No labels are used to select samples or change the experiment.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageOps

from common import load_config, normalize_reddit_id, read_jsonl, resolve_path


def normalized_title(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def overlap_groups(rows, field):
    grouped = defaultdict(list)
    for row in rows:
        if row.get(field):
            grouped[row[field]].append(row)
    return [
        {"key": key, "members": members}
        for key, members in sorted(grouped.items())
        if len({member["split"] for member in members}) > 1
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--min-observed-nodes", type=int, default=1)
    args = parser.parse_args()
    config, config_path = load_config(args.config)
    data_dir = resolve_path(config_path, config["data"]["output_dir"])
    manifest_path = data_dir / "manifest.csv"
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    manifest = {row["news_id"]: row for row in manifest_rows}
    if len(manifest) != len(manifest_rows):
        raise ValueError("Duplicate manifest news ID")
    news = {}
    eligible = set()
    input_hashes = {str(manifest_path): hashlib.sha256(manifest_path.read_bytes()).hexdigest()}
    for split in ("train", "validate", "test"):
        path = data_dir / config["data"][f"structure_{split}_jsonl"]
        input_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        for record in read_jsonl(path):
            inputs, meta = record["model_input"], record["metadata"]
            news_id = str(meta["news_id"])
            key = (split, news_id)
            identity = (inputs["news_text"], str(Path(inputs["image_path"]).resolve()))
            if key in news and news[key] != identity:
                raise ValueError(f"Inconsistent input for {key}")
            news[key] = identity
            if len(inputs["observed_nodes"]) >= args.min_observed_nodes:
                eligible.add(key)

    source_urls = {}
    for split in ("train", "validate", "test"):
        path = resolve_path(config_path, config["source"]["news_tsv"][split])
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                key = (split, normalize_reddit_id(row["id"]))
                if key in news:
                    if key in source_urls:
                        raise ValueError(f"Duplicate source record: {key}")
                    source_urls[key] = row.get("image_url", "")

    rows, errors = [], []
    for (split, news_id), (title, image_path) in sorted(news.items()):
        entry = manifest[news_id]
        row = {"news_id": news_id, "split": split, "title": title,
               "normalized_title": normalized_title(title), "image_path": image_path,
               "image_url": entry.get("image_url", ""),
               "generation_eligible": (split, news_id) in eligible}
        source_url = source_urls.get((split, news_id))
        row["url_matches_original_tsv"] = source_url == row["image_url"]
        if source_url is None or not row["url_matches_original_tsv"]:
            errors.append({"news_id": news_id, "split": split, "reason": "source_url_mismatch"})
        try:
            path = Path(image_path)
            row["file_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            with Image.open(path) as image:
                rgb = ImageOps.exif_transpose(image).convert("RGB")
                rgb.load()
                row["width"], row["height"] = rgb.size
                dimensions = f"{rgb.width}x{rgb.height}:RGB:".encode()
                row["decoded_pixels_sha256"] = hashlib.sha256(dimensions + rgb.tobytes()).hexdigest()
        except (OSError, ValueError) as exc:
            errors.append({"news_id": news_id, "split": split,
                           "reason": "image_read_error", "error": str(exc)})
        rows.append(row)

    scopes = {}
    for name, selected in (("all_structure_sft_news", rows),
                           ("generation_eligible_news", [r for r in rows if r["generation_eligible"]])):
        fields = ("news_id", "normalized_title", "file_sha256", "decoded_pixels_sha256", "image_url")
        groups = {field: overlap_groups(selected, field) for field in fields}
        scopes[name] = {"news_count": len(selected),
                        "split_counts": dict(Counter(r["split"] for r in selected)),
                        "cross_split_group_counts": {k: len(v) for k, v in groups.items()},
                        "cross_split_groups": groups}
    report = {"config": str(config_path), "min_observed_nodes": args.min_observed_nodes,
              "input_sha256": input_hashes, "scopes": scopes, "errors": errors,
              "interpretation": [
                  "No samples were removed, reassigned or relabeled by this audit.",
                  "Shared images can be faithfully linked but not news-specific.",
                  "Exact title/byte/pixel equality does not detect all near duplicates or semantic overlap.",
                  "Image decoding and URL agreement do not establish visual relevance or image authenticity.",
                  "Generation eligibility matches min-observed filtering with no sample limit; verify final coverage separately.",
              ]}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output_json.resolve()), "errors": len(errors),
                      "scopes": {k: {x: v[x] for x in ("news_count", "split_counts", "cross_split_group_counts")}
                                 for k, v in scopes.items()}}, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
