"""Validate human-filled generated-comment quality/placement annotations."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

RELEVANCE = {"relevant", "partly_relevant", "off_topic"}
PARENT_FIT = {"good", "uncertain", "wrong"}
IMMUTABLE = (
    "news_id", "official_split", "source_group", "generated_position",
    "image_path", "headline", "parent_node_index", "parent_node_type",
    "parent_content", "recent_ancestors", "generated_comment",
)


def load_unique(path: Path) -> dict[str, dict]:
    result = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        key = str(row["annotation_id"])
        if key in result:
            raise ValueError(f"Duplicate annotation ID at {path}:{number}: {key}")
        result[key] = row
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-jsonl", type=Path, required=True,
                        help="Unmodified generated_quality_annotation_train60.jsonl queue")
    parser.add_argument("--annotations-jsonl", type=Path, required=True,
                        help="Copy of the queue containing human annotations")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    reference = load_unique(args.reference_jsonl)
    annotations = load_unique(args.annotations_jsonl)
    if set(reference) != set(annotations):
        raise ValueError(f"Annotation IDs differ: missing={sorted(set(reference)-set(annotations))[:5]}, "
                         f"extra={sorted(set(annotations)-set(reference))[:5]}")
    counts = {"content_relevance": Counter(), "parent_fit": Counter(),
              "unsupported_specific_claim": Counter()}
    incomplete = []
    for key, row in annotations.items():
        if row["official_split"] != "train":
            raise ValueError(f"Non-training annotation: {key}")
        for field in IMMUTABLE:
            if row[field] != reference[key][field]:
                raise ValueError(f"Source text/metadata was edited: {key}/{field}")
        relevance = row.get("content_relevance")
        parent_fit = row.get("parent_fit")
        unsupported = row.get("unsupported_specific_claim")
        if relevance not in RELEVANCE | {None}:
            raise ValueError(f"Invalid content_relevance: {key}/{relevance}")
        if parent_fit not in PARENT_FIT | {None}:
            raise ValueError(f"Invalid parent_fit: {key}/{parent_fit}")
        if unsupported is not None and not isinstance(unsupported, bool):
            raise ValueError(f"unsupported_specific_claim must be true/false/null: {key}")
        if not isinstance(row.get("review_notes", ""), str):
            raise ValueError(f"review_notes must be text: {key}")
        counts["content_relevance"][str(relevance)] += 1
        counts["parent_fit"][str(parent_fit)] += 1
        counts["unsupported_specific_claim"][str(unsupported)] += 1
        if relevance is None or parent_fit is None:
            incomplete.append(key)
    if args.require_complete and incomplete:
        raise ValueError(f"Missing required human labels on {len(incomplete)} rows: {incomplete[:5]}")
    print(json.dumps({"rows": len(annotations), "train_only": True,
                      "immutable_fields_verified": True,
                      "rows_missing_content_or_parent_label": len(incomplete),
                      "label_counts": {field: dict(counter) for field, counter in counts.items()}},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
