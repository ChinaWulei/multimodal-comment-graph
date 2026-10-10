"""Audit exact train/validation media and title reuse in the local Fakeddit subset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


def normalized(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text)).casefold()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    with args.manifest_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle)
                if row["split"] in {"train", "validate"} and str(row["image_valid"]).lower() == "true"]
    train = [row for row in rows if row["split"] == "train"]
    validate = [row for row in rows if row["split"] == "validate"]
    ids = {split: {row["news_id"] for row in part} for split, part in (("train", train), ("validate", validate))}
    hashes = {}
    for row in rows:
        image = Path(row["image_path"])
        if not image.is_file():
            raise FileNotFoundError(image)
        hashes[row["news_id"]] = hashlib.sha256(image.read_bytes()).hexdigest()
    train_by_hash = defaultdict(list)
    for row in train:
        train_by_hash[hashes[row["news_id"]]].append(row)
    train_titles = {normalized(row["news_text"]) for row in train}
    seen_images = []
    seen_titles = []
    for row in validate:
        if normalized(row["news_text"]) in train_titles:
            seen_titles.append(row["news_id"])
        donors = train_by_hash.get(hashes[row["news_id"]], [])
        if donors:
            seen_images.append({"validation_news_id": row["news_id"],
                                "training_news_ids": [donor["news_id"] for donor in donors],
                                "image_sha256": hashes[row["news_id"]]})
    result = {"train_news": len(train), "validation_news": len(validate),
              "news_id_overlap": len(ids["train"] & ids["validate"]),
              "normalized_exact_title_overlap": len(seen_titles),
              "exact_image_overlap_news": len(seen_images),
              "validation_exact_title_overlap_ids": seen_titles,
              "validation_exact_image_overlap": seen_images,
              "class_counts_validation": dict(Counter(row["3_way_label"] for row in validate)),
              "limitation": "Exact bytes/text only; shared images may be legitimate reuse and near-duplicates are not detected."}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in {"validation_exact_image_overlap", "validation_exact_title_overlap_ids"}},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
