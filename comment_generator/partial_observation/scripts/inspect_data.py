from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from common import IMAGE_EXTENSIONS, LABEL_COLUMNS, load_config, resolve_path


def parse_args() -> argparse.Namespace:
    default = Path(__file__).resolve().parents[1] / "config" / "generator.yaml"
    parser = argparse.ArgumentParser(description="Inspect Partial Observation data sources.")
    parser.add_argument("--config", type=Path, default=default)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config, config_path = load_config(args.config)
    source = config["source"]
    images_dir = resolve_path(config_path, source["images_dir"])
    comments_path = resolve_path(config_path, source["comments_tsv"])
    reports_dir = config_path.parent.parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    news_info = {}
    for split, value in source["news_tsv"].items():
        path = resolve_path(config_path, value)
        frame = pd.read_csv(
            path,
            sep="\t",
            usecols=["id", "clean_title", "image_url", *LABEL_COLUMNS],
            dtype=str,
            keep_default_na=False,
            low_memory=False,
        )
        news_info[split] = {
            "path": str(path),
            "rows": len(frame),
            "columns": list(pd.read_csv(path, sep="\t", nrows=0).columns),
            "duplicate_ids": int(frame["id"].str.lower().duplicated(keep=False).sum()),
        }

    images = [
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    with comments_path.open("r", encoding="utf-8-sig", newline="") as handle:
        comments_columns = next(csv.reader(handle, delimiter="\t"))
    report = {
        "images_dir": str(images_dir),
        "image_count": len(images),
        "image_extensions": dict(Counter(path.suffix.lower() for path in images)),
        "comments_tsv": str(comments_path),
        "comments_columns": comments_columns,
        "comments_has_timestamp": any(
            name.lower() in {"created_utc", "timestamp", "created_at"}
            for name in comments_columns
        ),
        "news": news_info,
    }
    output = reports_dir / "inspection.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
