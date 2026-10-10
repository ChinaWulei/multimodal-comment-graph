from __future__ import annotations

"""Turn downloaded official-subset images into a split-preserving manifest.

This deliberately does not search the historical flat ``images/`` directory.  A
row is included only when its file was downloaded under the selected official
train/validate/test subset, so a subsequent tree build cannot silently mix the
172-image pilot data with official held-out news.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from common import IMAGE_EXTENSIONS, LABEL_COLUMNS, load_config, resolve_path, validate_image


def parse_args() -> argparse.Namespace:
    default = Path(__file__).resolve().parents[1] / "config" / "generator_structure.yaml"
    parser = argparse.ArgumentParser(description="Build manifest for downloaded official Fakeddit subset images.")
    parser.add_argument("--config", type=Path, default=default)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <data.output_dir>/official_subset",
    )
    return parser.parse_args()


def find_image(images_root: Path, split: str, news_id: str) -> tuple[Path | None, str]:
    directory = images_root / split
    matches = sorted(
        path for path in directory.glob(f"{news_id}.*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not matches:
        return None, "image_missing"
    if len(matches) > 1:
        return matches[0], "duplicate_image_files"
    return matches[0], ""


def main() -> None:
    args = parse_args()
    config, config_path = load_config(args.config)
    data_root = resolve_path(config_path, config["data"]["output_dir"])
    output_dir = args.output_dir.resolve() if args.output_dir else data_root / "official_subset"
    output_dir.mkdir(parents=True, exist_ok=True)
    images_root = data_root / "official_subset_images"

    rows: list[dict] = []
    duplicate_news = 0
    for split in ("train", "validate", "test"):
        selected_path = data_root / f"official_structure_subset_{split}.csv"
        if not selected_path.exists():
            raise FileNotFoundError(f"Missing selected-subset file: {selected_path}")
        selected = pd.read_csv(selected_path, dtype=str, keep_default_na=False)
        if selected["news_id"].duplicated().any():
            duplicate_news += int(selected["news_id"].duplicated(keep=False).sum())
        for record in selected.to_dict("records"):
            news_id = str(record["news_id"]).strip().lower()
            image_path, reason = find_image(images_root, split, news_id)
            image_info = validate_image(image_path) if image_path else {"image_valid": False, "image_format": "", "image_width": 0, "image_height": 0, "image_error": ""}
            if image_path and not image_info["image_valid"]:
                reason = "image_invalid" if not reason else f"{reason};image_invalid"
            rows.append(
                {
                    "news_id": news_id,
                    "split": split,
                    "news_text": str(record.get("news_text", "")).strip(),
                    "image_path": str(image_path.resolve()) if image_path else "",
                    "image_url": str(record.get("image_url", "")).strip(),
                    "image_valid": bool(image_info["image_valid"]),
                    "image_format": image_info["image_format"],
                    "image_width": image_info["image_width"],
                    "image_height": image_info["image_height"],
                    **{column: str(record.get(column, "")).strip() for column in LABEL_COLUMNS},
                    "reported_num_comments": str(record.get("reported_num_comments", "")),
                    "valid_comment_count": str(record.get("valid_comment_count", "")),
                    "valid_top_level_count": str(record.get("valid_top_level_count", "")),
                    "valid_reply_count": str(record.get("valid_reply_count", "")),
                    "eligible": bool(image_info["image_valid"]) and bool(str(record.get("news_text", "")).strip()),
                    "failure_reason": reason,
                }
            )

    manifest = pd.DataFrame(rows).sort_values(["split", "news_id"], kind="stable")
    manifest_path = output_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    eligible = manifest[manifest["eligible"]]
    stats = {
        "selected_rows": len(manifest),
        "downloaded_valid_images": int(manifest["image_valid"].sum()),
        "eligible_news": len(eligible),
        "eligible_by_split": eligible["split"].value_counts().sort_index().to_dict(),
        "eligible_3_way_label_distribution": eligible["3_way_label"].value_counts().sort_index().to_dict(),
        "failure_reasons": Counter(reason for reason in manifest["failure_reason"] if reason),
        "duplicate_news_rows_within_split": duplicate_news,
        "images_root": str(images_root),
        "manifest": str(manifest_path),
    }
    stats_path = output_dir / "manifest_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2, default=dict) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2, default=dict))


if __name__ == "__main__":
    main()
