from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    default = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    parser = argparse.ArgumentParser(description="Validate CHECKED SFT split leakage.")
    parser.add_argument("--config", type=Path, default=default)
    return parser.parse_args()


def resolve(config_path: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (config_path.parent / path).resolve()


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data = config["data"]
    data_dir = resolve(config_path, data["output_dir"])
    paths = {
        "train": data_dir / data["train_jsonl"],
        "validate": data_dir / data["val_jsonl"],
        "test": data_dir / data["test_jsonl"],
    }
    datasets = {split: read_jsonl(path) for split, path in paths.items()}
    news_ids: dict[str, set[str]] = {}
    target_ids: dict[str, set[str]] = {}
    target_texts: dict[str, set[str]] = {}
    errors: list[str] = []
    for split, records in datasets.items():
        news_ids[split] = set()
        target_ids[split] = set()
        target_texts[split] = set()
        for record in records:
            metadata = record["metadata"]
            model_input = record["model_input"]
            if metadata["split"] != split:
                errors.append(f"split mismatch: {record['sample_id']}")
            if set(model_input) != {"image_path", "news_text", "observed_comments"}:
                errors.append(f"forbidden model input field: {record['sample_id']}")
            if not Path(model_input["image_path"]).is_file():
                errors.append(f"missing image: {record['sample_id']}")
            if metadata["target_comment_id"] in set(metadata["observed_comment_ids"]):
                errors.append(f"observed-target id overlap: {record['sample_id']}")
            if record["target_comment"] in set(model_input["observed_comments"]):
                errors.append(f"observed-target text overlap: {record['sample_id']}")
            news_ids[split].add(metadata["news_id"])
            target_ids[split].add(metadata["target_comment_id"])
            target_texts[split].add(" ".join(record["target_comment"].casefold().split()))
    overlaps: dict[str, dict[str, int]] = {}
    split_names = list(datasets)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            key = f"{left}__{right}"
            overlaps[key] = {
                "news": len(news_ids[left] & news_ids[right]),
                "target_ids": len(target_ids[left] & target_ids[right]),
                "target_texts": len(target_texts[left] & target_texts[right]),
            }
            if any(overlaps[key].values()):
                errors.append(f"cross-split overlap: {key}={overlaps[key]}")
    report = {
        "valid": not errors,
        "samples": {split: len(records) for split, records in datasets.items()},
        "news": {split: len(ids) for split, ids in news_ids.items()},
        "overlaps": overlaps,
        "errors": errors[:100],
    }
    report_path = config_path.parent.parent / "reports" / "dataset_validation.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
